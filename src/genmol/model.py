# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
import hydra.utils
import lightning as L
import torch
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from bionemo.moco.interpolants import MDLM
from bionemo.moco.distributions.time import UniformTimeDistribution
from genmol.utils.utils_moco import AntitheticUniformTimeDistribution
from bionemo.moco.schedules.noise.continuous_noise_transforms import LogLinearExpNoiseTransform
from bionemo.moco.distributions.prior import DiscreteMaskedPrior

from genmol.backbone import TimeConditionedBertForMaskedLM
from genmol.diffusion import ContinuousUniformDiffusion
from genmol.utils.ema import ExponentialMovingAverage
from genmol.utils.utils_data import get_tokenizer
from genmol.utils.utils_save import clean_checkpoint, fast_forward_info

class GenMol(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # set up tokenizer
        self.tokenizer = get_tokenizer()
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.pad_index = self.tokenizer.pad_token_id

        # Checkpoints released before UDLM have no diffusion selector.  The
        # explicit MDLM fallback keeps their architecture and state keys exact.
        self.diffusion_type = str(self.config.training.get('diffusion', 'mdlm')).lower()
        if self.diffusion_type not in {'mdlm', 'udlm'}:
            raise ValueError("training.diffusion must be either 'mdlm' or 'udlm'")

        backbone_config = BertConfig.from_dict(dict(self.config.model))
        if self.diffusion_type == 'udlm':
            udlm_config = self.config.training.get('udlm', {})
            self.backbone = TimeConditionedBertForMaskedLM(
                backbone_config,
                time_embedding_size=int(udlm_config.get('time_embedding_size', 256)),
                zero_init_conditioning=bool(
                    udlm_config.get('zero_init_conditioning', True)
                ),
            )
        else:
            self.backbone = BertForMaskedLM(backbone_config)

        # Keep ``mdlm`` as the process attribute for downstream compatibility.
        # For a UDLM checkpoint it points to the uniform process instead.
        if self.diffusion_type == 'mdlm':
            if self.config.training.antithetic_sampling:
                time_distribution = AntitheticUniformTimeDistribution(
                    sampling_eps=self.config.training.sampling_eps
                )
            else:
                time_distribution = UniformTimeDistribution()
            prior = DiscreteMaskedPrior(
                num_classes=self.config.model.vocab_size,
                mask_dim=self.mask_index,
            )
            noise_schedule = LogLinearExpNoiseTransform()
            self.mdlm = MDLM(
                time_distribution=time_distribution,
                prior_distribution=prior,
                noise_schedule=noise_schedule,
            )
        else:
            udlm_config = self.config.training.get('udlm', {})
            excluded_token_ids = ()
            if udlm_config.get('exclude_special_tokens', False):
                excluded_token_ids = tuple(self.tokenizer.all_special_ids)
            self.mdlm = ContinuousUniformDiffusion(
                num_classes=self.config.model.vocab_size,
                excluded_token_ids=excluded_token_ids,
                sampling_eps=float(self.config.training.sampling_eps),
                noise_eps=float(udlm_config.get('noise_eps', 1e-3)),
                antithetic_sampling=bool(self.config.training.antithetic_sampling),
            )
        # set up ema
        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(self.backbone.parameters(), decay=self.config.training.ema)
        else:
            self.ema = None

    def initialize_from_mdlm_checkpoint(self, checkpoint_path, use_ema=True):
        """Warm-start UDLM's BERT only, resetting all training state.

        This is intentionally not a Lightning resume: the optimizer, learning
        rate schedule, global step, UDLM process, time conditioner, and EMA are
        all new.  When available, the source MDLM EMA weights are preferred
        because they are the weights used by the released GenMol sampler.
        """

        if self.diffusion_type != 'udlm':
            raise ValueError('MDLM backbone initialization is only valid for UDLM')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        source_state = checkpoint.get('state_dict', checkpoint)
        backbone_state = {
            key.removeprefix('backbone.'): value
            for key, value in source_state.items()
            if key.startswith('backbone.')
        }
        if not backbone_state:
            raise ValueError('checkpoint contains no backbone.* tensors')

        load_result = self.backbone.load_state_dict(backbone_state, strict=False)
        expected_missing = {
            key for key in self.backbone.state_dict() if key.startswith('time_conditioner.')
        }
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise ValueError(
                'MDLM backbone is incompatible with this UDLM architecture: '
                f'missing={load_result.missing_keys}, '
                f'unexpected={load_result.unexpected_keys}'
            )

        weights = 'raw'
        base_parameters = [
            parameter
            for name, parameter in self.backbone.named_parameters()
            if not name.startswith('time_conditioner.')
        ]
        if use_ema:
            ema_state = checkpoint.get('ema')
            shadow_parameters = None if ema_state is None else ema_state.get('shadow_params')
            if shadow_parameters is None:
                raise ValueError('requested MDLM EMA initialization, but checkpoint has no EMA')
            if len(shadow_parameters) != len(base_parameters):
                raise ValueError(
                    'MDLM EMA parameter count does not match the BERT backbone: '
                    f'{len(shadow_parameters)} != {len(base_parameters)}'
                )
            with torch.no_grad():
                for parameter, shadow in zip(base_parameters, shadow_parameters):
                    parameter.copy_(shadow)
            weights = 'ema'

        if self.ema:
            self.ema = ExponentialMovingAverage(
                self.backbone.parameters(),
                decay=self.config.training.ema,
            )
        return {
            'source_path': str(checkpoint_path),
            'weights': weights,
            'parameter_tensors': len(base_parameters),
        }

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)
        
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            {'_target_': 'transformers.get_constant_schedule_with_warmup',
             'num_warmup_steps': 2500},
             optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'lr'}
        return [optimizer], [scheduler_dict]

    def on_train_start(self):
        self.backbone.train()
        self.mdlm.to_device(self.device)
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(self.backbone.parameters()))
        
    def forward(self, x, attention_mask=None, t=None):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            if self.diffusion_type == 'udlm':
                if t is None:
                    raise ValueError("UDLM forward passes require one time value per sequence")
                noise_level = self.mdlm.sigma(t.to(device=x.device, dtype=torch.float32))
                return self.backbone(
                    x,
                    attention_mask,
                    noise_level=noise_level,
                )['logits']
            return self.backbone(x, attention_mask)['logits']

    def diffusion_token_mask(self, input_ids, attention_mask):
        """Select molecular content positions while preserving sequence framing."""

        token_mask = attention_mask.to(dtype=torch.bool)
        for token_id in (self.pad_index, self.bos_index, self.eos_index):
            if token_id is not None:
                token_mask &= input_ids != token_id
        return token_mask
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # sample time
        t = self.mdlm.sample_time(input_ids.shape[0], device=input_ids.device)
        if self.diffusion_type == 'udlm':
            loss_mask = self.diffusion_token_mask(input_ids, attention_mask)
            xt = self.mdlm.forward_process(input_ids, t, mutable_mask=loss_mask)
            logits = self(xt, attention_mask, t=t)
        else:
            loss_mask = attention_mask
            # Forward process to add absorbing mask tokens.
            xt = self.mdlm.forward_process(input_ids, t)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                logits = self.backbone(xt, attention_mask)["logits"]
        # compute loss
        if self.config.training.global_mean_loss:
            loss = self.mdlm.loss(
                logits,
                input_ids,
                xt,
                t,
                mask=loss_mask,
                global_mean=True,
            )
        else:
            loss = self.mdlm.loss(
                logits,
                input_ids,
                xt,
                t,
                mask=loss_mask,
            ).mean()
        self.log(name='train_loss',
                 value=loss.item(),
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        return loss
