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


import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import math
import numbers
import pickle
import torch
import random
import safe as sf
from types import MappingProxyType
from rdkit import Chem
from genmol.utils.utils_chem import safe_to_smiles, filter_by_substructure, mix_sequences, Slicer
from genmol.utils.bracket_safe_converter import BracketSAFEConverter, bracketsafe2safe
from genmol.utils.checkpoint_io import verified_checkpoint_file  # noqa: E402
from genmol.model import GenMol


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def _inference_weights_receipt(source, ema_applied, ema):
    """Return an immutable receipt for the weights installed in the backbone."""
    frozen_ema = None if ema is None else MappingProxyType(dict(ema))
    return MappingProxyType({
        'source': source,
        'ema_applied': ema_applied,
        'ema': frozen_ema,
    })


def _copy_inference_weights_receipt(receipt):
    """Return a JSON-serializable copy without exposing mutable internal state."""
    ema = receipt['ema']
    return {
        'source': receipt['source'],
        'ema_applied': receipt['ema_applied'],
        'ema': None if ema is None else dict(ema),
    }


def _ema_metadata_and_parameters(model):
    """Validate the EMA state against the trainable backbone parameters."""
    parameters = list(model.backbone.parameters())
    trainable_parameters = [
        parameter for parameter in parameters if parameter.requires_grad
    ]
    shadow_parameters = list(model.ema.shadow_params)
    if not trainable_parameters:
        raise RuntimeError('Cannot apply EMA: the backbone has no trainable parameters')
    if len(shadow_parameters) != len(trainable_parameters):
        raise RuntimeError(
            'Cannot apply EMA: shadow parameter count '
            f'{len(shadow_parameters)} does not match trainable backbone parameter '
            f'count {len(trainable_parameters)}'
        )

    for index, (shadow, parameter) in enumerate(
        zip(shadow_parameters, trainable_parameters)
    ):
        if not isinstance(shadow, torch.Tensor):
            raise RuntimeError(
                f'Cannot apply EMA: shadow parameter {index} is not a tensor'
            )
        if shadow.shape != parameter.shape:
            raise RuntimeError(
                f'Cannot apply EMA: shadow parameter {index} has shape '
                f'{tuple(shadow.shape)}, expected {tuple(parameter.shape)}'
            )
        if not torch.isfinite(shadow).all().item():
            raise RuntimeError(
                f'Cannot apply EMA: shadow parameter {index} contains non-finite values'
            )

    decay = model.ema.decay
    if hasattr(decay, 'item'):
        decay = decay.item()
    if isinstance(decay, bool) or not isinstance(decay, numbers.Real):
        raise RuntimeError('Cannot apply EMA: decay is not a real scalar')
    decay = float(decay)
    if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise RuntimeError('Cannot apply EMA: decay must be finite and in [0, 1]')

    num_updates = model.ema.num_updates
    if hasattr(num_updates, 'item'):
        num_updates = num_updates.item()
    if num_updates is not None:
        if (
            isinstance(num_updates, bool)
            or not isinstance(num_updates, numbers.Integral)
        ):
            raise RuntimeError(
                'Cannot apply EMA: num_updates must be a non-negative integer or null'
            )
        num_updates = int(num_updates)
        if num_updates < 0:
            raise RuntimeError('Cannot apply EMA: num_updates cannot be negative')

    return parameters, trainable_parameters, shadow_parameters, {
        'shadow_parameter_count': len(shadow_parameters),
        'decay': decay,
        'num_updates': num_updates,
    }


def load_model_from_path(
    path,
    expected_checkpoint_sha256=None,
    *,
    require_ema=False,
):
    with verified_checkpoint_file(
        path,
        expected_sha256=expected_checkpoint_sha256,
    ) as (checkpoint_file, _identity):
        model = GenMol.load_from_checkpoint(checkpoint_file)
    model.backbone.eval()
    if model.ema is None:
        if require_ema:
            raise RuntimeError(
                'EMA inference weights were required, but the checkpoint has no EMA state'
            )
        receipt = _inference_weights_receipt('raw_model', False, None)
    else:
        (
            parameters,
            trainable_parameters,
            shadow_parameters,
            ema_metadata,
        ) = _ema_metadata_and_parameters(model)
        model.ema.store(iter(parameters))
        model.ema.copy_to(iter(parameters))
        for index, (shadow, parameter) in enumerate(
            zip(shadow_parameters, trainable_parameters)
        ):
            expected = shadow.detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            if not torch.equal(parameter.detach(), expected):
                raise RuntimeError(
                    f'EMA copy verification failed for backbone parameter {index}'
                )
        receipt = _inference_weights_receipt('ema', True, ema_metadata)
    # Freeze the load-time fact. Sampler returns defensive plain-dict copies
    # when callers need JSON serialization.
    model.inference_weights = receipt
    return model


class Sampler:
    def __init__(
        self,
        path,
        expected_checkpoint_sha256=None,
        length_distribution=None,
        *,
        require_ema=False,
    ):
        self.model = load_model_from_path(
            path,
            expected_checkpoint_sha256,
            require_ema=require_ema,
        )
        self._inference_weights = self.model.inference_weights
        self.slicer = Slicer()
        self.dot_index = self.model.tokenizer('.')['input_ids'][1]
        self.pad_index = self.model.tokenizer.pad_token_id
        self.mdlm = self.model.mdlm
        self.mdlm.to_device(self.model.device)
        self.diffusion_type = getattr(self.model, 'diffusion_type', 'mdlm')
        if length_distribution is None:
            with open(os.path.join(ROOT_DIR, 'data/len.pk'), 'rb') as f:
                length_distribution = pickle.load(f)
        try:
            self.length_distribution = tuple(length_distribution)
        except TypeError as error:
            raise ValueError('length_distribution must be an integer sequence') from error
        if not self.length_distribution or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.length_distribution
        ):
            raise ValueError('length_distribution must be a nonempty integer sequence')

    @property
    def inference_weights(self):
        """Describe the backbone weights selected before any inference call."""
        return _copy_inference_weights_receipt(self._inference_weights)
        
    @torch.no_grad()
    def generate(
        self,
        x,
        softmax_temp=1.2,
        randomness=2,
        fix=True,
        gamma=0,
        w=2,
        num_steps=None,
        return_token_ids=False,
        **kwargs,
    ):
        """Generate molecules or raw IDs; ``randomness`` is MDLM-only."""
        x = x.to(self.model.device)
        attention_mask = x != self.pad_index
        if self.diffusion_type == 'udlm':
            if gamma and w:
                raise ValueError(
                    'GenMol MCG blends clean logits and is not a valid UDLM '
                    'posterior-space guidance rule; UDLM guidance is not yet enabled.'
                )
            editable_mask = x == self.model.mask_index
            prior = self.mdlm.sample_prior(x.shape, device=x.device)
            x = torch.where(editable_mask, prior, x)
            udlm_config = self.model.config.training.get('udlm', {})
            if num_steps is None:
                num_steps = int(udlm_config.get('sampling_steps', 128))
            if num_steps <= 0:
                raise ValueError('num_steps must be positive for UDLM sampling')
            inference_eps = float(udlm_config.get('inference_eps', 1e-5))
            if not 0 < inference_eps < 1:
                raise ValueError('UDLM inference_eps must lie strictly between 0 and 1')

            timesteps = torch.linspace(
                1.0,
                inference_eps,
                num_steps + 1,
                device=x.device,
                dtype=torch.float32,
            )
            for i in range(num_steps):
                t = timesteps[i].expand(x.shape[0])
                s = timesteps[i + 1].expand(x.shape[0])
                logits = self.model(x, attention_mask, t=t)
                x = self.mdlm.step(
                    logits,
                    x,
                    t,
                    s,
                    mutable_mask=editable_mask,
                    temperature=softmax_temp,
                )
        else:
            num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
            for i in range(num_steps):
                logits = self.model(x, attention_mask)

                if gamma and w:
                    x_poor = x.clone()
                    context_tokens = (x_poor[0] != self.model.bos_index).to(int) * \
                        (x_poor[0] != self.model.eos_index).to(int) * \
                        (x_poor[0] != self.model.mask_index).to(int) * \
                        (x_poor[0] != self.pad_index).to(int)
                    context_token_ids = context_tokens.nonzero(as_tuple=True)[0].tolist()
                    # mask 100 * gamma % of the context (given fragments) tokens
                    num_mask_poor = int(context_tokens.sum() * gamma)
                    mask_idx_poor = random.sample(context_token_ids, num_mask_poor)
                    x_poor[:, mask_idx_poor] = self.model.mask_index
                    logits_poor = self.model(x_poor, attention_mask=attention_mask)
                    logits = w * logits + (1 - w) * logits_poor

                x = self.mdlm.step_confidence(
                    logits,
                    x,
                    i,
                    num_steps,
                    softmax_temp,
                    randomness,
                )

        if return_token_ids:
            return x

        # decode to SAFE strings
        samples = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
        # convert to SMILES strings
        if self.model.config.training.get('use_bracket_safe'):
            samples = [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in samples]
        else:
            samples = [safe_to_smiles(s, fix=fix) for s in samples]
        # remove None and take the largest
        samples = [sorted(s.split('.'), key=len)[-1] for s in samples if s]
        return samples

    def _insert_mask(self, x, num_samples, min_add_len=18, **kwargs):
        x = x[0]
        x_new = []
        for _ in range(num_samples):
            add_seq_len = max(
                random.choice(self.length_distribution) - len(x), min_add_len
            )
            x_new.append(torch.hstack([x[:-1],
                                      torch.full((add_seq_len,), self.model.mask_index),
                                      x[-1:]]))
        pad_len = max([len(xx) for xx in x_new])
        x_new = [torch.hstack([xx,torch.full((pad_len - len(xx),), self.pad_index)]) for xx in x_new]
        return torch.stack(x_new)
    
    @torch.no_grad()
    def de_novo_generation(self, num_samples=1, softmax_temp=0.8, randomness=0.5, min_add_len=40, **kwargs):
        # Prepare Fully Masked Inputs
        x = torch.hstack([torch.full((1, 1), self.model.bos_index),
                          torch.full((1, 1), self.model.eos_index)])
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        x = x.to(self.model.device)
        return self.generate(x, softmax_temp, randomness, **kwargs)
    
    def fragment_linking_onestep(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0, min_add_len=30, **kwargs):
        if self.model.config.training.get('use_bracket_safe'):
            encoded_fragment = BracketSAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        else:
            encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        
        x = self.model.tokenizer([encoded_fragment + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        samples = self.generate(
            x, softmax_temp, randomness, gamma=gamma, **kwargs
        )
        samples = filter_by_substructure(samples, fragment)
        return samples
    
    def fragment_linking(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0, min_add_len=30, **kwargs):
        encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        prefix, suffix = encoded_fragment.split('.')

        x = self.model.tokenizer([prefix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        prefix_samples = self.generate(
            x, softmax_temp, randomness, gamma=gamma, **kwargs
        )

        x = self.model.tokenizer([suffix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        suffix_samples = self.generate(
            x, softmax_temp, randomness, gamma=gamma, **kwargs
        )
        
        samples = filter_by_substructure(mix_sequences(prefix_samples, suffix_samples,
                                                      *fragment.split('.'), num_samples), fragment)
        return samples
        
    def fragment_completion(self, fragment, num_samples=1, apply_filter=True, softmax_temp=1.2, randomness=2, gamma=0, **kwargs):
        if '*' not in fragment:     # superstructure generation
            cores = sf.utils.list_individual_attach_points(Chem.MolFromSmiles(fragment), depth=3)
            fragment = random.choice(cores)
            
        encoded_fragment = sf.SAFEConverter(ignore_stereo=True).encoder(fragment, allow_empty=True) + '.'
        x = self.model.tokenizer([encoded_fragment],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples)
        samples = self.generate(
            x, softmax_temp, randomness, gamma=gamma, **kwargs
        )

        if apply_filter:
            return filter_by_substructure(samples, fragment)
        return samples

    def mask_modification(self, smiles, min_len=30, **kwargs):
        encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
        x = self.model.tokenizer([encoded_smiles],
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=self.model.config.model.max_position_embeddings)['input_ids']
        if x.shape[-1] < min_len:
            return self.addmask(smiles, num_edit=min_len-x.shape[-1]+1, **kwargs)
        return self.remask(smiles, input_ids=x, **kwargs)

    def addmask(self, smiles, num_edit=3, **kwargs):
        try:
            samples = self.fragment_completion(smiles, mask_len=num_edit, apply_filter=False, **kwargs)
        except:
            return smiles
        if samples:
            return samples[0]
        return smiles
    
    def remask(self, smiles, input_ids=None, **kwargs):
        x = input_ids
        if x is None:
            encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
            x = self.model.tokenizer([encoded_smiles],
                                     return_tensors='pt',
                                     truncation=True,
                                     max_length=self.model.config.model.max_position_embeddings)['input_ids']
        
        # fragment mask replacement
        special_token_idx = [0] + (x[0] == self.dot_index).nonzero(as_tuple=True)[0].tolist() + [len(x[0]) - 1]
        frag_idx = random.randint(0, len(special_token_idx) - 2)
        mask_start_idx = special_token_idx[frag_idx] + 1
        mask_end_idx = special_token_idx[frag_idx + 1]
        num_insert_mask = random.randint(5, 15)
        num_insert_mask = min(num_insert_mask,
                              self.model.config.model.max_position_embeddings - x.shape[-1] + mask_end_idx - mask_start_idx)
        x = torch.hstack([x[:, :mask_start_idx],
                          torch.full((1, num_insert_mask), self.model.mask_index),
                          x[:, mask_end_idx:]])
        samples = self.generate(x, **kwargs)
        if samples:
            return samples[0]
        return smiles
