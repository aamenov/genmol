"""Denoising backbones specific to GenMol diffusion variants."""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import BertForMaskedLM
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

ADDITIVE_CONDITIONING = "additive"
FILM_ADALN_CONDITIONING = "film_adaln"
CONDITIONING_VARIANTS = frozenset(
    {ADDITIVE_CONDITIONING, FILM_ADALN_CONDITIONING}
)


def is_conditioning_parameter_name(name: str) -> bool:
    """Return whether a backbone parameter belongs only to time conditioning."""

    normalized = name.removeprefix("backbone.")
    return normalized.startswith("time_conditioner.") or (
        ".film_modulation." in normalized
    )


class TimestepEmbedder(nn.Module):
    """Embed a scalar continuous noise level with sinusoidal features.

    This follows the timestep embedding used by the official UDLM DiT.  GenMol
    keeps its BERT backbone to isolate the effect of the diffusion process, so
    the resulting vector is added to every token embedding instead of driving
    DiT's adaptive layer normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        *,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if frequency_embedding_size < 2:
            raise ValueError("frequency_embedding_size must be at least 2")
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        if zero_init_output:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def sinusoidal_embedding(
        values: torch.Tensor,
        dimension: int,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        """Map one scalar per batch element to cosine/sine features."""

        if values.ndim != 1:
            raise ValueError("values must be a one-dimensional batch tensor")
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=values.device, dtype=torch.float32)
            / half
        )
        arguments = values.float()[:, None] * frequencies[None]
        embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
        if dimension % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        features = self.sinusoidal_embedding(values, self.frequency_embedding_size)
        return self.mlp(features)


class _FilmConditionedBertEncoder(nn.Module):
    """Run stock BERT layers and FiLM each post-layer normalized activation.

    The wrapper adopts the already initialized ``BertEncoder.layer`` module
    list.  Original BERT parameter paths therefore remain exactly
    ``bert.encoder.layer.*``; each layer gains only a new
    ``film_modulation`` projection.
    """

    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.config = encoder.config
        self.layer = encoder.layer
        self.gradient_checkpointing = encoder.gradient_checkpointing
        for layer in self.layer:
            layer.film_modulation = nn.Linear(hidden_size, 2 * hidden_size)
            nn.init.zeros_(layer.film_modulation.weight)
            nn.init.zeros_(layer.film_modulation.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if conditioning.ndim != 2 or conditioning.shape != (
            hidden_states.shape[0],
            hidden_states.shape[-1],
        ):
            raise ValueError(
                "FiLM conditioning must have shape [batch, hidden_size]"
            )

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`..."
            )
            use_cache = False

        next_decoder_cache = () if use_cache else None
        for index, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[index] if head_mask is not None else None
            past_key_value = (
                past_key_values[index] if past_key_values is not None else None
            )
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            shift, scale = layer_module.film_modulation(conditioning).chunk(
                2, dim=-1
            )
            shift = shift[:, None, :].to(dtype=hidden_states.dtype)
            scale = scale[:, None, :].to(dtype=hidden_states.dtype)
            hidden_states = hidden_states * (1 + scale) + shift

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in (
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                )
                if value is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class TimeConditionedBertForMaskedLM(BertForMaskedLM):
    """BERT masked LM conditioned on one continuous noise level per sequence.

    Subclassing rather than wrapping ``BertForMaskedLM`` preserves all original
    ``backbone.bert.*`` checkpoint keys.  Consequently an MDLM backbone can be
    loaded explicitly into this model while the new conditioner is initialized
    separately.
    """

    def __init__(
        self,
        config: BertConfig,
        *,
        time_embedding_size: int = 256,
        zero_init_conditioning: bool = True,
        conditioning_variant: str = ADDITIVE_CONDITIONING,
    ) -> None:
        super().__init__(config)
        self.conditioning_variant = str(conditioning_variant).lower()
        if self.conditioning_variant not in CONDITIONING_VARIANTS:
            allowed = ", ".join(sorted(CONDITIONING_VARIANTS))
            raise ValueError(f"conditioning_variant must be one of: {allowed}")
        if (
            self.conditioning_variant == FILM_ADALN_CONDITIONING
            and zero_init_conditioning
        ):
            raise ValueError(
                "film_adaln requires a normally initialized timestep MLP; "
                "zeroing both it and the FiLM projections kills all conditioning "
                "weight gradients"
            )
        self.time_conditioner = TimestepEmbedder(
            config.hidden_size,
            frequency_embedding_size=time_embedding_size,
            zero_init_output=zero_init_conditioning,
        )
        if self.conditioning_variant == FILM_ADALN_CONDITIONING:
            self.bert.encoder = _FilmConditionedBertEncoder(
                self.bert.encoder,
                config.hidden_size,
            )

    def _film_bert_forward(
        self,
        *,
        conditioning: torch.Tensor,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        """Mirror the pinned Hugging Face BERT flow while passing FiLM explicitly."""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = self.config.use_cache if self.config.is_decoder else False

        if input_ids is not None:
            self.bert.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        batch_size, sequence_length = input_shape

        if token_type_ids is None:
            if hasattr(self.bert.embeddings, "token_type_ids"):
                token_type_ids = self.bert.embeddings.token_type_ids[
                    :, :sequence_length
                ].expand(batch_size, sequence_length)
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length), device=device
            )

        use_sdpa_attention_masks = (
            self.bert.attn_implementation == "sdpa"
            and self.bert.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    0,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask,
                    embedding_output.dtype,
                    tgt_len=sequence_length,
                )
        else:
            extended_attention_mask = self.bert.get_extended_attention_mask(
                attention_mask, input_shape
            )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device
                )
            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    embedding_output.dtype,
                    tgt_len=sequence_length,
                )
            else:
                encoder_extended_attention_mask = self.bert.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_extended_attention_mask = None

        prepared_head_mask = self.bert.get_head_mask(
            head_mask, self.config.num_hidden_layers
        )
        encoder_outputs = self.bert.encoder(
            embedding_output,
            conditioning=conditioning,
            attention_mask=extended_attention_mask,
            head_mask=prepared_head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.bert.pooler(sequence_output)
            if self.bert.pooler is not None
            else None
        )
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def _film_masked_lm_forward(
        self,
        *,
        conditioning: torch.Tensor,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        **kwargs,
    ):
        labels = kwargs.pop("labels", None)
        return_dict = kwargs.get("return_dict")
        outputs = self._film_bert_forward(
            conditioning=conditioning,
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        prediction_scores = self.cls(outputs[0])
        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = nn.CrossEntropyLoss()(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        resolved_return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if not resolved_return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        noise_level: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        if noise_level is None:
            raise ValueError("UDLM's time-conditioned BERT requires noise_level")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("provide input_ids or inputs_embeds, not both")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("one of input_ids or inputs_embeds is required")

        conditioning = self.time_conditioner(noise_level)
        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )
        if conditioning.shape[0] != batch_size:
            raise ValueError("noise_level must have one value per batch element")

        if self.conditioning_variant == ADDITIVE_CONDITIONING:
            if inputs_embeds is None:
                inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
            conditioned_embeddings = inputs_embeds + conditioning[:, None].to(
                inputs_embeds.dtype
            )
            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=conditioned_embeddings,
                **kwargs,
            )

        conditioning = torch.nn.functional.silu(conditioning)
        return self._film_masked_lm_forward(
            conditioning=conditioning,
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
