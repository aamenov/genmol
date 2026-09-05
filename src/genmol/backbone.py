"""Denoising backbones specific to GenMol diffusion variants."""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig


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
    ) -> None:
        super().__init__(config)
        self.time_conditioner = TimestepEmbedder(
            config.hidden_size,
            frequency_embedding_size=time_embedding_size,
            zero_init_output=zero_init_conditioning,
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

        if inputs_embeds is None:
            inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        conditioning = self.time_conditioner(noise_level)
        if conditioning.shape[0] != inputs_embeds.shape[0]:
            raise ValueError("noise_level must have one value per batch element")
        conditioned_embeddings = inputs_embeds + conditioning[:, None].to(inputs_embeds.dtype)
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=conditioned_embeddings,
            **kwargs,
        )
