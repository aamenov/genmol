"""Discrete diffusion processes used by GenMol.

The uniform process in this module implements the continuous-time UDLM
objective from Schiff et al., *Simple Guidance Mechanisms for Discrete
Diffusion Models* (Eq. 18), together with its finite-step reverse posterior.

The released UDLM code evaluates Eq. 18 as a difference of two terms.  That
form is algebraically correct but can lose precision when the prediction is
already good.  Here the same integrand is evaluated as a generalized KL:

    sum_j r_j * (exp(u_j) - 1 - u_j),

where ``r_j = x_bar_j / x_bar_i`` and
``u_j = log(x_bar_theta_j / x_bar_theta_i) - log(r_j)``.  Every summand is
non-negative, and ``expm1`` keeps the expression accurate near zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _broadcast_time(t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Append singleton dimensions until ``t`` broadcasts over ``reference``."""

    while t.ndim < reference.ndim:
        t = t.unsqueeze(-1)
    return t


def _expm1_minus_x(x: torch.Tensor) -> torch.Tensor:
    """Stably compute ``exp(x) - 1 - x`` with a smooth local series."""

    # Direct subtraction loses all useful digits close to zero.  Four terms
    # are ample at this threshold in float32 and preserve finite gradients.
    x2 = x * x
    series = x2 * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x / 120.0)))
    return torch.where(x.abs() < 1e-3, series, torch.expm1(x) - x)


class ContinuousUniformDiffusion(nn.Module):
    """Continuous-time uniform discrete diffusion (UDLM).

    Parameters
    ----------
    num_classes:
        Size of the model output vocabulary.
    excluded_token_ids:
        Optional model-token IDs excluded from the uniform corruption prior.
        An empty sequence exactly matches the official UDLM vocabulary policy.
        Excluding tokenizer control symbols is a molecule-specific ablation,
        not part of the paper's base method.
    sampling_eps:
        Lower bound for sampled training times.
    noise_eps:
        Residual clean probability at ``t=1`` in the official log-linear
        schedule.  The released loss uses the idealized ``alpha(t)=1-t``;
        corruption and sampling use ``alpha(t)=1-(1-noise_eps)t``.
    antithetic_sampling:
        Stratify one time sample per batch element as in MDLM/official UDLM.

    Notes
    -----
    Model tensors use the full vocabulary of size ``num_classes``.  Internally
    the process maps allowed token IDs to a compact alphabet of size ``N`` so
    the same exact equations also support a restricted corruption alphabet.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        excluded_token_ids: Iterable[int] = (),
        sampling_eps: float = 1e-3,
        noise_eps: float = 1e-3,
        antithetic_sampling: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 < sampling_eps < 1.0:
            raise ValueError("sampling_eps must lie strictly between 0 and 1")
        if not 0.0 < noise_eps < 1.0:
            raise ValueError("noise_eps must lie strictly between 0 and 1")

        excluded = {int(token_id) for token_id in excluded_token_ids}
        invalid = sorted(token_id for token_id in excluded if not 0 <= token_id < num_classes)
        if invalid:
            raise ValueError(f"excluded token IDs outside the vocabulary: {invalid}")

        token_ids = [token_id for token_id in range(num_classes) if token_id not in excluded]
        if len(token_ids) < 2:
            raise ValueError("the uniform diffusion alphabet must contain at least 2 tokens")
        token_to_index = torch.full((num_classes,), -1, dtype=torch.long)
        token_to_index[token_ids] = torch.arange(len(token_ids), dtype=torch.long)

        self.num_classes = int(num_classes)
        self.sampling_eps = float(sampling_eps)
        self.noise_eps = float(noise_eps)
        self.antithetic_sampling = bool(antithetic_sampling)
        self.register_buffer("diffusion_token_ids", torch.tensor(token_ids, dtype=torch.long))
        self.register_buffer("token_to_diffusion_index", token_to_index)

    @property
    def diffusion_vocab_size(self) -> int:
        return int(self.diffusion_token_ids.numel())

    def to_device(self, device: torch.device | str) -> "ContinuousUniformDiffusion":
        """Compatibility shim for GenMol callers that move the MDLM helper."""

        self.to(device)
        return self

    def sample_time(
        self,
        n_samples: int,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample continuous times in ``[sampling_eps, 1)``."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if device is None:
            device = self.diffusion_token_ids.device
        time = torch.rand(n_samples, device=device, generator=generator)
        if self.antithetic_sampling:
            offsets = torch.arange(n_samples, device=device, dtype=time.dtype) / n_samples
            time = (time / n_samples + offsets) % 1.0
        return (1.0 - self.sampling_eps) * time + self.sampling_eps

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Official log-linear total noise used to condition the denoiser."""

        return -torch.log1p(-(1.0 - self.noise_eps) * t)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Clean-data coefficient used by corruption and reverse sampling."""

        return torch.exp(-self.sigma(t))

    def _compact_indices(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        compact = self.token_to_diffusion_index[token_ids]
        return compact.clamp_min(0), compact >= 0

    def clean_log_probs(
        self,
        logits: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return float32 clean-token log probabilities on the UDLM alphabet."""

        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} logits, received {logits.shape[-1]}"
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        # Float64 is useful for reference checks; all lower-precision training
        # dtypes are deliberately promoted to float32 for the UDLM algebra.
        compute_logits = logits if logits.dtype == torch.float64 else logits.float()
        selected = compute_logits.index_select(-1, self.diffusion_token_ids)
        return (selected / float(temperature)).log_softmax(dim=-1)

    def sample_prior(
        self,
        shape: Sequence[int] | torch.Size,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample iid tokens from the uniform limiting distribution."""

        if device is None:
            device = self.diffusion_token_ids.device
        compact = torch.randint(
            self.diffusion_vocab_size,
            tuple(shape),
            device=device,
            generator=generator,
        )
        return self.diffusion_token_ids.to(device=device)[compact]

    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        *,
        mutable_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample ``z_t`` by replacing tokens with iid uniform noise."""

        if x0.ndim < 2:
            raise ValueError("x0 must have batch and sequence dimensions")
        if t.shape != (x0.shape[0],):
            raise ValueError(f"t must have shape ({x0.shape[0]},), received {tuple(t.shape)}")
        if mutable_mask is None:
            mutable_mask = torch.ones_like(x0, dtype=torch.bool)
        else:
            mutable_mask = mutable_mask.to(device=x0.device, dtype=torch.bool)
            if mutable_mask.shape != x0.shape:
                raise ValueError("mutable_mask must have the same shape as x0")

        _, allowed = self._compact_indices(x0)
        if torch.any(mutable_mask & ~allowed):
            bad_ids = torch.unique(x0[mutable_mask & ~allowed]).tolist()
            raise ValueError(f"mutable positions contain excluded token IDs: {bad_ids}")

        move_chance = 1.0 - self.alpha(t.to(device=x0.device, dtype=torch.float32))
        move_chance = _broadcast_time(move_chance, x0)
        replace = torch.rand(x0.shape, device=x0.device, generator=generator) < move_chance
        replace &= mutable_mask
        noise = self.sample_prior(x0.shape, device=x0.device, generator=generator)
        return torch.where(replace, noise, x0)

    def loss_per_token(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the continuous-time UDLM NELBO integrand.

        The returned tensor has shape ``(batch, length)``.  All algebra is
        evaluated in float32 even under mixed precision.
        """

        if logits.shape[:-1] != x0.shape or xt.shape != x0.shape:
            raise ValueError("logits, x0, and xt shapes are inconsistent")
        if t.shape != (x0.shape[0],):
            raise ValueError(f"t must have shape ({x0.shape[0]},), received {tuple(t.shape)}")
        if torch.any((t <= 0) | (t >= 1)):
            raise ValueError("continuous-time UDLM loss requires 0 < t < 1")

        if mask is None:
            token_mask = torch.ones_like(x0, dtype=torch.bool)
        else:
            token_mask = mask.to(device=x0.device, dtype=torch.bool)
            if token_mask.shape != x0.shape:
                raise ValueError("mask must have the same shape as x0")

        x0_compact, x0_allowed = self._compact_indices(x0)
        xt_compact, xt_allowed = self._compact_indices(xt)
        token_mask &= x0_allowed & xt_allowed
        if not torch.any(token_mask):
            raise ValueError("UDLM loss mask selects no tokens from the diffusion alphabet")

        log_x_theta = self.clean_log_probs(logits)
        dtype = log_x_theta.dtype
        alpha = (1.0 - t.to(device=logits.device, dtype=dtype))[:, None, None]
        one_minus_alpha = 1.0 - alpha
        vocab_size = float(self.diffusion_vocab_size)

        x0_one_hot = F.one_hot(
            x0_compact, num_classes=self.diffusion_vocab_size
        ).to(dtype=dtype)
        x_bar = vocab_size * alpha * x0_one_hot + one_minus_alpha

        # log(N * alpha * p_theta + 1 - alpha), kept stable when p is tiny.
        log_signal = torch.log(alpha * vocab_size) + log_x_theta
        log_floor = torch.log(one_minus_alpha).expand_as(log_signal)
        log_x_bar_theta = torch.logaddexp(log_signal, log_floor)
        log_x_bar = torch.log(x_bar)

        gather_index = xt_compact.unsqueeze(-1)
        log_x_bar_i = torch.gather(log_x_bar, -1, gather_index)
        log_x_bar_theta_i = torch.gather(log_x_bar_theta, -1, gather_index)
        log_r = log_x_bar - log_x_bar_i
        log_s = log_x_bar_theta - log_x_bar_theta_i
        u = log_s - log_r
        phi = _expm1_minus_x(u)
        phi.scatter_(-1, gather_index, 0.0)

        positive_coefficient = 1.0 / (vocab_size * alpha)
        per_token = (positive_coefficient * (log_r.exp() * phi).sum(-1, keepdim=True)).squeeze(-1)
        per_token = torch.where(token_mask, per_token, torch.zeros_like(per_token))

        return per_token

    def loss(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        global_mean: bool = False,
    ) -> torch.Tensor:
        """Reduce the UDLM loss with the same contract as BioNeMo MDLM.

        ``global_mean=True`` returns one scalar weighted over all selected
        tokens.  Otherwise this returns one length-normalized value per batch
        element, which the Lightning module subsequently averages.
        """

        if mask is None:
            token_mask = torch.ones_like(x0, dtype=torch.bool)
        else:
            token_mask = mask.to(device=x0.device, dtype=torch.bool)
        _, x0_allowed = self._compact_indices(x0)
        _, xt_allowed = self._compact_indices(xt)
        token_mask &= x0_allowed & xt_allowed
        per_token = self.loss_per_token(logits, x0, xt, t, mask=token_mask)
        if global_mean:
            return per_token.sum() / token_mask.sum()
        return per_token.sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1)

    def posterior_probs(
        self,
        logits: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute ``p_theta(z_s | z_t)`` on the compact diffusion alphabet."""

        if logits.shape[:-1] != xt.shape:
            raise ValueError("logits and xt shapes are inconsistent")
        if t.shape != (xt.shape[0],) or s.shape != (xt.shape[0],):
            raise ValueError("t and s must each have one value per batch element")
        if torch.any(s < 0) or torch.any(t > 1) or torch.any(s >= t):
            raise ValueError("posterior requires 0 <= s < t <= 1")

        log_x_theta = self.clean_log_probs(logits, temperature=temperature)
        x_theta = log_x_theta.exp()
        xt_compact, _ = self._compact_indices(xt)
        xt_one_hot = F.one_hot(
            xt_compact, num_classes=self.diffusion_vocab_size
        ).to(dtype=x_theta.dtype)

        alpha_t = self.alpha(t.to(device=logits.device, dtype=x_theta.dtype))[:, None, None]
        alpha_s = self.alpha(s.to(device=logits.device, dtype=x_theta.dtype))[:, None, None]
        alpha_t_given_s = alpha_t / alpha_s
        uniform_mass = 1.0 / float(self.diffusion_vocab_size)

        transition_to_xt = (
            alpha_t_given_s * xt_one_hot
            + (1.0 - alpha_t_given_s) * uniform_mass
        )
        predicted_marginal_s = alpha_s * x_theta + (1.0 - alpha_s) * uniform_mass
        posterior = transition_to_xt * predicted_marginal_s
        return posterior / posterior.sum(dim=-1, keepdim=True)

    def step(
        self,
        logits: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        *,
        mutable_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one reverse transition while clamping immutable context."""

        posterior = self.posterior_probs(
            logits, xt, t, s, temperature=temperature
        )
        sampled_compact = torch.multinomial(
            posterior.reshape(-1, self.diffusion_vocab_size),
            num_samples=1,
            generator=generator,
        ).reshape_as(xt)
        sampled = self.diffusion_token_ids[sampled_compact]
        if mutable_mask is None:
            return sampled
        mutable_mask = mutable_mask.to(device=xt.device, dtype=torch.bool)
        if mutable_mask.shape != xt.shape:
            raise ValueError("mutable_mask must have the same shape as xt")
        return torch.where(mutable_mask, sampled, xt)


# Short name used throughout the project and paper.
UDLM = ContinuousUniformDiffusion
