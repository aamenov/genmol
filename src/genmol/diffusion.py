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

import math
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


def _log_weighted_expm1_minus_x_value(
    log_weight: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Compute the weighted phi value in log space without building a graph."""

    if log_weight.shape != x.shape:
        log_weight, x = torch.broadcast_tensors(log_weight, x)

    zero = x == 0
    small = (x.abs() < 1e-3) & ~zero
    large_positive = x > 20.0
    large_negative = x < -20.0
    middle = ~(zero | small | large_positive | large_negative)

    # ``torch.where`` evaluates both inputs eagerly.  Give every inactive
    # branch benign finite arguments so an overflow there cannot contaminate
    # the selected branch's backward pass.
    small_x = torch.where(small, x, x.new_tensor(5e-4))
    small_weight = torch.where(small, log_weight, torch.zeros_like(log_weight))
    polynomial = 0.5 + small_x * (
        1.0 / 6.0
        + small_x * (1.0 / 24.0 + small_x / 120.0)
    )
    small_value = (
        small_weight + 2.0 * small_x.abs().log() + polynomial.log()
    ).exp()

    middle_x = torch.where(middle, x, x.new_tensor(1.0))
    middle_weight = torch.where(middle, log_weight, torch.zeros_like(log_weight))
    middle_value = (
        middle_weight + _expm1_minus_x(middle_x).log()
    ).exp()

    positive_x = torch.where(large_positive, x, x.new_tensor(21.0))
    positive_weight = torch.where(
        large_positive, log_weight, torch.zeros_like(log_weight)
    )
    positive_correction = torch.log1p(
        -(1.0 + positive_x) * torch.exp(-positive_x)
    )
    positive_value = (positive_weight + positive_x + positive_correction).exp()

    negative_x = torch.where(large_negative, x, x.new_tensor(-21.0))
    negative_weight = torch.where(
        large_negative, log_weight, torch.zeros_like(log_weight)
    )
    negative_log_phi = torch.logaddexp((-1.0 - negative_x).log(), negative_x)
    negative_value = (negative_weight + negative_log_phi).exp()

    result = torch.zeros_like(x)
    result = torch.where(small, small_value, result)
    result = torch.where(middle, middle_value, result)
    result = torch.where(large_positive, positive_value, result)
    return torch.where(large_negative, negative_value, result)


def _log_weighted_expm1_derivative(
    log_weight: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Stably compute ``exp(log_weight) * expm1(x)`` with its sign."""

    positive = x > 0
    negative = x < 0

    regular_positive = positive & (x <= 20.0)
    regular_positive_x = torch.where(
        regular_positive, x, x.new_tensor(1.0)
    )
    regular_positive_weight = torch.where(
        regular_positive, log_weight, torch.zeros_like(log_weight)
    )
    regular_positive_value = (
        regular_positive_weight + torch.expm1(regular_positive_x).log()
    ).exp()

    large_positive = x > 20.0
    large_positive_x = torch.where(
        large_positive, x, x.new_tensor(21.0)
    )
    large_positive_weight = torch.where(
        large_positive, log_weight, torch.zeros_like(log_weight)
    )
    large_positive_value = (
        large_positive_weight
        + large_positive_x
        + torch.log1p(-torch.exp(-large_positive_x))
    ).exp()

    negative_x = torch.where(negative, x, x.new_tensor(-1.0))
    negative_weight = torch.where(
        negative, log_weight, torch.zeros_like(log_weight)
    )
    negative_value = (
        negative_weight + (-torch.expm1(negative_x)).log()
    ).exp()

    result = torch.zeros_like(x)
    result = torch.where(regular_positive, regular_positive_value, result)
    result = torch.where(large_positive, large_positive_value, result)
    return torch.where(negative, -negative_value, result)


class _LogWeightedExpm1MinusX(torch.autograd.Function):
    """First-order-stable autograd for weighted ``exp(x)-1-x``."""

    @staticmethod
    def forward(
        ctx, log_weight: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        value = _log_weighted_expm1_minus_x_value(log_weight, x)
        ctx.save_for_backward(log_weight, x, value)
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        log_weight, x, value = ctx.saved_tensors
        derivative_x = _log_weighted_expm1_derivative(log_weight, x)
        return grad_output * value, grad_output * derivative_x


def _log_weighted_expm1_minus_x(
    log_weight: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Stably compute ``exp(log_weight) * (exp(x) - 1 - x)``.

    Both the value and first derivatives combine exponents before
    exponentiation.  This avoids ``inf * 0`` in the forward pass and avoids the
    ``output / phi`` intermediate produced by automatic differentiation of a
    naive ``exp(log_weight + log(phi))`` expression.
    """

    log_weight, x = torch.broadcast_tensors(log_weight, x)
    return _LogWeightedExpm1MinusX.apply(log_weight, x)


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
        outside_vocabulary = (token_ids < 0) | (token_ids >= self.num_classes)
        if torch.any(outside_vocabulary):
            invalid = torch.unique(token_ids[outside_vocabulary]).tolist()
            raise ValueError(f"token IDs outside the model vocabulary: {invalid}")
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


class ContinuousCategoricalDiffusion(ContinuousUniformDiffusion):
    """Exact rank-one categorical diffusion with stationary distribution ``pi``.

    This class implements our arbitrary-``pi`` extension of continuous-time
    UDLM; it is a derived experimental method, not a claim from the UDLM paper,
    and its molecular-generation performance is not yet established.  On the
    compact active alphabet its transition kernel is

    ``Q_{s,t}(j, i) = a * 1[j=i] + (1-a) * pi[i]``,

    where ``a = alpha(t) / alpha(s)``.  Consequently ``pi`` is stationary and
    ``q(z_t | x) = alpha(t) delta_x + (1-alpha(t)) pi``.

    Parameters
    ----------
    num_classes:
        Size of the model output vocabulary.
    stationary_probs:
        One strictly positive probability per *active compact* token, ordered
        by increasing model token ID after ``excluded_token_ids`` are removed.
        The values must sum to one.  Requiring the compact representation makes
        zero probability for excluded model tokens explicit rather than
        silently renormalizing a full-vocabulary distribution.
    excluded_token_ids:
        Model-token IDs outside the categorical diffusion alphabet.  They may
        appear only at immutable positions during corruption and reverse steps.
    sampling_eps, noise_eps, antithetic_sampling:
        As in :class:`ContinuousUniformDiffusion`.  Unlike the legacy uniform
        implementation, the loss here uses the derivative of the actual
        release schedule: ``beta(t) = (1-noise_eps) / alpha(t)``.

    Notes
    -----
    ``ContinuousUniformDiffusion`` is intentionally not refactored through this
    class.  Its valid-input behavior, state dictionary, and ``UDLM`` alias
    remain compatible with existing checkpoints and reference tests; shared
    token-range validation now rejects invalid negative IDs in both classes.

    :meth:`loss` returns the model-dependent continuous-time integral.  The
    residual-clean-mass schedule also has a small terminal
    ``KL(q(z_1 | x_0) || pi)`` exposed by :meth:`endpoint_prior_kl`.  That term
    is independent of model parameters, so omitting it does not change training
    gradients, but callers reporting a schedule-consistent full NELBO must add
    it explicitly.
    """

    def __init__(
        self,
        num_classes: int,
        stationary_probs: Sequence[float] | torch.Tensor,
        *,
        excluded_token_ids: Iterable[int] = (),
        sampling_eps: float = 1e-3,
        noise_eps: float = 1e-3,
        antithetic_sampling: bool = True,
    ) -> None:
        super().__init__(
            num_classes,
            excluded_token_ids=excluded_token_ids,
            sampling_eps=sampling_eps,
            noise_eps=noise_eps,
            antithetic_sampling=antithetic_sampling,
        )
        probabilities = (
            torch.as_tensor(stationary_probs, dtype=torch.float64, device="cpu")
            .detach()
            .clone()
        )
        expected_shape = (self.diffusion_vocab_size,)
        if probabilities.shape != expected_shape:
            raise ValueError(
                "stationary_probs must have one entry per active compact token: "
                f"expected {expected_shape}, received {tuple(probabilities.shape)}"
            )
        if not torch.isfinite(probabilities).all():
            raise ValueError("stationary_probs must contain only finite values")
        if torch.any(probabilities <= 0):
            raise ValueError(
                "stationary_probs must have full support on the active alphabet"
            )
        probability_sum = probabilities.sum()
        if not torch.isclose(
            probability_sum,
            torch.ones_like(probability_sum),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                "stationary_probs must sum to one; "
                f"received {probability_sum.item():.17g}"
            )
        # Normalize only accepted floating-point summation drift.  Clone above
        # ensures later mutation of the constructor input cannot alter the
        # registered process prior.
        self.register_buffer("stationary_probs", probabilities / probability_sum)

    def _apply(self, fn, recurse: bool = True):
        """Move module state while retaining the stationary prior in float64.

        The model algebra may run in float32, but narrowing the stored prior can
        erase valid rare categories.  PyTorch implements ``to()``, ``float()``,
        ``half()``, and device moves through ``_apply``; temporarily removing
        this buffer lets the rest of the module follow the requested transform
        while the prior changes device but returns to float64.
        """

        prior = self._buffers.pop("stationary_probs", None)
        try:
            result = super()._apply(fn, recurse=recurse)
        except Exception:
            if prior is not None:
                self._buffers["stationary_probs"] = prior
            raise
        if prior is not None:
            # Infer the requested device from an inherited integer buffer,
            # which ``_apply`` moves but never narrows.  Applying ``fn`` to the
            # prior first would already destroy sub-float32 probabilities.
            moved_prior = prior.to(
                device=self.diffusion_token_ids.device,
                dtype=torch.float64,
            )
            self._buffers["stationary_probs"] = moved_prior
        return result

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Return the exact jump rate ``-alpha'(t) / alpha(t)``."""

        # Evaluate the affine denominator directly.  For very small float32
        # times ``exp(-sigma(t))`` can round to exactly one even though the
        # analytically equivalent denominator still carries the intended
        # schedule definition.
        return (1.0 - self.noise_eps) / (
            1.0 - (1.0 - self.noise_eps) * t
        )

    def sample_prior(
        self,
        shape: Sequence[int] | torch.Size,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample iid model token IDs exactly from the stationary prior."""

        if device is None:
            device = self.stationary_probs.device
        sample_shape = tuple(shape)
        num_samples = math.prod(sample_shape)
        compact = torch.multinomial(
            self.stationary_probs.to(device=device),
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        ).reshape(sample_shape)
        return self.diffusion_token_ids.to(device=device)[compact]

    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        *,
        mutable_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample ``z_t`` by refreshing mutable tokens from ``pi``."""

        return super().forward_process(
            x0,
            t,
            mutable_mask=mutable_mask,
            generator=generator,
        )

    def loss_per_token(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Evaluate our exact arbitrary-``pi`` continuous-time loss.

        Write ``m_x(j) = alpha 1[x=j] + (1-alpha) pi_j`` and define the
        density relative to the stationary measure as ``m_bar = m / pi``.
        For observed current token ``i``, let

        ``R_j = m_bar_x(j) / m_bar_x(i)`` and
        ``S_j = m_bar_theta(j) / m_bar_theta(i)``.

        The returned integrand is

        ``beta sum_{j != i} pi_j R_j phi(log(S_j)-log(R_j))``,

        with ``phi(u)=exp(u)-1-u`` and schedule-consistent
        ``beta=-alpha'/alpha``.  Log density ratios and ``expm1`` make the
        computation stable near an oracle denoiser.
        """

        if logits.shape[:-1] != x0.shape or xt.shape != x0.shape:
            raise ValueError("logits, x0, and xt shapes are inconsistent")
        if t.shape != (x0.shape[0],):
            raise ValueError(f"t must have shape ({x0.shape[0]},), received {tuple(t.shape)}")
        if torch.any((t <= 0) | (t >= 1)):
            raise ValueError("continuous-time categorical loss requires 0 < t < 1")

        if mask is None:
            token_mask = torch.ones_like(x0, dtype=torch.bool)
        else:
            token_mask = mask.to(device=x0.device, dtype=torch.bool)
            if token_mask.shape != x0.shape:
                raise ValueError("mask must have the same shape as x0")

        x0_compact, x0_allowed = self._compact_indices(x0)
        xt_compact, xt_allowed = self._compact_indices(xt)
        selected_excluded = token_mask & (~x0_allowed | ~xt_allowed)
        if torch.any(selected_excluded):
            bad_ids = torch.unique(
                torch.cat(
                    (
                        x0[token_mask & ~x0_allowed],
                        xt[token_mask & ~xt_allowed],
                    )
                )
            ).tolist()
            raise ValueError(
                "categorical diffusion loss selects excluded token IDs; "
                f"mask immutable context explicitly, received {bad_ids}"
            )
        if not torch.any(token_mask):
            raise ValueError(
                "categorical diffusion loss mask selects no active tokens"
            )

        log_x_theta = self.clean_log_probs(logits)
        dtype = log_x_theta.dtype
        device = logits.device
        # Take logs before narrowing.  A valid float64 prior may contain masses
        # below float32's range; casting probabilities first would turn those
        # entries into zero and manufacture ``-inf`` density ratios.
        log_pi = self.stationary_probs.log().to(device=device, dtype=dtype)
        typed_time = t.to(device=device, dtype=dtype)
        refresh = (1.0 - self.noise_eps) * typed_time
        log_alpha = torch.log1p(-refresh)[:, None, None]
        log_refresh = refresh.log()[:, None, None]

        # m_theta / pi = alpha * p_theta / pi + (1-alpha).
        log_bar_theta = torch.logaddexp(
            log_alpha + log_x_theta - log_pi,
            log_refresh.expand_as(log_x_theta),
        )

        # m_x / pi differs from (1-alpha) only at the clean class.  Construct
        # that one entry without ever taking log(0) from a one-hot vector.
        log_bar_x = log_refresh.expand_as(log_x_theta).clone()
        clean_log_pi = log_pi[x0_compact]
        clean_log_density = torch.logaddexp(
            log_refresh.squeeze(-1).expand_as(clean_log_pi),
            log_alpha.squeeze(-1).expand_as(clean_log_pi) - clean_log_pi,
        )
        log_bar_x.scatter_(-1, x0_compact.unsqueeze(-1), clean_log_density.unsqueeze(-1))

        current_index = xt_compact.unsqueeze(-1)
        log_r = log_bar_x - torch.gather(log_bar_x, -1, current_index)
        log_s = log_bar_theta - torch.gather(log_bar_theta, -1, current_index)
        u = log_s - log_r
        weighted_phi = _log_weighted_expm1_minus_x(log_pi + log_r, u)
        weighted_phi = weighted_phi.scatter(-1, current_index, 0.0)

        rate = ((1.0 - self.noise_eps) * (-log_alpha).exp()).squeeze(-1)
        per_token = rate * weighted_phi.sum(dim=-1)
        return torch.where(token_mask, per_token, torch.zeros_like(per_token))

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
        """Reduce the model-dependent CT integral over explicitly selected tokens.

        Excluded control tokens are legal only where ``mask`` is false.  This
        stricter contract prevents an accidentally unmasked BOS/EOS position
        from silently disappearing from the training objective.
        """

        if mask is None:
            token_mask = torch.ones_like(x0, dtype=torch.bool)
        else:
            token_mask = mask.to(device=x0.device, dtype=torch.bool)
            if token_mask.shape != x0.shape:
                raise ValueError("mask must have the same shape as x0")
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
        """Compute the exact model reverse posterior on the active alphabet.

        For candidate earlier state ``j`` and observed current state ``i``,
        Bayes' rule uses likelihood
        ``a 1[j=i] + (1-a) pi_i``.  In particular the refresh likelihood is
        the scalar ``pi_i``; substituting the candidate-dependent ``pi_j`` is
        an incorrect but easy-to-miss implementation.
        """

        if logits.shape[:-1] != xt.shape:
            raise ValueError("logits and xt shapes are inconsistent")
        if t.shape != (xt.shape[0],) or s.shape != (xt.shape[0],):
            raise ValueError("t and s must each have one value per batch element")
        if torch.any(s < 0) or torch.any(t > 1) or torch.any(s >= t):
            raise ValueError("posterior requires 0 <= s < t <= 1")

        xt_compact, xt_allowed = self._compact_indices(xt)
        if not torch.all(xt_allowed):
            bad_ids = torch.unique(xt[~xt_allowed]).tolist()
            raise ValueError(
                "posterior current states must belong to the active alphabet; "
                f"received {bad_ids}"
            )

        log_x_theta = self.clean_log_probs(logits, temperature=temperature)
        dtype = log_x_theta.dtype
        device = logits.device
        log_pi = self.stationary_probs.log().to(device=device, dtype=dtype)
        typed_t = t.to(device=device, dtype=dtype)
        typed_s = s.to(device=device, dtype=dtype)
        refresh_t = (1.0 - self.noise_eps) * typed_t
        refresh_s = (1.0 - self.noise_eps) * typed_s
        log_alpha_t = torch.log1p(-refresh_t)[:, None, None]
        log_alpha_s = torch.log1p(-refresh_s)[:, None, None]
        log_alpha_t_given_s = (log_alpha_t - log_alpha_s).clamp_max(0.0)
        log_transition_refresh = torch.log(
            -torch.expm1(log_alpha_t_given_s)
        )
        log_pi_i = log_pi[xt_compact].unsqueeze(-1)
        log_transition = (
            log_transition_refresh + log_pi_i
        ).expand_as(log_x_theta).clone()
        current_index = xt_compact.unsqueeze(-1)
        current_log_transition = torch.logaddexp(
            log_transition_refresh + log_pi_i,
            log_alpha_t_given_s.expand_as(log_pi_i),
        )
        log_transition.scatter_(-1, current_index, current_log_transition)

        log_predicted_marginal_s = torch.logaddexp(
            log_alpha_s + log_x_theta,
            refresh_s.log()[:, None, None] + log_pi,
        )
        log_posterior = log_transition + log_predicted_marginal_s
        return (log_posterior - log_posterior.logsumexp(-1, keepdim=True)).exp()

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
        """Draw one reverse transition while preserving excluded context."""

        _, active = self._compact_indices(xt)
        if mutable_mask is None:
            mutable_mask = torch.ones_like(xt, dtype=torch.bool)
        else:
            mutable_mask = mutable_mask.to(device=xt.device, dtype=torch.bool)
            if mutable_mask.shape != xt.shape:
                raise ValueError("mutable_mask must have the same shape as xt")
        if torch.any(mutable_mask & ~active):
            bad_ids = torch.unique(xt[mutable_mask & ~active]).tolist()
            raise ValueError(f"mutable positions contain excluded token IDs: {bad_ids}")

        # posterior_probs is strict about its current-state support.  Substitute
        # an arbitrary active token only at positions that are clamped below.
        posterior_xt = torch.where(active, xt, self.diffusion_token_ids[0])
        posterior = self.posterior_probs(
            logits,
            posterior_xt,
            t,
            s,
            temperature=temperature,
        )
        sampled_compact = torch.multinomial(
            posterior.reshape(-1, self.diffusion_vocab_size),
            num_samples=1,
            generator=generator,
        ).reshape_as(xt)
        sampled = self.diffusion_token_ids[sampled_compact]
        return torch.where(mutable_mask, sampled, xt)

    def endpoint_prior_kl(
        self,
        x0: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Return ``KL(q(z_1 | x0) || pi)`` from residual ``noise_eps``.

        This exposes the small, nonzero terminal-prior mismatch of the release
        schedule instead of treating ``alpha(1)=noise_eps`` as exactly zero.
        The result has the shape of ``x0``; positions outside ``mask`` are zero.
        """

        x0_compact, active = self._compact_indices(x0)
        if mask is None:
            token_mask = torch.ones_like(x0, dtype=torch.bool)
        else:
            token_mask = mask.to(device=x0.device, dtype=torch.bool)
            if token_mask.shape != x0.shape:
                raise ValueError("mask must have the same shape as x0")
        selected_excluded = token_mask & ~active
        if torch.any(selected_excluded):
            bad_ids = torch.unique(x0[selected_excluded]).tolist()
            raise ValueError(
                "endpoint prior KL selects excluded clean tokens; "
                "mask immutable context explicitly; "
                f"received {bad_ids}"
            )
        pi_x = self.stationary_probs.to(device=x0.device)[x0_compact]
        residual = torch.as_tensor(
            self.noise_eps,
            device=x0.device,
            dtype=pi_x.dtype,
        )
        q_x = residual + (1.0 - residual) * pi_x
        per_token = (
            q_x * (q_x.log() - pi_x.log())
            + (1.0 - residual)
            * (1.0 - pi_x)
            * torch.log1p(-residual)
        )
        return torch.where(token_mask, per_token, torch.zeros_like(per_token))


# Short name used throughout the project and paper.
UDLM = ContinuousUniformDiffusion
