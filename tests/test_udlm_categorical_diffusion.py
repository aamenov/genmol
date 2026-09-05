import math

import pytest
import torch
from torch.nn import functional as F

from genmol.diffusion import (
    ContinuousCategoricalDiffusion,
    ContinuousUniformDiffusion,
    _log_weighted_expm1_minus_x,
)


def _rank_one_kernel(pi: torch.Tensor, coefficient: torch.Tensor) -> torch.Tensor:
    size = pi.numel()
    return coefficient * torch.eye(size, dtype=pi.dtype) + (
        1.0 - coefficient
    ) * pi.expand(size, -1)


def _direct_reverse_posterior(
    clean_probs: torch.Tensor,
    pi: torch.Tensor,
    current: int,
    alpha_s: torch.Tensor,
    alpha_t: torch.Tensor,
) -> torch.Tensor:
    marginal_s = alpha_s * clean_probs + (1.0 - alpha_s) * pi
    coefficient = alpha_t / alpha_s
    # The refresh likelihood is the scalar pi[current] for every candidate
    # earlier state.  Using the vector pi here is the regression this helper is
    # intentionally designed to catch.
    likelihood = torch.full_like(pi, (1.0 - coefficient) * pi[current])
    likelihood[current] += coefficient
    posterior = likelihood * marginal_s
    return posterior / posterior.sum()


def _schedule_consistent_uniform_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    *,
    noise_eps: float,
) -> torch.Tensor:
    """Literal uniform specialization of the arbitrary-pi exact loss."""

    vocab_size = logits.shape[-1]
    clean_probs = logits.log_softmax(-1).exp()
    alpha = (1.0 - (1.0 - noise_eps) * t)[:, None, None]
    x_bar = vocab_size * alpha * F.one_hot(x0, vocab_size) + 1.0 - alpha
    x_bar_theta = vocab_size * alpha * clean_probs + 1.0 - alpha
    current = xt.unsqueeze(-1)
    log_r = x_bar.log() - torch.gather(x_bar.log(), -1, current)
    log_s = x_bar_theta.log() - torch.gather(x_bar_theta.log(), -1, current)
    u = log_s - log_r
    phi = torch.expm1(u) - u
    phi.scatter_(-1, current, 0.0)
    coefficient = (1.0 - noise_eps) / (vocab_size * alpha)
    return (coefficient * (log_r.exp() * phi).sum(-1, keepdim=True)).squeeze(-1)


def _direct_categorical_rate_kl(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    pi: torch.Tensor,
    *,
    noise_eps: float,
) -> torch.Tensor:
    """Direct CTMC reverse-rate KL, independent of density-ratio algebra."""

    clean_probs = logits.log_softmax(-1).exp()
    refresh = ((1.0 - noise_eps) * t)[:, None, None]
    alpha = 1.0 - refresh
    true_marginal = alpha * F.one_hot(x0, pi.numel()) + refresh * pi
    model_marginal = alpha * clean_probs + refresh * pi
    current = xt.unsqueeze(-1)
    pi_current = pi[xt].unsqueeze(-1)
    beta = ((1.0 - noise_eps) / (1.0 - (1.0 - noise_eps) * t))[
        :, None, None
    ]
    true_rates = (
        beta
        * pi_current
        * true_marginal
        / torch.gather(true_marginal, -1, current)
    )
    model_rates = (
        beta
        * pi_current
        * model_marginal
        / torch.gather(model_marginal, -1, current)
    )
    true_rates = true_rates.scatter(-1, current, 0.0)
    model_rates = model_rates.scatter(-1, current, 0.0)
    return (
        true_rates * (true_rates.log() - model_rates.log())
        - true_rates
        + model_rates
    ).nan_to_num(nan=0.0).sum(-1)


@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        ([0.5, 0.5], "one entry per active compact token"),
        ([0.7, 0.3, 0.0], "full support"),
        ([0.7, 0.4, -0.1], "full support"),
        ([0.7, float("nan"), 0.3], "finite"),
        ([0.6, 0.2, 0.1], "sum to one"),
    ],
)
def test_stationary_prior_validation(probabilities, match):
    with pytest.raises(ValueError, match=match):
        ContinuousCategoricalDiffusion(5, probabilities, excluded_token_ids=(0, 4))


def test_compact_prior_is_copied_and_legacy_uniform_state_is_unchanged():
    probabilities = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    process = ContinuousCategoricalDiffusion(
        5,
        probabilities,
        excluded_token_ids=(0, 4),
    )
    probabilities[0] = 0.1

    assert torch.allclose(
        process.stationary_probs,
        torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64),
    )
    assert process.diffusion_token_ids.tolist() == [1, 2, 3]
    assert tuple(ContinuousUniformDiffusion(5).state_dict()) == (
        "diffusion_token_ids",
        "token_to_diffusion_index",
    )


def test_exact_prior_sampling_uses_only_active_model_token_ids():
    process = ContinuousCategoricalDiffusion(
        5,
        [0.72, 0.20, 0.08],
        excluded_token_ids=(0, 3),
    )
    samples = process.sample_prior(
        (100_000,),
        generator=torch.Generator().manual_seed(42),
    )
    frequencies = torch.bincount(samples, minlength=5).double() / samples.numel()

    assert frequencies[0] == 0
    assert frequencies[3] == 0
    assert torch.allclose(
        frequencies[torch.tensor([1, 2, 4])],
        torch.tensor([0.72, 0.20, 0.08], dtype=torch.float64),
        atol=0.004,
        rtol=0,
    )


def test_rank_one_kernel_stationarity_semigroup_and_forward_marginal():
    pi = torch.tensor([0.62, 0.28, 0.10], dtype=torch.float64)
    first = torch.tensor(0.73, dtype=torch.float64)
    second = torch.tensor(0.41, dtype=torch.float64)
    q_first = _rank_one_kernel(pi, first)
    q_second = _rank_one_kernel(pi, second)

    assert torch.allclose(pi @ q_first, pi, atol=1e-14, rtol=1e-14)
    assert torch.allclose(
        q_first @ q_second,
        _rank_one_kernel(pi, first * second),
        atol=1e-14,
        rtol=1e-14,
    )

    process = ContinuousCategoricalDiffusion(
        3,
        pi,
        noise_eps=0.2,
        antithetic_sampling=False,
    )
    t = torch.tensor(0.6, dtype=torch.float64)
    alpha = process.alpha(t)
    exact_marginal = (1.0 - alpha) * pi
    exact_marginal[1] += alpha
    matrix_marginal = F.one_hot(torch.tensor(1), 3).double() @ _rank_one_kernel(
        pi, alpha
    )
    assert torch.allclose(matrix_marginal, exact_marginal, atol=1e-14, rtol=1e-14)

    x0 = torch.ones((100_000, 1), dtype=torch.long)
    sampled = process.forward_process(
        x0,
        torch.full((x0.shape[0],), t.item()),
        generator=torch.Generator().manual_seed(7),
    )
    empirical = torch.bincount(sampled[:, 0], minlength=3).double() / x0.shape[0]
    assert torch.allclose(empirical, exact_marginal, atol=0.005, rtol=0)


def test_skew_prior_posterior_matches_direct_bayes_with_scalar_pi_i():
    pi = torch.tensor([0.72, 0.23, 0.05], dtype=torch.float64)
    process = ContinuousCategoricalDiffusion(3, pi, noise_eps=0.04)
    clean_probs = torch.tensor([0.51, 0.31, 0.18], dtype=torch.float64)
    logits = clean_probs.log().reshape(1, 1, -1)
    current = 2
    xt = torch.tensor([[current]])
    t = torch.tensor([0.83], dtype=torch.float64)
    s = torch.tensor([0.29], dtype=torch.float64)

    actual = process.posterior_probs(logits, xt, t, s)[0, 0]
    expected = _direct_reverse_posterior(
        clean_probs,
        pi,
        current,
        process.alpha(s)[0],
        process.alpha(t)[0],
    )
    # A candidate-dependent pi_j refresh factor gives a materially different
    # result for this deliberately skewed prior.
    coefficient = process.alpha(t)[0] / process.alpha(s)[0]
    wrong_likelihood = (1.0 - coefficient) * pi
    wrong_likelihood[current] += coefficient
    marginal_s = process.alpha(s)[0] * clean_probs + (1.0 - process.alpha(s)[0]) * pi
    wrong = wrong_likelihood * marginal_s
    wrong /= wrong.sum()

    assert torch.allclose(actual, expected, atol=1e-13, rtol=1e-13)
    assert not torch.allclose(actual, wrong, atol=1e-3, rtol=1e-3)
    assert torch.all(actual >= 0)
    assert actual.sum() == pytest.approx(1.0, abs=1e-14)


def test_oracle_denoiser_has_exact_posterior_and_zero_ct_loss():
    pi = torch.tensor([0.65, 0.24, 0.11], dtype=torch.float64)
    process = ContinuousCategoricalDiffusion(3, pi, noise_eps=0.03)
    x0 = torch.tensor([[1, 1, 1]])
    xt = torch.tensor([[0, 1, 2]])
    logits = torch.full((1, 3, 3), -torch.inf, dtype=torch.float64)
    logits.scatter_(-1, x0.unsqueeze(-1), 0.0)
    t = torch.tensor([0.71], dtype=torch.float64)
    s = torch.tensor([0.22], dtype=torch.float64)

    actual_posterior = process.posterior_probs(logits, xt, t, s)[0]
    expected_posterior = torch.stack(
        [
            _direct_reverse_posterior(
                F.one_hot(torch.tensor(1), 3).double(),
                pi,
                current,
                process.alpha(s)[0],
                process.alpha(t)[0],
            )
            for current in xt[0].tolist()
        ]
    )
    loss = process.loss_per_token(logits, x0, xt, t)

    assert torch.allclose(actual_posterior, expected_posterior, atol=1e-13, rtol=1e-13)
    assert torch.equal(loss, torch.zeros_like(loss))


def test_finite_reverse_kl_converges_to_ct_loss_at_first_order():
    pi = torch.tensor([0.63, 0.27, 0.10], dtype=torch.float64)
    process = ContinuousCategoricalDiffusion(3, pi, noise_eps=0.07)
    true_clean = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    model_clean = torch.tensor([0.30, 0.50, 0.20], dtype=torch.float64)
    current = 0
    t_value = 0.71
    t = torch.tensor([t_value], dtype=torch.float64)
    exact_rate = process.loss_per_token(
        model_clean.log().reshape(1, 1, -1),
        torch.tensor([[1]]),
        torch.tensor([[current]]),
        t,
    )[0, 0]

    errors = []
    for step_size in (1e-2, 5e-3, 2.5e-3):
        s = torch.tensor([t_value - step_size], dtype=torch.float64)
        true_posterior = _direct_reverse_posterior(
            true_clean,
            pi,
            current,
            process.alpha(s)[0],
            process.alpha(t)[0],
        )
        model_posterior = _direct_reverse_posterior(
            model_clean,
            pi,
            current,
            process.alpha(s)[0],
            process.alpha(t)[0],
        )
        finite_kl = (
            true_posterior * (true_posterior.log() - model_posterior.log())
        ).sum()
        errors.append(abs(finite_kl / step_size - exact_rate))

    # KL/h = loss + O(h): halving h should approximately halve the error.
    assert errors[1] < 0.55 * errors[0]
    assert errors[2] < 0.55 * errors[1]
    assert errors[2] < 0.006 * exact_rate


def test_uniform_prior_reduces_to_schedule_consistent_uniform_reference():
    noise_eps = 0.17
    process = ContinuousCategoricalDiffusion(
        4,
        torch.full((4,), 0.25, dtype=torch.float64),
        noise_eps=noise_eps,
    )
    logits = torch.randn(
        2,
        3,
        4,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(8),
        requires_grad=True,
    )
    x0 = torch.tensor([[0, 1, 3], [2, 0, 1]])
    xt = torch.tensor([[2, 1, 0], [3, 0, 2]])
    t = torch.tensor([0.19, 0.81], dtype=torch.float64)

    actual = process.loss_per_token(logits, x0, xt, t)
    expected = _schedule_consistent_uniform_loss(
        logits,
        x0,
        xt,
        t,
        noise_eps=noise_eps,
    )
    actual_gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]

    assert torch.allclose(actual, expected, atol=3e-12, rtol=3e-12)
    assert torch.allclose(actual_gradient, expected_gradient, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("time", [1e-3, 0.5, 0.9999])
def test_near_oracle_float32_loss_and_gradients_remain_finite(time):
    process = ContinuousCategoricalDiffusion(
        5,
        [0.799989, 0.15, 0.04, 0.01, 1e-5],
        noise_eps=1e-3,
    )
    x0 = torch.tensor([[0, 1, 4], [3, 2, 0]])
    xt = torch.tensor([[4, 1, 0], [3, 4, 2]])
    logits = torch.full((2, 3, 5), -12.0)
    logits.scatter_(-1, x0.unsqueeze(-1), 12.0)
    logits += 1e-3 * torch.randn(
        logits.shape,
        generator=torch.Generator().manual_seed(19),
    )
    logits.requires_grad_()

    loss = process.loss(
        logits,
        x0,
        xt,
        torch.full((2,), time),
        global_mean=True,
    )
    loss.backward()

    assert loss >= 0
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_float32_value_and_gradient_match_float64_reference():
    pi = [0.61, 0.26, 0.10, 0.029, 0.001]
    x0 = torch.tensor([[0, 1, 4, 2], [3, 2, 0, 1]])
    xt = torch.tensor([[4, 1, 0, 3], [3, 4, 2, 0]])
    t64 = torch.tensor([0.013, 0.93], dtype=torch.float64)
    base_logits = 1.5 * torch.randn(
        2,
        4,
        5,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(17),
    )

    values = []
    gradients = []
    for dtype in (torch.float64, torch.float32):
        process = ContinuousCategoricalDiffusion(5, pi, noise_eps=0.02)
        logits = base_logits.to(dtype).detach().requires_grad_()
        value = process.loss(
            logits,
            x0,
            xt,
            t64.to(dtype),
            global_mean=True,
        )
        gradient = torch.autograd.grad(value, logits)[0]
        values.append(value.detach().double())
        gradients.append(gradient.detach().double())

    assert torch.allclose(values[1], values[0], atol=5e-4, rtol=3e-6)
    assert torch.allclose(gradients[1], gradients[0], atol=2e-5, rtol=3e-5)


def test_randomized_skew_prior_loss_and_gradient_match_direct_rate_kl():
    generator = torch.Generator().manual_seed(123)
    pi = torch.tensor([0.731, 0.190, 0.061, 0.017, 0.001], dtype=torch.float64)
    process = ContinuousCategoricalDiffusion(5, pi, noise_eps=0.04)
    logits = torch.randn(
        4,
        7,
        5,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    x0 = torch.randint(5, (4, 7), generator=generator)
    xt = torch.randint(5, (4, 7), generator=generator)
    t = torch.tensor([0.0012, 0.073, 0.54, 0.997], dtype=torch.float64)

    actual = process.loss_per_token(logits, x0, xt, t)
    expected = _direct_categorical_rate_kl(
        logits,
        x0,
        xt,
        t,
        pi,
        noise_eps=process.noise_eps,
    )
    actual_gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]

    assert torch.allclose(actual, expected, atol=4e-12, rtol=4e-12)
    assert torch.allclose(actual_gradient, expected_gradient, atol=5e-11, rtol=5e-11)


def test_tiny_float32_time_remains_finite_and_matches_float64_reference():
    pi = [0.68, 0.21, 0.09, 0.02]
    base_logits = torch.tensor(
        [[[2.0, -0.5, 0.7, -1.2], [-1.0, 0.1, 1.8, -0.7]]],
        dtype=torch.float64,
    )
    x0 = torch.tensor([[0, 3]])
    xt = torch.tensor([[0, 3]])

    results = []
    gradients = []
    for dtype in (torch.float64, torch.float32):
        process = ContinuousCategoricalDiffusion(4, pi, noise_eps=1e-3)
        logits = base_logits.to(dtype).detach().requires_grad_()
        result = process.loss_per_token(
            logits,
            x0,
            xt,
            torch.tensor([1e-8], dtype=dtype),
        )
        gradient = torch.autograd.grad(result.sum(), logits)[0]
        results.append(result.detach().double())
        gradients.append(gradient.detach().double())

    assert torch.isfinite(results[1]).all()
    assert torch.isfinite(gradients[1]).all()
    assert torch.allclose(results[1], results[0], atol=2e-8, rtol=2e-5)
    assert torch.allclose(gradients[1], gradients[0], atol=2e-8, rtol=2e-5)


def test_log_weighted_phi_handles_zero_and_small_argument_with_huge_weight():
    log_weight = torch.tensor([90.0, 90.0, 90.0], requires_grad=True)
    argument = torch.tensor([0.0, 1e-4, 2e-3], requires_grad=True)

    actual = _log_weighted_expm1_minus_x(log_weight, argument)
    expected_small = math.exp(90.0) * (math.expm1(1e-4) - 1e-4)
    actual.sum().backward()

    assert actual[0] == 0
    assert torch.isfinite(actual).all()
    assert actual[1].detach().item() == pytest.approx(expected_small, rel=2e-4)
    assert torch.isfinite(log_weight.grad).all()
    assert torch.isfinite(argument.grad).all()
    expected_middle_gradient = math.exp(90.0) * math.expm1(2e-3)
    assert argument.grad[2].item() == pytest.approx(
        expected_middle_gradient,
        rel=2e-5,
    )


def test_subnormal_training_time_exact_oracle_loss_is_zero_not_nan():
    process = ContinuousCategoricalDiffusion(3, [0.7, 0.2, 0.1])
    logits = torch.full((1, 1, 3), -torch.inf)
    logits[..., 0] = 0.0

    loss = process.loss_per_token(
        logits,
        torch.tensor([[0]]),
        torch.tensor([[1]]),
        torch.tensor([1e-39]),
    )

    assert torch.equal(loss, torch.zeros_like(loss))


def test_subnormal_training_time_nonoracle_gradient_is_finite():
    process = ContinuousCategoricalDiffusion(3, [0.7, 0.2, 0.1])
    logits = torch.tensor(
        [[[0.0, -97.625865, -torch.inf]]],
        requires_grad=True,
    )

    loss = process.loss_per_token(
        logits,
        torch.tensor([[0]]),
        torch.tensor([[1]]),
        torch.tensor([1e-39]),
    ).sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_extreme_float64_prior_avoids_weight_times_phi_overflow():
    pi = torch.tensor([1.0 - 2e-17, 1e-17, 1e-17], dtype=torch.float64)
    process = ContinuousCategoricalDiffusion(3, pi, noise_eps=1e-3)
    logits = torch.tensor([[[-50.0, -50.0, 50.0]]], requires_grad=True)
    x0 = torch.tensor([[1]])
    xt = torch.tensor([[1]])
    t = torch.tensor([0.001])

    actual = process.loss_per_token(logits, x0, xt, t)
    expected = _direct_categorical_rate_kl(
        logits.double(),
        x0,
        xt,
        t.double(),
        pi,
        noise_eps=process.noise_eps,
    )
    actual.sum().backward()

    assert torch.isfinite(actual).all()
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(actual.double(), expected, atol=2e-3, rtol=8e-6)


def test_log_domain_posterior_preserves_sub_float32_prior_information():
    pi = torch.tensor([1.0 - 2e-46, 1e-46, 1e-46], dtype=torch.float64)
    process = ContinuousCategoricalDiffusion(3, pi)
    logits = torch.tensor([[[-50.0, -50.0, 50.0]]])
    xt = torch.tensor([[1]])
    t = torch.tensor([0.8])
    s = torch.tensor([0.2])

    actual = process.posterior_probs(logits, xt, t, s)[0, 0]
    expected = _direct_reverse_posterior(
        logits[0, 0].double().softmax(-1),
        pi,
        1,
        process.alpha(s.double())[0],
        process.alpha(t.double())[0],
    )

    assert torch.allclose(actual.double(), expected, atol=2e-7, rtol=2e-6)
    assert torch.all(actual > 0)


def test_endpoint_prior_kl_matches_formula_and_small_residual_asymptotic():
    pi = torch.tensor([0.70, 0.20, 0.10], dtype=torch.float64)
    residual = 1e-3
    process = ContinuousCategoricalDiffusion(3, pi, noise_eps=residual)
    x0 = torch.tensor([0, 2])
    actual = process.endpoint_prior_kl(x0)

    expected = []
    for clean_token in x0.tolist():
        terminal = (1.0 - residual) * pi.clone()
        terminal[clean_token] += residual
        expected.append((terminal * (terminal.log() - pi.log())).sum())
    expected = torch.stack(expected)
    leading_order = 0.5 * residual**2 * (pi[x0].reciprocal() - 1.0)

    assert torch.allclose(actual, expected, atol=2e-16, rtol=2e-12)
    assert torch.allclose(actual, leading_order, atol=0, rtol=0.003)
    assert torch.all(actual > 0)


def test_endpoint_prior_kl_masks_immutable_excluded_context():
    process = ContinuousCategoricalDiffusion(
        5,
        [0.70, 0.20, 0.10],
        excluded_token_ids=(0, 4),
    )
    x0 = torch.tensor([[0, 1, 3, 4]])
    active_mask = torch.tensor([[False, True, True, False]])

    masked = process.endpoint_prior_kl(x0, mask=active_mask)

    assert torch.equal(masked[~active_mask], torch.zeros(2, dtype=masked.dtype))
    assert torch.all(masked[active_mask] > 0)
    with pytest.raises(ValueError, match="selects excluded clean tokens"):
        process.endpoint_prior_kl(x0)


@pytest.mark.parametrize("invalid_token", [-2, -1, 5, 19])
def test_token_ids_outside_model_vocabulary_are_rejected(invalid_token):
    process = ContinuousCategoricalDiffusion(5, [0.4, 0.3, 0.2, 0.07, 0.03])
    x0 = torch.tensor([[invalid_token]])
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        process.loss_per_token(
            torch.zeros(1, 1, 5),
            x0,
            torch.zeros_like(x0),
            torch.tensor([0.5]),
        )


def test_loss_requires_explicit_false_mask_for_excluded_context():
    process = ContinuousCategoricalDiffusion(
        5,
        [0.60, 0.30, 0.10],
        excluded_token_ids=(0, 4),
    )
    x0 = torch.tensor([[0, 1, 2, 4]])
    xt = x0.clone()
    logits = torch.zeros(1, 4, 5)
    t = torch.tensor([0.5])
    active_mask = torch.tensor([[False, True, True, False]])

    with pytest.raises(ValueError, match="selects excluded token IDs"):
        process.loss(logits, x0, xt, t)
    value = process.loss(logits, x0, xt, t, mask=active_mask, global_mean=True)
    expected = process.loss_per_token(logits, x0, xt, t, mask=active_mask)[
        active_mask
    ].mean()

    assert torch.isfinite(value)
    assert value == expected


def test_categorical_state_dict_round_trip_preserves_prior_and_outputs():
    original = ContinuousCategoricalDiffusion(
        5,
        [0.61, 0.29, 0.10],
        excluded_token_ids=(0, 4),
        noise_eps=0.07,
    )
    restored = ContinuousCategoricalDiffusion(
        5,
        [1 / 3, 1 / 3, 1 / 3],
        excluded_token_ids=(0, 4),
        noise_eps=0.07,
    )
    restored.load_state_dict(original.state_dict())
    logits = torch.tensor([[[0.0, 0.3, -0.7, 1.1, -2.0]]], dtype=torch.float64)
    xt = torch.tensor([[2]])
    t = torch.tensor([0.8], dtype=torch.float64)
    s = torch.tensor([0.2], dtype=torch.float64)

    assert torch.equal(restored.stationary_probs, original.stationary_probs)
    assert torch.equal(
        restored.posterior_probs(logits, xt, t, s),
        original.posterior_probs(logits, xt, t, s),
    )


def test_dtype_conversion_never_narrows_the_stored_stationary_prior():
    tiny = 1e-46
    process = ContinuousCategoricalDiffusion(
        3,
        [1.0 - 2 * tiny, tiny, tiny],
    )

    process.float()
    assert process.stationary_probs.dtype == torch.float64
    assert torch.equal(
        process.stationary_probs,
        torch.tensor([1.0 - 2 * tiny, tiny, tiny], dtype=torch.float64),
    )
    process.half()
    assert process.stationary_probs.dtype == torch.float64
    assert torch.all(process.stationary_probs > 0)


def test_step_empirical_frequencies_match_exact_reverse_posterior():
    process = ContinuousCategoricalDiffusion(3, [0.72, 0.23, 0.05])
    sample_count = 80_000
    logits = torch.tensor([[[0.4, 1.2, -0.3]]]).expand(sample_count, -1, -1)
    xt = torch.full((sample_count, 1), 2)
    t = torch.full((sample_count,), 0.81)
    s = torch.full((sample_count,), 0.33)
    expected = process.posterior_probs(logits[:1], xt[:1], t[:1], s[:1])[0, 0]

    sampled = process.step(
        logits,
        xt,
        t,
        s,
        generator=torch.Generator().manual_seed(314),
    )
    empirical = torch.bincount(sampled[:, 0], minlength=3).float() / sample_count

    assert torch.allclose(empirical, expected, atol=0.004, rtol=0)


def test_excluded_tokens_are_allowed_only_as_immutable_context():
    process = ContinuousCategoricalDiffusion(
        5,
        [0.60, 0.30, 0.10],
        excluded_token_ids=(0, 4),
        noise_eps=0.1,
    )
    x0 = torch.tensor([[0, 1, 2, 4]])
    mutable = torch.tensor([[False, True, True, False]])
    t = torch.tensor([0.7])
    corrupted = process.forward_process(
        x0,
        t,
        mutable_mask=mutable,
        generator=torch.Generator().manual_seed(9),
    )
    logits = torch.full((1, 4, 5), -torch.inf)
    logits[..., 3] = 0.0
    stepped = process.step(
        logits,
        corrupted,
        t,
        torch.tensor([0.0]),
        mutable_mask=mutable,
        generator=torch.Generator().manual_seed(10),
    )

    assert torch.equal(corrupted[~mutable], x0[~mutable])
    assert torch.equal(stepped[~mutable], x0[~mutable])
    assert torch.equal(stepped[mutable], torch.tensor([3, 3]))
    with pytest.raises(ValueError, match="mutable positions contain excluded"):
        process.forward_process(x0, t, mutable_mask=torch.ones_like(mutable))
    with pytest.raises(ValueError, match="mutable positions contain excluded"):
        process.step(
            logits,
            x0,
            t,
            torch.tensor([0.0]),
            mutable_mask=torch.ones_like(mutable),
        )
    with pytest.raises(ValueError, match="current states must belong"):
        process.posterior_probs(logits, x0, t, torch.tensor([0.0]))


def test_beta_is_derivative_of_the_actual_release_schedule():
    process = ContinuousCategoricalDiffusion(3, [0.5, 0.3, 0.2], noise_eps=0.13)
    t = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    step = 1e-6
    numerical = -(
        process.alpha(t + step).log() - process.alpha(t - step).log()
    ) / (2.0 * step)

    assert torch.allclose(process.beta(t), numerical, atol=2e-10, rtol=2e-10)
    assert process.alpha(torch.tensor(1.0)) == pytest.approx(0.13, abs=1e-7)
    assert math.isfinite(process.beta(torch.tensor(1.0)).item())
