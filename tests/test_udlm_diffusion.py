import math

import pytest
import torch
from torch.nn import functional as F

from genmol.diffusion import ContinuousUniformDiffusion


def _official_literal_loss(logits, x0, xt, t):
    """Literal algebra from the released UDLM ``diffusion.py``."""

    vocab_size = logits.shape[-1]
    model_probs = logits.log_softmax(-1).exp()
    alpha = (1.0 - t)[:, None, None]
    x_bar = vocab_size * alpha * F.one_hot(x0, vocab_size) + 1.0 - alpha
    x_bar_theta = vocab_size * alpha * model_probs + 1.0 - alpha
    gather_index = xt[..., None]
    x_bar_i = torch.gather(x_bar, -1, gather_index)
    x_bar_theta_i = torch.gather(x_bar_theta, -1, gather_index)
    coefficient = -1.0 / (vocab_size * alpha)
    term_1 = vocab_size / x_bar_i - vocab_size / x_bar_theta_i
    term_2 = (
        (x_bar / x_bar_i)
        * (
            x_bar_theta_i.log()
            - x_bar_theta.log()
            + x_bar.log()
            - x_bar_i.log()
        )
    ).sum(-1, keepdim=True)
    return (coefficient * (term_1 - term_2)).squeeze(-1)


def test_schedule_matches_released_loglinear_definition():
    process = ContinuousUniformDiffusion(7, noise_eps=1e-3)
    t = torch.tensor([0.0, 0.25, 1.0])

    assert torch.allclose(process.alpha(t), 1.0 - 0.999 * t)
    assert process.alpha(t)[0] == 1.0
    assert process.alpha(t)[-1] == pytest.approx(1e-3, abs=2e-8)
    assert process.sigma(t)[0] == 0.0
    assert process.sigma(t)[-1] == pytest.approx(-math.log(1e-3), rel=1e-5)


def test_antithetic_times_cover_one_stratum_per_example():
    process = ContinuousUniformDiffusion(7, sampling_eps=1e-3)
    generator = torch.Generator().manual_seed(12)
    t = process.sample_time(8, generator=generator)
    normalized = (t - process.sampling_eps) / (1.0 - process.sampling_eps)
    strata = torch.floor(normalized * 8).to(torch.long)

    assert torch.equal(strata.sort().values, torch.arange(8))
    assert torch.all((t >= process.sampling_eps) & (t < 1.0))


def test_forward_process_has_uniform_marginal_and_clamps_context():
    process = ContinuousUniformDiffusion(4, noise_eps=0.2, antithetic_sampling=False)
    x0 = torch.zeros((100_000, 2), dtype=torch.long)
    mutable = torch.zeros_like(x0, dtype=torch.bool)
    mutable[:, 0] = True
    t = torch.full((x0.shape[0],), 0.5)
    xt = process.forward_process(
        x0, t, mutable_mask=mutable, generator=torch.Generator().manual_seed(3)
    )

    # alpha=.6, hence P(zt=x0)=alpha+(1-alpha)/N=.7.
    frequencies = torch.bincount(xt[:, 0], minlength=4).float() / x0.shape[0]
    assert frequencies[0] == pytest.approx(0.7, abs=0.006)
    assert torch.allclose(frequencies[1:], torch.full((3,), 0.1), atol=0.006)
    assert torch.equal(xt[:, 1], x0[:, 1])


def test_reverse_posterior_matches_direct_bayes_computation():
    process = ContinuousUniformDiffusion(3, noise_eps=0.05)
    logits = torch.log(torch.tensor([[[0.65, 0.25, 0.10]]], dtype=torch.float64))
    xt = torch.tensor([[2]])
    t = torch.tensor([0.8], dtype=torch.float64)
    s = torch.tensor([0.35], dtype=torch.float64)

    actual = process.posterior_probs(logits, xt, t, s)
    alpha_t = process.alpha(t)[0]
    alpha_s = process.alpha(s)[0]
    alpha_t_given_s = alpha_t / alpha_s
    p_x = logits.softmax(-1)[0, 0]
    q_zs = alpha_s * p_x + (1.0 - alpha_s) / 3
    q_zt_given_zs = torch.full((3,), (1.0 - alpha_t_given_s) / 3, dtype=torch.float64)
    q_zt_given_zs[2] += alpha_t_given_s
    expected = q_zs * q_zt_given_zs
    expected /= expected.sum()

    assert torch.allclose(actual[0, 0], expected, atol=1e-12, rtol=1e-12)
    assert torch.all(actual >= 0)
    assert torch.allclose(actual.sum(-1), torch.ones_like(actual[..., 0]))


def test_stable_ct_loss_matches_official_expression_and_gradient():
    process = ContinuousUniformDiffusion(5)
    logits = torch.randn(2, 3, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    logits.requires_grad_()
    x0 = torch.tensor([[0, 1, 4], [3, 2, 0]])
    xt = torch.tensor([[0, 3, 4], [1, 2, 4]])
    t = torch.tensor([0.13, 0.79], dtype=torch.float64)

    stable = process.loss_per_token(logits, x0, xt, t)
    literal = _official_literal_loss(logits, x0, xt, t)
    stable_gradient = torch.autograd.grad(stable.sum(), logits, retain_graph=True)[0]
    literal_gradient = torch.autograd.grad(literal.sum(), logits)[0]

    assert torch.allclose(stable, literal, atol=2e-12, rtol=2e-12)
    assert torch.allclose(stable_gradient, literal_gradient, atol=2e-11, rtol=2e-11)


def test_ct_loss_is_zero_for_perfect_clean_prediction():
    process = ContinuousUniformDiffusion(4)
    x0 = torch.tensor([[0, 1, 2, 3]])
    xt = torch.tensor([[3, 1, 0, 3]])
    logits = torch.full((1, 4, 4), -100.0, dtype=torch.float64)
    logits.scatter_(-1, x0[..., None], 100.0)

    loss = process.loss_per_token(
        logits, x0, xt, torch.tensor([0.4], dtype=torch.float64)
    )

    assert torch.all(loss >= 0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-14)


@pytest.mark.parametrize("time", [1e-3, 0.01, 0.5, 0.99, 0.9999])
def test_large_vocabulary_loss_and_gradients_are_finite(time):
    process = ContinuousUniformDiffusion(1880)
    logits = torch.randn(2, 3, 1880, generator=torch.Generator().manual_seed(18))
    logits.requires_grad_()
    x0 = torch.tensor([[5, 900, 1879], [44, 1200, 6]])
    xt = torch.tensor([[10, 900, 9], [44, 11, 1700]])
    loss = process.loss(
        logits, x0, xt, torch.full((2,), time), global_mean=True
    )
    loss.backward()

    assert loss >= 0
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_restricted_alphabet_never_corrupts_or_samples_to_special_tokens():
    process = ContinuousUniformDiffusion(8, excluded_token_ids=(0, 1, 2, 3, 4))
    prior = process.sample_prior((10_000,), generator=torch.Generator().manual_seed(4))
    x0 = torch.tensor([[1, 5, 6, 2]])
    mutable = torch.tensor([[False, True, True, False]])
    xt = process.forward_process(
        x0,
        torch.tensor([1.0]),
        mutable_mask=mutable,
        generator=torch.Generator().manual_seed(5),
    )

    assert set(prior.unique().tolist()) == {5, 6, 7}
    assert xt[0, 0] == 1 and xt[0, -1] == 2
    assert torch.all(xt[mutable] >= 5)


def test_step_resamples_editable_tokens_and_preserves_context():
    process = ContinuousUniformDiffusion(4)
    xt = torch.tensor([[0, 1, 2, 3]])
    mutable = torch.tensor([[False, True, True, False]])
    logits = torch.full((1, 4, 4), -80.0)
    logits[..., 0] = 80.0
    result = process.step(
        logits,
        xt,
        torch.tensor([0.5]),
        torch.tensor([0.0]),
        mutable_mask=mutable,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(result[~mutable], xt[~mutable])
    assert torch.equal(result[mutable], torch.zeros(2, dtype=torch.long))


def test_global_mean_uses_only_selected_tokens():
    process = ContinuousUniformDiffusion(3)
    logits = torch.zeros(1, 3, 3)
    x0 = torch.tensor([[0, 1, 2]])
    xt = torch.tensor([[2, 0, 1]])
    t = torch.tensor([0.4])
    mask = torch.tensor([[True, False, True]])

    per_token = process.loss_per_token(logits, x0, xt, t, mask=mask)
    reduced = process.loss(logits, x0, xt, t, mask=mask, global_mean=True)

    assert per_token[0, 1] == 0
    assert torch.allclose(reduced, per_token[[0, 0], [0, 2]].mean())
