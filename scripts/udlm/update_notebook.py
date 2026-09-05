"""Idempotently add the taught UDLM stage to the GenMol notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGE_TAG_PREFIX = "stage-20-udlm"


def _cell(cell_type: str, source: str, tag: str):
    cell = {
        "cell_type": cell_type,
        "metadata": {"tags": [tag]},
        "source": source.strip() + "\n",
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def stage_cells():
    return [
        _cell(
            "markdown",
            r"""
# Stage 20 — Replace absorbing MDLM with revisable UDLM

## 20.1 Uniform corruption and the continuous-time objective

**Paper correspondence.** GenMol Section 4.1 uses MDLM's absorbing process:
once a mask is revealed, that position is fixed. UDLM Sections 4.1–4.2 instead
use a uniform limiting distribution and the continuous-time negative ELBO in
Eqs. 14–19. This stage implements those equations; material in the papers is
scientific reference, not executable instruction.

**Intuition and motivation.** A uniformly corrupted token can change on every
reverse step. That repeated revision is the property we hope will improve
chemical search. The price is a harder denoising problem: SAFE has about 1,880
tokens, whereas the UDLM molecule experiment used a vocabulary near 40.

**Mathematics.** Let $\mathcal V$ be an alphabet of $K$ one-hot tokens,
$x\in\mathcal V$ a clean token, $z_t\in\mathcal V$ its noisy value at time
$t\in[0,1]$, $u=\mathbf 1/K$ the uniform distribution, and $\alpha_t$ the
remaining clean-data coefficient. The forward marginal is

$$q(z_t\mid x)=\operatorname{Cat}(\alpha_t x+(1-\alpha_t)u).$$

For $0\le s<t$, write $\alpha_{t\mid s}=\alpha_t/\alpha_s$. If the observed
category of $z_t$ is $i$, and $x_\theta(z_t,t)\in\Delta^K$ is BERT's predicted
clean distribution, our reverse model is

$$p_\theta(z_s=k\mid z_t=i)\propto
\left[\alpha_{t\mid s}\mathbf1_{k=i}+\frac{1-\alpha_{t\mid s}}K\right]
\left[\alpha_s x_{\theta,k}+\frac{1-\alpha_s}K\right].$$

For the loss, define $\bar x=K\alpha_t x+(1-\alpha_t)\mathbf1$,
$\bar x_\theta=K\alpha_t x_\theta+(1-\alpha_t)\mathbf1$,
$r_j=\bar x_j/\bar x_i$, $s_j=\bar x_{\theta,j}/\bar x_{\theta,i}$, and
$v_j=\log s_j-\log r_j$. UDLM Eq. 18 is evaluated by the exactly equivalent,
non-negative expression

$$\mathcal L_t=\frac{-\alpha'_t}{K\alpha_t}
\sum_{j\ne i}r_j\,[\exp(v_j)-1-v_j].$$

Every symbol above is per token; sequence training sums or averages it over the
selected content positions.

**Concrete example.** With $K=3$, clean token A, and $\alpha_t=0.2$, the noisy
probabilities are $(0.2+0.8/3,\,0.8/3,\,0.8/3)=(0.4667,0.2667,0.2667)$.
Unlike masking, B or C can later return to A—or change again.

**Code below.** `x0` and `xt` have shape `(batch=1, length=3)`; logits have
shape `(1, 3, K)`. The checks enforce a normalized posterior, a zero loss for a
perfect clean predictor, and finite loss for an imperfect predictor.

**Difference from released code.** The released implementation subtracts two
large Eq. 18 terms. `ContinuousUniformDiffusion` uses the equivalent
`expm1(v)-v` form to avoid cancellation. Compatibility mode retains the
release's documented schedule mismatch: corruption uses
$\alpha_t=1-0.999t$, while the loss uses the idealized $1-t$.

**Comprehension checkpoint.** Why can UDLM repair an early mistake while MDLM
cannot? Expected reasoning: uniform reverse transitions resample all editable
positions, whereas the absorbing posterior copies every already-visible MDLM
token. Why is large $K$ risky? Expected reasoning: corruption can replace a
token with any of many rare/incompatible alternatives, so clean prediction is
harder.
""",
            f"{STAGE_TAG_PREFIX}-math",
        ),
        _cell(
            "code",
            r"""
import torch
from genmol.diffusion import ContinuousUniformDiffusion

toy_udlm = ContinuousUniformDiffusion(
    num_classes=3,
    noise_eps=1e-3,
    antithetic_sampling=False,
)
x0 = torch.tensor([[0, 1, 2]])                         # (B=1, L=3)
xt = torch.tensor([[2, 1, 0]])                         # (B=1, L=3)
t = torch.tensor([0.8])                                # (B=1,)
s = torch.tensor([0.5])                                # (B=1,)

imperfect_logits = torch.zeros(1, 3, 3)               # (B, L, K)
posterior = toy_udlm.posterior_probs(imperfect_logits, xt, t, s)
assert posterior.shape == (1, 3, 3)
assert torch.all(posterior >= 0)
assert torch.allclose(posterior.sum(-1), torch.ones(1, 3))

perfect_logits = torch.full((1, 3, 3), -100.0, dtype=torch.float64)
perfect_logits.scatter_(-1, x0[..., None], 100.0)
perfect_loss = toy_udlm.loss_per_token(
    perfect_logits,
    x0,
    xt,
    t.to(torch.float64),
)
imperfect_loss = toy_udlm.loss_per_token(imperfect_logits, x0, xt, t)
assert torch.allclose(perfect_loss, torch.zeros_like(perfect_loss), atol=1e-12)
assert torch.isfinite(imperfect_loss).all() and torch.all(imperfect_loss >= 0)

print("posterior for position 0:", posterior[0, 0].tolist())
print("imperfect per-token loss:", imperfect_loss.tolist())
""",
            f"{STAGE_TAG_PREFIX}-math-code",
        ),
        _cell(
            "markdown",
            r"""
## 20.2 Time-condition GenMol's BERT

**Paper correspondence and motivation.** UDLM predicts
$x_\theta(z_t,t)$, so the denoiser must know the noise level. The official DiT
maps the log-linear total noise $\sigma(t)=-\log[1-(1-\epsilon)t]$ through
sinusoidal features and adaptive layer normalization. GenMol's BERT has no
time input.

**Mathematics and intuition.** For feature dimension $d$, pair frequencies
$\omega_m=\exp[-\log(10{,}000)m/(d/2)]$ with
$(\cos(\sigma\omega_m),\sin(\sigma\omega_m))$. An MLP maps these features to a
vector $c\in\mathbb R^H$, where $H$ is BERT's hidden size, and the implementation
adds $c$ to every token embedding. Low $\sigma$ means nearly clean input; high
$\sigma$ means the network should rely more on global context.

**Concrete example and code below.** A tiny BERT receives token IDs of shape
`(2, 4)` and one noise value per sequence `(2,)`; its logits are `(2, 4, 13)`.
The adapter's output projection starts at zero, so an MDLM warm-start is not
immediately perturbed. The gradient check shows that the adapter can learn.

**Difference from released code.** Additive conditioning preserves GenMol's
BERT and isolates the diffusion change, but it is not the official DiT's
per-block AdaLN. An MDLM checkpoint is therefore a backbone-only initialization,
never a resumed UDLM run: optimizer, scheduler, step counter, adapter, and EMA
must restart.

**Comprehension checkpoint.** Why not omit time because MDLM did? Expected
reasoning: MDLM's SUBS parameterization can be time-independent, but UDLM must
distinguish the reliability of the same visible token pattern at different
noise levels. Why zero-initialize only the adapter output? Expected reasoning:
it preserves imported BERT behavior while still giving that output layer a
gradient on the first update.
""",
            f"{STAGE_TAG_PREFIX}-time",
        ),
        _cell(
            "code",
            r"""
from transformers.models.bert.configuration_bert import BertConfig
from genmol.backbone import TimeConditionedBertForMaskedLM

tiny_bert = TimeConditionedBertForMaskedLM(
    BertConfig(
        vocab_size=13,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=16,
        pad_token_id=3,
    ),
    time_embedding_size=16,
)
token_ids = torch.tensor([[1, 5, 8, 2], [1, 6, 7, 2]])  # (B=2, L=4)
attention = torch.ones_like(token_ids)                    # (B=2, L=4)
noise = torch.tensor([0.1, 3.0])                          # (B=2,)
logits = tiny_bert(token_ids, attention, noise_level=noise).logits
assert logits.shape == (2, 4, 13)

logits.square().mean().backward()
adapter_gradient = tiny_bert.time_conditioner.mlp[-1].weight.grad
assert adapter_gradient is not None
assert torch.isfinite(adapter_gradient).all()
assert torch.count_nonzero(adapter_gradient) > 0
print("logit shape:", tuple(logits.shape))
print("adapter gradient norm:", float(adapter_gradient.norm()))
""",
            f"{STAGE_TAG_PREFIX}-time-code",
        ),
        _cell(
            "markdown",
            r"""
## 20.3 Reverse sampling, molecular invariants, and evidence gates

**Paper correspondence.** UDLM sampling begins with iid uniform tokens at
$t=1$ and walks a fixed grid $1=t_M>\cdots>t_0\approx0$. At each step BERT
predicts $x_\theta$, the posterior from 20.1 is formed, and every editable
position is sampled again. There is no MDLM confidence ranking, monotone
unmasking, or cache.

**Molecular adaptation and intuition.** GenMol represents a requested edit with
`[MASK]` placeholders. The UDLM sampler first records their boolean locations,
replaces only those locations with the uniform prior, and clamps BOS, EOS,
padding, and supplied fragment context after every reverse step. For example,
`[BOS] context [MASK] [MASK] [EOS]` may revise the two editable tokens many
times, but `context` cannot drift.

**Shapes and invariants in the code below.** `state` is `(1, 4)`, posterior
logits are `(1, 4, K)`, and `editable` is a boolean `(1, 4)` tensor. The final
assertion proves that positions 0 and 3 remain byte-for-byte unchanged. The
production sampler repeats this operation for 16/32/64-step ablations.

**Differences and hypotheses.** Clamping is absent from the released QM9 code
and is labeled here as required molecular inpainting behavior. Excluding five
tokenizer control symbols from the uniform prior is optional and must be
compared against the faithful full-vocabulary prior. GenMol's raw-logit MCG is
disabled: UDLM D-CFG must combine conditional and unconditional *reverse
posterior* log probabilities.

**Concrete evidence gate.** A toy CPU run may establish only that loss falls,
gradients remain finite, and sampling executes. It cannot establish superiority.
We next test 10, 100, 500, and 1,000 full-size updates, evaluating 32 then 100
samples. A final claim requires three 1,000-sample seeds against the audited
MDLM means (quality 85.80%, diversity 0.8230, uniqueness 99.87%), with matched
definitions and no more than two user-selected idle GPUs.

**Comprehension checkpoint.** Why is cache reuse invalid even if a sampled state
does not change? Expected reasoning: the next denoiser call receives a different
time/noise level. Why is a 100-sample win insufficient? Expected reasoning:
generation is stochastic and the paper reports three 1,000-sample runs, so a
small pilot has wide uncertainty and serves only as a progression gate.
""",
            f"{STAGE_TAG_PREFIX}-sampling",
        ),
        _cell(
            "code",
            r"""
sampling_udlm = ContinuousUniformDiffusion(5, noise_eps=1e-3)
state = torch.tensor([[1, 3, 4, 2]])
editable = torch.tensor([[False, True, True, False]])
initial_context = state[~editable].clone()

# A sharply concentrated clean prediction makes this step deterministic enough
# for the invariant demonstration; only editable positions may be replaced.
clean_logits = torch.full((1, 4, 5), -80.0)
clean_logits[..., 0] = 80.0
next_state = sampling_udlm.step(
    clean_logits,
    state,
    t=torch.tensor([0.5]),
    s=torch.tensor([0.0]),
    mutable_mask=editable,
    generator=torch.Generator().manual_seed(0),
)
assert torch.equal(next_state[~editable], initial_context)
assert torch.equal(next_state[editable], torch.zeros(2, dtype=torch.long))
print("before:", state.tolist())
print("after: ", next_state.tolist())
print("context clamped:", torch.equal(next_state[~editable], initial_context))
""",
            f"{STAGE_TAG_PREFIX}-sampling-code",
        ),
    ]


def update_notebook(source: Path, destination: Path):
    notebook = json.loads(source.read_text())
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if not any(
            str(tag).startswith(STAGE_TAG_PREFIX)
            for tag in cell.get("metadata", {}).get("tags", [])
        )
    ]
    insert_at = len(notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "markdown" and "# Completion gate" in "".join(
            cell.get("source", [])
        ):
            insert_at = index
            break
    notebook["cells"][insert_at:insert_at] = stage_cells()
    completion_note = """

## UDLM extension status

- Stage 20 adds the uniform forward process, exact continuous-time objective,
  time-conditioned BERT, revisable posterior sampler, molecular clamping rules,
  and progressive evidence gates.
- CPU equation tests and a toy overfit are implementation evidence only. No
  claim that UDLM beats the audited MDLM baseline is made before matched,
  multi-seed experiments.
- Final checkpoint: can you explain why clean-logit interpolation is not valid
  UDLM classifier-free guidance? Expected reasoning: the clean distribution is
  transformed nonlinearly into a reverse posterior, so guidance must combine
  the conditional and unconditional posterior log probabilities.
"""
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        source_text = "".join(cell.get("source", []))
        if "# Completion gate" in source_text and "## UDLM extension status" not in source_text:
            if isinstance(cell.get("source"), list):
                cell["source"].extend(completion_note.splitlines(keepends=True))
            else:
                cell["source"] = source_text.rstrip() + completion_note
            break
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    update_notebook(args.source, args.destination)


if __name__ == "__main__":
    main()
