# UDLM × GenMol research plan

## Claim being tested

The target is not merely to make uniform diffusion run. The final UDLM system
must beat the audited local GenMol MDLM control under a matched protocol.
The primary de-novo criterion is a better quality–diversity trade-off at equal
requested sample count, model size, training data, and evaluation definitions.
For a single operating point, success means:

- repaired validity and uniqueness are not lower;
- quality is above the local MDLM mean of 85.80%; and
- diversity is no more than 0.005 below the local MDLM mean of 0.8230.

Final evidence requires three independent 1,000-sample seeds and uncertainty;
a 32- or 100-sample pilot is only a gate. Generation speed is reported but is
hardware-dependent. A second, independent target is higher PMO top-10 AUC at
the same oracle-call budget.

## Faithful baseline before hypotheses

The first implementation follows UDLM at official revision `edb0f8c`:

1. The forward process replaces tokens with uniform vocabulary draws.
2. BERT predicts the clean-token distribution and receives the log-linear
   noise level through a learned sinusoidal time adapter.
3. Training uses continuous-time Eq. 18, evaluated with an algebraically exact
   non-negative form to avoid cancellation.
4. Sampling starts from iid uniform tokens and resamples every editable token
   through the exact reverse posterior on a fixed time grid.

Two implementation differences are repairs, not hypotheses: the unused
reconstruction forward pass is omitted, and the stable Eq. 18 identity replaces
the released subtraction of large terms. GenMol sequence framing is clamped:
BOS, EOS, padding, and supplied context cannot be overwritten. This is a
molecular inpainting adaptation absent from the UDLM QM9 release.

The compatibility mode deliberately retains the released schedule mismatch:
corruption/sampling use `alpha(t)=1-0.999t`, while Eq. 18 uses the idealized
`alpha(t)=1-t`. A schedule-consistent repair must be tested separately.

## Main risk and testable hypotheses

UDLM beat MDLM on QM9 with a vocabulary of about 40. GenMol's SAFE model has
about 1,880 states, and the UDLM paper itself reports that uniform diffusion
degrades as the vocabulary grows. The experiments therefore proceed in this
order:

1. **Full-vocabulary UDLM** — scientific control matching released behavior.
2. **Exclude control symbols** — remove PAD/BOS/EOS/UNK/MASK from the uniform
   corruption prior; this is an explicitly labeled chemistry adaptation.
3. **MDLM EMA warm-start** — reuse only BERT weights, while resetting optimizer,
   scheduler, step, time adapter, and EMA. This tests sample efficiency, not a
   from-scratch architecture comparison.
4. **Schedule-consistent loss** — use the same alpha and derivative in
   corruption, Eq. 18, and sampling.
5. **Smaller or structured corruption alphabet** — only if the controls show
   that the 1,880-way uniform prior is the limiting factor. Candidate versions
   are a smaller SAFE tokenizer and token-type-restricted noise. These change
   the model/data representation and require their own MDLM controls.

GenMol MCG is disabled for UDLM until posterior-space guidance is implemented.
Combining clean logits would not equal the UDLM paper's D-CFG rule.

## Progressive gates

1. CPU equation, gradient, legacy-checkpoint, and sampler tests.
2. Tiny CPU overfit on a fixed set of molecules; require falling loss, finite
   gradients, and an executable 16-step reverse chain.
3. One verified-idle GPU, full-size BERT, 10 optimizer steps; check memory,
   throughput, checkpoint save/load, and no NaNs.
4. Warm-start pilots at 100, 500, then 1,000 steps. Evaluate 32 samples first,
   then 100 samples at 16/32/64 reverse steps.
5. Advance only a promising candidate to 2,000–5,000 steps. Use at most two
   user-selected, freshly verified idle GPUs.
6. Run three 1,000-sample seeds and update the benchmark PDF only after a pilot
   clears the quality/diversity gate.

Every run records Git SHA, source checkpoint/hash, seed, physical-to-logical GPU
mapping, configuration, sample count, step count, wall time, raw generations,
strict and repaired metrics, and deviations from the paper/released code.
