# UDLM × GenMol research plan

## Claim being tested

The target is not merely to make uniform diffusion run. The final UDLM system
must beat the audited local GenMol MDLM control under a matched protocol.
The primary de-novo criterion is a better quality–diversity trade-off at equal
requested sample count, BERT width/depth, training data, and evaluation definitions.
The additive time conditioner adds 787,968 parameters (about 0.9%), so this is
not literally an equal-parameter comparison.
For a single operating point, success means:

- repaired validity and uniqueness are not lower;
- quality is above the local MDLM mean of 85.80%; and
- diversity is no more than 0.005 below the local MDLM mean of 0.8230.

Final evidence requires three independent 1,000-sample seeds and uncertainty;
a 32- or 100-sample pilot is only a gate. Generation speed is reported but is
hardware-dependent. A second, independent target is higher PMO top-10 AUC at
the same oracle-call budget.

The registered final decision uses one-sided 95% intervals: quality's lower
bound must exceed the MDLM control, diversity's lower delta bound must exceed
`-0.005`, and validity/uniqueness deltas must exceed `-0.005`. Means and sample
standard deviations remain paper-compatible summaries. Validity uses a pooled
request-level Newcombe--Wilson method-10 interval for two independent
proportions. Uniqueness, quality, and diversity use unpaired Welch intervals
over the three seed-level estimates per method. A molecule-row bootstrap is
forbidden: re-deduplicating resampled rows manufactures duplicates and does not
represent the uncertainty of these nonlinear per-run set metrics.

The frozen machine-readable protocol is
`experiments/udlm/protocols/de_novo_superiority_v1.json`. Its decision is an
intersection-union gate: all four point requirements and all four interval
requirements must pass for one candidate that was locked before final seeds
0, 1, and 2. The lock binds the completed training summary and exit receipt,
EMA checkpoint, exact EMA shadow-tensor count/decay/update metadata, and an
immutable `launch_manifest.json` by repository-relative path, raw-byte SHA-256,
and schema. The manifest records the exact ordered GPU UUIDs plus the complete
inventory, initial selection, and final just-before-launch telemetry. The final
UUID probes must still satisfy the registered idle-device policy. Runtime
config, training summary, and successful exit receipt must all bind the same
manifest snapshot and exact UUID list. The lock also binds source/config/
sampler/runner hashes, all disclosed pilot evidence, exact 128-NFE sampling
configuration, and one predeclared output directory per final seed.

A repository-global single-training-job lease is acquired before any GPU probe.
Its immutable record is bound into the launch manifest and validated again
before the exit receipt is published; for a handed-off training job, only then
may the receipt writer release that exact lease. The launcher may release its
exact owner lease on failure before tmux handoff. Unexplained or stale leases
fail closed for manual review. The matched R/S/E specification registers
sequential order
`release_uniform` (R), `schedule_uniform` (S), then `empirical_frequency` (E),
at most one training job at a time, and a validated successful receipt before
advancing. The global lease machine-enforces the one-job ceiling. Until a
predecessor-chain artifact is implemented, however, R-to-S-to-E ordering and
receipt-gated advancement remain an operator protocol that needs post-run
audit; an individual run manifest does not prove its predecessor completed.
This is cooperative host/worktree serialization, not a cluster-wide scheduler
reservation.

Training accounting records requested example exposure; it does not claim a
content-token exposure count. Pilot selection may use only seeds at least 1000.
The registered comparison that can make a candidate eligible uses generation
seeds 1000 and 1001, 256 requested samples per seed, 128 NFE, and the
released-compatible quality and diversity metrics. Smaller 32-sample or
32/64-NFE runs remain disclosed engineering evidence but cannot enter the
selection score. The machine-readable winner is the highest mean quality,
then highest mean diversity, then lexicographically smallest attempt ID.
Artifacts using benchmark-run schema 6 or aggregate-report schema 5 are
rejected; the required versions are benchmark 7, report 6, launch manifest 1,
training runtime config 2, training summary 3, and successful exit receipt 3.

The MDLM side of the gate is independently bound to
`experiments/udlm/baselines/mdlm_50000_rescore_attestation.json` (SHA-256
`6326b63c38c7052d0b47282d611618f77637496da2785779af69097fc1441323`).
That immutable attestation was produced from clean, already-pushed revision
`74482c2742ab5ad15def122c809a6b4e403e94cf` by three fresh CPU Python
interpreters, one for each seed 0, 1, and 2, with `PYTHONHASHSEED` equal to the
seed. Current code matched all 63,000 of 63,000 compared historical row cells
under the frozen per-field rules and reproduced the exact released-compatible
and strict per-seed metrics and aggregates. The rescore bound the verified SA
table and hashes for the runtime metric, chemistry, RDKit, SAFE, and runner
modules; it neither regenerated molecules nor rewrote or mutated historical
raw rows or summaries. Its network claim is deliberately narrow: offline
environment settings and Python-level guards covered four socket/name-resolution
APIs, but there was no OS- or process-level isolation and no claim about native
extensions, subprocesses, raw sockets, datagrams, or other unguarded APIs.

Before evaluating a UDLM candidate, the superiority gate verifies the frozen
MDLM manifest and this attestation by path and SHA-256, then checks their source,
row, aggregate, and metric-provenance bindings. Candidate choice is still made
only from the fully disclosed, immutable pilot ledger using the registered
eligible two-seed panel and the predeclared quality-then-diversity ordering; the
winner is committed and pushed in a single candidate lock before any final seed
is run.

## Faithful baseline before hypotheses

The first implementation follows UDLM at official revision `edb0f8c`:

1. The forward process replaces tokens with uniform vocabulary draws.
2. BERT predicts the clean-token distribution and receives the log-linear
   noise level through a learned sinusoidal time adapter.
3. Training uses continuous-time Eq. 18, evaluated with an algebraically exact
   non-negative form to avoid cancellation.
4. Sampling starts from iid uniform tokens and resamples every editable token
   through the exact reverse posterior on a fixed time grid. The official
   128-step grid is the faithful control; 32/64-step grids are speed ablations.

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
6. **Frequency-tempered categorical diffusion** — use a pinned training-prefix
   frequency estimate with a uniform floor, but only after deriving and testing
   its exact non-uniform posterior and objective. A diagnostic sampler is not a
   valid UDLM result and must never be promoted as one.

GenMol MCG is disabled for UDLM until posterior-space guidance is implemented.
Combining clean logits would not equal the UDLM paper's D-CFG rule.

## Progressive gates

1. CPU equation, gradient, legacy-checkpoint, and sampler tests.
2. Tiny CPU overfit on a fixed set of molecules; require falling loss, finite
   gradients, and an executable 16-step reverse chain.
3. After the user chooses a count of one or two GPUs, run the full-size
   warm-start R/S/E panel for 10 optimizer updates in that order. Each job must
   hold the global lease, use the same matched-panel contract, produce a valid
   receipt, save/reload its checkpoint, and remain finite. This is a health and
   plumbing check only. The current constant schedule warms up for 2,500 steps;
   with peak learning rate $3\times10^{-4}$, its learning rate is only about
   $1.2\times10^{-6}$ by update 10. Ten steps therefore cannot rank methods.
4. **Implemented but not yet registered or authorized scheduler screen:** on E
   only, seed 17, compare 100 updates of E-L0 (the current additive conditioner
   and constant schedule with 2,500-update warmup) against E-L1 (the same
   model/process with a 1,000-update half-cosine horizon, warmup 50, peak
   learning rate $3\times10^{-4}$, and clamped floor $3\times10^{-6}$). E-L1 is
   one optimizer-schedule bundle, not an isolated test of cosine curvature: over
   the first 100 optimizer updates its cumulative learning-rate exposure is
   approximately 37.57 times E-L0's. On the fixed denoising panel at
   $t\in\{0.1,0.5,0.9\}$, select E-L1 only if its pooled content-token loss is
   at least 2% lower, at least two of the three time bins improve, and no bin is
   more than 2% worse. A complete valid screen that misses a threshold retains
   E-L0; missing, malformed, or unmatched evidence yields no winner. The code
   and CPU tests exist, but the exact arm registry and selection record are not
   frozen, the launcher does not yet authorize these arms, and no GPU screen has
   run.
5. **Implemented but not yet registered or authorized conditioning screen:** on
   E only, seed 17, train 500 updates with the selected scheduler. Both E-A0 and
   E-A1 start independently from the same verified MDLM-EMA checkpoint; neither
   continues a 100-update scheduler-screen checkpoint. Both reseed the training
   RNG after model construction and warm-start loading so A1's extra parameter
   initialization does not shift the corruption/dropout stream. A0 retains the
   zero-output additive conditioner. A1 normally initializes the timestep MLP,
   applies an outer SiLU, and sends the result to one zero-initialized
   $H\to2H$ FiLM projection after each BERT layer. Select A1 only if it exactly
   preserves warm-start logits before training, pooled fixed-panel content-token
   loss is at least 2% lower, no $t\in\{0.1,0.5,0.9\}$ bin is more than 2% worse,
   clean-token accuracy is nondecreasing, every layer's FiLM group has finite
   nonzero gradients on the first backward, and the timestep MLP has finite
   nonzero gradients after the first **nonzero-learning-rate** FiLM update
   (backward three under either registered schedule, because optimizer update
   one uses learning rate zero). A complete valid screen that misses a selection
   condition retains A0; malformed or incomplete evidence yields no winner.
   This is implemented experimental plumbing, not an executed result.
6. Train matched R/S/E controls for 1,000 updates each with the selected
   scheduler and architecture. First decode 32 requests with generation seed
   1100 as an ineligible health diagnostic. Candidate eligibility still
   requires the registered 256-request runs for both seeds 1000 and 1001 at
   128 NFE. All smaller or mismatched panels remain disclosed but ineligible;
   final seeds 0, 1, and 2 are unavailable for tuning or selection. The user
   chooses one or two GPUs, and immediately before each sequential job the
   launcher scans the full NVIDIA inventory, dynamically selects genuinely idle
   physical GPUs, re-probes their exact UUIDs, and binds the telemetry through
   the launch/runtime/summary/receipt evidence chain.
7. Advance only a promising candidate to 2,000–5,000 steps if the registered
   pilot evidence justifies the cost.
8. Close and commit the complete pilot ledger, then commit and push one
   candidate lock. Only that locked revision may run the three 1,000-sample
   final seeds, once each in their predeclared directories at 128 NFE. Update
   the benchmark PDF only after the raw-row reporter and registered superiority
   gate both validate the result.

### What the prospective L1 and A1 arms change

For optimizer-update index $k\in\{0,\ldots,99\}$, peak learning rate
$\eta=3\times10^{-4}$, L0 warmup $w_0=2500$, L1 warmup $w_1=50$, L1 horizon
$h=1000$, and floor $\eta_{\min}=3\times10^{-6}$, the learning-rate paths are

$$
\eta_{L0}(k)=\eta\frac{k}{w_0},
$$

and

$$
\eta_{L1}(k)=
\begin{cases}
\eta k/w_1, & 0\le k<w_1,\\
\eta_{\min}+(\eta-\eta_{\min})
\frac{1+\cos\!\left(\pi(k-w_1)/(h-w_1)\right)}{2}, & w_1\le k\le h.
\end{cases}
$$

Here $k$ counts optimizer updates, not microbatches. Consequently,
$\sum_{k=0}^{99}\eta_{L0}(k)=5.94\times10^{-4}$ and
$\sum_{k=0}^{99}\eta_{L1}(k)=0.022317219370972547$, giving a ratio
$37.571076\ldots$. This sum is a transparent exposure diagnostic, not an
equivalent number of AdamW steps or a bound on parameter displacement. Because
warmup length, early learning rates, later half-cosine decay, and floor are one
bundle, any observed L1 improvement must be attributed to that bundle.

Let $S_{a,j}$ be the summed content-token loss and $N_{a,j}$ its integer token
denominator for arm $a$ and time bin $j$. Define
$\ell_{a,j}=S_{a,j}/N_{a,j}$ and pooled
$L_a=(\sum_j S_{a,j})/(\sum_j N_{a,j})$. The screen compares these fractions by
exact cross-products: L1 needs $L_{L1}\le0.98L_{L0}$, strict improvement in at
least two $\ell_{a,j}$ values, and $\ell_{L1,j}\le1.02\ell_{L0,j}$ in every bin.
For a concrete equal-denominator example, $N_{a,j}=100$, L0 sums
$(1000,600,200)$, and L1 sums $(970,570,202)$ give bin means
$(10,6,2)$ versus $(9.7,5.7,2.02)$: pooled loss improves about 3.2%, two bins
strictly improve, and the last regresses only 1%.

For A1, let batch size be $B$, sequence length $S$, BERT hidden width $H$, layer
index $l\in\{1,\ldots,L\}$, per-example UDLM noise vector
$\sigma\in\mathbb R^B$, normally initialized timestep MLP
$g:\mathbb R\to\mathbb R^H$, timestep
embedding $c=\operatorname{SiLU}(g(\sigma))\in\mathbb R^{B\times H}$, and the
ordinary post-LayerNorm output of BERT layer $l$ be
$y_l\in\mathbb R^{B\times S\times H}$. Each new projection computes

$$
[\beta_l,\gamma_l]=W_lc+b_l\in\mathbb R^{B\times2H},\qquad
h_l=(1+\gamma_l[:,\mathrm{None},:])\odot y_l+
\beta_l[:,\mathrm{None},:].
$$

$\beta_l$ is the shift, $\gamma_l$ the scale residual, $W_l$ and $b_l$ the
trainable projection with $W_l\in\mathbb R^{2H\times H}$ and
$b_l\in\mathbb R^{2H}$, and $\odot$ elementwise multiplication. The explicit
encoder loop passes $h_l$ into stock BERT layer $l+1$ (and the final $h_L$ to
the classifier); it uses neither hooks nor mutable forward state. Both $W_l$
and $b_l$ start at zero, so $h_l=y_l$ exactly for every $\sigma$ and the MDLM
warm-start logits are unchanged. The timestep MLP $g$ must *not* also have a
zero output: a nonzero $c$ lets each $W_l$ receive a gradient on backward one.
Since every $W_l$ is zero then, the gradient into $g$ is zero on backward one
by construction. Both registered schedules use learning-rate index zero on
optimizer update one, so that zero-rate step leaves $W_l=0$ and backward two
also gives zero gradient to $g$. Optimizer update two has positive learning rate
and changes $W_l$; the next backward (backward three) can make $g$'s gradient
nonzero. The gate is therefore phrased as "after the first nonzero-rate FiLM
update," rather than assuming that the first optimizer call changes weights.

For example, with $B=2$, $S=4$, $H=24$, and $L=2$, $c$ has shape `[2, 24]`,
each projection produces `[2, 48]`, each shift and scale has shape `[2, 24]`,
and broadcasting `[2, 1, 24]` over the four positions preserves a hidden tensor
of shape `[2, 4, 24]`. In the production model, $H=768$ and $L=12$; each
projection has weight shape `[1536, 768]` and bias shape `[1536]`, for
14,174,208 FiLM parameters. Together with the 787,968-parameter timestep MLP,
A1 has 14,962,176 conditioning parameters. Its explicit MDLM-EMA initializer
loads the same 202 base BERT tensors as A0, excludes four timestep and 24 FiLM
parameter tensors, then creates 230 fresh EMA shadows. A0 retains its legacy
state-key set; A1 checkpoints carry a structural conditioning manifest and must
not load as A0 or vice versa.

This topology is inspired by, but is not identical to, released UDLM. Released
GenMol's BERT has no time input. Official UDLM uses a rotary, pre-LayerNorm DiT
whose normally initialized timestep MLP is followed by an outer SiLU; every DiT
block emits two shift/scale/gate triplets and its output layer has another
shift/scale pair. A1 instead preserves GenMol's absolute-position, post-LayerNorm
BERT and applies only one shift/scale pair after each layer, with no residual
gate. It is a warm-start-compatible hypothesis, not a paper result.

Checkpoint: why must both 500-update arms reload the same MDLM EMA and reseed
after initialization? Expected reasoning: continuing a scheduler-screen model
would give one arm extra data exposure, while construction of A1 consumes RNG
for extra parameters; a fresh common checkpoint plus post-init reseed removes
those two avoidable confounds. Why is L1 a bundle rather than a clean cosine
ablation? Expected reasoning: its 50-update warmup changes the first-100-update
learning-rate sum by about 37.57 times, long before much cosine decay occurs.
Why are zero FiLM projections compatible with useful first-step gradients?
Expected reasoning: they make the forward map the identity, but the normally
initialized timestep MLP supplies nonzero $c$, so projection gradients can be
nonzero. Why does the timestep MLP wait until backward three under these
schedules? Expected reasoning: on backward one its upstream Jacobian contains
zero $W_l$; optimizer update one also has zero learning rate, so backward two
sees zero $W_l$ again. Optimizer update two is the first positive-rate update,
allowing backward three to propagate through nonzero FiLM weights.

The warm-start route is an operational sample-efficiency comparison: it uses
the MDLM checkpoint's previous data exposure. A method-only claim additionally
requires a from-scratch UDLM run and an equal-extra-step MDLM continuation.

Every run records Git SHA, source checkpoint/hash, seed, physical-to-logical GPU
mapping, configuration, sample count, step count, wall time, raw generations,
strict and repaired metrics, and deviations from the paper/released code.
Quality additionally binds the ignored local `oracle/fpscores.pkl` input to
SHA-256 `24a4392f5c673e79c0446af3c4d8e458293b5fecaa244328e76741ead9d21dbf`
and PyTDC 0.4.1 source hashes. The runner loads those verified bytes directly
into the resident TDC SA table and disables TDC's implicit downloader while
scoring; missing, replaced, symlinked, or mismatched inputs fail before model
startup.
