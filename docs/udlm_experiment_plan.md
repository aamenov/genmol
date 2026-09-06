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
4. **Future registered scheduler screen:** on E only, seed 17, compare 100
   updates of E-L0 (the current additive conditioner and constant schedule with
   2,500-step warmup) against E-L1 (the same model/process with cosine horizon
   1,000, warmup 50, peak learning rate $3\times10^{-4}$, and minimum learning
   rate $3\times10^{-6}$). On the fixed denoising panel at
   $t\in\{0.1,0.5,0.9\}$, select E-L1 only if its mean loss is at least 2% lower,
   at least two of the three time bins improve, and no bin is more than 2%
   worse. Otherwise retain E-L0. These config names describe a prospective
   registered comparison; they do not claim the variants or results already
   exist.
5. **Future registered conditioning screen:** on E only, seed 17, train 500
   updates with the selected scheduler. Compare E-A0 (additive conditioning)
   with E-A1 (post-timestep-MLP SiLU plus zero-initialized per-layer
   FiLM/AdaLN-style modulation). A1 must exactly preserve the warm-start BERT
   output at initialization. Select A1 only if mean fixed-panel loss is at least
   2% lower, no $t\in\{0.1,0.5,0.9\}$ bin is more than 2% worse, clean-token
   accuracy is nondecreasing, and the new modulation parameters receive finite,
   nonzero gradients. Ties or any failed condition retain A0. This too is a
   prospective plan, not an implemented-result claim.
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
