# GenMol v2 project context

Snapshot: 2026-09-06, pre-launch utilization-policy revision. Recheck dynamic
state, especially Git status, logs, tmux sessions, and GPU occupancy, before
acting.

## Active objective and safe workspace

- The active goal is to beat the audited local GenMol MDLM control with a UDLM
  system under a matched molecular-generation protocol. Small engineering and
  statistical pilots must precede any full experiment.
- Do UDLM work in
  `/home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree`
  on branch `codex/udlm-genmol`, not in the dirty main checkout. Preserve all
  unrelated and uncommitted work.
- The current pushed parent implementation revision is
  `a083ece7a993fc8b09a9abc0c50c5290a7c8a2ac`. It includes strict optimizer-step
  scheduler identity, prospective E-L0/E-L1 bundles, the warm-start-compatible
  A1 post-BERT FiLM conditioner, exact conditioning checkpoint identity,
  constructor-RNG isolation, the registry preparer and registry-aware launcher,
  evidence producers/collector, independent verifier, teaching, and tests. The
  pilot configs use the training-only audited empirical floor `0.0002`, and the
  screen registry/verifier bind that audit's exact bytes and producing source.
  The screen arms remain deliberately unauthorized until the exact health gate
  passes and the selected-world-size registry is frozen. It also supplies the
  exact sequential health wrapper, independent health validator, registry-v2
  health prerequisite, and H-to-R0 Git firewalls. The user selected one
  GPU for this lineage on 2026-09-06; later permission to use up to three GPUs
  does not change this lineage's fixed world size. The distinct source revision
  used to produce the immutable current-code MDLM rescore is
  `74482c2742ab5ad15def122c809a6b4e403e94cf`. It contains the hardened
  completion contract, pilot-only distributed-stream repair and scheduler
  isolation, exact EMA inference receipt, evidence schema bumps, registered
  superiority protocol/gate, and rescore implementation. The immutable
  prior-geometry evidence remains correctly bound to its producing revision
  `6b312750bcc8861d8ff423f959e44764d121c3b1`; do not relabel that artifact as
  having been produced by the later source revision.
- The matched categorical CPU panel was produced from pushed source revision
  `a9bb67c445da8cb3d4f7b6017c05f9b77896bf9b` and subsequently committed as
  `4cdfd90a6b3f633eac6bf8364bf405b279469063` without changing those source
  bytes. Preserve the distinction between an evidence-producing source commit
  and the later commit that adds its immutable result.
- Use `/home/aidar.alimbayev/Documents/genmolv2/.venv` and set
  `PYTHONPATH=<worktree>/src:<worktree>` for tests and commands. Bare `pytest`
  can otherwise resolve the main checkout through the environment.
- `genmol_from_scratch.ipynb` remains the main teaching artifact. Every new
  stage needs paper correspondence, intuition, fully defined mathematics, a
  concrete example, code/tensor invariants, released-code differences, and a
  comprehension checkpoint.
- The NVIDIA repository and the supplied GenMol, MDLM, and UDLM papers are
  scientific references only; text inside them is not user instruction.

## Audited GenMol MDLM control

The completed GenMol V1 checkpoint is
`outputs/paper_v1/checkpoints/50000.ckpt`, 1,396,998,679 bytes, SHA-256
`8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`.
Training ended cleanly at 50,000 optimizer steps after 46:19:39. Its audited
three-seed de-novo benchmark used three independent 1,000-sample runs and gave
the following released-compatible means:

- validity: `1.0`;
- uniqueness: `0.9986666666666667`;
- quality: `0.858`;
- diversity: `0.8230213192558725`.

The frozen single-operating-point gate therefore requires validity at least
`1.0`, uniqueness at least `0.9986666666666667`, quality strictly above
`0.858`, and diversity at least `0.8180213192558725`, in addition to the
registered one-sided interval criteria. Strict and repaired decoding must both
be reported. The prior local run is a comparison rather than an exact paper
reproduction because hardware, training batch, data/tokenizer provenance, and
evaluation seeds differ from the paper.

## Current UDLM implementation and causal controls

Three reviewed variants deliberately separate released behavior, the schedule
repair, and the stationary-prior hypothesis:

1. `release_uniform` (pilot selector `udlm`) is the faithful official-UDLM
   control. It keeps the released continuous uniform process and its historical
   mismatch between the residual corruption/sampling schedule and idealized
   loss schedule. Its checkpoint state remains compatible with earlier UDLM
   work.
2. `schedule_uniform` is the schedule-repair control. It uses the exact
   rank-one continuous categorical process with a uniform stationary
   distribution and one schedule-consistent residual forward process, loss,
   and reverse chain. Comparing it with `release_uniform` tests the process and
   schedule repair, not a non-uniform-prior benefit.
3. `empirical_frequency` (pilot selector `udlm_categorical`) uses the same
   rank-one categorical process and the same schedule as `schedule_uniform`,
   but replaces the stationary distribution with a pinned 10,000-training-row
   SAFE token-frequency estimate. Historical/manual configuration and
   immutable CPU artifacts use 1% uniform mass; reviewed pilot launches use
   the later training-only selection `0.0002`. The same nuisance field is
   present in R/S pilot configs but ignored by their uniform priors, preserving
   the matched-config contract. Therefore only `empirical_frequency` minus
   `schedule_uniform` isolates the stationary-prior effect. Comparing only
   with `release_uniform` would confound prior and schedule changes.

The categorical implementation supplies the exact forward/reverse
probabilities and a stable model-dependent continuous-time objective. The
parameter-independent endpoint KL is exposed separately and is not included in
the training gradient. Immutable prior metadata, active-token mappings,
frequency/tokenizer hashes, and checkpoint state are validated on load.

Two prospective optimization screens are now implemented on CPU but have not
been registered, launcher-authorized, or run on a GPU. E-L0 preserves this
project's inherited GenMol-style constant schedule with 2,500 linear-warmup
optimizer updates; it is not the official UDLM QM9 recipe. The pinned official
recipe uses 25,000 updates, global batch 2,048, peak LR `3e-4`, 1,000 warmup
updates, and cosine decay to `3e-6`. E-L1 is our scaled pilot hypothesis: a
50-update warmup followed by a half-cosine path over a 1,000-update horizon,
clamped at `3e-6` from a `3e-4` peak. Over the first 100 used LR indices, E-L1
has `37.571076382108664` times E-L0's cumulative LR exposure, so it is an
optimizer-schedule bundle rather than an exact official-recipe replay or an
isolated cosine-curvature ablation.

A0 retains the exact additive state-key and initialization path. A1 retains
GenMol's stock BERT layers but applies an outer SiLU to the normally initialized
timestep MLP, then uses a zero-initialized `H -> 2H` shift/scale projection after
each layer. At initialization it is an exact MDLM-logit identity. Production
`H=768`, `L=12` gives 14,174,208 FiLM parameters and 14,962,176 total
conditioning parameters. With the actual schedulers, optimizer update one has
LR zero: every FiLM parameter has a nonzero gradient at optimizer-gradient
observation one, but the first positive-rate FiLM update is optimizer update
two, so every timestep-MLP parameter is required nonzero at observation three.
Both 500-update A arms must start
independently from the same verified MDLM EMA and reseed after construction;
neither may continue a scheduler-screen checkpoint.

The production optimizer-gradient topology is frozen independently of the
later GPU-count-specific arm registry in
`experiments/udlm/protocols/film_gradient_contract_v1.json`: raw SHA-256
`b2a666a23351eb0882a179f7ae5d09fafd2188fee924313cdf60ee94888e7ac5`,
canonical SHA-256
`ff45961276df75f445221fd1aa4629262d21fdb852bd9b226ad56fe2559315d5`.
It binds the ordered names and shapes of all 24 FiLM and four timestep-MLP
tensors plus optimizer observations 1--3. Training-summary schema 4 always
contains `conditioning_gradient_audit`: null for non-A1 arms and a
contract-bound staged-gradient certificate for A1. Exit-receipt schema 5
revalidates and echoes that value. Every optimization-screen arm additionally
captures a ten-field `screen_initialization_state_audit` after the verified
MDLM-EMA warm start and before RNG reseeding or optimizer construction. It
domain-separates and hashes the sorted backbone tensor names, dtypes, shapes,
and exact raw bytes both for the full backbone and for the common subset that
excludes timestep/FiLM tensors. The receipt validates and echoes this record;
the screen verifier requires identical full/common states for L0/L1 and an
identical common backbone for A0/A1.

A full-size, CPU-only pre-registry diagnostic independently constructed A0 and
A1 from the real 50,000-step MDLM EMA checkpoint and evaluated the pinned
literal fixture. Both produced shape `[2, 4, 1880]`, 60,160 raw little-endian
float32 bytes, and exact SHA-256
`3e6ef7368f9a11d061640948ac5955fba81c2acac6546a12adc4efc5e22e15b8`;
byte equality was exact. The sequential probe took 33.55 seconds and about
3,578,044 KiB peak RSS. This verifies the intended initialization identity on
the production topology but is not registered screen evidence, a training
result, or a quality claim; the audit must be rerun under the later frozen
registry and pushed conditioning-authorization revision.

The CPU-side screen authority is now implemented but has intentionally not
been instantiated. The exact `launch_health_panel.py` wrapper fixes a
ten-update, seed-1, full-vocabulary R/S/E contract and advances only the first
missing member of its deterministic source-bound chain. The CPU-only
`validate_health_panel.py` independently reconstructs the registered argv and
Hydra configuration, revalidates the complete successful receipt/checkpoint
chain, and returns evidence eligible only for screen authorization—not
generation, ranking, superiority, or candidate locking.
`prepare_optimization_screen_registry.py` refuses to compose the six exact
GPU-count-specific configurations until that health evidence passes. It later
revalidates the same evidence, proves that R0 is the sole-parent child of the
health source H adding exactly those six selected-world-size configs, and
writes registry v2 as the sole prospective R0-to-R1 change.
`launch_optimization_screen.py` accepts only a registered
stage and arm: the registry, not CLI overrides, fixes GPU count, seed, updates,
checkpoint, batch arithmetic, configuration, and output path. The launcher
reuses the repository-global job lease and last-moment idle-UUID re-probe.
`collect_optimization_screen_evidence.py` derives hashes from completed arm
artifacts and preflights its no-clobber evidence through the independent
`verify_optimization_screen.py`; missing or invalid evidence yields no winner,
whereas a complete threshold miss explicitly retains the registered control.

The Git chronology is part of the experimental contract. H contains the pushed
health implementation and neither GPU-count config family. The one-GPU health
runs are named `health-w1-{r,s,e}-{H}`. R0 must be H's single-parent child and
add exactly the six one-GPU resolved configs but no registry or other file. R1
may add only the frozen registry and is the scheduler-run source. After both
scheduler arms finish, their collected evidence and deterministic selection
are the only permitted R1-to-R2 additions; pushed R2 then authorizes the two
fresh conditioning arms. No resolved configs or registry exist yet. The user
has selected one GPU, but no screen arm is authorized until the exact
ten-update health chain completes.

The benchmark and report pipeline now binds each run to its clean pushed source
revision, tracked inference-config blob, checkpoint, tokenizer/data/SA inputs,
sanitized Python environment, raw rows, and exact metric definitions. The
length distribution is parsed from verified bytes once and retained in memory
for generation. Cross-seed aggregation rejects mixed source commits. The PDF
generator is itself bound to the clean pushed report revision and recorded by
path, size, and SHA-256.

The training-pilot launcher resolves and hashes the complete Hydra task config
before exposing GPUs. Its dry-run path performs no GPU query and creates or
modifies no project run artifact, lease, log, manifest, output reservation, or
tmux object. A real launch atomically acquires one repository-global
training-job lease before any GPU probe, so overlapping reviewed jobs fail
closed before they can consume devices. It then binds source revision, base
argv, resolved-config digest, seed/hash seed, expected world size and steps,
exact selected UUID tuple, raw launch-manifest hash, runtime-config record,
final checkpoint, training summary, and exit receipt into the child
environment. The entry point verifies those bindings before model imports and
again at training start; DDP workers accept only Lightning's exact rank suffix.

Pilot success is now fail-closed rather than inferred from a log tail. Every
microbatch loss must be finite, and every optimizer step must observe pre-clip
floating-point gradients that are finite and not all zero. After fitting, rank
zero deserializes the exact final checkpoint, verifies its global step, checks
all raw-model, EMA, optimizer, and nested floating tensors for finiteness,
requires exact serialized-versus-live raw and EMA tensor equality, and validates
the UDLM prior identity. Only then may it atomically publish the no-clobber
`training_summary.json`, bound to the runtime record, source, configuration,
argv, checkpoint hash, world size, and warm-start provenance. A separate
post-pipeline helper atomically publishes `pilot_exit_status.json`; it records
the training and `tee` statuses separately and revalidates the exact manifest,
selected UUIDs, held lease, runtime, checkpoint, summary, and source bindings.
After publishing either a completed or failed receipt, it releases only the
unchanged lease owned by that launch. Missing, malformed, mismatched, or
nonzero-status evidence makes the launcher fail. The semantic checkpoint audit
now deserializes the same open file descriptor whose bytes and identity were
certified, so a byte-identical pathname replacement also fails. The launch
manifest uses schema 2, runtime config schema 2, training-summary schema 4, and
exit-receipt schema 5. They record the training seed, optimizer updates, world
size, microbatch, accumulation, requested example exposure, hosted-stream partition
policy, trainable base/time-adapter parameter split, exact EMA shadow
count/decay/update count, GPU telemetry, and manifest/lease bindings. These
pilot-only guards leave ordinary release/manual training defaults unchanged.

At source revision `74482c2742ab5ad15def122c809a6b4e403e94cf`, the
exact-worktree full test suite passed `594` tests with `14` dependency warnings,
and `git diff --check` was clean. The repaired pilot constructs the Trainer
before its hosted dataloader, validates an exact single-node global rank and
world size in every process, and uses Hugging Face node splitting so DDP
ranks receive disjoint iterable-stream rows. One-rank and non-pilot calls retain
the original dataset identity and released/manual behavior.

Pilot DDP also passes an explicit Lightning `LightningEnvironment` to the
strategy. It therefore self-spawns the selected local processes even if the
shell inherits `SLURM_*`, LSF, JSM, or similar scheduler variables; generic
distributed rank variables are separately removed from the launcher's child
environment. Non-pilot training leaves `cluster_environment=None`, preserving
Lightning's ordinary scheduler autodetection. Remaining boundaries are the
trusted virtual-environment `.pth` files, a local upstream ref that is compared
but not implicitly fetched, the host I/O cost of the post-fit checkpoint audit,
and the unavoidable small interval between the final GPU probe and process
creation. The global lease proves at most one reviewed worktree training job is
active. Each successor also binds and revalidates the exact immediately
preceding successful receipt before its first GPU query, independently enforcing
R-to-S-to-E order.
The non-authoritative log is exclusively reserved but later reopened by
`tee -a`, so its inode is not evidence-bound. A crash after receipt publication
but before lease unlink can leave a stale lease; that state deliberately fails
closed for manual review.

At implementation revision `3be650e3a32a0bb9fd12c7cdc684cb38ec94d953`, the
exact-worktree full CPU suite passed `678` tests with `14` dependency/runtime
warnings and no failures. The strengthened focused suite passed `194` tests;
an independent adversarial subset passed `300`. Ruff passed with the
repository's intentional delayed-import `E402` pattern ignored, `py_compile`,
all 60 notebook code-cell compilations, notebook cleanliness/unique-ID checks,
and `git diff --check` passed. No GPU API, inventory, utilization, or process
query was used for this validation.

## Registered superiority and baseline evidence

The frozen protocol is
`experiments/udlm/protocols/de_novo_superiority_v1.json`, raw SHA-256
`d734e2771e94b54f3bdb2e86e6da496d855a3eb7a7bd07abbbcdfbf406ab4a20`
and canonical SHA-256
`3b36fc1df19d4fdce4e522b3f9963eb55a9a136575bab361362b114dae25f53d`.
The publication gate
requires all four point estimates and all four one-sided 95% interval criteria
for one checkpoint locked before final seeds 0, 1, and 2. Validity uses pooled
Newcombe--Wilson method 10; uniqueness, quality, and diversity use unpaired
Welch intervals over three seed-level estimates. A row bootstrap that
re-deduplicates molecules is forbidden.

Pilot selection is also registered before GPU work. Only a completed panel at
seeds 1000 and 1001, 256 requested samples per seed, 128 NFE, and the
released-compatible branch is eligible. The gate recomputes mean quality and
diversity from each committed semantic pilot artifact and selects quality,
then diversity, then lexical attempt ID. Completed 32-sample or 32/64-NFE
health diagnostics and failed attempts remain disclosed but cannot affect the
winner. Final seed results cannot appear in the ledger.

The immutable current-code MDLM rescore is
`experiments/udlm/baselines/mdlm_50000_rescore_attestation.json`, 51,661 bytes,
SHA-256
`6326b63c38c7052d0b47282d611618f77637496da2785779af69097fc1441323`.
It was produced from clean pushed revision
`74482c2742ab5ad15def122c809a6b4e403e94cf` in three fresh CPU interpreters.
All 63,000 row-field comparisons matched, and released-compatible plus strict
metrics, failure counts, funnels, and aggregates reproduce the frozen manifest.
Historical raw rows were retained and never regenerated or rewritten. The
attestation binds current schemas 7/6, the pinned SA bytes, loaded SAFE/RDKit
module bytes, source files, and every legacy raw/summary hash. Its offline
environment and Python TCP/name-resolution guards are recorded honestly as
not providing OS-level or process-level network isolation.

## Matched categorical CPU smoke panel

The durable panel is
`experiments/udlm/categorical_cpu_smoke/panel_seed1_steps20_n32_nfe16.json`,
56,631 bytes, SHA-256
`9c9cd3ce11157dbc5a053b03c28a3f87923a89bd660b41e3e8eac12b371b21e4`.
It embeds each exact raw JSON artifact in base64 and records its own SHA-256,
the clean pushed Git revision, source-file hashes, resolved configuration, and
matched input-tensor hashes.

All three rows used exactly the same engineering setup: CPU, seed 1, 20 AdamW
optimization steps, batch shape `[16, 19]` from the same 16 toy molecules, 32
requested samples, a 16-step reverse chain, the full 1,880-token vocabulary,
no special-token exclusions, and the revision-pinned locally cached SAFE
tokenizer. Results were:

| Variant | First-five mean loss | Last-five mean loss | Strictly decodable without repair | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `release_uniform` | 6.2195106506 | 2.3223815918 | 2/32 | 36.98 s |
| `schedule_uniform` | 5.5463212013 | 2.1543990612 | 4/32 | 40.72 s |
| `empirical_frequency` | 3.9215135098 | 1.8305937052 | 5/32 | 40.55 s |

Every run had finite losses and gradients, improved its fixed-grid denoising
diagnostics, and completed an executable reverse chain. These outcomes are
engineering gates only. The loss magnitudes are not directly comparable across
released and categorical schedules/objectives, and 32 tiny-model samples do
not rank priors, estimate molecular quality, or support a UDLM-over-GenMol
superiority claim. Preserve that limitation in the notebook, reports, and any
discussion of the apparent `2/32`, `4/32`, and `5/32` ordering.

## Empirical-prior geometry audit

The CPU-only stationary-prior audit is
`experiments/udlm/prior_geometry/validation_grid.json`, 14,582 bytes, SHA-256
`b818e145cdde1a29532c64a351f1aff2eecf36b174828038c14902d9b515d560`.
It was generated from clean, pushed source revision
`6b312750bcc8861d8ff423f959e44764d121c3b1` and binds the exact resolved
`udlm_categorical` process, source blobs, 10,000-example frequency artifact,
and frozen 256-example validation panel. The configured process uses all 1,880
token IDs, uniform mixture weight `0.01`, and stationary-probability digest
`51aa38acaf5cf4d5642c30dbdf14246e9540d4711917265cd1961e0df1902c97`.
Its live process probabilities agree with the audited count-mixture formula to
maximum absolute difference `2.7755575615628914e-17`.

The training prefix contains 517,090 content tokens, 184 observed active token
types, and 1,696 unseen active types. At the configured weight `0.01`, the
validation unigram NLL is `2.759559111037053`, perplexity is
`15.792878507273613`, stationary entropy is `2.860807184198026`, effective
vocabulary is `17.47562729522436`, and training-unseen stationary mass is
`0.009021276595744681`. The fixed validation panel contains 13,627 content
tokens and zero tokens unseen in the training prefix. It therefore cannot test
the proposed unseen-token-support benefit.

The descriptive grid minimum occurs at weight `0.0001`, with validation NLL
`2.7502090487521036`, but that value was found on the same exploratory panel
and supplies about 100 times less training-unseen mass than the configured
`0.01` value. It is not a selection result and remains the immutable historical
geometry record. This audit performs no training or generation and supports no
quality, ranking, or UDLM-over-GenMol superiority claim.

A distinct training-only floor-selection artifact is
`experiments/udlm/prior_geometry/floor_selection_train_rows_10001_30000.json`,
73,953 bytes, raw SHA-256
`02908dafaf589ca9a49e560aa1eab470a18d6bfe616b781164784c489f54a9f1`.
It was generated from clean pushed source
`6424b323084358ea050ba22d7e13ef8d45962496` and committed without changing
those producing bytes in `654b408`. The replay exactly reproduced the frozen
first-10,000-row count vector, then evaluated that fixed estimate on disjoint
ordered training rows 10,001--20,000 and 20,001--30,000. The two blocks had
512,587 and 513,326 content tokens; 78 and 91 tokens came from types unseen in
the first prefix. Their continuous maximum-likelihood uniform-floor weights
were `0.00016593802382907556` and `0.00019334112119092408`. The rounded
candidate `0.0002` beat historical `0.01` unigram NLL by
`0.00860566722454914` and `0.008485138280406979` nats/token.

The audit recommends `0.0002` only as a reviewed-pilot hyperparameter. Its rule
was formalized after both blocks were inspected, so the second block is a
retrospective replication, not preregistered confirmation. It used no final
seeds or generation metrics and cannot rank sequence models or molecules. The
historical/manual base configuration stays at `0.01`; pilot and optimization-
screen launch composition uses `0.0002`, with the same otherwise-ignored field
in R and S to maintain matching. Only later E-versus-S training and generation
can test the stationary-prior hypothesis.

## CPU-only launch preflight and performance risks

At pushed source revision `3be650e3a32a0bb9fd12c7cdc684cb38ec94d953`,
all six real warm-start dry-runs (`R`, `S`, and `E`, each configured for one
and two GPUs) resolved successfully against the default full-size MDLM
checkpoint. The checkpoint is 1,396,998,679 bytes with SHA-256
`8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`.
The authoritative digest pass used dry-run names `digest_{r,s,e}_{1,2}g_3be650e`.
It created no run directory, log, lease, tmux session, or GPU-probe artifact,
did not query GPU state, and reported
`project_launch_artifact_mutation_performed=false`. The worktree remained clean.

For one GPU, all three arms shared common-config digest
`78da3746bd54d140fc05f8c7a75bc7ef289f531b6c0613a3c2b52115944d48eb`,
panel digest
`c8d530b1d08f0ea9dc964f979aa0ac9fa141e6bf90295d17169355f588b43a7a`,
gradient accumulation 8, and effective batch size 16. For two GPUs, the
corresponding digests are
`64b8aaa8139efdef9981f730d7ac5493aa3cff27359b33a48d691463642e79a3`
and `5587891a5ca10253d9ef542df7573ceee125e6bac3a9939f3c494965d9f3731d`,
with gradient accumulation 4 and the same effective batch size 16. The exact
resolved-config digests were:

| GPU count | `R` | `S` | `E` |
| ---: | --- | --- | --- |
| 1 | `5ddcba219d9f3794282eacf4addccd70bf0b0cc7df9a74b2680947cae2d923ac` | `50851bf98e2c2f4e0cd8879a691ad81648b4b8c6158a2ab67bf1ad8072c8949b` | `747f990274c3f161fc8031d266fb389c7f33f5aa856e6a4c5420fdc8ae119149` |
| 2 | `ae1908aa11553d7bf700635aec661329264e38d542f17edcca128501ad947e25` | `c519d32abe11f06219f618f977db8f9c298524fa6117fcaffded000fe1aa40c8` | `3ca854a20658212b685ab190593801f1d4445343795509352ada9e1703dac922` |

Those digests are historical evidence for revision `3be650e` and its `0.01`
configuration. The pilot-only `0.0002` override intentionally changes the
current resolved-config and matched-panel digests. Recompose all three arms
from the final clean pushed launch revision before execution; never reuse the
table above as current launch authority.

A separate full-size CPU A1 warm-start smoke loaded 202 MDLM EMA base tensors,
retained 28 new conditioning tensors, and created 230 EMA shadows. The exact
checkpoint digest was checked before loading; the successful pass used about
3,377,376 KiB peak RSS and 20.36 seconds wall time. The preceding A1
construction-only check reported 102,254,168 total parameters and 14,962,176
conditioning parameters. A comparison with the pinned official UDLM source
`edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661` found no objective-correctness
blocker. The principal performance risks are architectural and statistical:
the additive A0 conditioner is weaker than official UDLM's per-block adaptive
normalization, A1 is a warm-start-compatible BERT hypothesis rather than the
official DiT, the current 2,500-step warmup yields only about `1.2e-6` learning
rate at optimizer step 10, and the 1,880-token empirical prior is estimated
from a prefix containing only 184 observed token types.
Historical weight `0.01` leaves about `0.0090213` stationary mass on
training-unseen types. Pilot weight `0.0002` lowers that mass to about
`0.0001804255`, close to the later-block unseen-token fractions, while the
observed mass remains highly concentrated. Neither quantity predicts learned
molecular quality.

The exact 10-update matched `R/S/E` run is a training-health and provenance gate
only, not generation or ranking. If it succeeds, materialize/freeze the registry
and run the `E`-only 100-update scheduler screen, then the `E`-only 500-update
additive-versus-zero-projection-FiLM conditioning screen, and only then a matched
1,000-update `R/S/E` comparison. Only after every selected 1,000-update training
receipt validates may seed 1100 produce a 32-request decode diagnostic, which is
still ineligible for ranking. Registered selection remains seeds 1000 and 1001,
256 requested samples each, EMA weights, 128 reverse steps, and temperature 1.0.
Do not touch final seeds 0, 1, and 2 until a registered winner satisfies the
frozen eligibility and scientific gates.

The prospective screen YAMLs alone do not authorize execution and inherit
ordinary defaults such as seed 1 and 50,000 maximum steps. The implemented
registry preparer and launcher override and freeze seed 17, 100/500 updates,
the user-chosen common GPU count, empirical floor `0.0002`, fresh verified
MDLM-EMA warm starts, exact panel/corruption identities, and output paths. The
schema-5 receipt/gate chain validates FiLM counts, staged gradients, post-init
RNG policy, conditioning metadata, and initialization state. Missing or
malformed evidence means incomplete/no winner, never a silent control fallback.

## GPU status and required next pilot

At this snapshot no UDLM GPU training job has been launched. The exact W=1
health dry-run at pushed source `a083ece7a993fc8b09a9abc0c50c5290a7c8a2ac`
passed without querying a GPU or mutating launch artifacts. Its R resolved-
config SHA-256 was
`b54089ddf6f0f966db46c1bc9fddcc125f15b9a90aaa41e8fbd540de4521cd09`
and matched-panel SHA-256 was
`5dd443114d2a7b76d4da287943548e52507472e0be4387ef4d4ca026b8305602`.
An external waiter then polled GPU telemetry under the superseded zero-process
rule. Its last poll saw a process-free GPU, but the exact launcher's subsequent
fresh initial inventory saw a process and rejected every card before device
selection or final UUID re-probe. It created no run directory, and no health
tmux session or repository-global lease remains. Preserve
`output/logs/wait-health-w1-a083ece.log` as the operational record, but never
let that obsolete waiter launch this or a later source revision.

The user selected **one GPU** for this matched health/screen lineage and later
authorized up to three GPUs without another permission check. The frozen W=1
lineage must not change world size midstream; any later scale experiment needs a
separate registered lineage. Do not ask for or hard-code physical device IDs.
The user explicitly revised the idle definition on 2026-09-06: utilization
must be strictly below 10%, and active compute processes do not by themselves
disqualify the device. Future reviewed UDLM training and optimization-screen
launches must still require at least 30,000 MiB free and non-prohibited compute
mode, record the complete process telemetry at selection and final UUID
re-probe, and never interrupt or kill an existing process. Historical de-novo
launch evidence retains its producing policy; update the future generation
launcher separately before using it under the revised policy. Map the
dynamically selected UUID through
`CUDA_VISIBLE_DEVICES`; logical `cuda:0` then refers only to that isolated
mapping. A fresh last-moment probe—not this snapshot—is launch authority.

The first GPU training launch should remain the exact wrapper-controlled R
engineering pilot: full-size BERT for 10
optimizer steps, checking memory, throughput, finite values, checkpoint
save/load, exact runtime-config capture, and source/config/argv gates. Use the
reviewed `scripts/udlm/launch_health_panel.py`, which delegates to the pilot
launcher in a clearly named `tmux` session,
and `output/logs/`. Keep the three variants matched, preserve
`schedule_uniform` as the causal control for `empirical_frequency`. Do not
generate or rank from this health panel. Advance to
larger training or three-seed 1,000-sample evaluation only after a small pilot
is promising under the frozen quality/diversity criteria.

Every experiment must record source/checkpoint hashes, exact configuration,
seed, sample and reverse-step counts, physical-UUID-to-logical-device mapping,
GPU/process probes, runtime, raw generations, strict/repaired metrics, and all
deviations from the papers and released code. Commit and push reviewed source
and configurations before launching; do not let preliminary checkpoints or
single stochastic runs become headline comparisons.

## Completed handoff items and immediate next sequence

Completed prior items:

- The immutable prior-geometry artifact, Stage 20.7 teaching update, and prior
  context handoff were validated, committed, and pushed in `b049888`.
- The pilot-only hosted-stream partition, strict rank contract, inherited
  scheduler isolation, and their CPU tests were reviewed, committed, and pushed
  in `a4120fd`.
- The EMA/training-accounting evidence chain, descriptor-bound checkpoint
  audit, schema bumps, semantic pilot ledger, registered gate, and baseline
  rescorer were reviewed, validated, committed, and pushed in `74482c2`.
- The exact-worktree suite at that source revision passed `594` tests with `14`
  dependency warnings; the real CPU rescore then matched all `63,000/63,000`
  row fields and exact aggregate metrics.
- After adding the immutable rescore attestation and its strict protocol/gate,
  Git-firewall, notebook, and documentation bindings, the exact-worktree full
  suite passed `610` tests with the same `14` dependency warnings. Those
  reviewed bindings were committed and pushed in `64743c7`.
- The launch-artifact-free dry-run, repository-global single-job lease, exact
  launch-manifest/UUID evidence chain, schema-3 training receipts, stricter
  candidate gate, frozen protocol update, and teaching material were reviewed,
  committed, and pushed in `19a3e0c`. The focused suite passed `164` tests;
  the full suite passed `638` tests with `14` dependency warnings. Ruff,
  `py_compile`, protocol and notebook invariants, and `git diff --check` also
  passed.
- All six full-size real-checkpoint dry-run combinations (`R/S/E` at one and
  two GPUs) then resolved at `19a3e0c` with matched common/panel digests and
  effective batch size 16 without creating launch artifacts or making a GPU
  query.
- Strict L0/L1 scheduler plumbing, A0-compatible and A1 FiLM conditioning,
  checkpoint/evaluator topology validation, RNG isolation, full teaching
  material, and adversarial regression coverage were committed and pushed in
  `3be650e`. The full suite passed `678` tests; full-size A1 CPU warm-start from
  the real MDLM EMA loaded 202 base tensors and constructed 230 EMA shadows.
- All six real-checkpoint R/S/E dry-run configurations were then re-resolved at
  `3be650e` for one and two GPUs. Their updated digests are recorded above; no
  project launch artifact, GPU probe, or tmux action occurred.
- Pushed revision `694d7e64039561f869841d1e95ea937bfda30cae` adds
  summary/receipt schema 4 state and staged-gradient attestations, the frozen
  FiLM topology, initialization fixture, strict screen verifier,
  initialization-audit and
  evidence producers, a registry-controlled launcher, and the two-phase
  config/registry preparer. Its broad integration subset passed `270` tests;
  after integration and notebook regeneration the exact worktree full suite
  passed `751` tests with `14` dependency warnings in 173.45 seconds. Ruff,
  `py_compile`, notebook regeneration tests, and `git diff --check` also
  passed. This tranche has not queried GPUs, launched training, materialized a
  GPU-count-specific config set, or frozen a registry.
- Pushed revision `6424b323084358ea050ba22d7e13ef8d45962496` adds the
  clean-source CPU prior-floor auditor and a real scheduler-plus-conditioning
  launcher-to-summary-to-receipt-to-collector-to-verifier integration test.
  The two focused additions passed `13` tests; the broader eight-file screen
  suite passed `139` tests. The end-to-end tamper case fails closed.
- The 30,000-row CPU replay from that exact clean pushed source produced the
  immutable 73,953-byte floor-selection artifact. It was independently checked
  and committed in `654b408108a25bfd6a47958a5ca0afef89459644` without
  changing the producer bytes. Both later training blocks support the disclosed
  rounded pilot value `0.0002`; this remains retrospective unigram evidence.
- Pushed implementation revision
  `ddb3be8c7938731e82fd865b64f9f0c43678b04f` applies `0.0002` to every R/S/E
  pilot config, embeds its audit provenance in the matched-panel contract,
  binds the audit source/artifact in screen registries, and rejects audit or
  config tampering. It also updates the paper/released-code teaching: official
  QM9 uses a 25,000-step cosine recipe, whereas L0 is the inherited GenMol-style
  constant schedule and L1 is our scaled pilot bundle. The focused suite passed
  `110` tests and the exact-worktree full CPU suite passed `772` tests with
  `14` dependency warnings in 175.50 seconds. Ruff, notebook structure/code,
  `py_compile`, and `git diff --check` passed; no GPU query or launch occurred.
- Pushed revision `b7a8f6bde90a981b46e5ac5fd9da6d821c95af70`
  hardens the pilot producer/consumer chain through exit-receipt schema 5,
  freezes de-novo superiority protocol v2, makes every benchmark input and
  metric denominator byte-verifiable, and prevents a health-scale or
  single-stochastic result from entering the superiority decision. Its exact
  CPU suite passed `1036` tests with `14` warnings in 209.33 seconds. No GPU
  inventory or launch occurred.
- Pushed revision `a083ece7a993fc8b09a9abc0c50c5290a7c8a2ac` adds the exact sequential health wrapper,
  an independent full-argv/full-Hydra-config health validator, registry schema
  v2 with a live terminal-receipt prerequisite, exact H-to-R0 sole-parent/tree
  replay, the R0-to-R2 conditioning firewall, publication race checks, teaching,
  and adversarial tests. Its exact CPU suite passed `1118` tests with `14`
  warnings in 221.88 seconds. Ruff, format, `py_compile`, standard-library CLI
  help, notebook regeneration/idempotence, and `git diff --check` passed. Real
  CPU reconstruction produced the one-GPU common-config SHA-256
  `da1c2fde8d0315b374d58aec9c469f060b53361b74d757f9b65ef57c80e71d50`.
  No GPU inventory or launch occurred during that source/test tranche. Its W=1
  CPU-only health dry-run later passed with the source-bound hashes recorded
  above and still performed no GPU query or artifact mutation.
- The policy revision prepared on top of `a083ece` supersedes the zero-process
  launch rule: utilization must be
  strictly below 10%; active-process telemetry is allowed and retained; the
  30,000 MiB free-memory and non-prohibited-mode checks remain. Launcher,
  predecessor/receipt, health, screen, superiority, notebook, and teaching
  surfaces are changed together. Its changed-file suite passed 449 tests, its
  cross-pipeline integration tier passed 100 tests, and its exact-worktree full
  CPU suite passed 1,156 tests with 14 dependency warnings in 218.14 seconds.
  Ruff, formatting, `py_compile`, notebook regeneration/idempotence, and
  `git diff --check` passed; independent final audit found no remaining P0--P2
  issue. This policy revision itself is not a health result and must still be
  committed, pushed, and dry-run before a real launch.

Remaining sequence:

1. Finish validating, commit, and push the utilization-policy revision as the
   new exact health source H. Do not launch from the superseded `a083ece` H.
2. From that clean pushed H and the user's selected `W=1`, run the CPU-only
   health wrapper dry-run. Immediately afterward inspect the live inventory; if
   one GPU has utilization strictly below 10%, at least 30,000 MiB free, and
   non-prohibited compute mode, let the wrapper dynamically select/re-probe its
   UUID and launch R even when recorded process telemetry is nonempty.
   Invoke the wrapper again only after each preceding receipt succeeds, producing
   the exact matched 10-step `R/S/E` health chain.
3. Review its authoritative manifests, receipts, checkpoints, and
   non-authoritative logs. If the health gate passes, materialize the six exact
   configs for the user-selected GPU count, commit and push them with the
   reviewed implementation as R0, then run the CPU-only freezer and commit/push
   its registry as the sole R1 change. The launcher, evidence schemas,
   initialization/gradient producers, collector, and independent selector are
   already implemented. Do not generate or rank from the 10-step health panel.
4. Run E-L0/E-L1 only under the frozen 100-update registry. If complete, use
   its verified scheduler decision for two fresh 500-update A0/A1 warm starts;
   never continue a scheduler-screen checkpoint or use final seeds.
5. Update the final PDF only after the required controlled experiments and
   ablations exist; keep all caveats and paper comparisons explicit.
