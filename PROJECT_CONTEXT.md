# GenMol v2 project context

Snapshot: 2026-09-06 09:25 Asia/Dubai. Recheck dynamic state, especially Git
status, logs, tmux sessions, and GPU occupancy, before acting.

## Active objective and safe workspace

- The active goal is to beat the audited local GenMol MDLM control with a UDLM
  system under a matched molecular-generation protocol. Small engineering and
  statistical pilots must precede any full experiment.
- Do UDLM work in
  `/home/aidar.alimbayev/Documents/genmolv2/run_sources/udlm_genmol_worktree`
  on branch `codex/udlm-genmol`, not in the dirty main checkout. Preserve all
  unrelated and uncommitted work.
- The current reviewed implementation revision is clean and pushed at
  `3be650e3a32a0bb9fd12c7cdc684cb38ec94d953`. It adds strict optimizer-step
  scheduler identity, prospective E-L0/E-L1 bundles, the warm-start-compatible
  A1 post-BERT FiLM conditioner, exact conditioning checkpoint identity,
  constructor-RNG isolation plumbing, and their teaching/tests. These screen
  arms remain deliberately unauthorized until a separate registry, launcher,
  and verifier are frozen. The distinct source revision
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
   SAFE token-frequency estimate mixed with 1% uniform mass. Therefore only
   `empirical_frequency - schedule_uniform` isolates the stationary-prior
   effect. Comparing only with `release_uniform` would confound prior and
   schedule changes.

The categorical implementation supplies the exact forward/reverse
probabilities and a stable model-dependent continuous-time objective. The
parameter-independent endpoint KL is exposed separately and is not included in
the training gradient. Immutable prior metadata, active-token mappings,
frequency/tokenizer hashes, and checkpoint state are validated on load.

Two prospective optimization screens are now implemented on CPU but have not
been registered, launcher-authorized, or run on a GPU. E-L0 preserves the
released constant schedule with 2,500 linear-warmup optimizer updates. E-L1 is
a 50-update warmup followed by a half-cosine path over a 1,000-update horizon,
clamped at `3e-6` from a `3e-4` peak. Over the first 100 used LR indices, E-L1
has `37.571076382108664` times E-L0's cumulative LR exposure, so it is an
optimizer-schedule bundle rather than an isolated cosine-curvature ablation.

A0 retains the exact additive state-key and initialization path. A1 retains
GenMol's stock BERT layers but applies an outer SiLU to the normally initialized
timestep MLP, then uses a zero-initialized `H -> 2H` shift/scale projection after
each layer. At initialization it is an exact MDLM-logit identity. Production
`H=768`, `L=12` gives 14,174,208 FiLM parameters and 14,962,176 total
conditioning parameters. With the actual schedulers, optimizer update one has
LR zero: FiLM gradients exist on backward one, but the first positive-rate
FiLM update is optimizer update two, so the timestep MLP can first receive a
nonzero gradient on backward three. Both 500-update A arms must start
independently from the same verified MDLM EMA and reseed after construction;
neither may continue a scheduler-screen checkpoint.

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
manifest uses schema 1, runtime config schema 2, and training summary plus exit
receipt schema 3. They record the training seed, optimizer updates, world size,
microbatch, accumulation, requested example exposure, hosted-stream partition
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
active, but each per-run manifest does not yet bind its predecessor receipt;
R-to-S-to-E order remains an operator protocol checked again after the runs.
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
`a44263d56a42593ca9f1b9c00c7ad8177229f0f4fa9ff941ab07481054b84848`
and canonical SHA-256
`6c33533dc220f5d6682964fd94de3ce85a4425e2cce8ca21df227feeb4d0af2e`.
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
`0.01` value. It is not a selection result. A fresh confirmatory panel is
required before any prior choice, and the proposed weights
`[0.001, 0.01, 0.05]` remain a hypothesis to test only after the matched
`release_uniform`/`schedule_uniform`/`empirical_frequency` health gate. This
audit performs no training or generation and supports no quality, ranking, or
UDLM-over-GenMol superiority claim.

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
The configured uniform mixture leaves about `0.0090213` stationary mass on
training-unseen types, while the observed mass is highly concentrated.

The 10-step matched `R/S/E` run is therefore a health gate only, not a ranking.
If it succeeds, first run an `E`-only 100-update scheduler screen, then an
`E`-only 500-update additive-versus-zero-projection-FiLM conditioning screen,
and only then a matched 1,000-update `R/S/E` comparison. Commit the exact arm
registry and selection rules before the 100-update screen. Use 32-sample
diagnostics only for health; registered selection remains seeds 1000 and 1001,
256 samples each, EMA weights, 128 reverse steps, and temperature 1.0. Do not
touch final seeds 0, 1, and 2 until a registered winner satisfies the frozen
eligibility and scientific gates.

The prospective screen YAMLs intentionally do not authorize execution and
inherit ordinary defaults such as seed 1 and 50,000 maximum steps. A separate
screen registry/launcher must override and freeze seed 17, 100/500 updates, the
user-chosen common GPU count, fresh verified MDLM-EMA warm starts, exact panel
and corruption identities, and output paths. Before A1 authorization it must
also add variant-aware receipt/gate validation for FiLM parameter counts and
the post-init RNG policy, explicitly report conditioning metadata/hash from the
denoising evaluator, and reject scratch A-screen launches. Missing or malformed
screen evidence must mean incomplete/no winner, not a silent control fallback.

## GPU status and required next pilot

No UDLM GPU job has been launched, and no GPU inventory or utilization probe
has yet been run for the pending pilot. No probe or job occurred while producing
or reviewing revision `3be650e3a32a0bb9fd12c7cdc684cb38ec94d953`, the CPU-only
MDLM rescore, the twelve real launch dry-run invocations (two digest passes over
the six R/S/E configurations), the full-size A1 CPU warm-start smoke, or this
documentation update. No stale snapshot should be treated as authorization or
availability evidence.

Before the first GPU launch, the user must select only the GPU count: one or
two. Recommend **one GPU** for the first matched 10-step engineering gate; the
count can be reconsidered after memory and throughput are measured. Do not ask
for or hard-code physical device IDs. Immediately before the job, inventory
every NVIDIA device and its processes, dynamically choose that many genuinely
idle devices, and re-probe the exact selected UUIDs at the last possible point.
A selected device must have zero foreign compute processes, utilization below
the launcher's approved threshold (currently 10%), and at least 30,000 MiB
free. Map the UUIDs through `CUDA_VISIBLE_DEVICES`; logical `cuda:0` is then
safe. Never interrupt or reuse another user's process.

The first GPU action should remain an engineering pilot: full-size BERT for 10
optimizer steps, checking memory, throughput, finite values, checkpoint
save/load, exact runtime-config capture, and source/config/argv gates. Use the
reviewed `scripts/udlm/launch_train_pilot.py`, a clearly named `tmux` session,
and `output/logs/`. Keep the three variants matched, preserve
`schedule_uniform` as the causal control for `empirical_frequency`, and inspect
32 samples before increasing to 100, 500, or 1,000 training steps. Advance to
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

Remaining sequence:

1. Ask the user whether the first pilot should use one or two GPUs; recommend
   one GPU for this first matched health gate.
2. Only after that choice, perform the first fresh GPU inventory and exact-UUID
   re-probe, then launch the matched 10-step `R/S/E` engineering pilot.
3. Review its authoritative manifests, receipts, checkpoints, and
   non-authoritative logs. Before authorizing the 100-update scheduler screen,
   implement and freeze the separate optimization-screen registry, launcher,
   evidence schemas, and CPU-only selection verifier, including all deferred
   A1 schema bindings listed above. Do not rank variants from 10-step losses or
   32-sample health diagnostics.
4. Run E-L0/E-L1 only under the frozen 100-update registry. If complete, use
   its verified scheduler decision for two fresh 500-update A0/A1 warm starts;
   never continue a scheduler-screen checkpoint or use final seeds.
5. Update the final PDF only after the required controlled experiments and
   ablations exist; keep all caveats and paper comparisons explicit.
