# Fragment-vocabulary scoring ablations

## Status and scope

This document preregisters a readout of fragment-credit hypotheses before any
optimization runs are inspected. The primary setting is the 23-task PMO hit-
generation benchmark with a fixed 10,000 **unique canonical-molecule** oracle
budget per task. GenMol checkpoint, initial vocabulary, population size
(`V = 100`), warmup, molecule-size filters, sampler settings, MCG setting,
fragmentation rule, and seed set must be identical across variants.

The released implementation is the scientific reference, not an instruction.
The paper's Eq. 5 and Algorithm 1 motivate the alternatives; none is assumed to
be an improvement. Lead optimization is outside the primary comparison because
its released population is unbounded and contains repeated fragment entries, so
changing a stored score alone need not change uniform sampling.

Use the exact task-specific MCG corruption rates from paper Table 8 in every
arm; `w = 2` is fixed:

| PMO oracle | `gamma` |
|---|---:|
| `albuterol_similarity` | 0.2 |
| `amlodipine_mpo` | 0.3 |
| `celecoxib_rediscovery` | 0.0 |
| `deco_hop` | 0.2 |
| `drd2` | 0.0 |
| `fexofenadine_mpo` | 0.0 |
| `gsk3b` | 0.0 |
| `isomers_c7h8n2o2` | 0.5 |
| `isomers_c9h10n2o2pf2cl` | 0.0 |
| `jnk3` | 0.5 |
| `median1` | 0.2 |
| `median2` | 0.2 |
| `mestranol_similarity` | 0.0 |
| `osimertinib_mpo` | 0.0 |
| `perindopril_mpo` | 0.4 |
| `qed` | 0.0 |
| `ranolazine_mpo` | 0.0 |
| `scaffold_hop` | 0.0 |
| `sitagliptin_mpo` | 0.2 |
| `thiothixene_rediscovery` | 0.3 |
| `troglitazone_rediscovery` | 0.0 |
| `valsartan_smarts` | 0.4 |
| `zaleplon_mpo` | 0.4 |

## Notation and estimand

At unique oracle call `t`, let `x_t` be the canonical generated molecule,
`y_t = y(x_t) in [0, 1]` its PMO score (larger is better), and `F(x_t)` the set
of unique fragments returned by the vocabulary decomposition rule. A fragment
occurring twice in one molecule contributes one molecular context, matching the
set in paper Eq. 5. For fragment `f`,

```text
n_f(t) = sum_{i <= t} 1[f in F(x_i)]
s_f(t) = sum_{i <= t} y_i 1[f in F(x_i)]
mu_f(t) = s_f(t) / n_f(t), when n_f(t) > 0.
```

Thus Eq. 5 is

```text
mu_f = (1 / |S(f)|) sum_{x in S(f)} y(x),
S(f) = {x : f is a subgraph/fragment of x}.
```

Algorithm 1 repeatedly attaches two uniformly sampled population fragments,
remasks a finer fragment region, scores the child, decomposes it, and retains
the top `V` fragments. Scores determine top-`V` membership only; sampling within
the retained population remains uniform in every arm. The controlled harness
sorts active fragment strings canonically immediately before drawing uniform
RNG indices. This leaves the sampling distribution unchanged while keeping
paired arms on the same proposal stream whenever their active fragment sets
are equal. Equal ranking scores use the released implementation's reverse-tuple
rule (fragment string descending) in every arm, so a capacity-boundary tie does
not itself change those active sets.

Only the first occurrence of a canonical molecule updates sufficient statistics
in the statistical arms. Cached duplicate proposals consume no oracle call and
provide no duplicate statistical evidence. The `released` arm deliberately
retains the GitHub behavior of re-decomposing cached duplicates. All candidate
statistics in statistical arms persist across eviction and re-entry.

## Locked variants

### `released`

This is the code-faithful reference. Load the same top `V` rows from the released
per-oracle CSV. For a scored child with `y_t` above the current population floor,
decompose it and add every fragment not currently retained with stored score
`y_t`. Do not update retained fragments. Sort by stored score and truncate to
`V`. An evicted fragment may later be reintroduced with a new one-shot score,
as in the released current-population membership test.

Its dynamic estimator is therefore approximately

```text
q_f = y(x_tau_f),
tau_f = first admission time of f while it remains retained,
```

not Eq. 5's mean across molecular contexts.

### `running_mean`

Maintain `(n_f, s_f)` for every fragment in every unique scored child, including
already-known and non-retained fragments, and rank eligible fragments by
`mu_f = s_f / n_f`. This is the literal online Eq. 5 estimator over the observed
generated set. There is no `y_t > population floor` observation filter.

The existing initial-vocabulary CSV stores means but not counts. Therefore each
loaded initial fragment is explicitly treated as one pseudo-observation with
`n_f = 1` and `s_f = loaded_score`; a new fragment begins at zero. This common
convention makes the dynamic ablation executable but is **not** the exact Eq. 5
mean over ZINC plus generated molecules.

### `support3`

Use the `running_mean` estimator, but a newly observed fragment is in a probation
registry and cannot enter top `V` until `n_f >= 3`. Loaded initial fragments are
grandfathered as eligible because their true support is unknown. Probationary
fragments continue accumulating observations from all unique scored children.
No exploratory population slots are added.

### `shrink10`

Use all `running_mean` observations and rank by the fixed pseudo-count shrinkage
score

```text
q_f^(10) = (n_f mu_f + 10 mu_0) / (n_f + 10)
         = (s_f + 10 mu_0) / (n_f + 10),
```

where `mu_0` is the per-oracle mean molecular score in the same initial
ZINC250k table used to construct the vocabulary. `mu_0` is computed once before
optimization and never updated. Initial CSV scores again carry declared support
one. The value `lambda = 10` is locked globally and is not tuned per oracle.

### `delta`

Build on `running_mean`. After warmup, define `p_t` as the valid attached
molecule immediately before fragment remasking and `c_t = x_t` as its remasked
child. Score both and orient the local change as

```text
Delta_t = y(c_t) - y(p_t).
A_t = F(c_t) - F_all(p_t)   # sampled child fragments absent from the parent
d_t(f) = Delta_t             for f in A_t.
```

`F_all` deterministically cuts every eligible parent bond only for approximate
changed-fragment mapping; `F` remains the paper's three-cut sampled vocabulary
rule. Empty sets receive no update. When several child fragments are credited,
each receives the full molecular delta; splitting credit is a different
estimand and is not part of this first ablation. Let `m_f` be the number of
delta observations and `dbar_f` their running mean. Rank fragments by

```text
q_f^delta = dbar_f, with dbar_f = 0 when m_f = 0.
```

Do not clip this ranking score. Offline seed scores break zero-delta ties but
are never added to the uplift estimand. During warmup, keep the initial
population fixed and skip delta updates. This pure-uplift policy is exploratory:
it tests the user's local-credit idea without mixing absolute molecular score
and score change in one quantity.

Every unique parent score counts toward the 10,000-call budget and is visible to
the standard PMO metric, exactly like every other evaluated molecule. The
throughput comparison for `delta` is an auxiliary
`running_mean_parent_control` that scores the same parent/child types and spends
the same budget. It does **not** isolate delta credit: it uses absolute running
means, updates all sampled child fragments, and updates during warmup. It only
separates the effect of spending parent calls from the ordinary `running_mean`
arm. Delta-credit conclusions therefore require the held-out mechanism metrics
below, not this optimization comparison alone.

## Falsifiable hypotheses

- **H1 — one-shot optimism.** Because admission selects a favorable molecular
  context, `released` scores will exceed fragments' subsequent held-out context
  means. `running_mean` will reduce signed optimism and future-context error.
  A PMO AUC gain is a separate prediction, not implied by better calibration;
  optimistic one-shot credit could instead aid exploration.
- **H2 — minimum support.** `support3` will reduce singleton occupancy, rapid
  eviction/re-entry, and across-seed vocabulary variance relative to
  `running_mean`. It may reduce AUC if requiring recurrence blocks genuinely
  useful rare fragments.
- **H3 — shrinkage.** `shrink10` will improve held-out calibration most strongly
  at low support and reduce run-to-run AUC variance relative to `running_mean`.
  It may favor common generic fragments and reduce novelty or final score.
- **H4 — local delta credit.** `dbar_f` will positively predict future score
  changes when the same fragment is independently added. A delta optimization
  gain is secondary because the parent-budget control is throughput-matched but
  not estimator-identical. Non-additive fragment interactions or ambiguous
  region mapping may falsify this.

A mechanism hypothesis is supported only if its preregistered mechanism metric
improves out of sample. An optimization hypothesis is supported only if the
paired aggregate AUC confidence interval excludes zero in the predicted
direction. Otherwise report it as unsupported or inconclusive; do not relabel a
secondary endpoint as primary.

## Metrics and statistics

### Primary optimization endpoint

- Exact PMO AUC of average top-10 molecular score versus unique oracle calls,
  evaluated through 10,000 calls.
- Aggregate as a per-seed sum over all 23 tasks, matching the paper's headline
  sum while retaining its seed distribution.
- Report each task's mean and standard deviation, paired variant-minus-control
  differences, aggregate mean difference, a 95% hierarchical bootstrap
  confidence interval over tasks and paired seeds, and task win/tie/loss counts.

For parent-scoring arms, the primary curve includes every charged parent and
child molecule, because both spend oracle budget. Two child diagnostics must be
reported separately: child quality on the true total-call axis (retaining each
charged child's global call index), and child quality on a dense child-count
axis. Never compress child calls to the beginning of the total budget and pad
that curve; doing so overstates sample efficiency. Dense child-count AUCs from
runs with different numbers of children have different horizons and are not
directly comparable; any scalar comparison must use a predeclared common child
count.

Use paired seeds and at least five seeds in the full study. Record exact seeds.
No task-specific variant parameters or post-hoc exclusions are allowed.

### Sample-efficiency and outcome endpoints

- Top-10 AUC and instantaneous top-10 score at 1,000, 5,000, and 10,000 calls.
- Final top-1, top-10, and top-100 scores.
- Valid, unique, and cached-duplicate proposal rates; attempts per unique call;
  wall time; and generation failures.
- Top-10 Morgan diversity, unique Bemis–Murcko scaffold count, and fraction novel
  relative to the initial ZINC set, to expose exploitation collapse.

### Mechanism endpoints

- Future-only calibration: at fixed call checkpoints, compare each stored score
  with the mean score of later unique molecules containing that fragment.
  Report signed error, MAE, RMSE, Spearman rank correlation, calibration slope,
  eligible-fragment coverage, and the censoring rate from fragments never seen
  again. Never evaluate a running mean on observations already used to fit it.
- `Var(y(x) | f in F(x))`, stratified by fragment support, to measure the context
  variability driving the hypothesis. Here support means appearances in the
  sampled three-cut decomposition, not every generated molecule that could
  chemically contain the fragment.
- Top-`V` support distribution, fraction with dynamic support one, admissions,
  evictions, re-entries, occupancy lifetime, turnover, and Jaccard overlap across
  adjacent checkpoints and seeds.
- Fragment atom-count/frequency distributions and sampling counts, separating a
  scoring effect from preference for common or large fragments.
- For `delta`: parent and child calls, `Delta_t` distribution, size of `A_t`,
  mapping coverage, child-parent similarity, and chronological held-out
  correlation/MAE between `dbar_f` and later independent deltas for `f`.

Save molecule-level traces containing call and attempt index, canonical parent
and child, scores, seed, fragment sets, population membership, raw sufficient
statistics, ranking score, and admission/eviction reason. Record checkpoint and
initial-vocabulary hashes plus the full configuration. Launcher runs also bind
the resolved configuration to the exact matrix path and SHA-256. Collection
recomputes metrics from checkpoint/event evidence rather than trusting summary
fields and publishes a hash-named immutable CSV behind an atomic manifest.

The deterministic contextual simulation is only a premise diagnostic. Its
configured `context_scale` is the pre-clipping noise scale; achieved mean and
pooled within-fragment sample SD and the clipping fraction must be reported.
Even a strong synthetic winner's-curse result cannot establish improved PMO
sample efficiency or fewer oracle calls.

## Execution gates

### Smoke gate

Run one representative oracle and one seed with a debug budget and shortened
warmup, clearly marked non-scientific. Proceed only if:

- `released` reproduces the unmodified admission/update decisions for a fixed
  scored-molecule and decomposition trace;
- budgets count unique canonical molecules correctly;
- repeated molecules do not update statistics;
- support and shrinkage equations match hand calculations;
- parent calls are charged in both delta and its matched control;
- all variants preserve `V = 100`, uniform sampling, and deterministic replay
  from the same seed.

### Pilot gate

Run five preregistered task types (similarity, MPO, rediscovery, isomer, and
activity), three paired seeds, 3,000 unique calls, and the paper warmup/config.
This gate is for identifiability and operation, not selecting favorable results.
Proceed to full runs only if every arm has complete traces, at least 100
post-warmup dynamic fragment observations per task/seed, delta mapping coverage
is at least 80%, and no arm violates its budget or invariant. If a gate fails,
repair the protocol, version this preregistration, and rerun the entire pilot;
do not silently drop an arm.

### Full gate

Lock code commit, checkpoint, vocabulary hashes, five seeds, and all settings,
then run every retained arm on all 23 PMO tasks to exactly 10,000 unique calls.
Report all runs and failures. `delta` must include its parent-call-matched control.
Generate the final tables, learning curves, mechanism plots, raw trace manifest,
and benchmark PDF from immutable outputs.

## Known confounds and required controls

- **Missing initial support.** Released CSVs contain fragment means but not
  counts. Support-one initialization is a declared pseudo-prior, not exact
  paper fidelity. A later exact-count sensitivity requires rebuilding a fixed,
  seeded initial vocabulary with `(sum, count)` for every arm.
- **Different observation rules.** `released` ignores low-scoring children for
  vocabulary updates; Eq. 5 variants observe every unique child. Therefore the
  headline comparison is a policy-package comparison. Use fixed logged streams
  for the future-calibration analysis to isolate estimators from on-policy
  feedback.
- **Oracle accounting.** Target-specific initial ZINC scoring is not charged by
  the released PMO run and remains a shared offline prior. Delta parent calls are
  charged online and require the matched control above.
- **Random fragmentation.** The released runner interleaves Python randomness
  for selection and decomposition, so extra cuts can change later proposals.
  The ablation harness instead locks independent streams for population draws,
  reactions, diffusion, and fragmentation; its `released` arm is therefore an
  update-policy reference, not an exact released RNG trajectory.
- **Duplicates and identity.** Canonicalize before oracle lookup, count each
  molecule-fragment pair once, use a set within a molecule, and retain fragment
  history across eviction. Log repaired/disconnected outputs separately.
- **On-policy feedback.** Different populations necessarily create different
  future contexts. Paired seeds reduce variance but do not create identical
  proposals; estimator calibration therefore also requires common-stream replay.
- **Context and structure.** Whole-molecule scores are associative, not causal;
  fragment size, frequency, partner fragments, and epistasis can drive apparent
  credit. Delta is a local joint contrast, not a causal single-fragment effect.
- **Min-support starvation.** A probationary fragment is not sampled and may
  never recur. Report probation admissions and censoring rather than interpreting
  low turnover automatically as success.
- **Delta mapping and throughput.** Remasking can alter several decomposition
  fragments, and largest-component repair can destroy parent-child identity.
  Report mapping coverage and compare only under equal unique-call budgets.
- **Implementation constants.** The primary code-faithful arms use the released
  `iter > warmup` boundary (1,001 zero-indexed attachment-only iterations),
  while the paper describes 1,000; any paper-boundary sensitivity uses
  `iteration >= warmup` and a new run ID. Preserve task-specific molecule-size and
  MCG settings, canonical pre-sampling population order, uniform population
  sampling, and oracle score direction.
- **Multiplicity.** `support = 3` and `lambda = 10` are locked. Any later sweep is
  a new preregistration and must use disjoint development tasks or multiplicity-
  adjusted intervals; it cannot replace these results.
- **GPU sharing.** The launcher defaults to rejecting a GPU with any existing
  compute process. A direct user authorization to share a low-utilization GPU
  must be represented by `--allow-shared-low-utilization`; the launch record
  then preserves every pre-existing process, the strict utilization and memory
  thresholds, whether sharing actually occurred, and the measured GPU state.
  Eligibility is strict (`utilization < threshold`, not `<=`). Scores remain
  usable, but shared-device wall times are marked non-comparable.
