# Fragment-vocabulary experiment results

These are immutable, validation-first snapshots of the fragment-credit
experiments. Raw checkpoints and molecule-level traces remain under `output/`
on the experiment host; their hashes are retained in each collection manifest.

## QED 50k Bayesian prior-strength ablation

This exploratory sensitivity study used the final local 50,000-step checkpoint,
QED with `gamma = 0`, 1,000 unique-molecule oracle calls per variant-seed run,
population size 100, warmup 100 with the declared legacy off-by-one behavior,
and paired seeds 0, 1, and 2. The Bayesian arms used a fixed neutral proxy prior
mean of 0.5 and strengths 1, 3, 10, and 30. The original ZINC molecule-level
scores needed for a data-derived prior were unavailable. The checkpoint
SHA-256 is
`8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6`.

All 18 runs completed at exactly 1,000 calls with checkpoint consistency and no
resumes. The collector replayed the checkpoints and event streams before
publishing the immutable CSV and report. Values below are mean +/- sample SD
over the three paired seeds.

| Variant | Prior strength | Top-10 AUC | Final top-10 | Top-100 AUC | Final top-100 | Paired top-10 AUC delta vs running mean |
|---|---:|---:|---:|---:|---:|---:|
| `released` | n/a | 0.894028 +/- 0.001708 | 0.946821 +/- 0.000549 | 0.853075 +/- 0.003966 | 0.933910 +/- 0.003292 | -0.001842 +/- 0.003984 |
| `running_mean` | 0 | 0.895869 +/- 0.002388 | 0.947197 +/- 0.000532 | 0.857370 +/- 0.004860 | 0.938896 +/- 0.001739 | 0.000000 +/- 0.000000 |
| `shrink1` | 1 | 0.898073 +/- 0.000738 | 0.946953 +/- 0.000143 | 0.854082 +/- 0.005275 | 0.925878 +/- 0.001675 | +0.002204 +/- 0.002767 |
| `shrink3` | 3 | 0.896511 +/- 0.002984 | 0.945883 +/- 0.001473 | 0.850591 +/- 0.006188 | 0.923429 +/- 0.006982 | +0.000642 +/- 0.001985 |
| `shrink10` | 10 | 0.897395 +/- 0.000801 | 0.946290 +/- 0.000463 | 0.848850 +/- 0.002930 | 0.922418 +/- 0.001726 | +0.001526 +/- 0.001807 |
| `shrink30` | 30 | 0.896810 +/- 0.001299 | 0.945458 +/- 0.000724 | 0.851198 +/- 0.002153 | 0.922835 +/- 0.002996 | +0.000941 +/- 0.001751 |

Running-mean updating improved the observed mean top-10 AUC by 0.001842 and
final top-100 by 0.004986 relative to the released one-shot update. Among the
tested shrinkage arms, strength 1 had the largest observed mean top-10 AUC and
its paired AUC deltas were positive in all three seeds. However, all four
shrinkage strengths had lower final top-10 and substantially lower final
top-100 than running mean. There was no monotonic dose response. These results
describe a small exploratory screen; they do not establish an optimal strength
or a general improvement in sample efficiency.

The jobs ran from clean code commit `47f7692cc21eda826511ce311c9dfa7573df4c0c`
on physical GPUs 3--6. Every launch snapshot was below the strict 10%
utilization threshold and had at least 47,163 MiB free. Low-utilization
processes were already present on those GPUs, as explicitly authorized, so
wall-time comparisons are suppressed. The controller span was approximately
1,070 seconds; individual run records ranged from 147.2 to 299.1 seconds.

The authoritative files are in `qed_50k_bayes_strength_1k_v1/`: the schema-3
collection manifest, hash-named CSV, hash-named six-page PDF, and report
manifest. The PDF reports every top-1/top-10/top-100 endpoint and normalized
AUC, per-seed values, paired deltas, trajectories, configurations, provenance,
and caveats. The paper's QED reference is 0.942 +/- 0.000 PMO AUC top-10 at
10,000 calls; it is not directly comparable to this 1,000-call local study.

## QED 40k core comparison

`qed_40k_core_1k_v1/` is the validated three-seed archive for the earlier 40k
checkpoint using the same 1,000-call QED protocol for `released`,
`running_mean`, and `shrink10`. Its mean top-10 AUC values were 0.893836,
0.896226, and 0.897346, respectively. The 50k rerun preserves the same
descriptive trade-off: running mean improves broad final top-100 performance,
while strength-10 shrinkage gives a small top-10 AUC increase but lowers final
top-100. The two checkpoint screens are exploratory and are not independent
confirmatory evidence.

## QED engineering smoke, v2

The repaired smoke used the preliminary 40k model checkpoint, QED, seed 0, a
200-unique-molecule budget, population size 100, warmup 20, `gamma = 0`, and the
legacy count-one approximation for seed fragments. `shrink10` used the declared
proxy prior mean 0.5. It ran each arm serially on physical GPU UUID
`GPU-997e881a-bd08-aa54-45f3-9b3c6fdf6ece`; every launch snapshot recorded 0%
utilization, no pre-existing compute process, and no GPU sharing. The matrix is
`../../configs/smoke_40k_v2.yaml` (SHA-256
`16c546d12d598a376ee0d0be72e0de90bc409f81cd1ced7528d9cf09cfbd758c`).
The model SHA-256 is
`2c153c347a749c671661d7d4d8f04d4bf8857a4a6d0fef87cceab40d2785f88e`
and the QED vocabulary SHA-256 is
`6b98420f3fa88835e50cb28b960225dc6c7857a29f3057fb1ca22b4211c1806c`.

All values below were recomputed from event streams and checkpoints by the
collector, rather than copied from the run summaries.

| Variant | All-call top-10 AUC | Final top-1 | Final top-10 | Charged children | Child AUC on total-call axis | Dense child-count AUC |
|---|---:|---:|---:|---:|---:|---:|
| `released` | 0.882666 | 0.947478 | 0.944204 | 200 | 0.882666 | 0.882666 |
| `running_mean` | 0.884202 | 0.947785 | 0.944228 | 200 | 0.884202 | 0.884202 |
| `support3` | 0.884443 | 0.947318 | 0.944736 | 200 | 0.884443 | 0.884443 |
| `shrink10` | 0.888697 | 0.948331 | 0.945430 | 200 | 0.888697 | 0.888697 |
| `running_mean_parent_control` | 0.877888 | 0.945658 | 0.938600 | 110 | 0.875175 | 0.835720 |
| `delta` | 0.868360 | 0.948442 | 0.938188 | 108 | 0.863501 | 0.823399 |

The hash-named CSV and `smoke_40k_v2/collection_manifest.json` are the
authoritative schema-3 compact archive. The manifest retains configurations,
commands, launch attempts, GPU snapshots, code identity, source paths, and
source hashes. It also proves that all six matrix jobs are present. The four
`context_*.analyzer-81060b4a04d3.json` files are replay-validated descriptive
analyses for the absolute-score statistical arms and bind the analyzer plus its
imported experiment-I/O helper by SHA-256.

Historical v2 run manifests preserve a cosmetic provenance defect: the first
`git status --porcelain` entry lost one leading status-column space. Dirty-state,
commit, and tracked-diff hashes are unaffected. The recorder is repaired for
future runs. Untracked file contents were not captured by the historical
tracked-diff hash. The launch commands prove that durable event writes were
enabled, but this flag was omitted from the historical hashed resolved config;
future configs record it explicitly.

This is an engineering smoke, not an efficacy result. It has one oracle, one
seed, only 200 calls, a shortened warmup, a preliminary model checkpoint, and a
proxy shrinkage prior. The small apparent advantage for `shrink10` cannot be
interpreted as a treatment effect. Parent-scoring arms spend part of the budget
on parents, and their dense child-count AUCs end after only 108--110 children;
those scalars do not share the 200-child horizon of the other arms.

## Fragment-context diagnostic from the v2 smoke

The analyzer associates each accepted child's whole-molecule QED score with
each unique fragment emitted by that event's sampled cut. It then validates and
replays the final population and oracle state before reporting repeated-context
statistics.

| Variant | Fragment observations | Distinct fragments | Repeated fragments | Singleton fraction | Pooled within-fragment sample SD | First-score minus final-mean |
|---|---:|---:|---:|---:|---:|---:|
| `running_mean` | 997 | 724 | 82 | 0.886740 | 0.117994 | 0.010110 |
| `support3` | 994 | 709 | 73 | 0.897038 | 0.118966 | -0.000278 |
| `shrink10` | 1010 | 711 | 80 | 0.887482 | 0.121645 | 0.013099 |
| `running_mean_parent_control` | 563 | 437 | 44 | 0.899314 | 0.118874 | 0.021506 |

The nonzero conditional dispersion supports the premise that the same sampled
fragment can occur in materially different-score molecular contexts. The
first-score statistic is not held out, however: its final mean includes the
first observation, and each policy generates its own adaptive contexts. These
diagnostics are descriptive and non-causal, and they do not establish fewer
oracle calls.

## Paired contextual simulation, v2

The CPU-only mechanism diagnostic used 200 latent fragments, capacity 20, 20
observations per fragment, 30 paired replicates, seed 20260905, and configured
pre-clipping context scales 0.05, 0.2, and 0.4. Latent means, observation order,
and standardized context offsets were paired across scales. The exact summary
and provenance are in `context_variance_v2/`.

At scale 0.2, the released one-shot policy had mean top-20 regret 0.070967,
mean oracle-set Jaccard 0.277346, and mean optimism 0.237382. The exhaustive
running-mean and support variants recovered the empirical panel ranking exactly
at the endpoint by construction. At scale 0.4, the released values were
0.140778 regret, 0.136964 Jaccard, and 0.368932 optimism. This demonstrates a
winner's-curse mechanism in the toy model only; it does not use molecules or a
PMO oracle. The shrinkage arm also uses privileged complete-panel prior
information and is not an online PMO comparison.

## Superseded v1 smoke

`smoke_40k_v1/` is retained for auditability. It preceded canonical cross-arm
sampling-order and child-metric-axis repairs, and its historical launch records
do not contain the full v2 policy evidence. Do not combine or average its
numbers with v2.

`micro_40k_v1/` contains the earlier five-call released-policy engineering
check. It validates only the end-to-end execution path and has no algorithmic
interpretation.
