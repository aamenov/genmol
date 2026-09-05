# Fragment-vocabulary experiment results

These are immutable, validation-first snapshots of the fragment-credit
experiments. Raw checkpoints and molecule-level traces remain under `output/`
on the experiment host; their hashes are retained in each collection manifest.

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
