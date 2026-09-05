# UDLM CPU smoke results

These runs are integration checks, not evidence that UDLM beats GenMol. Both
used implementation commit `02595d1`, seed 1, the same 16-molecule toy set, a
two-layer 64-hidden BERT, 100 optimizer updates, and a 32-step reverse chain.

| Uniform prior | First/last 5-step loss | Fixed loss at t=0.5, before/after | Strict valid / 16 |
|---|---:|---:|---:|
| Full 1,880-token vocabulary | 6.220 / 0.540 | 6.235 / 1.600 | 5 |
| Exclude five control tokens | 6.226 / 0.628 | 6.351 / 1.430 | 3 |

Both variants pass the intended gate: the objective falls sharply, gradients
remain finite, clean-token accuracy rises from zero, and the reverse chain
produces strictly decodable molecules. The validity counts are tiny stochastic
samples from an intentionally underfit toy model and do not rank the two prior
policies. Full JSON outputs are retained alongside this note.
