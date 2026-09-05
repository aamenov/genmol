# Superseded: confounded delta-y v1 screen

Do not interpret this run as a matched delta-versus-absolute credit experiment.
All 12 jobs completed successfully at 1,000 unique oracle calls, but the two
target arms had different seed-statistic initialization:

- `delta` used neutral totals and counts of zero, with absolute seed scores only
  breaking ties;
- `running_mean_delta_control` used absolute seed-score pseudo-observations with
  count one.

The initialization difference affects vocabulary admission after the first
post-warmup update, so the resulting contrast combines initialization and
credit-target effects. The collection manifest and hash-named CSV are retained
for auditability. The preliminary PDF and its report manifest are not
authoritative and are intentionally excluded from version control. Experiment
v2 corrects the control, records the initialization and credit policies
explicitly, and reruns the full matrix under a new experiment ID.
