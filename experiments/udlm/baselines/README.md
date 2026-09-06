# Frozen local MDLM comparator

`mdlm_50000.json` is the compact, committed manifest for the audited local
three-seed GenMol/MDLM benchmark. It freezes exact unrounded metrics, counts,
checkpoint identity, source-artifact hashes, metric definitions, and historical
caveats so UDLM gates do not depend on an untracked output directory or rounded
display values.

The source benchmark requested 1,000 molecules for each seed 0, 1, and 2 from
the 50,000-step checkpoint. The released-compatible means were validity
1.000000, uniqueness 0.9986666666666667, quality 0.858000, and diversity
0.8230213192558725. Strict unrepaired diagnostics remain separate.

This manifest is a comparator, not an exact paper reproduction. In particular,
the historical run used a dirty source tree and a shared GPU policy that is no
longer permitted. The raw rows remain in the original project output and are
identified by per-seed SHA-256 digests; they are not duplicated here.

The source aggregate SHA-256 is
`b474efc593b665489359425dbe1ed0873f8ae1d44b77478b871aff6d6b555904`.
The committed manifest SHA-256 is
`6da46fc615dedbcca436da087a2c1e9145f5d110036e0c15bb431ded3c2e5539`.

## Immutable current-code rescore

`experiments/udlm/baselines/mdlm_50000_rescore_attestation.json` is the
immutable current-code rescore attestation for that comparator. Its SHA-256 is
`6326b63c38c7052d0b47282d611618f77637496da2785779af69097fc1441323`,
and it was produced from clean, already-pushed revision
`74482c2742ab5ad15def122c809a6b4e403e94cf`. Three fresh CPU Python
interpreters rescored seeds 0, 1, and 2 with `PYTHONHASHSEED` equal to the
generation seed. All 63,000 of 63,000 row cells matched the historical rows
under the frozen comparison rules, and the released-compatible and strict
per-seed metrics and aggregates matched exactly.

The attestation binds the pinned SA table and the hashes of the metric,
chemistry, RDKit, SAFE, and runner modules actually loaded at runtime. It did
not regenerate molecules, rewrite schema-2 summaries, or otherwise mutate the
historical raw rows. Offline environment variables and guards on four Python
socket/name-resolution APIs were active, but this was not OS- or process-level
network isolation; native extensions, subprocesses, raw sockets, datagrams,
and other unguarded APIs are outside that claim.

The registered superiority gate pins both this attestation and the compact
manifest by path and SHA-256, and validates their source revision, row-level
agreement, metric provenance, and exact aggregates before considering a UDLM
result. UDLM candidate selection remains separate and predeclared: only the
registered eligible pilot panel enters the quality-then-diversity score, every
attempt stays in the immutable pilot ledger, and one candidate is locked and
pushed before final seeds 0, 1, and 2 are run.
