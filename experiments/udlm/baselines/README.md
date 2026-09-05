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
