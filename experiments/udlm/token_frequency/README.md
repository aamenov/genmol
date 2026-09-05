# Pinned SAFE token-frequency diagnostic

This CPU-only diagnostic analyzes the first 10,000 rows of the pinned SAFE-GPT
training stream. It is hypothesis-selection evidence, not a generative
benchmark and not evidence that UDLM beats GenMol.

The 10,000 examples contain 517,090 editable content tokens (median length 48).
Only 184 of GenMol's 1,880 base token IDs occur; 34 types cover 99% of tokens,
68 cover 99.9%, and 139 cover 99.99%. Empirical token perplexity is 15.80.

The diagnostic supports the large-vocabulary risk hypothesis:

- IDs 1680–1879 have zero occurrences in this prefix, but a faithful uniform
  prior assigns them 10.64% mass per position.
- At the observed median length, the probability of at least one such uniform
  draw is 99.55%.
- The five tokenizer control symbols account for only 0.266% per position and
  a 12.00% sequence-level hit probability. Excluding them is therefore too
  small a change to address the much larger vocabulary-tail mismatch.

The ID-tail boundary is a diagnostic heuristic, not a claim that every token
above 1679 is chemically impossible. Likewise, zero occurrences in a 10,000-row
prefix does not prove zero mass in the full corpus. The next scientific step is
to derive an exact non-uniform categorical process with a uniform floor, while
retaining full-uniform UDLM as the faithful control.

Provenance is embedded in `train_first_10000.json`: dataset revision
`b83175cd...`, tokenizer revision `3d5fa098...`, tokenizer SHA-256
`0db5f4db...`, ordered text digest `53aee8e5...`, seed-free prefix selection,
and implementation commit `56a96b2`. The artifact SHA-256 is
`088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed`.
