# Fixed UDLM validation panel

`first_256.json` freezes the first 256 non-empty rows from the revision-pinned
SAFE-GPT validation stream. It is intended for matched denoising diagnostics
(time bins, changed/unchanged tokens, and frequency buckets), not as evidence
of de novo molecular-generation quality.

## Provenance

- Dataset: `datamol-io/safe-gpt`, validation split, revision
  `b83175cd7394e7a4027478a35b2f9d1dda3ac62f`
- Tokenizer: `datamol-io/safe-gpt`, revision
  `3d5fa0988383e898d5ac5db7cd52bf715bc37061`
- `tokenizer.json` SHA-256:
  `0db5f4dbdc7e8ff759e98483759611a426e187ee7f3f0a91edc8800abe7bf140`
- Materializer source commit: `21680680fb39844f302249fc1987a76f8976f602`
- Selection: first 256 non-empty streaming rows; source rows 139 and 140
  were empty and skipped, so the last selected source index is 257.
- Ordered SAFE-text digest:
  `7eeedb706c4748b1fcd8e91b65a7b1d79a58053c994b7ce0c4b1ed273cdf616c`
- Ordered token-ID digest:
  `a641f0335f0c155040fdce6dcb131af3dd50be364a29e54c7517968e6f3e2be2`
- `first_256.json` SHA-256:
  `e2493da4f3cb3217b48c78dc2901dc7524d7d90dc3959cffa87a1b6f8a9a7658`

The JSON intentionally excludes raw SAFE strings. It stores token IDs, content
lengths, source indices, and per-row hashes. Content lengths range from 18 to
254 tokens (median 47.5).

## Reproduce and validate

Run from this repository root with the project virtual environment and the
worktree source first on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD/src:$PWD" \
  /home/aidar.alimbayev/Documents/genmolv2/.venv/bin/python \
  scripts/udlm/materialize_validation_panel.py \
  --sample-count 256 \
  --output experiments/udlm/validation_panel/first_256.json
```

The materializer refuses to overwrite an existing panel unless `--force` is
passed. Structural validation checks increasing source indices, BOS/EOS
framing, vocabulary bounds, content counts, and the ordered token-ID digest.
