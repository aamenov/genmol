# Full-model CPU trainer smoke

This is integration evidence, not evidence that UDLM beats GenMol. At commit
`4205f6e`, the production 12-layer BERT, hosted SAFE stream, exact UDLM
loss, MDLM-EMA warm-start, optimizer, scheduler, EMA update, Lightning
checkpoint hook, and checkpoint reload completed one CPU optimizer step.

- No GPU was visible or used.
- The effective global batch was 2 and seed was 1.
- Lightning displayed training loss 1.490 and stopped normally at one step.
- Reload recovered diffusion type `udlm`, 210 model-state entries, and 206 EMA
  entries.
- The 1.41 GB checkpoint remains under ignored `output/`; its path, byte count,
  and SHA-256 are recorded in `result.json`, but the binary is not committed.

The run used the then-configured 64 sampling steps, although it did not sample.
The faithful official sampling control is now explicitly 128 steps; 32/64 are
speed ablations.

The historical smoke loaded the hosted training stream without enforcing its
revision. It is therefore unsuitable as dataset-reproducible metric evidence;
subsequent training code pins dataset revision `b83175cd...` and tokenizer
revision `3d5fa098...` with a tokenizer checksum.
