# Uniform Diffusion Language Model reference

The implementation in `src/genmol/diffusion.py` and the sinusoidal timestep
embedding in `src/genmol/backbone.py` were derived with reference to:

- Yair Schiff et al., *Simple Guidance Mechanisms for Discrete Diffusion
  Models*, arXiv:2412.10193.
- `kuleshov-group/discrete-diffusion-guidance`, immutable source revision
  `edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661`.

That repository is distributed under the Apache License 2.0. The complete
license text is already included in this project as
`LICENSE/3rd_party/LICENSE_MDLM` and `LICENSE/license_code.txt`.

The implementation deliberately repairs a cancellation-prone but
algebraically equivalent form of the continuous-time loss and skips an unused
reconstruction forward pass. These differences are documented in the source.
