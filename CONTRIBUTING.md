# Contributing to gated-sae-tf

Thanks for your interest in improving `gated-sae-tf`! This is a small, focused
library — a faithful TensorFlow/Keras implementation of the gated sparse
autoencoder — and contributions that keep it clean and well-tested are very
welcome.

## Development setup

```bash
git clone https://github.com/aishwaryanatesh-hub/gated-sae-tf
cd gated-sae-tf
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The test suite is CPU-only and uses small synthetic tensors, so it runs in
seconds and never downloads a dataset.

## Guidelines

- **Keep core dependencies minimal.** The core install is `keras`, `tensorflow`,
  `numpy`. Anything else (matplotlib, etc.) belongs behind an optional extra.
- **Match the existing style.** Type hints and docstrings on public functions;
  follow the surrounding code's conventions.
- **Add a test** for any behavior change. Prefer fast synthetic tests.
- **Don't commit trained weights or datasets.** They're gitignored for a reason.

## Roadmap

- Additional SAE variants (vanilla ReLU, JumpReLU, TopK).
- Activation-store helpers for running SAEs on transformer activations.
- Pretrained dictionaries / model-hub integration.

Open an issue to discuss larger changes before sending a PR.
