# gated-sae-tf

**A TensorFlow/Keras *Sparse Gated Autoencoder* (Gated SAE)** for dictionary
learning and mechanistic interpretability.

Almost all open-source sparse autoencoder (SAE) tooling is written in PyTorch.
`gated-sae-tf` fills the gap for TensorFlow/Keras users: a clean, tested,
`pip`-installable implementation of the **gated** SAE from
[Rajamanoharan et al. (2024)](https://arxiv.org/abs/2404.16014), with the
training recipe (warmup→cosine LR, gradient clipping, decoder normalization,
auxiliary loss) and the interpretability tooling (sparsity stats, decoder
sharpness, feature galleries) packaged for reuse.

![Gated SAE overview](https://raw.githubusercontent.com/aishwaryanatesh-hub/gated-sae-tf/main/assets/gated_sae_overview.png)

![Gated SAE architecture](https://raw.githubusercontent.com/aishwaryanatesh-hub/gated-sae-tf/main/assets/gated_sae_architecture.png)

## What is a sparse gated autoencoder?

A sparse autoencoder learns an overcomplete dictionary of features that
reconstruct an input while keeping only a few features active at a time. The
**gated** variant separates two decisions that a plain ReLU SAE conflates:

- a **gate** path decides *which* features are active: `f_gate = 1[π_gate > 0]`,
- a **magnitude** path decides *how much* they fire: `f_mag = relu(W_mag·x_c + b_mag)`,

with the magnitude weights **tied** to the gate weights via a per-feature
rescaling, `W_mag = exp(r_mag) ⊙ W_gate`. The sparse code is `f̃ = f_gate ⊙ f_mag`
and the reconstruction is `x̂ = W_dec·f̃ + b_dec` (decoder columns kept unit-norm).

Training minimizes

```
L = L_reconstruct + λ · L_sparsity + α · L_aux
```

where `L_sparsity = Σ relu(π_gate)` drives sparsity and `L_aux` reconstructs the
input from `relu(π_gate)` through a frozen (`stop_gradient`) decoder to keep the
gate path well-conditioned. This decoupling gives sharper, more monosemantic
features than an L1-penalized ReLU SAE without the shrinkage bias.

## Install

```bash
pip install gated-sae-tf            # core (TensorFlow + Keras 3)
pip install "gated-sae-tf[viz]"     # + matplotlib for the gallery helpers
pip install "git+https://github.com/aishwaryanatesh-hub/gated-sae-tf"  # bleeding edge
```

Requires Python ≥ 3.10 and TensorFlow ≥ 2.16 (where Keras 3 is the default).
The package does not pin a TensorFlow build variant, so an existing
`tensorflow[and-cuda]` install is respected.

## Quickstart

```python
import keras, numpy as np
from gated_sae import GatedSAE, WarmupCosineDecay, sparsity_report

(x_train, _), _ = keras.datasets.fashion_mnist.load_data()
X = x_train.reshape(-1, 784).astype("float32") / 255.0

sae = GatedSAE(input_dim=784, encoding_dim=784 * 8,   # 8x overcomplete
               lambda_sparse=1e-3, aux_weight=0.1, clip_norm=1.0)
sae(X[:2])                          # build the weights
sae.b_dec.assign(X.mean(axis=0))    # init decoder bias to the data mean

steps = (len(X) // 256) * 20
sae.compile(optimizer=keras.optimizers.Adam(
    WarmupCosineDecay(1e-3, warmup_steps=steps // 10, total_steps=steps),
    beta_1=0.0, beta_2=0.999))       # beta_1=0 per the report

sae.fit(X, epochs=20, batch_size=256)
print(sparsity_report(sae, X))       # L0, alive/dead features, top-k share
```

See [`examples/train_your_first_gated_sae.ipynb`](examples/train_your_first_gated_sae.ipynb)
for the full walkthrough on Fashion MNIST.

## API

| Symbol | Description |
| --- | --- |
| `GatedSAE(input_dim, encoding_dim, lambda_sparse=1e-3, aux_weight=1e-2, clip_norm=1.0)` | The model. Custom `train_step`/`test_step`, `encode`/`decode`, `set_lambda` for annealing, full `get_config` serialization. |
| `WarmupCosineDecay(peak_lr, warmup_steps, total_steps)` | Linear warmup → cosine-decay-to-0 LR schedule. |
| `sparsity_report(model, X, batch_size=512, k=20)` | Dict of L0 mean/median, alive/dead counts, alive fraction, top-k activation share. |
| `decoder_sharpness(model)` | `(per_feature_kurtosis, mean)` — higher kurtosis = sharper, more localized features. |
| `plot_decoder_gallery(model, codes, top_n=10)` | Grid of the top decoder directions (`[viz]` extra). |
| `plot_feature_gallery(model, codes, X, labels, class_names, ...)` | Combined decoder-direction + top-activating-images view with MONO/POLY tagging (`[viz]` extra). |

## Roadmap

- Additional SAE variants (vanilla ReLU, JumpReLU, TopK).
- Activation-store helpers for SAEs on transformer activations.
- Pretrained dictionaries / model hub integration.

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## References & further reading

**Architecture.** This library implements the gated SAE formulation of:

- Rajamanoharan, S., Conmy, A., Lieberum, T., Varma, V., Kramár, J., Shah, R., &
  Nanda, N. (2024). *Improving Dictionary Learning with Gated Sparse
  Autoencoders.* arXiv:2404.16014. <https://arxiv.org/abs/2404.16014>

**Interpretability lineage** (Anthropic, Transformer Circuits Thread):

- Bricken, T., et al. (2023). *Towards Monosemanticity: Decomposing Language
  Models With Dictionary Learning.* Transformer Circuits Thread.
  <https://transformer-circuits.pub/2023/monosemantic-features>
- Elhage, N., et al. (2022). *Toy Models of Superposition.* Transformer Circuits
  Thread. <https://transformer-circuits.pub/2022/toy_model/index.html>
- Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B.,
  Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L.,
  McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E.,
  Batson, J., Jermyn, A., Carter, S., Olah, C., & Henighan, T. (2024). *Scaling
  Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.*
  Transformer Circuits Thread.
  <https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html>

```bibtex
@article{templeton2024scaling,
  title  = {Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet},
  author = {Templeton, Adly and Conerly, Tom and Marcus, Jonathan and Lindsey, Jack and Bricken, Trenton and Chen, Brian and Pearce, Adam and Citro, Craig and Ameisen, Emmanuel and Jones, Andy and Cunningham, Hoagy and Turner, Nicholas L. and McDougall, Callum and MacDiarmid, Monte and Freeman, C. Daniel and Sumers, Theodore R. and Rees, Edward and Batson, Joshua and Jermyn, Adam and Carter, Shan and Olah, Chris and Henighan, Tom},
  year   = {2024},
  journal= {Transformer Circuits Thread},
  url    = {https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html}
}
```

**Related work.**

- Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A.,
  Sutskever, I., Leike, J., & Wu, J. (2024). *Scaling and evaluating sparse
  autoencoders* (TopK SAEs). arXiv:2406.04093.
  <https://arxiv.org/abs/2406.04093>

## License

MIT © 2026 Aishwarya Natesh. See [LICENSE](LICENSE).
