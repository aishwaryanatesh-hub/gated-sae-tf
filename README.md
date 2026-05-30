# 🧩 gated-sae-tf

### sparse gated autoencoders for TensorFlow & Keras, batteries included

Almost every open-source sparse-autoencoder (SAE) toolkit speaks PyTorch. If you
live in TensorFlow/Keras, you've mostly been left to port things yourself.
`gated-sae-tf` closes that gap: a clean, tested, `pip`-installable **gated SAE**
for dictionary learning and mechanistic interpretability, with the whole
training recipe and the interpretability tooling already wired up.

`🧠 gated SAE` · `📉 warmup→cosine LR` · `🔬 sparsity + sharpness reports` · `🖼️ feature galleries` · `✅ 15 tests` · `📦 pip install`

![Gated SAE overview](https://raw.githubusercontent.com/aishwaryanatesh-hub/gated-sae-tf/main/assets/gated_sae_overview.png)

## 💭 Why this exists

I kept wanting to train sparse autoencoders on Keras models without
reimplementing the gated SAE from scratch or dragging everything over to
PyTorch. So I packaged the version I kept rewriting: the gated formulation from
Rajamanoharan et al. (2024), the training tricks that actually make it converge
(warmup→cosine LR, gradient clipping, decoder normalization, the auxiliary
loss), and the little interpretability utilities you reach for thirty seconds
later — sparsity stats, decoder sharpness, feature galleries — all in one
importable place.

## 🧠 What's a gated sparse autoencoder?

A sparse autoencoder learns an overcomplete dictionary of features that
reconstruct an input while keeping only a few of them active at a time. The
**gated** variant splits two decisions that a plain ReLU SAE tangles together:

- a **gate** path decides *which* features fire: `f_gate = 1[π_gate > 0]`
- a **magnitude** path decides *how much* they fire: `f_mag = relu(W_mag·x_c + b_mag)`

with the magnitudes tied to the gate via a per-feature rescale,
`W_mag = exp(r_mag) ⊙ W_gate`. The sparse code is `f̃ = f_gate ⊙ f_mag` and the
reconstruction is `x̂ = W_dec·f̃ + b_dec`, decoder columns kept unit-norm.

Training minimizes:

```
L = L_reconstruct + λ · L_sparsity + α · L_aux
```

`L_sparsity = Σ relu(π_gate)` drives the sparsity, and `L_aux` reconstructs the
input from `relu(π_gate)` through a frozen (`stop_gradient`) decoder so the gate
path stays well-conditioned. Decoupling the two gives sharper, more
monosemantic features than an L1-penalized ReLU SAE, without the shrinkage
bias. ✨

## 📦 Install

```bash
pip install gated-sae-tf            # core (TensorFlow + Keras 3)
pip install "gated-sae-tf[viz]"     # + matplotlib for the gallery helpers
```

Needs Python ≥ 3.10 and TensorFlow ≥ 2.16 (where Keras 3 is the default). The
package doesn't pin a TensorFlow build variant, so an existing
`tensorflow[and-cuda]` install is respected.

## 🚀 Quickstart

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
    beta_1=0.0, beta_2=0.999))       # beta_1=0 per the paper

sae.fit(X, epochs=20, batch_size=256)
print(sparsity_report(sae, X))       # L0, alive/dead features, top-k share
```

That trains an 8× overcomplete gated SAE on Fashion-MNIST and prints the L0
mean/median, alive/dead feature counts, and the top-k activation share. A full
end-to-end walkthrough notebook ships in the repository (`examples/`).

## 🧰 The API, at a glance

| Symbol | What it does |
| --- | --- |
| `GatedSAE(input_dim, encoding_dim, lambda_sparse=1e-3, aux_weight=1e-2, clip_norm=1.0)` | The model. Custom `train_step`/`test_step`, `encode`/`decode`, `set_lambda` for annealing, full `get_config` serialization. |
| `WarmupCosineDecay(peak_lr, warmup_steps, total_steps)` | Linear warmup → cosine-decay-to-zero LR schedule. |
| `sparsity_report(model, X, batch_size=512, k=20)` | Dict of L0 mean/median, alive/dead counts, alive fraction, top-k activation share. |
| `decoder_sharpness(model)` | `(per_feature_kurtosis, mean)` — higher kurtosis means sharper, more localized features. |
| `plot_decoder_gallery(model, codes, top_n=10)` | Grid of the top decoder directions. Needs the `[viz]` extra. |
| `plot_feature_gallery(model, codes, X, labels, class_names, ...)` | Decoder-direction + top-activating-images view with MONO/POLY tagging. Needs the `[viz]` extra. |

## 🗺️ Roadmap

- More SAE variants (vanilla ReLU, JumpReLU, TopK)
- Activation-store helpers for SAEs on transformer activations
- Pretrained dictionaries / model-hub integration

## 💛 Contributing

Issues and PRs are genuinely welcome — a new SAE variant, a docs fix, a bug
report, all of it. The full walkthrough notebook, the contributing guide, and
the extended bibliography live alongside this file in the repository
(`GUIDE.md` and `CONTRIBUTING.md`).

## 🤖 How it was made

Built with Claude, shaped and reviewed by me. The implementation follows the
gated SAE paper; training, serialization, and every interpretability utility are
covered by a 15-test suite that runs in CI.

## 📚 References & citing

This library implements the gated SAE of **Rajamanoharan et al. (2024)**,
*Improving Dictionary Learning with Gated Sparse Autoencoders*, arXiv:2404.16014
— https://arxiv.org/abs/2404.16014

It sits in the interpretability lineage of **Anthropic's** Transformer Circuits
work on dictionary learning:

- Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models
  With Dictionary Learning.* Transformer Circuits Thread —
  https://transformer-circuits.pub/2023/monosemantic-features
- Templeton et al. (2024). *Scaling Monosemanticity: Extracting Interpretable
  Features from Claude 3 Sonnet.* Transformer Circuits Thread —
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

Full bibliography, BibTeX, and related work (Elhage et al., Gao et al.) are in
`GUIDE.md` in the repository.

## ⚖️ License

MIT © 2026 Aishwarya Natesh.
