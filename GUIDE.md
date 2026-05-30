# 📖 Guide, references & further reading

This file is the linked-out companion to the [README](README.md).

## 🚀 Full walkthrough

The end-to-end tutorial trains a gated SAE on Fashion-MNIST, then walks through
sparsity stats, decoder sharpness, and the feature galleries:

- [`examples/train_your_first_gated_sae.ipynb`](examples/train_your_first_gated_sae.ipynb)

## 🧪 Bleeding edge

```bash
pip install "git+https://github.com/aishwaryanatesh-hub/gated-sae-tf"
```

## 💛 Contributing

Contributions are genuinely welcome — new SAE variants, docs, bug reports, and
better examples especially. See [CONTRIBUTING.md](CONTRIBUTING.md) to get
started. The fastest way to find a first contribution is to run the walkthrough
notebook and fix the first rough edge you hit.

## 📚 References

### Architecture

This library implements the gated SAE formulation of:

- Rajamanoharan, S., Conmy, A., Lieberum, T., Varma, V., Kramár, J., Shah, R., &
  Nanda, N. (2024). *Improving Dictionary Learning with Gated Sparse
  Autoencoders.* arXiv:2404.16014. <https://arxiv.org/abs/2404.16014>

### Interpretability lineage (Anthropic, Transformer Circuits Thread)

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

@article{rajamanoharan2024gated,
  title  = {Improving Dictionary Learning with Gated Sparse Autoencoders},
  author = {Rajamanoharan, Senthooran and Conmy, Arthur and Lieberum, Tom and Varma, Vikrant and Kram{\'a}r, J{\'a}nos and Shah, Rohin and Nanda, Neel},
  year   = {2024},
  journal= {arXiv preprint arXiv:2404.16014},
  url    = {https://arxiv.org/abs/2404.16014}
}
```

### Related work

- Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A.,
  Sutskever, I., Leike, J., & Wu, J. (2024). *Scaling and evaluating sparse
  autoencoders* (TopK SAEs). arXiv:2406.04093.
  <https://arxiv.org/abs/2406.04093>

## ⚖️ License

MIT © 2026 Aishwarya Natesh. See [LICENSE](LICENSE).
