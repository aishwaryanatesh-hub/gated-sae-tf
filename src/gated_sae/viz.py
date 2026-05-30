"""Visualization helpers for Gated SAE dictionaries.

These require matplotlib, an optional dependency::

    pip install "gated-sae-tf[viz]"
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "gated_sae.viz requires matplotlib. Install it with: "
        'pip install "gated-sae-tf[viz]"'
    ) from exc


def plot_decoder_gallery(
    model,
    codes: np.ndarray,
    top_n: int = 10,
    img_shape: Tuple[int, int] = (28, 28),
    vlim: float = 0.15,
):
    """Plot the decoder directions of the most-active features as images.

    Parameters
    ----------
    model:
        Trained :class:`~gated_sae.GatedSAE`.
    codes:
        Sparse codes ``(n_samples, encoding_dim)`` used to rank features by mean
        activation (e.g. from ``model.encode(X)[0]``).
    top_n:
        Number of top features to show.
    img_shape:
        Shape to reshape each decoder row into for display.
    vlim:
        Symmetric color limit for the diverging RdBu colormap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    W_dec = np.asarray(model.W_dec)
    mean_act = np.asarray(codes).mean(axis=0)
    top = np.argsort(mean_act)[::-1][:top_n]

    fig, axes = plt.subplots(1, top_n, figsize=(2 * top_n, 2.4))
    if top_n == 1:
        axes = [axes]
    for ax, feat_idx in zip(axes, top):
        ax.imshow(W_dec[feat_idx].reshape(img_shape), cmap="RdBu_r",
                  vmin=-vlim, vmax=vlim)
        ax.set_title(f"F#{feat_idx}\n{mean_act[feat_idx]:.3f}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"Top {top_n} Decoder Directions (RdBu, fixed scale)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_feature_gallery(
    model,
    codes: np.ndarray,
    X: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str],
    top_features: int = 20,
    top_images: int = 5,
    img_shape: Tuple[int, int] = (28, 28),
    feature_idx: Optional[Sequence[int]] = None,
    vlim: float = 0.15,
):
    """Combined view: decoder direction + the images that most activate it.

    For each of the top features (by mean activation, unless ``feature_idx`` is
    given), shows the decoder direction next to the ``top_images`` inputs that
    activate it most, and tags the feature MONO (all top images share a class)
    or POLY (mixed classes).

    Parameters
    ----------
    model, codes:
        As in :func:`plot_decoder_gallery`.
    X:
        Inputs aligned with ``codes`` (shape ``(n_samples, input_dim)``).
    labels:
        Integer class label per row of ``X``.
    class_names:
        Class-index -> display name.
    top_features, top_images:
        Grid dimensions.
    feature_idx:
        Explicit feature indices to display (overrides the top-by-activation
        ranking).
    img_shape, vlim:
        Display options, as above.

    Returns
    -------
    matplotlib.figure.Figure
    """
    W_dec = np.asarray(model.W_dec)
    codes = np.asarray(codes)
    X = np.asarray(X)
    labels = np.asarray(labels)
    mean_act = codes.mean(axis=0)

    if feature_idx is None:
        feature_idx = np.argsort(mean_act)[::-1][:top_features]
    else:
        feature_idx = np.asarray(feature_idx)[:top_features]
    n_features = len(feature_idx)

    fig, axes = plt.subplots(
        n_features, top_images + 2, figsize=(2 * (top_images + 2), n_features * 1.6))
    fig.patch.set_facecolor("#111111")
    if n_features == 1:
        axes = axes[np.newaxis, :]

    for row, feat in enumerate(feature_idx):
        feat_acts = codes[:, feat]
        top_img = np.argsort(feat_acts)[::-1][:top_images]

        # Column 0: text summary + MONO/POLY tag
        ax = axes[row, 0]
        ax.set_facecolor("#111111")
        seen = [class_names[labels[i]] for i in top_img]
        unique = set(seen)
        if len(unique) == 1:
            tag_label, tag_color = f"MONO: {next(iter(unique))}", "#44cc44"
        else:
            tag_label, tag_color = f"POLY: {len(unique)} classes", "#ee8833"
        ax.text(0.5, 0.6, f"F#{feat}\nmean={mean_act[feat]:.3f}", ha="center",
                va="center", fontsize=7, color="white", transform=ax.transAxes,
                family="monospace")
        ax.text(0.5, 0.15, tag_label, ha="center", va="center", fontsize=6,
                color=tag_color, fontweight="bold", transform=ax.transAxes)
        ax.axis("off")

        # Column 1: decoder direction
        ax = axes[row, 1]
        ax.imshow(W_dec[feat].reshape(img_shape), cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.axis("off")

        # Columns 2..: top activating images
        for col, img_idx in enumerate(top_img):
            ax = axes[row, col + 2]
            ax.imshow(X[img_idx].reshape(img_shape), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{class_names[labels[img_idx]]} ({feat_acts[img_idx]:.2f})",
                         fontsize=5, color="#cccccc", pad=1)
            ax.axis("off")

    headers = ["", "Decoder\nWeight"] + [f"Top {i + 1}" for i in range(top_images)]
    for col, label in enumerate(headers):
        axes[0, col].set_title(label, fontsize=7, color="#aaaaaa", pad=4)

    fig.suptitle("Decoder Directions + Top-Activating Images", fontsize=11,
                 color="white", y=1.005)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    return fig
