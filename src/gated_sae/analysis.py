"""Analysis utilities: sparsity statistics and decoder sharpness."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sparsity_report(model, X, batch_size: int = 512, k: int = 20) -> dict:
    """Summarize how sparse and how alive a trained model's codes are.

    Encodes ``X`` in batches and reports L0 (number of active features per
    example), the alive/dead feature split, and how concentrated activation is
    in the top-``k`` features.

    Parameters
    ----------
    model:
        A trained :class:`~gated_sae.GatedSAE` (anything with an ``encode``
        method returning ``(f_tilde, ...)``).
    X:
        Input array of shape ``(n_samples, input_dim)``.
    batch_size:
        Encoding batch size (keeps memory bounded for large overcomplete SAEs).
    k:
        Number of top features for the activation-share statistic.

    Returns
    -------
    dict with keys::

        l0_mean, l0_median   # active features per example
        alive, dead          # feature counts (alive = ever fires on X)
        n_features           # == encoding_dim
        alive_frac           # alive / n_features
        topk, topk_share     # k, and fraction of total activation in top-k
    """
    X = np.asarray(X, dtype="float32")
    n = X.shape[0]
    n_features = int(model.encoding_dim)

    l0 = np.empty(n, dtype="int64")
    act_sum = np.zeros(n_features, dtype="float64")
    ever_active = np.zeros(n_features, dtype=bool)

    for start in range(0, n, batch_size):
        batch = X[start:start + batch_size]
        codes, _, _ = model.encode(batch)
        codes = np.asarray(codes)
        l0[start:start + batch_size] = (codes > 0).sum(axis=1)
        act_sum += codes.sum(axis=0)
        ever_active |= codes.max(axis=0) > 0

    mean_act = act_sum / n
    alive = int(ever_active.sum())
    sorted_act = np.sort(mean_act)[::-1]
    total = sorted_act.sum()
    if total > 0:
        topk_share = float(np.cumsum(sorted_act)[min(k, n_features) - 1] / total)
    else:
        topk_share = 0.0

    return {
        "l0_mean": float(l0.mean()),
        "l0_median": float(np.median(l0)),
        "alive": alive,
        "dead": n_features - alive,
        "n_features": n_features,
        "alive_frac": alive / n_features,
        "topk": int(min(k, n_features)),
        "topk_share": topk_share,
    }


def decoder_sharpness(model) -> Tuple[np.ndarray, float]:
    """Per-feature kurtosis of the decoder directions.

    Higher excess (Fisher) kurtosis means a decoder direction concentrates its
    weight on a few pixels — i.e. a sharper, more localized spatial feature.
    This matches ``scipy.stats.kurtosis`` (Fisher, biased) but avoids the scipy
    dependency.

    Returns
    -------
    per_feature:
        Array of shape ``(encoding_dim,)`` with the kurtosis of each decoder row.
    mean:
        Mean kurtosis across features.
    """
    W_dec = np.asarray(model.W_dec).astype("float64")
    z = W_dec - W_dec.mean(axis=1, keepdims=True)
    m2 = np.mean(z ** 2, axis=1)
    m4 = np.mean(z ** 4, axis=1)
    # Guard against constant rows (m2 == 0) to avoid divide-by-zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_feature = np.where(m2 > 0, m4 / (m2 ** 2) - 3.0, 0.0)
    return per_feature, float(per_feature.mean())
