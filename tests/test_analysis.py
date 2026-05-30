import numpy as np
import tensorflow as tf

from gated_sae import GatedSAE, decoder_sharpness, sparsity_report


class _StubModel:
    """Minimal model exposing the attributes the analysis functions need."""

    def __init__(self, codes, W_dec):
        self._codes = np.asarray(codes, dtype="float32")
        self.W_dec = np.asarray(W_dec, dtype="float32")
        self.encoding_dim = self._codes.shape[1]

    def encode(self, X):
        # Return precomputed codes for the requested slice size (the report
        # batches X, so we just hand back rows we were seeded with).
        n = len(X)
        return self._codes[:n], None, None


def test_sparsity_report_keys_and_values():
    # 5 samples, 4 features. Feature 0 always on, feature 3 always off (dead).
    codes = np.array([
        [2.0, 0.0, 1.0, 0.0],
        [3.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [4.0, 2.0, 0.0, 0.0],
        [2.0, 0.0, 1.0, 0.0],
    ])
    model = _StubModel(codes, np.random.randn(4, 6))
    # Encode whole array in one batch for the stub.
    rep = sparsity_report(model, np.zeros((5, 6)), batch_size=5, k=2)

    assert set(rep) >= {"l0_mean", "l0_median", "alive", "dead", "n_features",
                        "alive_frac", "topk", "topk_share"}
    assert rep["n_features"] == 4
    assert rep["dead"] == 1          # feature 3 never fires
    assert rep["alive"] == 3
    assert 0.0 <= rep["topk_share"] <= 1.0
    assert rep["l0_mean"] > 0


def test_decoder_sharpness_shape_and_ordering():
    # Row 0: a sharp spike (high kurtosis). Row 1: roughly flat (low kurtosis).
    n = 100
    spike = np.zeros(n); spike[0] = 10.0
    flat = np.linspace(-1, 1, n)
    W_dec = np.stack([spike, flat])
    model = _StubModel(np.zeros((2, 2)), W_dec)

    per_feature, mean = decoder_sharpness(model)
    assert per_feature.shape == (2,)
    assert np.isclose(mean, per_feature.mean())
    # The spike is far more leptokurtic than the flat ramp.
    assert per_feature[0] > per_feature[1]


def test_decoder_sharpness_on_real_model():
    m = GatedSAE(20, 32)
    m(tf.zeros((2, 20)))
    per_feature, mean = decoder_sharpness(m)
    assert per_feature.shape == (32,)
    assert np.isfinite(mean)
