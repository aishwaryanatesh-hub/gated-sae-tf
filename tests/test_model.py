import keras
import numpy as np
import tensorflow as tf

from gated_sae import GatedSAE


def _build(input_dim=20, encoding_dim=64, **kw):
    model = GatedSAE(input_dim, encoding_dim, **kw)
    model(tf.zeros((2, input_dim)))  # build weights
    return model


def test_weight_shapes_and_tying_param_count():
    m = _build(20, 64)
    assert m.W_gate.shape == (20, 64)
    assert m.b_gate.shape == (64,)
    assert m.r_mag.shape == (64,)
    assert m.b_mag.shape == (64,)
    assert m.W_dec.shape == (64, 20)
    assert m.b_dec.shape == (20,)
    # Weight-tying: magnitude path adds only `encoding_dim` params (r_mag),
    # not a full (input_dim, encoding_dim) matrix.
    total = sum(int(np.prod(w.shape)) for w in m.weights)
    expected = 20 * 64 + 64 + 64 + 64 + 64 * 20 + 20
    assert total == expected


def test_encode_shapes_and_gate_is_binary():
    m = _build(20, 64)
    x = tf.random.normal((8, 20))
    f_tilde, pi_gate, relu_pi = m.encode(x)
    assert f_tilde.shape == (8, 64)
    assert pi_gate.shape == (8, 64)
    assert relu_pi.shape == (8, 64)
    # f_tilde must be zero wherever the gate is closed (pi_gate <= 0).
    gate_closed = (pi_gate.numpy() <= 0)
    assert np.all(f_tilde.numpy()[gate_closed] == 0.0)
    # relu(pi_gate) is non-negative
    assert np.all(relu_pi.numpy() >= 0.0)


def test_normalize_decoder_unit_norm_rows():
    m = _build(20, 64)
    m.W_dec.assign(tf.random.normal((64, 20)) * 5.0)
    m._normalize_decoder()
    norms = np.linalg.norm(m.W_dec.numpy(), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_loss_components_nonnegative_and_keys():
    m = _build(20, 64)
    m.compile(optimizer=keras.optimizers.Adam(1e-3, beta_1=0.0))
    x = tf.random.uniform((16, 20))
    logs = m.train_step(x)
    for k in ("loss", "L_reconstruct", "L_sparsity", "L_aux", "grad_norm"):
        assert k in logs
    assert float(logs["L_reconstruct"]) >= 0
    assert float(logs["L_sparsity"]) >= 0
    assert float(logs["L_aux"]) >= 0


def test_clip_norm_none_path():
    m = _build(20, 64, clip_norm=None)
    m.compile(optimizer=keras.optimizers.Adam(1e-3, beta_1=0.0))
    x = tf.random.uniform((16, 20))
    logs = m.train_step(x)
    assert "grad_norm" in logs
    assert float(logs["grad_norm"]) >= 0


def test_fit_reduces_reconstruction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 20)).astype("float32")
    m = _build(20, 64, lambda_sparse=1e-4)
    m.compile(optimizer=keras.optimizers.Adam(1e-2, beta_1=0.0))
    hist = m.fit(X, epochs=5, batch_size=64, verbose=0)
    recon = hist.history["L_reconstruct"]
    assert recon[-1] < recon[0]


def test_set_lambda():
    m = _build(20, 64, lambda_sparse=1e-3)
    assert m.lambda_sparse == 1e-3
    m.set_lambda(5e-4)
    assert m.lambda_sparse == 5e-4
    assert m.lambda_sparse_init == 1e-3  # untouched reference value


def test_get_config_from_config_roundtrip():
    m = _build(20, 64, lambda_sparse=2e-3, aux_weight=0.1, clip_norm=0.5)
    cfg = m.get_config()
    clone = GatedSAE.from_config(cfg)
    assert clone.input_dim == 20
    assert clone.encoding_dim == 64
    assert clone.lambda_sparse == 2e-3
    assert clone.aux_weight == 0.1
    assert clone.clip_norm == 0.5


def test_save_load_roundtrip(tmp_path):
    m = _build(20, 64)
    m.compile(optimizer=keras.optimizers.Adam(1e-3, beta_1=0.0))
    x = tf.random.uniform((8, 20))
    before = m(x).numpy()
    path = tmp_path / "model.keras"
    m.save(path)
    loaded = keras.models.load_model(path)
    after = loaded(x).numpy()
    assert np.allclose(before, after, atol=1e-5)
