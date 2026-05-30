"""Gated Sparse Autoencoder (Gated SAE) implemented in TensorFlow / Keras 3.

The architecture follows Rajamanoharan et al. (2024), "Improving Dictionary
Learning with Gated Sparse Autoencoders" (arXiv:2404.16014).
"""

from __future__ import annotations

from typing import Optional

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="gated_sae")
class GatedSAE(keras.Model):
    r"""Gated Sparse Autoencoder.

    A dual-path encoder separates *which* features fire (the gate path) from
    *how much* they fire (the magnitude path), with the magnitude weights tied
    to the gate weights through a per-feature rescaling ``exp(r_mag)``:

    .. code-block:: text

        x_c        = x - b_dec
        pi_gate    = W_gate @ x_c + b_gate
        f_gate     = 1[pi_gate > 0]                       # hard gate
        W_mag      = exp(r_mag) * W_gate                   # weight-tying
        f_mag      = relu(W_mag @ x_c + b_mag)
        f_tilde    = f_gate * f_mag                        # sparse code
        x_hat      = W_dec @ f_tilde + b_dec               # reconstruction

    The decoder rows (dictionary directions) are L2-normalized after every
    optimizer step. Training minimizes::

        L = L_reconstruct + lambda_sparse * L_sparsity + aux_weight * L_aux

    where ``L_sparsity = sum(relu(pi_gate))`` encourages few active features and
    ``L_aux`` reconstructs the input from ``relu(pi_gate)`` through a frozen
    (``stop_gradient``) decoder, which keeps the gate path well-conditioned.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input vectors (e.g. ``784`` for flattened MNIST).
    encoding_dim:
        Number of dictionary features (the SAE is typically overcomplete, so
        this is several times ``input_dim``).
    lambda_sparse:
        Weight on the L1-style sparsity penalty.
    aux_weight:
        Weight on the auxiliary (frozen-decoder) reconstruction loss.
    clip_norm:
        Global gradient-norm clip threshold applied each step. Pass ``None`` to
        disable clipping (recovers the un-clipped baseline).

    Notes
    -----
    Train with ``beta_1=0`` in Adam, per the reference report. Initialize
    ``b_dec`` to the training-set mean (``model.b_dec.assign(train_mean)``)
    before calling ``fit`` for best results.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        lambda_sparse: float = 1e-3,
        aux_weight: float = 1e-2,
        clip_norm: Optional[float] = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.lambda_sparse = lambda_sparse
        self.lambda_sparse_init = lambda_sparse  # reference value for annealing
        self.aux_weight = aux_weight
        self.clip_norm = clip_norm

        self.W_gate = self.add_weight(
            name="W_gate", shape=(input_dim, encoding_dim),
            initializer="glorot_uniform", trainable=True)
        self.b_gate = self.add_weight(
            name="b_gate", shape=(encoding_dim,),
            initializer="zeros", trainable=True)
        self.r_mag = self.add_weight(
            name="r_mag", shape=(encoding_dim,),
            initializer="zeros", trainable=True)
        self.b_mag = self.add_weight(
            name="b_mag", shape=(encoding_dim,),
            initializer="zeros", trainable=True)
        self.W_dec = self.add_weight(
            name="W_dec", shape=(encoding_dim, input_dim),
            initializer="glorot_uniform", trainable=True)
        self.b_dec = self.add_weight(
            name="b_dec", shape=(input_dim,),
            initializer="zeros", trainable=True)

    # ── public knobs ────────────────────────────────────────────────────────
    def set_lambda(self, new_lambda: float) -> None:
        """Update ``lambda_sparse`` (useful for sparsity annealing schedules)."""
        self.lambda_sparse = new_lambda

    # ── internals ───────────────────────────────────────────────────────────
    def _compute_W_mag(self) -> tf.Tensor:
        """Magnitude-path weights, tied to the gate path via ``exp(r_mag)``."""
        return self.W_gate * tf.exp(self.r_mag)[tf.newaxis, :]

    def _normalize_decoder(self) -> None:
        """L2-normalize each decoder row (dictionary direction) in place."""
        self.W_dec.assign(tf.math.l2_normalize(self.W_dec, axis=1))

    # ── forward ─────────────────────────────────────────────────────────────
    def encode(self, x: tf.Tensor):
        """Encode inputs to sparse codes.

        Returns
        -------
        f_tilde:
            The sparse code ``f_gate * f_mag`` of shape ``(batch, encoding_dim)``.
        pi_gate:
            Pre-activation gate logits.
        relu_pi_gate:
            ``relu(pi_gate)`` — used by the sparsity and auxiliary losses.
        """
        x_centered = x - self.b_dec
        pi_gate = tf.matmul(x_centered, self.W_gate) + self.b_gate
        relu_pi_gate = tf.nn.relu(pi_gate)
        f_gate = tf.cast(pi_gate > 0, x.dtype)
        W_mag = self._compute_W_mag()
        f_mag = tf.nn.relu(tf.matmul(x_centered, W_mag) + self.b_mag)
        f_tilde = f_gate * f_mag
        return f_tilde, pi_gate, relu_pi_gate

    def decode(self, f: tf.Tensor, stop_grad: bool = False) -> tf.Tensor:
        """Decode a code back to input space. ``stop_grad`` freezes the decoder."""
        W = tf.stop_gradient(self.W_dec) if stop_grad else self.W_dec
        b = tf.stop_gradient(self.b_dec) if stop_grad else self.b_dec
        return tf.matmul(f, W) + b

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        f_tilde, _, _ = self.encode(x)
        return self.decode(f_tilde)

    # ── losses (shared by train/test) ────────────────────────────────────────
    def _compute_losses(self, x):
        f_tilde, pi_gate, relu_pi_gate = self.encode(x)
        x_hat = self.decode(f_tilde, stop_grad=False)
        L_reconstruct = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=-1))
        L_sparsity = tf.reduce_mean(tf.reduce_sum(relu_pi_gate, axis=-1))
        x_hat_frozen = self.decode(relu_pi_gate, stop_grad=True)
        L_aux = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat_frozen), axis=-1))
        total_loss = (
            L_reconstruct
            + self.lambda_sparse * L_sparsity
            + self.aux_weight * L_aux
        )
        return total_loss, L_reconstruct, L_sparsity, L_aux

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data

        with tf.GradientTape() as tape:
            total_loss, L_reconstruct, L_sparsity, L_aux = self._compute_losses(x)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        if self.clip_norm is not None:
            gradients, global_norm = tf.clip_by_global_norm(gradients, self.clip_norm)
        else:
            global_norm = tf.linalg.global_norm(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self._normalize_decoder()

        return {
            "loss": total_loss,
            "L_reconstruct": L_reconstruct,
            "L_sparsity": L_sparsity,
            "L_aux": L_aux,
            "grad_norm": global_norm,
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        total_loss, L_reconstruct, L_sparsity, L_aux = self._compute_losses(x)
        return {
            "loss": total_loss,
            "L_reconstruct": L_reconstruct,
            "L_sparsity": L_sparsity,
            "L_aux": L_aux,
        }

    # ── serialization ────────────────────────────────────────────────────────
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "lambda_sparse": self.lambda_sparse,
            "aux_weight": self.aux_weight,
            "clip_norm": self.clip_norm,
        })
        return config
