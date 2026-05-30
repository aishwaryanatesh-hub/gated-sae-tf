"""Learning-rate schedules for training Gated SAEs."""

from __future__ import annotations

import math

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="gated_sae")
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to zero.

    Linearly ramps the learning rate from 0 to ``peak_lr`` over the first
    ``warmup_steps`` steps, then cosine-decays it back to ~0 by ``total_steps``.
    A short warmup is important when training with ``beta_1=0`` (no momentum),
    where early updates are otherwise noisy.

    Parameters
    ----------
    peak_lr:
        Maximum learning rate, reached at the end of warmup.
    warmup_steps:
        Number of optimizer steps spent ramping up.
    total_steps:
        Total number of optimizer steps over the whole training run.
    """

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.peak_lr = tf.cast(peak_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Linear warmup
        warmup_lr = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        # Cosine decay
        progress = (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0)
        cosine_lr = self.peak_lr * 0.5 * (
            1.0 + tf.cos(math.pi * tf.minimum(progress, 1.0)))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": float(self.peak_lr),
            "warmup_steps": float(self.warmup_steps),
            "total_steps": float(self.total_steps),
        }
