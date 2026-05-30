"""gated-sae-tf: a TensorFlow/Keras Sparse Gated Autoencoder.

Public API
----------
- :class:`GatedSAE`            -- the model
- :class:`WarmupCosineDecay`   -- the learning-rate schedule
- :func:`sparsity_report`      -- L0 / alive-dead / top-k activation stats
- :func:`decoder_sharpness`    -- per-feature decoder kurtosis
- ``plot_decoder_gallery``, ``plot_feature_gallery`` -- visualization
  helpers (require the optional ``[viz]`` extra; imported lazily).
"""

from __future__ import annotations

from .analysis import decoder_sharpness, sparsity_report
from .model import GatedSAE
from .schedules import WarmupCosineDecay

__version__ = "0.1.1"

__all__ = [
    "GatedSAE",
    "WarmupCosineDecay",
    "sparsity_report",
    "decoder_sharpness",
    "plot_decoder_gallery",
    "plot_feature_gallery",
    "__version__",
]

# Lazily expose the matplotlib-dependent viz helpers so importing the core
# package never requires matplotlib.
_VIZ_EXPORTS = {"plot_decoder_gallery", "plot_feature_gallery"}


def __getattr__(name):
    if name in _VIZ_EXPORTS:
        from . import viz
        return getattr(viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
