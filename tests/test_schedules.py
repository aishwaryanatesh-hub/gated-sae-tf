import numpy as np

from gated_sae import WarmupCosineDecay


def test_warmup_and_decay_shape():
    peak, warmup, total = 1e-3, 100, 1000
    sched = WarmupCosineDecay(peak, warmup, total)

    assert float(sched(0)) < 1e-5  # ~0 at the very start
    assert np.isclose(float(sched(warmup)), peak, rtol=1e-4)  # peak at warmup end
    assert float(sched(warmup // 2)) < peak  # still ramping mid-warmup
    assert float(sched(total)) < 1e-4  # decayed back to ~0 at the end


def test_warmup_is_monotonic_increasing():
    sched = WarmupCosineDecay(1e-3, 100, 1000)
    lrs = [float(sched(s)) for s in range(0, 100, 10)]
    assert all(b >= a for a, b in zip(lrs, lrs[1:]))


def test_get_config_roundtrip():
    sched = WarmupCosineDecay(1e-3, 100, 1000)
    cfg = sched.get_config()
    clone = WarmupCosineDecay.from_config(cfg)
    for s in (0, 50, 100, 500, 1000):
        assert np.isclose(float(sched(s)), float(clone(s)), atol=1e-8)
