"""Synthetic gauge records for null and injection tests."""
import numpy as np

from . import theory


def regular_wave(x, h, period, amp=0.01, refl=0.0, refl_phase=0.7, dk_rel=0.0, noise=5e-4,
                 fs=100.0, duration=60.0, harm2=0.03, seed=0, g=theory.G):
    """Incident + reflected regular wave at gauges x; true k = k_exact(1 + dk_rel)."""
    rng = np.random.default_rng(seed)
    w = 2 * np.pi / period
    k = theory.wavenumber(w, h, g) * (1 + dk_rel)
    t = np.arange(0, duration, 1 / fs)
    x = np.asarray(x, float)
    rec = np.empty((t.size, x.size))
    for j, xj in enumerate(x):
        inc = amp * np.cos(w * t - k * xj)
        ref = refl * amp * np.cos(w * t + k * xj + refl_phase)
        bound = harm2 * amp * np.cos(2 * (w * t - k * xj))          # simple bound 2nd harmonic
        rec[:, j] = inc + ref + bound + noise * rng.standard_normal(t.size)
    return rec, k
