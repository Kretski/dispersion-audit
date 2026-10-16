"""Focused-group (broadband) k(f): null and injection on synthetic records."""
import numpy as np

from dispersion_audit import broadband, theory

FS, H = 125.0, 0.6
X = np.array([7.407, 8.15, 8.845])


def group(dk=0.0, noise=3e-4, amp=0.007, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 60, 1 / FS)
    freqs = np.linspace(0.4, 1.8, 120)
    S = np.exp(-0.5 * ((freqs - 0.9) / 0.2) ** 2)
    a = amp * S / S.sum()
    rec = np.zeros((t.size, X.size))
    for fi, ai in zip(freqs, a):
        w = 2 * np.pi * fi
        k = theory.wavenumber(w, H) * (1 + dk)
        rec += ai * np.cos(k * (X[None, :] - 8.1) - w * (t[:, None] - 28.0))
    rec += noise * rng.standard_normal(rec.shape)
    return np.vstack([noise * rng.standard_normal((int(30 * FS), X.size)), rec])


def run(dk, seed):
    r = group(dk=dk, seed=seed)
    win = broadband.auto_windows(r, FS, 24.0, search=(30, 90))
    rows = broadband.analyze_transient(r, FS, X, H, win, noise_window=(0, 30), band=(0.5, 1.5))
    return broadband.summarize(rows)


def test_null_unbiased_and_conservative():
    res = np.array([run(0.0, s)[:2] for s in range(8)])
    assert abs(res[:, 0].mean()) < 2 * res[:, 1].mean() / np.sqrt(len(res)) + 3e-4
    assert res[:, 1].mean() > 0.7 * res[:, 0].std()


def test_injection():
    m, s, *_ = run(0.01, 1)
    assert abs(m - 0.01) < 4 * s
