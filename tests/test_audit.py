"""Null, injection and coverage tests on synthetic gauge records (pytest)."""
import numpy as np

from dispersion_audit import gauges, synthetic, theory

H = 0.5
X = np.array([0.0, 0.31, 0.74, 1.27, 1.93])        # unequal spacing


def test_dispersion_relation_roundtrip():
    k = np.array([0.1, 1.0, 5.0, 30.0])
    assert np.allclose(theory.wavenumber(theory.omega(k, H), H), k, rtol=1e-12)


def test_null_and_coverage():
    pulls = []
    for seed in range(40):
        rec, k = synthetic.regular_wave(X, H, period=1.4, refl=0.25, seed=seed)
        r = gauges.analyze_regular(rec, 100.0, X, H, n_boot=150, seed=seed)
        pulls.append(r["dk_rel"] / r["sigma_dk_rel"])
    pulls = np.array(pulls)
    assert abs(pulls.mean()) < 0.5, pulls.mean()
    assert 0.6 < pulls.std() < 1.5, pulls.std()


def test_injection_recovered():
    for dk in (0.01, -0.02):
        rec, k = synthetic.regular_wave(X, H, period=1.0, refl=0.3, dk_rel=dk, seed=3)
        r = gauges.analyze_regular(rec, 100.0, X, H, n_boot=100)
        assert abs(r["dk_rel"] - dk) < 4 * r["sigma_dk_rel"] + 1e-4, (r["dk_rel"], dk)
        assert abs(r["refl"] - 0.3) < 0.03


def test_depth_fit():
    res = []
    for T in (0.9, 1.3, 2.0, 3.0):
        rec, k = synthetic.regular_wave(X, 0.505, period=T, refl=0.2, seed=int(T * 10))
        res.append(gauges.analyze_regular(rec, 100.0, X, H, n_boot=100))
    h_fit, chi2 = gauges.fit_depth(res, H)
    assert abs(h_fit - 0.505) < 0.002, h_fit
