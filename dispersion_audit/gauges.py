"""Wavenumber from synchronous wave-gauge records, with incident/reflected separation.

For a regular wave of frequency f, each gauge j at position x_j gives a complex
amplitude Z_j (least-squares fit of cos/sin at f and its harmonics over a steady
window). The linear model is

    Z_j = a_I exp(-i k x_j) + a_R exp(+i k x_j)            (Mansard & Funke 1980 form)

with k left FREE: for every trial k the amplitudes a_I, a_R follow from linear least
squares, and k minimizes the residual. The fitted k is then compared with the exact
linear k(f, h). Uncertainty comes from a parametric bootstrap with the per-gauge noise
estimated from the time-series residual.
"""
import numpy as np
from scipy.optimize import minimize_scalar

from . import theory


def fundamental(x, fs):
    """Dominant frequency of a record (zero-padded FFT peak, parabolic refinement)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    nfft = 16 * int(2 ** np.ceil(np.log2(n)))
    s = np.abs(np.fft.rfft(x * np.hanning(n), nfft))
    i = int(np.argmax(s[1:])) + 1
    a, b, c = np.log(s[i - 1:i + 2] + 1e-300)
    return (i + 0.5 * (a - c) / (a - 2 * b + c)) * fs / nfft


def refine_frequency(records, fs, f0):
    """Common frequency maximizing the summed harmonic fit (handles FFT bias on short windows)."""
    def cost(f):
        return -sum(np.sum(np.abs(complex_amplitude(r, fs, f)[0]) ** 2) for r in records.T)
    res = minimize_scalar(cost, bounds=(0.97 * f0, 1.03 * f0), method="bounded",
                          options={"xatol": 1e-7 * f0})
    return res.x


def complex_amplitude(x, fs, f, n_harm=3):
    """Complex amplitude at f (convention eta = Re[Z exp(i 2 pi f t)]) and residual noise std."""
    x = np.asarray(x, float)
    t = np.arange(len(x)) / fs
    cols = [np.ones_like(t)]
    for m in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * m * f * t), np.sin(2 * np.pi * m * f * t)]
    M = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(M, x, rcond=None)
    resid = x - M @ c
    Z = c[1] - 1j * c[2]
    sigma_Z = resid.std(ddof=M.shape[1]) * np.sqrt(2.0 / len(x))   # std of Re and Im parts
    return np.array([Z]), sigma_Z, c


def _solve(k, x, Z, reflect):
    A = np.exp(-1j * k * x)[:, None]
    if reflect:
        A = np.column_stack([A[:, 0], np.exp(1j * k * x)])
    a, *_ = np.linalg.lstsq(A, Z, rcond=None)
    r = Z - A @ a
    return float(np.sum(np.abs(r) ** 2)), a


def fit_k(x, Z, k_guess, reflect=True, span=0.5, n_grid=2001):
    """k minimizing the incident(+reflected) model residual, searched in k_guess*(1 ± span)."""
    x = np.asarray(x, float)
    ks = np.linspace(k_guess * (1 - span), k_guess * (1 + span), n_grid)
    S = np.array([_solve(k, x, Z, reflect)[0] for k in ks])
    i = int(np.argmin(S))
    lo, hi = ks[max(i - 2, 0)], ks[min(i + 2, n_grid - 1)]
    res = minimize_scalar(lambda k: _solve(k, x, Z, reflect)[0], bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10 * k_guess})
    k = res.x
    S0, a = _solve(k, x, Z, reflect)
    edge = i in (0, n_grid - 1)
    return k, a, S0, edge


def analyze_regular(records, fs, x, h, f=None, reflect=True, n_boot=300, seed=0, g=theory.G):
    """Audit one regular-wave run.

    records : array (n_samples, n_gauges), steady window only, same clock for all gauges
    x       : gauge positions along the wave direction [m] (any origin)
    h       : still-water depth [m]
    Returns a dict with measured k, its bootstrap std, exact k, relative error, reflection.
    """
    records = np.asarray(records, float)
    x = np.asarray(x, float)
    n_g = records.shape[1]
    if n_g < (3 if reflect else 2):
        raise ValueError(f"need >= {3 if reflect else 2} gauges ({'with' if reflect else 'without'} reflection)")
    if f is None:
        f = refine_frequency(records, fs, fundamental(records[:, 0], fs))
    Z, sig = [], []
    for j in range(n_g):
        z, s, _ = complex_amplitude(records[:, j], fs, f)
        Z.append(z[0])
        sig.append(s)
    Z, sig = np.array(Z), np.array(sig)
    w = 2 * np.pi * f
    k_ex = float(theory.wavenumber(w, h, g))
    spacing = np.min(np.diff(np.sort(x)))
    alias = spacing > np.pi / k_ex                       # adjacent gauges more than L/2 apart
    k, a, S0, edge = fit_k(x, Z, k_ex, reflect)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        Zb = Z + sig * (rng.standard_normal(n_g) + 1j * rng.standard_normal(n_g))
        boot.append(fit_k(x, Zb, k, reflect, span=0.2, n_grid=401)[0])
    sk = float(np.std(boot, ddof=1))
    out = {"f": f, "omega": w, "k_meas": k, "sigma_k": sk, "k_exact": k_ex,
           "dk_rel": k / k_ex - 1, "sigma_dk_rel": sk / k_ex, "kh": k_ex * h,
           "a_inc": abs(a[0]), "refl": (abs(a[1]) / abs(a[0])) if reflect else np.nan,
           "chi2_per_dof": S0 / max(np.mean(sig ** 2) * (2 * n_g - 2 * (2 if reflect else 1) - 1), 1e-300),
           "edge": edge, "spatial_alias_risk": bool(alias)}
    return out


def fit_depth(results, h0, g=theory.G):
    """Depth that best explains the measured k across several frequencies (weighted LSQ).
    With a single frequency depth and a dispersion error are degenerate: returns None."""
    if len(results) < 2:
        return None
    w = np.array([r["omega"] for r in results])
    k = np.array([r["k_meas"] for r in results])
    s = np.array([max(r["sigma_k"], 1e-12) for r in results])
    cost = lambda hh: np.sum(((k - theory.wavenumber(w, hh, g)) / s) ** 2)
    res = minimize_scalar(cost, bounds=(0.5 * h0, 1.5 * h0), method="bounded", options={"xatol": 1e-7})
    return res.x, res.fun


def delta_lambda(results, h, g=theory.G):
    """Long-wave summary (only meaningful for kh <~ 1): residual y_meas - y_exact fitted as dLambda k^2."""
    w = np.array([r["omega"] for r in results])
    k = np.array([r["k_meas"] for r in results])
    s = np.array([r["sigma_k"] for r in results])
    dy = theory.y_of(w, k, h, g) - theory.y_of(w, theory.wavenumber(w, h, g), h, g)
    sy = 2 * (theory.y_of(w, k, h, g) + 1) * s / k               # dy/dk * sigma_k
    W = 1 / sy ** 2
    A = k ** 2
    dl = np.sum(W * A * dy) / np.sum(W * A ** 2)
    return dl, 1 / np.sqrt(np.sum(W * A ** 2))
