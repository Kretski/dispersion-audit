"""Wavenumber k(f) over a frequency band from a transient wave group (e.g. focused NewWave).

All gauges are recorded on one clock. A common time window (chosen to contain the group
at every gauge and to end before reflections from the far end arrive) is Fourier
transformed; for every frequency the phases follow the incident-wave model

    phi_j(f) = phi_0(f) - k(f) x_j        (convention X ~ A exp(i(phi_0 - k x)))

k(f) is the slope of phase against position. The 2*pi ambiguity is resolved
hierarchically: the closest gauge pair fixes the branch using the exact linear k as a
guide (wide tolerance), then the full array refines it. Per-frequency phase noise is
estimated from a still-water segment recorded before the waves arrive.

Limitations: no reflection separation here (window must exclude reflections); bound
(second-order) harmonics contaminate frequencies near 2 f_peak at large steepness;
gauges close to the wavemaker may contain evanescent modes.
"""
import numpy as np

from . import theory


def _spectra(seg, fs):
    n = seg.shape[0]
    w = np.hanning(n)
    X = np.fft.rfft((seg - seg.mean(0)) * w[:, None], axis=0)
    f = np.fft.rfftfreq(n, 1 / fs)
    return f, X, np.sqrt(np.sum(w ** 2))


def auto_windows(records, fs, length, search=None):
    """Per-gauge window of `length` s centred on the envelope maximum (Hilbert transform)."""
    from scipy.signal import hilbert
    rec = np.asarray(records, float)
    lo, hi = (0, rec.shape[0]) if search is None else (int(search[0] * fs), int(search[1] * fs))
    out = []
    for j in range(rec.shape[1]):
        e = np.abs(hilbert(rec[lo:hi, j] - rec[lo:hi, j].mean()))
        c = (lo + int(np.argmax(e))) / fs
        out.append((c - length / 2, c + length / 2))
    return out


def analyze_transient(records, fs, x, h, window, noise_window, band=None, min_snr=10.0,
                      g=theory.G):
    """Return per-frequency measured k, its std, exact k and relative error.

    records      : (n_samples, n_gauges), metres, common clock
    x            : gauge positions [m]
    window       : (t0, t1) seconds for all gauges, or a list of per-gauge (t0, t1) of equal
                   length (phases are referred back to the common clock); each window must
                   contain the group at that gauge and end before reflections arrive
    noise_window : (t0, t1) seconds of still water, used for the phase-noise estimate
    band         : (fmin, fmax) Hz; default: where the amplitude at every gauge > min_snr x noise
    """
    records = np.asarray(records, float)
    x = np.asarray(x, float)
    order = np.argsort(x)
    x, records = x[order], records[:, order]
    if np.ndim(window[0]) == 0:
        window = [window] * len(x)
    else:
        window = [window[o] for o in order]
    starts = [int(round(w0 * fs)) for w0, _ in window]
    n = int(round((window[0][1] - window[0][0]) * fs))
    n0 = int(noise_window[0] * fs)
    if int(noise_window[1] * fs) - n0 < n:
        raise ValueError("noise window must be at least as long as the analysis window")
    seg = np.column_stack([records[s0:s0 + n, j] for j, s0 in enumerate(starts)])
    if seg.shape[0] != n:
        raise ValueError("a window runs past the end of the record")
    f, X, _ = _spectra(seg, fs)
    X = X * np.exp(-2j * np.pi * f[:, None] * (np.array(starts)[None, :] / fs))   # common clock
    if records[n0:n0 + n].shape[0] != n:
        raise ValueError("noise window runs past the end of the record")
    _, N, _ = _spectra(records[n0:n0 + n], fs)
    amp = np.abs(X)
    namp = np.abs(N)
    # smooth the noise floor over a few bins (single-segment periodogram is noisy)
    ker = np.ones(9) / 9
    nfl = np.column_stack([np.convolve(namp[:, j], ker, mode="same") for j in range(namp.shape[1])])
    snr = amp / np.maximum(nfl, 1e-30)
    ok = np.all(snr > min_snr, axis=1)
    if band is not None:
        ok &= (f >= band[0]) & (f <= band[1])
    rows = []
    for i in np.where(ok)[0]:
        if f[i] <= 0:
            continue
        w = 2 * np.pi * f[i]
        k_ex = float(theory.wavenumber(w, h, g))
        ph = np.angle(X[i])
        sig_ph = 1.0 / snr[i]                       # phase std ~ noise/amplitude (per Re/Im)
        # 1) branch from the closest pair, guided by exact k (tolerance: half a branch)
        dxs = np.diff(x)
        j = int(np.argmin(dxs))
        dph = -(ph[j + 1] - ph[j])                    # = k dx (mod 2 pi)
        m = np.round((k_ex * dxs[j] - dph) / (2 * np.pi))
        k_est = (dph + 2 * np.pi * m) / dxs[j]
        # 2) unwrap every gauge relative to the first with k_est, then weighted LSQ slope
        pred = ph[0] - k_est * (x - x[0])
        unw = ph + 2 * np.pi * np.round((pred - ph) / (2 * np.pi))
        W = 1 / sig_ph ** 2
        xm = np.sum(W * x) / np.sum(W)
        slope = np.sum(W * (x - xm) * (unw - np.sum(W * unw) / np.sum(W))) / np.sum(W * (x - xm) ** 2)
        k = -slope
        sk = 1 / np.sqrt(np.sum(W * (x - xm) ** 2))
        resid = unw - (np.sum(W * unw) / np.sum(W) + slope * (x - xm))
        chi2 = float(np.sum(W * resid ** 2))
        branch_margin = abs(k_ex * dxs[j] - dph - 2 * np.pi * m) / np.pi   # < 1 required; near 1 = risky
        rows.append({"f": f[i], "kh": k_ex * h, "k_meas": k, "sigma_k": sk, "k_exact": k_ex,
                     "dk_rel": k / k_ex - 1, "sigma_dk_rel": sk / k_ex, "snr_min": float(snr[i].min()),
                     "chi2": chi2, "dof": len(x) - 2, "branch_margin": float(branch_margin)})
    return rows


def summarize(rows, step=2):
    """Weighted mean of dk/k over the band, using every `step`-th frequency bin (adjacent
    Hann-window bins are correlated). Returns mean, std (inflated by sqrt(chi2/dof) when
    that exceeds 1, PDG-style), chi2 and dof of the constancy fit."""
    rows = rows[::step]
    d = np.array([r["dk_rel"] for r in rows])
    s = np.array([r["sigma_dk_rel"] for r in rows])
    W = 1 / s ** 2
    m = np.sum(W * d) / np.sum(W)
    chi2 = float(np.sum(W * (d - m) ** 2))
    dof = max(len(d) - 1, 1)
    sm = 1 / np.sqrt(np.sum(W)) * max(1.0, np.sqrt(chi2 / dof))
    return m, sm, chi2, dof
