"""UiO flume, shoal case C02A07F01 (test430013, 7 mm): phase propagation over variable depth.

Probes 2-4 (7.407 / 8.150 / 8.845 m) stand over the shoal top (h = 0.19 m; probe 2 at the
edge of the up-ramp, h = 0.196 m). Predicted phase between probes is the WKB integral
of k(omega, h(x)) dx with exact linear dispersion; residuals are fitted with (a) the top
depth, (b) a position offset per probe (dk ~ k) and (c) a timing offset per probe (~ omega).

Data (CC BY 4.0): Lande, Ø. (2026), doi:10.18710/FZN3AL.
  python examples/uio_shoal.py test430013_bud_ampl7mm_focused_x8.15m.csv
"""
import datetime as dt
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dispersion_audit import broadband, theory  # noqa: E402

G = 9.819
X = np.array([1.0, 7.407, 8.15, 8.845])
REF = 2                                    # probe 3 (8.15 m)


def depth(x, htop=0.19):
    """Shoal geometry from the dataset schematic."""
    x = np.asarray(x, float)
    h = np.full_like(x, 0.6)
    a, b, c, d = 5.832, 7.432, 9.032, 9.832
    up = (x > a) & (x <= b)
    h[up] = 0.6 - (0.6 - htop) * (x[up] - a) / (b - a)
    h[(x > b) & (x <= c)] = htop
    dn = (x > c) & (x < d)
    h[dn] = htop + (0.6 - htop) * (x[dn] - c) / (d - c)
    return h


def phase_between(w, xa, xb, htop=0.19, n=400):
    """WKB phase -integral of k dx from xa to xb (sign follows the incident-wave convention)."""
    xs = np.linspace(min(xa, xb), max(xa, xb), n)
    I = np.trapezoid(theory.wavenumber(w, depth(xs, htop), g=G), xs)
    return -np.sign(xb - xa) * I


def load(fn):
    t, d = [], []
    for line in open(fn):
        p = line.strip().split(",")
        try:
            t.append(dt.datetime.strptime(p[0], "%m/%d/%Y %H:%M:%S.%f").timestamp())
            d.append([float(v) for v in p[1:5]])
        except (ValueError, IndexError):
            pass
    t = np.array(t)
    return t - t[0], np.array(d)


def main(fn):
    t, d = load(fn)
    fs = round(1 / np.median(np.diff(t)))
    e = d - d[:int(5 * fs)].mean(0)
    WL = 16.0
    win = broadband.auto_windows(e, fs, WL, search=(5, t[-1] - 5))
    n = int(WL * fs)
    st = [int(round(w0 * fs)) for w0, _ in win]
    seg = np.column_stack([e[s0:s0 + n, j] for j, s0 in enumerate(st)])
    f = np.fft.rfftfreq(n, 1 / fs)
    F = np.fft.rfft((seg - seg.mean(0)) * np.hanning(n)[:, None], axis=0)
    F *= np.exp(-2j * np.pi * f[:, None] * np.array(st)[None, :] / fs)
    nz = e[int(43 * fs):int(43 * fs) + n]
    N = np.abs(np.fft.rfft((nz - nz.mean(0)) * np.hanning(n)[:, None], axis=0))
    N = np.column_stack([np.convolve(N[:, j], np.ones(9) / 9, "same") for j in range(4)])
    snr = np.abs(F) / np.maximum(N, 1e-30)
    sel = [i for i in range(len(f)) if 0.75 <= f[i] <= 1.6 and np.all(snr[i, 1:] > 10)][::2]
    rows = []
    for i in sel:
        w = 2 * np.pi * f[i]
        meas = {j: np.angle(F[i, j] / F[i, REF]) for j in (1, 3)}
        sig = {j: np.hypot(1 / snr[i, j], 1 / snr[i, REF]) for j in (1, 3)}
        rows.append((f[i], w, meas, sig))

    def resid(htop):
        out = []
        for fi, w, meas, sig in rows:
            for j in (1, 3):
                r = meas[j] - phase_between(w, X[REF], X[j], htop)
                out.append(((r + np.pi) % (2 * np.pi) - np.pi, sig[j], j, w))
        return out

    print(f"{os.path.basename(fn)}: fs {fs} Hz, band {f[sel[0]]:.2f}-{f[sel[-1]]:.2f} Hz, "
          f"kh on top {theory.wavenumber(2*np.pi*f[sel[0]], 0.19, g=G)*0.19:.2f}-"
          f"{theory.wavenumber(2*np.pi*f[sel[-1]], 0.19, g=G)*0.19:.2f}")
    chi2 = lambda h: sum((r / s) ** 2 for r, s, _, _ in resid(h))
    best = minimize_scalar(chi2, bounds=(0.17, 0.21), method="bounded")
    npts = 2 * len(rows)
    print(f"  documented top depth 190 mm: chi2 {chi2(0.19):.1f} for {npts} points")
    print(f"  best-fit top depth {best.x * 1e3:.1f} mm: chi2 {best.fun:.1f} for {npts - 1} dof")
    rr = resid(0.19)
    for j in (1, 3):
        r = np.array([a for a, _, jj, _ in rr if jj == j])
        s = np.array([b for _, b, jj, _ in rr if jj == j])
        w = np.array([c for _, _, jj, c in rr if jj == j])
        k = theory.wavenumber(w, depth(X[j]), g=G)
        for name, col, unit in (("position offset", -k, "mm"), ("timing offset", -w, "ms")):
            a = np.sum(col * r / s ** 2) / np.sum(col ** 2 / s ** 2)
            sa = 1 / np.sqrt(np.sum(col ** 2 / s ** 2))
            c2 = np.sum(((r - a * col) / s) ** 2)
            print(f"  probe {j + 1} ({X[j]} m): {name:16s} {a * 1e3:+6.1f} ± {sa * 1e3:.1f} {unit}  "
                  f"chi2 {c2:.1f}/{len(r) - 1}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test430013_bud_ampl7mm_focused_x8.15m.csv")
