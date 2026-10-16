"""UiO flume dissipation tests (probes at 1, 5, 8, 12 m; 250 Hz): k(f) vs exact linear theory,
and a comparison of two explanations for any excess: probe-position scale (dk ~ k) versus a
uniform extra slowness / timing (dk ~ omega).

Data (CC BY 4.0): Lande, Ø. (2026), doi:10.18710/FZN3AL, waveprobes/tank_dissipation_tests/.
  python examples/uio_dissipation.py test460040_*.csv test460041_*.csv test460042_*.csv
"""
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dispersion_audit import broadband  # noqa: E402

H, G = 0.6, 9.819
X = np.array([5.0, 8.0, 12.0])             # probes 2-4 (probe 1 at 1 m excluded: near field)


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


def main(files):
    res = []
    for fn in files:
        t, d = load(fn)
        fs = round(1 / np.median(np.diff(t)))
        nw = (t[-1] - 21, t[-1] - 1)                       # still water at the end of the record
        e = d - np.median(d[int(nw[0] * fs):int(nw[1] * fs)], 0)
        win = broadband.auto_windows(e[:, 1:], fs, 20.0, search=(5, nw[0]))
        rows = broadband.analyze_transient(e[:, 1:], fs, X, H, win, noise_window=nw, band=(0.9, 1.6), g=G)
        m, s, chi2, dof = broadband.summarize(rows)
        rr = rows[::2]
        k = np.array([r["k_exact"] for r in rr])
        w = 2 * np.pi * np.array([r["f"] for r in rr])
        dk = np.array([r["k_meas"] - r["k_exact"] for r in rr])
        sk = np.array([r["sigma_k"] for r in rr])
        fits = {}
        for name, col in (("position (dk ~ k)", k), ("slowness/timing (dk ~ omega)", w)):
            a = np.sum(col * dk / sk ** 2) / np.sum(col ** 2 / sk ** 2)
            fits[name] = (a, float(np.sum(((dk - a * col) / sk) ** 2)))
        res.append(m)
        print(f"{os.path.basename(fn)}: fs {fs} Hz  dk/k = {m:+.2e} ± {s:.1e} (chi2 {chi2:.1f}/{dof})")
        for name, (a, c2) in fits.items():
            print(f"    {name:30s} coefficient {a:.5f}   chi2 {c2:.1f} / {len(dk) - 1}")
    if len(res) > 1:
        print(f"repeats: mean {np.mean(res):+.2e}, scatter {np.std(res, ddof=1):.1e}")


if __name__ == "__main__":
    main(sys.argv[1:])
