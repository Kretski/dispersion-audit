"""First real-data audit: UiO wave flume, flat bed, focused group A = 7 mm (case C01A07F01).

Data (CC BY 4.0): Lande, Ø. (2026), doi:10.18710/FZN3AL, file
waveprobes/test370060_bud_ampl7mm_focused_x8.15m.csv  (download it next to this script).

  python examples/uio_flume_A07.py path/to/test370060_bud_ampl7mm_focused_x8.15m.csv
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dispersion_audit import broadband, theory  # noqa: E402

FS, H = 125.0, 0.6
X = np.array([1.0, 7.407, 8.15, 8.845])          # documented probe positions


def load(fn):
    rows = []
    for line in open(fn):
        p = line.strip().split(",")
        try:
            rows.append([float(v) for v in p[1:5]])
        except ValueError:
            rows.append([np.nan] * 4)
    d = np.array(rows)
    return d - np.nanmean(d[:600], 0)              # elevation relative to pre-run still water


def main(fn):
    e = load(fn)
    print("k(f) from probes 2-4 (x = 7.407, 8.150, 8.845 m), per-gauge 20 s windows, "
          "noise from 42-64 s")
    r, x = e[:, 1:], X[1:]
    win = broadband.auto_windows(r, FS, 20.0, search=(8, 45))
    rows = broadband.analyze_transient(r, FS, x, H, win, noise_window=(42, 64))
    for q in rows[::2]:
        print(f"  f {q['f']:.3f} Hz  kh {q['kh']:.2f}  dk/k {q['dk_rel']:+.2e} ± {q['sigma_dk_rel']:.1e}")
    m, s, chi2, dof = broadband.summarize(rows)
    print(f"band mean dk/k = {m:+.2e} ± {s:.1e}  (chi2 {chi2:.1f} / {dof})")

    print("\neffective probe positions relative to probe 3, assuming exact linear k "
          "(constant in f -> position offset; proportional to c_p -> channel time skew)")
    n = int(20 * FS)
    st = [int(round(w[0] * FS)) for w in broadband.auto_windows(e, FS, 20.0, search=(8, 45))]
    seg = np.column_stack([e[s0:s0 + n, j] for j, s0 in enumerate(st)])
    f = np.fft.rfftfreq(n, 1 / FS)
    F = np.fft.rfft((seg - seg.mean(0)) * np.hanning(n)[:, None], axis=0)
    F *= np.exp(-2j * np.pi * f[:, None] * np.array(st)[None, :] / FS)
    offs = []
    for fi in np.arange(0.85, 1.6, 0.075):
        i = np.argmin(abs(f - fi))
        k = theory.wavenumber(2 * np.pi * f[i], H)
        row = []
        for j in range(4):
            dph = np.angle(F[i, j] / F[i, 2])
            nom = X[j] - X[2]
            mm = np.round((-k * nom - dph) / (2 * np.pi))
            row.append(-(dph + 2 * np.pi * mm) / k - nom)
        offs.append(row)
    offs = np.array(offs) * 1e3
    print("  offset from documented position [mm], mean ± spread over 0.85-1.6 Hz:")
    for j in range(4):
        print(f"    probe {j + 1} (x = {X[j]:.3f} m): {offs[:, j].mean():+6.1f} ± {offs[:, j].std():.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test370060_bud_ampl7mm_focused_x8.15m.csv")
