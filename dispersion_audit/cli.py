"""Command line: audit regular-wave gauge records.

  python -m dispersion_audit.cli run.txt --fs 100 --x 0,0.35,0.80,1.40 --h 0.50 --t0 20 --t1 80
  python -m dispersion_audit.cli run1.txt run2.txt ... --fs 100 --x ... --h ... --scale 0.001

Each file: whitespace/comma separated columns, one column per gauge, same clock,
in the order of --x. --scale converts the file units to metres (e.g. 0.001 for mm).
"""
import argparse
import os

import numpy as np

from . import gauges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--fs", type=float, required=True, help="sampling rate [Hz]")
    ap.add_argument("--x", required=True, help="gauge positions [m], comma separated, column order")
    ap.add_argument("--h", type=float, required=True, help="still-water depth [m]")
    ap.add_argument("--t0", type=float, default=None, help="start of steady window [s]")
    ap.add_argument("--t1", type=float, default=None, help="end of steady window [s]")
    ap.add_argument("--columns", default=None, help="columns to use (0-based, comma separated)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-reflection", action="store_true")
    args = ap.parse_args()
    x = np.array([float(v) for v in args.x.split(",")])
    results = []
    print(f"{'file':28s} {'f [Hz]':>8s} {'kh':>6s} {'k_meas':>9s} {'± σ':>8s} {'k_exact':>9s} "
          f"{'δk/k':>9s} {'± σ':>8s} {'refl':>5s} {'χ²/dof':>7s}")
    for fn in args.files:
        d = np.loadtxt(fn, delimiter="," if fn.endswith(".csv") else None, ndmin=2)
        if args.columns:
            d = d[:, [int(c) for c in args.columns.split(",")]]
        if d.shape[1] != len(x):
            raise SystemExit(f"{fn}: {d.shape[1]} columns but {len(x)} positions")
        i0 = 0 if args.t0 is None else int(args.t0 * args.fs)
        i1 = len(d) if args.t1 is None else int(args.t1 * args.fs)
        r = gauges.analyze_regular(d[i0:i1] * args.scale, args.fs, x, args.h,
                                   reflect=not args.no_reflection)
        results.append(r)
        flag = (" EDGE" if r["edge"] else "") + (" ALIAS?" if r["spatial_alias_risk"] else "")
        print(f"{os.path.basename(fn)[-28:]:28s} {r['f']:8.4f} {r['kh']:6.3f} {r['k_meas']:9.5f} {r['sigma_k']:8.5f} "
              f"{r['k_exact']:9.5f} {r['dk_rel']:+9.2e} {r['sigma_dk_rel']:8.1e} {r['refl']:5.2f} "
              f"{r['chi2_per_dof']:7.2f}{flag}")
    hd = gauges.fit_depth(results, args.h)
    if hd is not None:
        print(f"\ndepth that best fits all runs: {hd[0]:.4f} m (given {args.h} m), χ² = {hd[1]:.1f} "
              f"for {len(results) - 1} dof")
    else:
        print("\none run only: depth and dispersion error cannot be separated")
    if results and max(r["kh"] for r in results) < 1.0 and len(results) >= 2:
        dl, s = gauges.delta_lambda(results, args.h)
        print(f"long-wave summary: δΛ = {dl:+.3e} ± {s:.1e} m²  (Λ_phys = {-args.h ** 2 / 3:.4e} m²)")


if __name__ == "__main__":
    main()
