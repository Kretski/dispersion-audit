"""Write synthetic regular-wave runs (5 gauges, 25% reflection, 0.5 mm noise, depth 0.505 m)
to examples/data/ so the command line can be tried without lab data."""
import os
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dispersion_audit import synthetic

X = [0.0, 0.31, 0.74, 1.27, 1.93]
os.makedirs("examples/data", exist_ok=True)
for T in (1.0, 1.5, 2.2, 3.0):
    rec, k = synthetic.regular_wave(X, 0.505, period=T, refl=0.25, seed=int(T * 10), duration=80)
    np.savetxt(f"examples/data/synthetic_T{T:.1f}.txt", rec * 1000, fmt="%.3f")   # in mm
print("written; try:\n  python -m dispersion_audit.cli examples/data/synthetic_T*.txt --fs 100 "
      "--x 0,0.31,0.74,1.27,1.93 --h 0.50 --t0 20 --t1 80 --scale 0.001")
