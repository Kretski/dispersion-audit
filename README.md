# dispersion-audit

Measure how well a **wave basin** or a **numerical wave model** reproduces linear
wave dispersion, and express the error as one number with an honest uncertainty.

Linear water waves obey ω² = g k tanh(k h). A numerical model (finite grid, few vertical
layers, simplified equations) or a basin experiment (depth error, reflections, wavemaker
effects) can deviate from it. `dispersion-audit` measures the wavenumber k actually
present in the data and compares it with the exact value:

- **from gauge records** (lab or model output): several synchronous wave gauges along the
  wave direction; k is fitted with the incident and reflected waves separated, the
  reflection coefficient comes out as a by-product;
- **from a solver** (self-test): the effective dispersion of a discretized equation,
  compared with its known von Neumann prediction.

For long waves (k h ≲ 1) the result is summarized as a quartic coefficient,
ω² = g h k² (1 + Λ k² + …), with Λ_phys = −h²/3 and a grid term Λ_grid ≈ −Δx²/12.

## What is and is not new

Nothing here is new physics. Numerical dispersion (von Neumann analysis) and
incident/reflected wave separation (Mansard & Funke 1980) are textbook methods.
What the package adds is discipline: the measured quantity is a **residual against the
exact theory**, every estimate comes with a bootstrap uncertainty, and the method is
checked with null tests, injected errors and coverage tests before it is applied to data.

## Quick start

```bash
pip install numpy scipy pytest
python -m pytest -q tests                      # null, injection, coverage, depth fit (~1.5 min)
python examples/make_synthetic_runs.py
python -m dispersion_audit.cli examples/data/synthetic_T*.txt \
       --fs 100 --x 0,0.31,0.74,1.27,1.93 --h 0.50 --t0 20 --t1 80 --scale 0.001
```

Output per run: frequency, k h, measured k ± σ, exact k, relative error δk/k ± σ,
reflection coefficient, χ²/dof of the gauge fit; then the depth that best explains all runs.

## Data needed from a basin

1. **≥ 3 gauges (better 4–6), recorded on the same clock**, along the wave direction.
2. **Positions to ~1 mm**, preferably unequal spacing; adjacent gauges closer than half a
   wavelength (the tool flags spatial-aliasing risk).
3. **Still-water depth to ~0.5 mm**, or several periods so depth can be fitted jointly:
   1 mm of depth error is larger than the grid error of a fine model.
4. **Regular waves, small steepness**, flat bottom, several periods (k h from ~0.3 to ~3).
5. **Steady window** after the wave front has passed (30–60 s or more).
6. Calibration coefficients of the gauges (units → metres).

Resistive gauges are preferred to ultrasonic ones (point measurement, no internal delay).

## Validation status

| Test | Result |
|---|---|
| Exact-dispersion solver (null) | δΛ = 1e-16 |
| Injected Λ in a spectral solver | recovered to 1e-16 |
| 2nd-order FD shallow-water solver | Λ matches von Neumann (−Δx²/12 + ghΔt²/12) to 4 digits |
| Linear Boussinesq solver, h = 0.5 m | total Λ error 0.7 / 1.2 / 3.4 % of h²/3 at Δx = 5 / 10 / 20 cm |
| Synthetic gauges, 25 % reflection, 0.5 mm noise | pull mean 0.0–0.5, pull std 0.9–1.1 (40 runs per case) |
| Injected δk/k = +1 %, −2 % | recovered within 4σ, reflection within 0.03 |
| Depth fit, 4 periods | 0.505 m recovered to < 2 mm |

Known limitations:
- **Not yet applied to real basin data or to an external model (e.g. SWASH).** This is the next step.
- A small bias of ~5×10⁻⁵ in δk/k (≈ 0.3–0.5 σ) remains at k h ≈ 2 in the synthetic tests.
- Linear theory only: strongly nonlinear or breaking waves are outside the scope.
- The long-wave Λ summary is meaningful only for k h ≲ 1; for deeper water δk/k is reported.
- A raw fit of Λ against −h²/3 has a truncation bias of ~1.7 % (k h ≤ 0.8); always use the
  residual against the exact relation (as the code does).

## Layout

```
dispersion_audit/theory.py     exact linear dispersion, wavenumber solver
dispersion_audit/gauges.py     complex amplitudes, k fit with reflection, bootstrap, depth fit
dispersion_audit/synthetic.py  synthetic gauge records (null / injection tests)
dispersion_audit/cli.py        command line for regular-wave runs
examples/solver_audit.py       effective Λ of three reference solvers
examples/residual_audit.py     residual version (solver vs own PDE, solver vs exact)
tests/test_audit.py            pytest suite
```

## Author

Dimitar Kretski, independent researcher, Varna, Bulgaria — ORCID 0000-0001-5108-2243.
Related: [residual-first-framework](https://github.com/Kretski/residual-first-framework).

## License

MIT (see `LICENSE`).
