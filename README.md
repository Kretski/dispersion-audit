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
| Focused group, synthetic (broadband k(f)) | unbiased; injected +1 % recovered; σ conservative |

### First real data: UiO wave flume (Lande 2026, doi:10.18710/FZN3AL, CC BY 4.0)

Flat bed, h = 0.6 m, focused groups, g = 9.819 m/s² (as used by the data authors), band
0.85–1.6 Hz (main test) and 0.9–1.6 Hz (dissipation tests); k h ≈ 1.8–5.8, practically deep
water, so a depth error cannot matter.

| Run | Probes | Amplitude | δk/k (measured vs exact linear) |
|---|---|---|---|
| main test, 2024 (C01A07F01) | 7.407 / 8.150 / 8.845 m | 7 mm | **+1.39 % ± 0.25 %** |
| dissipation test, 2025, 3 repeats | 5 / 8 / 12 m | 7 mm | **+0.49 % ± 0.05 %** (repeat scatter 0.01 %) |
| dissipation test, 2025 | 5 / 8 / 12 m | 25 mm | −0.2 % ± 0.2 % (strongly f-dependent) |
| dissipation test, 2025 | 5 / 8 / 12 m | 45 mm | −1.3 … −2.5 %, growing with f |
| shoal, Nov 2024 (C02A07F01) | 7.407 / 8.150 / 8.845 m over h = 0.19 m | 7 mm | documented geometry fits (χ² 20 / 14); see below |

What the numbers support:
- The amplitude trend (+0.49 % → −0.2 % → −1.3 … −2.5 %) has the sign and size of Stokes amplitude
  dispersion (steeper waves travel faster), so the tool resolves real nonlinear physics.
- Linear (7 mm) waves show a **repeatable positive excess in k** in both campaigns, i.e. they
  appear slightly slower than ideal linear theory.

What is not resolved (open question for the data authors, not a claim of error):
- In the 2025 runs the excess fits δk ∝ ω (a uniform extra slowness ≈ 3.8 ms/m) far better
  than a probe-position scale error (δk ∝ k). Probe-specific timing offsets are ruled out:
  the 25 mm run from the same day and probes shows no such offsets.
- The measured amplitude decay is 2–5× too small to explain the excess by laminar boundary
  layers alone (for which the phase shift equals the damping); surface contamination,
  probe-position uncertainty (dissipation-test positions are given to 0.1 m) or a
  combination remain possible.
- In the 2024 main test the array is short (1.44 m) and the data cannot distinguish a
  position offset (≈ 21 mm) from a timing offset (≈ 17 ms); the excess per metre is about
  three times larger than in 2025.
- The probe at 1 m from the wavemaker is excluded (likely near-field effects).

Shoal case (variable depth, WKB phase integral of exact linear k over the documented
geometry; kh = 0.8–1.9 on the shoal top, where k is depth-sensitive, about −0.2 % per mm):
- The documented top depth (190 mm) fits at χ² 20 / 14; a best-fit depth of 184.7 mm improves
  this to 13.5 / 13 (≈ 2.6 σ, weak).
- Probe 4 (8.845 m) is consistent with its documented position (+0.9 ± 2.8 mm relative to
  probe 3); probe 2 (7.407 m) shows −9.5 ± 2.8 mm (or −8.8 ± 2.6 ms). In the flat-bed main
  test, two months earlier with the same probe positions, probe 2 gave −15 ± 4 mm, so the
  probe-2 offset is seen in two independent runs; probe 4 differs between them (+6 ± 4 vs
  +1 ± 3 mm), which is within the uncertainties.

Scripts: `examples/uio_flume_A07.py` (main test), `examples/uio_dissipation.py` (dissipation tests),
`examples/uio_shoal.py` (shoal case). One-run and three-repeat results are
reproducible from the public files.

## Layout

```
dispersion_audit/theory.py     exact linear dispersion, wavenumber solver
dispersion_audit/gauges.py     complex amplitudes, k fit with reflection, bootstrap, depth fit
dispersion_audit/synthetic.py  synthetic gauge records (null / injection tests)
dispersion_audit/broadband.py  k(f) from a transient wave group (per-gauge windows)
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
