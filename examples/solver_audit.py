#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dispersion_audit.py — measure the effective quartic dispersion coefficient Λ of a
wave solver, ω² = c²k²(1 + Λk² + μk⁴ + ...), and compare it with exact theory.

Physics: linear water waves, ω² = g k tanh(kh)  →  Λ_phys = −h²/3, μ = 2h⁴/15.

Part A (solver audit). Three linear solvers on a periodic domain, standing-wave
initial condition η = cos kx, u = 0; ω measured from the time series at one point
(least-squares sinusoid fit over many periods). Λ_eff fitted from
y(k) = ω²/(ghk²) − 1 = Λ k² + μ k⁴ over kh ≤ KH_FIT.
  SWE   : hydrostatic shallow-water, staggered 2nd-order FD, leapfrog
          predicted Λ = −Δx²/12 + gh Δt²/12           (no physical dispersion)
  BOUSS : linear Peregrine Boussinesq, same discretization
          predicted Λ = −h²/3 − Δx²/12 + gh Δt²/12
  EXACT : spectral, exact dispersion (null test: must give −h²/3)
  INJ   : spectral with ω² = ghk²(1 + Λ_inj k²) (injection test)

Part B (measurement feasibility). Synthetic flume: regular waves, 20 gauges,
Gaussian gauge noise; k estimated from the phase across gauges at the forcing
frequency, Λ fitted over a set of periods. Gives σ(Λ) for a realistic lab setup
and checks bias with an injected offset δΛ.
"""
import numpy as np
from scipy.optimize import least_squares

g = 9.81
H = 0.50            # water depth [m]
L = 40.0            # periodic domain [m]
KH_FIT = 0.8        # fit range in kh
COURANT = 0.5
rng = np.random.default_rng(20261016)


def kgrid(N, dx):
    return 2 * np.pi * np.fft.fftfreq(N, dx)


def run(solver, k_mode, dx, n_periods=60, lam_inj=0.0):
    """Time-step a standing wave of wavenumber k_mode; return measured ω."""
    N = int(round(L / dx))
    x = dx * np.arange(N)
    k = kgrid(N, dx)
    c = np.sqrt(g * H)
    dt = COURANT * dx / c
    keff = (2 / dx) * np.sin(k * dx / 2)                 # staggered 2nd-order FD symbol
    shift = np.exp(1j * k * dx / 2)                       # u lives at x + dx/2
    eta_hat = np.fft.fft(np.cos(k_mode * x))
    T_est = 2 * np.pi / (c * k_mode)
    nsteps = int(n_periods * T_est / dt)
    rec = np.empty(nsteps)
    if solver in ("SWE", "BOUSS"):
        # D: d/dx from u-points to η-points; G: d/dx from η-points to u-points
        D = 1j * keff / shift
        G = 1j * keff * shift
        M = 1.0 + (H ** 2 / 3) * keff ** 2 if solver == "BOUSS" else 1.0
        u_hat = np.zeros(N, complex)
        u_hat += -0.5 * dt * g * G * eta_hat / M          # start u at t = +dt/2 (η at rest velocity 0)
        for n in range(nsteps):
            rec[n] = eta_hat.real.sum() / N           # η at x = 0
            eta_hat = eta_hat - dt * H * D * u_hat
            u_hat = u_hat - dt * g * G * eta_hat / M
        t = dt * np.arange(nsteps)
    else:
        # spectral: the initial condition is a single mode, which evolves as cos(ω t)
        if solver == "EXACT":
            w = np.sqrt(g * k_mode * np.tanh(k_mode * H))
        else:  # INJ
            w = np.sqrt(g * H * k_mode ** 2 * (1 + lam_inj * k_mode ** 2))
        t = dt * np.arange(nsteps)
        rec = np.cos(w * t)
    # frequency: FFT peak (zero-padded, parabolic interpolation) as the start value,
    # then a least-squares sinusoid fit (no theory used for the start value)
    nfft = 8 * int(2 ** np.ceil(np.log2(len(rec))))
    spec = np.abs(np.fft.rfft(rec * np.hanning(len(rec)), nfft))
    i = int(np.argmax(spec[1:])) + 1
    a, b, cc = np.log(spec[i - 1:i + 2])
    w0 = 2 * np.pi * (i + 0.5 * (a - cc) / (a - 2 * b + cc)) / (nfft * dt)
    res = least_squares(lambda p: p[0] * np.cos(p[1] * t + p[2]) - rec, [1.0, w0, 0.0],
                        x_scale=[1, w0, 1])
    return abs(res.x[1])


def fit_lambda(ks, ws):
    y = ws ** 2 / (g * H * ks ** 2) - 1
    A = np.vstack([ks ** 2, ks ** 4]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # Λ, μ


def part_a():
    print("Part A — effective Λ of solvers (h = %.2f m, Λ_phys = −h²/3 = %.4e m²)" % (H, -H ** 2 / 3))
    khs = np.linspace(0.1, KH_FIT, 8)
    rows = []
    for solver in ("EXACT", "SWE", "BOUSS"):
        for dx in (0.05, 0.10, 0.20):
            dk = 2 * np.pi / L
            ks = np.round(khs / H / dk) * dk              # modes that fit the periodic box
            ws = np.array([run(solver, kk, dx) for kk in ks])
            lam, mu = fit_lambda(ks, ws)
            dt = COURANT * dx / np.sqrt(g * H)
            num = -dx ** 2 / 12 + g * H * dt ** 2 / 12
            pred = {"EXACT": -H ** 2 / 3, "SWE": num, "BOUSS": -H ** 2 / 3 + num}[solver]
            rows.append((solver, dx, lam, pred))
            print(f"  {solver:6s} dx {dx:.2f} m   Λ_meas {lam:+.4e}   Λ_pred {pred:+.4e}   "
                  f"meas/pred {lam / pred:7.4f}   Λ_meas − Λ_phys {lam + H ** 2 / 3:+.3e}")
    print("\n  injection (spectral, ω² = ghk²(1+Λk²)), dx 0.10:")
    for lam_inj in (-H ** 2 / 3, -H ** 2 / 3 * 1.05, 0.0, 1e-3):
        dk = 2 * np.pi / L
        ks = np.round(khs / H / dk) * dk
        ws = np.array([run("INJ", kk, 0.10, lam_inj=lam_inj) for kk in ks])
        y = ws ** 2 / (g * H * ks ** 2) - 1
        lam = np.linalg.lstsq(np.vstack([ks ** 2, ks ** 4]).T, y, rcond=None)[0][0]
        print(f"    Λ_inj {lam_inj:+.4e}   Λ_meas {lam:+.4e}   diff {lam - lam_inj:+.1e}")
    return rows


def flume_k(omega, lam_true, gauges, amp, noise, fs, T_rec):
    """Synthetic gauge records of a progressive wave; returns k estimated from phase slope."""
    # true k from ω² = ghk²(1+Λk²) via exact tanh part replaced by the quartic model + μ term
    from scipy.optimize import brentq
    mu = 2 * H ** 4 / 15
    f = lambda kk: g * H * kk ** 2 * (1 + lam_true * kk ** 2 + mu * kk ** 4) - omega ** 2
    k_true = brentq(f, 1e-4, 50)
    t = np.arange(0, T_rec, 1 / fs)
    ph = []
    for xg in gauges:
        rec = amp * np.cos(k_true * xg - omega * t) + noise * rng.standard_normal(t.size)
        z = np.sum(rec * np.exp(1j * omega * t))           # complex demodulation at ω
        ph.append(np.angle(z))
    ph = np.unwrap(np.array(ph))
    A = np.vstack([gauges, np.ones_like(gauges)]).T
    coef, res, *_ = np.linalg.lstsq(A, ph, rcond=None)
    return coef[0], k_true


def part_b(trials=400):
    print("\nPart B — flume feasibility (20 gauges over 10 m, a = 2 cm, noise 0.5 mm, 60 s at 50 Hz)")
    gauges = np.linspace(2.0, 12.0, 20)
    khs = np.linspace(0.2, KH_FIT, 7)
    mu = 2 * H ** 4 / 15
    omegas = []
    for kh in khs:
        kk = kh / H
        omegas.append(np.sqrt(g * H * kk ** 2 * (1 - H ** 2 / 3 * kk ** 2 + mu * kk ** 4)))
    for dlam in (0.0, -0.01 * H ** 2 / 3, 1e-3, -0.05 * H ** 2 / 3):
        lam_true = -H ** 2 / 3 + dlam
        est = []
        for _ in range(trials):
            ks = np.array([flume_k(w, lam_true, gauges, 0.02, 5e-4, 50, 60.0)[0] for w in omegas])
            ws = np.array(omegas)
            y = ws ** 2 / (g * H * ks ** 2) - 1 - mu * ks ** 4       # μ fixed at theory
            est.append(np.sum(y * ks ** 2) / np.sum(ks ** 4))
        est = np.array(est)
        print(f"  δΛ_inj {dlam:+.3e}   mean(Λ̂ − Λ_phys) {est.mean() + H ** 2 / 3:+.3e}   "
              f"σ(Λ̂) {est.std(ddof=1):.2e}   σ/|Λ_phys| {est.std(ddof=1) / (H ** 2 / 3):.2e}")
    print(f"  reference: grid term Δx²/12 at Δx = 0.05/0.10/0.20 m = "
          f"{0.05 ** 2 / 12:.1e} / {0.1 ** 2 / 12:.1e} / {0.2 ** 2 / 12:.1e} m²")


if __name__ == "__main__":
    part_a()
    part_b()
