#!/usr/bin/env python3
"""Residual version: δy(k) = y_solver(k) − y_ref(k), y = ω²/(ghk²) − 1, fitted as δΛ k² + δμ k⁴.
ref = exact tanh (total error) or the solver's own continuum equations (numerical error only)."""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver_audit as da
H, g, L = da.H, da.g, da.L
khs = np.linspace(0.1, da.KH_FIT, 8)
dk = 2*np.pi/L
ks = np.round(khs/H/dk)*dk
y_exact = np.tanh(ks*H)/(ks*H) - 1
y_bous = 1/(1 + (ks*H)**2/3) - 1
y_swe = np.zeros_like(ks)
A = np.vstack([ks**2, ks**4]).T
fit = lambda d: np.linalg.lstsq(A, d, rcond=None)[0]
print(f"h = {H} m, kh = {khs[0]:.1f}–{khs[-1]:.1f}; numerical prediction δΛ_num = −Δx²/12 + ghΔt²/12")
print(f"{'solver':6s} {'dx':>5s} {'δΛ vs own PDE':>14s} {'pred δΛ_num':>12s} {'ratio':>7s} {'δΛ vs exact':>12s} {'|δΛ_tot|/(h²/3)':>16s}")
for solver, yc in (("SWE", y_swe), ("BOUSS", y_bous)):
    for dx in (0.05, 0.10, 0.20):
        ws = np.array([da.run(solver, kk, dx) for kk in ks])
        y = ws**2/(g*H*ks**2) - 1
        dt = da.COURANT*dx/np.sqrt(g*H)
        pred = -dx**2/12 + g*H*dt**2/12
        own = fit(y - yc)[0]
        tot = fit(y - y_exact)[0]
        print(f"{solver:6s} {dx:5.2f} {own:+14.4e} {pred:+12.4e} {own/pred:7.4f} {tot:+12.4e} {abs(tot)/(H**2/3):16.4f}")
# null: exact solver vs exact
ws = np.array([da.run("EXACT", kk, 0.1) for kk in ks])
print(f"null EXACT vs exact: δΛ = {fit(ws**2/(g*H*ks**2)-1-y_exact)[0]:+.1e}")
# Abbott-type coincidence: SWE grid dispersion equals physical when Δx² − ghΔt² = 4h²
print(f"SWE would mimic Λ_phys if Δx²(1−C²)/12 = h²/3, i.e. Δx = {np.sqrt(4*H**2/(1-da.COURANT**2)):.2f} m at C = {da.COURANT}")
