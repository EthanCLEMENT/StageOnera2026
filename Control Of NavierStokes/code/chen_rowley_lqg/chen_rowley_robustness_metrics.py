#!/usr/bin/env python3
"""
Robustness and stability diagnostics for the Chen--Rowley complex
Ginzburg--Landau LQG reproduction script.

This imports your local ChenRowley.py, builds the nominal LQG controller, and reports:

1. Nominal closed-loop stability/performance
   - spectral abscissa max Re(lambda)
   - stability gap -max Re(lambda)
   - H2 norm of w -> z
   - signed-frequency Hinf norm of w -> z

2. Loop robustness metrics using the standard negative-feedback loop
       u = -Kc(s) y,       L(s) = P_yu(s) Kc(s)
       S = (I + L)^(-1),   T = L(I + L)^(-1),   KS = Kc S
   - ||S||_inf: sensitivity peak; modulus margin = 1/||S||_inf
   - ||T||_inf: complementary sensitivity peak; inverse is an unweighted
     multiplicative-uncertainty robustness radius
   - ||KS||_inf: additive plant-output uncertainty robustness metric
   - min_w sigma_min(I+L(iw)): return-difference margin

3. Fixed-controller robustness sweeps
   - mu0/supercritical-factor sweep, keeping the nominal controller fixed
   - actuator/sensor placement jitter, keeping the nominal controller fixed

Notes
-----
The plant/controller are complex-valued, so frequency responses are generally not
symmetric. This script scans signed frequencies by default.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import eigvals, solve, solve_continuous_lyapunov, svdvals


def load_module(path: str):
    p = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("chenrowley_module", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def spectral_abscissa(A: np.ndarray) -> float:
    return float(np.max(eigvals(A).real))


def stable_h2_from_lft(ZA: np.ndarray, ZB: np.ndarray, ZC: np.ndarray) -> float:
    alpha = spectral_abscissa(ZA)
    if alpha >= 0.0:
        return float("inf")
    Wc = solve_continuous_lyapunov(ZA, -(ZB @ ZB.conj().T))
    val = float(np.real(np.trace(ZC @ Wc @ ZC.conj().T)))
    return float(np.sqrt(max(0.0, val)))


def signed_log_grid(wmin: float, wmax: float, ngrid: int, include_zero: bool = True) -> np.ndarray:
    if not (0.0 < wmin < wmax):
        raise ValueError("Need 0 < wmin < wmax")
    pos = np.exp(np.linspace(np.log(wmin), np.log(wmax), ngrid))
    parts = [-pos[::-1]]
    if include_zero:
        parts.append(np.array([0.0]))
    parts.append(pos)
    return np.concatenate(parts)


def fixed_controller_lft(mod, A_actual, x, M, xa_actual, xs_actual, disturbance, params_actual,
                         A_nom, B2_nom, C2_nom, F_nom, L_nom):
    """Closed-loop LFT with actual plant and nominal fixed LQG controller."""
    B1a, B2a, C1a, C2a, D12a, D21a = mod.placement_matrices(
        x, M, xa_actual, xs_actual, disturbance, params_actual
    )
    ZA = np.block([
        [A_actual, -B2a @ F_nom],
        [L_nom @ C2a, A_nom - B2_nom @ F_nom - L_nom @ C2_nom],
    ])
    ZB = np.vstack([B1a, L_nom @ D21a])
    ZC = np.hstack([C1a, -D12a @ F_nom])
    return ZA, ZB, ZC


def evaluate_lft(mod, ZA, ZB, ZC, wmin: float, wmax: float, ngrid: int, refine: bool = True) -> dict:
    alpha = spectral_abscissa(ZA)
    if alpha >= 0.0:
        return {
            "alpha": alpha,
            "h2": float("inf"),
            "hinf": float("inf"),
            "wpeak": np.nan,
            "neg_hinf": np.nan,
            "pos_hinf": np.nan,
        }
    h2 = stable_h2_from_lft(ZA, ZB, ZC)
    hres = mod.hinf_norm_grid(
        ZA, ZB, ZC,
        wmin=wmin, wmax=wmax, ngrid=ngrid,
        refine=refine, signed=True, include_zero=True, return_grid=False,
    )
    return {
        "alpha": alpha,
        "h2": h2,
        "hinf": hres.hinf,
        "wpeak": hres.omega,
        "neg_hinf": hres.neg_hinf,
        "pos_hinf": hres.pos_hinf,
    }


def print_eval(label: str, r: dict) -> None:
    print(label)
    print(f"  spectral abscissa alpha=max Re(lambda) = {r['alpha']:.10g}")
    print(f"  exponential stability gap -alpha       = {-r['alpha']:.10g}")
    print(f"  H2(w->z)                              = {r['h2']:.10g}")
    print(f"  signed Hinf(w->z)                    = {r['hinf']:.10g} at omega={r['wpeak']:.10g}")
    print(f"    negative-frequency peak             = {r['neg_hinf']:.10g}")
    print(f"    positive-frequency peak             = {r['pos_hinf']:.10g}")


def loop_robustness_metrics(mod, A, x, M, xa, xs, disturbance, params, out, wmin, wmax, ngrid) -> dict:
    """Compute loop metrics for u=-Kc y, L=P_yu Kc, S=(I+L)^-1.

    Kc is the positive controller in the standard negative-feedback convention:
        controller realization: qhatdot = Ak qhat + Lgain y,  u = -F qhat = -Kc y.
    """
    _, B2, _, C2, _, _ = mod.placement_matrices(x, M, xa, xs, disturbance, params)
    F = out.F
    Lgain = out.L
    Ak = A - B2 @ F - Lgain @ C2

    ms = C2.shape[0]
    n = A.shape[0]
    nk = Ak.shape[0]
    Iplant = np.eye(n, dtype=complex)
    Ik = np.eye(nk, dtype=complex)
    Iy = np.eye(ms, dtype=complex)

    omegas = signed_log_grid(wmin, wmax, ngrid, include_zero=True)
    rows = []

    for w in omegas:
        s = 1j * float(w)
        P = C2 @ solve(s * Iplant - A, B2, assume_a="gen")          # y/u
        Kc = F @ solve(s * Ik - Ak, Lgain, assume_a="gen")          # positive Kc, u=-Kc y
        Lop = P @ Kc                                                # loop L=P Kc
        return_difference = Iy + Lop
        S = solve(return_difference, Iy, assume_a="gen")            # (I+L)^-1
        T = Lop @ S
        KS = Kc @ S

        sigma_S = float(svdvals(S)[0])
        sigma_T = float(svdvals(T)[0])
        sigma_KS = float(svdvals(KS)[0])
        sigma_L = float(svdvals(Lop)[0])
        sigma_RD_min = float(svdvals(return_difference)[-1])
        rows.append((w, sigma_S, sigma_T, sigma_KS, sigma_L, sigma_RD_min))

    arr = np.asarray(rows, dtype=float)
    w = arr[:, 0]

    def peak(col: int) -> tuple[float, float]:
        k = int(np.argmax(arr[:, col]))
        return float(arr[k, col]), float(w[k])

    def trough(col: int) -> tuple[float, float]:
        k = int(np.argmin(arr[:, col]))
        return float(arr[k, col]), float(w[k])

    Ms, w_Ms = peak(1)
    Mt, w_Mt = peak(2)
    Mks, w_Mks = peak(3)
    Lpk, w_Lpk = peak(4)
    rd_min, w_rd = trough(5)

    return {
        "Ms": Ms,
        "w_Ms": w_Ms,
        "Mt": Mt,
        "w_Mt": w_Mt,
        "Mks": Mks,
        "w_Mks": w_Mks,
        "loop_peak": Lpk,
        "w_loop_peak": w_Lpk,
        "return_diff_min": rd_min,
        "w_return_diff_min": w_rd,
        "modulus_margin": 1.0 / Ms if Ms > 0 else np.inf,
        "mult_unc_radius": 1.0 / Mt if Mt > 0 else np.inf,
        "additive_output_unc_radius": 1.0 / Mks if Mks > 0 else np.inf,
        "grid": arr,
    }


def print_loop_metrics(r: dict) -> None:
    print("\nLoop robustness metrics, standard negative-feedback convention u=-Kc y")
    print(f"  Ms = ||S||inf                         = {r['Ms']:.10g} at omega={r['w_Ms']:.10g}")
    print(f"  modulus margin 1/Ms                  = {r['modulus_margin']:.10g}")
    print(f"  Mt = ||T||inf                         = {r['Mt']:.10g} at omega={r['w_Mt']:.10g}")
    print(f"  unweighted multiplicative radius 1/Mt = {r['mult_unc_radius']:.10g}")
    print(f"  ||Kc S||inf                           = {r['Mks']:.10g} at omega={r['w_Mks']:.10g}")
    print(f"  additive-output uncertainty radius    = {r['additive_output_unc_radius']:.10g}")
    print(f"  min sigma_min(I+L)                    = {r['return_diff_min']:.10g} at omega={r['w_return_diff_min']:.10g}")
    print(f"  peak sigma_max(L)                     = {r['loop_peak']:.10g} at omega={r['w_loop_peak']:.10g}")


def mu_sweep(mod, args, params_nom, x, M, A_nom, out_nom) -> None:
    print("\nFixed-controller mu0/supercritical-factor sweep")
    print("  factor means mu0 = factor * mu_c. Controller remains fixed at the nominal design.")
    print("  factor, alpha_open, alpha_closed, H2, Hinf, omega_peak")
    for fac in args.mu_factors:
        params_actual = replace(params_nom, supercritical_factor=float(fac))
        _, A_actual, _, _ = mod.build_gl_operator(args.N, params_actual, supercritical=True)
        ZA, ZB, ZC = fixed_controller_lft(
            mod, A_actual, x, M, args.xa, args.xs, args.disturbance, params_actual,
            A_nom, out_nom.B2, out_nom.C2, out_nom.F, out_nom.L,
        )
        r = evaluate_lft(mod, ZA, ZB, ZC, args.wmin, args.wmax, args.ngrid, refine=False)
        print(f"  {fac:.6g}, {spectral_abscissa(A_actual):.6g}, {r['alpha']:.6g}, "
              f"{r['h2']:.6g}, {r['hinf']:.6g}, {r['wpeak']:.6g}")


def jitter_check(mod, args, params_nom, x, M, A_nom, out_nom) -> None:
    print("\nFixed-controller actuator/sensor placement-error check")
    print("  Actual positions are randomly perturbed; controller remains fixed at nominal design.")
    print("  sigma_x, stable_frac, worst_alpha, median_Hinf, worst_Hinf, median_H2, worst_H2")
    rng = np.random.default_rng(args.seed)
    z0 = np.r_[args.xa, args.xs].astype(float)
    ma = len(args.xa)
    for sig in args.jitter_sigmas:
        vals = []
        stable = 0
        for _ in range(args.jitter_samples):
            z = z0 + rng.normal(scale=float(sig), size=z0.shape)
            xa = z[:ma]
            xs = z[ma:]
            ZA, ZB, ZC = fixed_controller_lft(
                mod, A_nom, x, M, xa, xs, args.disturbance, params_nom,
                A_nom, out_nom.B2, out_nom.C2, out_nom.F, out_nom.L,
            )
            r = evaluate_lft(mod, ZA, ZB, ZC, args.wmin, args.wmax, args.ngrid, refine=False)
            vals.append(r)
            stable += int(r["alpha"] < 0.0)
        alphas = np.array([v["alpha"] for v in vals], dtype=float)
        hinfs = np.array([v["hinf"] for v in vals], dtype=float)
        h2s = np.array([v["h2"] for v in vals], dtype=float)
        print(f"  {float(sig):.6g}, {stable / args.jitter_samples:.3f}, {np.max(alphas):.6g}, "
              f"{np.nanmedian(hinfs):.6g}, {np.nanmax(hinfs):.6g}, "
              f"{np.nanmedian(h2s):.6g}, {np.nanmax(h2s):.6g}")


def save_loop_csv(path: str, r: dict) -> None:
    header = "omega,sigma_S,sigma_T,sigma_KS,sigma_loop,sigma_min_return_difference"
    np.savetxt(path, r["grid"], delimiter=",", header=header, comments="")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--module", default="./ChenRowley.py", help="path to your ChenRowley.py reproduction script")
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--subcritical", action="store_true", help="use mu0=0.96 mu_c instead of 1.03 mu_c")
    p.add_argument("--disturbance", choices=["everywhere", "upstream"], default="everywhere")
    p.add_argument("--xa", nargs="+", type=float, default=[-1.03])
    p.add_argument("--xs", nargs="+", type=float, default=[0.98])
    p.add_argument("--wmin", type=float, default=1e-4)
    p.add_argument("--wmax", type=float, default=5.0)
    p.add_argument("--ngrid", type=int, default=151)
    p.add_argument("--final-ngrid", type=int, default=401)
    p.add_argument("--mu-factors", nargs="*", type=float, default=[0.96, 1.00, 1.03, 1.05, 1.08, 1.10, 1.15])
    p.add_argument("--no-mu-sweep", action="store_true")
    p.add_argument("--jitter-samples", type=int, default=40)
    p.add_argument("--jitter-sigmas", nargs="*", type=float, default=[0.05, 0.10, 0.25, 0.50, 1.00])
    p.add_argument("--no-jitter", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--loop-csv", default=None, help="optional CSV for loop metrics vs signed frequency")
    args = p.parse_args(argv)

    mod = load_module(args.module)
    params = mod.GLParams()
    supercritical = not args.subcritical
    x, A, M, info = mod.build_gl_operator(args.N, params, supercritical=supercritical)
    out = mod.lqg_controller_and_norm(A, x, M, args.xa, args.xs, args.disturbance, params)

    print("Nominal LQG design")
    print(f"  N={args.N}, disturbance={args.disturbance}")
    print(f"  xa={args.xa}")
    print(f"  xs={args.xs}")
    print(f"  mu0={info['mu0']:.12g}, mu_c={info['mu_c']:.12g}")
    print(f"  open-loop spectral abscissa={spectral_abscissa(A):.10g}")
    print(f"  LQG/H2 value from Riccati formula={out.h2:.10g}")

    nom = evaluate_lft(mod, out.ZA, out.ZB, out.ZC, args.wmin, args.wmax, args.final_ngrid, refine=True)
    print_eval("\nNominal closed-loop LFT", nom)

    loop = loop_robustness_metrics(
        mod, A, x, M, args.xa, args.xs, args.disturbance, params, out,
        args.wmin, args.wmax, args.final_ngrid,
    )
    print_loop_metrics(loop)
    if args.loop_csv:
        save_loop_csv(args.loop_csv, loop)
        print(f"  saved loop-metric frequency sweep CSV: {args.loop_csv}")

    if not args.no_mu_sweep:
        mu_sweep(mod, args, params, x, M, A, out)
    if not args.no_jitter:
        jitter_check(mod, args, params, x, M, A, out)


if __name__ == "__main__":
    main()
