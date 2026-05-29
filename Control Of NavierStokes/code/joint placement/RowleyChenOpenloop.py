#!/usr/bin/env python3
"""Check open-loop eigenvalue, H2, and signed-frequency Hinf for ChenRowley.py.

This helper imports the reproduction script as a module and computes open-loop
norms for the transfer

    d -> z = beta * M^(1/2) q,
    qdot = A q + B d,

with B = I for disturbances everywhere or B = Gaussian(x, xd, sigma) for the
upstream disturbance.  H2/Hinf are true finite norms only when A is stable
(e.g. use --subcritical).  For the supercritical case, the script still prints
frequency samples, but labels the true norms as undefined/infinite.

Example:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python check_open_loop_signed_freqs.py \
      --module "/stck/eclement/Control Of NavierStokes/src2/ChenRowley.py" \
      --N 100 --subcritical --both-disturbances --wmin -5 --wmax 5 --ngrid 2001
"""


import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals, solve, solve_continuous_lyapunov, svdvals
from scipy.optimize import minimize_scalar


def import_module_from_path(path: str):
    path = str(Path(path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location("chenrowley_user_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def disturbance_B(cr, x, params, disturbance: str) -> np.ndarray:
    if disturbance == "everywhere":
        return np.eye(x.size, dtype=complex)
    if disturbance == "upstream":
        return cr.gaussian(x, params.xd, params.sigma)[:, None].astype(complex)
    raise ValueError(f"unknown disturbance {disturbance!r}")


def h2_open_loop(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    Wc = solve_continuous_lyapunov(A, -(B @ B.conj().T))
    h2_sq = np.real(np.trace(C @ Wc @ C.conj().T))
    return float(np.sqrt(max(0.0, h2_sq)))


def sigma_at(A: np.ndarray, B: np.ndarray, C: np.ndarray, omega: float) -> float:
    n = A.shape[0]
    G = C @ solve(1j * omega * np.eye(n, dtype=complex) - A, B, assume_a="gen")
    return float(svdvals(G)[0])


def signed_hinf_grid(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    wmin: float,
    wmax: float,
    ngrid: int,
    refine: bool = True,
):
    if wmin >= wmax:
        raise ValueError("wmin must be smaller than wmax")
    ws = np.linspace(wmin, wmax, ngrid)
    sig = np.array([sigma_at(A, B, C, float(w)) for w in ws])

    def best_in_mask(mask):
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return np.nan, np.nan
        local = idxs[np.argmax(sig[idxs])]
        best_w = float(ws[local])
        best_s = float(sig[local])
        if refine and 0 < local < len(ws) - 1:
            lo, hi = float(ws[local - 1]), float(ws[local + 1])
            res = minimize_scalar(
                lambda om: -sigma_at(A, B, C, float(om)),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-9},
            )
            if res.success and -res.fun > best_s:
                best_w = float(res.x)
                best_s = float(-res.fun)
        return best_s, best_w

    neg = best_in_mask(ws < 0.0)
    pos = best_in_mask(ws > 0.0)
    allp = best_in_mask(np.ones_like(ws, dtype=bool))
    return allp, neg, pos, ws, sig


def run_case(cr, args, disturbance: str) -> None:
    params = cr.GLParams()
    x, A, M, info = cr.build_gl_operator(args.N, params, supercritical=not args.subcritical)
    lam = eigvals(A)
    lead = lam[np.argmax(lam.real)]
    stable = bool(np.max(lam.real) < 0.0)

    B = disturbance_B(cr, x, params, disturbance)
    C = params.beta_cost * np.diag(np.sqrt(np.diag(M).real))

    print("\n" + "=" * 72)
    print(f"Open-loop check: N={args.N}, disturbance={disturbance}, "
          f"{'subcritical' if args.subcritical else 'supercritical'}")
    print(f"mu0 = {info['mu0']:.12g}, mu_c = {info['mu_c']:.12g}")
    print(f"analytical lambda0 = {info['lambda0_analytical']:.12g}")
    print(f"numerical leading eigenvalue = {lead:.12g}")
    print(f"max Re(lambda) = {np.max(lam.real):.12g}")
    print(f"dominant modal frequency Im(lambda0) = {lead.imag:.12g}")

    if stable:
        h2 = h2_open_loop(A, B, C)
        print(f"open-loop H2  ||d -> beta*M^(1/2)q||_2 = {h2:.12g}")
    else:
        print("open-loop H2 is undefined/infinite because A is unstable.")

    (hall, wall), (hneg, wneg), (hpos, wpos), ws, sig = signed_hinf_grid(
        A, B, C, args.wmin, args.wmax, args.ngrid, refine=not args.no_refine
    )
    if stable:
        print(f"signed-grid Hinf over [{args.wmin:g}, {args.wmax:g}] = {hall:.12g} at omega={wall:.12g}")
        print(f"  negative-frequency peak = {hneg:.12g} at omega={wneg:.12g}")
        print(f"  positive-frequency peak = {hpos:.12g} at omega={wpos:.12g}")
    else:
        print("Frequency-response samples below are resolvent gains only; true Hinf is undefined/infinite.")
        print(f"largest sampled resolvent gain over [{args.wmin:g}, {args.wmax:g}] = {hall:.12g} at omega={wall:.12g}")
        print(f"  negative sampled peak = {hneg:.12g} at omega={wneg:.12g}")
        print(f"  positive sampled peak = {hpos:.12g} at omega={wpos:.12g}")

    wmodal = float(lead.imag)
    for om in [wmodal, -wmodal, 0.0, -0.65, 0.65]:
        if args.wmin <= om <= args.wmax:
            print(f"  sigma_max(G(i*{om:.8g})) = {sigma_at(A, B, C, om):.12g}")

    if args.csv:
        base = Path(args.csv)
        if args.both_disturbances:
            out = base.with_name(base.stem + f"_{disturbance}" + base.suffix)
        else:
            out = base
        np.savetxt(out, np.column_stack([ws, sig]), delimiter=",", header="omega,sigma_max", comments="")
        print(f"saved frequency sweep CSV: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", required=True, help="Path to ChenRowley.py / chen_rowley_2011_reproduce.py")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--subcritical", action="store_true", help="Use mu0=0.96 mu_c. Required for finite open-loop H2/Hinf.")
    parser.add_argument("--disturbance", choices=["everywhere", "upstream"], default="everywhere")
    parser.add_argument("--both-disturbances", action="store_true")
    parser.add_argument("--wmin", type=float, default=-5.0)
    parser.add_argument("--wmax", type=float, default=5.0)
    parser.add_argument("--ngrid", type=int, default=2001)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--csv", help="Optional CSV output path for omega,sigma_max")
    args = parser.parse_args()

    # Avoid pathological oversubscription on clusters.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cr = import_module_from_path(args.module)
    dists = ["everywhere", "upstream"] if args.both_disturbances else [args.disturbance]
    for dist in dists:
        run_case(cr, args, dist)


if __name__ == "__main__":
    main()
