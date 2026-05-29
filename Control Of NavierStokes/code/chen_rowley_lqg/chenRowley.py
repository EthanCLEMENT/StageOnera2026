#!/usr/bin/env python3
"""
Reproduce key computations from Chen & Rowley (2011),
"H2 optimal actuator and sensor placement in the linearised complex
Ginzburg--Landau system".

What this script does
---------------------
1. Builds the linearised complex Ginzburg--Landau operator with the weighted
   Hermite collocation used in the paper.
2. Verifies the analytical leading eigenvalue lambda_0 against the discrete A.
3. Builds Gaussian actuators/sensors and the LQG/H2 plant.
4. Computes the LQG controller, the closed-loop H2 norm, and an approximate
   H-infinity norm of the LQG-closed-loop system over signed frequencies
   omega in (-infty, +infty).
5. Searches for H2-optimal actuator/sensor placements using the paper's
   analytical gradient of Gamma_2 = ||G||_2^2.
6. Optionally searches for a placement that minimizes the approximate Hinf norm
   of the LQG-closed-loop system. This is an extension: the Chen--Rowley paper
   optimizes H2/LQG placement, not Hinf placement.

Dependencies
------------
    pip install numpy scipy matplotlib

Examples
--------
Fast smoke test:
    python chen_rowley_2011_reproduce_signedfreq.py --N 40 --evaluate-reference --skip-opt

Publication-like SISO H2 optimization, disturbances everywhere:
    python chen_rowley_2011_reproduce_signedfreq.py --N 100 --disturbance everywhere --optimize-h2 --ma 1 --ms 1 --x0 -1 1

Evaluate paper's SISO upstream-disturbance placement and estimate signed-frequency Hinf:
    python chen_rowley_2011_reproduce_signedfreq.py --N 100 --disturbance upstream --xa -8.48 --xs -5.55 --hinf --wmin 1e-4 --wmax 1e4

Two-actuator/two-sensor H2 placement, disturbances everywhere:
    python chen_rowley_2011_reproduce_signedfreq.py --N 100 --disturbance everywhere --optimize-h2 --ma 2 --ms 2 --x0 -4 3 -3 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import (
    eigvals,
    solve,
    solve_continuous_are,
    solve_continuous_lyapunov,
    svdvals,
)
from scipy.optimize import minimize, minimize_scalar

Disturbance = Literal["everywhere", "upstream"]


@dataclass(frozen=True)
class GLParams:
    # Table 1 values in Chen & Rowley (2011)
    U: float = 2.0
    cu: float = 0.2
    cd: float = -1.0
    mu2: float = -0.01
    sigma: float = 0.4
    beta_cost: float = 7.0
    sensor_noise_half: float = 2.0e-4  # V^(1/2)
    xd: float = -11.0
    supercritical_factor: float = 1.03
    subcritical_factor: float = 0.96

    @property
    def gamma(self) -> complex:
        return 1.0 + 1j * self.cd

    @property
    def nu(self) -> complex:
        return self.U + 2j * self.cu

    @property
    def h(self) -> complex:
        return np.sqrt(-2.0 * self.mu2 * self.gamma)

    @property
    def mu_t(self) -> float:
        Umax = self.U + 2.0 * self.cd * self.cu
        return Umax**2 / (4.0 * abs(self.gamma) ** 2)

    @property
    def mu_c(self) -> float:
        return self.mu_t + abs(self.h) * np.cos(np.angle(self.gamma) / 2.0) / 2.0

    def mu0(self, supercritical: bool = True) -> float:
        return (self.supercritical_factor if supercritical else self.subcritical_factor) * self.mu_c

    def leading_eigenvalue_analytical(self, supercritical: bool = True) -> complex:
        mu0 = self.mu0(supercritical)
        return mu0 - self.cu**2 - self.nu**2 / (4.0 * self.gamma) - 0.5 * self.h

    def hermite_scale(self) -> float:
        # The paper states that x_j are roots of H_N(chi x).  chi is complex;
        # using Re(chi) reproduces the stated N=100 extent x in +/-56.06.
        chi = (-self.mu2 / (2.0 * self.gamma)) ** 0.25
        return float(np.real(chi))


def weighted_hermite_differentiation(N: int, b: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted Hermite nodes and first/second differentiation matrices.

    The unscaled roots r_j are the zeros of the physicists' H_N(r).  The physical
    nodes are x_j = r_j / b.  The first derivative matrix is the Weideman--Reddy
    weighted Hermite collocation matrix for functions exp(-r^2/2) p(r).

    For roots of H_N, the diagonal of the weighted first derivative matrix is zero.
    Higher derivative matrices are powers of D1 for this basis.
    """
    r, gh_w = hermgauss(N)
    r = np.asarray(r, dtype=float)
    gh_w = np.asarray(gh_w, dtype=float)

    ii = np.arange(N)
    ri = r[:, None]
    rj = r[None, :]

    # alpha_i/alpha_j * c_i/c_j in a stable form.
    # alpha(r)=exp(-r^2/2).  For Hermite roots, |c_i/c_j| = sqrt(w_j/w_i),
    # where w_j are Gauss-Hermite quadrature weights for exp(-r^2).
    sign = (-1.0) ** (ii[None, :] - ii[:, None])
    C = sign * np.exp(-0.5 * ri**2 + 0.5 * rj**2) * np.sqrt(gh_w[None, :] / gh_w[:, None])

    D_unscaled = np.zeros((N, N), dtype=float)
    offdiag = ~np.eye(N, dtype=bool)
    D_unscaled[offdiag] = C[offdiag] / (ri - rj)[offdiag]
    # diagonal is exactly zero for the weighted Hermite D^(1)

    x = r / b
    D1 = (b * D_unscaled).astype(complex)
    D2 = D1 @ D1
    return x, D1, D2


def trapezoid_mass_matrix(x: np.ndarray) -> np.ndarray:
    """Diagonal trapezoid-rule mass matrix M for a nonuniform grid."""
    w = np.empty_like(x, dtype=float)
    w[0] = 0.5 * (x[1] - x[0])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return np.diag(w)


def build_gl_operator(
    N: int = 100,
    params: GLParams = GLParams(),
    supercritical: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return x, A, M, info for qdot = A q."""
    b = params.hermite_scale()
    x, D1, D2 = weighted_hermite_differentiation(N, b)
    mu0 = params.mu0(supercritical)
    mu = mu0 - params.cu**2 + 0.5 * params.mu2 * x**2
    A = -params.nu * D1 + np.diag(mu) + params.gamma * D2
    M = trapezoid_mass_matrix(x)
    info = {
        "b": b,
        "mu_t": params.mu_t,
        "mu_c": params.mu_c,
        "mu0": mu0,
        "amplification_extent": np.sqrt(-2.0 * (mu0 - params.cu**2) / params.mu2),
        "lambda0_analytical": params.leading_eigenvalue_analytical(supercritical),
    }
    return x, A, M, info


def gaussian(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))


def gaussian_position_derivative(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    g = gaussian(x, x0, sigma)
    return ((x - x0) / sigma**2) * g


def placement_matrices(
    x: np.ndarray,
    M: np.ndarray,
    xa: Sequence[float],
    xs: Sequence[float],
    disturbance: Disturbance,
    params: GLParams = GLParams(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build B1, B2, C1, C2, D12, D21 for the generalized plant."""
    xa = np.atleast_1d(np.asarray(xa, dtype=float))
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    N = x.size
    ma = xa.size
    ms = xs.size

    B2 = np.column_stack([gaussian(x, a, params.sigma) for a in xa]).astype(complex)
    S = np.column_stack([gaussian(x, s, params.sigma) for s in xs]).astype(complex)
    C2 = S.T @ M  # s^T M, as in the paper; s is real-valued here.

    if disturbance == "everywhere":
        Wh = np.eye(N, dtype=complex)
    elif disturbance == "upstream":
        Wh = gaussian(x, params.xd, params.sigma)[:, None].astype(complex)
    else:
        raise ValueError(f"unknown disturbance: {disturbance}")

    B1 = np.hstack([Wh, np.zeros((N, ms), dtype=complex)])

    M_sqrt = np.diag(np.sqrt(np.diag(M).real))
    C1 = np.vstack([
        params.beta_cost * M_sqrt,
        np.zeros((ma, N), dtype=complex),
    ])
    D12 = np.vstack([
        np.zeros((N, ma), dtype=complex),
        np.eye(ma, dtype=complex),
    ])
    D21 = np.hstack([
        np.zeros((ms, Wh.shape[1]), dtype=complex),
        params.sensor_noise_half * np.eye(ms, dtype=complex),
    ])
    return B1, B2, C1, C2, D12, D21


@dataclass
class LQGResult:
    h2: float
    h2_squared: float
    X: np.ndarray
    Y: np.ndarray
    F: np.ndarray
    L: np.ndarray
    ZA: np.ndarray
    ZB: np.ndarray
    ZC: np.ndarray
    B1: np.ndarray
    B2: np.ndarray
    C1: np.ndarray
    C2: np.ndarray
    D12: np.ndarray
    D21: np.ndarray


def lqg_controller_and_norm(
    A: np.ndarray,
    x: np.ndarray,
    M: np.ndarray,
    xa: Sequence[float],
    xs: Sequence[float],
    disturbance: Disturbance,
    params: GLParams = GLParams(),
    balanced_care: bool = False,
) -> LQGResult:
    """Compute LQG controller and H2 norm of the closed-loop map w -> z."""
    B1, B2, C1, C2, D12, D21 = placement_matrices(x, M, xa, xs, disturbance, params)
    ma = B2.shape[1]
    ms = C2.shape[0]
    R = np.eye(ma, dtype=complex)
    V = (params.sensor_noise_half**2) * np.eye(ms, dtype=complex)

    # Riccati equations (3.9a,b)
    X = solve_continuous_are(
        A, B2, C1.conj().T @ C1, R, balanced=balanced_care
    )
    Y = solve_continuous_are(
        A.conj().T, C2.conj().T, B1 @ B1.conj().T, V, balanced=balanced_care
    )
    X = 0.5 * (X + X.conj().T)
    Y = 0.5 * (Y + Y.conj().T)

    F = B2.conj().T @ X  # R = I
    L = Y @ C2.conj().T @ np.linalg.inv(V)

    # LFT (3.11), with controller qhat_dot=(A-BF-LC)qhat+L y, u=-F qhat.
    ZA = np.block([
        [A, -B2 @ F],
        [L @ C2, A - B2 @ F - L @ C2],
    ])
    ZB = np.vstack([B1, L @ D21])
    ZC = np.hstack([C1, -D12 @ F])

    # N-by-N formula (3.14a).  This is much cheaper than solving 2N Lyapunovs.
    Vinv = np.linalg.inv(V)
    h2_sq = np.real(
        np.trace(C1 @ Y @ C1.conj().T)
        + np.trace(Vinv @ C2 @ Y @ X @ Y @ C2.conj().T)
    )
    h2_sq = max(0.0, h2_sq)
    return LQGResult(
        h2=np.sqrt(h2_sq),
        h2_squared=h2_sq,
        X=X,
        Y=Y,
        F=F,
        L=L,
        ZA=ZA,
        ZB=ZB,
        ZC=ZC,
        B1=B1,
        B2=B2,
        C1=C1,
        C2=C2,
        D12=D12,
        D21=D21,
    )


def h2_from_lft(ZA: np.ndarray, ZB: np.ndarray, ZC: np.ndarray) -> float:
    """Independent H2 check from the 2N closed-loop Lyapunov equation."""
    Wc = solve_continuous_lyapunov(ZA, -(ZB @ ZB.conj().T))
    val = np.real(np.trace(ZC @ Wc @ ZC.conj().T))
    return np.sqrt(max(0.0, val))


@dataclass(frozen=True)
class HinfGridResult:
    hinf: float
    omega: float
    neg_hinf: Optional[float]
    neg_omega: Optional[float]
    pos_hinf: Optional[float]
    pos_omega: Optional[float]
    zero_gain: Optional[float]
    omegas: Optional[np.ndarray] = None
    sigmas: Optional[np.ndarray] = None


def hinf_norm_grid(
    ZA: np.ndarray,
    ZB: np.ndarray,
    ZC: np.ndarray,
    wmin: float = 1e-4,
    wmax: float = 1e4,
    ngrid: int = 500,
    refine: bool = True,
    signed: bool = True,
    include_zero: bool = True,
    return_grid: bool = False,
) -> HinfGridResult:
    """Approximate Hinf norm by frequency gridding.

    By default this scans signed frequencies, i.e. omega in
    [-wmax, -wmin] union {0} union [wmin, wmax].  This matters for the
    complex Ginzburg--Landau system because sigma_max(G(i omega)) is not
    generally symmetric in omega.

    Returns an HinfGridResult.  This is a numerical estimate, not a certified
    Hinf algorithm.
    """
    if wmin <= 0 or wmax <= 0 or wmin >= wmax:
        raise ValueError("Require 0 < wmin < wmax.")
    n = ZA.shape[0]
    I = np.eye(n, dtype=complex)

    def sigma_max_w(w: float) -> float:
        G = ZC @ solve(1j * float(w) * I - ZA, ZB, assume_a="gen")
        return float(svdvals(G)[0])

    pos = np.exp(np.linspace(np.log(wmin), np.log(wmax), ngrid))
    if signed:
        parts = [-pos[::-1]]
        if include_zero:
            parts.append(np.array([0.0]))
        parts.append(pos)
        omegas = np.concatenate(parts)
    else:
        parts = []
        if include_zero:
            parts.append(np.array([0.0]))
        parts.append(pos)
        omegas = np.concatenate(parts)

    sigmas = np.array([sigma_max_w(w) for w in omegas])

    def refine_one(k: int) -> tuple[float, float]:
        best_w = float(omegas[k])
        best_s = float(sigmas[k])
        if refine and 0 < k < len(omegas) - 1:
            lo = float(omegas[k - 1])
            hi = float(omegas[k + 1])
            # Avoid refining across the artificial gap around zero.  The zero
            # point is meaningful, but the log grid is discontinuous there.
            if lo < best_w < hi and not (lo < 0.0 < hi):
                res = minimize_scalar(lambda ww: -sigma_max_w(float(ww)), bounds=(lo, hi), method="bounded")
                if res.success and -res.fun > best_s:
                    best_w = float(res.x)
                    best_s = float(-res.fun)
        return best_s, best_w

    k_all = int(np.argmax(sigmas))
    best_s, best_w = refine_one(k_all)

    neg_hinf = neg_omega = None
    if np.any(omegas < 0):
        neg_indices = np.where(omegas < 0)[0]
        k_neg = int(neg_indices[np.argmax(sigmas[neg_indices])])
        neg_hinf, neg_omega = refine_one(k_neg)
        if neg_hinf > best_s:
            best_s, best_w = neg_hinf, neg_omega

    pos_hinf = pos_omega = None
    if np.any(omegas > 0):
        pos_indices = np.where(omegas > 0)[0]
        k_pos = int(pos_indices[np.argmax(sigmas[pos_indices])])
        pos_hinf, pos_omega = refine_one(k_pos)
        if pos_hinf > best_s:
            best_s, best_w = pos_hinf, pos_omega

    zero_gain = None
    if include_zero:
        iz = np.where(omegas == 0.0)[0]
        if iz.size:
            zero_gain = float(sigmas[int(iz[0])])
            if zero_gain > best_s:
                best_s, best_w = zero_gain, 0.0

    return HinfGridResult(
        hinf=float(best_s),
        omega=float(best_w),
        neg_hinf=None if neg_hinf is None else float(neg_hinf),
        neg_omega=None if neg_omega is None else float(neg_omega),
        pos_hinf=None if pos_hinf is None else float(pos_hinf),
        pos_omega=None if pos_omega is None else float(pos_omega),
        zero_gain=zero_gain,
        omegas=omegas if return_grid else None,
        sigmas=sigmas if return_grid else None,
    )


def print_hinf_result(prefix: str, res: HinfGridResult, signed: bool) -> None:
    if signed:
        print(f"{prefix}estimated signed-frequency Hinf={res.hinf:.10g} at omega={res.omega:.10g}")
        if res.neg_hinf is not None:
            print(f"{prefix}  negative-frequency peak={res.neg_hinf:.10g} at omega={res.neg_omega:.10g}")
        if res.pos_hinf is not None:
            print(f"{prefix}  positive-frequency peak={res.pos_hinf:.10g} at omega={res.pos_omega:.10g}")
        if res.zero_gain is not None:
            print(f"{prefix}  sigma_max(G(i*0))={res.zero_gain:.10g}")
    else:
        print(f"{prefix}estimated positive-frequency Hinf={res.hinf:.10g} at omega={res.omega:.10g}")


def save_hinf_csv(path: str, res: HinfGridResult) -> None:
    if res.omegas is None or res.sigmas is None:
        raise ValueError("HinfGridResult was created without return_grid=True.")
    data = np.column_stack([res.omegas, res.sigmas])
    np.savetxt(path, data, delimiter=",", header="omega,sigma_max", comments="")


def h2_squared_and_gradient(
    z: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    M: np.ndarray,
    ma: int,
    ms: int,
    disturbance: Disturbance,
    params: GLParams = GLParams(),
    balanced_care: bool = False,
) -> Tuple[float, np.ndarray]:
    """Gamma_2 and analytical gradient from equations (3.16)--(3.20)."""
    xa = np.asarray(z[:ma], dtype=float)
    xs = np.asarray(z[ma : ma + ms], dtype=float)
    out = lqg_controller_and_norm(A, x, M, xa, xs, disturbance, params, balanced_care)

    X, Y = out.X, out.Y
    B2, C2 = out.B2, out.C2
    ma, ms = B2.shape[1], C2.shape[0]
    V = (params.sensor_noise_half**2) * np.eye(ms, dtype=complex)
    Vinv = np.linalg.inv(V)

    grad = np.zeros(ma + ms, dtype=float)

    # Actuator derivatives: dB2/dxa_j is only column j.
    AF = A - B2 @ B2.conj().T @ X  # A - B R^-1 B^* X, R = I
    for j, a in enumerate(xa):
        dBj = np.zeros_like(B2)
        dBj[:, j] = gaussian_position_derivative(x, a, params.sigma)
        Qx = X @ (dBj @ B2.conj().T + B2 @ dBj.conj().T) @ X
        dX = solve_continuous_lyapunov(AF.conj().T, Qx)
        grad[j] = np.real(np.trace(Vinv @ C2 @ Y @ dX @ Y @ C2.conj().T))

    # Sensor derivatives: dC2/dxs_k is only row k.
    AL = A - Y @ C2.conj().T @ Vinv @ C2
    for k, s in enumerate(xs):
        dC = np.zeros_like(C2)
        dC[k, :] = gaussian_position_derivative(x, s, params.sigma).T @ M
        Qy = Y @ (dC.conj().T @ Vinv @ C2 + C2.conj().T @ Vinv @ dC) @ Y
        dY = solve_continuous_lyapunov(AL, Qy)
        grad[ma + k] = np.real(np.trace(B2.conj().T @ X @ dY @ X @ B2))

    return out.h2_squared, grad


def optimize_h2_placement(
    A: np.ndarray,
    x: np.ndarray,
    M: np.ndarray,
    x0: Sequence[float],
    ma: int,
    ms: int,
    disturbance: Disturbance,
    params: GLParams = GLParams(),
    maxiter: int = 100,
    use_gradient: bool = True,
) -> Tuple[np.ndarray, float, object]:
    """Minimize Gamma_2 = H2^2 over actuator and sensor positions."""
    x0 = np.asarray(x0, dtype=float)
    if x0.size != ma + ms:
        raise ValueError(f"x0 must have length ma+ms={ma+ms}; got {x0.size}")

    def fun(z):
        val, _ = h2_squared_and_gradient(z, A, x, M, ma, ms, disturbance, params)
        if not np.isfinite(val):
            return 1e300
        return val

    def jac(z):
        _, g = h2_squared_and_gradient(z, A, x, M, ma, ms, disturbance, params)
        return g

    if use_gradient:
        res = minimize(fun, x0, jac=jac, method="BFGS", options={"gtol": 1e-5, "maxiter": maxiter})
    else:
        res = minimize(fun, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3})
    return np.asarray(res.x, dtype=float), float(np.sqrt(max(0.0, res.fun))), res


def optimize_hinf_placement_grid(
    A: np.ndarray,
    x: np.ndarray,
    M: np.ndarray,
    x0: Sequence[float],
    ma: int,
    ms: int,
    disturbance: Disturbance,
    params: GLParams = GLParams(),
    maxiter: int = 60,
    ngrid: int = 200,
    wmin: float = 1e-4,
    wmax: float = 1e4,
    signed: bool = True,
) -> Tuple[np.ndarray, float, float, object]:
    """Extension: minimize gridded Hinf norm of the LQG closed loop.

    This is derivative-free and can be slow.  It is not a certified Hinf optimal
    controller or placement computation.
    """
    x0 = np.asarray(x0, dtype=float)

    def fun(z):
        try:
            out = lqg_controller_and_norm(A, x, M, z[:ma], z[ma:], disturbance, params)
            hres = hinf_norm_grid(out.ZA, out.ZB, out.ZC, wmin=wmin, wmax=wmax, ngrid=ngrid, refine=False, signed=signed)
            return hres.hinf
        except Exception:
            return 1e300

    res = minimize(fun, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    out = lqg_controller_and_norm(A, x, M, res.x[:ma], res.x[ma:], disturbance, params)
    hres = hinf_norm_grid(out.ZA, out.ZB, out.ZC, wmin=wmin, wmax=wmax, ngrid=max(ngrid, 500), refine=True, signed=signed)
    return np.asarray(res.x, dtype=float), float(hres.hinf), float(hres.omega), res


REFERENCE_PLACEMENTS = {
    # Values reported in text/figures of Chen & Rowley (2011).
    ("everywhere", 1, 1): {"xa": [-1.03], "xs": [0.98], "h2": 46.1},
    ("upstream", 1, 1): {"xa": [-8.48], "xs": [-5.55], "h2": 3.86},
    ("everywhere", 2, 2): {"xa": [-3.78, 2.71], "xs": [-2.75, 3.74], "h2": 34.0},
}


def print_reference_evaluations(A, x, M, disturbance_filter: Optional[str], params: GLParams, do_hinf: bool, signed_hinf: bool, wmin: float, wmax: float, ngrid: int):
    for (dist, ma, ms), ref in REFERENCE_PLACEMENTS.items():
        if disturbance_filter is not None and dist != disturbance_filter:
            continue
        print(f"\nReference placement: disturbance={dist}, ma={ma}, ms={ms}")
        print(f"  paper xa={ref['xa']}, xs={ref['xs']}, H2≈{ref['h2']}")
        out = lqg_controller_and_norm(A, x, M, ref["xa"], ref["xs"], dist, params)
        print(f"  computed H2={out.h2:.8g}")
        print(f"  closed-loop max Re(lambda)={np.max(eigvals(out.ZA).real):.4e}")
        if do_hinf:
            hres = hinf_norm_grid(out.ZA, out.ZB, out.ZC, wmin=wmin, wmax=wmax, ngrid=ngrid, signed=signed_hinf)
            print_hinf_result("  ", hres, signed_hinf)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N", type=int, default=60, help="Hermite collocation size; use 100 for the paper, 40-70 for faster testing.")
    parser.add_argument("--subcritical", action="store_true", help="Use mu0=0.96 mu_c instead of supercritical mu0=1.03 mu_c.")
    parser.add_argument("--disturbance", choices=["everywhere", "upstream"], default="everywhere")
    parser.add_argument("--ma", type=int, default=1, help="number of actuators for optimization")
    parser.add_argument("--ms", type=int, default=1, help="number of sensors for optimization")
    parser.add_argument("--x0", nargs="*", type=float, help="initial guess: xa entries followed by xs entries")
    parser.add_argument("--xa", nargs="*", type=float, help="actuator positions for direct evaluation")
    parser.add_argument("--xs", nargs="*", type=float, help="sensor positions for direct evaluation")
    parser.add_argument("--optimize-h2", action="store_true")
    parser.add_argument("--no-gradient", action="store_true", help="use Nelder-Mead instead of analytical gradient for H2 optimization")
    parser.add_argument("--optimize-hinf", action="store_true", help="extension: optimize gridded Hinf norm of the LQG closed loop")
    parser.add_argument("--hinf", action="store_true", help="estimate Hinf for direct/reference/optimized evaluations")
    parser.add_argument("--positive-frequencies-only", action="store_true", help="old behavior: estimate Hinf over omega >= 0 only; default scans positive and negative frequencies")
    parser.add_argument("--wmin", type=float, default=1e-4, help="minimum nonzero absolute frequency for Hinf grid")
    parser.add_argument("--wmax", type=float, default=1e4, help="maximum absolute frequency for Hinf grid")
    parser.add_argument("--ngrid", type=int, default=500, help="number of positive log-spaced frequencies for Hinf grid; signed scan uses about 2*ngrid+1 points")
    parser.add_argument("--freq-csv", type=str, default=None, help="optional CSV path for the last Hinf frequency sweep")
    parser.add_argument("--evaluate-reference", action="store_true", help="evaluate paper's reference placements")
    parser.add_argument("--skip-opt", action="store_true", help="do not run optimization even if defaults would otherwise suggest it")
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args(argv)
    signed_hinf = not args.positive_frequencies_only

    params = GLParams()
    x, A, M, info = build_gl_operator(args.N, params, supercritical=not args.subcritical)

    print("Complex Ginzburg--Landau discretization")
    print(f"  N={args.N}, x in [{x[0]:.5g}, {x[-1]:.5g}], b={info['b']:.8g}")
    print(f"  mu_t={info['mu_t']:.8g}, mu_c={info['mu_c']:.8g}, mu0={info['mu0']:.8g}")
    print(f"  amplification region approx +/-{info['amplification_extent']:.5g}")

    lam_num = eigvals(A)
    lead = lam_num[np.argmax(lam_num.real)]
    print("\nLeading eigenvalue check")
    print(f"  analytical lambda0 = {info['lambda0_analytical']:.12g}")
    print(f"  numerical  lambda0 = {lead:.12g}")
    print(f"  absolute error     = {abs(lead - info['lambda0_analytical']):.3e}")

    if args.evaluate_reference:
        print_reference_evaluations(A, x, M, args.disturbance if args.disturbance else None, params, args.hinf, signed_hinf, args.wmin, args.wmax, args.ngrid)

    if args.xa is not None and args.xs is not None:
        out = lqg_controller_and_norm(A, x, M, args.xa, args.xs, args.disturbance, params)
        print(f"\nDirect evaluation: disturbance={args.disturbance}")
        print(f"  xa={args.xa}")
        print(f"  xs={args.xs}")
        print(f"  H2={out.h2:.10g}")
        print(f"  H2 from LFT Lyapunov={h2_from_lft(out.ZA, out.ZB, out.ZC):.10g}")
        print(f"  closed-loop max Re(lambda)={np.max(eigvals(out.ZA).real):.4e}")
        if args.hinf:
            hres = hinf_norm_grid(out.ZA, out.ZB, out.ZC, wmin=args.wmin, wmax=args.wmax, ngrid=args.ngrid, signed=signed_hinf, return_grid=args.freq_csv is not None)
            print_hinf_result("  ", hres, signed_hinf)
            if args.freq_csv is not None:
                save_hinf_csv(args.freq_csv, hres)
                print(f"  saved Hinf frequency sweep CSV: {args.freq_csv}")

    if args.optimize_h2 and not args.skip_opt:
        if args.x0 is None:
            # sensible defaults based on paper's figures
            if args.ma == args.ms == 1:
                args.x0 = [-1.0, 1.0] if args.disturbance == "everywhere" else [-8.5, -5.5]
            elif args.ma == args.ms == 2 and args.disturbance == "everywhere":
                args.x0 = [-4.0, 3.0, -3.0, 4.0]
            else:
                args.x0 = list(np.linspace(-4, 4, args.ma)) + list(np.linspace(-3, 5, args.ms))
        zopt, h2opt, res = optimize_h2_placement(
            A,
            x,
            M,
            args.x0,
            args.ma,
            args.ms,
            args.disturbance,
            params,
            maxiter=args.maxiter,
            use_gradient=not args.no_gradient,
        )
        print(f"\nH2 placement optimization: disturbance={args.disturbance}")
        print(f"  success={res.success}, message={res.message}")
        print(f"  xa={zopt[:args.ma].tolist()}")
        print(f"  xs={zopt[args.ma:].tolist()}")
        print(f"  H2={h2opt:.10g}")
        out = lqg_controller_and_norm(A, x, M, zopt[: args.ma], zopt[args.ma :], args.disturbance, params)
        if args.hinf:
            hres = hinf_norm_grid(out.ZA, out.ZB, out.ZC, wmin=args.wmin, wmax=args.wmax, ngrid=args.ngrid, signed=signed_hinf, return_grid=args.freq_csv is not None)
            print_hinf_result("  ", hres, signed_hinf)
            if args.freq_csv is not None:
                save_hinf_csv(args.freq_csv, hres)
                print(f"  saved Hinf frequency sweep CSV: {args.freq_csv}")

    if args.optimize_hinf and not args.skip_opt:
        if args.x0 is None:
            args.x0 = [-1.0, 1.0] if args.disturbance == "everywhere" else [-8.5, -5.5]
        zopt, hinf, wpeak, res = optimize_hinf_placement_grid(
            A,
            x,
            M,
            args.x0,
            args.ma,
            args.ms,
            args.disturbance,
            params,
            maxiter=args.maxiter,
            ngrid=args.ngrid,
            wmin=args.wmin,
            wmax=args.wmax,
            signed=signed_hinf,
        )
        print(f"\nApprox-Hinf placement optimization of LQG closed loop: disturbance={args.disturbance}")
        print("  Note: extension beyond the H2/LQG optimization in the paper.")
        print(f"  success={res.success}, message={res.message}")
        print(f"  xa={zopt[:args.ma].tolist()}")
        print(f"  xs={zopt[args.ma:].tolist()}")
        print(f"  estimated signed-frequency Hinf={hinf:.10g} at omega={wpeak:.10g}")


if __name__ == "__main__":
    main()
