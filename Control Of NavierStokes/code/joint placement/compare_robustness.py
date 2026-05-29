#!/usr/bin/env python3
"""
Modal truncation + reduced-order Hinf controller search for the Chen/Rowley
Ginzburg--Landau FE model used in yo.py.

Purpose
-------
1. Assemble the same FE model as yo.py.
2. Compute generalized global modes A v = lambda M v.
3. Build M-orthonormal modal subspaces from the least stable modes.
4. Check convergence of the open-loop FE energy-norm Hinf(q->q) on truncated models.
5. Pick an active modal dimension.
6. Optimize actuator/sensor placement + a low-order dynamic SISO controller on the
   truncated model with PyGRANSO, using only:
       objective: signed-frequency Hinf(q->q)
       constraint: closed-loop alpha + stability_margin <= 0
   No Ms/Mt/KS robustness constraints are imposed in this script.

Run
---
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python yo_modal_truncation_hinf_reduced.py

Notes
-----
This is intentionally dense/serial for the truncation and reduced optimization.
Run with one MPI rank. The full FE model may still be assembled with dolfinx/PETSc.
"""

import csv
import time
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar

import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

import ufl
from dolfinx import mesh, fem
from petsc4py import PETSc
from mpi4py import MPI

# =============================================================================
# User knobs
# =============================================================================

# FE model: same defaults as yo.py
MU_0 = 0.38
MU_2 = -0.01
C_D = -1.0
C_U = 0.2
U_MEAN = 2.0
X_MIN = -56.06
X_MAX = 56.06
NX = 400
FE_ORDER = 2
SIGMA_GAUSS = 0.4

# Modal truncation
MODE_COUNTS = [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80]
MAX_EIG_MODES = max(MODE_COUNTS)
ACTIVE_REL_TOL = 0.02       # choose smallest k within 2% of largest tested k
MIN_ACTIVE_K = 6

# Reduced Hinf frequency scan
WMIN = 1e-4
WMAX = 20.0
NGRID = 180
NREFINE = 8

# PyGRANSO reduced Hinf optimization
PLACEMENT_BOUNDS = (-5.0, 5.0)       # bounds for xa and xs
K_BOUNDS = (-200.0, 200.0)           # larger than the robust-constrained run
STABILITY_MARGIN = 1e-4
MAXIT = 60
CTRL_ORDER_MODE = "auto"             # "auto" or integer below
CTRL_ORDER_CAP = 16                   # if auto, ctrl order = min(active_k, cap)
FD_STEP = 1e-4
PRINT_LEVEL = 1

# Initial point: old Hinf-only static point is a good warm start
XA_INIT = -0.6783155331
XS_INIT = +0.6773918546
K_INIT = -83.923781394

# If controller order is 4, this is the previous stable denominator seed.
A4_INIT = np.array([4.4, 16.3, 18.6, 7.7], dtype=float)

# =============================================================================
# Utilities
# =============================================================================

def require_serial(comm):
    if comm.size != 1:
        raise RuntimeError(
            "This reduced modal script is dense/serial. Run with one MPI rank."
        )


def petsc_mat_to_numpy(A: PETSc.Mat) -> np.ndarray:
    """Serial PETSc Mat -> dense NumPy array."""
    require_serial(A.comm)
    A = A.convert("aij")
    A.assemble()
    n, m = A.getSize()
    rows = np.arange(n, dtype=PETSc.IntType)
    cols = np.arange(m, dtype=PETSc.IntType)
    return np.asarray(A.getValues(rows, cols), dtype=np.complex128)


def petsc_vec_to_numpy(v: PETSc.Vec) -> np.ndarray:
    require_serial(v.comm)
    return np.asarray(v.getArray(readonly=True), dtype=np.complex128).copy()


def tanh_to_interval(eta, lo, hi):
    eta = float(eta)
    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    t = np.tanh(eta)
    return mid + rad * t


def interval_to_tanh(x, lo, hi, eps=1e-12):
    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    y = (float(x) - mid) / rad
    y = np.clip(y, -1.0 + eps, 1.0 - eps)
    return float(np.arctanh(y))


def signed_log_grid(wmin=WMIN, wmax=WMAX, ngrid=NGRID):
    wp = np.logspace(np.log10(wmin), np.log10(wmax), int(ngrid))
    return np.concatenate([-wp[::-1], [0.0], wp])


def stable_companion_coeffs(r, pole_min=0.4, pole_max=3.0):
    """
    Return a[0:r] for companion L with characteristic
        lambda^r + a[r-1] lambda^{r-1} + ... + a[0].
    """
    poles = -np.linspace(pole_min, pole_max, int(r))
    poly = np.poly(poles)  # [1, c_{r-1}, ..., c0]
    return np.asarray(poly[1:][::-1], dtype=float)


def companion_matrix(a):
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    r = a.size
    L = np.zeros((r, r), dtype=np.complex128)
    if r > 1:
        L[:-1, 1:] = np.eye(r - 1)
    L[-1, :] = -a
    return L


def modal_hinf_qtoq(Ared, wmin=WMIN, wmax=WMAX, ngrid=NGRID, nrefine=NREFINE):
    """Energy-norm q->q Hinf for xdot=Ared x + d, z=x."""
    n = Ared.shape[0]
    I = np.eye(n, dtype=np.complex128)

    def gain(w):
        X = la.solve(1j * float(w) * I - Ared, I, assume_a="gen")
        return float(la.svdvals(X)[0])

    ws = signed_log_grid(wmin, wmax, ngrid)
    vals = np.array([gain(w) for w in ws], dtype=float)
    order = np.argsort(vals)[::-1]

    best_w = float(ws[order[0]])
    best_g = float(vals[order[0]])

    for idx in order[:min(nrefine, len(ws))]:
        idx = int(idx)
        lo = float(ws[max(0, idx - 1)])
        hi = float(ws[min(len(ws) - 1, idx + 1)])
        if hi > lo + 1e-15:
            res = minimize_scalar(lambda ww: -gain(ww), bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-8, "maxiter": 80})
            g = float(-res.fun)
            w = float(res.x)
            if g > best_g:
                best_g = g
                best_w = w

    neg_peak = float(np.max(vals[ws < 0])) if np.any(ws < 0) else np.nan
    pos_peak = float(np.max(vals[ws > 0])) if np.any(ws > 0) else np.nan
    return best_g, best_w, neg_peak, pos_peak


# =============================================================================
# FE assembly, same model as yo.py
# =============================================================================

def assemble_fe_model():
    comm = MPI.COMM_WORLD
    require_serial(comm)

    domain = mesh.create_interval(comm, NX, [X_MIN, X_MAX])
    x = ufl.SpatialCoordinate(domain)

    gamma = fem.Constant(domain, PETSc.ScalarType(1.0 + 1j * C_D))
    nu = fem.Constant(domain, PETSc.ScalarType(U_MEAN + 2j * C_U))
    mu_x = (MU_0 - C_U**2) + (MU_2 * x[0] ** 2) / 2.0

    V = fem.functionspace(domain, ("CG", FE_ORDER))
    q = ufl.TrialFunction(V)
    phi = ufl.TestFunction(V)

    def boundary(xx):
        return np.isclose(xx[0], X_MIN) | np.isclose(xx[0], X_MAX)

    bc = fem.dirichletbc(
        PETSc.ScalarType(0.0 + 0.0j),
        fem.locate_dofs_geometrical(V, boundary),
        V,
    )

    m_form = ufl.inner(q, phi) * ufl.dx
    a_form = (
        nu * ufl.inner(q, ufl.grad(phi)[0]) * ufl.dx
        + mu_x * ufl.inner(q, phi) * ufl.dx
        - gamma * ufl.inner(ufl.grad(q)[0], ufl.grad(phi)[0]) * ufl.dx
    )

    A_full = fem.petsc.assemble_matrix(fem.form(a_form), bcs=[bc])
    A_full.assemble()
    M_full = fem.petsc.assemble_matrix(fem.form(m_form), bcs=[bc])
    M_full.assemble()

    bc_dofs = np.asarray(bc.dof_indices()[0], dtype=PETSc.IntType)
    nfull = M_full.getSize()[0]
    all_dofs = np.arange(nfull, dtype=PETSc.IntType)
    mask = np.ones(nfull, dtype=bool)
    mask[bc_dofs] = False
    free = all_dofs[mask]
    is_free = PETSc.IS().createGeneral(free, comm=comm)

    A = A_full.createSubMatrix(is_free, is_free)
    M = M_full.createSubMatrix(is_free, is_free)
    A.assemble()
    M.assemble()

    return domain, V, phi, bc, free, A, M


def gaussian_function(Vspace, x0, sig):
    g = fem.Function(Vspace)
    g.interpolate(lambda xx: np.exp(-0.5 * ((xx[0] - float(x0)) / float(sig)) ** 2))
    return g


def actuator_vector_numpy(xa, V, phi, bc, free):
    b_fun = gaussian_function(V, xa, SIGMA_GAUSS)
    b_full = fem.petsc.assemble_vector(fem.form(b_fun * ufl.conj(phi) * ufl.dx))
    b_full.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b_full, [bc])
    b_arr = petsc_vec_to_numpy(b_full)
    return b_arr[np.asarray(free, dtype=int)]


def sensor_c_vector_numpy(xs, V, M_np, free):
    s_fun = gaussian_function(V, xs, SIGMA_GAUSS)
    s_full = s_fun.x.petsc_vec.copy()
    s_full.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    s_arr = petsc_vec_to_numpy(s_full)
    s_red = s_arr[np.asarray(free, dtype=int)]
    return M_np @ s_red


# =============================================================================
# Modal basis and reduced controller model
# =============================================================================

def compute_rightmost_eigs(A_np, M_np):
    """Dense generalized eigendecomposition sorted by decreasing real part."""
    evals, Vraw = la.eig(A_np, M_np)
    order = np.argsort(-np.real(evals))
    return evals[order], Vraw[:, order]


def make_modal_basis_from_raw(A_np, M_np, Vraw_sorted, k):
    """
    Build an M-orthonormal basis from the first k sorted right eigenvectors.
    The subspace is span{Vraw_sorted[:, :k]}.
    """
    Vsub = Vraw_sorted[:, :int(k)]
    G = Vsub.conj().T @ M_np @ Vsub
    G = 0.5 * (G + G.conj().T)
    d, U = la.eigh(G)
    if np.min(d) <= 1e-12:
        raise RuntimeError(f"Modal Gramian is ill-conditioned for k={k}: min eig={np.min(d)}")
    Phi = Vsub @ U @ np.diag(1.0 / np.sqrt(d))
    Ar = Phi.conj().T @ A_np @ Phi
    return Phi, Ar

def reduced_plant_at_placement(Phi_k, Ar_k, xa, xs, V, phi, bc, free, M_np):
    b = actuator_vector_numpy(xa, V, phi, bc, free)
    c = sensor_c_vector_numpy(xs, V, M_np, free)
    br = Phi_k.conj().T @ b
    cr = Phi_k.conj().T @ c
    return Ar_k, br.reshape(-1), cr.reshape(-1)


def make_augmented_closed_loop(Ar, br, cr, a, nvec, K):
    """
    Positive-feedback convention matching yo.py:
        Acl = [[Ar + br*K*cr^H, br*n^H],
               [e_r*cr^H,       L(a)   ]]
    """
    m = Ar.shape[0]
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    nvec = np.asarray(nvec, dtype=np.complex128).reshape(-1)
    r = a.size
    br = np.asarray(br, dtype=np.complex128).reshape(m)
    cr = np.asarray(cr, dtype=np.complex128).reshape(m)

    A11 = Ar + float(K) * np.outer(br, cr.conj())
    A12 = np.outer(br, nvec.conj())
    er = np.zeros(r, dtype=np.complex128)
    er[-1] = 1.0
    A21 = np.outer(er, cr.conj())
    A22 = companion_matrix(a)

    return np.block([[A11, A12], [A21, A22]])


def closed_loop_alpha(Acl):
    return float(np.max(np.real(la.eigvals(Acl))))


def closed_loop_hinf_qtoq(Acl, m, wmin=WMIN, wmax=WMAX, ngrid=NGRID, nrefine=NREFINE):
    """
    Hinf from plant-coordinate energy input d to plant state x, for
    z = x and augmented state [x; xc].
    """
    naug = Acl.shape[0]
    Iaug = np.eye(naug, dtype=np.complex128)
    B = np.zeros((naug, m), dtype=np.complex128)
    B[:m, :] = np.eye(m, dtype=np.complex128)

    def gain(w):
        X = la.solve(1j * float(w) * Iaug - Acl, B, assume_a="gen")
        G = X[:m, :]
        return float(la.svdvals(G)[0])

    ws = signed_log_grid(wmin, wmax, ngrid)
    vals = np.array([gain(w) for w in ws], dtype=float)
    order = np.argsort(vals)[::-1]
    best_w = float(ws[order[0]])
    best_g = float(vals[order[0]])

    for idx in order[:min(nrefine, len(ws))]:
        idx = int(idx)
        lo = float(ws[max(0, idx - 1)])
        hi = float(ws[min(len(ws) - 1, idx + 1)])
        if hi > lo + 1e-15:
            res = minimize_scalar(lambda ww: -gain(ww), bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-8, "maxiter": 80})
            g = float(-res.fun)
            w = float(res.x)
            if g > best_g:
                best_g, best_w = g, w

    neg_peak = float(np.max(vals[ws < 0])) if np.any(ws < 0) else np.nan
    pos_peak = float(np.max(vals[ws > 0])) if np.any(ws > 0) else np.nan
    return best_g, best_w, neg_peak, pos_peak


# =============================================================================
# PyGRANSO reduced Hinf-only optimizer
# =============================================================================

class ReducedHinfDesignProblem:
    def __init__(self, Phi, Ar, V, phi, bc, free, M_np, ctrl_order):
        self.Phi = Phi
        self.Ar = Ar
        self.V = V
        self.phi = phi
        self.bc = bc
        self.free = free
        self.M_np = M_np
        self.m = Ar.shape[0]
        self.r = int(ctrl_order)
        self.nvar = 2 + self.r + self.r + 1
        self.history = []
        self.best_feasible = None

    def decode(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)
        xa = tanh_to_interval(z[0], *PLACEMENT_BOUNDS)
        xs = tanh_to_interval(z[1], *PLACEMENT_BOUNDS)
        a = z[2:2 + self.r]
        nvec = z[2 + self.r:2 + 2 * self.r]
        K = tanh_to_interval(z[-1], *K_BOUNDS)
        return xa, xs, a, nvec, K

    def evaluate(self, z, fine=False):
        xa, xs, a, nvec, K = self.decode(z)
        Ar, br, cr = reduced_plant_at_placement(
            self.Phi, self.Ar, xa, xs, self.V, self.phi, self.bc, self.free, self.M_np
        )
        Acl = make_augmented_closed_loop(Ar, br, cr, a, nvec, K)
        alpha = closed_loop_alpha(Acl)
        if fine:
            h, w, hn, hp = closed_loop_hinf_qtoq(Acl, self.m, ngrid=max(400, NGRID), nrefine=12)
        else:
            h, w, hn, hp = closed_loop_hinf_qtoq(Acl, self.m)
        return {
            "f": float(h),
            "omega": float(w),
            "neg_peak": float(hn),
            "pos_peak": float(hp),
            "alpha": float(alpha),
            "c": float(alpha + STABILITY_MARGIN),
            "xa": float(xa),
            "xs": float(xs),
            "K": float(K),
            "a": np.asarray(a, dtype=float).copy(),
            "n": np.asarray(nvec, dtype=float).copy(),
        }

    def finite_diff_grad(self, z, which="f"):
        z = np.asarray(z, dtype=float).reshape(-1)
        g = np.zeros_like(z)
        for j in range(z.size):
            h = FD_STEP * max(1.0, abs(z[j]))
            zp = z.copy(); zm = z.copy()
            zp[j] += h; zm[j] -= h
            fp = self.evaluate(zp)[which]
            fm = self.evaluate(zm)[which]
            g[j] = (fp - fm) / (2.0 * h)
        return g

    def combined_fn(self, X):
        z = X.z.detach().cpu().numpy().reshape(-1)
        try:
            out = self.evaluate(z)
            f = out["f"]
            c = out["c"]
            gf = self.finite_diff_grad(z, "f")
            gc = self.finite_diff_grad(z, "c")
        except Exception as exc:
            # Fail safely; give PyGRANSO a large value.
            f = 1e12
            c = 1e6
            gf = np.zeros_like(z)
            gc = np.zeros_like(z)
            out = {"f": f, "omega": np.nan, "alpha": np.nan, "c": c,
                   "xa": np.nan, "xs": np.nan, "K": np.nan,
                   "a": np.full(self.r, np.nan), "n": np.full(self.r, np.nan),
                   "error": repr(exc)}

        rec = dict(out)
        rec["call"] = len(self.history) + 1
        self.history.append(rec)
        if np.isfinite(c) and c <= 0.0:
            if self.best_feasible is None or f < self.best_feasible["f"]:
                self.best_feasible = rec

        if MPI.COMM_WORLD.rank == 0:
            print(
                f"[red-hinf] call={rec['call']:03d} f={f:.6e} "
                f"alpha={out['alpha']:+.3e} xa={out['xa']:+.4f} xs={out['xs']:+.4f} "
                f"K={out['K']:+.4e} omega={out['omega']:+.4e} c={c:+.3e}",
                flush=True,
            )

        dev = X.z.device
        dt = X.z.dtype
        f_t = float(f)
        gf_t = torch.tensor(gf.reshape(-1, 1), device=dev, dtype=dt)
        ci_t = torch.tensor([[float(c)]], device=dev, dtype=dt)
        gci_t = torch.tensor(gc.reshape(-1, 1), device=dev, dtype=dt)
        return [f_t, gf_t, ci_t, gci_t, None, None]


def write_history_csv(filename, history):
    if MPI.COMM_WORLD.rank != 0:
        return
    if not history:
        return
    keys = ["call", "f", "omega", "neg_peak", "pos_peak", "alpha", "c", "xa", "xs", "K"]
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rec in history:
            w.writerow({k: rec.get(k, np.nan) for k in keys})


def write_best_txt(filename, best):
    if MPI.COMM_WORLD.rank != 0:
        return
    with open(filename, "w") as f:
        if best is None:
            f.write("No feasible design found.\n")
            return
        f.write(f"best f       = {best['f']:.12e}\n")
        f.write(f"best omega   = {best['omega']:.12e}\n")
        f.write(f"best alpha   = {best['alpha']:.12e}\n")
        f.write(f"best xa/xs   = {best['xa']:+.12e}, {best['xs']:+.12e}\n")
        f.write(f"best K       = {best['K']:+.12e}\n")
        f.write("best a       = " + np.array2string(best['a'], precision=12) + "\n")
        f.write("best n       = " + np.array2string(best['n'], precision=12) + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    comm = MPI.COMM_WORLD
    require_serial(comm)

    if comm.rank == 0:
        print("\n" + "=" * 100)
        print("MODAL TRUNCATION + REDUCED HINF-ONLY CONTROLLER SEARCH")
        print("=" * 100)
        print(f"FE: Nx={NX}, order={FE_ORDER}, mu0={MU_0}, sigma={SIGMA_GAUSS}")

    domain, V, phi, bc, free, A_petsc, M_petsc = assemble_fe_model()
    A_np = petsc_mat_to_numpy(A_petsc)
    M_np = petsc_mat_to_numpy(M_petsc)
    n = A_np.shape[0]

    if comm.rank == 0:
        print(f"free dofs = {n}")
        print("Computing dense generalized eigen-decomposition A v = lambda M v ...")

    evals, Vraw_sorted = compute_rightmost_eigs(A_np, M_np)

    if comm.rank == 0:
        with open("modal_rightmost_modes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["j", "real_lambda", "imag_lambda", "abs_lambda"])
            for j in range(min(MAX_EIG_MODES, evals.size)):
                w.writerow([j + 1, np.real(evals[j]), np.imag(evals[j]), abs(evals[j])])
        print("Rightmost eigenvalues written to modal_rightmost_modes.csv")
        print("First 12 rightmost eigenvalues:")
        for j in range(min(12, evals.size)):
            print(f"  {j+1:3d}: lambda={evals[j].real:+.8e}{evals[j].imag:+.8e}j")

    # Modal Hinf convergence table.
    conv_rows = []
    if comm.rank == 0:
        print("\nOpen-loop modal Hinf(q->q) convergence:")
        print("  k, alpha_k, Hinf_k, omega_peak, neg_peak, pos_peak")

    for k in MODE_COUNTS:
        Phi_k, Ar_k = make_modal_basis_from_raw(A_np, M_np, Vraw_sorted, k)
        h, w, hn, hp = modal_hinf_qtoq(Ar_k)
        alpha_k = float(np.max(np.real(la.eigvals(Ar_k))))
        conv_rows.append({"k": k, "alpha": alpha_k, "hinf": h, "omega": w,
                          "neg_peak": hn, "pos_peak": hp})
        if comm.rank == 0:
            print(f"  {k:3d}, {alpha_k:+.6e}, {h:.8e}, {w:+.6e}, {hn:.8e}, {hp:.8e}")

    href = conv_rows[-1]["hinf"]
    active_k = conv_rows[-1]["k"]
    for rec in conv_rows:
        rel = abs(rec["hinf"] - href) / max(abs(href), 1e-30)
        rec["rel_to_max_k"] = rel
        if rec["k"] >= MIN_ACTIVE_K and rel <= ACTIVE_REL_TOL:
            active_k = rec["k"]
            break

    if comm.rank == 0:
        with open("modal_hinf_convergence.csv", "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(conv_rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(conv_rows)
        print("modal_hinf_convergence.csv written")
        print(f"\nSelected active modal dimension k = {active_k} "
              f"(tol={ACTIVE_REL_TOL:.1%} relative to k={MODE_COUNTS[-1]})")

    # Controller order selection.
    if CTRL_ORDER_MODE == "auto":
        ctrl_order = int(min(active_k, CTRL_ORDER_CAP))
    else:
        ctrl_order = int(CTRL_ORDER_MODE)

    Phi, Ar = make_modal_basis_from_raw(A_np, M_np, Vraw_sorted, active_k)

    if ctrl_order == 4:
        a0 = A4_INIT.copy()
    else:
        a0 = stable_companion_coeffs(ctrl_order)
    n0 = np.zeros(ctrl_order, dtype=float)

    z0 = np.concatenate([
        np.array([
            interval_to_tanh(XA_INIT, *PLACEMENT_BOUNDS),
            interval_to_tanh(XS_INIT, *PLACEMENT_BOUNDS),
        ], dtype=float),
        a0,
        n0,
        np.array([interval_to_tanh(K_INIT, *K_BOUNDS)], dtype=float),
    ])

    if comm.rank == 0:
        print("\nReduced Hinf-only PyGRANSO optimization")
        print(f"  active plant modes = {active_k}")
        print(f"  controller order   = {ctrl_order}")
        print(f"  variables          = {z0.size}")
        print(f"  K bounds           = {K_BOUNDS}")
        print(f"  placement bounds   = {PLACEMENT_BOUNDS}")
        print("  constraints        = alpha + stability_margin <= 0")
        print("  no Ms/Mt/KS robustness constraints")

    problem = ReducedHinfDesignProblem(Phi, Ar, V, phi, bc, free, M_np, ctrl_order)

    var_spec = {"z": [z0.size, 1]}
    opts = pygransoStruct()
    opts.torch_device = torch.device("cpu")
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = MAXIT
    opts.print_frequency = 1
    opts.print_level = PRINT_LEVEL
    opts.quadprog_info_msg = False
    opts.x0 = torch.tensor(z0.reshape(-1, 1), dtype=torch.double)

    soln = pygranso(var_spec=var_spec, combined_fn=problem.combined_fn, user_opts=opts)

    best = problem.best_feasible
    if best is not None:
        # Fine reevaluation at the stored best parameters.
        # Reconstruct z from best values is awkward due tanh; use best history record for reporting.
        pass

    write_history_csv("reduced_hinf_pygranso_history.csv", problem.history)
    write_best_txt("reduced_hinf_pygranso_best.txt", best)

    if comm.rank == 0:
        print("\n" + "=" * 100)
        print("REDUCED HINF-ONLY RUN DONE")
        print(f"elapsed = {time.time() - t0:.1f} s")
        if best is None:
            print("No feasible design found.")
        else:
            print(f"best f       = {best['f']:.10e}")
            print(f"best omega   = {best['omega']:+.10e}")
            print(f"best alpha   = {best['alpha']:+.10e}")
            print(f"best xa/xs   = {best['xa']:+.10f}, {best['xs']:+.10f}")
            print(f"best K       = {best['K']:+.10e}")
            print("best a       =", best["a"])
            print("best n       =", best["n"])
        print("Files written:")
        print("  modal_rightmost_modes.csv")
        print("  modal_hinf_convergence.csv")
        print("  reduced_hinf_pygranso_history.csv")
        print("  reduced_hinf_pygranso_best.txt")
        print("=" * 100)


if __name__ == "__main__":
    main()