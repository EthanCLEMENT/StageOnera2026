"""
Append this file after the definitions in yo.py.

Purpose
-------
Run four SISO placement/controller parametrization experiments with PyGRANSO:

    0) K = 0, a = 0, n = 0: placement-only baseline.
       This is intentionally flat because the feedback path is zero.

    1) K free, a = 0, n = 0: static output feedback + placement.
       Implemented with ClosedLoopSOFF so that zero controller dynamics are not
       artificially added as neutral poles.

    2) K = 0, a free, n free: strictly proper dynamic controller + placement.
       Uses ClosedLoopDynamicCompanionSISOStrictlyProper with r = 4.

    3) K free, a free, n free: dynamic controller with feedthrough + placement.
       Uses ClosedLoopDynamicCompanionSISO with r = 4.

Expected symbols from yo.py
---------------------------
A, M, V, phi, bc, is_free, sigma_gauss, x_min, x_max,
build_actuator_vector, build_sensor_state_vector, sensor_output_vector,
ClosedLoopSOFF, ClosedLoopDynamicCompanionSISO,
ClosedLoopDynamicCompanionSISOStrictlyProper, pygranso, pygransoStruct,
MPI, torch, np, plt.
"""

from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from mpi4py import MPI


# -----------------------------------------------------------------------------
# User-tunable experiment settings
# -----------------------------------------------------------------------------

R_PLACEMENT = 4

# Initial placement. These are the feedforward values already used in your file.
XA0 = 1.03
XS0 = -0.98

# Keep placement search in the physical wavemaker region, not the whole domain.
# Widen this if you want to search the entire interval [x_min, x_max].
PLACEMENT_BOUNDS = (-8.0, 8.0)

# Stable starting denominator for the dynamic-controller cases.
# Do not start a free-a dynamic controller at a = 0: the companion block has
# neutral poles there and the augmented closed-loop problem becomes ill-posed.
A_CENTER_STABLE = np.array([4.4, 16.3, 18.6, 7.7], dtype=float)
N_CENTER_ZERO = np.zeros(R_PLACEMENT, dtype=float)

# Previous static value used in yo.py. It is only a starting point for K-free cases.
K_CENTER_GUESS = -1.89522178

# Affine scaling from optimizer z to physical parameters p.
# p = center + scale * z, but only nonzero scale components are active.
XA_SCALE = 2.0
XS_SCALE = 2.0
A_SCALE_FREE = np.maximum(0.15 * np.abs(A_CENTER_STABLE), 0.2)
N_SCALE_FREE = 2.5e-2 * np.ones(R_PLACEMENT)
K_SCALE_FREE = 0.5

# Finite-difference step for placement derivatives in physical x units.
# Controller derivatives are taken analytically from your existing oracle.
FD_X = 1.0e-4

# PyGRANSO/oracle settings. Start conservative; increase maxit later.
OPT_SETTINGS = dict(
    stab_margin=1e-5,
    stab_penalty_weight=1e5,
    num_guess=8,
    wmin=1e-6,
    wmax=2e1,
    base_seed=2,
    candidate_npos=80,
    disk_b=20.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
    maxit=20,
    print_level=1,
    verbose=True,
)


# -----------------------------------------------------------------------------
# Parameter convention
# -----------------------------------------------------------------------------
# The full physical vector is always
#
#     p = [xa, xs, a_0,...,a_{r-1}, n_0,...,n_{r-1}, K]
#
# Static cases ignore a,n. Strictly-proper dynamic cases ignore K.
# Freezing is done by setting the corresponding scale to zero.
# -----------------------------------------------------------------------------


def idx_slices(r=R_PLACEMENT):
    a_sl = slice(2, 2 + r)
    n_sl = slice(2 + r, 2 + 2 * r)
    k_idx = 2 + 2 * r
    return a_sl, n_sl, k_idx


def split_p(p, r=R_PLACEMENT):
    p = np.asarray(p, dtype=float).reshape(-1)
    a_sl, n_sl, k_idx = idx_slices(r)
    xa = float(p[0])
    xs = float(p[1])
    a = p[a_sl].copy()
    n = p[n_sl].copy()
    K = float(p[k_idx])
    return xa, xs, a, n, K


def make_center(xa=XA0, xs=XS0, a=None, n=None, K=0.0, r=R_PLACEMENT):
    if a is None:
        a = np.zeros(r)
    if n is None:
        n = np.zeros(r)
    return np.concatenate([
        np.array([xa, xs], dtype=float),
        np.asarray(a, dtype=float).reshape(r),
        np.asarray(n, dtype=float).reshape(r),
        np.array([K], dtype=float),
    ])


def make_scale(free_x=False, free_a=False, free_n=False, free_K=False, r=R_PLACEMENT):
    return np.concatenate([
        np.array([XA_SCALE if free_x else 0.0,
                  XS_SCALE if free_x else 0.0], dtype=float),
        A_SCALE_FREE if free_a else np.zeros(r),
        N_SCALE_FREE if free_n else np.zeros(r),
        np.array([K_SCALE_FREE if free_K else 0.0], dtype=float),
    ])


@dataclass
class PlacementCase:
    name: str
    mode: str          # "static", "strict", or "full"
    center: np.ndarray
    scale: np.ndarray
    note: str = ""


CASES = [
    PlacementCase(
        name="00_baseline_K0_a0_n0_place_only",
        mode="static",
        center=make_center(K=0.0),
        scale=make_scale(free_x=True),
        note="No feedback path: objective is independent of placement; use as open-loop baseline.",
    ),
    PlacementCase(
        name="01_static_K_free_place",
        mode="static",
        center=make_center(K=K_CENTER_GUESS),
        scale=make_scale(free_x=True, free_K=True),
        note="Static output feedback, using ClosedLoopSOFF so no neutral controller states are added.",
    ),
    PlacementCase(
        name="02_strict_K0_a_free_n_free_place",
        mode="strict",
        center=make_center(a=A_CENTER_STABLE, n=N_CENTER_ZERO, K=0.0),
        scale=make_scale(free_x=True, free_a=True, free_n=True),
        note="Strictly proper dynamic controller: K fixed to zero.",
    ),
    PlacementCase(
        name="03_full_K_free_a_free_n_free_place",
        mode="full",
        center=make_center(a=A_CENTER_STABLE, n=N_CENTER_ZERO, K=K_CENTER_GUESS),
        scale=make_scale(free_x=True, free_a=True, free_n=True, free_K=True),
        note="Dynamic controller with direct feedthrough K.",
    ),
]


# -----------------------------------------------------------------------------
# Closed-loop construction at a given placement
# -----------------------------------------------------------------------------


def build_b_c_for_placement(xa, xs):
    b_red, _ = build_actuator_vector(xa, sigma_gauss, V, phi, bc, is_free)
    s_red, _ = build_sensor_state_vector(xs, sigma_gauss, V, is_free)
    c_red = sensor_output_vector(M, s_red)
    return b_red, c_red


def make_closed_loop(case, p):
    xa, xs, a, n, K = split_p(p, R_PLACEMENT)
    b_red, c_red = build_b_c_for_placement(xa, xs)

    if case.mode == "static":
        cl = ClosedLoopSOFF(A, M, b_red, c_red, feedback_sign=+1.0)
        theta = K

    elif case.mode == "strict":
        cl = ClosedLoopDynamicCompanionSISOStrictlyProper(
            A=A, M=M, b=b_red, c=c_red,
            r=R_PLACEMENT,
            optimize_mc_gain=False,
            fixed_mc_gain=1.0,
        )
        theta = np.concatenate([a, n])

    elif case.mode == "full":
        cl = ClosedLoopDynamicCompanionSISO(
            A=A, M=M, b=b_red, c=c_red,
            r=R_PLACEMENT,
            optimize_mc_gain=False,
            fixed_mc_gain=1.0,
        )
        theta = np.concatenate([a, n, np.array([K])])

    else:
        raise ValueError(f"Unknown case.mode={case.mode!r}")

    return cl, theta


# -----------------------------------------------------------------------------
# Oracle evaluation and gradients
# -----------------------------------------------------------------------------


def eval_oracle_physical(case, p, seed, settings=OPT_SETTINGS):
    cl, theta = make_closed_loop(case, p)

    if case.mode == "static":
        out = cl.objective_constraint_oracle_newton_disk(
            float(theta),
            stab_margin=settings["stab_margin"],
            num_guess=settings["num_guess"],
            wmin=settings["wmin"],
            wmax=settings["wmax"],
            seed=seed,
            disk_b=settings["disk_b"],
            disk_eigs_tol=settings["disk_eigs_tol"],
            disk_nev=settings["disk_nev"],
            disk_eps_probe=settings["disk_eps_probe"],
            disk_axis_warn=settings["disk_axis_warn"],
        )
    else:
        out = cl.objective_constraint_oracle_newton_disk(
            theta,
            stab_margin=settings["stab_margin"],
            num_guess=settings["num_guess"],
            wmin=settings["wmin"],
            wmax=settings["wmax"],
            seed=seed,
            candidate_npos=settings["candidate_npos"],
            disk_b=settings["disk_b"],
            disk_eigs_tol=settings["disk_eigs_tol"],
            disk_nev=settings["disk_nev"],
            disk_eps_probe=settings["disk_eps_probe"],
            disk_axis_warn=settings["disk_axis_warn"],
        )

    return out


def one_sided_or_central_fd(case, p, j, seed, settings=OPT_SETTINGS, h=FD_X):
    """Finite-difference df/dp_j and dc/dp_j for placement variables."""
    p = np.asarray(p, dtype=float).copy()
    lo, hi = PLACEMENT_BOUNDS

    # Respect placement bounds for xa/xs with one-sided differences near bounds.
    hp = h
    hm = h
    pp = p.copy()
    pm = p.copy()
    pp[j] += hp
    pm[j] -= hm

    if j in (0, 1):
        if pp[j] > hi:
            pp[j] = p[j]
            pm[j] = p[j] - h
            out0 = eval_oracle_physical(case, p, seed, settings)
            outm = eval_oracle_physical(case, pm, seed, settings)
            df = (float(out0["f"]) - float(outm["f"])) / h
            dc = (float(out0["c"]) - float(outm["c"])) / h
            return df, dc
        if pm[j] < lo:
            pp[j] = p[j] + h
            pm[j] = p[j]
            outp = eval_oracle_physical(case, pp, seed, settings)
            out0 = eval_oracle_physical(case, p, seed, settings)
            df = (float(outp["f"]) - float(out0["f"])) / h
            dc = (float(outp["c"]) - float(out0["c"])) / h
            return df, dc

    outp = eval_oracle_physical(case, pp, seed, settings)
    outm = eval_oracle_physical(case, pm, seed, settings)
    df = (float(outp["f"]) - float(outm["f"])) / (2.0 * h)
    dc = (float(outp["c"]) - float(outm["c"])) / (2.0 * h)
    return df, dc


def physical_gradients(case, p, out0, seed, settings=OPT_SETTINGS):
    """
    Gradient with respect to the full physical vector p.

    Placement entries xa,xs use finite differences.
    Controller entries use your existing analytic oracle gradients.
    """
    r = R_PLACEMENT
    a_sl, n_sl, k_idx = idx_slices(r)

    grad_f = np.zeros(2 + 2 * r + 1, dtype=float)
    grad_c = np.zeros_like(grad_f)

    # Placement gradients by finite difference.
    # These are the only new derivatives needed for placement optimization.
    for j in (0, 1):
        grad_f[j], grad_c[j] = one_sided_or_central_fd(case, p, j, seed, settings)

    # Controller gradients from the existing oracle.
    if case.mode == "static":
        grad_f[k_idx] = float(out0.get("dfdk", 0.0))
        grad_c[k_idx] = float(out0.get("dcdk", 0.0))

    elif case.mode == "strict":
        g_f = np.asarray(out0["dfdtheta"], dtype=float).reshape(-1)
        g_c = np.asarray(out0["dcdtheta"], dtype=float).reshape(-1)
        grad_f[a_sl] = g_f[:r]
        grad_f[n_sl] = g_f[r:2 * r]
        grad_c[a_sl] = g_c[:r]
        grad_c[n_sl] = g_c[r:2 * r]

    elif case.mode == "full":
        g_f = np.asarray(out0["dfdtheta"], dtype=float).reshape(-1)
        g_c = np.asarray(out0["dcdtheta"], dtype=float).reshape(-1)
        grad_f[a_sl] = g_f[:r]
        grad_f[n_sl] = g_f[r:2 * r]
        grad_f[k_idx] = g_f[2 * r]
        grad_c[a_sl] = g_c[:r]
        grad_c[n_sl] = g_c[r:2 * r]
        grad_c[k_idx] = g_c[2 * r]

    return grad_f, grad_c


# -----------------------------------------------------------------------------
# PyGRANSO wrapper with placement variables and placement bounds
# -----------------------------------------------------------------------------


def best_feasible_placement(history):
    feasible = [rec for rec in history if np.all(np.asarray(rec["ci"]) <= 0.0)]
    if len(feasible) == 0:
        return None
    return min(feasible, key=lambda rec: rec["f_raw"])


def make_pygranso_fn_placement(case, settings=OPT_SETTINGS):
    center = np.asarray(case.center, dtype=float).reshape(-1)
    scale = np.asarray(case.scale, dtype=float).reshape(-1)
    active = np.flatnonzero(np.abs(scale) > 0.0)

    if active.size == 0:
        raise ValueError(f"Case {case.name} has no active variables.")

    history = []
    state = {"calls": 0}

    x_lo, x_hi = PLACEMENT_BOUNDS

    def z_to_p(z):
        p = center.copy()
        p[active] = center[active] + scale[active] * z
        return p

    def combined_fn(X_struct):
        state["calls"] += 1
        z_torch = X_struct.z
        z = z_torch.detach().cpu().numpy().reshape(-1)
        p = z_to_p(z)

        seed = settings["base_seed"] + state["calls"]
        out0 = eval_oracle_physical(case, p, seed, settings)

        f_raw = float(out0["f"])
        c_stab = float(out0["c"])
        grad_p_f, grad_p_c = physical_gradients(case, p, out0, seed, settings)

        # Chain rule p = center + scale*z.
        grad_z_f = grad_p_f[active] * scale[active]
        grad_z_c = grad_p_c[active] * scale[active]

        # Penalty on stability constraint to help PyGRANSO early on.
        c_plus = max(c_stab, 0.0)
        f_return = f_raw + settings["stab_penalty_weight"] * c_plus**2
        grad_z_return = grad_z_f.copy()
        if c_stab > 0.0:
            grad_z_return += 2.0 * settings["stab_penalty_weight"] * c_stab * grad_z_c

        # Inequality constraints ci <= 0:
        #   0: spectral abscissa margin
        #   1-4: placement box constraints
        ci_vals = np.array([
            c_stab,
            p[0] - x_hi,
            x_lo - p[0],
            p[1] - x_hi,
            x_lo - p[1],
        ], dtype=float)

        grad_p_ci = np.zeros((center.size, ci_vals.size), dtype=float)
        grad_p_ci[:, 0] = grad_p_c
        grad_p_ci[0, 1] = +1.0
        grad_p_ci[0, 2] = -1.0
        grad_p_ci[1, 3] = +1.0
        grad_p_ci[1, 4] = -1.0
        grad_z_ci = grad_p_ci[active, :] * scale[active, None]

        xa, xs, a, n, K = split_p(p, R_PLACEMENT)
        rec = dict(
            case=case.name,
            mode=case.mode,
            call=state["calls"],
            z=z.copy(),
            p=p.copy(),
            xa=xa,
            xs=xs,
            a=a.copy(),
            n=n.copy(),
            K=K,
            f_raw=f_raw,
            f_return=float(f_return),
            alpha=float(out0["alpha"]),
            c=c_stab,
            ci=ci_vals.copy(),
            omega_star=float(out0["omega_star"]),
            certified=bool(out0["hinf_info"]["certified"]),
            cert_info=out0["hinf_info"]["cert_info"],
            note=case.note,
        )
        history.append(rec)

        if settings["verbose"] and MPI.COMM_WORLD.rank == 0:
            print(
                f"[{case.name}] call={state['calls']:03d} "
                f"f={f_raw:.8e} f_ret={f_return:.8e} "
                f"alpha={rec['alpha']:+.4e} c={c_stab:+.4e} "
                f"xa={xa:+.4f} xs={xs:+.4f} K={K:+.4e} "
                f"omega*={rec['omega_star']:+.4e} cert={rec['certified']}"
            )

        dev = z_torch.device
        dt = z_torch.dtype

        f = float(f_return)
        f_grad = torch.tensor(grad_z_return.reshape(-1, 1), device=dev, dtype=dt)
        ci = torch.tensor(ci_vals.reshape(-1, 1), device=dev, dtype=dt)
        ci_grad = torch.tensor(grad_z_ci, device=dev, dtype=dt)

        return [f, f_grad, ci, ci_grad, None, None]

    combined_fn.history = history
    combined_fn.active = active
    combined_fn.center = center
    combined_fn.scale = scale
    return combined_fn


def run_case(case, settings=OPT_SETTINGS):
    device = torch.device("cpu")
    dtype = torch.double

    comb_fn = make_pygranso_fn_placement(case, settings)
    nvar = comb_fn.active.size

    opts = pygransoStruct()
    opts.torch_device = device
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = settings["maxit"]
    opts.print_frequency = 1
    opts.print_level = settings["print_level"]
    opts.quadprog_info_msg = False
    opts.x0 = torch.zeros((nvar, 1), device=device, dtype=dtype)

    if MPI.COMM_WORLD.rank == 0:
        print("\n" + "=" * 100)
        print(f"Running {case.name}")
        print(case.note)
        print("active physical indices:", comb_fn.active)
        print("center:", comb_fn.center)
        print("scale :", comb_fn.scale)

    soln = pygranso(var_spec={"z": [nvar, 1]}, combined_fn=comb_fn, user_opts=opts)
    best = best_feasible_placement(comb_fn.history)

    # If there is no feasible iterate, keep the least penalized one for diagnostics.
    if best is None and len(comb_fn.history) > 0:
        best = min(comb_fn.history, key=lambda rec: rec["f_return"])
        best = dict(best)
        best["warning"] = "No feasible iterate found; this is the least penalized iterate."

    return dict(case=case, soln=soln, history=comb_fn.history, best=best)


def run_all_placement_cases(cases=CASES, settings=OPT_SETTINGS):
    results = []
    for case in cases:
        results.append(run_case(case, settings))
    return results


# -----------------------------------------------------------------------------
# Plotting and diagnostics
# -----------------------------------------------------------------------------


def signed_log_grid(wmin=1e-6, wmax=2e1, npos=250):
    wp = np.logspace(np.log10(wmin), np.log10(wmax), int(npos))
    return np.concatenate([-wp[::-1], [0.0], wp])


def gain_at(case, p, omega):
    cl, theta = make_closed_loop(case, p)
    if case.mode == "static":
        return float(cl.gain_of_omega(float(theta), float(omega)))
    if hasattr(cl, "hinf_branch_value"):
        return float(cl.hinf_branch_value(theta, float(omega)))
    return float(cl.gain_of_omega(theta, float(omega)))


def companion_L(a):
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    r = a.size
    L = np.zeros((r, r), dtype=np.complex128)
    if r > 1:
        for j in range(r - 1):
            L[j, j + 1] = 1.0
    L[-1, :] = -a
    return L


def controller_transfer(a, n, K, omega):
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    n = np.asarray(n, dtype=np.complex128).reshape(-1)
    K = complex(K)

    # Avoid a singular solve when the dynamic numerator is exactly zero.
    if np.linalg.norm(n) == 0.0:
        return K

    r = a.size
    L = companion_L(a)
    er = np.zeros(r, dtype=np.complex128)
    er[-1] = 1.0
    s = 1j * float(omega)
    x = np.linalg.solve(s * np.eye(r, dtype=np.complex128) - L, er)
    return K + np.vdot(n, x)


def print_results_table(results):
    if MPI.COMM_WORLD.rank != 0:
        return

    print("\n" + "=" * 100)
    print("Best feasible/diagnostic result by case")
    print("=" * 100)
    header = f"{'case':42s} {'f':>12s} {'alpha':>12s} {'c':>12s} {'xa':>9s} {'xs':>9s} {'K':>12s} {'omega*':>12s}"
    print(header)
    print("-" * len(header))
    for res in results:
        b = res["best"]
        if b is None:
            print(f"{res['case'].name:42s}  NO RESULT")
            continue
        print(
            f"{res['case'].name:42s} "
            f"{b['f_raw']:12.5e} {b['alpha']:12.5e} {b['c']:12.5e} "
            f"{b['xa']:9.4f} {b['xs']:9.4f} {b['K']:12.5e} {b['omega_star']:12.5e}"
        )
        if "warning" in b:
            print("   WARNING:", b["warning"])
        if b.get("note"):
            print("   note:", b["note"])


def plot_objective_history(results):
    if MPI.COMM_WORLD.rank != 0:
        return
    plt.figure(figsize=(8, 5))
    for res in results:
        hist = res["history"]
        if len(hist) == 0:
            continue
        ys = [h["f_raw"] for h in hist]
        plt.semilogy(np.arange(1, len(ys) + 1), ys, marker="o", label=res["case"].name)
    plt.xlabel("PyGRANSO call")
    plt.ylabel(r"$\|G\|_\infty$ candidate")
    plt.title("Objective history")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_best_bar(results):
    if MPI.COMM_WORLD.rank != 0:
        return
    labels = []
    vals = []
    for res in results:
        b = res["best"]
        if b is not None:
            labels.append(res["case"].name.replace("_", "\n"))
            vals.append(b["f_raw"])
    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=0, fontsize=8)
    plt.ylabel(r"best feasible $\|G\|_\infty$")
    plt.title("Best objective by parametrization")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_placement_scatter(results):
    if MPI.COMM_WORLD.rank != 0:
        return
    plt.figure(figsize=(8, 4))
    for i, res in enumerate(results):
        b = res["best"]
        if b is None:
            continue
        plt.scatter([b["xa"]], [0.0], marker="^", s=90)
        plt.scatter([b["xs"]], [0.0], marker="v", s=90)
        plt.text(b["xa"], 0.04 + 0.04 * i, f"A{i}", ha="center", fontsize=9)
        plt.text(b["xs"], -0.08 - 0.04 * i, f"S{i}", ha="center", fontsize=9)
    plt.axhline(0.0, linewidth=1)
    plt.xlim(PLACEMENT_BOUNDS)
    plt.yticks([])
    plt.xlabel("x")
    plt.title("Optimized actuator/sensor locations: triangles up = actuator, down = sensor")
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_gain_curves(results, wmin=1e-6, wmax=2e1, npos=180):
    ws = signed_log_grid(wmin=wmin, wmax=wmax, npos=npos)
    curves = []

    # All ranks must participate in PETSc/SLEPc computations.
    for res in results:
        b = res["best"]
        if b is None:
            curves.append(None)
            continue
        case = res["case"]
        p = b["p"]
        vals = np.array([gain_at(case, p, w) for w in ws], dtype=float)
        curves.append(vals)

    if MPI.COMM_WORLD.rank != 0:
        return

    plt.figure(figsize=(8, 5))
    for res, vals in zip(results, curves):
        if vals is None:
            continue
        plt.semilogy(ws, vals, label=res["case"].name)
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"gain")
    plt.title("Closed-loop gain curves at the best iterate")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_controller_bode(results, wmin=1e-4, wmax=2e2, npos=400):
    wp = np.logspace(np.log10(wmin), np.log10(wmax), int(npos))
    if MPI.COMM_WORLD.rank != 0:
        return

    plt.figure(figsize=(8, 5))
    for res in results:
        b = res["best"]
        if b is None:
            continue
        _, _, a, n, K = split_p(b["p"], R_PLACEMENT)
        Hc = np.array([controller_transfer(a, n, K, w) for w in wp])
        plt.loglog(wp, np.abs(Hc), label=res["case"].name)
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$|K_c(i\omega)|$")
    plt.title("Controller magnitude by parametrization")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def make_all_plots(results):
    print_results_table(results)
    plot_objective_history(results)
    plot_best_bar(results)
    plot_placement_scatter(results)
    plot_gain_curves(results, wmin=OPT_SETTINGS["wmin"], wmax=OPT_SETTINGS["wmax"], npos=160)
    plot_controller_bode(results)


# -----------------------------------------------------------------------------
# Main call
# -----------------------------------------------------------------------------

# Uncomment this block after appending the file to yo.py.
#
# results_place = run_all_placement_cases(CASES, OPT_SETTINGS)
# make_all_plots(results_place)
#
# if MPI.COMM_WORLD.rank == 0:
#     np.savez(
#         "siso_placement_results.npz",
#         names=np.array([r["case"].name for r in results_place], dtype=object),
#         best_p=np.array([r["best"]["p"] if r["best"] is not None else np.full(2 + 2 * R_PLACEMENT + 1, np.nan)
#                          for r in results_place]),
#         best_f=np.array([r["best"]["f_raw"] if r["best"] is not None else np.nan
#                          for r in results_place]),
#     )
