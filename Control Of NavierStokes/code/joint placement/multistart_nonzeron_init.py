
# import librairies
import numpy as np
from scipy.signal import StateSpace
import sys
import os
from scipy.signal import impulse
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from slepc4py import SLEPc
from scipy.sparse import csr_matrix
from petsc4py import PETSc
import scipy.sparse as sp
from petsc4py import PETSc
from slepc4py import SLEPc
from numpy.linalg import solve
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from mpi4py import MPI

import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from petsc4py import PETSc
from dolfinx import fem
import ufl
import time
bisect_log = []
import time


from collections import defaultdict
from mpi4py import MPI

# -----------------------------------------------------------------------------
# MPI task-group layout
# -----------------------------------------------------------------------------
# PETSc/SLEPc calls are collective over the communicator used to build matrices.
# To safely parallelize independent frequency/oracle evaluations, split WORLD
# into independent FE_COMM task groups.  Default: one rank per task group,
# which works well for these small-ish 1D FE/SLEPc solves and avoids duplicated
# serial dense/PyGRANSO work being mistaken for parallel speedup.
WORLD = MPI.COMM_WORLD
WORLD_RANK = WORLD.Get_rank()
WORLD_SIZE = WORLD.Get_size()

TASK_GROUP_SIZE = int(os.environ.get("MPI_TASK_GROUP_SIZE", "1"))
if TASK_GROUP_SIZE < 1:
    TASK_GROUP_SIZE = 1
if TASK_GROUP_SIZE > WORLD_SIZE:
    TASK_GROUP_SIZE = WORLD_SIZE
if WORLD_SIZE % TASK_GROUP_SIZE != 0:
    if WORLD_RANK == 0:
        print(
            f"WARNING: WORLD_SIZE={WORLD_SIZE} is not divisible by "
            f"MPI_TASK_GROUP_SIZE={TASK_GROUP_SIZE}; reducing group size to 1.",
            flush=True,
        )
    TASK_GROUP_SIZE = 1

NUM_TASK_GROUPS = WORLD_SIZE // TASK_GROUP_SIZE
TASK_GROUP_ID = WORLD_RANK // TASK_GROUP_SIZE
TASK_RANK = WORLD_RANK % TASK_GROUP_SIZE
FE_COMM = WORLD.Split(color=TASK_GROUP_ID, key=TASK_RANK)
FE_RANK = FE_COMM.Get_rank()
FE_SIZE = FE_COMM.Get_size()
IS_WORLD_ROOT = (WORLD_RANK == 0)
IS_GROUP_LEADER = (FE_RANK == 0)

# Communicator containing only task-group leaders, used to gather Python results.
LEADER_COMM = WORLD.Split(color=0 if IS_GROUP_LEADER else MPI.UNDEFINED, key=TASK_GROUP_ID)

# Protect against accidental nested task maps.  Nested maps fall back to local
# serial execution inside the already-assigned task group, avoiding deadlocks.
_MPI_TASK_MAP_DEPTH = 0


def _env_flag(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _root_print(*args, **kwargs):
    if IS_WORLD_ROOT:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def _progress_enabled():
    return _env_flag("MPI_PROGRESS", True)


def _progress_print(*args, **kwargs):
    if IS_WORLD_ROOT and _progress_enabled():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def mpi_task_map(func, items, label="mpi_task_map", use_parallel=True):
    """Map independent Python tasks over MPI task groups.

    Every WORLD rank must call this function in the same order.  Each task group
    evaluates a subset of items collectively over FE_COMM; task-group leaders
    gather picklable results and broadcast the ordered list back to all ranks.
    """
    global _MPI_TASK_MAP_DEPTH
    items = list(items)
    nitems = len(items)
    if nitems == 0:
        return []

    # Nested calls cannot repartition all groups because the groups may already
    # be working on different outer tasks.  Compute locally instead.
    if (not use_parallel) or NUM_TASK_GROUPS <= 1 or _MPI_TASK_MAP_DEPTH > 0:
        _MPI_TASK_MAP_DEPTH += 1
        try:
            return [func(item) for item in items]
        finally:
            _MPI_TASK_MAP_DEPTH -= 1

    _MPI_TASK_MAP_DEPTH += 1
    try:
        if _env_flag("MPI_TASK_TRACE", False) and IS_WORLD_ROOT:
            print(f"[mpi-task-map] {label}: {nitems} item(s) over {NUM_TASK_GROUPS} task group(s)", flush=True)
        local_results = []
        for i, item in enumerate(items):
            if i % NUM_TASK_GROUPS == TASK_GROUP_ID:
                local_results.append((i, func(item)))

        gathered = None
        if IS_GROUP_LEADER:
            gathered = LEADER_COMM.gather(local_results, root=0)
            if TASK_GROUP_ID == 0:
                ordered = [None] * nitems
                for chunk in gathered:
                    for i, value in chunk:
                        ordered[i] = value
                missing = [i for i, value in enumerate(ordered) if value is None]
                if missing:
                    raise RuntimeError(f"{label}: missing MPI task results for indices {missing}")
            else:
                ordered = None
        else:
            ordered = None

        ordered = WORLD.bcast(ordered, root=0)
        if _env_flag("MPI_TASK_TRACE", False) and IS_WORLD_ROOT:
            print(f"[mpi-task-map] {label}: done", flush=True)
        return ordered
    except Exception as exc:
        # Make failures fail all ranks rather than leaving some stuck in collectives.
        if IS_WORLD_ROOT:
            print(f"ERROR in {label}: {repr(exc)}", flush=True)
        WORLD.Abort(911)
        raise
    finally:
        _MPI_TASK_MAP_DEPTH -= 1


if IS_WORLD_ROOT:
    print(
        "MPI layout: "
        f"WORLD_SIZE={WORLD_SIZE}, TASK_GROUP_SIZE={TASK_GROUP_SIZE}, "
        f"NUM_TASK_GROUPS={NUM_TASK_GROUPS}, FE_SIZE={FE_SIZE}",
        flush=True,
    )

class Profiler:
    def __init__(self, comm):
        self.comm = comm
        self.time_total = defaultdict(float)
        self.calls = defaultdict(int)
        self.meta_sum = defaultdict(float)   # optional numeric counters
        self.meta_max = defaultdict(float)

    def add_time(self, key, dt):
        self.time_total[key] += float(dt)
        self.calls[key] += 1

    def add_count(self, key, val=1):
        self.meta_sum[key] += float(val)

    def add_max(self, key, val):
        self.meta_max[key] = max(self.meta_max[key], float(val))

    def scoped(self, key):
        return _ProfileScope(self, key)

    def report(self, sort_by="time"):
        # parallel reduction: max time is the relevant wall clock for parallel sections
        keys = sorted(set(self.time_total) | set(self.calls) | set(self.meta_sum) | set(self.meta_max))

        global_time = {}
        global_calls = {}
        global_meta_sum = {}
        global_meta_max = {}

        for k in keys:
            global_time[k] = self.comm.allreduce(self.time_total.get(k, 0.0), op=MPI.MAX)
            global_calls[k] = int(self.comm.allreduce(self.calls.get(k, 0), op=MPI.SUM))
            global_meta_sum[k] = self.comm.allreduce(self.meta_sum.get(k, 0.0), op=MPI.SUM)
            global_meta_max[k] = self.comm.allreduce(self.meta_max.get(k, 0.0), op=MPI.MAX)

        if IS_WORLD_ROOT:
            rows = []
            total = sum(global_time.values())

            for k in keys:
                t = global_time[k]
                c = global_calls[k]
                avg = t / c if c > 0 else 0.0
                rows.append((k, t, c, avg, global_meta_sum[k], global_meta_max[k]))

            if sort_by == "time":
                rows.sort(key=lambda x: x[1], reverse=True)
            elif sort_by == "calls":
                rows.sort(key=lambda x: x[2], reverse=True)

            print("\n=== Profiling summary (max time over ranks) ===")
            print(f"{'name':30s} {'time [s]':>12s} {'calls':>10s} {'avg [ms]':>12s} {'meta_sum':>12s} {'meta_max':>12s}")
            print("-" * 95)
            for name, t, c, avg, s, m in rows:
                pct = 100.0 * t / (total + 1e-30)
                print(f"{name:30s} {t:12.6f} {c:10d} {1000*avg:12.3f} {s:12.1f} {m:12.1f}   ({pct:5.1f}%)")
            print("-" * 95)
            print(f"{'TOTAL':30s} {total:12.6f}")
        return global_time, global_calls, global_meta_sum, global_meta_max


class _ProfileScope:
    def __init__(self, profiler, key):
        self.profiler = profiler
        self.key = key

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.profiler.add_time(self.key, time.perf_counter() - self.t0)

prof = Profiler(FE_COMM)
class SectionTimer:
    def __init__(self, comm):
        self.comm = comm
        self.t = {"gain": 0.0, "shiftinvert": 0.0}
    def add(self, key, dt):
        self.t[key] += dt
    def report(self):
        # max across ranks ~ true wall-time for parallel sections
        out = {}
        for k, v in self.t.items():
            out[k] = self.comm.allreduce(v, op=MPI.MAX)
        if IS_WORLD_ROOT:
            total = sum(out.values())
            print("\n=== Timing summary (max over ranks) ===")
            for k in out:
                print(f"{k:12s}: {out[k]:.6f} s  ({100*out[k]/(total+1e-30):.1f}%)")
            print(f"{'total':12s}: {total:.6f} s")
        return out

timer = SectionTimer(FE_COMM)

# Parameters 
mu_0 = 0.38 # overall amplification 
mu_2 = -0.01 # degree of non-parallelism 
mu_t = 0.32 # transitional mu 
mu_c = 0.40 # critical mu
c_d = -1.0 # dispersion parameter 
c_u = 0.2 # most unstable wavenumber 
U = 2.0 # mean advection velocity
x_min = -56.06 
x_max = 56.06
Nx = 400 # resolution

domain = mesh.create_interval(FE_COMM, Nx, [x_min, x_max])
x = ufl.SpatialCoordinate(domain)
gamma = fem.Constant(domain, 1 + 1j*c_d) # diffusion coefficient
nu = fem.Constant(domain, U + 2j*c_u) # advection coefficient 
mu_x = (mu_0 - c_u**2) + (mu_2*x[0]**2)/2 # region of amplification

# if mu_0 - c_u^2 < 0 then the flow is stable everywhere

V = fem.functionspace(domain, ("CG", 2)) # continous Galerkin, x ordre
q = ufl.TrialFunction(V) # state
phi = ufl.TestFunction(V) # test function

# Dirichlet BCs at q = 0
def boundary(x):
    return np.isclose(x[0], x_min) | np.isclose(x[0], x_max) 

bc = fem.dirichletbc(PETSc.ScalarType(0.0 + 0.0j),fem.locate_dofs_geometrical(V, boundary),V)

m_form = ufl.inner(q, phi) * ufl.dx # mass matrix or left hand side 

# Weak form
a_form = (
    nu * ufl.inner(q, ufl.grad(phi)[0]) * ufl.dx
    + mu_x * ufl.inner(q, phi) * ufl.dx
    - gamma * ufl.inner(ufl.grad(q)[0], ufl.grad(phi)[0]) * ufl.dx # right hand side 
)

A = fem.petsc.assemble_matrix(fem.form(a_form), bcs=[bc]) # Assemble a bilinear form into a matrix
A.assemble()

M = fem.petsc.assemble_matrix(fem.form(m_form), bcs=[bc])
M.assemble()

# Parameters from the paper
sigma_gauss = 0.4
xa, xs = -1.03, 0.98 # location of the sensors

# gaussian function from Chen & Rowley for the actuators and the sensors
def gaussian_function(Vspace, x0, sig):
    g = fem.Function(Vspace)
    g.interpolate(lambda x: np.exp(-0.5 * ((x[0] - x0) / sig) ** 2))
    return g

b_fun = gaussian_function(V, xa, sigma_gauss)
s_fun = gaussian_function(V, xs, sigma_gauss)

B_load = fem.petsc.assemble_vector(fem.form(b_fun * ufl.conj(phi) * ufl.dx))
B_load.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(B_load, [bc])  

s_vec = s_fun.x.petsc_vec.copy()
s_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

bc_dofs = bc.dof_indices()[0]  # in recent dolfinx; if this fails, print(type(bc)) and we adjust
bc_dofs = np.array(bc_dofs, dtype=PETSc.IntType)

n = M.getSize()[0]

all_dofs = np.arange(n, dtype=PETSc.IntType)
mask = np.ones(n, dtype=bool)
mask[bc_dofs] = False
free = all_dofs[mask]

is_free = PETSc.IS().createGeneral(free, comm=M.comm)
A = A.createSubMatrix(is_free, is_free)
M = M.createSubMatrix(is_free, is_free)
A.assemble(); M.assemble()

###############################################################################################
def T(A: PETSc.Mat) -> PETSc.Mat:
    AH = PETSc.Mat().createHermitianTranspose(A)
    AH.assemble()
    return AH
def _block_is(comm, sizes):
    is_list = []
    start = 0
    for sz in sizes:
        is_list.append(PETSc.IS().createStride(sz, start, 1, comm=comm))
        start += sz
    return is_list

def eye_scaled(comm, n, alpha, like_mat: PETSc.Mat) -> PETSc.Mat:
    I = PETSc.Mat().create(comm=comm)
    I.setSizes([n, n])
    I.setType(like_mat.getType())
    I.setUp()

    d = PETSc.Vec().create(comm=comm)
    d.setSizes(n)
    d.setFromOptions()
    d.set(alpha)
    I.setDiagonal(d)
    I.assemble()
    return I

def rel_diff(A: PETSc.Mat, B: PETSc.Mat) -> float:
    A = A.convert("aij"); A.assemble()
    B = B.convert("aij"); B.assemble()
    R = A.copy()
    R.axpy(-1.0, B)
    R.assemble()
    return R.norm(PETSc.NormType.NORM_FROBENIUS) / (B.norm(PETSc.NormType.NORM_FROBENIUS) + 1e-30)

def matmul_sparse(A: PETSc.Mat, B: PETSc.Mat) -> PETSc.Mat:
    C = A.matMult(B)
    C.assemble()
    return C

def descriptor_matrix_boyd_openloop(E: PETSc.Mat) -> PETSc.Mat:
    """
    Build N = [[E^2, 0],
               [0,   E^T]]
    
    """
    comm = E.comm
    n, _ = E.getSize()

    E2 = matmul_sparse(E, E)   # E^2
    ET = T(E)                  # E^T 

    blocks = [
        [E2, None],
        [None, ET],
    ]

    sizes = [n, n]
    isrows = _block_is(comm, sizes)
    iscols = _block_is(comm, sizes)

    N = PETSc.Mat().createNest(blocks, isrows=isrows, iscols=iscols, comm=comm)
    N.assemble()
    return N, (E2, ET)

def hamiltonian_matrix_boyd_openloop(A: PETSc.Mat, E: PETSc.Mat, gamma: float) -> PETSc.Mat:
    """
    Build M_gamma = [[E*A,        gamma^{-1} I],
                     [-gamma^{-1} E,  -A^T     ]]
    so that sN - M_gamma 
    """
    comm = A.comm
    n, _ = A.getSize()

    EA = matmul_sparse(E, A)          # E*A
    
    AH = PETSc.Mat().createHermitianTranspose(A)
    AH.assemble()
    ATneg = AH.copy()
    ATneg.scale(-1.0)
    ATneg.assemble()
    Iginv = eye_scaled(comm, n, 1.0/gamma, like_mat=A)  # gamma^{-1} I

    Einvblock = E.copy()
    Einvblock.scale(-1.0/gamma)       # -(gamma^{-1}) E
    Einvblock.assemble()

    blocks = [
        [EA,     Iginv],
        [Einvblock, ATneg],
    ]

    sizes = [n, n]
    isrows = _block_is(comm, sizes)
    iscols = _block_is(comm, sizes)

    Mgam = PETSc.Mat().createNest(blocks, isrows=isrows, iscols=iscols, comm=comm)
    Mgam.assemble()
    return Mgam, (EA, Iginv, Einvblock, ATneg)

E = M   
gamma = 1.0

N, (E2, ET) = descriptor_matrix_boyd_openloop(E)
Mgam, (EA, Iginv, Einvblock, ATneg) = hamiltonian_matrix_boyd_openloop(A, E, gamma)

n, _ = E.getSize()
is2 = _block_is(E.comm, [n, n])



def energy_identity_check(S: PETSc.Mat, ksp_S: PETSc.KSP, M: PETSc.Mat,
                          Aop: PETSc.Mat, v: PETSc.Vec):
    # u = S^{-1} v
    u = S.createVecRight()
    b = S.createVecLeft()
    v.copy(b)
    ksp_S.solve(b, u)

    Mu = u.duplicate(); M.mult(u, Mu)

    # left = v^H (Aop v)
    Av = v.duplicate(); Aop.mult(v, Av)
    left = v.dot(Av)

    # right = u^H M u
    right = u.dot(Mu)

    rel = abs(left - right) / (abs(left) + abs(right) + 1e-30)

    if IS_WORLD_ROOT:
        print("energy check: v^H(Aop v) =", left)
        print("energy check: (S^{-1}v)^H M (S^{-1}v) =", right)
        print("energy check rel diff =", rel)

    return rel


def build_S(A: PETSc.Mat, M: PETSc.Mat, omega: float) -> PETSc.Mat:
    s = 1j * omega
    S = A.copy()
    S.scale(-1.0)      # -A
    S.axpy(s, M)       # -A + s M  = iω M - A
    S.assemble()
    return S

def make_lu_ksp(S: PETSc.Mat) -> PETSc.KSP:
    prof.add_count("lu_factorizations")
    with prof.scoped("make_lu_ksp"):
        ksp = PETSc.KSP().create(S.comm)
        ksp.setOperators(S)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorShift(shift_type=PETSc.Mat.FactorShiftType.NONZERO, amount=1e-12)
        ksp.setUp()
        ksp.setErrorIfNotConverged(True)
        return ksp

class AomegaCtx:
    def __init__(self, S: PETSc.Mat, M: PETSc.Mat):
        self.M = M
        self.S = S
        self.ksp = make_lu_ksp(S)

        self.u = S.createVecRight()
        self.v = M.createVecLeft()
        self.rhsT = S.createVecRight()
        self.zT = S.createVecLeft()

    def solve_hermitian(self, b: PETSc.Vec, y: PETSc.Vec):
        with prof.scoped("solve_transpose"):
            b.copy(self.rhsT)
            self.rhsT.conjugate()
            self.ksp.solveTranspose(self.rhsT, self.zT)
            self.zT.conjugate()
            self.zT.copy(y)
            self.zT.conjugate()
        prof.add_count("solve_transpose_calls")

    def mult(self, A, x, y):
        with prof.scoped("Aomega_mult"):
            with prof.scoped("solve_forward"):
                self.ksp.solve(x, self.u)
            prof.add_count("solve_forward_calls")

            self.M.mult(self.u, self.v)

            self.solve_hermitian(self.v, y)

def make_Aomega_shell(A: PETSc.Mat, M: PETSc.Mat, omega: float) -> PETSc.Mat:
    S = build_S(A, M, omega)
    n, _ = M.getSize()

    ctx = AomegaCtx(S, M)

    Aop = PETSc.Mat().createPython([n, n], comm=M.comm)
    Aop.setPythonContext(ctx)
    Aop.setUp()
    Aop.assemble()

    return Aop

def rayleigh_quotient_check(Aop: PETSc.Mat, M: PETSc.Mat, v: PETSc.Vec):
    Av = v.duplicate(); Aop.mult(v, Av)
    Mv = v.duplicate(); M.mult(v, Mv)

    num = v.dot(Av)     # v^H A v
    den = v.dot(Mv)     # v^H M v
    rq  = num / den

    # report
    if IS_WORLD_ROOT:
        print("Rayleigh quotient num v^HAv =", num)
        print("Rayleigh quotient den v^HMv =", den)
        print("Rayleigh quotient rq =", rq)

    return rq




def validate_near_shift_candidates(cl, k, gamma, lambdas, ncheck=6, probe_eps=1e-6):
    """
    Probe the gain at the imaginary parts of the first few eigenvalues nearest the shift.
    If any probe exceeds gamma, certification fails.
    """
    for lam in lambdas[:min(len(lambdas), ncheck)]:
        w = float(lam.imag)
        for h in (-probe_eps, 0.0, probe_eps):
            g = cl.gain_of_omega(k, w + h)
            if g > gamma:
                return True, {
                    "reason": "gain_violation_from_near_shift_eig",
                    "omega": float(w + h),
                    "gain": float(g),
                    "lambda_hit": lam,
                }
    return False, None


def certify_global_peak_disk_cl_safe(
    cl,
    k,
    gamma,
    b=10.0,
    eigs_tol=1e-8,
    nev=60,
    eps_probe=1e-6,
    axis_warn=1e-3,
    min_interval=None,
    midpoint_guard=True,
):
    """
    Conservative disk certification on [-b,b].

    Returns
    -------
    certified : bool
    info : dict
    """
    Acl = cl.Acl(k)

    N = build_N_from_M(cl.M)
    Mgam = build_Mgamma(Acl, cl.M, gamma)

    N = N.convert("aij"); N.assemble()
    Mgam = Mgam.convert("aij"); Mgam.assemble()

    if min_interval is None:
        min_interval = eigs_tol

    intervals = [(-float(b), 0.0), (0.0, float(b))]
    visited = []

    # low-frequency guard
    for w in [0.0, -1e-12, 1e-12, -1e-9, 1e-9, -1e-6, 1e-6]:
        g = cl.gain_of_omega(k, w)
        if g > gamma:
            return False, {
                "reason": "low_freq_guard_hit",
                "omega": float(w),
                "gain": float(g),
            }

    while intervals:
        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        intervals.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        lo, hi = intervals.pop(0)
        theta = 0.5 * (lo + hi)

        if midpoint_guard:
            g_theta = cl.gain_of_omega(k, theta)
            if g_theta > gamma:
                return False, {
                    "reason": "midpoint_gain_violation",
                    "omega": float(theta),
                    "gain": float(g_theta),
                }

        lambdas, nconv = eigs_close_to_shift_pencil(
            N, Mgam, theta,
            nev=nev,
            tol=eigs_tol,
            max_it=2000,
        )

        visited.append((lo, hi, theta, nconv))

        # No spectral information => no exclusion
        if nconv == 0 or lambdas.size == 0:
            continue

        shift = 1j * theta

        # Sort already happens in eigs_close_to_shift_pencil, but keep explicit
        lambdas = lambdas[np.argsort(np.abs(lambdas - shift))]

        # Probe gain at the first few nearby eigenvalue imaginary parts
        hit, info = validate_near_shift_candidates(
            cl, k, gamma, lambdas,
            ncheck=6,
            probe_eps=eps_probe,
        )
        if hit:
            return False, info

        nearest = lambdas[0]
        nearest_dist = float(abs(nearest - shift))

        # If the nearest eigenvalue is close to the imaginary axis, do not exclude aggressively
        if abs(nearest.real) <= axis_warn:
            # keep searching elsewhere; do not remove this interval yet
            continue

        # Only now do we use the exclusion radius
        if np.isfinite(nearest_dist) and nearest_dist > 0.0:
            intervals += split_by_removed_middle(
                lo, hi, theta,
                r=nearest_dist,
                min_progress=0.5 * min_interval
            )

    _progress_print(f"[disk-cert] done: fully excluded after {len(visited)} interval visit(s)")
    return True, {
        "reason": "fully_excluded",
        "window": (-float(b), float(b)),
        "visited": visited,
    }



def sigma_max_energy_resolvent(A: PETSc.Mat, M: PETSc.Mat, omega: float,
                               eps_tol=1e-8, max_it=200, ncv=40) -> float:
    with prof.scoped("sigma_max_energy_resolvent"):
        Aop = make_Aomega_shell(A, M, omega)

        eps = SLEPc.EPS().create(comm=M.comm)
        eps.setOperators(Aop, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)

        with prof.scoped("eps_solve_resolvent"):
            eps.solve()

        prof.add_count("eps_solve_resolvent_calls")
        prof.add_max("eps_iter_resolvent_max", eps.getIterationNumber())
        prof.add_count("eps_iter_resolvent_sum", eps.getIterationNumber())

        nconv = eps.getConverged()
        if nconv < 1:
            raise RuntimeError(f"EPS did not converge for sigma_max at omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = float(np.real(lam))
        lam = max(lam, 0.0)
        return float(np.sqrt(lam))

def is_hermitian(S: PETSc.Mat, tol=1e-10, norm_type=PETSc.NormType.FROBENIUS):
    # Form S^H
    Sh = S.copy()
    Sh = T(S)

    # D = S - S^H
    D = S.copy()
    D.axpy(-1.0, Sh)
    D.assemble()

    # Norms
    norm_S = S.norm(norm_type)
    norm_D = D.norm(norm_type)

    rel_err = norm_D / (norm_S + 1e-30)

    return rel_err < tol, rel_err

def gain_of_omega(A: PETSc.Mat, M: PETSc.Mat, omega: float) -> float:
    prof.add_count("gain_eval")
    prof.add_max("max_abs_omega", abs(omega))
    with prof.scoped("gain_of_omega"):
        return sigma_max_energy_resolvent(A, M, omega)

def build_N_from_M(M: PETSc.Mat):
    # Boyd open-loop N
    N, _ = descriptor_matrix_boyd_openloop(M)
    return N
def build_Mgamma(A: PETSc.Mat, M: PETSc.Mat, gamma: float):
    Mgam, _ = hamiltonian_matrix_boyd_openloop(A, M, gamma)
    return Mgam
N     = build_N_from_M(M)
Mgam  = build_Mgamma(A, M, gamma)



#  Estimate upper bound b    
def estimate_b_bound(A, M, B_load, c_vec, eps=1e-6, w0=1e-6, wmax=1e6):
    """
    Input: Transfer function H , tolerance eps
    Output: Upper bound of the critical interval b 

    Algo: 

    set w := 1
    set G := 0
    while ‖H(iw) - G ‖_2 > eps:
        set G := H(iw)
        set w := 2w
    end while
    set b := w
    """
    omega = w0
    Gprev_pos = 0.0
    Gprev_neg = 0.0

    while True:
        Gpos = gain_of_omega(A, M, omega)
        Gneg = gain_of_omega(A, M, -omega)

        if abs(Gpos - Gprev_pos) <= eps and abs(Gneg - Gprev_neg) <= eps:
            return omega

        Gprev_pos = Gpos
        Gprev_neg = Gneg
        omega *= 2.0

        if omega > wmax:
            return omega
    
def prune_intervals(intervals, min_len):
    """Drop tiny/degenerate intervals."""
    out = []
    for lo, hi in intervals:
        if hi > lo and (hi - lo) > min_len:
            out.append((float(lo), float(hi)))
    return out
def low_freq_guard(A, M, gamma):
    global bisect_log
    for w in [0.0, -1e-12, 1e-12, -1e-9, 1e-9, -1e-6, 1e-6, -1e-5, 1e-5]:
        t0 = time.perf_counter()
        g = gain_of_omega(A,M,w)
        bisect_log.append((w, g))
        timer.add("gain", time.perf_counter() - t0)
        if g > gamma:
            return True
    return False

def split_by_removed_middle(lo, hi, theta, r, min_progress):
    """
    Remove [theta-r, theta+r] (clipped), but ensure at least min_progress.
    Returns remaining sub-intervals (0, 1, or 2 of them).
    """
    r_eff = max(float(r), float(min_progress))

    mid_lo = max(lo, theta - r_eff)
    mid_hi = min(hi, theta + r_eff)

    out = []
    if mid_lo - lo > 0:
        out.append((lo, mid_lo))
    if hi - mid_hi > 0:
        out.append((mid_hi, hi))
    return out

class KOpCtx:
    def __init__(self, N, ksp_minus, ksp_plus):
        self.N = N
        self.ksp_minus = ksp_minus
        self.ksp_plus  = ksp_plus
        # work vecs
        self.v1 = N.createVecRight()
        self.v2 = N.createVecRight()
        self.v3 = N.createVecRight()

    def mult(self, A, x, y):
        # K(xi) = (Mgam + xi*N)^-1 N(Mgam - xi*N)^-1 N
        # v1 = N*x
        self.N.mult(x, self.v1)
        # v2 = (Mgam - xi N)^(-1) v1
        self.ksp_minus.solve(self.v1, self.v2)
        # v3 = N*v2
        self.N.mult(self.v2, self.v3)
        # y  = (Mgam + xi N)^(-1) v3
        self.ksp_plus.solve(self.v3, y)
def make_K_shell(N: PETSc.Mat, Mgam: PETSc.Mat, xi: complex) -> PETSc.Mat:
    # N    = N.convert("aij");    N.assemble()
    # Mgam = Mgam.convert("aij"); Mgam.assemble()

    # Sminus = Mgam - xi*N, Splus = Mgam + xi*N
    Sminus = Mgam.copy(); Sminus.axpy(-xi, N); Sminus.assemble()
    Splus  = Mgam.copy(); Splus.axpy(+xi, N); Splus.assemble()
    #Sminus = Sminus.convert("aij"); Sminus.assemble()
    #Splus  = Splus.convert("aij");  Splus.assemble()
    #ensure_diagonal_entries(Sminus, value=0.0)
    #ensure_diagonal_entries(Splus,  value=0.0)

    ksp_minus = make_lu_ksp(Sminus)
    ksp_plus  = make_lu_ksp(Splus)

    n = N.getSize()[0]
    ctx = KOpCtx(N, ksp_minus, ksp_plus)

    K = PETSc.Mat().createPython([n, n], comm=N.comm)
    K.setPythonContext(ctx)
    K.setUp()
    K.assemble()
    return K
def lambdas_from_etas(etas, xi, theta, max_keep=None):
    """
    Convert eta eigenvalues of K(xi) into lambda candidates,
    choosing the sqrt branch closest to shift i*theta.
    Returns lambdas sorted by distance to i*theta.
    """
    shift = 1j*theta
    lambdas = []

    for eta in etas:
        if eta == 0:
            continue
        z = xi*xi + 1.0/eta          # lambda^2
        lam0 = np.sqrt(z)            # principal
        lam1 = -lam0                 # other root

        # choose the one closer to the shift
        lam = lam0 if abs(lam0 - shift) <= abs(lam1 - shift) else lam1

        # store both ±lam (for completeness), but keep lam first
        lambdas.append(lam)
        lambdas.append(-lam)

    lambdas = np.array(lambdas, dtype=complex)

    # sort by distance to shift
    order = np.argsort(np.abs(lambdas - shift))
    lambdas = lambdas[order]

    if max_keep is not None and len(lambdas) > max_keep:
        lambdas = lambdas[:max_keep]

    return lambdas

def eta_to_lambda_near_shift(eta, shift):
    z = shift*shift + 1.0/eta          # lambda^2
    lam0 = np.sqrt(z)
    lam1 = -lam0
    lam  = lam0 if abs(lam0 - shift) <= abs(lam1 - shift) else lam1
    return lam, -lam
def eigs_close_to_shift_pencil(N: PETSc.Mat, Mgam: PETSc.Mat, theta: float,
                               nev=8, tol=1e-8, max_it=200):
    prof.add_count("pencil_shiftinvert_calls")
    with prof.scoped("eigs_close_to_shift_pencil"):
        s0 = 1j * theta

        eps = SLEPc.EPS().create(comm=N.comm)
        eps.setOperators(Mgam, N)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(s0)

        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(s0)
        eps.setDimensions(nev, ncv=max(4 * nev, 40))
        eps.setTolerances(tol, max_it)
        eps.setFromOptions()

        with prof.scoped("eps_solve_pencil"):
            eps.solve()

        prof.add_count("eps_iter_pencil_sum", eps.getIterationNumber())
        prof.add_max("eps_iter_pencil_max", eps.getIterationNumber())

        nconv = eps.getConverged()
        vals = [eps.getEigenvalue(i) for i in range(nconv)]
        vals = np.array(vals, dtype=complex)
        if vals.size == 0:
            return vals, nconv
        vals = vals[np.argsort(np.abs(vals - s0))]
        return vals, nconv

def eigs_close_to_shift_even_pencil(N, Mgam, theta, nev=16, tol=1e-6):
    shift = 1j*theta
    K = make_K_shell(N, Mgam, shift)
    eps = SLEPc.EPS().create(comm=N.comm)
    eps.setOperators(K)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev, ncv=max(4*nev, 40))
    eps.setTolerances(1e-6, PETSc.DECIDE)

    eps.setFromOptions()
    eps.solve()
    nconv = eps.getConverged()
    reason = eps.getConvergedReason()
    its    = eps.getIterationNumber()
    if IS_WORLD_ROOT:
        print("EPS reason:", reason, "its:", its, "nconv:", nconv)

    etas = [eps.getEigenvalue(i) for i in range(min(nconv, nev))]

    # map etas -> lambdas, with branch selection near shift
    lams = []
    for eta in etas:
        if eta == 0:
            continue
        
        lam, lam_neg = eta_to_lambda_near_shift(eta, shift)
        lams += [lam, lam_neg]

    lams = np.array(lams, dtype=complex)
    if lams.size == 0:
        return lams, tol, nconv

    # sort by distance to shift (matches MATLAB "close to shift")
    lams = lams[np.argsort(np.abs(lams - shift))]

    return lams, tol, nconv

def validate_imag_candidates(A, M, gamma, lambdas, eps_eigs):
    global bisect_log
    for lam in lambdas:
        # for lambda_i with Re(lambda_i) = 0 do
        if abs(lam.real) <= 1e-8:
            w = lam.imag
            for h in [-eps_eigs, 0.0, +eps_eigs]:
                g = gain_of_omega(A,M,w+h)
                bisect_log.append((w+h, g))
                if  g > gamma: 
                    return True
    return False
def certify_global_peak_disk_cl(
    cl,
    k,
    gamma,
    b=10.0,
    eigs_tol=1e-6,
    nev=12,
    eps_eigs=1e-6,
    min_interval=None,
):
    Acl = cl.Acl(k)

    N = build_N_from_M(cl.M)
    Mgam = build_Mgamma(Acl, cl.M, gamma)

    N = N.convert("aij"); N.assemble()
    Mgam = Mgam.convert("aij"); Mgam.assemble()

    if min_interval is None:
        min_interval = eigs_tol

    intervals = [(-float(b), 0.0), (0.0, float(b))]

    for w in [0.0, -1e-12, 1e-12, -1e-9, 1e-9, -1e-6, 1e-6]:
        g = cl.gain_of_omega(k, w)
        if g > gamma:
            return False, {"reason": "low_freq_guard_hit", "omega": float(w), "gain": float(g)}

    visited = []

    while intervals:
        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        intervals.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        lo, hi = intervals.pop(0)
        theta = 0.5 * (lo + hi)

        # very safe debug check
        g_theta = cl.gain_of_omega(k, theta)
        if g_theta > gamma:
            return False, {
                "reason": "midpoint_gain_violation",
                "omega": float(theta),
                "gain": float(g_theta),
            }

        lambdas, nconv = eigs_close_to_shift_pencil(
            N, Mgam, theta, nev=nev, tol=eigs_tol
        )

        visited.append((lo, hi, theta, nconv))

        # no spectral info => exclude nothing
        if nconv == 0 or lambdas.size == 0:
            continue

        hit, info = validate_near_shift_candidates(
            cl, k, gamma, lambdas, ncheck=5, probe_eps=eps_eigs
        )
        if hit:
            return False, info

        shift = 1j * theta
        r = float(np.min(np.abs(lambdas - shift)))

        # if nearest returned eigenvalue is still too close to imag axis,
        # be conservative and do not exclude aggressively
        nearest = lambdas[0]
        if abs(nearest.real) <= 1e-3:
            continue

        # optional old imag-axis test
        for lam in lambdas:
            if abs(lam.real) <= eps_eigs:
                w = float(lam.imag)
                for h in (-eps_eigs, 0.0, eps_eigs):
                    g = cl.gain_of_omega(k, w + h)
                    if g > gamma:
                        return False, {
                            "reason": "imag_axis_hit",
                            "omega": float(w + h),
                            "gain": float(g),
                            "lambda_hit": lam,
                        }

        if not np.isfinite(r):
            continue

        intervals += split_by_removed_middle(
            lo, hi, theta,
            r=r,
            min_progress=0.5 * min_interval
        )

    return True, {
        "reason": "fully_excluded",
        "window": (-float(b), float(b)),
        "visited": visited,
    }
def inner_check(gamma, A, M, tol_imag=1e-8, eigs_tol=1e-6, nev=16):
    N = build_N_from_M(M) 
    Mgam = build_Mgamma(A,M,gamma)
    Mgam = Mgam.convert("aij"); Mgam.assemble()
    N    = N.convert("aij");    N.assemble()
    # Call b for upper bound
    # b = estimate_b_bound(A, M, B_load, s_vec, eps=1e-6, w0=1.0, wmax=1e12)
    b = 10.0

    # check if
    if max(gain_of_omega(A,M,0.5*b), gain_of_omega(A,M,-0.5*b)) > gamma:
        return True
    
    # set critical interval
    intervals = [(-float(b), 0.0), (0.0, float(b))]
    min_interval = None
    if min_interval is None: 
        min_interval = eigs_tol

    if low_freq_guard(A, M, gamma):
        return True

    # while nci > 0 do 
    while intervals:

        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        # Sort to work on the largest remaining interval first
        intervals.sort(key=lambda interval: interval[1] - interval[0],reverse=True)
        lo, hi = intervals.pop(0) # removes and returns the first element of the list
        if hi <= lo:
            continue
        
        # theta = (CI11 + CI12)/2
        theta = 0.5*(lo + hi)

        # Compute lambdas the eigenvalues of sN - Mgam close to i*theta
        t0 = time.perf_counter()
        lambdas, nconv = eigs_close_to_shift_pencil(N, Mgam, theta, nev=nev, tol=eigs_tol)
        timer.add("shiftinvert", time.perf_counter() - t0)
        if nconv == 0 or lambdas.size == 0:
            intervals += split_by_removed_middle(lo, hi, theta, r=0.0, min_progress=0.5*min_interval)
            print("Nconv == 0 and here is the interval", intervals)
            continue

        shift = 1j*theta    
        r = float(np.min(np.abs(lambdas - shift)))

        for lam in lambdas[:min(len(lambdas), 6)]:
            if abs(lam.real) <= 1e-8:
                print("test")
                for lam in lambdas[:6]:
                    w = lam.imag
                    g = gain_of_omega(A, M, w)
                print("lam =", lam, "gain at Im(lam) =", g)
                if validate_imag_candidates(A, M, gamma, np.array([lam]), eps_eigs=1e-6):
                    return True

        if not np.isfinite(r):
            intervals += split_by_removed_middle(lo,hi, theta, r = 0.0, min_progress=0.5*min_interval)
            print("if not n.isfinite print here and thats the interval", intervals)
            continue

        if r <= 0.5*(hi - lo):
            intervals += split_by_removed_middle(lo, hi, theta, r=r, min_progress=0.5*min_interval)
            print("r < 0.5*(hi - lo)", intervals)
    return False  

def outer_bisection(A, M, tol_gamma=1e-8, tol_imag=1e-10, max_bisect=100):
    gamma_low = 0.0
    gamma_high = 1e-6

    while inner_check(gamma_high, A, M):
        gamma_low = gamma_high
        gamma_high *= 2.0

    for _ in range(max_bisect):
        gamma = 0.5*(gamma_low + gamma_high)
        violates = inner_check(gamma, A, M, tol_imag) 

        if violates:
            gamma_low = gamma
        else:
            gamma_high = gamma

        if gamma_high - gamma_low <= tol_gamma * max(1.0, gamma_high):
            return gamma_high

    return gamma_high

#ws = np.logspace(-14, 6, 4000)
#vals = []
#for w in ws:
    #vals.append(gain_of_omega(A,M,w))

#mx = max(vals)
#w_at = ws[int(np.argmax(vals))]

#print("max |G(iw)| on sweep =", mx, "at w =", w_at)
#mx = max(vals)
#w_at = ws[int(np.argmax(vals))]

#print("mx:", mx)

#print("ratio max/gamma_star =", mx/gamma_star)
  
# ws_pos = np.logspace(-6, 2, 400)
# ws = np.concatenate([-ws_pos[::-1], [0.0], ws_pos])
# # adjust range as needed
# vals = []

# for w in ws:
#     g = gain_of_omega(A, M, w)
#     vals.append(g)
# mx = max(vals)
# w_at = ws[int(np.argmax(vals))]

# print("max |G(iw)| on sweep =", mx, "at w =", w_at)
# vals = np.array(vals)

# if IS_WORLD_ROOT:
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(7,5))
#     plt.semilogy(ws, vals, linewidth=2)   # linear x, log y
#     plt.xlabel(r'$\omega$')
#     plt.ylabel(r'$\|G(i\omega)\|$')
#     plt.title('Resolvent Gain vs Frequency')
#     plt.grid(True, which="both", ls="--", alpha=0.5)
#     plt.tight_layout()
#     plt.show()
# #print("inner_check(1.0) =", inner_check(1.0, A, M, B_load, s_vec))
# #print("inner_check(1e-6) =", inner_check(1e-6, A, M, B_load,s_vec))
# start_time = time.time()

# gamma_star = outer_bisection(A, M )
# timer.report()
# print("gamma_star =", gamma_star)
# #print("ratio max/gamma_star =", mx/gamma_star)
# end_time = time.time()
# #print("inner_check(1.0) =", inner_check(1.0, A, M, B_load, s_vec))
# #print("inner_check(1e-6) =", inner_check(1e-6, A, M, B_load,s_vec))
# elapsed_time = end_time - start_time
# print(f"Execution time: {elapsed_time} seconds")
# if IS_WORLD_ROOT:
#     ws_log = np.array([p[0] for p in bisect_log])
#     gs_log = np.array([p[1] for p in bisect_log])

#     plt.figure(figsize=(7,5))
#     plt.plot(ws, vals, label="Gain curve")  # linear plot
#     plt.axhline(gamma_star, color='r', linestyle='--', label='gamma*')

#     plt.scatter(ws_log, gs_log, color='black', s=20, zorder=5, label='Bisection checks')

#     plt.xlabel(r'$\omega$')
#     plt.ylabel(r'$\|G(i\omega)\|$')
#     plt.legend()
#     plt.grid(True)  # no need for "which='both'" in linear scale
#     plt.show()
    
from petsc4py import PETSc
import numpy as np
from slepc4py import SLEPc

# -----------------------
# Small utilities
# -----------------------
def mat_to_aij(M):
    A = M.convert("aij")
    A.assemble()
    return A

def vec_norm(v):
    return v.norm(PETSc.NormType.NORM_2)

def mat_norm(A):
    return A.norm(PETSc.NormType.NORM_FROBENIUS)

def rel_mat_diff(A, B):
    A = mat_to_aij(A); B = mat_to_aij(B)
    R = A.copy()
    R.axpy(-1.0, B)
    R.assemble()
    den = mat_norm(B) + 1e-30
    return mat_norm(R) / den

def is_hermitian_mat(A, tol=1e-10):
    A = mat_to_aij(A)
    AH = PETSc.Mat().createHermitianTranspose(A)
    AH.assemble()
    rel = rel_mat_diff(A, AH)
    return (rel < tol), rel

def random_vec_like(mat, side="right", seed=1):
    # deterministic random vector (same across ranks)
    comm = mat.comm
    rng = np.random.default_rng(seed=seed)
    if side == "right":
        v = mat.createVecRight()
    else:
        v = mat.createVecLeft()
    # Fill local part deterministically
    lo, hi = v.getOwnershipRange()
    nloc = hi - lo
    data = rng.standard_normal(nloc) + 1j*rng.standard_normal(nloc)
    idx = np.arange(lo, hi, dtype=PETSc.IntType)   # <- int32 or int64 depending on PETSc
    v.setValues(idx, data.astype(np.complex128))
    v.assemble()
    return v

def ksp_lu(S):
    ksp = PETSc.KSP().create(S.comm)
    ksp.setOperators(S)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    ksp.setErrorIfNotConverged(True)
    ksp.setUp()
    return ksp

def solve(S, b):
    x = S.createVecRight()
    ksp = ksp_lu(S)
    ksp.solve(b, x)
    return x

def solveH_via_transpose(S, b):
    """
    Solve S^H y = b using: S^T conj(y) = conj(b)
    """
    rhs = b.duplicate(); b.copy(rhs); rhs.conjugate()
    z = S.createVecLeft()
    ksp = ksp_lu(S)
    ksp.solveTranspose(rhs, z)   # solves S^T z = rhs
    z.conjugate()                # y = conj(z)
    return z

# -----------------------
# Core building blocks you already have
# -----------------------
def build_S(A, M, omega):
    s = 1j * omega
    S = A.copy()
    S.scale(-1.0)      # -A
    S.axpy(s, M)       # -A + iω M
    S.assemble()
    return S

# -----------------------
# Tests
# -----------------------
def test_basic(A, M):
    comm = A.comm
    nA, mA = A.getSize()
    nM, mM = M.getSize()
    if IS_WORLD_ROOT:
        print("\n[TEST] basic")
        print("A size:", (nA, mA), "type:", A.getType(), "||A||F:", mat_norm(A))
        print("M size:", (nM, mM), "type:", M.getType(), "||M||F:", mat_norm(M))
    assert (nA == mA == nM == mM), "A and M must be square and same size"
    assert np.isfinite(mat_norm(A)) and np.isfinite(mat_norm(M)), "Non-finite matrix norm"

def test_mass_hermitian(M, tol=1e-10):
    ok, rel = is_hermitian_mat(M, tol=tol)
    if IS_WORLD_ROOT:
        print("\n[TEST] M Hermitian:", ok, "rel_err:", rel)
    # This should be essentially Hermitian for standard FE mass matrices
    if not ok and IS_WORLD_ROOT:
        print("WARNING: M is not Hermitian to tolerance. BC handling or assembly may be modifying it.")
    return ok, rel

def test_mass_positive(M, ntrials=5):
    """
    Quick stochastic check: v^H M v should be >= 0 for SPD/PSD.
    Not a proof, but catches sign/adjoint disasters.
    """
    comm = M.comm
    min_val = +np.inf
    for k in range(ntrials):
        v = random_vec_like(M, "right", seed=10+k)
        Mv = v.duplicate(); M.mult(v, Mv)
        q = v.dot(Mv)  # v^H M v
        min_val = min(min_val, np.real(q))
        if IS_WORLD_ROOT:
            print(f"[TEST] M positivity trial {k}: v^H M v = {q}")
    if IS_WORLD_ROOT:
        print("[TEST] M positivity min real(v^H M v) =", min_val)
    return min_val

def test_adjoint_operator(A, tol=1e-10):
    """
    Checks that PETSc's HermitianTranspose behaves sensibly:
    <x, A y> == <A^H x, y> for random x,y.
    """
    comm = A.comm
    AH = PETSc.Mat().createHermitianTranspose(A); AH.assemble()
    for k in range(3):
        x = random_vec_like(A, "right", seed=100+k)
        y = random_vec_like(A, "right", seed=200+k)
        Ay = y.duplicate(); A.mult(y, Ay)
        lhs = x.dot(Ay)  # x^H A y
        AHx = x.duplicate(); AH.mult(x, AHx)
        rhs = AHx.dot(y) # (A^H x)^H y = x^H A y
        rel = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30)
        if IS_WORLD_ROOT:
            print(f"\n[TEST] adjoint identity trial {k}: rel_err = {rel}")
        if rel > tol:
            if IS_WORLD_ROOT:
                print("FAILED: adjoint identity not satisfied. A may be ill-assembled or not in expected format.")
            return False, rel
    return True, 0.0

def test_resolvent_shell_matches_reference(A, M, omega, tol=1e-8):
    """
    Compare your intended operator:
        Aop = S^{-H} M S^{-1}
    applied to random v, against explicit reference.
    """
    comm = A.comm
    S = build_S(A, M, omega)

    v = random_vec_like(M, "right", seed=333)
    # reference: u = S^{-1} v, w = M u, y = S^{-H} w
    u = solve(S, v)
    Mu = u.duplicate(); M.mult(u, Mu)
    y_ref = solveH_via_transpose(S, Mu)

    # We'll check residual of S^H y_ref = Mu:
    # residual r = S^H y_ref - Mu should be small.
    SH = PETSc.Mat().createHermitianTranspose(S); SH.assemble()
    SHy = Mu.duplicate(); SH.mult(y_ref, SHy)
    r = SHy.duplicate(); SHy.copy(r); r.axpy(-1.0, Mu)
    rel_res = vec_norm(r) / (vec_norm(Mu) + 1e-30)

    if IS_WORLD_ROOT:
        print(f"\n[TEST] resolvent reference at omega={omega}")
        print("relative residual of S^H y = Mu :", rel_res)

    assert rel_res < tol, "Hermitian solve via transpose is not accurate enough (S^H solve inconsistent)."
    return rel_res

def test_energy_identity(A, M, omega, tol=1e-8):
    """
    Checks: v^H (S^{-H} M S^{-1}) v == (S^{-1} v)^H M (S^{-1} v)
    which must hold if Aop is built correctly.
    """
    comm = A.comm
    S = build_S(A, M, omega)
    v = random_vec_like(M, "right", seed=444)

    u = solve(S, v)
    Mu = u.duplicate(); M.mult(u, Mu)
    y = solveH_via_transpose(S, Mu)  # y = S^{-H} M u = Aop v

    left = v.dot(y)
    right = u.dot(Mu)
    rel = abs(left - right) / (abs(left) + abs(right) + 1e-30)

    if IS_WORLD_ROOT:
        print(f"\n[TEST] energy identity at omega={omega}")
        print("v^H Aop v =", left)
        print("u^H M u  =", right)
        print("rel_err  =", rel)

    assert rel < tol, "Energy identity failed: resolvent operator inconsistent with M-inner product."
    return rel

def test_boyd_blocks_consistency(A, M, gamma, descriptor_matrix_boyd_openloop, hamiltonian_matrix_boyd_openloop):
    """
    Sanity checks:
    - N is block diagonal with E^2 and E^H
    - Mgam uses -A^H (not an uninitialized matrix)
    """
    comm = A.comm
    E = M
    N, (E2, EH) = descriptor_matrix_boyd_openloop(E)
    Mgam, blocks = hamiltonian_matrix_boyd_openloop(A, E, gamma)

    # Check E2 ~ E*E
    E2_ref = E.matMult(E); E2_ref.assemble()
    e2_rel = rel_mat_diff(E2, E2_ref)

    # Check EH is actually Hermitian transpose of E
    EH_ref = PETSc.Mat().createHermitianTranspose(E); EH_ref.assemble()
    eh_rel = rel_mat_diff(EH, EH_ref)

    if IS_WORLD_ROOT:
        print("\n[TEST] Boyd blocks")
        print("rel_diff(E2, E*E)   =", e2_rel)
        print("rel_diff(EH, E^H)   =", eh_rel)

    assert e2_rel < 1e-10, "E2 block is not E*E"
    assert eh_rel < 1e-10, "EH block is not E^H"

    # Now the critical one: the bottom-right block should be -A^H
    # We extract it if Mgam is MatNest:
    if Mgam.getType().lower() == "nest":
        sub = Mgam.getNestSubMatrix(1, 1)
        AH = PETSc.Mat().createHermitianTranspose(A)  # A^H
        AH.assemble()
        ATneg = AH.copy()
        ATneg.scale(-1.0)                             # -A^H
        ATneg.assemble()
        br_rel = rel_mat_diff(sub, ATneg)
        if IS_WORLD_ROOT:
            print("rel_diff(Mgam(1,1), -A^H) =", br_rel)
        assert br_rel < 1e-10, "Hamiltonian bottom-right block is not -A^H (very likely your current bug)."
    else:
        if IS_WORLD_ROOT:
            print("Mgam is not nest type; skipping explicit BR block check.")

    return True

def run_all_tests(A, M, gamma,
                  descriptor_matrix_boyd_openloop,
                  hamiltonian_matrix_boyd_openloop,
                  omegas=(0.0, 1e-6, 1e-3, 1.0, 10.0)):

    test_basic(A, M)
    test_mass_hermitian(M)
    test_mass_positive(M, ntrials=5)
    test_adjoint_operator(A)

    for w in omegas:
        # Avoid exactly w=0 if you know S can be singular there
        test_resolvent_shell_matches_reference(A, M, omega=w if w != 0.0 else 1e-12)
        test_energy_identity(A, M, omega=w if w != 0.0 else 1e-12)

    test_boyd_blocks_consistency(A, M, gamma,
                                 descriptor_matrix_boyd_openloop,
                                 hamiltonian_matrix_boyd_openloop)

    if IS_WORLD_ROOT:
        print("\nAll tests completed.\n")


#run_all_tests(A, M, gamma,
              #descriptor_matrix_boyd_openloop,
              #hamiltonian_matrix_boyd_openloop)

def rightmost_eig_realpart(A, M, nev=5, tol=1e-8, max_it=200):
    eps = SLEPc.EPS().create(comm=A.comm)
    eps.setOperators(A, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setDimensions(nev=nev, ncv=max(4*nev, 40))
    eps.setTolerances(tol, max_it)
    eps.setFromOptions()
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError("No eigenpairs converged")
    vals = [eps.getEigenvalue(i) for i in range(nconv)]
    vals = np.array(vals, dtype=complex)
    vals = vals[np.argsort(-np.real(vals))]  # sort by real part desc
    return vals[:min(len(vals), nev)]


# gamma_dc = gain_of_omega(A, M, -15.0) 
# if IS_WORLD_ROOT:
#     print("DC gain (omega=0) =", gamma_dc)


def rightmost_eigs_shiftinvert(A, M, target=-0.0177-0.648j, nev=5, tol=1e-8, max_it=1000):
    eps = SLEPc.EPS().create(comm=A.comm)
    eps.setOperators(A, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

    # Shift-invert around target
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target)

    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(target)

    eps.setDimensions(nev=nev, ncv=max(80, 6*nev))
    eps.setTolerances(tol, max_it)
    eps.setFromOptions()
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError("No eigenpairs converged")

    vals = np.array([eps.getEigenvalue(i) for i in range(min(nconv, nev))], dtype=complex)
    vals = vals[np.argsort(-np.real(vals))]  # sort by real part
    return vals


#vals = rightmost_eigs_shiftinvert(A,M)
#if IS_WORLD_ROOT:
    #print("Rightmost eigf: ", vals)

class ProjectedSolveCtx:
    def __init__(self, Aom, M, v, lam):
        self.Aom = Aom
        self.M = M
        self.v = v.copy()
        self.lam = lam

        self.Mv = v.duplicate()
        M.mult(v, self.Mv)

        self.tmp1 = v.duplicate()
        self.tmp2 = v.duplicate()
        self.tmp3 = v.duplicate()

    def project(self, x, y):
        # y = x - v * (v^H M x)
        self.M.mult(x, self.tmp1)          # tmp1 = Mx
        alpha = self.v.dot(self.tmp1)      # v^H M x
        x.copy(y)
        y.axpy(-alpha, self.v)

    def mult(self, A, x, y):
        # xp = P x
        self.project(x, self.tmp2)

        # tmp3 = Aom xp
        self.Aom.mult(self.tmp2, self.tmp3)

        # tmp1 = M xp
        self.M.mult(self.tmp2, self.tmp1)

        # tmp3 = (Aom - lam M) xp
        self.tmp3.axpy(-self.lam, self.tmp1)

        # y = P tmp3
        self.project(self.tmp3, y)
 
class Aomega:
    def __init__(self, S: PETSc.Mat, M: PETSc.Mat):
        self.M = M
        self.S = S
        self.ksp = make_lu_ksp(S)

        self.u = S.createVecRight() # u = S^-1
        self.r = S.createVecLeft() # r = M u

        self.rhsT = S.createVecRight()
        self.zT = S.createVecLeft()
    def solve_hermitian(self, b: PETSc.Vec, y: PETSc.Vec):
        """We wanna solve S^H y = b by S^H y = b <=> S^T conj(y) = conj(b)"""
        b.copy(self.rhsT)
        self.rhsT.conjugate() # rhsT = conj(b)
        self.ksp.solveTranspose(self.rhsT, self.zT) # S^T zT = rhsT
        self.zT.conjugate() # zT = conj(y)
        self.zT.copy(y) # y = conj(zT)
        self.zT.conjugate() # restore zT

    def mult(self, A, x, y):
        self.ksp.solve(x, self.u) # u = S^{-1} x
        self.M.mult(self.u, self.r) # r = M u
        self.solve_hermitian(self.r, y) # y = (S^H)^{-1} r

def make_Aomega_shell_ctx(A: PETSc.Mat, M: PETSc.Mat, omega: float) -> PETSc.Mat:
    S = build_S(A, M, omega)
    n, _ = M.getSize()

    ctx = Aomega(S, M)

    AomegaMat = PETSc.Mat().createPython([n, n], comm=M.comm)
    AomegaMat.setPythonContext(ctx)
    AomegaMat.setUp()
    AomegaMat.assemble()

    return AomegaMat, ctx, S

def assemble_python_mat_to_aij(Aop: PETSc.Mat) -> PETSc.Mat:
    """
    Explicitly assemble a MatShell / Python Mat into an AIJ matrix by columns.
    Works in serial. Good for debugging.
    """
    comm = Aop.comm
    rank = comm.getRank()
    size = comm.getSize()
    if size != 1:
        raise NotImplementedError("This debug assembler is written for serial runs only.")

    n, m = Aop.getSize()
    if n != m:
        raise ValueError("Expected square matrix.")

    Aexp = PETSc.Mat().createAIJ([n, n], comm=comm)
    Aexp.setUp()

    ej = Aop.createVecRight()
    col = Aop.createVecLeft()

    for j in range(n):
        ej.set(0.0)
        ej.setValue(j, 1.0)
        ej.assemble()

        Aop.mult(ej, col)

        rows, vals = col.getValues(range(n)), None
        # petsc4py Vec.getValues takes indices and returns the values
        vals = rows
        nz = np.nonzero(np.abs(vals) > 1e-14)[0]
        if nz.size > 0:
            Aexp.setValues(nz.tolist(), [j], vals[nz].reshape(-1, 1))

    Aexp.assemble()
    return Aexp


def make_rank1_lift_matrix(M: PETSc.Mat, v: PETSc.Vec, tau: float) -> PETSc.Mat:
    """
    Build the rank-1 operator:
        tau * v (v^H M .)
    as an explicit AIJ matrix.

    Assumes v is M-normalized if you want this to 'lift' the null direction naturally.
    """
    comm = M.comm
    rank = comm.getRank()
    size = comm.getSize()
    if size != 1:
        raise NotImplementedError("This debug helper is written for serial runs only.")

    n, _ = M.getSize()

    Mv = v.duplicate()
    M.mult(v, Mv)   # Mv = M v

    # Outer product: tau * v * (Mv)^H
    v_arr = v.getArray(readonly=True).copy()
    Mv_arr = Mv.getArray(readonly=True).copy()

    R = PETSc.Mat().createAIJ([n, n], comm=comm)
    R.setUp()

    for i in range(n):
        row_vals = tau * v_arr[i] * np.conjugate(Mv_arr)
        nz = np.nonzero(np.abs(row_vals) > 1e-14)[0]
        if nz.size > 0:
            R.setValues([i], nz.tolist(), row_vals[nz])

    R.assemble()
    return R


def explicit_projected_operator_debug(Aom: PETSc.Mat, M: PETSc.Mat, v: PETSc.Vec, lam: float,
                                      tau: float = 0.0) -> PETSc.Mat:
    """
    Assemble K = P(Aom - lam M)P explicitly from your ProjectedSolveCtx.
    Optionally add a rank-1 lift:
        K_tau = K + tau * v (v^H M .)
    so LU can factor it on the full space.
    """
    proj = ProjectedSolveCtx(Aom, M, v, lam)

    Kshell = PETSc.Mat().createPython(Aom.getSize(), comm=M.comm)
    Kshell.setPythonContext(proj)
    Kshell.setUp()
    Kshell.assemble()

    Kexp = assemble_python_mat_to_aij(Kshell)

    if tau != 0.0:
        R = make_rank1_lift_matrix(M, v, tau)
        Kexp.axpy(1.0, R)
        Kexp.assemble()

    return Kexp


def solve_with_lu(Kexp: PETSc.Mat, rhs: PETSc.Vec) -> PETSc.Vec:
    """
    Solve Kexp z = rhs with direct LU.
    """
    ksp = PETSc.KSP().create(comm=Kexp.comm)
    ksp.setOperators(Kexp)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    ksp.setUp()
    ksp.setErrorIfNotConverged(True)

    z = rhs.duplicate()
    z.set(0.0)
    ksp.solve(rhs, z)
    return z


def residual_norm(K: PETSc.Mat, z: PETSc.Vec, rhs: PETSc.Vec) -> float:
    """
    Return ||K z - rhs||_2
    """
    r = rhs.duplicate()
    K.mult(z, r)      # r = K z
    r.axpy(-1.0, rhs) # r = K z - rhs
    return r.norm()

def get_lambda_dot(A: PETSc.Mat, M: PETSc.Mat, omega: float, eps_tol=1e-8, max_it=500, ncv=40):
    prof.add_count("lambda_dot_eval")
    with prof.scoped("get_lambda_dot"):
        comm = M.comm
        Aom, ctx, S = make_Aomega_shell_ctx(A, M, omega)

        eps = SLEPc.EPS().create(comm=comm)
        eps.setOperators(Aom, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()

        with prof.scoped("eps_solve_lambda_dot"):
            eps.solve()

        prof.add_count("eps_solve_lambda_dot_calls")
        prof.add_count("eps_iter_lambda_dot_sum", eps.getIterationNumber())
        prof.add_max("eps_iter_lambda_dot_max", eps.getIterationNumber())

        if eps.getConverged() < 1:
            raise RuntimeError(f"EPS did not converge for omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = float(np.real(lam))
        lam = max(lam, 0.0)
        g = float(np.sqrt(lam))

        v = S.createVecRight()
        eps.getEigenvector(0, v)

        Mv = v.duplicate()
        M.mult(v, Mv)
        den = v.dot(Mv)
        scale = 1.0 / np.sqrt(np.real(den))
        v.scale(scale)
        Mv.scale(scale)

        u = S.createVecRight()
        ctx.ksp.solve(v, u)

        r = M.createVecLeft()
        M.mult(u, r)

        y = S.createVecRight()
        ctx.solve_hermitian(r, y)

        w = S.createVecRight()
        ctx.ksp.solve(r, w)

        Mw = w.duplicate()
        M.mult(w, Mw)

        wMw = np.real(w.dot(Mw))
        yMw = np.real(y.dot(Mw))

        vA2v = 2.0 * wMw - 4.0 * yMw

        yw = y.duplicate()
        y.copy(yw)
        yw.axpy(-1.0, w)

        Myw = r.duplicate()
        M.mult(yw, Myw)

        Adotv = y.duplicate()
        ctx.solve_hermitian(Myw, Adotv)
        Adotv.scale(-1j)

        numerator = -1j * r.dot(yw)

        denominator = v.dot(Mv)
        lamdot = float(np.real(numerator / (denominator + 1e-30)))

        gdot = lamdot / (2.0 * g) if g > 0.0 else 0.0

        h = Adotv.duplicate()
        Adotv.copy(h)
        h.axpy(-lamdot, Mv)

        proj = ProjectedSolveCtx(Aom, M, v, lam)
        hp = h.duplicate()
        proj.project(h, hp)

        K = PETSc.Mat().createPython(Aom.getSize(), comm=M.comm)
        K.setPythonContext(proj)
        K.setUp()
        K.assemble()

        kspz = PETSc.KSP().create(comm=M.comm)
        kspz.setOperators(K)
        kspz.setType("gmres")
        pc = kspz.getPC()
        pc.setType("none")
        kspz.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
        kspz.setFromOptions()
        kspz.setUp()
        kspz.setErrorIfNotConverged(True)

        rhs = hp.duplicate()
        hp.copy(rhs)
        rhs.scale(-1.0)

        z = hp.duplicate()
        z.set(0.0)
        kspz.solve(rhs, z)
        proj.project(z, z)

        lambdadotdot = vA2v + 2.0 * np.real(z.dot(h))

        return g, gdot, lam, lamdot, lambdadotdot

def random_initial_omegas(num_guess, wmin=1e-8, wmax=1e3, seed=None):
    rng = np.random.default_rng(seed)
    logw = rng.uniform(np.log10(wmin), np.log10(wmax), size=num_guess)
    return 10.0**logw

def random_initial_guesses(A,M,omegas):
    gains = []
    for w in omegas:
        gains.append(gain_of_omega(A,M,float(w)))
    gains = np.array(gains, dtype = float)
    idx = int(np.argmax(gains))
    return gains, omegas[idx], gains[idx]

def check_derivatives(A, M, omega, rel_step=1e-4, verbose=True):
    h = rel_step * max(1.0, abs(float(omega)))

    # center
    g0, gdot0, lam0, lamdot0, lamddot0 = get_lambda_dot(A, M, omega)

    # neighbors
    gp, gdotp, lamp, lamdotp, _ = get_lambda_dot(A, M, omega + h)
    gm, gdotm, lamm, lamdotm, _ = get_lambda_dot(A, M, omega - h)

    # finite differences
    lamdot_fd  = (lamp - lamm) / (2.0*h)
    lamddot_fd = (lamdotp - lamdotm) / (2.0*h)
    lamddot_fd2 = (lamp - 2.0*lam0 + lamm) / (h*h)

    rel1 = abs(lamdot0 - lamdot_fd) / (abs(lamdot_fd) + abs(lamdot0) + 1e-30)
    rel2 = abs(lamddot0 - lamddot_fd) / (abs(lamddot_fd) + abs(lamddot0) + 1e-30)
    rel2b = abs(lamddot0 - lamddot_fd2) / (abs(lamddot_fd2) + abs(lamddot0) + 1e-30)

    if IS_WORLD_ROOT and verbose:
        print(f"\nomega = {omega}")
        print("lam        =", lam0)
        print("lamdot     =", lamdot0)
        print("lamdot FD  =", lamdot_fd)
        print("rel err λ' =", rel1)
        print("lamddot    =", lamddot0)
        print("lamddot FD(dλ') =", lamddot_fd)
        print("rel err λ'' (via dλ') =", rel2)
        print("lamddot FD(λ)   =", lamddot_fd2)
        print("rel err λ'' (via λ)   =", rel2b)

    return {
        "lam": lam0,
        "lamdot": lamdot0,
        "lamdot_fd": lamdot_fd,
        "rel_lamdot": rel1,
        "lamddot": lamddot0,
        "lamddot_fd": lamddot_fd,
        "rel_lamddot": rel2,
        "lamddot_fd2": lamddot_fd2,
        "rel_lamddot2": rel2b,
    }

#test_ws = [1e-3, 1e-1, 1.0, 10.0, ws[np.argmax(vals)]]
#print("test_ws =", test_ws)
#for w in test_ws:
    #print("checking omega =", w)
    #check_derivatives(A, M, float(w))

def check_Adotv_sign(A, M, omega, rel_step=1e-6):
    h = rel_step * max(1.0, abs(float(omega)))

    # center objects
    Aom0, ctx0, S0 = make_Aomega_shell_ctx(A, M, omega)

    # get dominant eigenvector v at center
    eps = SLEPc.EPS().create(comm=M.comm)
    eps.setOperators(Aom0, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setDimensions(nev=1, ncv=40)
    eps.solve()

    v = S0.createVecRight()
    eps.getEigenvector(0, v)

    # normalize v^H M v = 1
    Mv = v.duplicate()
    M.mult(v, Mv)
    den = v.dot(Mv)
    v.scale(1.0 / np.sqrt(np.real(den)))

    # analytic Adotv at center
    # u = S^{-1} v
    u = S0.createVecRight()
    ctx0.ksp.solve(v, u)

    r = M.createVecLeft()
    M.mult(u, r)

    y = S0.createVecRight()
    ctx0.solve_hermitian(r, y)

    w = S0.createVecRight()
    ctx0.ksp.solve(r, w)

    yw = y.duplicate()
    y.copy(yw)
    yw.axpy(-1.0, w)

    Myw = r.duplicate()
    M.mult(yw, Myw)

    Adotv = y.duplicate()
    ctx0.solve_hermitian(Myw, Adotv)
    Adotv.scale(1j)   # test your current sign

    Aomp, _, _ = make_Aomega_shell_ctx(A, M, omega + h)
    Aomm, _, _ = make_Aomega_shell_ctx(A, M, omega - h)

    Avp = v.duplicate()
    Avm = v.duplicate()
    Aomp.mult(v, Avp)
    Aomm.mult(v, Avm)

    Adotv_fd = v.duplicate()
    Avp.copy(Adotv_fd)
    Adotv_fd.axpy(-1.0, Avm)
    Adotv_fd.scale(1.0 / (2.0 * h))

    # compare both signs
    diff_plus = Adotv_fd.duplicate()
    Adotv_fd.copy(diff_plus)
    diff_plus.axpy(-1.0, Adotv)

    diff_minus = Adotv_fd.duplicate()
    Adotv_fd.copy(diff_minus)
    diff_minus.axpy(+1.0, Adotv)

    return diff_plus.norm(), diff_minus.norm()

def sigma_max_2norm_resolvent(A: PETSc.Mat, M: PETSc.Mat, omega: float,
                              eps_tol=1e-8, max_it=200, ncv=40) -> float:
    """
    Compute ||(iω M - A)^(-1)||_2
    by forming R^H R as a shell operator and taking its largest eigenvalue.
    """
    S = build_S(A, M, omega)
    n, _ = A.getSize()

    ksp = make_lu_ksp(S)

    class RHR_Ctx:
        def __init__(self, S, ksp):
            self.S = S
            self.ksp = ksp
            self.u = S.createVecRight()
            self.rhsT = S.createVecRight()
            self.zT = S.createVecLeft()

        def solve_hermitian(self, b: PETSc.Vec, y: PETSc.Vec):
            # Solve S^H y = b via transpose solve
            b.copy(self.rhsT)
            self.rhsT.conjugate()
            self.ksp.solveTranspose(self.rhsT, self.zT)
            self.zT.conjugate()
            self.zT.copy(y)
            self.zT.conjugate()

        def mult(self, mat, x, y):
            # y = R^H R x = S^{-H} S^{-1} x
            self.ksp.solve(x, self.u)          # u = S^{-1} x
            self.solve_hermitian(self.u, y)    # y = S^{-H} u

    ctx = RHR_Ctx(S, ksp)

    RHR = PETSc.Mat().createPython([n, n], comm=A.comm)
    RHR.setPythonContext(ctx)
    RHR.setUp()
    RHR.assemble()

    eps = SLEPc.EPS().create(comm=A.comm)
    eps.setOperators(RHR)
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setDimensions(nev=1, ncv=ncv)
    eps.setTolerances(eps_tol, max_it)
    eps.setFromOptions()
    eps.solve()

    if eps.getConverged() < 1:
        raise RuntimeError(f"EPS did not converge for 2-norm at omega={omega}")

    lam = eps.getEigenvalue(0)
    lam = max(float(np.real(lam)), 0.0)
    return np.sqrt(lam)

#omega_peak = ws[np.argmax(vals)]

#gain_M = gain_of_omega(A, M, float(omega_peak))
#gain_2 = sigma_max_2norm_resolvent(A, M, float(omega_peak))

#ratio = gain_M / gain_2

#if IS_WORLD_ROOT:
    #print("omega_peak =", omega_peak)
    #print("||R(iw)||_M =", gain_M)
    #print("||R(iw)||_2 =", gain_2)
    #print("ratio ||R||_M / ||R||_2 =", ratio)


def local_newton_peak(A, M, omega0, max_it=20, tol_w=1e-8, tol_lamdot=1e-8,
                      trust_radius=0.25, backtrack=0.5, min_step=1e-10):
    prof.add_count("newton_runs")
    with prof.scoped("local_newton_peak"):
        omega = float(omega0)
        g, gdot, lam, lamdot, lamddot = get_lambda_dot(A, M, omega)
        best = (omega, g)

        for it in range(max_it):
            prof.add_count("newton_iterations")
            if IS_WORLD_ROOT:
                print(f"start={omega0:.3e}, end={omega:.6e}, g={g:.6e}")
            if abs(lamdot) <= tol_lamdot * max(1.0, abs(lam)) and lamddot < 0.0:
                prof.add_count("newton_converged")
                return omega, g, {"status": "converged", "it": it}

            if np.isfinite(lamddot) and abs(lamddot) > 1e-14 and lamddot < 0.0:
                step = -lamdot / lamddot
                prof.add_count("newton_true_steps")
            else:
                step = trust_radius * np.sign(lamdot) if lamdot != 0 else 0.0
                prof.add_count("newton_fallback_steps")

            step = np.clip(step, -trust_radius, trust_radius)

            accepted = False
            alpha = 1.0
            while alpha >= min_step:
                prof.add_count("newton_backtrack_trials")
                omega_trial = omega + alpha * step

                # cheap acceptance test
                g_trial = gain_of_omega(A, M, omega_trial)

                if g_trial >= g:
                    # only compute the expensive derivatives
                    g_new, gdot_new, lam_new, lamdot_new, lamddot_new = get_lambda_dot(A, M, omega_trial)

                    prof.add_count("newton_accepted_steps")
                    omega, g = omega_trial, g_new
                    lam, lamdot, lamddot = lam_new, lamdot_new, lamddot_new

                    if g > best[1]:
                        best = (omega, g)

                    accepted = True
                    break

                alpha *= backtrack

            if not accepted:
                prof.add_count("newton_stalled")
                return best[0], best[1], {"status": "stalled", "it": it}

            if abs(alpha * step) <= tol_w * max(1.0, abs(omega)):
                prof.add_count("newton_small_step")
                return omega, g, {"status": "small_step", "it": it}
        

        prof.add_count("newton_maxit")
        return best[0], best[1], {"status": "max_it", "it": max_it}

def optimize_peak_multistart(A, M, omegas0):
    results = []
    for w0 in omegas0:
        try:
            w, g, info = local_newton_peak(A, M, float(w0))
            results.append((g, w, info))
        except Exception as e:
            results.append((-np.inf, float(w0), {"status": f"failed: {e}"}))

    results.sort(key=lambda t: t[0], reverse=True)
    g_best, w_best, info_best = results[0]
    return w_best, g_best, results

def certify_global_peak(A, M, gamma, b=10.0, eigs_tol=1e-6, nev=10, eps_eigs=1e-6):
    prof.add_count("certify_calls")
    with prof.scoped("certify_global_peak"):
        N = build_N_from_M(M)
        Mgam = build_Mgamma(A, M, gamma)

        N = N.convert("aij"); N.assemble()
        Mgam = Mgam.convert("aij"); Mgam.assemble()

        intervals = [(-float(b), float(b))]
        min_interval = eigs_tol

        while intervals:
            prof.add_count("certify_interval_iterations")

            intervals = prune_intervals(intervals, min_interval)
            if not intervals:
                return True, {"reason": "fully_excluded"}

            intervals.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
            lo, hi = intervals.pop(0)
            theta = 0.5 * (lo + hi)

            lambdas, nconv = eigs_close_to_shift_pencil(N, Mgam, theta, nev=nev, tol=eigs_tol)

            if nconv == 0 or lambdas.size == 0:
                intervals += split_by_removed_middle(lo, hi, theta, r=0.0, min_progress=0.5 * min_interval)
                continue

            shift = 1j * theta
            r = float(np.min(np.abs(lambdas - shift)))
            prof.add_max("certify_radius_max", r)

            for lam in lambdas:
                if abs(lam.real) <= 1e-8:
                    prof.add_count("imag_axis_candidates")
                    w = lam.imag
                    for h in (-eps_eigs, 0.0, eps_eigs):
                        g = gain_of_omega(A, M, w + h)
                        if g > gamma:
                            return False, {"reason": "imag_axis_hit", "omega": float(w + h), "gain": float(g)}

            intervals += split_by_removed_middle(lo, hi, theta, r=r, min_progress=0.5 * min_interval)

        return True, {"reason": "fully_excluded"}

def hinf_newton_plus_one_boyd_check(
    A, M,
    num_guess=1,
    wmin=1e-6,
    wmax=1e2,
    seed=1,
    b=10.0,
):
    # multistart
    omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
    w_best, g_best, all_results = optimize_peak_multistart(A, M, omegas0)

    certified, cert_info = certify_global_peak(A, M, g_best, b=b)

    return {
        "omega_star": w_best,
        "gamma_star": g_best,
        "certified": certified,
        "cert_info": cert_info,
        "local_results": all_results,
    }

from mpi4py import MPI
import time

#comm = WORLD
#rank = comm.rank

#comm.Barrier()
#start = time.perf_counter()

#res = hinf_newton_plus_one_boyd_check(A, M, num_guess=1, wmin=1e-6, wmax=1e2, seed=2, b=10.0)

#comm.Barrier()
#end = time.perf_counter()
#runtime = end - start

#if IS_WORLD_ROOT:
    #print("omega_star =", res["omega_star"])
    #print("gamma_star =", res["gamma_star"])
    #print("certified  =", res["certified"])
    #print("cert_info  =", res["cert_info"])
    #print("runtime (s) =", runtime)

#prof.report(sort_by="time")

def one_boyd_check_can_increase_gamma(
    A, M, gamma_star, omega_star,
    rel_increase=1e-3,   # test gamma slightly above Newton value
    nev=12,
    eigs_tol=1e-7,
    axis_tol=1e-8,
):
    """
    Fast Boyd/Balakrishnan-style check:
      - build pencil at gamma_test = (1+rel_increase)*gamma_star
      - do one shift-invert solve near i*omega_star
      - if we see an eigenvalue near the imaginary axis, gamma can still increase

    Returns
    -------
    can_increase : bool
        True  -> found an (approx) imaginary-axis eigenvalue at larger gamma
        False -> no such eigenvalue found in this check
    info : dict
    """
    gamma_test = (1.0 + rel_increase) * float(gamma_star)

    N = build_N_from_M(M)
    Mgam = build_Mgamma(A, M, gamma_test)

    N = N.convert("aij"); N.assemble()
    Mgam = Mgam.convert("aij"); Mgam.assemble()

    # single Mitchell-style check near the Newton peak frequency
    lambdas, nconv = eigs_close_to_shift_pencil(
        N, Mgam, float(omega_star), nev=nev, tol=eigs_tol
    )

    if nconv == 0 or lambdas.size == 0:
        return False, {
            "status": "no_eigs_returned",
            "gamma_test": gamma_test,
            "omega_shift": float(omega_star),
        }

    # look for an approximate imaginary-axis hit
    idx_axis = np.where(np.abs(np.real(lambdas)) <= axis_tol)[0]

    if idx_axis.size > 0:
        k = idx_axis[np.argmin(np.abs(np.imag(lambdas[idx_axis]) - float(omega_star)))]
        lam_hit = lambdas[k]
        return True, {
            "status": "imag_axis_hit",
            "gamma_test": gamma_test,
            "omega_shift": float(omega_star),
            "lambda_hit": lam_hit,
            "omega_hit": float(np.imag(lam_hit)),
        }

    # no axis hit found in this one check
    j = int(np.argmin(np.abs(np.real(lambdas))))
    return False, {
        "status": "no_imag_axis_hit",
        "gamma_test": gamma_test,
        "omega_shift": float(omega_star),
        "closest_lambda": lambdas[j],
        "closest_realpart": float(np.real(lambdas[j])),
    }

def hinf_newton_plus_one_boyd_check(
    A, M,
    num_guess=1,
    wmin=1e-6,
    wmax=1e2,
    seed=1,
    boyd_rel_increase=1e-3,
    boyd_nev=12,
    boyd_eigs_tol=1e-7,
    boyd_axis_tol=1e-8,
):
    omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
    w_best, g_best, all_results = optimize_peak_multistart(A, M, omegas0)

    can_increase, boyd_info = one_boyd_check_can_increase_gamma(
        A, M,
        gamma_star=g_best,
        omega_star=w_best,
        rel_increase=boyd_rel_increase,
        nev=boyd_nev,
        eigs_tol=boyd_eigs_tol,
        axis_tol=boyd_axis_tol,
    )

    return {
        "omega_star": w_best,
        "gamma_star": g_best,
        "boyd_passed": not can_increase,
        "boyd_info": boyd_info,
        "local_results": all_results,
    }

def few_boyd_checks_can_increase_gamma(
    A, M, gamma_star, omega_star,
    rel_increases=(1e-3, 5e-3, 1e-2),
    nev=12, eigs_tol=1e-7, axis_tol=1e-8,
):
    checks = []
    for rel in rel_increases:
        hit, info = one_boyd_check_can_increase_gamma(
            A, M, gamma_star, omega_star,
            rel_increase=rel, nev=nev,
            eigs_tol=eigs_tol, axis_tol=axis_tol
        )
        checks.append((rel, hit, info))
        if hit:
            return True, {"status": "can_increase", "checks": checks}
    return False, {"status": "no_hit_found", "checks": checks}

#comm.Barrier()
#start = time.perf_counter()

#res = hinf_newton_plus_one_boyd_check(
    #A, M,
    #num_guess=1,
    #wmin=1e-6,
    #wmax=1e2,
    #seed=2,
    #boyd_rel_increase=1e-3,
#)

#comm.Barrier()
#end = time.perf_counter()
#runtime = end - start

#if IS_WORLD_ROOT:
    #print("omega_star   =", res["omega_star"])
    #print("gamma_star   =", res["gamma_star"])
    #print("boyd_passed  =", res["boyd_passed"])
    #print("boyd_info    =", res["boyd_info"])
    #print("runtime (s)  =", runtime)



def restrict_vec_to_is(v_full: PETSc.Vec, iset: PETSc.IS) -> PETSc.Vec:
    """
    Return a standalone copy of the subvector v_full[iset].
    """
    subv = v_full.getSubVector(iset)
    out = subv.copy()
    v_full.restoreSubVector(iset, subv)
    return out

def gather_full_vec_numpy(v: PETSc.Vec) -> np.ndarray:
    """
    Gather a distributed PETSc Vec into a global numpy array.
    Fine for a first working implementation.
    """
    comm = v.comm.tompi4py()
    local = v.getArray(readonly=True).copy()
    parts = comm.allgather(local)
    return np.concatenate(parts)

def make_rank1_mat_from_vecs(left: PETSc.Vec, right: PETSc.Vec) -> PETSc.Mat:
    """
    Build the explicit AIJ matrix left * right^H.

    WARNING:
    This is dense rank-one assembled as an explicit matrix.
    It is fine for a first working version, but later you may want
    to replace it by a shell / Woodbury update.
    """
    comm = left.comm
    n = left.getSize()
    m = right.getSize()
    if n != m:
        raise ValueError("left and right must have same global size")

    R = PETSc.Mat().createAIJ([n, n], comm=comm)
    R.setUp()

    right_full = gather_full_vec_numpy(right)
    lo, hi = left.getOwnershipRange()
    left_loc = left.getArray(readonly=True)

    for i_loc, i_glob in enumerate(range(lo, hi)):
        row = left_loc[i_loc] * np.conjugate(right_full)
        nz = np.flatnonzero(np.abs(row) > 1e-14)
        if nz.size:
            R.setValues([i_glob], nz.tolist(), row[nz])

    R.assemble()
    return R

# ============================================================
# Closed-loop static output feedback wrapper
# ============================================================

class ClosedLoopSOF:
    """
    Static output feedback:
        u = feedback_sign * k * y
        y = c^H q
    so
        Acl(k) = A + feedback_sign * k * b c^H

    Assumptions:
    - k is scalar
    - disturbance/output weighting remains the same weighted resolvent as in your codei
    - A, M are already restricted to the free dofs
    - b, c are vectors on the same reduced space
    """

    def __init__(
        self,
        A: PETSc.Mat,
        M: PETSc.Mat,
        b: PETSc.Vec,
        c: PETSc.Vec,
        feedback_sign: float = -1.0,
    ):
        self.A = A
        self.M = M
        self.b = b.copy()
        self.c = c.copy()
        self.feedback_sign = float(feedback_sign)

        self.BcH = make_rank1_mat_from_vecs(self.b, self.c)

    def Acl(self, k: float) -> PETSc.Mat:
        Acl = self.A.copy()
        Acl.axpy(self.feedback_sign * float(k), self.BcH)
        Acl.assemble()
        return Acl

    # --------------------------------------------------------
    # Closed-loop Hinf ingredients
    # --------------------------------------------------------

    def gain_of_omega(self, k: float, omega: float) -> float:
        Acl = self.Acl(k)
        return gain_of_omega(Acl, self.M, float(omega))

    def hinf_branch_value_grad(self, k: float, omega: float,
                               eps_tol=1e-8, max_it=400, ncv=40):
        """
        Value and scalar gradient of the fixed-frequency branch
            g(k, omega) = sigma_max( R_cl(i omega, k) )

        Uses the generalized Hermitian EVP
            A_omega v = lambda M v,
        where lambda = sigma^2.

        Returns
        -------
        sigma : float
        dsigma_dk : float
        """
        Acl = self.Acl(k)
        Aom, ctx, S = make_Aomega_shell_ctx(Acl, self.M, float(omega))

        eps = SLEPc.EPS().create(comm=self.M.comm)
        eps.setOperators(Aom, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError(f"GHEP did not converge at omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = max(float(np.real(lam)), 0.0)
        sigma = float(np.sqrt(lam))

        # generalized eigenvector v: Aom v = lambda M v
        v = S.createVecRight()
        eps.getEigenvector(0, v)

        # normalize so that v^H M v = 1
        Mv = v.duplicate()
        self.M.mult(v, Mv)
        den = v.dot(Mv)
        scale = 1.0 / np.sqrt(np.real(den))
        v.scale(scale)

        # x = (i omega M - Acl)^(-1) v
        x = S.createVecRight()
        ctx.ksp.solve(v, x)

        # y = (1/sigma) (i omega M - Acl)^(-H) M x
        Mx = x.duplicate()
        self.M.mult(x, Mx)

        y = S.createVecRight()
        ctx.solve_hermitian(Mx, y)
        if sigma > 0.0:
            y.scale(1.0 / sigma)
        else:
            y.set(0.0)

        # dsigma/dk = feedback_sign * Re( (b^H y) (c^H x) )
        dsigma_dk = self.feedback_sign * np.real(self.b.dot(y) * self.c.dot(x))

        return sigma, float(dsigma_dk)

    def hinf_oracle(self, k: float,
                    num_guess=4, wmin=1e-6, wmax=1e2, seed=1,
                    branch_eps_tol=1e-8, branch_max_it=400, branch_ncv=40):
        """
        Outer oracle for the Hinf objective:
        - builds Acl(k)
        - runs your existing multistart Newton on omega
        - returns gamma_star and dgamma_star/dk using the active branch gradient
        """
        Acl = self.Acl(k)

        omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
        omega_star, gamma_star, local_results = optimize_peak_multistart(Acl, self.M, omegas0)

        gamma_branch, dgamma_dk = self.hinf_branch_value_grad(
            k, omega_star,
            eps_tol=branch_eps_tol,
            max_it=branch_max_it,
            ncv=branch_ncv,
        )

        return {
            "omega_star": float(omega_star),
            "gamma_star": float(gamma_branch),
            "dgamma_dk": float(dgamma_dk),
            "local_results": local_results,
        }
    def objective_constraint_oracle_newton_disk(
        self,
        k: float,
        stab_margin=1e-5,
        num_guess=8,
        wmin=1e-6,
        wmax=1e2,
        seed=1,
        disk_b=10.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
    ):
        h = self.hinf_value_grad_newton_disk(
            k,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        alpha, dalpha_dk = self.spectral_abscissa_and_grad(k)

        return {
            "f": h["gamma_star"],
            "dfdk": h["dgamma_dk"],
            "omega_star": h["omega_star"],
            "alpha": alpha,
            "c": alpha + float(stab_margin),
            "dcdk": dalpha_dk,
            "hinf_info": h,
        }
    # --------------------------------------------------------
    # Closed-loop spectral abscissa ingredients
    # --------------------------------------------------------

    def spectral_abscissa_and_grad(self, k: float,
                                   eps_tol=1e-8, max_it=400, ncv=40):
        """
        Returns
        -------
        alpha : float
            spectral abscissa of the generalized EVP Acl v = lambda M v
        dalpha_dk : float
            gradient of the active simple branch wrt scalar k

        Formula:
            lambda'(k) = u^H (dAcl/dk) v / (u^H M v)
                       = feedback_sign * (u^H b)(c^H v) / (u^H M v)
        """
        Acl = self.Acl(k)

        eps = SLEPc.EPS().create(comm=self.M.comm)
        eps.setOperators(Acl, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.setTwoSided(True)
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError("GNHEP did not converge for spectral abscissa")

        lam = eps.getEigenvalue(0)

        v = Acl.createVecRight()
        u = Acl.createVecLeft()
        eps.getEigenvector(0, v)
        eps.getLeftEigenvector(0, u)

        Mv = v.duplicate()
        self.M.mult(v, Mv)
        denom = u.dot(Mv)

        if abs(denom) < 1e-30:
            raise RuntimeError("u^H M v is too small to normalize spectral-abscissa gradient")

        dlam_dk = self.feedback_sign * (u.dot(self.b) * self.c.dot(v)) / denom
        dalpha_dk = float(np.real(dlam_dk))

        return float(np.real(lam)), dalpha_dk

    # --------------------------------------------------------
    # Single scalar controller oracle for optimization
    # --------------------------------------------------------

    def objective_constraint_oracle(self, k: float,
                                    stab_margin=1e-8,
                                    num_guess=4, wmin=1e-6, wmax=1e2, seed=1):
        """
        Returns the scalar objective/constraint info you need at one k:

            f(k)   = Hinf norm
            df/dk  = branch gradient at active peak
            c(k)   = alpha(k) + stab_margin   <= 0
            dc/dk  = spectral-abscissa branch gradient
        """
        h = self.hinf_oracle(k, num_guess=num_guess, wmin=wmin, wmax=wmax, seed=seed)
        alpha, dalpha_dk = self.spectral_abscissa_and_grad(k)

        return {
            "f": h["gamma_star"],
            "dfdk": h["dgamma_dk"],
            "omega_star": h["omega_star"],
            "alpha": alpha,
            "c": alpha + float(stab_margin),
            "dcdk": dalpha_dk,
            "hinf_info": h,
        }
    
## Restrict actuator/sensor vectors to the same free dofs as A and M
b = restrict_vec_to_is(B_load, is_free)
c = restrict_vec_to_is(s_vec,  is_free)

## Negative feedback: u = -k y
#cl = ClosedLoopSOF(A, M, b, c, feedback_sign=-1.0)

#k0 = 0.0

## one frequency sample
#g0 = cl.gain_of_omega(k0, omega=1.0)
#if IS_WORLD_ROOT:
    #print("closed-loop gain at omega=1:", g0)

## one full oracle call
#out = cl.objective_constraint_oracle(
    #k0,
    #stab_margin=1e-8,
    #num_guess=1,
    #wmin=1e-6,
    #wmax=1e2,
    #seed=2,
#)

#if IS_WORLD_ROOT:
    #print("f(k0)        =", out["f"])
    #print("df/dk        =", out["dfdk"])
    #print("omega_star   =", out["omega_star"])
    #print("alpha(k0)    =", out["alpha"])
    #print("c(k0)        =", out["c"])
    #print("dc/dk        =", out["dcdk"])

# ============================================================
# Closed-loop gradients for scalar static output feedback
# ============================================================

class ClosedLoopSOFF:
    """
    Static output feedback:
        u = feedback_sign * k * y
        y = c^H q

    so
        Acl(k) = A + feedback_sign * k * b c^H

    Here:
    - A, M are PETSc matrices on the reduced/free space
    - b, c are PETSc vectors on that same space
    - k is scalar
    """

    def __init__(self, A, M, b, c, feedback_sign=-1.0):
        self.A = A
        self.M = M
        self.b = b.copy()
        self.c = c.copy()
        self.feedback_sign = float(feedback_sign)

        # explicit rank-1 matrix b c^H for a first working version
        self.BcH = make_rank1_mat_from_vecs(self.b, self.c)

    def Acl(self, k: float) -> PETSc.Mat:
        Acl = self.A.copy()
        Acl.axpy(self.feedback_sign * float(k), self.BcH)
        Acl.assemble()
        return Acl

    def gain_of_omega(self, k: float, omega: float) -> float:
        Acl = self.Acl(k)
        return gain_of_omega(Acl, self.M, float(omega))

    def spectral_abscissa_and_grad(
        self,
        k: float,
        eps_tol=1e-8,
        max_it=400,
        ncv=40,
    ):
        """
        Returns
        -------
        alpha : float
            spectral abscissa of Acl(k) v = lambda M v
        dalpha_dk : float
            branch gradient of Re(lambda) wrt scalar k

        Formula:
            d lambda / dk
              = feedback_sign * (u^H b) (c^H v) / (u^H M v)

        where
            Acl v = lambda M v,
            u^H Acl = lambda u^H M.
        """
        Acl = self.Acl(k)

        eps = SLEPc.EPS().create(comm=self.M.comm)
        eps.setOperators(Acl, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setTwoSided(True)   # needed for left eigenvectors
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError("GNHEP did not converge for spectral abscissa")

        lam = eps.getEigenvalue(0)

        v = Acl.createVecRight()
        u = Acl.createVecLeft()
        eps.getEigenvector(0, v)
        eps.getLeftEigenvector(0, u)

        Mv = v.duplicate()
        self.M.mult(v, Mv)

        denom = u.dot(Mv)   # u^H M v
        if abs(denom) < 1e-30:
            raise RuntimeError("u^H M v is too small in spectral_abscissa_and_grad")

        dlam_dk = self.feedback_sign * (u.dot(self.b) * self.c.dot(v)) / denom
        dalpha_dk = float(np.real(dlam_dk))

        return float(np.real(lam)), dalpha_dk

    def hinf_branch_value_grad(
        self,
        k: float,
        omega: float,
        eps_tol=1e-8,
        max_it=400,
        ncv=40,
    ):
        """
        Fixed-frequency branch:
            g_omega(k) = sigma_max( R_cl(i omega, k) )

        with
            R_cl(s,k) = M^{1/2} (s M - Acl(k))^{-1} M^{-1/2}.

        We use the generalized Hermitian EVP
            A_omega v = lambda M v,
            A_omega = S^{-H} M S^{-1},
            S = i omega M - Acl(k),

        so lambda = g_omega(k)^2.

        Returns
        -------
        sigma : float
            g_omega(k)
        dsigma_dk : float
            branch gradient wrt scalar k

        Implemented formula:
            dsigma/dk = feedback_sign * Re( (b^H y) (c^H x) )

        with
            x = S^{-1} v,
            y = (1/sigma) S^{-H} M x,
            v^H M v = 1.
        """
        Acl = self.Acl(k)
        Aom, ctx, S = make_Aomega_shell_ctx(Acl, self.M, float(omega))

        eps = SLEPc.EPS().create(comm=self.M.comm)
        eps.setOperators(Aom, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError(f"GHEP did not converge at omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = max(float(np.real(lam)), 0.0)
        sigma = float(np.sqrt(lam))

        # generalized eigenvector v, normalized by v^H M v = 1
        v = S.createVecRight()
        eps.getEigenvector(0, v)

        Mv = v.duplicate()
        self.M.mult(v, Mv)
        den = v.dot(Mv)
        if abs(den) < 1e-30:
            raise RuntimeError("v^H M v is too small in hinf_branch_value_grad")

        v.scale(1.0 / np.sqrt(np.real(den)))

        # x = S^{-1} v
        x = S.createVecRight()
        ctx.ksp.solve(v, x)

        # y = (1/sigma) S^{-H} M x
        Mx = x.duplicate()
        self.M.mult(x, Mx)

        y = S.createVecRight()
        ctx.solve_hermitian(Mx, y)

        if sigma > 1e-30:
            y.scale(1.0 / sigma)
            dsigma_dk = self.feedback_sign * float(np.real(y.dot(self.b) * self.c.dot(x)))
        else:
            dsigma_dk = 0.0

        return sigma, dsigma_dk

    def hinf_value_grad(
        self,
        k: float,
        num_guess=1,
        wmin=1e-6,
        wmax=1e2,
        seed=1,
        branch_eps_tol=1e-8,
        branch_max_it=400,
        branch_ncv=40,
    ):
        """
        Full Hinf objective oracle for scalar k:
        - run multistart Newton on omega for Acl(k)
        - evaluate branch value/gradient at the active peak

        In the smooth case (unique active peak, simple singular value),
        this is the gradient of the full Hinf norm.
        """
        Acl = self.Acl(k)

        omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
        omega_star, gamma_star, local_results = optimize_peak_multistart(Acl, self.M, omegas0)

        gamma_branch, dgamma_dk = self.hinf_branch_value_grad(
            k,
            omega_star,
            eps_tol=branch_eps_tol,
            max_it=branch_max_it,
            ncv=branch_ncv,
        )

        return {
            "omega_star": float(omega_star),
            "gamma_star": float(gamma_branch),
            "dgamma_dk": float(dgamma_dk),
            "local_results": local_results,
        }

    def objective_constraint_oracle(
        self,
        k: float,
        stab_margin=1e-8,
        num_guess=1,
        wmin=1e-6,
        wmax=1e2,
        seed=1,
    ):
        """
        Returns the scalar objective/constraint data:
            f(k)   = Hinf norm
            df/dk  = Hinf branch gradient at active peak
            c(k)   = alpha(k) + stab_margin <= 0
            dc/dk  = spectral-abscissa branch gradient
        """
        h = self.hinf_value_grad(
            k,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
        )

        alpha, dalpha_dk = self.spectral_abscissa_and_grad(k)

        return {
            "f": h["gamma_star"],
            "dfdk": h["dgamma_dk"],
            "omega_star": h["omega_star"],
            "alpha": alpha,
            "c": alpha + float(stab_margin),
            "dcdk": dalpha_dk,
            "hinf_info": h,
        }
    def objective_constraint_oracle_newton_disk(
        self,
        k: float,
        stab_margin=1e-5,
        num_guess=8,
        wmin=1e-6,
        wmax=1e2,
        seed=1,
        disk_b=10.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
    ):
        h = self.hinf_value_grad_newton_disk(
            k,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        alpha, dalpha_dk = self.spectral_abscissa_and_grad(k)

        return {
            "f": h["gamma_star"],
            "dfdk": h["dgamma_dk"],
            "omega_star": h["omega_star"],
            "alpha": alpha,
            "c": alpha + float(stab_margin),
            "dcdk": dalpha_dk,
            "hinf_info": h,
        }
    def hinf_value_grad_newton_disk(
        self,
        k: float,
        num_guess=8,
        wmin=1e-6,
        wmax=1e2,
        seed=1,
        branch_eps_tol=1e-8,
        branch_max_it=400,
        branch_ncv=40,
        disk_b=10.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
        disk_midpoint_guard=True,
    ):
        """
        Hinf oracle:
        1) Newton/multistart gives a candidate peak
        2) safe disk validator checks whether a larger peak is present
        3) if disk finds a larger peak, use that violating frequency instead

        Returns
        -------
        dict with gamma_star, omega_star, dgamma_dk, certified, cert_info
        """
        Acl = self.Acl(k)

        # step 1: candidate from multistart Newton
        omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
        w_best, g_best, local_results = optimize_peak_multistart(Acl, self.M, omegas0)

        gamma_star = float(g_best)
        omega_star = float(w_best)

        # step 2: safe disk validation
        certified, cert_info = certify_global_peak_disk_cl_safe(
            self,
            k,
            gamma=gamma_star,
            b=disk_b,
            eigs_tol=disk_eigs_tol,
            nev=disk_nev,
            eps_probe=disk_eps_probe,
            axis_warn=disk_axis_warn,
            midpoint_guard=disk_midpoint_guard,
        )

        # step 3: if disk found a larger point, override candidate
        if not certified:
            if "omega" in cert_info and "gain" in cert_info:
                omega_star = float(cert_info["omega"])
                gamma_star = float(cert_info["gain"])

        # gradient at selected active frequency
        _, dgamma_dk = self.hinf_branch_value_grad(
            k,
            omega_star,
            eps_tol=branch_eps_tol,
            max_it=branch_max_it,
            ncv=branch_ncv,
        )

        return {
            "omega_star": float(omega_star),
            "gamma_star": float(gamma_star),
            "dgamma_dk": float(dgamma_dk),
            "certified": bool(certified),
            "cert_info": cert_info,
            "local_results": local_results,
        }

    def objective_constraint_oracle_newton_disk(
        self,
        k: float,
        stab_margin=1e-5,
        num_guess=8,
        wmin=1e-6,
        wmax=1e2,
        seed=1,
        disk_b=10.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
    ):
        h = self.hinf_value_grad_newton_disk(
            k,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        alpha, dalpha_dk = self.spectral_abscissa_and_grad(k)

        return {
            "f": h["gamma_star"],
            "dfdk": h["dgamma_dk"],
            "omega_star": h["omega_star"],
            "alpha": alpha,
            "c": alpha + float(stab_margin),
            "dcdk": dalpha_dk,
            "hinf_info": h,
        }
    
## actuator/sensor restricted to free dofs
#b = restrict_vec_to_is(B_load, is_free)
#c = restrict_vec_to_is(s_vec,  is_free)

## negative feedback u = -k y
#cl = ClosedLoopSOFF(A, M, b, c, feedback_sign=-1.0)

#k0 = 0.0
#out = cl.objective_constraint_oracle(
    #k0,
    #stab_margin=1e-8,
    #num_guess=1,
    #wmin=1e-6,
    #wmax=1e2,
    #seed=2,
#)

#if IS_WORLD_ROOT:
    #print("f(k0)      =", out["f"])
    #print("df/dk      =", out["dfdk"])
    #print("omega_star =", out["omega_star"])
    #print("alpha(k0)  =", out["alpha"])
    #print("c(k0)      =", out["c"])
    #print("dc/dk      =", out["dcdk"])

#def fd_check_scalar_oracle(cl, k, h=1e-6, seed=2):
    #o0 = cl.objective_constraint_oracle(k, seed=seed)
    #op = cl.objective_constraint_oracle(k + h, seed=seed)
    #om = cl.objective_constraint_oracle(k - h, seed=seed)

    #df_fd = (op["f"] - om["f"]) / (2.0 * h)
    #dc_fd = (op["c"] - om["c"]) / (2.0 * h)

    #if IS_WORLD_ROOT:
        #print("\nFD check at k =", k)
        #print("analytic df/dk =", o0["dfdk"])
        #print("FD       df/dk =", df_fd)
        #print("rel err df     =", abs(o0["dfdk"] - df_fd) / (abs(o0["dfdk"]) + abs(df_fd) + 1e-30))
        #print("analytic dc/dk =", o0["dcdk"])
        #print("FD       dc/dk =", dc_fd)
        #print("rel err dc     =", abs(o0["dcdk"] - dc_fd) / (abs(o0["dcdk"]) + abs(dc_fd) + 1e-30))

#fd_check_scalar_oracle(cl,k0)


def make_pygranso_combined_fn(
    cl,
    stab_margin=1e-3,
    num_guess=1,
    wmin=1e-6,
    wmax=1e2,
    seed=2,
    verbose=False,
):
    """
    Wraps your PETSc/SLEPc oracle into a PyGRANSO combined_fn for scalar k.

    Returns:
        combined_fn(X_struct) -> [f, f_grad, ci, ci_grad, ce, ce_grad]
    """

    def combined_fn(X_struct):
        # PyGRANSO variable
        k_torch = X_struct.k
        k = float(k_torch.item())

        out = cl.objective_constraint_oracle(
            k,
            stab_margin=stab_margin,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
        )

        f_val = float(out["f"])
        dfdk  = float(out["dfdk"])
        c_val = float(out["c"])
        dcdk  = float(out["dcdk"])

        if verbose and IS_WORLD_ROOT:
            print(
                f"[oracle] k={k:+.6e}  "
                f"f={f_val:.6e}  df/dk={dfdk:.6e}  "
                f"alpha={out['alpha']:.6e}  c={c_val:.6e}  dc/dk={dcdk:.6e}  "
                f"omega*={out['omega_star']:.6e}"
            )

        # Manual-gradient mode:
        # return objective/constraint values + gradients as torch tensors
        dev = k_torch.device
        dt  = k_torch.dtype

        f = f_val
        f_grad = torch.tensor([[dfdk]], device=dev, dtype=dt)

        ci = torch.tensor([[c_val]], device=dev, dtype=dt)
        ci_grad = torch.tensor([[dcdk]], device=dev, dtype=dt)

        ce = None
        ce_grad = None

        return [f, f_grad, ci, ci_grad, ce, ce_grad]

    return combined_fn


#device = torch.device("cpu")
#torch_dtype = torch.double

## scalar variable k
#var_spec = {"k": [1, 1]}

## initial guess
#k0 = 0.0

#opts = pygransoStruct()
#opts.torch_device = device
#opts.double_precision = True
#opts.globalAD = False
#opts.maxit = 50
#opts.print_frequency = 1
#opts.print_level = 1
#opts.x0 = torch.tensor([[k0]], device=device, dtype=torch_dtype)

#comb_fn = make_pygranso_combined_fn(
    #cl,
    #stab_margin=1e-8,
    #num_guess=1,
    #wmin=1e-6,
    #wmax=1e2,
    #seed=2,
    #verbose=True,
#)

#soln = pygranso(
    #var_spec=var_spec,
    #combined_fn=comb_fn,
    #user_opts=opts,
#)

#if IS_WORLD_ROOT:
    #print("\n=== PyGRANSO result ===")
    #print("final x =", soln.final.x)

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def sample_gain_curve(cl, k, ws):
    vals = []
    for w in ws:
        vals.append(cl.gain_of_omega(k, float(w)))
    return np.array(vals, dtype=float)

def summarize_controller(cl, k, num_guess=6, wmin=1e-6, wmax=1e2, seed=2):
    out = cl.objective_constraint_oracle(
        k,
        stab_margin=1e-8,
        num_guess=num_guess,
        wmin=wmin,
        wmax=wmax,
        seed=seed,
    )
    return {
        "k": k,
        "hinf": out["f"],
        "omega_star": out["omega_star"],
        "alpha": out["alpha"],
        "constraint": out["c"],
    }

def plot_gain_comparison(cl, k_list, labels=None, wmin=1e-4, wmax=1e2, npts=600):
    comm = WORLD
    ws_pos = np.logspace(np.log10(wmin), np.log10(wmax), npts)
    ws = np.concatenate([-ws_pos[::-1], [0.0], ws_pos])

    curves = []
    summaries = []

    for k in k_list:
        vals = sample_gain_curve(cl, k, ws)
        curves.append(vals)
        summaries.append(summarize_controller(cl, k))

    if IS_WORLD_ROOT:
        if labels is None:
            labels = [f"k={k:.4e}" for k in k_list]

        plt.figure(figsize=(8, 5))
        for vals, lab, summ in zip(curves, labels, summaries):
            plt.semilogy(ws, vals, label=f"{lab}  ($H_\\infty$={summ['hinf']:.3e})")
            plt.scatter([summ["omega_star"]], [summ["hinf"]], s=35, zorder=5)

        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$\|R_{cl}(i\omega)\|$")
        plt.title("Closed-loop gain curve")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\n=== Controller summaries ===")
        for summ, lab in zip(summaries, labels):
            print(
                f"{lab:15s}  "
                f"Hinf={summ['hinf']:.6e}  "
                f"omega*={summ['omega_star']:.6e}  "
                f"alpha={summ['alpha']:.6e}  "
                f"c={summ['constraint']:.6e}"
            )

    return ws, curves, summaries

## open loop = k = 0
#k_open = 0.0

## replace by your current candidate
#k_test = 1.383047

#ws, curves, summaries = plot_gain_comparison(
    #cl,
    #[k_open, k_test],
    #labels=["open loop", "current closed loop"],
    #wmin=1e-4,
    #wmax=1e2,
    #npts=500,
#)

def deduplicate_local_peaks(local_results, freq_tol=1e-3, gain_tol_rel=1e-3):
    """
    local_results: list of tuples (g, w, info) from optimize_peak_multistart

    Returns a deduplicated list of peaks sorted by gain descending:
        [{"gain": g, "omega": w, "info": info, "count": m}, ...]
    """
    cleaned = []
    for g, w, info in local_results:
        if not np.isfinite(g):
            continue
        cleaned.append({"gain": float(g), "omega": float(w), "info": info, "count": 1})

    cleaned.sort(key=lambda p: p["gain"], reverse=True)

    peaks = []
    for cand in cleaned:
        merged = False
        for p in peaks:
            same_freq = abs(cand["omega"] - p["omega"]) <= freq_tol
            close_gain = abs(cand["gain"] - p["gain"]) <= gain_tol_rel * max(1.0, p["gain"])
            if same_freq and close_gain:
                p["count"] += 1
                # keep the better of the two
                if cand["gain"] > p["gain"]:
                    p["gain"] = cand["gain"]
                    p["omega"] = cand["omega"]
                    p["info"] = cand["info"]
                merged = True
                break
        if not merged:
            peaks.append(cand)

    peaks.sort(key=lambda p: p["gain"], reverse=True)
    return peaks


def select_top_peaks_for_boyd(
    peaks,
    top_n=3,
    rel_keep=0.05,
    abs_keep=None,
):
    """
    Keep up to top_n peaks, but only those not too far below the best one.

    rel_keep=0.05 keeps peaks with gain >= 95% of best.
    If you want literally the top_n no matter what, set rel_keep=None.
    """
    if len(peaks) == 0:
        return []

    gamma_star = peaks[0]["gain"]

    out = []
    for p in peaks:
        keep = True
        if rel_keep is not None:
            keep = keep and (p["gain"] >= (1.0 - rel_keep) * gamma_star)
        if abs_keep is not None:
            keep = keep and (gamma_star - p["gain"] <= abs_keep)
        if keep:
            out.append(p)
        if len(out) >= top_n:
            break
    if len(out) == 0:
        out = [peaks[0]]

    return out


def few_boyd_checks_at_peaks(
    A, M,
    gamma_star,
    peaks,
    rel_increases=(1e-3, 5e-3, 1e-2),
    nev=12,
    eigs_tol=1e-7,
    axis_tol=1e-8,
):
    """
    Run the cheaper Boyd/Balakrishnan-style check at several candidate peak shifts.

    We test the SAME gamma_star (inflated by rel_increase) near each selected peak.
    If any check detects an imag-axis hit, we declare 'can increase'.

    Returns
    -------
    can_increase : bool
    info : dict
    """
    checks = []

    for peak in peaks:
        omega_shift = float(peak["omega"])
        gain_peak = float(peak["gain"])

        for rel in rel_increases:
            hit, info = one_boyd_check_can_increase_gamma(
                A, M,
                gamma_star=gamma_star,
                omega_star=omega_shift,
                rel_increase=rel,
                nev=nev,
                eigs_tol=eigs_tol,
                axis_tol=axis_tol,
            )

            checks.append({
                "omega_shift": omega_shift,
                "peak_gain": gain_peak,
                "rel_increase": rel,
                "hit": hit,
                "info": info,
            })

            if hit:
                return True, {
                    "status": "can_increase",
                    "gamma_star": float(gamma_star),
                    "checks": checks,
                }

    return False, {
        "status": "no_hit_found",
        "gamma_star": float(gamma_star),
        "checks": checks,
    }
def hinf_newton_plus_few_boyd_checks_cl(
    cl,
    k,
    num_guess=12,
    wmin=1e-6,
    wmax=1e2,
    seed=1,
    top_n=3,
    rel_keep=None,
    rel_increases=(1e-3, 5e-3, 1e-2),
    freq_tol=1e-3,
    gain_tol_rel=1e-3,
    boyd_nev=12,
    boyd_eigs_tol=1e-7,
    boyd_axis_tol=1e-8,
):
    Acl = cl.Acl(k)

    omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
    w_best, g_best, local_results = optimize_peak_multistart(Acl, cl.M, omegas0)

    peaks = deduplicate_local_peaks(
        local_results,
        freq_tol=freq_tol,
        gain_tol_rel=gain_tol_rel,
    )

    selected_peaks = select_top_peaks_for_boyd(
        peaks,
        top_n=top_n,
        rel_keep=rel_keep,
    )

    can_increase, boyd_info = few_boyd_checks_at_peaks(
        Acl, cl.M,
        gamma_star=g_best,
        peaks=selected_peaks,
        rel_increases=rel_increases,
        nev=boyd_nev,
        eigs_tol=boyd_eigs_tol,
        axis_tol=boyd_axis_tol,
    )

    return {
        "omega_star": float(w_best),
        "gamma_star": float(g_best),
        "boyd_passed": not can_increase,
        "boyd_info": boyd_info,
        "all_local_results": local_results,
        "dedup_peaks": peaks,
        "selected_peaks": selected_peaks,
        "initial_guesses": omegas0,
    }

def hinf_newton_plus_few_boyd_checks(
    A, M,
    num_guess=12,
    wmin=1e-6,
    wmax=1e2,
    seed=1,
    top_n=3,
    rel_keep=0.05,
    rel_increases=(1e-3, 5e-3, 1e-2),
    freq_tol=1e-3,
    gain_tol_rel=1e-3,
    boyd_nev=12,
    boyd_eigs_tol=1e-7,
    boyd_axis_tol=1e-8,
):
    """
    1) multistart Newton
    2) deduplicate local peaks
    3) select top few candidate peaks
    4) run a few Boyd checks at those shifts

    Returns a richer diagnostics dictionary.
    """
    omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
    w_best, g_best, local_results = optimize_peak_multistart(A, M, omegas0)

    peaks = deduplicate_local_peaks(
        local_results,
        freq_tol=freq_tol,
        gain_tol_rel=gain_tol_rel,
    )

    selected_peaks = select_top_peaks_for_boyd(
        peaks,
        top_n=top_n,
        rel_keep=rel_keep,
    )

    can_increase, boyd_info = few_boyd_checks_at_peaks(
        A, M,
        gamma_star=g_best,
        peaks=selected_peaks,
        rel_increases=rel_increases,
        nev=boyd_nev,
        eigs_tol=boyd_eigs_tol,
        axis_tol=boyd_axis_tol,
    )

    return {
        "omega_star": float(w_best),
        "gamma_star": float(g_best),
        "boyd_passed": not can_increase,
        "boyd_info": boyd_info,
        "all_local_results": local_results,
        "dedup_peaks": peaks,
        "selected_peaks": selected_peaks,
        "initial_guesses": omegas0,
    }


#res = hinf_newton_plus_few_boyd_checks(
    #A, M,
    #num_guess=16,
    #wmin=1e-6,
    #wmax=1e2,
    #seed=2,
    #top_n=3,
    #rel_keep=None,                  # keep literally the top 3
    #rel_increases=(1e-3, 5e-3),     # cheap first pass
    #boyd_nev=12,
    #boyd_eigs_tol=1e-7,
    #boyd_axis_tol=1e-8,
#)

#if IS_WORLD_ROOT:
    #print("omega_star   =", res["omega_star"])
    #print("gamma_star   =", res["gamma_star"])
    #print("boyd_passed  =", res["boyd_passed"])
    #print("selected peaks:")
    #for p in res["selected_peaks"]:
        #print(f"  omega={p['omega']:+.6e}, gain={p['gain']:.6e}, count={p['count']}")
    #print("boyd_info status =", res["boyd_info"]["status"])


#res = hinf_newton_plus_few_boyd_checks_cl(
    #cl,
    #k_test,
    #num_guess=16,
    #wmin=1e-6,
    #wmax=1e2,
    #seed=2,
    #top_n=3,
    #rel_keep=None,
#)

#if IS_WORLD_ROOT:
    #print("omega_star   =", res["omega_star"])
    #print("gamma_star   =", res["gamma_star"])
    #print("boyd_passed  =", res["boyd_passed"])
    #print("selected peaks:")
    #for p in res["selected_peaks"]:
        #print(f"  omega={p['omega']:+.6e}, gain={p['gain']:.6e}, count={p['count']}")
    #print("boyd_info status =", res["boyd_info"]["status"])


#res_peak = hinf_newton_plus_few_boyd_checks_cl(
    #cl,
    #k_test,
    #num_guess=16,
    #wmin=1e-6,
    #wmax=1e2,
    #seed=2,
#)

#gamma_star = res_peak["gamma_star"]

#certified, cert_info = certify_global_peak_disk_cl(
    #cl,
    #k_test,
    #gamma=gamma_star,
    #b=10.0,
    #eigs_tol=1e-6,
    #nev=16,
    #eps_eigs=1e-6,
#)

#if IS_WORLD_ROOT:
    #print("gamma_star  =", gamma_star)
    #print("certified   =", certified)
    #print("cert_info   =", cert_info["reason"])

import numpy as np

def make_signed_log_grid(wmin=1e-4, wmax=1e2, npos=400):
    ws_pos = np.logspace(np.log10(wmin), np.log10(wmax), npos)
    ws = np.concatenate([-ws_pos[::-1], [0.0], ws_pos])
    return ws

def sweep_gain_and_derivative_cl(
    cl,
    k,
    ws,
    rel_step=1e-6,
):
    """
    Sweep the closed-loop operator Acl(k) over frequencies ws and compute:
        g(omega), lambda'(omega), lambda''(omega)

    using your existing get_lambda_dot(A, M, omega) routine.
    """
    Acl = cl.Acl(k)

    gains = []
    lamdots = []
    lamddots = []

    for w in ws:
        g, gdot, lam, lamdot, lamddot = get_lambda_dot(Acl, cl.M, float(w))
        gains.append(float(g))
        lamdots.append(float(lamdot))
        lamddots.append(float(lamddot))

    return np.array(gains), np.array(lamdots), np.array(lamddots)

def detect_peak_brackets_from_lamdot(
    ws,
    gains,
    lamdots,
    lamddots=None,
    deriv_tol=1e-8,
    gain_floor_rel=1e-3,
):
    """
    Detect candidate peak brackets using sign change of lambda'(omega):
        lambda' > 0 before, lambda' < 0 after

    Also keeps near-zero derivative points if curvature is negative.
    """
    ws = np.asarray(ws)
    gains = np.asarray(gains)
    lamdots = np.asarray(lamdots)

    gmax = np.max(gains)
    brackets = []

    def sgn(x):
        if x > deriv_tol:
            return 1
        if x < -deriv_tol:
            return -1
        return 0

    for i in range(len(ws) - 1):
        if max(gains[i], gains[i+1]) < gain_floor_rel * gmax:
            continue

        s0 = sgn(lamdots[i])
        s1 = sgn(lamdots[i+1])

        # strict + to - sign change
        if s0 > 0 and s1 < 0:
            brackets.append({
                "type": "sign_change",
                "i": i,
                "w_left": float(ws[i]),
                "w_right": float(ws[i+1]),
                "w0": float(0.5 * (ws[i] + ws[i+1])),
                "g_left": float(gains[i]),
                "g_right": float(gains[i+1]),
            })
            continue

        # catch near-stationary sampled points with negative curvature
        if lamddots is not None:
            if abs(lamdots[i]) <= deriv_tol and lamddots[i] < 0 and gains[i] >= gain_floor_rel * gmax:
                brackets.append({
                    "type": "near_stationary",
                    "i": i,
                    "w_left": float(ws[max(i-1, 0)]),
                    "w_right": float(ws[min(i+1, len(ws)-1)]),
                    "w0": float(ws[i]),
                    "g_left": float(gains[max(i-1, 0)]),
                    "g_right": float(gains[min(i+1, len(ws)-1)]),
                })

    return brackets

def refine_peak_brackets_with_newton(
    cl,
    k,
    brackets,
    max_refine=None,
):
    """
    Refine each bracket midpoint with local_newton_peak on Acl(k).
    """
    Acl = cl.Acl(k)
    if max_refine is not None:
        brackets = brackets[:max_refine]

    local_results = []
    for b in brackets:
        w0 = float(b["w0"])
        try:
            w, g, info = local_newton_peak(Acl, cl.M, w0)
            local_results.append((g, w, info))
        except Exception as e:
            local_results.append((-np.inf, w0, {"status": f"failed: {e}"}))

    peaks = deduplicate_local_peaks(local_results, freq_tol=1e-3, gain_tol_rel=1e-3)
    return local_results, peaks

def scan_and_refine_peaks_cl(
    cl,
    k,
    wmin=1e-4,
    wmax=1e2,
    npos=600,
    deriv_tol=1e-8,
    gain_floor_rel=1e-3,
    max_refine=20,
):
    """
    Full workflow:
      1) sweep closed-loop gain and derivative
      2) detect peak brackets from lambda' sign changes
      3) refine with Newton
    """
    ws = make_signed_log_grid(wmin=wmin, wmax=wmax, npos=npos)
    gains, lamdots, lamddots = sweep_gain_and_derivative_cl(cl, k, ws)

    brackets = detect_peak_brackets_from_lamdot(
        ws, gains, lamdots, lamddots=lamddots,
        deriv_tol=deriv_tol,
        gain_floor_rel=gain_floor_rel,
    )

    local_results, peaks = refine_peak_brackets_with_newton(
        cl, k, brackets, max_refine=max_refine
    )

    return {
        "ws": ws,
        "gains": gains,
        "lamdots": lamdots,
        "lamddots": lamddots,
        "brackets": brackets,
        "local_results": local_results,
        "peaks": peaks,
    }


#scan = scan_and_refine_peaks_cl(
    #cl,
    #k_test,
    #wmin=1e-4,
    #wmax=2e1,
    #npos=800,
    #deriv_tol=1e-7,
    #gain_floor_rel=1e-3,
    #max_refine=30,
#)

#if IS_WORLD_ROOT:
    #print("Detected derivative brackets:")
    #for b in scan["brackets"][:20]:
        #print(f"{b['type']:15s}  [{b['w_left']:+.6e}, {b['w_right']:+.6e}]  w0={b['w0']:+.6e}")

    #print("\nRefined peaks:")
    #for p in scan["peaks"]:
        #print(f"omega={p['omega']:+.6e}, gain={p['gain']:.6e}, count={p['count']}")



#import matplotlib.pyplot as plt

#if IS_WORLD_ROOT:
    #ws = scan["ws"]
    #gains = scan["gains"]
    #lamdots = scan["lamdots"]

    #plt.figure(figsize=(8,5))
    #plt.semilogy(ws, gains)
    #for p in scan["peaks"]:
        #plt.scatter([p["omega"]], [p["gain"]], s=40)
    #plt.xlabel(r"$\omega$")
    #plt.ylabel(r"$\|G_{cl}(i\omega)\|$")
    #plt.title("Closed-loop gain with refined peaks")
    #plt.grid(True, which="both", ls="--", alpha=0.4)
    #plt.tight_layout()
    #plt.show()

    #plt.figure(figsize=(8,5))
    #plt.plot(ws, lamdots)
    #plt.axhline(0.0, linestyle="--")
    #plt.xlabel(r"$\omega$")
    #plt.ylabel(r"$\lambda'(\omega)$")
    #plt.title(r"Derivative sweep: sign changes bracket peaks")
    #plt.grid(True, alpha=0.4)
    #plt.tight_layout()
    #plt.show()
#w_sus = -3.072484

#for dw in [-1e-2, -5e-3, -1e-3, -5e-4, 0.0, 5e-4, 1e-3, 5e-3, 1e-2]:
    #w = w_sus + dw
    #g = cl.gain_of_omega(k_test, w)
    #if IS_WORLD_ROOT:
        #print(f"w={w:+.9e}, g={g:.9e}")


#theta = -3.072484
#gamma = 98.16936476559681

#Acl = cl.Acl(k_test)
#N = build_N_from_M(cl.M)
#Mgam = build_Mgamma(Acl, cl.M, gamma)

#N = N.convert("aij"); N.assemble()
#Mgam = Mgam.convert("aij"); Mgam.assemble()

#lams, nconv = eigs_close_to_shift_pencil(
    #N, Mgam, theta,
    #nev=80,          
    #tol=5e-1,
    #max_it=2000
#)

#if IS_WORLD_ROOT:
    #print("nconv =", nconv)
    #for lam in lams[:20]:
        #print("lam =", lam, "dist to i theta =", abs(lam - 1j*theta))

import numpy as np

def make_signed_log_grid(wmin=1e-4, wmax=1e2, npos=500):
    ws_pos = np.logspace(np.log10(wmin), np.log10(wmax), npos)
    return np.concatenate([-ws_pos[::-1], [0.0], ws_pos])

def hinf_value_grad_by_sweep(
    cl,
    k,
    ws,
    branch_eps_tol=1e-8,
    branch_max_it=400,
    branch_ncv=40,
):
    """
    Sweep-based Hinf oracle:
      - evaluate gain on a fixed frequency grid
      - take the maximum sampled gain
      - compute branch gradient at the maximizing sampled frequency

    Returns
    -------
    dict with:
      gamma_star  : max sampled gain
      omega_star  : maximizing sampled frequency
      dgamma_dk   : branch gradient at omega_star
      gains       : sampled gain curve
    """
    gains = []
    for w in ws:
        gains.append(cl.gain_of_omega(k, float(w)))
    gains = np.array(gains, dtype=float)

    idx = int(np.argmax(gains))
    omega_star = float(ws[idx])
    gamma_star = float(gains[idx])

    _, dgamma_dk = cl.hinf_branch_value_grad(
        k,
        omega_star,
        eps_tol=branch_eps_tol,
        max_it=branch_max_it,
        ncv=branch_ncv,
    )

    return {
        "gamma_star": gamma_star,
        "omega_star": omega_star,
        "dgamma_dk": float(dgamma_dk),
        "gains": gains,
        "idx_star": idx,
    }
def objective_constraint_oracle_by_sweep(
    cl,
    k,
    ws,
    stab_margin=1e-8,
):
    """
    Objective/constraint oracle using sampled Hinf max instead of Newton peak search.
    """
    h = hinf_value_grad_by_sweep(cl, k, ws)
    alpha, dalpha_dk = cl.spectral_abscissa_and_grad(k)

    return {
        "f": h["gamma_star"],
        "dfdk": h["dgamma_dk"],
        "omega_star": h["omega_star"],
        "alpha": alpha,
        "c": alpha + float(stab_margin),
        "dcdk": dalpha_dk,
        "hinf_info": h,
    }

import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

def make_pygranso_combined_fn_sweep(
    cl,
    ws,
    stab_margin=1e-8,
    verbose=True,
):
    history = []

    def combined_fn(X_struct):
        k_torch = X_struct.k
        k = float(k_torch.item())

        out = objective_constraint_oracle_by_sweep(
            cl,
            k,
            ws,
            stab_margin=stab_margin,
        )

        f_val = float(out["f"])
        dfdk  = float(out["dfdk"])
        c_val = float(out["c"])
        dcdk  = float(out["dcdk"])

        history.append({
            "k": k,
            "f": f_val,
            "dfdk": dfdk,
            "alpha": float(out["alpha"]),
            "c": c_val,
            "dcdk": dcdk,
            "omega_star": float(out["omega_star"]),
            "gains": out["hinf_info"]["gains"].copy(),
            "ws": np.array(ws, copy=True),
        })

        if verbose and IS_WORLD_ROOT:
            print(
                f"[sweep oracle] "
                f"k={k:+.6e}  "
                f"f={f_val:.6e}  "
                f"omega*={out['omega_star']:+.6e}  "
                f"alpha={out['alpha']:.6e}  "
                f"c={c_val:.6e}"
            )

        dev = k_torch.device
        dt = k_torch.dtype

        f = f_val
        f_grad = torch.tensor([[dfdk]], device=dev, dtype=dt)
        ci = torch.tensor([[c_val]], device=dev, dtype=dt)
        ci_grad = torch.tensor([[dcdk]], device=dev, dtype=dt)

        return [f, f_grad, ci, ci_grad, None, None]

    combined_fn.history = history
    return combined_fn

## frequency grid used at every iteration
#ws = make_signed_log_grid(wmin=1e-3, wmax=2e1, npos=500)

## scalar variable k
#var_spec = {"k": [1, 1]}

#device = torch.device("cpu")
#dtype = torch.double

#opts = pygransoStruct()
#opts.torch_device = device
#opts.double_precision = True
#opts.globalAD = False
#opts.maxit = 20
#opts.print_frequency = 1
#opts.print_level = 1
#opts.x0 = torch.tensor([[0.0]], device=device, dtype=dtype)

#comb_fn = make_pygranso_combined_fn_sweep(
    #cl,
    #ws,
    #stab_margin=1e-5,  
    #verbose=True,
#)

#soln = pygranso(
    #var_spec=var_spec,
    #combined_fn=comb_fn,
    #user_opts=opts,
#)

import matplotlib.pyplot as plt

def plot_pygranso_sweep_history(history, every=1):
    if not IS_WORLD_ROOT:
        return

    plt.figure(figsize=(8, 5))
    for i, rec in enumerate(history):
        if i % every != 0:
            continue
        ws = rec["ws"]
        gains = rec["gains"]
        plt.semilogy(ws, gains, label=f"it {i}, k={rec['k']:.3e}")

    plt.xlabel(r"$\omega$")
    plt.ylabel(r"sampled $\|G_{cl}(i\omega)\|$")
    plt.title("Sweep-based gain curve during PyGRANSO iterations")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_pygranso_history_summary(history):
    if not IS_WORLD_ROOT:
        return

    print("\n=== Sweep-based PyGRANSO history ===")
    for i, rec in enumerate(history):
        print(
            f"it={i:02d}  "
            f"k={rec['k']:+.6e}  "
            f"f={rec['f']:.6e}  "
            f"omega*={rec['omega_star']:+.6e}  "
            f"alpha={rec['alpha']:.6e}  "
            f"c={rec['c']:.6e}"
        )


#print_pygranso_history_summary(history)
#plot_pygranso_sweep_history(history, every=1)

def suspicious_intervals_from_disk(
    cl,
    k,
    gamma,
    b=10.0,
    eigs_tol=1e-8,
    nev=60,
    axis_warn=1e-3,
    min_interval=None,
):
    Acl = cl.Acl(k)

    N = build_N_from_M(cl.M)
    Mgam = build_Mgamma(Acl, cl.M, gamma)

    N = N.convert("aij"); N.assemble()
    Mgam = Mgam.convert("aij"); Mgam.assemble()

    if min_interval is None:
        min_interval = eigs_tol

    intervals = [(-float(b), 0.0), (0.0, float(b))]
    suspicious = []

    while intervals:
        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        intervals.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        lo, hi = intervals.pop(0)
        theta = 0.5 * (lo + hi)

        lambdas, nconv = eigs_close_to_shift_pencil(
            N, Mgam, theta,
            nev=nev,
            tol=eigs_tol,
            max_it=2000,
        )

        if nconv == 0 or lambdas.size == 0:
            suspicious.append((lo, hi, theta, "no_eigs"))
            continue

        shift = 1j * theta
        lambdas = lambdas[np.argsort(np.abs(lambdas - shift))]
        nearest = lambdas[0]
        nearest_dist = float(abs(nearest - shift))

        if abs(nearest.real) <= axis_warn:
            suspicious.append((lo, hi, theta, nearest))
            continue

        if np.isfinite(nearest_dist) and nearest_dist > 0.0:
            intervals += split_by_removed_middle(
                lo, hi, theta,
                r=nearest_dist,
                min_progress=0.5 * min_interval
            )

    return suspicious


import numpy as np

def validate_near_shift_candidates(cl, k, gamma, lambdas, ncheck=6, probe_eps=1e-6):
    """
    Probe the gain at the imaginary parts of the first few eigenvalues nearest the shift.
    If any probe exceeds gamma, certification fails.
    """
    for lam in lambdas[:min(len(lambdas), ncheck)]:
        w = float(lam.imag)
        for h in (-probe_eps, 0.0, probe_eps):
            ww = w + h
            g = cl.gain_of_omega(k, ww)
            if g > gamma:
                return True, {
                    "reason": "gain_violation_from_near_shift_eig",
                    "omega": float(ww),
                    "gain": float(g),
                    "lambda_hit": lam,
                }
    return False, None


def certify_global_peak_disk_cl_safe(
    cl,
    k,
    gamma,
    b=10.0,
    eigs_tol=1e-8,
    nev=60,
    eps_probe=1e-6,
    axis_warn=1e-3,
    min_interval=None,
    midpoint_guard=True,
):
    """
    Conservative disk certification on [-b,b].

    Returns
    -------
    certified : bool
    info : dict
    """
    Acl = cl.Acl(k)

    N = build_N_from_M(cl.M)
    Mgam = build_Mgamma(Acl, cl.M, gamma)

    N = N.convert("aij"); N.assemble()
    Mgam = Mgam.convert("aij"); Mgam.assemble()

    if min_interval is None:
        min_interval = eigs_tol

    intervals = [(-float(b), 0.0), (0.0, float(b))]
    visited = []

    # low-frequency guard
    for w in [0.0, -1e-12, 1e-12, -1e-9, 1e-9, -1e-6, 1e-6]:
        g = cl.gain_of_omega(k, w)
        if g > gamma:
            return False, {
                "reason": "low_freq_guard_hit",
                "omega": float(w),
                "gain": float(g),
            }

    while intervals:
        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        intervals.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        lo, hi = intervals.pop(0)
        theta = 0.5 * (lo + hi)

        if midpoint_guard:
            g_theta = cl.gain_of_omega(k, theta)
            if g_theta > gamma:
                return False, {
                    "reason": "midpoint_gain_violation",
                    "omega": float(theta),
                    "gain": float(g_theta),
                }

        lambdas, nconv = eigs_close_to_shift_pencil(
            N, Mgam, theta,
            nev=nev,
            tol=eigs_tol,
            max_it=2000,
        )

        visited.append((lo, hi, theta, nconv))

        # no spectral information => no exclusion
        if nconv == 0 or lambdas.size == 0:
            continue

        shift = 1j * theta
        lambdas = lambdas[np.argsort(np.abs(lambdas - shift))]

        # probe gain at first few nearby eigenvalue imaginary parts
        hit, info = validate_near_shift_candidates(
            cl, k, gamma, lambdas,
            ncheck=6,
            probe_eps=eps_probe,
        )
        if hit:
            return False, info

        nearest = lambdas[0]
        nearest_dist = float(abs(nearest - shift))

        # if nearest eigenvalue is somewhat close to imag axis, do not exclude aggressively
        if abs(nearest.real) <= axis_warn:
            continue

        if np.isfinite(nearest_dist) and nearest_dist > 0.0:
            intervals += split_by_removed_middle(
                lo, hi, theta,
                r=nearest_dist,
                min_progress=0.5 * min_interval
            )

    return True, {
        "reason": "fully_excluded",
        "window": (-float(b), float(b)),
        "visited": visited,
    }

def make_pygranso_combined_fn_newton_disk(
    cl,
    stab_margin=1e-5,
    num_guess=8,
    wmin=1e-6,
    wmax=1e2,
    base_seed=2,
    disk_b=10.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
    verbose=True,
):
    history = []
    state = {"calls": 0}

    def combined_fn(X_struct):
        state["calls"] += 1

        k_torch = X_struct.k
        k = float(k_torch.item())

        out = cl.objective_constraint_oracle_newton_disk(
            k,
            stab_margin=stab_margin,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=base_seed + state["calls"],
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        f_val = float(out["f"])
        dfdk  = float(out["dfdk"])
        c_val = float(out["c"])
        dcdk  = float(out["dcdk"])

        history.append({
            "k": k,
            "f": f_val,
            "dfdk": dfdk,
            "alpha": float(out["alpha"]),
            "c": c_val,
            "dcdk": dcdk,
            "omega_star": float(out["omega_star"]),
            "certified": bool(out["hinf_info"]["certified"]),
            "cert_info": out["hinf_info"]["cert_info"],
        })

        if verbose and IS_WORLD_ROOT:
            print(
                f"[newton+disk] "
                f"k={k:+.6e}  "
                f"f={f_val:.6e}  "
                f"omega*={out['omega_star']:+.6e}  "
                f"alpha={out['alpha']:.6e}  "
                f"c={c_val:.6e}  "
                f"cert={out['hinf_info']['certified']}"
            )

        dev = k_torch.device
        dt = k_torch.dtype

        f = f_val
        f_grad = torch.tensor([[dfdk]], device=dev, dtype=dt)
        ci = torch.tensor([[c_val]], device=dev, dtype=dt)
        ci_grad = torch.tensor([[dcdk]], device=dev, dtype=dt)

        return [f, f_grad, ci, ci_grad, None, None]

    combined_fn.history = history
    return combined_fn

#device = torch.device("cpu")
#dtype = torch.double

#var_spec = {"k": [1, 1]}

#opts = pygransoStruct()
#opts.torch_device = device
#opts.double_precision = True
#opts.globalAD = False
#opts.maxit = 40
#opts.print_frequency = 1
#opts.print_level = 1

## start from a feasible point you already trust
#opts.x0 = torch.tensor([[1.383047]], device=device, dtype=dtype)

#comb_fn = make_pygranso_combined_fn_newton_disk(
    #cl,
    #stab_margin=1e-5,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #base_seed=2,
    #disk_b=20.0,
    #disk_eigs_tol=1e-8,
    #disk_nev=60,
    #disk_eps_probe=1e-6,
    #disk_axis_warn=1e-3,
    #verbose=True,
#)

#soln = pygranso(
    #var_spec=var_spec,
    #combined_fn=comb_fn,
    #user_opts=opts,
#)

def best_feasible_from_history(history):
    feas = [rec for rec in history if rec["c"] <= 0.0]
    if len(feas) == 0:
        return None
    return min(feas, key=lambda rec: rec["f"])

#hist = comb_fn.history
#best_feas = best_feasible_from_history(hist)

#if IS_WORLD_ROOT:
    #print("\n=== Best feasible iterate (newton+disk) ===")
    #if best_feas is None:
        #print("No feasible iterate found.")
    #else:
        #for k, v in best_feas.items():
            #print(f"{k}: {v}")



# ============================================================
# MIMO actuator/sensor construction
# ============================================================

def build_actuator_vector(x0, sigma, V, phi, bc, is_free):
    """
    Returns reduced actuator vector b_j on free dofs:
        b_j ~ discretization of exp(-(x-x0)^2/(2 sigma^2))
    assembled in the same variational way as your current B_load.
    """
    b_fun = gaussian_function(V, x0, sigma)

    b_full = fem.petsc.assemble_vector(fem.form(b_fun * ufl.conj(phi) * ufl.dx))
    b_full.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                       mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b_full, [bc])

    b_red = restrict_vec_to_is(b_full, is_free)
    return b_red, b_fun


def build_sensor_state_vector(x0, sigma, V, is_free):
    """
    Returns reduced nodal sensor shape s_j on free dofs.
    This is the 's' in the paper. The actual output row uses c_j = M s_j.
    """
    s_fun = gaussian_function(V, x0, sigma)

    s_full = s_fun.x.petsc_vec.copy()
    s_full.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)

    s_red = restrict_vec_to_is(s_full, is_free)
    return s_red, s_fun


def sensor_output_vector(M, s_red):
    """
    Paper-consistent discrete sensing:
        y_j = s_j^H M q
    so store c_j = M s_j, giving y_j = c_j^H q.
    """
    c_red = M.createVecLeft()
    M.mult(s_red, c_red)
    return c_red


def make_rank1_mat_from_vecs_parallel(left: PETSc.Vec, right: PETSc.Vec,
                                      tol: float = 1e-14) -> PETSc.Mat:
    """
    Parallel-safe explicit rank-1 matrix:
        left * right^H

    """
    comm = left.comm
    n = left.getSize()
    m = right.getSize()
    if n != m:
        raise ValueError("left and right must have same global size")

    R = PETSc.Mat().createAIJ([n, n], comm=comm)
    R.setUp()

    right_full = gather_full_vec_numpy(right)   
    lo, hi = left.getOwnershipRange()
    left_loc = left.getArray(readonly=True)

    for i_loc, i_glob in enumerate(range(lo, hi)):
        row = left_loc[i_loc] * np.conjugate(right_full)
        nz = np.flatnonzero(np.abs(row) > tol)
        if nz.size:
            R.setValues([i_glob], nz.astype(PETSc.IntType).tolist(), row[nz])

    R.assemble()
    return R

# ============================================================
# Helpers for a full 2x2 real static gain matrix K
# ============================================================

def kvec_to_K(kvec):
    """
    kvec = [k11, k12, k21, k22]
    """
    k11, k12, k21, k22 = np.asarray(kvec, dtype=float).reshape(4,)
    return np.array([[k11, k12],
                     [k21, k22]], dtype=np.complex128)

def K_to_kvec(K):
    K = np.asarray(K)
    return np.array([np.real(K[0, 0]),
                     np.real(K[0, 1]),
                     np.real(K[1, 0]),
                     np.real(K[1, 1])], dtype=float)
# ============================================================
# MIMO disk certification helpers
# ============================================================

def validate_near_shift_candidates_mimo(cl_mimo, K, gamma, lambdas, ncheck=6, probe_eps=1e-6):
    """
    Probe the gain at the imaginary parts of the first few eigenvalues nearest the shift.
    If any probe exceeds gamma, certification fails.
    """
    for lam in lambdas[:min(len(lambdas), ncheck)]:
        w = float(lam.imag)
        for h in (-probe_eps, 0.0, probe_eps):
            ww = w + h
            g = cl_mimo.gain_of_omega(K, ww)
            if g > gamma:
                return False, {
                    "reason": "gain_violation_from_near_shift_eig",
                    "omega": float(ww),
                    "gain": float(g),
                    "lambda_hit": lam,
                }
    return True, None


def certify_global_peak_disk_mimo_safe(
    cl_mimo,
    K,
    gamma,
    b=10.0,
    eigs_tol=1e-8,
    nev=60,
    eps_probe=1e-6,
    axis_warn=1e-3,
    min_interval=None,
    midpoint_guard=True,
):
    """
    Conservative disk certification for the MIMO static closed-loop system
    on the window [-b, b].

    Returns
    -------
    certified : bool
    info : dict
    """
    Acl = cl_mimo.Acl(K)

    N = build_N_from_M(cl_mimo.M)
    Mgam = build_Mgamma(Acl, cl_mimo.M, gamma)

    N = N.convert("aij"); N.assemble()
    Mgam = Mgam.convert("aij"); Mgam.assemble()

    if min_interval is None:
        min_interval = eigs_tol

    intervals = [(-float(b), 0.0), (0.0, float(b))]
    visited = []

    # low-frequency guard
    for w in [0.0, -1e-12, 1e-12, -1e-9, 1e-9, -1e-6, 1e-6]:
        g = cl_mimo.gain_of_omega(K, w)
        if g > gamma:
            return False, {
                "reason": "low_freq_guard_hit",
                "omega": float(w),
                "gain": float(g),
            }

    while intervals:
        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        intervals.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        lo, hi = intervals.pop(0)
        theta = 0.5 * (lo + hi)

        if midpoint_guard:
            g_theta = cl_mimo.gain_of_omega(K, theta)
            if g_theta > gamma:
                return False, {
                    "reason": "midpoint_gain_violation",
                    "omega": float(theta),
                    "gain": float(g_theta),
                }

        lambdas, nconv = eigs_close_to_shift_pencil(
            N, Mgam, theta,
            nev=nev,
            tol=eigs_tol,
            max_it=2000,
        )

        visited.append((lo, hi, theta, nconv))

        if nconv == 0 or lambdas.size == 0:
            continue

        shift = 1j * theta
        lambdas = lambdas[np.argsort(np.abs(lambdas - shift))]

        ok_local, info = validate_near_shift_candidates_mimo(
            cl_mimo, K, gamma, lambdas,
            ncheck=6,
            probe_eps=eps_probe,
        )
        if not ok_local:
            return False, info

        nearest = lambdas[0]
        nearest_dist = float(abs(nearest - shift))

        # conservative near-axis behavior
        if abs(nearest.real) <= axis_warn:
            continue

        if np.isfinite(nearest_dist) and nearest_dist > 0.0:
            intervals += split_by_removed_middle(
                lo, hi, theta,
                r=nearest_dist,
                min_progress=0.5 * min_interval,
            )

    return True, {
        "reason": "fully_excluded",
        "window": (-float(b), float(b)),
        "visited": visited,
    }
class ClosedLoopSOFMIMO:
    """
    Static output feedback:
        u = feedback_sign * K y
        y_j = c_j^H q, with c_j = M s_j

    so
        Acl(K) = A + feedback_sign * sum_{i,j} K[i,j] b_i c_j^H
    """

    def __init__(self, A: PETSc.Mat, M: PETSc.Mat,
                 b_list, c_list, feedback_sign=-1.0):
        self.A = A
        self.M = M
        self.b_list = [b.copy() for b in b_list]
        self.c_list = [c.copy() for c in c_list]
        self.feedback_sign = float(feedback_sign)

        self.ma = len(self.b_list)
        self.ms = len(self.c_list)

        # Preassemble all b_i c_j^H blocks
        self.BcH = []
        for i in range(self.ma):
            row = []
            for j in range(self.ms):
                row.append(make_rank1_mat_from_vecs_parallel(
                    self.b_list[i], self.c_list[j]
                ))
            self.BcH.append(row)

    def Acl(self, K):
        K = np.asarray(K, dtype=np.complex128)
        if K.shape != (self.ma, self.ms):
            raise ValueError(f"K must have shape {(self.ma, self.ms)}, got {K.shape}")

        Acl = self.A.copy()
        for i in range(self.ma):
            for j in range(self.ms):
                kij = self.feedback_sign * K[i, j]
                if abs(kij) > 0.0:
                    Acl.axpy(PETSc.ScalarType(kij), self.BcH[i][j])
        Acl.assemble()
        return Acl

    def gain_of_omega(self, K, omega: float) -> float:
        Acl = self.Acl(K)
        return gain_of_omega(Acl, self.M, float(omega))

    def spectral_abscissa(self, K, nev=5, tol=1e-8, max_it=400):
        Acl = self.Acl(K)
        vals = rightmost_eig_realpart(Acl, self.M, nev=nev, tol=tol, max_it=max_it)
        return float(np.real(vals[0])), vals

    def spectral_abscissa_and_grad(self, K, eps_tol=1e-8, max_it=400, ncv=40):
        """
        Returns
        -------
        alpha : float
        G     : real gradient matrix d alpha / d K_ij
        """
        Acl = self.Acl(K)

        eps = SLEPc.EPS().create(comm=self.M.comm)
        eps.setOperators(Acl, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setTwoSided(True)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError("GNHEP did not converge for spectral abscissa")

        lam = eps.getEigenvalue(0)

        v = Acl.createVecRight()
        u = Acl.createVecLeft()
        eps.getEigenvector(0, v)
        eps.getLeftEigenvector(0, u)

        Mv = v.duplicate()
        self.M.mult(v, Mv)
        denom = u.dot(Mv)
        if abs(denom) < 1e-30:
            raise RuntimeError("u^H M v too small in spectral_abscissa_and_grad")

        G = np.zeros((self.ma, self.ms), dtype=float)
        for i in range(self.ma):
            ubi = u.dot(self.b_list[i])   # u^H b_i
            for j in range(self.ms):
                cjv = self.c_list[j].dot(v)  # c_j^H v
                dlam = self.feedback_sign * (ubi * cjv) / denom
                G[i, j] = float(np.real(dlam))

        return float(np.real(lam)), G

    def hinf_branch_value_grad(self, K, omega, eps_tol=1e-8, max_it=400, ncv=40):
        """
        Fixed-frequency branch gradient wrt each K_ij.
        """
        Acl = self.Acl(K)
        Aom, ctx, S = make_Aomega_shell_ctx(Acl, self.M, float(omega))

        eps = SLEPc.EPS().create(comm=self.M.comm)
        eps.setOperators(Aom, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError(f"GHEP did not converge at omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = max(float(np.real(lam)), 0.0)
        sigma = float(np.sqrt(lam))

        v = S.createVecRight()
        eps.getEigenvector(0, v)

        Mv = v.duplicate()
        self.M.mult(v, Mv)
        den = v.dot(Mv)
        if abs(den) < 1e-30:
            raise RuntimeError("v^H M v too small in hinf_branch_value_grad")
        v.scale(1.0 / np.sqrt(np.real(den)))

        # x = S^{-1} v
        x = S.createVecRight()
        ctx.ksp.solve(v, x)

        # y = (1/sigma) S^{-H} M x
        Mx = x.duplicate()
        self.M.mult(x, Mx)

        y = S.createVecRight()
        ctx.solve_hermitian(Mx, y)

        G = np.zeros((self.ma, self.ms), dtype=float)

        if sigma > 1e-30:
            y.scale(1.0 / sigma)
            for i in range(self.ma):
                biy = self.b_list[i].dot(y)      # b_i^H y
                for j in range(self.ms):
                    cjx = self.c_list[j].dot(x)  # c_j^H x
                    G[i, j] = self.feedback_sign * float(np.real(biy * cjx))

        return sigma, G

    def hinf_value_grad(self, K, num_guess=8, wmin=1e-6, wmax=2e1, seed=1):
        """
        Use existing omega-search on the closed-loop Acl(K),
        then evaluate the active-branch gradient wrt K.
        """
        Acl = self.Acl(K)

        omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
        omega_star, gamma_star, local_results = optimize_peak_multistart(Acl, self.M, omegas0)

        gamma_branch, G = self.hinf_branch_value_grad(K, omega_star)

        return {
            "omega_star": float(omega_star),
            "gamma_star": float(gamma_branch),
            "Ggamma": G,
            "local_results": local_results,
        }

    def objective_constraint_oracle_full(
        self,
        kvec,
        stab_margin=1e-8,
        num_guess=8,
        wmin=1e-6,
        wmax=2e1,
        seed=1,
    ):
        """
        Full 2x2 static gain optimization over the 4 real entries:
            kvec = [k11, k12, k21, k22]
        """
        K = kvec_to_K(kvec)

        h = self.hinf_value_grad(
            K,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
        )

        alpha, Galpha = self.spectral_abscissa_and_grad(K)

        df = np.array([
            h["Ggamma"][0, 0],
            h["Ggamma"][0, 1],
            h["Ggamma"][1, 0],
            h["Ggamma"][1, 1],
        ], dtype=float)

        dc = np.array([
            Galpha[0, 0],
            Galpha[0, 1],
            Galpha[1, 0],
            Galpha[1, 1],
        ], dtype=float)

        return {
            "K": K,
            "f": h["gamma_star"],
            "dfdk": df,
            "omega_star": h["omega_star"],
            "alpha": alpha,
            "c": alpha + float(stab_margin),
            "dcdk": dc,
            "hinf_info": h,
        } 
    
    def hinf_value_grad_newton_disk(
        self,
        K,
        num_guess=8,
        wmin=1e-6,
        wmax=2e1,
        seed=1,
        branch_eps_tol=1e-8,
        branch_max_it=400,
        branch_ncv=40,
        disk_b=10.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
        disk_midpoint_guard=True,
    ):
        """
        MIMO Hinf oracle:
        1) multistart Newton gives a candidate peak
        2) disk certification checks whether a larger peak is present
        3) if certification fails because a violating frequency is found,
            use that violating frequency instead

        Returns
        -------
        dict with gamma_star, omega_star, gradient matrix, certified, cert_info
        """
        Acl = self.Acl(K)

        # step 1: candidate from multistart Newton
        omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
        w_best, g_best, local_results = optimize_peak_multistart(Acl, self.M, omegas0)

        gamma_star = float(g_best)
        omega_star = float(w_best)

        # step 2: safe disk certification
        certified, cert_info = certify_global_peak_disk_mimo_safe(
            self,
            K,
            gamma=gamma_star,
            b=disk_b,
            eigs_tol=disk_eigs_tol,
            nev=disk_nev,
            eps_probe=disk_eps_probe,
            axis_warn=disk_axis_warn,
            midpoint_guard=disk_midpoint_guard,
        )

        # step 3: if disk found a larger point, override candidate
        if not certified:
            if "omega" in cert_info and "gain" in cert_info:
                omega_star = float(cert_info["omega"])
                gamma_star = float(cert_info["gain"])

        # gradient at selected active frequency
        _, G = self.hinf_branch_value_grad(
            K,
            omega_star,
            eps_tol=branch_eps_tol,
            max_it=branch_max_it,
            ncv=branch_ncv,
        )

        return {
            "omega_star": float(omega_star),
            "gamma_star": float(gamma_star),
            "Ggamma": G,
            "certified": bool(certified),
            "cert_info": cert_info,
            "local_results": local_results,
        }


    def objective_constraint_oracle_full_newton_disk(
        self,
        kvec,
        stab_margin=1e-8,
        num_guess=8,
        wmin=1e-6,
        wmax=2e1,
        seed=1,
        disk_b=10.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
    ):
        """
        Full 2x2 static gain optimization over the 4 real entries,
        but with Newton + disk certification for the Hinf peak.
        """
        K = kvec_to_K(kvec)

        h = self.hinf_value_grad_newton_disk(
            K,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        alpha, Galpha = self.spectral_abscissa_and_grad(K)

        df = np.array([
            h["Ggamma"][0, 0],
            h["Ggamma"][0, 1],
            h["Ggamma"][1, 0],
            h["Ggamma"][1, 1],
        ], dtype=float)

        dc = np.array([
            Galpha[0, 0],
            Galpha[0, 1],
            Galpha[1, 0],
            Galpha[1, 1],
        ], dtype=float)

        return {
            "K": K,
            "f": h["gamma_star"],
            "dfdk": df,
            "omega_star": h["omega_star"],
            "alpha": alpha,
            "c": alpha + float(stab_margin),
            "dcdk": dc,
            "hinf_info": h,
        }
# ============================================================
# Your own 2-actuator / 2-sensor placement
# ============================================================

#xa_list = [-4.0,  3.0]   # choose what you want
#xs_list = [-2.5,  4.0]   # choose what you want

#b_list = []
#c_list = []

#for xa in xa_list:
    #b_j, _ = build_actuator_vector(xa, sigma_gauss, V, phi, bc, is_free)
    #b_list.append(b_j)

#for xs in xs_list:
    #s_j, _ = build_sensor_state_vector(xs, sigma_gauss, V, is_free)
    #c_j = sensor_output_vector(M, s_j)   # IMPORTANT: c_j = M s_j
    #c_list.append(c_j)

#cl_mimo = ClosedLoopSOFMIMO(A, M, b_list, c_list, feedback_sign=-1.0)

#K0 = np.zeros((2, 2), dtype=np.complex128)
#g_test = cl_mimo.gain_of_omega(K0, omega=1.0)
#alpha_test, _ = cl_mimo.spectral_abscissa_and_grad(K0)

#if IS_WORLD_ROOT:
    #print("open-loop gain at omega=1:", g_test)
    #print("open-loop spectral abscissa:", alpha_test)

#def make_pygranso_combined_fn_mimo_full_newton_disk(
    #cl_mimo,
    #stab_margin=1e-8,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #base_seed=2,
    #disk_b=20.0,
    #disk_eigs_tol=1e-8,
    #disk_nev=60,
    #disk_eps_probe=1e-6,
    #disk_axis_warn=1e-3,
    #verbose=True,
#):
    #history = []
    #state = {"calls": 0}

    #def combined_fn(X_struct):
        #state["calls"] += 1

        #k11 = float(X_struct.k11.item())
        #k12 = float(X_struct.k12.item())
        #k21 = float(X_struct.k21.item())
        #k22 = float(X_struct.k22.item())

        #kvec = np.array([k11, k12, k21, k22], dtype=float)

        #out = cl_mimo.objective_constraint_oracle_full_newton_disk(
            #kvec,
            #stab_margin=stab_margin,
            #num_guess=num_guess,
            #wmin=wmin,
            #wmax=wmax,
            #seed=base_seed + state["calls"],
            #disk_b=disk_b,
            #disk_eigs_tol=disk_eigs_tol,
            #disk_nev=disk_nev,
            #disk_eps_probe=disk_eps_probe,
            #disk_axis_warn=disk_axis_warn,
        #)

        #f_val = float(out["f"])
        #df = np.asarray(out["dfdk"], dtype=float)
        #c_val = float(out["c"])
        #dc = np.asarray(out["dcdk"], dtype=float)

        #history.append({
            #"kvec": kvec.copy(),
            #"K": out["K"].copy(),
            #"f": f_val,
            #"dfdk": df.copy(),
            #"alpha": float(out["alpha"]),
            #"c": c_val,
            #"dcdk": dc.copy(),
            #"omega_star": float(out["omega_star"]),
            #"certified": bool(out["hinf_info"]["certified"]),
            #"cert_info": out["hinf_info"]["cert_info"],
        #})

        #if verbose and IS_WORLD_ROOT:
            #print(
                #f"[mimo full + disk] "
                #f"k11={k11:+.4e}  k12={k12:+.4e}  "
                #f"k21={k21:+.4e}  k22={k22:+.4e}  "
                #f"f={f_val:.6e}  omega*={out['omega_star']:+.6e}  "
                #f"alpha={out['alpha']:.6e}  c={c_val:.6e}  "
                #f"cert={out['hinf_info']['certified']}"
            #)

        #dev = X_struct.k11.device
        #dt = X_struct.k11.dtype

        #f = f_val
        #f_grad = torch.tensor([[df[0]], [df[1]], [df[2]], [df[3]]], device=dev, dtype=dt)

        #ci = torch.tensor([[c_val]], device=dev, dtype=dt)
        #ci_grad = torch.tensor([[dc[0]], [dc[1]], [dc[2]], [dc[3]]], device=dev, dtype=dt)

        #return [f, f_grad, ci, ci_grad, None, None]

    #combined_fn.history = history
    #return combined_fn

#device = torch.device("cpu")
#dtype = torch.double

#var_spec = {
    #"k11": [1, 1],
    #"k12": [1, 1],
    #"k21": [1, 1],
    #"k22": [1, 1],
#}

#opts = pygransoStruct()
#opts.torch_device = device
#opts.double_precision = True
#opts.globalAD = False
#opts.maxit = 40
#opts.print_frequency = 1
#opts.print_level = 1

# Start from zero if open loop is already stable.
# Otherwise use a stabilizing initial guess.
#opts.x0 = torch.tensor([[0.0],
                        #[0.0],
                        #[0.0],
                        #[0.0]], device=device, dtype=dtype)

#comb_fn = make_pygranso_combined_fn_mimo_full_newton_disk(
    #cl_mimo,
    #stab_margin=1e-8,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #base_seed=2,
    #disk_b=20.0,
    #disk_eigs_tol=1e-8,
    #disk_nev=60,
    #disk_eps_probe=1e-6,
    #disk_axis_warn=1e-3,
    #verbose=True,
#)
#soln = pygranso(
    #var_spec=var_spec,
    #combined_fn=comb_fn,
    #user_opts=opts,
#)

#if IS_WORLD_ROOT:
    #print("\n=== PyGRANSO MIMO-full result ===")
    #print("final x =", soln.final.x)

    #import matplotlib.pyplot as plt

def plot_hinf_comparison_mimo(cl_mimo, K_list, labels=None, wmin=1e-4, wmax=2e1, npts=500):
    comm = WORLD

    ws_pos = np.logspace(np.log10(wmin), np.log10(wmax), npts)
    ws = np.concatenate([-ws_pos[::-1], [0.0], ws_pos])

    curves = []
    peaks = []

    for K in K_list:
        vals = np.array([cl_mimo.gain_of_omega(K, float(w)) for w in ws], dtype=float)
        idx = int(np.argmax(vals))
        curves.append(vals)
        peaks.append((ws[idx], vals[idx]))

    if IS_WORLD_ROOT:
        if labels is None:
            labels = [f"case {i}" for i in range(len(K_list))]

        plt.figure(figsize=(8, 5))
        for vals, lab, pk in zip(curves, labels, peaks):
            plt.semilogy(ws, vals, label=f"{lab}  (Hinf≈{pk[1]:.3e})")
            plt.scatter([pk[0]], [pk[1]], s=35, zorder=5)

        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$\|G_{cl}(i\omega)\|$")
        plt.title("2x2 static MIMO sampled Hinf curve")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\n=== Peak summary ===")
        for lab, pk in zip(labels, peaks):
            print(f"{lab:18s}  omega*={pk[0]:+.6e}  Hinf≈{pk[1]:.6e}")

    return ws, curves, peaks

#xopt = np.array(soln.final.x.detach().cpu().numpy(), dtype=float).reshape(-1)
#K_opt = kvec_to_K(xopt)

#K_open = np.zeros((2, 2), dtype=np.complex128)

#ws, curves, peaks = plot_hinf_comparison_mimo(
    #cl_mimo,
    #[K_open, K_opt],
    #labels=["open loop", "optimized full static MIMO"],
    #wmin=1e-4,
    #wmax=2e1,
    #npts=500,
#)

#if IS_WORLD_ROOT:
    #print("\nOptimized K =")
    #print(K_opt)


def assemble_gaussian_load_vector(Vspace, phi, bc, is_free, x0, sigma):
    """
    FE load vector for a Gaussian centered at x0:
        g(x) = exp(-(x-x0)^2 / (2 sigma^2))

    Returned vector is restricted to the free dofs.
    """
    g_fun = fem.Function(Vspace)
    g_fun.interpolate(lambda x: np.exp(-0.5 * ((x[0] - x0) / sigma) ** 2))

    vec = fem.petsc.assemble_vector(fem.form(g_fun * ufl.conj(phi) * ufl.dx))
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(vec, [bc])

    return restrict_vec_to_is(vec, is_free)

def validate_near_shift_candidates_mimo(cl, K, gamma, lambdas, ncheck=6, probe_eps=1e-6):
    for lam in lambdas[:min(len(lambdas), ncheck)]:
        w = float(lam.imag)
        for h in (-probe_eps, 0.0, probe_eps):
            ww = w + h
            g = cl.gain_of_omega(K, ww)
            if g > gamma:
                return True, {
                    "reason": "gain_violation_from_near_shift_eig",
                    "omega": float(ww),
                    "gain": float(g),
                    "lambda_hit": lam,
                }
    return False, None


def certify_global_peak_disk_cl_safe_mimo(
    cl,
    K,
    gamma,
    b=20.0,
    eigs_tol=1e-8,
    nev=60,
    eps_probe=1e-6,
    axis_warn=1e-3,
    min_interval=None,
    midpoint_guard=True,
):
    K = np.asarray(K, dtype=float).reshape(2, 2)
    Acl = cl.Acl(K)

    N = build_N_from_M(cl.M)
    Mgam = build_Mgamma(Acl, cl.M, gamma)

    N = N.convert("aij"); N.assemble()
    Mgam = Mgam.convert("aij"); Mgam.assemble()

    if min_interval is None:
        min_interval = eigs_tol

    intervals = [(-float(b), 0.0), (0.0, float(b))]
    visited = []

    # low-frequency guard
    for w in [0.0, -1e-12, 1e-12, -1e-9, 1e-9, -1e-6, 1e-6]:
        g = cl.gain_of_omega(K, w)
        if g > gamma:
            return False, {
                "reason": "low_freq_guard_hit",
                "omega": float(w),
                "gain": float(g),
            }

    while intervals:
        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        intervals.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        lo, hi = intervals.pop(0)
        theta = 0.5 * (lo + hi)

        if midpoint_guard:
            g_theta = cl.gain_of_omega(K, theta)
            if g_theta > gamma:
                return False, {
                    "reason": "midpoint_gain_violation",
                    "omega": float(theta),
                    "gain": float(g_theta),
                }

        lambdas, nconv = eigs_close_to_shift_pencil(
            N, Mgam, theta,
            nev=nev,
            tol=eigs_tol,
            max_it=2000,
        )

        visited.append((lo, hi, theta, nconv))

        if nconv == 0 or lambdas.size == 0:
            continue

        shift = 1j * theta
        lambdas = lambdas[np.argsort(np.abs(lambdas - shift))]

        hit, info = validate_near_shift_candidates_mimo(
            cl, K, gamma, lambdas,
            ncheck=6,
            probe_eps=eps_probe,
        )
        if hit:
            return False, info

        nearest = lambdas[0]
        nearest_dist = float(abs(nearest - shift))

        if abs(nearest.real) <= axis_warn:
            continue

        if np.isfinite(nearest_dist) and nearest_dist > 0.0:
            intervals += split_by_removed_middle(
                lo, hi, theta,
                r=nearest_dist,
                min_progress=0.5 * min_interval
            )

    return True, {
        "reason": "fully_excluded",
        "window": (-float(b), float(b)),
        "visited": visited,
    }
def build_chen_rowley_2x2_vectors(Vspace, phi, bc, is_free, sigma=0.4):
    """
    Chen & Rowley 2011, disturbances everywhere, conjugate-gradient-refined 2x2 placement:
        actuators: (-3.78,  2.71)
        sensors:   (-2.75,  3.74)

    Returns
    -------
    b_list : list of 2 PETSc.Vec
        actuator vectors
    c_list : list of 2 PETSc.Vec
        sensor vectors (used as rows via c_j^H q)
    """
    xa = (-3.78, 2.71)
    xs = (-2.75, 3.74)

    b_list = [assemble_gaussian_load_vector(Vspace, phi, bc, is_free, x0, sigma) for x0 in xa]
    c_list = [assemble_gaussian_load_vector(Vspace, phi, bc, is_free, x0, sigma) for x0 in xs]

    return b_list, c_list

class ClosedLoopMIMO2x2:
    """
    2x2 real static output feedback:
        u = feedback_sign * K y
        y = C q,   with C = [c1^H; c2^H]
        B = [b1 b2]

    so
        Acl(K) = A + feedback_sign * B K C
               = A + feedback_sign * sum_{i,j} K_ij b_i c_j^H

    Assumes:
    - K is a real 2x2 numpy array
    - A, M are PETSc matrices on the reduced/free space
    - b_list, c_list each have length 2 and contain PETSc.Vec on that same space
    """

    def __init__(self, A, M, b_list, c_list, feedback_sign=-1.0):
        assert len(b_list) == 2 and len(c_list) == 2

        self.A = A
        self.M = M
        self.b_list = [b.copy() for b in b_list]
        self.c_list = [c.copy() for c in c_list]
        self.feedback_sign = float(feedback_sign)

        # Precompute the four rank-1 matrices b_i c_j^H
        self.rank1 = [[make_rank1_mat_from_vecs(self.b_list[i], self.c_list[j]) for j in range(2)]
                      for i in range(2)]

    def Acl(self, K):
        K = np.asarray(K, dtype=float).reshape(2, 2)

        Acl = self.A.copy()
        for i in range(2):
            for j in range(2):
                if abs(K[i, j]) > 0.0:
                    Acl.axpy(self.feedback_sign * float(K[i, j]), self.rank1[i][j])
        Acl.assemble()
        return Acl

    def gain_of_omega(self, K, omega):
        Acl = self.Acl(K)
        return gain_of_omega(Acl, self.M, float(omega))

    # ------------------------------------------------------------
    # Small linear-algebra helpers for B^H z and C x
    # ------------------------------------------------------------

    def zHB(self, z):
        """
        returns [z^H b1, z^H b2] as a complex numpy array of length 2
        """
        return np.array([z.dot(self.b_list[i]) for i in range(2)], dtype=np.complex128)

    def Cv(self, v):
        """
        returns [c1^H v, c2^H v] as a complex numpy array of length 2
        """
        return np.array([self.c_list[j].dot(v) for j in range(2)], dtype=np.complex128)

    # ------------------------------------------------------------
    # Spectral abscissa and branch gradient wrt real 2x2 K
    # ------------------------------------------------------------

    def spectral_abscissa_and_grad(self, K, eps_tol=1e-8, max_it=400, ncv=40):
        """
        Active-branch gradient for alpha(K) = max Re(lambda(Acl(K), M))

        For a simple active generalized eigenpair
            Acl v = lambda M v,
            u^H Acl = lambda u^H M,
        the directional derivative is
            d lambda[K;H] = feedback_sign * u^H B H C v / (u^H M v)

        For real K, the gradient matrix is
            G_alpha = feedback_sign * Re( a b^T / (u^H M v) )
        with
            a_i = u^H b_i,
            b_j = c_j^H v.
        """
        K = np.asarray(K, dtype=float).reshape(2, 2)
        Acl = self.Acl(K)

        eps = SLEPc.EPS().create(comm=self.M.comm)
        eps.setOperators(Acl, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setTwoSided(True)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError("GNHEP did not converge for spectral abscissa")

        lam = eps.getEigenvalue(0)

        v = Acl.createVecRight()
        u = Acl.createVecLeft()
        eps.getEigenvector(0, v)
        eps.getLeftEigenvector(0, u)

        Mv = v.duplicate()
        self.M.mult(v, Mv)
        denom = u.dot(Mv)   # u^H M v

        if abs(denom) < 1e-30:
            raise RuntimeError("u^H M v too small in spectral_abscissa_and_grad")

        a = self.zHB(u)   # a_i = u^H b_i
        b = self.Cv(v)    # b_j = c_j^H v

        # real 2x2 gradient for real-valued K
        G = self.feedback_sign * np.real(np.outer(a, b) / denom)

        return float(np.real(lam)), G

    # ------------------------------------------------------------
    # Fixed-frequency Hinf branch value and gradient wrt real 2x2 K
    # ------------------------------------------------------------

    def hinf_branch_value_grad(self, K, omega, eps_tol=1e-8, max_it=400, ncv=40):
        """
        Fixed-frequency branch:
            g_omega(K) = sigma_max( R_cl(i omega, K) )

        We use the generalized Hermitian EVP
            A_omega v = lambda M v,
            A_omega = S^{-H} M S^{-1},
            S = i omega M - Acl(K),
        so sigma = sqrt(lambda).

        For real K, with z = S^{-H} M x and x = S^{-1} v,
            d sigma[K;H] = feedback_sign * Re( z^H B H C x ) / sigma

        Therefore the real 2x2 gradient matrix is
            G_hinf = feedback_sign * Re( a b^T ) / sigma
        with
            a_i = z^H b_i,
            b_j = c_j^H x.
        """
        K = np.asarray(K, dtype=float).reshape(2, 2)
        Acl = self.Acl(K)

        Aom, ctx, S = make_Aomega_shell_ctx(Acl, self.M, float(omega))

        eps = SLEPc.EPS().create(comm=self.M.comm)
        eps.setOperators(Aom, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError(f"GHEP did not converge at omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = max(float(np.real(lam)), 0.0)
        sigma = float(np.sqrt(lam))

        v = S.createVecRight()
        eps.getEigenvector(0, v)

        Mv = v.duplicate()
        self.M.mult(v, Mv)
        den = v.dot(Mv)
        if abs(den) < 1e-30:
            raise RuntimeError("v^H M v too small in hinf_branch_value_grad")

        # normalize v^H M v = 1
        v.scale(1.0 / np.sqrt(np.real(den)))

        # x = S^{-1} v
        x = S.createVecRight()
        ctx.ksp.solve(v, x)

        # z = S^{-H} M x
        Mx = x.duplicate()
        self.M.mult(x, Mx)

        z = S.createVecRight()
        ctx.solve_hermitian(Mx, z)

        if sigma <= 1e-30:
            G = np.zeros((2, 2), dtype=float)
            return sigma, G

        a = self.zHB(z)   # a_i = z^H b_i
        b = self.Cv(x)    # b_j = c_j^H x

        G = self.feedback_sign * np.real(np.outer(a, b)) / sigma
        return sigma, G

    # ------------------------------------------------------------
    # Newton + safe disk Hinf oracle
    # ------------------------------------------------------------

    def hinf_value_grad_newton_disk(
        self,
        K,
        num_guess=8,
        wmin=1e-6,
        wmax=2e1,
        seed=1,
        branch_eps_tol=1e-8,
        branch_max_it=400,
        branch_ncv=40,
        disk_b=20.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
        disk_midpoint_guard=True,
    ):
        K = np.asarray(K, dtype=float).reshape(2, 2)
        Acl = self.Acl(K)

        omegas0 = random_initial_omegas(num_guess, wmin=wmin, wmax=wmax, seed=seed)
        w_best, g_best, local_results = optimize_peak_multistart(Acl, self.M, omegas0)

        gamma_star = float(g_best)
        omega_star = float(w_best)

        certified, cert_info = certify_global_peak_disk_cl_safe_mimo(
            self,
            K,
            gamma=gamma_star,
            b=disk_b,
            eigs_tol=disk_eigs_tol,
            nev=disk_nev,
            eps_probe=disk_eps_probe,
            axis_warn=disk_axis_warn,
            midpoint_guard=disk_midpoint_guard,
        )

        if not certified and "omega" in cert_info and "gain" in cert_info:
            omega_star = float(cert_info["omega"])
            gamma_star = float(cert_info["gain"])

        _, G = self.hinf_branch_value_grad(
            K,
            omega_star,
            eps_tol=branch_eps_tol,
            max_it=branch_max_it,
            ncv=branch_ncv,
        )

        return {
            "omega_star": omega_star,
            "gamma_star": gamma_star,
            "dGamma_dK": G,
            "certified": bool(certified),
            "cert_info": cert_info,
            "local_results": local_results,
        }

    def objective_constraint_oracle_newton_disk(
        self,
        K,
        stab_margin=1e-5,
        num_guess=8,
        wmin=1e-6,
        wmax=2e1,
        seed=1,
        disk_b=20.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
    ):
        K = np.asarray(K, dtype=float).reshape(2, 2)

        h = self.hinf_value_grad_newton_disk(
            K,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        alpha, G_alpha = self.spectral_abscissa_and_grad(K)

        return {
            "f": h["gamma_star"],
            "dfdK": h["dGamma_dK"],
            "omega_star": h["omega_star"],
            "alpha": alpha,
            "c": alpha + float(stab_margin),
            "dcdK": G_alpha,
            "hinf_info": h,
        }

# Chen & Rowley 2x2 placements, disturbances everywhere
b_list_2x2, c_list_2x2 = build_chen_rowley_2x2_vectors(
    V, phi, bc, is_free, sigma=0.4
)

cl_mimo = ClosedLoopMIMO2x2(
    A, M,
    b_list=b_list_2x2,
    c_list=c_list_2x2,
    feedback_sign=-1.0,   # u = -K y
)

#K0 = np.zeros((2, 2), dtype=float)

#out0 = cl_mimo.objective_constraint_oracle_newton_disk(
    #K0,
    #stab_margin=1e-5,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #seed=2,
    #disk_b=20.0,
    #disk_eigs_tol=1e-8,
    #disk_nev=60,
    #disk_eps_probe=1e-6,
    #disk_axis_warn=1e-3,
#)

#if IS_WORLD_ROOT:
    #print("=== 2x2 MIMO, K=0 ===")
    #print("f(K0)      =", out0["f"])
    #print("omega_star =", out0["omega_star"])
    #print("alpha(K0)  =", out0["alpha"])
    #print("c(K0)      =", out0["c"])
    #print("dfdK       =\n", out0["dfdK"])
    #print("dcdK       =\n", out0["dcdK"])
    #print("certified  =", out0["hinf_info"]["certified"])
    #print("cert_info  =", out0["hinf_info"]["cert_info"]["reason"])


#def fd_check_mimo_oracle(cl_mimo, K, H, h=1e-6, seed=2):
    #K = np.asarray(K, dtype=float).reshape(2,2)
    #H = np.asarray(H, dtype=float).reshape(2,2)

    #o0 = cl_mimo.objective_constraint_oracle_newton_disk(K, seed=seed)
    #op = cl_mimo.objective_constraint_oracle_newton_disk(K + h*H, seed=seed)
    #om = cl_mimo.objective_constraint_oracle_newton_disk(K - h*H, seed=seed)

    #df_fd = (op["f"] - om["f"]) / (2*h)
    #dc_fd = (op["c"] - om["c"]) / (2*h)

    #df_an = float(np.sum(o0["dfdK"] * H))
    #dc_an = float(np.sum(o0["dcdK"] * H))

    #if IS_WORLD_ROOT:
        #print("\n=== FD check MIMO oracle ===")
        #print("analytic <dfdK,H> =", df_an)
        #print("FD       <dfdK,H> =", df_fd)
        #print("analytic <dcdK,H> =", dc_an)
        #print("FD       <dcdK,H> =", dc_fd)

#H = np.array([[1.0, 0.0],
              #[0.0, 0.0]])

#fd_check_mimo_oracle(cl_mimo, K0, H, h=1e-6, seed=3)


def make_pygranso_combined_fn_mimo2x2(
    cl_mimo,
    stab_margin=1e-3,
    num_guess=8,
    wmin=1e-6,
    wmax=2e1,
    base_seed=2,
    disk_b=20.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
    verbose=True,
):
    history = []
    state = {"calls": 0}

    def combined_fn(X_struct):
        state["calls"] += 1

        K_torch = X_struct.K
        K = K_torch.detach().cpu().numpy().astype(float)

        out = cl_mimo.objective_constraint_oracle_newton_disk(
            K,
            stab_margin=stab_margin,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=base_seed + state["calls"],
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        f_val = float(out["f"])
        Gf_mat = np.asarray(out["dfdK"], dtype=float)   # 2x2
        c_val = float(out["c"])
        Gc_mat = np.asarray(out["dcdK"], dtype=float)   # 2x2

        history.append({
            "K": K.copy(),
            "f": f_val,
            "dfdK": Gf_mat.copy(),
            "alpha": float(out["alpha"]),
            "c": c_val,
            "dcdK": Gc_mat.copy(),
            "omega_star": float(out["omega_star"]),
            "certified": bool(out["hinf_info"]["certified"]),
            "cert_info": out["hinf_info"]["cert_info"],
        })

        if verbose and IS_WORLD_ROOT:
            print(
                f"[mimo 2x2] call={state['calls']:03d}  "
                f"f={f_val:.6e}  "
                f"omega*={out['omega_star']:+.6e}  "
                f"alpha={out['alpha']:.6e}  "
                f"c={c_val:.6e}  "
                f"cert={out['hinf_info']['certified']}"
            )
            print("K =\n", K)

        dev = K_torch.device
        dt = K_torch.dtype

        # IMPORTANT: PyGRANSO wants column vectors of length n=4
        Gf_vec = torch.tensor(Gf_mat.reshape(-1, 1), device=dev, dtype=dt)
        Gc_vec = torch.tensor(Gc_mat.reshape(-1, 1), device=dev, dtype=dt)

        f = f_val
        ci = torch.tensor([[c_val]], device=dev, dtype=dt)

        return [f, Gf_vec, ci, Gc_vec, None, None]

    combined_fn.history = history
    return combined_fn

def best_feasible_from_history_mimo(history):
    feas = [rec for rec in history if rec["c"] <= 0.0]
    if len(feas) == 0:
        return None
    return min(feas, key=lambda rec: rec["f"])

#device = torch.device("cpu")
#dtype = torch.double

#var_spec = {"K": [2, 2]}

#opts = pygransoStruct()
#opts.torch_device = device
#opts.double_precision = True
#opts.globalAD = False
#opts.maxit = 30
#opts.print_frequency = 1
#opts.print_level = 1

## start from the open-loop controller
#opts.x0 = torch.zeros((4, 1), device=device, dtype=dtype)
#comb_fn = make_pygranso_combined_fn_mimo2x2(
    #cl_mimo,
    #stab_margin=1e-3,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #base_seed=2,
    #disk_b=20.0,
    #disk_eigs_tol=1e-8,
    #disk_nev=60,
    #disk_eps_probe=1e-6,
    #disk_axis_warn=1e-3,
    #verbose=True,
#)

#soln = pygranso(
    #var_spec=var_spec,
    #combined_fn=comb_fn,
    #user_opts=opts,
#)

#hist = comb_fn.history
#best_feas = best_feasible_from_history_mimo(hist)

#if IS_WORLD_ROOT:
    #print("\n=== Best feasible 2x2 iterate ===")
    #if best_feas is None:
        #print("No feasible iterate found.")
    #else:
        #print("f        =", best_feas["f"])
        #print("alpha    =", best_feas["alpha"])
        #print("c        =", best_feas["c"])
        #print("omega*   =", best_feas["omega_star"])
        #print("K* =\n", best_feas["K"])
        #print("certified =", best_feas["certified"])
        #print("cert_info =", best_feas["cert_info"])

#if IS_WORLD_ROOT:
    #print("\n=== 2x2 history ===")
    #for i, rec in enumerate(hist):
        #print(
            #f"it={i:02d}  "
            #f"f={rec['f']:.6e}  "
            #f"alpha={rec['alpha']:.6e}  "
            #f"c={rec['c']:.6e}  "
            #f"omega*={rec['omega_star']:+.6e}  "
            #f"cert={rec['certified']}"
        #)
        #print(rec["K"])

#def make_signed_log_grid(wmin=1e-4, wmax=2e1, npos=1200):
    #ws_pos = np.logspace(np.log10(wmin), np.log10(wmax), npos)
    #return np.concatenate([-ws_pos[::-1], [0.0], ws_pos])

#def sample_gain_curve_mimo(cl_mimo, K, ws):
    #vals = []
    #for w in ws:
        #vals.append(cl_mimo.gain_of_omega(K, float(w)))
    #return np.array(vals, dtype=float)

#def plot_best_mimo_peak(cl_mimo, K_star, wmin=1e-4, wmax=2e1, npos=1200):
    #ws = make_signed_log_grid(wmin=wmin, wmax=wmax, npos=npos)

    #K0 = np.zeros((2,2), dtype=float)

    #vals_open = sample_gain_curve_mimo(cl_mimo, K0, ws)
    #vals_star = sample_gain_curve_mimo(cl_mimo, K_star, ws)

    #i0 = int(np.argmax(vals_open))
    #is_ = int(np.argmax(vals_star))

    #w0_peak, g0_peak = float(ws[i0]), float(vals_open[i0])
    #ws_peak, gs_peak = float(ws[is_]), float(vals_star[is_])

    #if IS_WORLD_ROOT:
        #print("Open-loop sampled peak : omega =", w0_peak, " gain =", g0_peak)
        #print("Best MIMO sampled peak : omega =", ws_peak, " gain =", gs_peak)

        ## full plot
        #plt.figure(figsize=(8,5))
        #plt.semilogy(ws, vals_open, label="open loop")
        #plt.semilogy(ws, vals_star, label="best feasible 2x2 MIMO")
        #plt.scatter([w0_peak], [g0_peak], s=40, zorder=5)
        #plt.scatter([ws_peak], [gs_peak], s=40, zorder=5)
        #plt.xlabel(r"$\omega$")
        #plt.ylabel(r"$\|G_{cl}(i\omega)\|$")
        #plt.title("Open loop vs best feasible 2x2 MIMO")
        #plt.grid(True, which="both", ls="--", alpha=0.4)
        #plt.legend()
        #plt.tight_layout()
        #plt.show()

        ## zoom near the best peak
        #zoom_halfwidth = max(0.5, 0.15*abs(ws_peak))
        #mask = (ws >= ws_peak - zoom_halfwidth) & (ws <= ws_peak + zoom_halfwidth)

        #plt.figure(figsize=(8,5))
        #plt.semilogy(ws[mask], vals_star[mask], label="best feasible 2x2 MIMO")
        #plt.scatter([ws_peak], [gs_peak], s=40, zorder=5, label="sampled peak")
        #plt.xlabel(r"$\omega$")
        #plt.ylabel(r"$\|G_{cl}(i\omega)\|$")
        #plt.title("Zoom on dominant closed-loop peak")
        #plt.grid(True, which="both", ls="--", alpha=0.4)
        #plt.legend()
        #plt.tight_layout()
        #plt.show

    #return {
        #"ws": ws,
        #"vals_open": vals_open,
        #"vals_star": vals_star,
        #"open_peak": (w0_peak, g0_peak),
        #"best_peak": (ws_peak, gs_peak),
    #}

#K_star = best_feas["K"] 

#plot_data = plot_best_mimo_peak(
    #cl_mimo,
    #K_star,
    #wmin=1e-4,
    #wmax=2e1,
    #npos=1500,
#)

# ============================================================
# Small helpers
# ============================================================

def gather_full_vec_numpy(v: PETSc.Vec) -> np.ndarray:
    """Gather a distributed PETSc Vec into a global numpy array."""
    comm = v.comm.tompi4py()
    local = v.getArray(readonly=True).copy()
    parts = comm.allgather(local)
    return np.concatenate(parts)


def T(A: PETSc.Mat) -> PETSc.Mat:
    AH = PETSc.Mat().createHermitianTranspose(A)
    AH.assemble()
    return AH


def _block_is(comm, sizes):
    is_list = []
    start = 0
    for sz in sizes:
        is_list.append(PETSc.IS().createStride(sz, start, 1, comm=comm))
        start += sz
    return is_list


def eye_scaled(comm, n, alpha, like_mat: PETSc.Mat | None = None) -> PETSc.Mat:
    I = PETSc.Mat().create(comm=comm)
    I.setSizes([n, n])
    if like_mat is not None:
        I.setType(like_mat.getType())
    I.setUp()

    d = PETSc.Vec().create(comm=comm)
    d.setSizes(n)
    d.setFromOptions()
    d.set(alpha)
    I.setDiagonal(d)
    I.assemble()
    return I


def matmul_sparse(A: PETSc.Mat, B: PETSc.Mat) -> PETSc.Mat:
    C = A.matMult(B)
    C.assemble()
    return C


def make_lu_ksp(S: PETSc.Mat) -> PETSc.KSP:
    ksp = PETSc.KSP().create(S.comm)
    ksp.setOperators(S)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorShift(shift_type=PETSc.Mat.FactorShiftType.NONZERO, amount=1e-12)
    ksp.setUp()
    ksp.setErrorIfNotConverged(True)
    return ksp


def make_dense_small_mat(comm, A: np.ndarray) -> PETSc.Mat:
    A = np.asarray(A, dtype=np.complex128)
    m, n = A.shape

    M = PETSc.Mat().createAIJ([m, n], nnz=n, comm=comm)
    M.setUp()

    rlo, rhi = M.getOwnershipRange()
    cols = np.arange(n, dtype=PETSc.IntType)

    for i in range(rlo, rhi):
        M.setValues([i], cols.tolist(), A[i, :])

    M.assemble()
    return M

def make_rect_outer_vec_row(left: PETSc.Vec, row: np.ndarray, tol: float = 1e-14) -> PETSc.Mat:
    """
    Build the rectangular matrix left * row^H, shape (n, r),
    where left is a PETSc Vec of length n and row is a numpy vector of length r.
    """
    comm = left.comm
    n = left.getSize()
    row = np.asarray(row, dtype=np.complex128).reshape(-1)
    r = row.size

    R = PETSc.Mat().createAIJ([n, r], comm=comm)
    R.setUp()

    lo, hi = left.getOwnershipRange()
    left_loc = left.getArray(readonly=True)
    nz_cols = np.flatnonzero(np.abs(row) > tol)
    if nz_cols.size == 0:
        R.assemble()
        return R

    rowH_nz = np.conjugate(row[nz_cols])
    for i_loc, i_glob in enumerate(range(lo, hi)):
        vals = left_loc[i_loc] * rowH_nz
        if vals.size:
            R.setValues([i_glob], nz_cols.astype(PETSc.IntType).tolist(), vals)
    R.assemble()
    return R


def make_rect_outer_col_vecH(col: np.ndarray, right: PETSc.Vec, tol: float = 1e-14) -> PETSc.Mat:
    """
    Build the rectangular matrix col * right^H, shape (r, n),
    where col is a numpy vector of length r and right is a PETSc Vec of length n.
    """
    comm = right.comm
    col = np.asarray(col, dtype=np.complex128).reshape(-1)
    r = col.size
    n = right.getSize()

    R = PETSc.Mat().createAIJ([r, n], comm=comm)
    R.setUp()

    right_full = gather_full_vec_numpy(right)
    rightH_full = np.conjugate(right_full)

    rlo, rhi = R.getOwnershipRange()
    nz_cols = np.flatnonzero(np.abs(rightH_full) > tol)
    if nz_cols.size == 0:
        R.assemble()
        return R

    for i in range(rlo, rhi):
        if abs(col[i]) > tol:
            vals = col[i] * rightH_full[nz_cols]
            R.setValues([i], nz_cols.astype(PETSc.IntType).tolist(), vals)
    R.assemble()
    return R


def build_nested_aij(blocks, sizes, comm) -> PETSc.Mat:
    isrows = _block_is(comm, sizes)
    iscols = _block_is(comm, sizes)
    N = PETSc.Mat().createNest(blocks, isrows=isrows, iscols=iscols, comm=comm)
    N.assemble()
    A = N.convert("aij")
    A.assemble()
    return A


def split_vec_by_is(v: PETSc.Vec, iset: PETSc.IS) -> PETSc.Vec:
    subv = v.getSubVector(iset)
    out = subv.copy()
    v.restoreSubVector(iset, subv)
    return out


def set_subvector(dst: PETSc.Vec, iset: PETSc.IS, src: PETSc.Vec):
    subv = dst.getSubVector(iset)
    src.copy(subv)
    dst.restoreSubVector(iset, subv)


# ============================================================
# Dynamic controller in controllable canonical form
# ============================================================

class ClosedLoopDynamicCompanionSISO:
    r"""
    Dynamic SISO controller in controllable canonical form.

    Controller:
        xdot = L(a) x + mc_gain * e_r * y
        u    = n^H x
        y    = c^H q

    Closed loop:
        [ M   0 ] [qdot] = [ A      b n^H     ] [q] + [ w ]
        [ 0   I ] [xdot]   [ mc e_r c^H  L(a) ] [x]   [ 0 ]

    The Hinf branch is computed for the physical map
        w -> z,
    with
        w acting on q only and z measuring q only.

    Parameters
    ----------
    A, M : PETSc.Mat
        Plant matrices on the reduced/free space.
    b, c : PETSc.Vec
        Actuator and output vectors on the same reduced/free space.
        Here y = c^H q.
    r : int
        Controller order.
    optimize_mc_gain : bool
        If True, include mc_gain as the last optimization variable.
        Otherwise mc_gain is fixed.
    fixed_mc_gain : float
        Fixed value used when optimize_mc_gain=False.
    """

    def __init__(
        self,
        A: PETSc.Mat,
        M: PETSc.Mat,
        b: PETSc.Vec,
        c: PETSc.Vec,
        r: int,
        optimize_mc_gain: bool = False,
        fixed_mc_gain: float = 1.0,
    ):
        self.A = A
        self.M = M
        self.b = b.copy()
        self.c = c.copy()
        self.r = int(r)
        self.optimize_mc_gain = bool(optimize_mc_gain)
        self.fixed_mc_gain = float(fixed_mc_gain)

        self.comm = A.comm
        self.n = A.getSize()[0]

        self.is_q, self.is_x = _block_is(self.comm, [self.n, self.r])

        self.Ir = eye_scaled(self.comm, self.r, 1.0, like_mat=A)
        self.zero_n_r = PETSc.Mat().createAIJ([self.n, self.r], comm=self.comm)
        self.zero_r_n = PETSc.Mat().createAIJ([self.r, self.n], comm=self.comm)
        self.zero_r_r = PETSc.Mat().createAIJ([self.r, self.r], comm=self.comm)
        self.zero_n_r.assemble(); self.zero_r_n.assemble(); self.zero_r_r.assemble()

        self.er = np.zeros(self.r, dtype=np.complex128)
        self.er[-1] = 1.0

        # Z = diag(M, 0), used in the fixed-frequency Hinf branch operator.
        self.Z = build_nested_aij(
            [[self.M, None], [None, self.zero_r_r]],
            [self.n, self.r],
            self.comm,
        )

    # --------------------------------------------------------
    # Parameter pack/unpack
    # --------------------------------------------------------

    @property
    def num_params(self) -> int:
        # PyGRANSO optimizes real variables.  Complex a and n are represented as
        # independent real/imaginary coordinates:
        # [Re(a), Im(a), Re(n), Im(n), (mc_gain)]
        return 4 * self.r + (1 if self.optimize_mc_gain else 0)

    def unpack_theta(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        expected = self.num_params
        legacy_expected = 2 * self.r + (1 if self.optimize_mc_gain else 0)

        # Backward compatibility for old real-only vectors: [a, n, (mc_gain)].
        if theta.size == legacy_expected:
            a = theta[:self.r].astype(np.complex128)
            nvec = theta[self.r:2 * self.r].astype(np.complex128)
            if self.optimize_mc_gain:
                mc_gain = float(theta[-1])
            else:
                mc_gain = self.fixed_mc_gain
            return a, nvec, mc_gain

        if theta.size != expected:
            raise ValueError(
                f"theta must have length {expected} for complex a/n "
                f"or legacy length {legacy_expected}, got {theta.size}"
            )

        r = self.r
        a = theta[0:r] + 1j * theta[r:2 * r]
        nvec = theta[2 * r:3 * r] + 1j * theta[3 * r:4 * r]

        off = 4 * r
        if self.optimize_mc_gain:
            mc_gain = float(theta[off])
        else:
            mc_gain = self.fixed_mc_gain

        return a.astype(np.complex128), nvec.astype(np.complex128), mc_gain

    def pack_theta(self, a, nvec, mc_gain=None):
        a = np.asarray(a, dtype=np.complex128).reshape(self.r)
        nvec = np.asarray(nvec, dtype=np.complex128).reshape(self.r)

        parts = [np.real(a), np.imag(a), np.real(nvec), np.imag(nvec)]
        if self.optimize_mc_gain:
            if mc_gain is None:
                raise ValueError("mc_gain must be provided when optimize_mc_gain=True")
            parts.append(np.array([float(mc_gain)], dtype=float))
        return np.concatenate(parts)

    # --------------------------------------------------------
    # Controller matrices
    # --------------------------------------------------------

    def L_companion_numpy(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=np.complex128).reshape(self.r)
        L = np.zeros((self.r, self.r), dtype=np.complex128)
        for i in range(self.r - 1):
            L[i, i + 1] = 1.0
        L[-1, :] = -a
        return L

    def Ecl(self) -> PETSc.Mat:
        return build_nested_aij(
            [[self.M, None], [None, self.Ir]],
            [self.n, self.r],
            self.comm,
        )

    def Acl(self, theta) -> PETSc.Mat:
        a, nvec, mc_gain = self.unpack_theta(theta)

        BN = make_rect_outer_vec_row(self.b, nvec)
        MC = make_rect_outer_col_vecH(mc_gain * self.er, self.c)
        L = make_dense_small_mat(self.comm, self.L_companion_numpy(a))

        return build_nested_aij(
            [[self.A, BN], [MC, L]],
            [self.n, self.r],
            self.comm,
        )

    # --------------------------------------------------------
    # Sparse Boyd/Balakrishnan pencil from your slide deck
    # --------------------------------------------------------

    def sparse_boyd_pencil_no_Minv(self, theta, gamma: float):
        """
        Build the left-scaled Hamiltonian pencil from the slide deck,
        i.e. the version where the M^{-1} block has been removed by
        multiplying on the left by G = diag(M, I, I, I).

        The returned pair (Nhat, Mhat) satisfies
            s Nhat - Mhat
        with block sizes [n, r, n, r].
        """
        a, nvec, mc_gain = self.unpack_theta(theta)
        L_np = self.L_companion_numpy(a)
        L = make_dense_small_mat(self.comm, L_np)
        LH = T(L)

        Mb = self.b.duplicate()
        self.M.mult(self.b, Mb)

        BN_E = make_rect_outer_vec_row(Mb, nvec)             # M b n^H
        MC = make_rect_outer_col_vecH(mc_gain * self.er, self.c)  # mc e_r c^H
        CMH = make_rect_outer_vec_row(self.c, mc_gain * self.er)  # c (mc e_r)^H
        NBH = make_rect_outer_col_vecH(nvec, self.b)         # n b^H

        M2 = matmul_sparse(self.M, self.M)
        MA = matmul_sparse(self.M, self.A)
        AH = T(self.A)

        GS11 = M2.copy()
        GS11.scale(0.0)  # placeholder shape only

        Iginv = eye_scaled(self.comm, self.n, -1.0 / float(gamma), like_mat=self.A)
        GinvM = self.M.copy(); GinvM.scale(1.0 / float(gamma)); GinvM.assemble()

        Nhat = build_nested_aij(
            [
                [M2, None, None, None],
                [None, self.Ir, None, None],
                [None, None, self.M, None],
                [None, None, None, self.Ir],
            ],
            [self.n, self.r, self.n, self.r],
            self.comm,
        )

        Mhat = build_nested_aij(
            [
                [MA,          BN_E,  -Iginv,    None],
                [MC,          L,     None,      None],
                [-GinvM,      None,  -AH,       None],
                [None,        None,  -NBH,      -LH],
            ],
            [self.n, self.r, self.n, self.r],
            self.comm,
        )

        # Signs are chosen so that s Nhat - Mhat reproduces the slide form:
        # [ sM^2 - MA,      -MBN,     -(1/gamma)I,  0 ]
        # [ -MC,            sI-L,      0,           0 ]
        # [ +(1/gamma)M,     0,        sM+A^H,      0 ]
        # [ 0,               0,        N B^H,       sI+L^H ]
        return Nhat, Mhat

    # --------------------------------------------------------
    # Frequency-domain linear algebra for the physical q->q map
    # --------------------------------------------------------

    def build_S_aug(self, theta, omega: float):
        Ecl = self.Ecl()
        Acl = self.Acl(theta)
        S = Acl.copy()
        S.scale(-1.0)
        S.axpy(1j * float(omega), Ecl)
        S.assemble()
        return S, Acl, Ecl

    def qvec_to_aug(self, qvec: PETSc.Vec, template: PETSc.Mat) -> PETSc.Vec:
        aug = template.createVecRight()
        aug.set(0.0)
        set_subvector(aug, self.is_q, qvec)
        return aug

    def split_aug_vec(self, v_aug: PETSc.Vec):
        return split_vec_by_is(v_aug, self.is_q), split_vec_by_is(v_aug, self.is_x)

    def solve_H_via_transpose(self, ksp: PETSc.KSP, S: PETSc.Mat, b: PETSc.Vec) -> PETSc.Vec:
        rhs = S.createVecRight()
        z = S.createVecLeft()
        b.copy(rhs)
        rhs.conjugate()
        ksp.solveTranspose(rhs, z)
        z.conjugate()
        return z

    def apply_Z(self, u_aug: PETSc.Vec) -> PETSc.Vec:
        uq, _ = self.split_aug_vec(u_aug)
        Zu = self.Z.createVecRight()
        Zu.set(0.0)
        Muq = uq.duplicate()
        self.M.mult(uq, Muq)
        set_subvector(Zu, self.is_q, Muq)
        return Zu

    # --------------------------------------------------------
    # Spectral abscissa + gradient in canonical coordinates
    # --------------------------------------------------------

    def spectral_abscissa_and_grad(self, theta, eps_tol=1e-8, max_it=400, ncv=40):
        Acl = self.Acl(theta)
        Ecl = self.Ecl()

        eps = SLEPc.EPS().create(comm=self.comm)
        eps.setOperators(Acl, Ecl)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setTwoSided(True)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError("GNHEP did not converge for spectral abscissa")

        lam = eps.getEigenvalue(0)

        v = Acl.createVecRight()
        u = Acl.createVecLeft()
        eps.getEigenvector(0, v)
        eps.getLeftEigenvector(0, u)

        Eclv = v.duplicate()
        Ecl.mult(v, Eclv)
        denom = u.dot(Eclv)
        if abs(denom) < 1e-30:
            raise RuntimeError("u^H Ecl v too small in spectral_abscissa_and_grad")

        grad = np.zeros(self.num_params, dtype=float)
        for j in range(self.num_params):
            dA = self.dAcl_basis(j)
            dAv = v.duplicate()
            dA.mult(v, dAv)
            grad[j] = float(np.real(u.dot(dAv) / denom))

        return float(np.real(lam)), grad
    # --------------------------------------------------------
    # Fixed-frequency Hinf branch + gradient in canonical coordinates
    # --------------------------------------------------------

    class _DynQToQCtx:
        def __init__(self, parent, S_aug, ksp_aug):
            self.parent = parent
            self.S_aug = S_aug
            self.ksp_aug = ksp_aug
            self.u_aug = S_aug.createVecRight()
            self.y_aug = S_aug.createVecRight()

        def mult(self, A, x_q, y_q):
            rhs_aug = self.parent.qvec_to_aug(x_q, self.S_aug)
            self.ksp_aug.solve(rhs_aug, self.u_aug)
            Zu = self.parent.apply_Z(self.u_aug)
            self.y_aug = self.parent.solve_H_via_transpose(self.ksp_aug, self.S_aug, Zu)
            yq, _ = self.parent.split_aug_vec(self.y_aug)
            yq.copy(y_q)

    def make_q_to_q_shell(self, theta, omega: float):
        S_aug, Acl, Ecl = self.build_S_aug(theta, omega)
        ksp_aug = make_lu_ksp(S_aug)

        Tq = PETSc.Mat().createPython([self.n, self.n], comm=self.comm)
        Tq.setPythonContext(self._DynQToQCtx(self, S_aug, ksp_aug))
        Tq.setUp()
        Tq.assemble()
        return Tq, S_aug, ksp_aug

    def hinf_branch_value_grad(self, theta, omega, eps_tol=1e-8, max_it=400, ncv=40):
        Tq, S_aug, ksp_aug = self.make_q_to_q_shell(theta, float(omega))

        eps = SLEPc.EPS().create(comm=self.comm)
        eps.setOperators(Tq, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError(f"GHEP did not converge for Hinf branch at omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = max(float(np.real(lam)), 0.0)
        sigma = float(np.sqrt(lam))

        vq = self.M.createVecRight()
        eps.getEigenvector(0, vq)

        Mvq = vq.duplicate()
        self.M.mult(vq, Mvq)
        den = vq.dot(Mvq)
        if abs(den) < 1e-30:
            raise RuntimeError("v_q^H M v_q too small in hinf_branch_value_grad")

        vq.scale(1.0 / np.sqrt(np.real(den)))

        rhs_aug = self.qvec_to_aug(vq, S_aug)
        x_aug = S_aug.createVecRight()
        ksp_aug.solve(rhs_aug, x_aug)

        Zx = self.apply_Z(x_aug)
        z_aug = self.solve_H_via_transpose(ksp_aug, S_aug, Zx)

        if sigma <= 1e-30:
            return sigma, np.zeros(self.num_params, dtype=float)

        grad = np.zeros(self.num_params, dtype=float)
        for j in range(self.num_params):
            dA = self.dAcl_basis(j)
            dAx = x_aug.duplicate()
            dA.mult(x_aug, dAx)
            grad[j] = float(np.real(z_aug.dot(dAx)) / sigma)

        return sigma, grad
    def dAcl_basis(self, j: int) -> PETSc.Mat:
        if not (0 <= j < self.num_params):
            raise IndexError("parameter index out of range")

        r = self.r

        # Re(a_j): L[-1, j] = -a_j, so derivative is -1.
        if j < r:
            ja = j
            dL = np.zeros((r, r), dtype=np.complex128)
            dL[-1, ja] = -1.0
            dLmat = make_dense_small_mat(self.comm, dL)
            return build_nested_aij(
                [[None, None],
                 [None, dLmat]],
                [self.n, r],
                self.comm,
            )

        # Im(a_j): derivative of -a_j wrt Im(a_j) is -i.
        if j < 2 * r:
            ja = j - r
            dL = np.zeros((r, r), dtype=np.complex128)
            dL[-1, ja] = -1j
            dLmat = make_dense_small_mat(self.comm, dL)
            return build_nested_aij(
                [[None, None],
                 [None, dLmat]],
                [self.n, r],
                self.comm,
            )

        # Re(n_j): upper-right block is b n^H, derivative coefficient is +1.
        if j < 3 * r:
            jn = j - 2 * r
            dn = np.zeros(r, dtype=np.complex128)
            dn[jn] = 1.0
            dBN = make_rect_outer_vec_row(self.b, dn)
            return build_nested_aij(
                [[None, dBN],
                 [None, None]],
                [self.n, r],
                self.comm,
            )

        # Im(n_j): derivative of conj(n_j) wrt Im(n_j) is -i.
        # make_rect_outer_vec_row conjugates its row argument, so row=+i gives -i.
        if j < 4 * r:
            jn = j - 3 * r
            dn = np.zeros(r, dtype=np.complex128)
            dn[jn] = 1j
            dBN = make_rect_outer_vec_row(self.b, dn)
            return build_nested_aij(
                [[None, dBN],
                 [None, None]],
                [self.n, r],
                self.comm,
            )

        off = 4 * r

        # mc_gain: affects only lower-left block mc_gain * e_r * c^H
        if self.optimize_mc_gain and j == off:
            dMC = make_rect_outer_col_vecH(self.er, self.c)
            return build_nested_aij(
                [[None, None],
                 [dMC, None]],
                [self.n, r],
                self.comm,
            )

        raise IndexError("parameter index out of range")
    
    def hinf_branch_value(self, theta, omega, eps_tol=1e-8, max_it=400, ncv=40):
        """
        Value only:
            sigma(theta, omega)
        for the physical q->q map.
        """
        Tq, _, _ = self.make_q_to_q_shell(theta, float(omega))

        eps = SLEPc.EPS().create(comm=self.comm)
        eps.setOperators(Tq, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError(f"GHEP did not converge for Hinf value at omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = max(float(np.real(lam)), 0.0)
        return float(np.sqrt(lam))

    def gain_of_omega(self, theta, omega):
        return self.hinf_branch_value(theta, omega)

    def hinf_value_grad_newton_disk(
        self,
        theta,
        num_guess=8,
        wmin=1e-6,
        wmax=2e1,
        seed=1,
        candidate_npos=80,
        branch_eps_tol=1e-8,
        branch_max_it=400,
        branch_ncv=40,
        disk_b=20.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
        disk_midpoint_guard=True,
    ):
        """
        Dynamic Hinf oracle:
        1) candidate peak search on the actual dynamic q->q map
        2) disk certification with the dynamic pencil
        3) exact branch gradient at the selected active frequency
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)

        # candidate search on the true dynamic objective sigma(theta, omega)
        omega_star, gamma_star, local_results = find_peak_candidate_dynamic(
            self,
            theta,
            wmin=wmin,
            wmax=wmax,
            npos=candidate_npos,
            nrefine=max(3, num_guess),
        )

        gamma_star = float(gamma_star)
        omega_star = float(omega_star)

        # disk certification on the dynamic Hamiltonian pencil
        certified, cert_info = certify_global_peak_disk_dynamic_safe(
            self,
            theta,
            gamma=gamma_star,
            b=disk_b,
            eigs_tol=disk_eigs_tol,
            nev=disk_nev,
            eps_probe=disk_eps_probe,
            axis_warn=disk_axis_warn,
            midpoint_guard=disk_midpoint_guard,
        )

        # if certification finds a violating frequency, use that instead
        if not certified:
            if "omega" in cert_info and "gain" in cert_info:
                omega_star = float(cert_info["omega"])
                gamma_star = float(cert_info["gain"])
            else:
                raise RuntimeError(
                    "Disk certification failed without an explicit violating frequency; "
                    "refusing to return an uncertified Hinf objective."
                )

        gamma_branch, grad = self.hinf_branch_value_grad(
            theta,
            omega_star,
            eps_tol=branch_eps_tol,
            max_it=branch_max_it,
            ncv=branch_ncv,
        )

        # make sure the reported objective is the same branch we differentiate
        gamma_star = float(gamma_branch)

        return {
            "omega_star": float(omega_star),
            "gamma_star": float(gamma_star),
            "grad": np.asarray(grad, dtype=float).reshape(-1),
            "certified": bool(certified),
            "cert_info": cert_info,
            "local_results": local_results,
        }

    def objective_constraint_oracle_newton_disk(
        self,
        theta,
        stab_margin=1e-5,
        num_guess=8,
        wmin=1e-6,
        wmax=2e1,
        seed=1,
        candidate_npos=80,
        disk_b=20.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
    ):
        """
        Objective:
            f(theta) = certified/violating peak gamma(theta)
        Constraint:
            c(theta) = alpha(theta) + stab_margin <= 0
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)

        h = self.hinf_value_grad_newton_disk(
            theta,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
            candidate_npos=candidate_npos,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        alpha, g_alpha = self.spectral_abscissa_and_grad(theta)

        return {
            "f": float(h["gamma_star"]),
            "dfdtheta": np.asarray(h["grad"], dtype=float).reshape(-1),
            "omega_star": float(h["omega_star"]),
            "alpha": float(alpha),
            "c": float(alpha + stab_margin),
            "dcdtheta": np.asarray(g_alpha, dtype=float).reshape(-1),
            "hinf_info": h,
        }
# ============================================================
# Finite-difference checks for the analytic branch gradients
# ============================================================

def fd_check_dynamic_branches(cl_dyn, theta, dtheta, omega, h=1e-6):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    dtheta = np.asarray(dtheta, dtype=float).reshape(-1)

    alpha0, galpha = cl_dyn.spectral_abscissa_and_grad(theta)
    alphap, _ = cl_dyn.spectral_abscissa_and_grad(theta + h * dtheta)
    alpham, _ = cl_dyn.spectral_abscissa_and_grad(theta - h * dtheta)

    sigma0, gsigma = cl_dyn.hinf_branch_value_grad(theta, omega)
    sigmap, _ = cl_dyn.hinf_branch_value_grad(theta + h * dtheta, omega)
    sigmam, _ = cl_dyn.hinf_branch_value_grad(theta - h * dtheta, omega)

    out = {
        "alpha": alpha0,
        "sigma": sigma0,
        "analytic_alpha_dir": float(np.dot(galpha, dtheta)),
        "fd_alpha_dir": float((alphap - alpham) / (2.0 * h)),
        "analytic_sigma_dir": float(np.dot(gsigma, dtheta)),
        "fd_sigma_dir": float((sigmap - sigmam) / (2.0 * h)),
        "grad_alpha": galpha,
        "grad_sigma": gsigma,
    }
    return out


# ============================================================
# Example usage
# ============================================================
#
# # Assuming A, M, b, c already exist exactly as in your current script:
# #   A, M : PETSc.Mat
# #   b, c : PETSc.Vec
#
cl_dyn = ClosedLoopDynamicCompanionSISO(
    A=A,
    M=M,
    b=b,
    c=c,
    r=1,
    optimize_mc_gain=False,
    fixed_mc_gain=1.0,
)


import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def make_signed_log_grid(wmin=1e-4, wmax=2e1, npos=1200):
    ws_pos = np.logspace(np.log10(wmin), np.log10(wmax), npos)
    return np.concatenate([-ws_pos[::-1], [0.0], ws_pos])

def sample_open_vs_dynamic_zero_controller(A, M, cl_dyn, theta0, ws):
    vals_open = np.array([gain_of_omega(A, M, float(w)) for w in ws], dtype=float)
    vals_dyn  = np.array([cl_dyn.hinf_branch_value(theta0, float(w)) for w in ws], dtype=float)
    return vals_open, vals_dyn

def plot_open_vs_dynamic_zero_controller(A, M, cl_dyn, theta0,
                                         wmin=1e-4, wmax=2e1, npos=1200):
    ws = make_signed_log_grid(wmin=wmin, wmax=wmax, npos=npos)
    vals_open, vals_dyn = sample_open_vs_dynamic_zero_controller(A, M, cl_dyn, theta0, ws)

    if IS_WORLD_ROOT:
        rel_max = np.max(np.abs(vals_dyn - vals_open) / (np.abs(vals_open) + 1e-30))
        i_open = int(np.argmax(vals_open))
        i_dyn  = int(np.argmax(vals_dyn))

        print("\n=== Open-loop vs dynamic(theta0) ===")
        print("max relative difference =", rel_max)
        print("open-loop peak : omega =", ws[i_open], " gain =", vals_open[i_open])
        print("dynamic peak   : omega =", ws[i_dyn],  " gain =", vals_dyn[i_dyn])

        plt.figure(figsize=(8, 5))
        plt.semilogy(ws, vals_open, label="open loop")
        plt.semilogy(ws, vals_dyn, "--", label="dynamic, controller = 0")
        plt.scatter([ws[i_open]], [vals_open[i_open]], s=35)
        plt.scatter([ws[i_dyn]],  [vals_dyn[i_dyn]],  s=35)
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$\|G(i\omega)\|$")
        plt.title("Open loop vs dynamic controller set to zero")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ws, vals_open, vals_dyn


theta0 = np.array([
    1.0,    # stable controller poles at -1
    0.0,    # n = 0  -> controller output identically zero

], dtype=float)

# ws, vals_open, vals_dyn = plot_open_vs_dynamic_zero_controller(
#     A, M, cl_dyn, theta0,
#     wmin=1e-4,
#     wmax=2e1,
#     npos=1500,
# )

class ClosedLoopDynamicCompanionSISO:
    r"""
    Dynamic SISO controller in controllable canonical form, with static feedthrough Dk = K.

    Controller:
        xdot = L(a) x + mc_gain * e_r * y
        u    = n^H x + K * y
        y    = c^H q

    Closed loop:
        [ M   0 ] [qdot] = [ A + b K c^H     b n^H     ] [q] + [ w ]
        [ 0   I ] [xdot]   [ mc e_r c^H      L(a)      ] [x]   [ 0 ]

    Parameters
    ----------
    A, M : PETSc.Mat
        Plant matrices on the reduced/free space.
    b, c : PETSc.Vec
        Actuator and output vectors on the same reduced/free space.
        Here y = c^H q.
    r : int
        Controller order.
    optimize_mc_gain : bool
        If True, include mc_gain as an optimization variable.
        Otherwise mc_gain is fixed.
    fixed_mc_gain : float
        Fixed value used when optimize_mc_gain=False.

    Parameter vector
    ----------------
    theta = [Re(a), Im(a), Re(n), Im(n), (mc_gain), K]
    where a and n are complex controller coefficients represented by real
    optimization coordinates.
    """

    def __init__(
        self,
        A: PETSc.Mat,
        M: PETSc.Mat,
        b: PETSc.Vec,
        c: PETSc.Vec,
        r: int,
        optimize_mc_gain: bool = False,
        fixed_mc_gain: float = 1.0,
    ):
        self.A = A
        self.M = M
        self.b = b.copy()
        self.c = c.copy()
        self.r = int(r)
        self.optimize_mc_gain = bool(optimize_mc_gain)
        self.fixed_mc_gain = float(fixed_mc_gain)

        self.comm = A.comm
        self.n = A.getSize()[0]

        self.is_q, self.is_x = _block_is(self.comm, [self.n, self.r])

        self.Ir = eye_scaled(self.comm, self.r, 1.0, like_mat=A)
        self.zero_n_r = PETSc.Mat().createAIJ([self.n, self.r], comm=self.comm)
        self.zero_r_n = PETSc.Mat().createAIJ([self.r, self.n], comm=self.comm)
        self.zero_r_r = PETSc.Mat().createAIJ([self.r, self.r], comm=self.comm)
        self.zero_n_r.assemble(); self.zero_r_n.assemble(); self.zero_r_r.assemble()

        self.er = np.zeros(self.r, dtype=np.complex128)
        self.er[-1] = 1.0

        # Z = diag(M, 0), used in the fixed-frequency Hinf branch operator
        self.Z = build_nested_aij(
            [[self.M, None], [None, self.zero_r_r]],
            [self.n, self.r],
            self.comm,
        )

        # NEW: static feedthrough rank-one block b c^H
        self.BcH = make_rank1_mat_from_vecs(self.b, self.c)

    # --------------------------------------------------------
    # Parameter pack/unpack
    # --------------------------------------------------------

    @property
    def num_params(self) -> int:
        # PyGRANSO optimizes real variables.  Complex a and n are represented as
        # independent real/imaginary coordinates:
        # [Re(a), Im(a), Re(n), Im(n), (mc_gain), K]
        return 4 * self.r + (1 if self.optimize_mc_gain else 0) + 1  # +1 for real K

    def unpack_theta(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        expected = self.num_params
        legacy_expected = 2 * self.r + (1 if self.optimize_mc_gain else 0) + 1

        # Backward compatibility for old real-only vectors: [a, n, (mc_gain), K].
        if theta.size == legacy_expected:
            a = theta[:self.r].astype(np.complex128)
            nvec = theta[self.r:2 * self.r].astype(np.complex128)
            off = 2 * self.r
            if self.optimize_mc_gain:
                mc_gain = float(theta[off])
                off += 1
            else:
                mc_gain = self.fixed_mc_gain
            Kstat = float(theta[off])
            return a, nvec, mc_gain, Kstat

        if theta.size != expected:
            raise ValueError(
                f"theta must have length {expected} for complex a/n "
                f"or legacy length {legacy_expected}, got {theta.size}"
            )

        r = self.r
        a = theta[0:r] + 1j * theta[r:2 * r]
        nvec = theta[2 * r:3 * r] + 1j * theta[3 * r:4 * r]

        off = 4 * r
        if self.optimize_mc_gain:
            mc_gain = float(theta[off])
            off += 1
        else:
            mc_gain = self.fixed_mc_gain

        Kstat = float(theta[off])
        return a.astype(np.complex128), nvec.astype(np.complex128), mc_gain, Kstat

    def pack_theta(self, a, nvec, mc_gain=None, Kstat=0.0):
        a = np.asarray(a, dtype=np.complex128).reshape(self.r)
        nvec = np.asarray(nvec, dtype=np.complex128).reshape(self.r)

        parts = [np.real(a), np.imag(a), np.real(nvec), np.imag(nvec)]
        if self.optimize_mc_gain:
            if mc_gain is None:
                raise ValueError("mc_gain must be provided when optimize_mc_gain=True")
            parts.append(np.array([float(mc_gain)], dtype=float))

        parts.append(np.array([float(Kstat)], dtype=float))
        return np.concatenate(parts)

    # --------------------------------------------------------
    # Controller matrices
    # --------------------------------------------------------

    def L_companion_numpy(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=np.complex128).reshape(self.r)
        L = np.zeros((self.r, self.r), dtype=np.complex128)
        for i in range(self.r - 1):
            L[i, i + 1] = 1.0
        L[-1, :] = -a
        return L

    def Ecl(self) -> PETSc.Mat:
        return build_nested_aij(
            [[self.M, None], [None, self.Ir]],
            [self.n, self.r],
            self.comm,
        )

    def Acl(self, theta) -> PETSc.Mat:
        a, nvec, mc_gain, Kstat = self.unpack_theta(theta)

        # A11 = A + b K c^H
        A11 = self.A.copy()
        if abs(Kstat) > 0.0:
            A11.axpy(PETSc.ScalarType(Kstat), self.BcH)
        A11.assemble()

        BN = make_rect_outer_vec_row(self.b, nvec)
        MC = make_rect_outer_col_vecH(mc_gain * self.er, self.c)
        L  = make_dense_small_mat(self.comm, self.L_companion_numpy(a))

        return build_nested_aij(
            [[A11, BN], [MC, L]],
            [self.n, self.r],
            self.comm,
        )

    # --------------------------------------------------------
    # Sparse Boyd/Balakrishnan pencil
    # --------------------------------------------------------

    def sparse_boyd_pencil_no_Minv(self, theta, gamma: float):
        """
        Build the left-scaled Hamiltonian pencil
            s Nhat - Mhat
        with block sizes [n, r, n, r].
        """
        a, nvec, mc_gain, Kstat = self.unpack_theta(theta)

        # A11 = A + b K c^H
        A11 = self.A.copy()
        if abs(Kstat) > 0.0:
            A11.axpy(PETSc.ScalarType(Kstat), self.BcH)
        A11.assemble()

        L  = make_dense_small_mat(self.comm, self.L_companion_numpy(a))
        LH = T(L)

        Mb = self.b.duplicate()
        self.M.mult(self.b, Mb)

        BN_E = make_rect_outer_vec_row(Mb, nvec)                  # M b n^H
        MC   = make_rect_outer_col_vecH(mc_gain * self.er, self.c)
        NBH  = make_rect_outer_col_vecH(nvec, self.b)

        M2   = matmul_sparse(self.M, self.M)
        MA11 = matmul_sparse(self.M, A11)
        A11H = T(A11)

        Iginv = eye_scaled(self.comm, self.n, -1.0 / float(gamma), like_mat=self.A)
        GinvM = self.M.copy()
        GinvM.scale(1.0 / float(gamma))
        GinvM.assemble()

        Nhat = build_nested_aij(
            [
                [M2,   None, None, None],
                [None, self.Ir, None, None],
                [None, None, self.M, None],
                [None, None, None, self.Ir],
            ],
            [self.n, self.r, self.n, self.r],
            self.comm,
        )

        Mhat = build_nested_aij(
            [
                [MA11,        BN_E,   -Iginv,   None],
                [MC,          L,      None,     None],
                [-GinvM,      None,   -A11H,    None],
                [None,        None,   -NBH,     -LH],
            ],
            [self.n, self.r, self.n, self.r],
            self.comm,
        )

        return Nhat, Mhat

    # --------------------------------------------------------
    # Frequency-domain linear algebra for the physical q->q map
    # --------------------------------------------------------

    def build_S_aug(self, theta, omega: float):
        Ecl = self.Ecl()
        Acl = self.Acl(theta)
        S = Acl.copy()
        S.scale(-1.0)
        S.axpy(1j * float(omega), Ecl)
        S.assemble()
        return S, Acl, Ecl

    def qvec_to_aug(self, qvec: PETSc.Vec, template: PETSc.Mat) -> PETSc.Vec:
        aug = template.createVecRight()
        aug.set(0.0)
        set_subvector(aug, self.is_q, qvec)
        return aug

    def split_aug_vec(self, v_aug: PETSc.Vec):
        return split_vec_by_is(v_aug, self.is_q), split_vec_by_is(v_aug, self.is_x)

    def solve_H_via_transpose(self, ksp: PETSc.KSP, S: PETSc.Mat, b: PETSc.Vec) -> PETSc.Vec:
        rhs = S.createVecRight()
        z = S.createVecLeft()
        b.copy(rhs)
        rhs.conjugate()
        ksp.solveTranspose(rhs, z)
        z.conjugate()
        return z

    def apply_Z(self, u_aug: PETSc.Vec) -> PETSc.Vec:
        uq, _ = self.split_aug_vec(u_aug)
        Zu = self.Z.createVecRight()
        Zu.set(0.0)
        Muq = uq.duplicate()
        self.M.mult(uq, Muq)
        set_subvector(Zu, self.is_q, Muq)
        return Zu

    # --------------------------------------------------------
    # Spectral abscissa + exact gradient
    # --------------------------------------------------------

    def spectral_abscissa_and_grad(self, theta, eps_tol=1e-8, max_it=400, ncv=40):
        Acl = self.Acl(theta)
        Ecl = self.Ecl()

        eps = SLEPc.EPS().create(comm=self.comm)
        eps.setOperators(Acl, Ecl)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setTwoSided(True)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError("GNHEP did not converge for spectral abscissa")

        lam = eps.getEigenvalue(0)

        v = Acl.createVecRight()
        u = Acl.createVecLeft()
        eps.getEigenvector(0, v)
        eps.getLeftEigenvector(0, u)

        Eclv = v.duplicate()
        Ecl.mult(v, Eclv)
        denom = u.dot(Eclv)
        if abs(denom) < 1e-30:
            raise RuntimeError("u^H Ecl v too small in spectral_abscissa_and_grad")

        grad = np.zeros(self.num_params, dtype=float)
        for j in range(self.num_params):
            dA = self.dAcl_basis(j)
            dAv = v.duplicate()
            dA.mult(v, dAv)
            grad[j] = float(np.real(u.dot(dAv) / denom))

        return float(np.real(lam)), grad

    # --------------------------------------------------------
    # Fixed-frequency Hinf branch + exact gradient
    # --------------------------------------------------------

    class _DynQToQCtx:
        def __init__(self, parent, S_aug, ksp_aug):
            self.parent = parent
            self.S_aug = S_aug
            self.ksp_aug = ksp_aug
            self.u_aug = S_aug.createVecRight()
            self.y_aug = S_aug.createVecRight()

        def mult(self, A, x_q, y_q):
            rhs_aug = self.parent.qvec_to_aug(x_q, self.S_aug)
            self.ksp_aug.solve(rhs_aug, self.u_aug)
            Zu = self.parent.apply_Z(self.u_aug)
            self.y_aug = self.parent.solve_H_via_transpose(self.ksp_aug, self.S_aug, Zu)
            yq, _ = self.parent.split_aug_vec(self.y_aug)
            yq.copy(y_q)

    def make_q_to_q_shell(self, theta, omega: float):
        S_aug, Acl, Ecl = self.build_S_aug(theta, omega)
        ksp_aug = make_lu_ksp(S_aug)

        Tq = PETSc.Mat().createPython([self.n, self.n], comm=self.comm)
        Tq.setPythonContext(self._DynQToQCtx(self, S_aug, ksp_aug))
        Tq.setUp()
        Tq.assemble()
        return Tq, S_aug, ksp_aug

    def hinf_branch_value_grad(self, theta, omega, eps_tol=1e-8, max_it=400, ncv=40):
        Tq, S_aug, ksp_aug = self.make_q_to_q_shell(theta, float(omega))

        eps = SLEPc.EPS().create(comm=self.comm)
        eps.setOperators(Tq, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError(f"GHEP did not converge for Hinf branch at omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = max(float(np.real(lam)), 0.0)
        sigma = float(np.sqrt(lam))

        vq = self.M.createVecRight()
        eps.getEigenvector(0, vq)

        Mvq = vq.duplicate()
        self.M.mult(vq, Mvq)
        den = vq.dot(Mvq)
        if abs(den) < 1e-30:
            raise RuntimeError("v_q^H M v_q too small in hinf_branch_value_grad")

        vq.scale(1.0 / np.sqrt(np.real(den)))

        rhs_aug = self.qvec_to_aug(vq, S_aug)
        x_aug = S_aug.createVecRight()
        ksp_aug.solve(rhs_aug, x_aug)

        Zx = self.apply_Z(x_aug)
        z_aug = self.solve_H_via_transpose(ksp_aug, S_aug, Zx)

        if sigma <= 1e-30:
            return sigma, np.zeros(self.num_params, dtype=float)

        grad = np.zeros(self.num_params, dtype=float)
        for j in range(self.num_params):
            dA = self.dAcl_basis(j)
            dAx = x_aug.duplicate()
            dA.mult(x_aug, dAx)
            grad[j] = float(np.real(z_aug.dot(dAx)) / sigma)

        return sigma, grad

    def hinf_branch_value(self, theta, omega, eps_tol=1e-8, max_it=400, ncv=40):
        Tq, _, _ = self.make_q_to_q_shell(theta, float(omega))

        eps = SLEPc.EPS().create(comm=self.comm)
        eps.setOperators(Tq, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError(f"GHEP did not converge for Hinf value at omega={omega}")

        lam = eps.getEigenvalue(0)
        lam = max(float(np.real(lam)), 0.0)
        return float(np.sqrt(lam))

    def gain_of_omega(self, theta, omega):
        return self.hinf_branch_value(theta, omega)

    # --------------------------------------------------------
    # Exact basis perturbations
    # --------------------------------------------------------

    def dAcl_basis(self, j: int) -> PETSc.Mat:
        if not (0 <= j < self.num_params):
            raise IndexError("parameter index out of range")

        r = self.r

        # Re(a_j): bottom-right block L[-1, j] = -a_j.
        if j < r:
            ja = j
            dL = np.zeros((r, r), dtype=np.complex128)
            dL[-1, ja] = -1.0
            dLmat = make_dense_small_mat(self.comm, dL)
            return build_nested_aij(
                [[None, None],
                 [None, dLmat]],
                [self.n, r],
                self.comm,
            )

        # Im(a_j): derivative of -a_j wrt Im(a_j) is -i.
        if j < 2 * r:
            ja = j - r
            dL = np.zeros((r, r), dtype=np.complex128)
            dL[-1, ja] = -1j
            dLmat = make_dense_small_mat(self.comm, dL)
            return build_nested_aij(
                [[None, None],
                 [None, dLmat]],
                [self.n, r],
                self.comm,
            )

        # Re(n_j): upper-right block b n^H, derivative coefficient is +1.
        if j < 3 * r:
            jn = j - 2 * r
            dn = np.zeros(r, dtype=np.complex128)
            dn[jn] = 1.0
            dBN = make_rect_outer_vec_row(self.b, dn)
            return build_nested_aij(
                [[None, dBN],
                 [None, None]],
                [self.n, r],
                self.comm,
            )

        # Im(n_j): derivative of conj(n_j) wrt Im(n_j) is -i.
        # make_rect_outer_vec_row conjugates its row argument, so row=+i gives -i.
        if j < 4 * r:
            jn = j - 3 * r
            dn = np.zeros(r, dtype=np.complex128)
            dn[jn] = 1j
            dBN = make_rect_outer_vec_row(self.b, dn)
            return build_nested_aij(
                [[None, dBN],
                 [None, None]],
                [self.n, r],
                self.comm,
            )

        off = 4 * r

        # mc_gain: lower-left block
        if self.optimize_mc_gain:
            if j == off:
                dMC = make_rect_outer_col_vecH(self.er, self.c)
                return build_nested_aij(
                    [[None, None],
                     [dMC, None]],
                    [self.n, r],
                    self.comm,
                )
            off += 1

        # K: top-left static block b c^H
        if j == off:
            return build_nested_aij(
                [[self.BcH, None],
                 [None, None]],
                [self.n, r],
                self.comm,
            )

        raise IndexError("parameter index out of range")

    # --------------------------------------------------------
    # Hinf oracle + disk certification
    # --------------------------------------------------------

    def hinf_value_grad_newton_disk(
        self,
        theta,
        num_guess=8,
        wmin=1e-6,
        wmax=2e1,
        seed=1,
        candidate_npos=80,
        branch_eps_tol=1e-8,
        branch_max_it=400,
        branch_ncv=40,
        disk_b=20.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
        disk_midpoint_guard=True,
    ):
        theta = np.asarray(theta, dtype=float).reshape(-1)

        omega_star, gamma_star, local_results = find_peak_candidate_dynamic(
            self,
            theta,
            wmin=wmin,
            wmax=wmax,
            npos=candidate_npos,
            nrefine=max(3, num_guess),
        )

        gamma_star = float(gamma_star)
        omega_star = float(omega_star)

        if _env_flag("HINF_USE_DISK_CERT", False):
            certified, cert_info = certify_global_peak_disk_dynamic_safe(
                self,
                theta,
                gamma=gamma_star,
                b=disk_b,
                eigs_tol=disk_eigs_tol,
                nev=disk_nev,
                eps_probe=disk_eps_probe,
                axis_warn=disk_axis_warn,
                midpoint_guard=disk_midpoint_guard,
            )

            if not certified:
                if "omega" in cert_info and "gain" in cert_info:
                    omega_star = float(cert_info["omega"])
                    gamma_star = float(cert_info["gain"])
                else:
                    raise RuntimeError(
                        "Disk certification failed without an explicit violating frequency"
                    )
        else:
            certified = True
            cert_info = {
                "reason": "disk_certification_skipped_by_HINF_USE_DISK_CERT_0",
                "note": "Set HINF_USE_DISK_CERT=1 for the original disk-certifying oracle.",
            }

        gamma_branch, grad = self.hinf_branch_value_grad(
            theta,
            omega_star,
            eps_tol=branch_eps_tol,
            max_it=branch_max_it,
            ncv=branch_ncv,
        )

        gamma_star = float(gamma_branch)

        return {
            "omega_star": float(omega_star),
            "gamma_star": float(gamma_star),
            "grad": np.asarray(grad, dtype=float).reshape(-1),
            "certified": bool(certified),
            "cert_info": cert_info,
            "local_results": local_results,
        }

    def objective_constraint_oracle_newton_disk(
        self,
        theta,
        stab_margin=1e-5,
        num_guess=8,
        wmin=1e-6,
        wmax=2e1,
        seed=1,
        candidate_npos=80,
        disk_b=20.0,
        disk_eigs_tol=1e-8,
        disk_nev=60,
        disk_eps_probe=1e-6,
        disk_axis_warn=1e-3,
    ):
        theta = np.asarray(theta, dtype=float).reshape(-1)

        h = self.hinf_value_grad_newton_disk(
            theta,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
            candidate_npos=candidate_npos,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        alpha, g_alpha = self.spectral_abscissa_and_grad(theta)

        return {
            "f": float(h["gamma_star"]),
            "dfdtheta": np.asarray(h["grad"], dtype=float).reshape(-1),
            "omega_star": float(h["omega_star"]),
            "alpha": float(alpha),
            "c": float(alpha + stab_margin),
            "dcdtheta": np.asarray(g_alpha, dtype=float).reshape(-1),
            "hinf_info": h,
        }
    
# dynamic-with-Dk should reduce to static SISO when n=0
cl_dyn = ClosedLoopDynamicCompanionSISO(
    A=A, M=M, b=b, c=c,
    r=1,
    optimize_mc_gain=False,
    fixed_mc_gain=1.0,
)

# your old static class
cl_stat = ClosedLoopSOFF(A, M, b, c, feedback_sign=+1.0)   # match sign convention of A + b K c^H

a_stable = 0.3
if _env_flag("RUN_STATIC_DYNAMIC_SANITY", False):
    for K in [0.0, 0.01, -0.01, 0.05]:
        theta = np.array([a_stable, 0.0, K], dtype=float)   # [a, n, K]

        if IS_WORLD_ROOT:
            print(f"\nK = {K:+.6e}", flush=True)
        for w in [-1.0, -0.8, -0.65, -0.5, 0.5]:
            gd = cl_dyn.gain_of_omega(theta, w)
            gs = cl_stat.gain_of_omega(K, w)
            rel = abs(gd - gs) / (abs(gs) + 1e-30)
            if IS_WORLD_ROOT:
                print(f"w={w:+.6f}  dyn={gd:.8e}  stat={gs:.8e}  rel={rel:.3e}", flush=True)

        ad, _ = cl_dyn.spectral_abscissa_and_grad(theta)
        a_s, _ = cl_stat.spectral_abscissa_and_grad(K)
        if IS_WORLD_ROOT:
            print(f"alpha_dyn  = {ad:.16e}", flush=True)
            print(f"alpha_stat = {a_s:.16e}", flush=True)
else:
    _root_print("Skipping static/dynamic sanity checks; set RUN_STATIC_DYNAMIC_SANITY=1 to enable.")

# dynamic-with-Dk should reduce to static SISO when n=0
cl_dyn = ClosedLoopDynamicCompanionSISO(
    A=A, M=M, b=b, c=c,
    r=1,
    optimize_mc_gain=False,
    fixed_mc_gain=1.0,
)
import numpy as np
from scipy.optimize import minimize_scalar

def make_signed_log_grid(wmin=1e-6, wmax=2e1, npos=80):
    """
    Symmetric log grid on [-wmax,-wmin] U {0} U [wmin,wmax].
    """
    ws_pos = np.logspace(np.log10(wmin), np.log10(wmax), int(npos))
    return np.concatenate([-ws_pos[::-1], [0.0], ws_pos])


def sample_dynamic_gain_curve(cl_dyn, theta, ws):
    """
    Sample the true dynamic fixed-frequency value:
        omega -> sigma(theta, omega)

    MPI: the signed-frequency grid is embarrassingly parallel, so distribute
    omega samples over task groups.  Each sample still uses collective PETSc/SLEPc
    operations inside its FE_COMM group.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    ws = np.asarray(ws, dtype=float).reshape(-1)

    def _eval_one(w):
        return float(cl_dyn.hinf_branch_value(theta, float(w)))

    _progress_print(f"[gain-grid] sampling {len(ws)} signed frequencies over {NUM_TASK_GROUPS} task group(s)")
    vals = mpi_task_map(_eval_one, list(ws), label="sample_dynamic_gain_curve")
    _progress_print(f"[gain-grid] done; max sampled gain={float(np.max(vals)):.6e}")
    return np.array(vals, dtype=float)


def refine_dynamic_peak_in_bracket(cl_dyn, theta, w_left, w_mid, w_right,
                                   xatol=1e-8, maxiter=80):
    """
    Local bounded refinement inside one bracket [w_left, w_right].
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)

    # always test the midpoint itself
    best_w = float(w_mid)
    best_g = float(cl_dyn.hinf_branch_value(theta, best_w))

    if w_right > w_left + 1e-15:
        res = minimize_scalar(
            lambda w: -cl_dyn.hinf_branch_value(theta, float(w)),
            bounds=(float(w_left), float(w_right)),
            method="bounded",
            options={"xatol": xatol, "maxiter": maxiter},
        )
        w_loc = float(res.x)
        g_loc = float(-res.fun)
        if g_loc > best_g:
            best_w, best_g = w_loc, g_loc

    return best_w, best_g


def find_peak_candidate_dynamic(
    cl_dyn,
    theta,
    wmin=1e-6,
    wmax=2e1,
    npos=80,
    nrefine=8,
    xatol=1e-8,
    maxiter=80,
):
    """
    Candidate peak search for the true dynamic q->q gain.

    Strategy:
      1) coarse signed-log sweep
      2) take the top few sampled peaks
      3) locally refine each one with bounded scalar optimization

    Returns
    -------
    omega_star : float
    gamma_star : float
    info       : dict
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)

    _progress_print(f"[peak-search] start: npos={npos}, wmin={wmin:g}, wmax={wmax:g}, nrefine={nrefine}")
    ws = make_signed_log_grid(wmin=wmin, wmax=wmax, npos=npos)
    vals = sample_dynamic_gain_curve(cl_dyn, theta, ws)

    order = np.argsort(vals)[::-1]

    best_idx = int(order[0])
    best_w = float(ws[best_idx])
    best_g = float(vals[best_idx])

    nref = min(len(ws), max(1, int(nrefine)))
    refine_items = []
    for idx in order[:nref]:
        idx = int(idx)
        refine_items.append((
            idx,
            float(ws[max(0, idx - 1)]),
            float(ws[idx]),
            float(ws[min(len(ws) - 1, idx + 1)]),
            float(vals[idx]),
        ))

    def _refine_one(item):
        idx, w_left, w_mid, w_right, grid_gain = item
        w_loc, g_loc = refine_dynamic_peak_in_bracket(
            cl_dyn,
            theta,
            w_left,
            w_mid,
            w_right,
            xatol=xatol,
            maxiter=maxiter,
        )
        return {
            "grid_index": int(idx),
            "grid_omega": float(w_mid),
            "grid_gain": float(grid_gain),
            "refined_omega": float(w_loc),
            "refined_gain": float(g_loc),
            "bracket": (float(w_left), float(w_right)),
        }

    _progress_print(f"[peak-search] refining {len(refine_items)} candidate bracket(s)")
    local_results = mpi_task_map(_refine_one, refine_items, label="refine_dynamic_peak_candidates")
    for rec in local_results:
        if float(rec["refined_gain"]) > best_g:
            best_w = float(rec["refined_omega"])
            best_g = float(rec["refined_gain"])

    _progress_print(f"[peak-search] done: omega*={best_w:+.6e}, gamma*={best_g:.6e}")
    return best_w, best_g, {
        "grid_ws": ws,
        "grid_vals": vals,
        "local_results": local_results,
    }

def validate_near_shift_candidates_dynamic(
    cl_dyn,
    theta,
    gamma,
    lambdas,
    ncheck=6,
    probe_eps=1e-6,
):
    """
    Probe the actual dynamic gain near the first few returned pencil eigenvalues.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)

    for lam in lambdas[:min(len(lambdas), int(ncheck))]:
        w = float(np.imag(lam))
        for h in (-probe_eps, 0.0, probe_eps):
            ww = w + h
            g = cl_dyn.hinf_branch_value(theta, ww)
            if g > gamma:
                return True, {
                    "reason": "gain_violation_from_near_shift_eig",
                    "omega": float(ww),
                    "gain": float(g),
                    "lambda_hit": lam,
                }

    return False, None


def certify_global_peak_disk_dynamic_safe(
    cl_dyn,
    theta,
    gamma,
    b=20.0,
    eigs_tol=1e-8,
    nev=60,
    eps_probe=1e-6,
    axis_warn=1e-3,
    min_interval=None,
    midpoint_guard=True,
):
    """
    Safe disk certification for the dynamic controller using
    cl_dyn.sparse_boyd_pencil_no_Minv(theta, gamma).

    MPI: the adaptive disk-exclusion loop is batched.  At each pass we take the
    largest remaining intervals, evaluate up to one interval per task group, and
    merge the resulting exclusions.  This preserves the conservative nature of
    the original loop while using all available task groups on the expensive
    shift-invert pencil solves.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)

    _progress_print(f"[disk-cert] start: gamma={float(gamma):.6e}, b={float(b):.6e}, nev={int(nev)}")
    Nhat, Mhat = cl_dyn.sparse_boyd_pencil_no_Minv(theta, gamma)
    Nhat = Nhat.convert("aij"); Nhat.assemble()
    Mhat = Mhat.convert("aij"); Mhat.assemble()

    if min_interval is None:
        min_interval = eigs_tol

    visited = []

    def _gain_at(w):
        return (float(w), float(cl_dyn.hinf_branch_value(theta, float(w))))

    # Low-frequency guard is independent across probe frequencies.
    low_probe_ws = [-1e-12, 1e-12, -1e-9, 1e-9, -1e-6, 1e-6]
    for w, g in mpi_task_map(_gain_at, low_probe_ws, label="disk_dynamic_low_freq_guard"):
        if g > gamma:
            return False, {
                "reason": "low_freq_guard_hit",
                "omega": float(w),
                "gain": float(g),
            }

    def _process_interval(ab):
        lo, hi = map(float, ab)
        theta_shift = 0.5 * (lo + hi)

        if midpoint_guard:
            g_mid = float(cl_dyn.hinf_branch_value(theta, theta_shift))
            if g_mid > gamma:
                return {
                    "status": "violation",
                    "info": {
                        "reason": "midpoint_gain_violation",
                        "omega": float(theta_shift),
                        "gain": float(g_mid),
                    },
                    "visited": (lo, hi, theta_shift, 0),
                    "new_intervals": [],
                }

        lambdas, nconv = eigs_close_to_shift_pencil(
            Nhat,
            Mhat,
            theta_shift,
            nev=nev,
            tol=eigs_tol,
            max_it=2000,
        )

        if nconv == 0 or lambdas.size == 0:
            return {
                "status": "ok",
                "info": None,
                "visited": (lo, hi, theta_shift, nconv),
                "new_intervals": [],
            }

        shift = 1j * theta_shift
        lambdas = lambdas[np.argsort(np.abs(lambdas - shift))]

        # Probe nearby candidate frequencies.  This is small, so serial inside
        # the task group avoids nested global task maps.
        hit, info = validate_near_shift_candidates_dynamic(
            cl_dyn,
            theta,
            gamma,
            lambdas,
            ncheck=6,
            probe_eps=eps_probe,
        )
        if hit:
            return {
                "status": "violation",
                "info": info,
                "visited": (lo, hi, theta_shift, nconv),
                "new_intervals": [],
            }

        nearest = lambdas[0]
        nearest_dist = float(abs(nearest - shift))

        if abs(nearest.real) <= axis_warn:
            return {
                "status": "ok",
                "info": None,
                "visited": (lo, hi, theta_shift, nconv),
                "new_intervals": [],
            }

        new_intervals = []
        if np.isfinite(nearest_dist) and nearest_dist > 0.0:
            new_intervals = split_by_removed_middle(
                lo,
                hi,
                theta_shift,
                r=nearest_dist,
                min_progress=0.5 * min_interval,
            )

        return {
            "status": "ok",
            "info": None,
            "visited": (lo, hi, theta_shift, nconv),
            "new_intervals": new_intervals,
        }

    intervals = [(-float(b), 0.0), (0.0, float(b))]
    while intervals:
        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        intervals.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        batch_size = max(1, min(NUM_TASK_GROUPS, len(intervals)))
        batch = intervals[:batch_size]
        intervals = intervals[batch_size:]

        _progress_print(f"[disk-cert] interval batch: {len(batch)} active interval(s), remaining queue={len(intervals)}")
        results = mpi_task_map(_process_interval, batch, label="disk_dynamic_interval_batch")
        for res in results:
            visited.append(res["visited"])
            if res["status"] == "violation":
                return False, res["info"]
            intervals += list(res["new_intervals"])

    _progress_print(f"[disk-cert] done: fully excluded after {len(visited)} interval visit(s)")
    return True, {
        "reason": "fully_excluded",
        "window": (-float(b), float(b)),
        "visited": visited,
    }
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

def make_pygranso_combined_fn_dynamic_Konly(
    cl_dyn,
    a_fixed,
    n_fixed,
    K_center,
    K_scale=0.25,
    stab_margin=1e-5,
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
    verbose=True,
):
    """
    Optimize only the static feedthrough K inside the dynamic controller,
    keeping a and n fixed.

    theta = [a_fixed, n_fixed, K_center + K_scale * eta]
    """

    history = []
    state = {"calls": 0}

    a_fixed = float(a_fixed)
    n_fixed = float(n_fixed)
    K_center = float(K_center)
    K_scale = float(K_scale)

    def combined_fn(X_struct):
        state["calls"] += 1

        eta_torch = X_struct.eta
        eta = float(eta_torch.item())

        K = K_center + K_scale * eta
        theta = np.array([a_fixed, n_fixed, K], dtype=float)

        out = cl_dyn.objective_constraint_oracle_newton_disk(
            theta,
            stab_margin=stab_margin,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=base_seed + state["calls"],
            candidate_npos=candidate_npos,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        f_val = float(out["f"])
        g_theta_f = np.asarray(out["dfdtheta"], dtype=float).reshape(-1)
        c_val = float(out["c"])
        g_theta_c = np.asarray(out["dcdtheta"], dtype=float).reshape(-1)

        # theta = [a, n, K], and K = K_center + K_scale * eta
        df_deta = K_scale * g_theta_f[2]
        dc_deta = K_scale * g_theta_c[2]

        history.append({
            "eta": eta,
            "K": K,
            "theta": theta.copy(),
            "f": f_val,
            "alpha": float(out["alpha"]),
            "c": c_val,
            "omega_star": float(out["omega_star"]),
            "certified": bool(out["hinf_info"]["certified"]),
            "cert_info": out["hinf_info"]["cert_info"],
        })

        if verbose and IS_WORLD_ROOT:
            print(
                f"[dynamic K-only] call={state['calls']:03d}  "
                f"K={K:+.6e}  "
                f"f={f_val:.6e}  "
                f"omega*={out['omega_star']:+.6e}  "
                f"alpha={out['alpha']:.6e}  "
                f"c={c_val:.6e}  "
                f"cert={out['hinf_info']['certified']}"
            )

        dev = eta_torch.device
        dt  = eta_torch.dtype

        f = f_val
        f_grad = torch.tensor([[df_deta]], device=dev, dtype=dt)

        ci = torch.tensor([[c_val]], device=dev, dtype=dt)
        ci_grad = torch.tensor([[dc_deta]], device=dev, dtype=dt)

        return [f, f_grad, ci, ci_grad, None, None]

    combined_fn.history = history
    return combined_fn


def best_feasible_from_history_Konly(history):
    feas = [rec for rec in history if rec["c"] <= 0.0]
    if len(feas) == 0:
        return None
    return min(feas, key=lambda rec: rec["f"])



def make_signed_log_grid(wmin=1e-6, wmax=2e1, npos=1500):
    ws_pos = np.logspace(np.log10(wmin), np.log10(wmax), int(npos))
    return np.concatenate([-ws_pos[::-1], [1e-12], ws_pos])

def plot_hinf_all_frequency_dynamic(cl_dyn, theta_star, wmin=1e-6, wmax=2e1, npos=1500):
    theta_star = np.asarray(theta_star, dtype=float).reshape(-1)

    ws = make_signed_log_grid(wmin=wmin, wmax=wmax, npos=npos)
    vals = np.array([cl_dyn.hinf_branch_value(theta_star, float(w)) for w in ws], dtype=float)

    ipeak = int(np.argmax(vals))
    w_peak = float(ws[ipeak])
    g_peak = float(vals[ipeak])

    if IS_WORLD_ROOT:
        print("\n=== Hinf frequency sweep at theta* ===")
        print("theta* =", theta_star)
        print("sampled peak omega =", w_peak)
        print("sampled peak gain  =", g_peak)

        plt.figure(figsize=(8, 5))
        plt.semilogy(ws, vals, label=r"$\sigma(\theta^\star,\omega)$")
        plt.scatter([w_peak], [g_peak], s=40, zorder=5, label="sampled peak")
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$\|G_{cl}(i\omega)\|$")
        plt.title(r"$H_\infty$ profile over all frequencies at $\theta^\star$")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # optional zoom around the peak
        zoom_halfwidth = max(0.25, 0.15 * abs(w_peak))
        mask = (ws >= w_peak - zoom_halfwidth) & (ws <= w_peak + zoom_halfwidth)

        plt.figure(figsize=(8, 5))
        plt.semilogy(ws[mask], vals[mask], label=r"$\sigma(\theta^\star,\omega)$ near peak")
        plt.scatter([w_peak], [g_peak], s=40, zorder=5, label="sampled peak")
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$\|G_{cl}(i\omega)\|$")
        plt.title("Zoom near dominant peak")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "ws": ws,
        "vals": vals,
        "peak": (w_peak, g_peak),
    }
# ------------------------------------------------------------
# K-only dynamic test
# ------------------------------------------------------------
# IMPORTANT:
# old static optimum was k_static ≈ +1.3 with negative feedback
# current dynamic class uses A + b*K*c^H
# so the matching dynamic K is about -1.3

#device = torch.device("cpu")
#dtype = torch.double

#var_spec = {"eta": [1, 1]}

#opts = pygransoStruct()
#opts.torch_device = device
#opts.double_precision = True
#opts.globalAD = False
#opts.maxit = 20
#opts.print_frequency = 1
#opts.print_level = 1
#opts.quadprog_info_msg = False

## eta = 0 means K = K_center
#opts.x0 = torch.zeros((1, 1), device=device, dtype=dtype)

## r = 1, fixed mc_gain = 1, keep a and n fixed
#a_fixed = 0.3
#n_fixed = 0.0

## use the sign-converted static optimum as center
#K_center = 0
#K_scale = 0.0025

#comb_fn = make_pygranso_combined_fn_dynamic_Konly(
    #cl_dyn,
    #a_fixed=a_fixed,
    #n_fixed=n_fixed,
    #K_center=K_center,
    #K_scale=K_scale,
    #stab_margin=1e-5,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #base_seed=2,
    #candidate_npos=80,
    #disk_b=20.0,
    #disk_eigs_tol=1e-8,
    #disk_nev=60,
    #disk_eps_probe=1e-6,
    #disk_axis_warn=1e-3,
    #verbose=True,
#)

#soln = pygranso(
    #var_spec=var_spec,
    #combined_fn=comb_fn,
    #user_opts=opts,
#)

#best_feas = best_feasible_from_history_Konly(comb_fn.history)

#if IS_WORLD_ROOT:
    #print("\n=== Best feasible dynamic iterate (K-only) ===")
    #if best_feas is None:
        #print("No feasible iterate found.")
    #else:
        #print("f        =", best_feas["f"])
        #print("alpha    =", best_feas["alpha"])
        #print("c        =", best_feas["c"])
        #print("omega*   =", best_feas["omega_star"])
        #print("K*       =", best_feas["K"])
        #print("theta*   =", best_feas["theta"])
        #print("certified =", best_feas["certified"])
        #print("cert_info =", best_feas["cert_info"])


#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import minimize_scalar

#def oracle_dynamic_Konly(cl_dyn, a_fixed, K,
                         #stab_margin=1e-5,
                         #num_guess=8,
                         #wmin=1e-6,
                         #wmax=2e1,
                         #seed=2,
                         #candidate_npos=80,
                         #disk_b=20.0,
                         #disk_eigs_tol=1e-8,
                         #disk_nev=60,
                         #disk_eps_probe=1e-6,
                         #disk_axis_warn=1e-3):
    #theta = np.array([a_fixed, 0.0, K], dtype=float)   # [a, n, K], with n=0
    #return cl_dyn.objective_constraint_oracle_newton_disk(
        #theta,
        #stab_margin=stab_margin,
        #num_guess=num_guess,
        #wmin=wmin,
        #wmax=wmax,
        #seed=seed,
        #candidate_npos=candidate_npos,
        #disk_b=disk_b,
        #disk_eigs_tol=disk_eigs_tol,
        #disk_nev=disk_nev,
        #disk_eps_probe=disk_eps_probe,
        #disk_axis_warn=disk_axis_warn,
    #)

#def scan_K_dynamic_Konly(cl_dyn, a_fixed,
                         #Kmin=-6.0, Kmax=1.0, nK=121,
                         #stab_margin=1e-5,
                         #num_guess=8,
                         #wmin=1e-6,
                         #wmax=2e1,
                         #seed=2,
                         #candidate_npos=80,
                         #disk_b=20.0,
                         #disk_eigs_tol=1e-8,
                         #disk_nev=60,
                         #disk_eps_probe=1e-6,
                         #disk_axis_warn=1e-3):
    #Ks = np.linspace(Kmin, Kmax, nK)
    #alphas = np.empty_like(Ks)
    #cs = np.empty_like(Ks)
    #fs = np.empty_like(Ks)
    #omegas = np.empty_like(Ks)
    #fs[:] = np.nan
    #omegas[:] = np.nan

    #for i, K in enumerate(Ks):
        #out = oracle_dynamic_Konly(
            #cl_dyn, a_fixed, K,
            #stab_margin=stab_margin,
            #num_guess=num_guess,
            #wmin=wmin,
            #wmax=wmax,
            #seed=seed,
            #candidate_npos=candidate_npos,
            #disk_b=disk_b,
            #disk_eigs_tol=disk_eigs_tol,
            #disk_nev=disk_nev,
            #disk_eps_probe=disk_eps_probe,
            #disk_axis_warn=disk_axis_warn,
        #)
        #alphas[i] = out["alpha"]
        #cs[i] = out["c"]
        #if out["c"] <= 0.0:
            #fs[i] = out["f"]
            #omegas[i] = out["omega_star"]

        #if IS_WORLD_ROOT:
            #print(f"K={K:+.6f}  alpha={out['alpha']:+.6e}  "
                  #f"c={out['c']:+.6e}  f={out['f']:.6e}  "
                  #f"omega*={out['omega_star']:+.6e}")

    #return Ks, alphas, cs, fs, omegas

#def contiguous_true_intervals(mask):
    #intervals = []
    #start = None
    #for i, m in enumerate(mask):
        #if m and start is None:
            #start = i
        #elif not m and start is not None:
            #intervals.append((start, i - 1))
            #start = None
    #if start is not None:
        #intervals.append((start, len(mask) - 1))
    #return intervals

#def refine_best_feasible_K(cl_dyn, a_fixed, Ks, cs, fs,
                           #stab_margin=1e-5,
                           #num_guess=8,
                           #wmin=1e-6,
                           #wmax=2e1,
                           #seed=2,
                           #candidate_npos=80,
                           #disk_b=20.0,
                           #disk_eigs_tol=1e-8,
                           #disk_nev=60,
                           #disk_eps_probe=1e-6,
                           #disk_axis_warn=1e-3):
    #feasible = np.isfinite(fs) & (cs <= 0.0)
    #if not np.any(feasible):
        #return None

    #idx_best = int(np.nanargmin(fs))
    #K_best_grid = float(Ks[idx_best])

    ## Find the feasible interval containing the best grid point
    #ints = contiguous_true_intervals(feasible)
    #bracket = None
    #for i0, i1 in ints:
        #if i0 <= idx_best <= i1:
            #bracket = (float(Ks[i0]), float(Ks[i1]))
            #break

    #if bracket is None:
        #return {
            #"K_star": K_best_grid,
            #"f_star": float(fs[idx_best]),
            #"alpha_star": np.nan,
            #"omega_star": np.nan,
            #"source": "grid",
        #}

    #def obj(K):
        #out = oracle_dynamic_Konly(
            #cl_dyn, a_fixed, float(K),
            #stab_margin=stab_margin,
            #num_guess=num_guess,
            #wmin=wmin,
            #wmax=wmax,
            #seed=seed,
            #candidate_npos=candidate_npos,
            #disk_b=disk_b,
            #disk_eigs_tol=disk_eigs_tol,
            #disk_nev=disk_nev,
            #disk_eps_probe=disk_eps_probe,
            #disk_axis_warn=disk_axis_warn,
        #)
        ## Keep the bounded solve inside the feasible interval
        #if out["c"] > 0.0:
            #return 1e12 + 1e9 * out["c"]
        #return out["f"]

    #res = minimize_scalar(obj, bounds=bracket, method="bounded",
                          #options={"xatol": 1e-3, "maxiter": 30})

    #out = oracle_dynamic_Konly(
        #cl_dyn, a_fixed, float(res.x),
        #stab_margin=stab_margin,
        #num_guess=num_guess,
        #wmin=wmin,
        #wmax=wmax,
        #seed=seed,
        #candidate_npos=candidate_npos,
        #disk_b=disk_b,
        #disk_eigs_tol=disk_eigs_tol,
        #disk_nev=disk_nev,
        #disk_eps_probe=disk_eps_probe,
        #disk_axis_warn=disk_axis_warn,
    #)

    #return {
        #"K_star": float(res.x),
        #"f_star": float(out["f"]),
        #"alpha_star": float(out["alpha"]),
        #"omega_star": float(out["omega_star"]),
        #"source": "bounded_refine",
    #}

#def plot_K_scan(Ks, alphas, fs, K_best=None, f_best=None):
    #if not IS_WORLD_ROOT:
        #return

    #fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    #ax[0].plot(Ks, alphas, label=r"$\alpha(K)$")
    #ax[0].axhline(0.0, linestyle="--")
    #ax[0].set_ylabel(r"$\alpha(K)$")
    #ax[0].grid(True, alpha=0.3)
    #ax[0].legend()

    #ax[1].plot(Ks, fs, label=r"feasible $H_\infty(K)$")
    #if K_best is not None and f_best is not None:
        #ax[1].scatter([K_best], [f_best], s=40, zorder=5)
    #ax[1].set_xlabel("K")
    #ax[1].set_ylabel(r"$H_\infty(K)$")
    #ax[1].grid(True, alpha=0.3)
    #ax[1].legend()

    #plt.tight_layout()
    #plt.show()


#a_fixed = 0.3

#Ks, alphas, cs, fs, omegas = scan_K_dynamic_Konly(
    #cl_dyn,
    #a_fixed=a_fixed,
    #Kmin=-6.0,
    #Kmax=1.0,
    #nK=81,
    #stab_margin=1e-5,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #seed=2,
    #candidate_npos=80,
    #disk_b=20.0,
#)

#best = refine_best_feasible_K(
    #cl_dyn,
    #a_fixed=a_fixed,
    #Ks=Ks,
    #cs=cs,
    #fs=fs,
    #stab_margin=1e-5,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #seed=2,
    #candidate_npos=80,
    #disk_b=20.0,
#)

#if IS_WORLD_ROOT:
    #print("\n=== Best feasible K from direct 1D search ===")
    #print(best)
class ClosedLoopDynamicCompanionSISOStrictlyProper(ClosedLoopDynamicCompanionSISO):
    r"""
    Dynamic SISO controller in controllable canonical form, strictly proper (Dk = 0).

    Controller:
        xdot = L(a) x + mc_gain * e_r * y
        u    = n^H x
        y    = c^H q

    Closed loop:
        [ M   0 ] [qdot] = [ A              b n^H     ] [q] + [ w ]
        [ 0   I ] [xdot]   [ mc e_r c^H     L(a)      ] [x]   [ 0 ]

    Parameter vector
    ----------------
    theta = [Re(a), Im(a), Re(n), Im(n), (mc_gain)]
    where a and n are complex controller coefficients represented by real
    optimization coordinates.
    """

    @property
    def num_params(self) -> int:
        # PyGRANSO optimizes real variables.  Complex a and n are represented as
        # independent real/imaginary coordinates:
        # [Re(a), Im(a), Re(n), Im(n), (mc_gain)]
        return 4 * self.r + (1 if self.optimize_mc_gain else 0)

    def unpack_theta(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        expected = self.num_params
        legacy_expected = 2 * self.r + (1 if self.optimize_mc_gain else 0)

        # Backward compatibility for old real-only vectors: [a, n, (mc_gain)].
        if theta.size == legacy_expected:
            a = theta[:self.r].astype(np.complex128)
            nvec = theta[self.r:2 * self.r].astype(np.complex128)
            off = 2 * self.r
            if self.optimize_mc_gain:
                mc_gain = float(theta[off])
            else:
                mc_gain = self.fixed_mc_gain
            return a, nvec, mc_gain

        if theta.size != expected:
            raise ValueError(
                f"theta must have length {expected} for complex a/n "
                f"or legacy length {legacy_expected}, got {theta.size}"
            )

        r = self.r
        a = theta[0:r] + 1j * theta[r:2 * r]
        nvec = theta[2 * r:3 * r] + 1j * theta[3 * r:4 * r]

        off = 4 * r
        if self.optimize_mc_gain:
            mc_gain = float(theta[off])
        else:
            mc_gain = self.fixed_mc_gain

        return a.astype(np.complex128), nvec.astype(np.complex128), mc_gain

    def pack_theta(self, a, nvec, mc_gain=None):
        a = np.asarray(a, dtype=np.complex128).reshape(self.r)
        nvec = np.asarray(nvec, dtype=np.complex128).reshape(self.r)

        parts = [np.real(a), np.imag(a), np.real(nvec), np.imag(nvec)]
        if self.optimize_mc_gain:
            if mc_gain is None:
                raise ValueError("mc_gain must be provided when optimize_mc_gain=True")
            parts.append(np.array([float(mc_gain)], dtype=float))

        return np.concatenate(parts)

    def Acl(self, theta) -> PETSc.Mat:
        a, nvec, mc_gain = self.unpack_theta(theta)

        BN = make_rect_outer_vec_row(self.b, nvec)
        MC = make_rect_outer_col_vecH(mc_gain * self.er, self.c)
        L  = make_dense_small_mat(self.comm, self.L_companion_numpy(a))

        return build_nested_aij(
            [[self.A, BN], [MC, L]],
            [self.n, self.r],
            self.comm,
        )

    def sparse_boyd_pencil_no_Minv(self, theta, gamma: float):
        """
        Build the left-scaled Hamiltonian pencil
            s Nhat - Mhat
        with block sizes [n, r, n, r].
        Strictly proper case: no static feedthrough K*y.
        """
        a, nvec, mc_gain = self.unpack_theta(theta)

        L  = make_dense_small_mat(self.comm, self.L_companion_numpy(a))
        LH = T(L)

        Mb = self.b.duplicate()
        self.M.mult(self.b, Mb)

        BN_E = make_rect_outer_vec_row(Mb, nvec)                  # M b n^H
        MC   = make_rect_outer_col_vecH(mc_gain * self.er, self.c)
        NBH  = make_rect_outer_col_vecH(nvec, self.b)

        M2  = matmul_sparse(self.M, self.M)
        MA  = matmul_sparse(self.M, self.A)
        AH  = T(self.A)

        Iginv = eye_scaled(self.comm, self.n, -1.0 / float(gamma), like_mat=self.A)
        GinvM = self.M.copy()
        GinvM.scale(1.0 / float(gamma))
        GinvM.assemble()

        Nhat = build_nested_aij(
            [
                [M2,   None, None, None],
                [None, self.Ir, None, None],
                [None, None, self.M, None],
                [None, None, None, self.Ir],
            ],
            [self.n, self.r, self.n, self.r],
            self.comm,
        )

        Mhat = build_nested_aij(
            [
                [MA,          BN_E,   -Iginv,   None],
                [MC,          L,      None,     None],
                [-GinvM,      None,   -AH,      None],
                [None,        None,   -NBH,     -LH],
            ],
            [self.n, self.r, self.n, self.r],
            self.comm,
        )

        return Nhat, Mhat

    def dAcl_basis(self, j: int) -> PETSc.Mat:
        if not (0 <= j < self.num_params):
            raise IndexError("parameter index out of range")

        r = self.r

        # Re(a_j): bottom-right block L[-1, j] = -a_j.
        if j < r:
            ja = j
            dL = np.zeros((r, r), dtype=np.complex128)
            dL[-1, ja] = -1.0
            dLmat = make_dense_small_mat(self.comm, dL)
            return build_nested_aij(
                [[None, None],
                 [None, dLmat]],
                [self.n, r],
                self.comm,
            )

        # Im(a_j): derivative of -a_j wrt Im(a_j) is -i.
        if j < 2 * r:
            ja = j - r
            dL = np.zeros((r, r), dtype=np.complex128)
            dL[-1, ja] = -1j
            dLmat = make_dense_small_mat(self.comm, dL)
            return build_nested_aij(
                [[None, None],
                 [None, dLmat]],
                [self.n, r],
                self.comm,
            )

        # Re(n_j): upper-right block b n^H, derivative coefficient is +1.
        if j < 3 * r:
            jn = j - 2 * r
            dn = np.zeros(r, dtype=np.complex128)
            dn[jn] = 1.0
            dBN = make_rect_outer_vec_row(self.b, dn)
            return build_nested_aij(
                [[None, dBN],
                 [None, None]],
                [self.n, r],
                self.comm,
            )

        # Im(n_j): derivative of conj(n_j) wrt Im(n_j) is -i.
        # make_rect_outer_vec_row conjugates its row argument, so row=+i gives -i.
        if j < 4 * r:
            jn = j - 3 * r
            dn = np.zeros(r, dtype=np.complex128)
            dn[jn] = 1j
            dBN = make_rect_outer_vec_row(self.b, dn)
            return build_nested_aij(
                [[None, dBN],
                 [None, None]],
                [self.n, r],
                self.comm,
            )

        off = 4 * r

        # mc_gain: lower-left block
        if self.optimize_mc_gain and j == off:
            dMC = make_rect_outer_col_vecH(self.er, self.c)
            return build_nested_aij(
                [[None, None],
                 [dMC, None]],
                [self.n, r],
                self.comm,
            )

        raise IndexError("parameter index out of range")

cl_dyn_sp = ClosedLoopDynamicCompanionSISOStrictlyProper(
    A=A, M=M, b=b, c=c,
    r=1,
    optimize_mc_gain=False,
    fixed_mc_gain=1.0,
)

theta = cl_dyn_sp.pack_theta(
    a=np.array([-0.3 + 0.0j], dtype=np.complex128),
    nvec=np.array([0.0 + 0.0j], dtype=np.complex128),
)


import numpy as np
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


def make_pygranso_combined_fn_dynamic_strictly_proper(
    cl_dyn,
    theta_center,
    theta_scale,
    stab_margin=1e-5,
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
    verbose=True,
):
    """
    Generic PyGRANSO wrapper for the strictly proper dynamic controller.

    The optimization variable is z, with affine map
        theta = theta_center + theta_scale * z
    applied elementwise.

    This lets you:
      - optimize all parameters      : choose all theta_scale > 0
      - fix some parameters          : choose theta_scale[j] = 0 for those entries

    For the strictly proper class, theta has the complex real-coordinate form
        [Re(a), Im(a), Re(n), Im(n), (mc_gain)]

    Parameters
    ----------
    cl_dyn : ClosedLoopDynamicCompanionSISOStrictlyProper
    theta_center : array_like, shape (num_params,)
        Base point in physical coordinates.
    theta_scale : array_like, shape (num_params,)
        Elementwise scaling from optimization variable z to theta.
        Zero entries freeze parameters.
    """

    history = []
    state = {"calls": 0}

    theta_center = np.asarray(theta_center, dtype=float).reshape(-1)
    theta_scale = np.asarray(theta_scale, dtype=float).reshape(-1)

    if theta_center.size != cl_dyn.num_params:
        raise ValueError(
            f"theta_center must have length {cl_dyn.num_params}, got {theta_center.size}"
        )
    if theta_scale.size != cl_dyn.num_params:
        raise ValueError(
            f"theta_scale must have length {cl_dyn.num_params}, got {theta_scale.size}"
        )
    if np.any(~np.isfinite(theta_center)) or np.any(~np.isfinite(theta_scale)):
        raise ValueError("theta_center and theta_scale must be finite")
    if np.all(np.abs(theta_scale) == 0.0):
        raise ValueError("theta_scale cannot be identically zero")

    def combined_fn(X_struct):
        state["calls"] += 1

        z_torch = X_struct.z
        z = z_torch.detach().cpu().numpy().reshape(-1)

        theta = theta_center + theta_scale * z

        out = cl_dyn.objective_constraint_oracle_newton_disk(
            theta,
            stab_margin=stab_margin,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=base_seed + state["calls"],
            candidate_npos=candidate_npos,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        f_val = float(out["f"])
        c_val = float(out["c"])

        g_theta_f = np.asarray(out["dfdtheta"], dtype=float).reshape(-1)
        g_theta_c = np.asarray(out["dcdtheta"], dtype=float).reshape(-1)

        # Chain rule through theta = theta_center + theta_scale * z
        df_dz = theta_scale * g_theta_f
        dc_dz = theta_scale * g_theta_c

        history.append({
            "z": z.copy(),
            "theta": theta.copy(),
            "f": f_val,
            "alpha": float(out["alpha"]),
            "c": c_val,
            "omega_star": float(out["omega_star"]),
            "certified": bool(out["hinf_info"]["certified"]),
            "cert_info": out["hinf_info"]["cert_info"],
        })

        if verbose and IS_WORLD_ROOT:
            print(
                f"[dynamic strictly proper] call={state['calls']:03d}  "
                f"theta={theta}  "
                f"f={f_val:.6e}  "
                f"omega*={out['omega_star']:+.6e}  "
                f"alpha={out['alpha']:.6e}  "
                f"c={c_val:.6e}  "
                f"cert={out['hinf_info']['certified']}"
            )

        dev = z_torch.device
        dt = z_torch.dtype

        f = f_val
        f_grad = torch.tensor(df_dz.reshape(-1, 1), device=dev, dtype=dt)

        ci = torch.tensor([[c_val]], device=dev, dtype=dt)
        ci_grad = torch.tensor(dc_dz.reshape(-1, 1), device=dev, dtype=dt)

        return [f, f_grad, ci, ci_grad, None, None]

    combined_fn.history = history
    return combined_fn


def best_feasible_from_history(history):
    feas = [rec for rec in history if rec["c"] <= 0.0]
    if len(feas) == 0:
        return None
    return min(feas, key=lambda rec: rec["f"])



device = torch.device("cpu")
dtype = torch.double

cl_dyn_sp = ClosedLoopDynamicCompanionSISOStrictlyProper(
    A=A, M=M, b=b, c=c,
    r=4,
    optimize_mc_gain=False,
    fixed_mc_gain=1.0,
)

a_center = np.array([24.0, 50.0, 35.0, 10.0], dtype=np.complex128)
n_center = np.zeros(4, dtype=np.complex128)

# Complex parameter layout used by cl_dyn_sp:
# theta = [Re(a), Im(a), Re(n), Im(n)] because optimize_mc_gain=False.
theta_center = cl_dyn_sp.pack_theta(a_center, n_center)

# Set a_*_scale entries nonzero if you want to optimize a too.
# Existing behavior froze a, so these default to zero.
a_scale_re = np.zeros(4, dtype=float)
a_scale_im = np.zeros(4, dtype=float)

# n is now optimized in both real and imaginary directions.
n_scale_re = 2.5e-6 * np.ones(4, dtype=float)
n_scale_im = 2.5e-6 * np.ones(4, dtype=float)

theta_scale = np.concatenate([a_scale_re, a_scale_im, n_scale_re, n_scale_im])

var_spec = {"z": [cl_dyn_sp.num_params, 1]}


import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


def make_signed_log_grid(wmin=1e-6, wmax=2e1, npos=1500):
    ws_pos = np.logspace(np.log10(wmin), np.log10(wmax), int(npos))
    return np.concatenate([-ws_pos[::-1], [1e-12], ws_pos])

def sample_open_vs_dynamic_strictly_proper(A, M, cl_dyn_sp, theta, ws):
    vals_open = np.array([gain_of_omega(A, M, float(w)) for w in ws], dtype=float)
    vals_dyn  = np.array([cl_dyn_sp.gain_of_omega(theta, float(w)) for w in ws], dtype=float)
    return vals_open, vals_dyn

def plot_open_vs_dynamic_strictly_proper(A, M, cl_dyn_sp, theta,
                                         wmin=1e-6, wmax=2e1, npos=1500):
    ws = make_signed_log_grid(wmin=wmin, wmax=wmax, npos=npos)
    vals_open, vals_dyn = sample_open_vs_dynamic_strictly_proper(A, M, cl_dyn_sp, theta, ws)

    i_open = int(np.argmax(vals_open))
    i_dyn  = int(np.argmax(vals_dyn))

    if IS_WORLD_ROOT:
        print("\n=== Strictly proper dynamic controller curve ===")
        print("theta* =", theta)
        print("open-loop peak : omega =", ws[i_open], " gain =", vals_open[i_open])
        print("controlled peak: omega =", ws[i_dyn],  " gain =", vals_dyn[i_dyn])

        plt.figure(figsize=(8, 5))
        plt.semilogy(ws, vals_open, label="open loop")
        plt.semilogy(ws, vals_dyn,  label="strictly proper dynamic, r=2")
        plt.scatter([ws[i_open]], [vals_open[i_open]], s=35)
        plt.scatter([ws[i_dyn]],  [vals_dyn[i_dyn]],  s=35)
        plt.axvline(ws[i_dyn], linestyle="--", alpha=0.4)

        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$\|G(i\omega)\|$")
        plt.title("Open loop vs strictly proper dynamic controller")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ws, vals_open, vals_dyn

#ws, vals_open, vals_dyn = plot_open_vs_dynamic_strictly_proper(
    #A, M, cl_dyn_sp, theta_star,
    #wmin=1e-6,
    #wmax=2e1,
    #npos=1500,
#)


#if IS_WORLD_ROOT:
    #mask = (ws > -1.0) & (ws < -0.1)

    #plt.figure(figsize=(8, 5))
    #plt.semilogy(ws[mask], vals_open[mask], label="open loop")
    #plt.semilogy(ws[mask], vals_dyn[mask],  label="strictly proper dynamic, r=2")
    #plt.axvline(-0.4823801594275983, linestyle="--", alpha=0.4, label="reported peak")
    #plt.xlabel(r"$\omega$")
    #plt.ylabel(r"$\|G(i\omega)\|$")
    #plt.title("Zoom near controlled peak")
    #plt.grid(True, which="both", ls="--", alpha=0.4)
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

#opts = pygransoStruct()
#opts.torch_device = device
#opts.double_precision = True
#opts.globalAD = False
#opts.maxit = 20
#opts.print_frequency = 1
#opts.print_level = 1
#opts.quadprog_info_msg = False
#opts.x0 = torch.zeros((cl_dyn_sp.num_params, 1), device=device, dtype=dtype)

#comb_fn = make_pygranso_combined_fn_dynamic_strictly_proper(
    #cl_dyn=cl_dyn_sp,
    #theta_center=theta_center,
    #theta_scale=theta_scale,
    #stab_margin=1e-5,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #base_seed=2,
    #candidate_npos=80,
    #disk_b=20.0,
    #disk_eigs_tol=1e-8,
    #disk_nev=60,
    #disk_eps_probe=1e-6,
    #disk_axis_warn=1e-3,
    #verbose=True,
#)

#soln = pygranso(
    #var_spec=var_spec,
    #combined_fn=comb_fn,
    #user_opts=opts,
#)

#best_feas = best_feasible_from_history(comb_fn.history)

#if IS_WORLD_ROOT:
    #print("\n=== Best feasible dynamic iterate (strictly proper) ===")
    #if best_feas is None:
        #print("No feasible iterate found.")
    #else:
        #print("f         =", best_feas["f"])
        #print("alpha     =", best_feas["alpha"])
        #print("c         =", best_feas["c"])
        #print("omega*    =", best_feas["omega_star"])
        #print("theta*    =", best_feas["theta"])
        #print("certified =", best_feas["certified"])
        #print("cert_info =", best_feas["cert_info"])

#device = torch.device("cpu")
#dtype = torch.double

#cl_dyn = ClosedLoopDynamicCompanionSISO(
    #A=A, M=M, b=b, c=c,
    #r=4,
    #optimize_mc_gain=False,
    #fixed_mc_gain=1.0,
#)

#a_center = np.array([4.4, 16.3, 18.6, 7.7], dtype=float)
#n_center = np.array([0.0,0.0, 0.0,0.0], dtype=float)
#K_center = -1.37229774

#theta_center = np.concatenate([a_center, n_center, np.array([K_center])])

#theta_scale = np.array([
    #0.0, 0.0, 0.0,0.0,
    #0.0, 0.0,0.0,0.00,
    #0.5
#], dtype=float)

#var_spec = {"z": [cl_dyn.num_params, 1]}

#opts = pygransoStruct()
#opts.torch_device = device
#opts.double_precision = True
#opts.globalAD = False
#opts.maxit = 20
#opts.print_frequency = 1
#opts.print_level = 1
#opts.quadprog_info_msg = False
#opts.x0 = torch.zeros((cl_dyn.num_params, 1), device=device, dtype=dtype)
#opts.wolfe1 = 1e-4
#opts.wolfe2 = 0.5

#comb_fn = make_pygranso_combined_fn_dynamic_strictly_proper(
    #cl_dyn=cl_dyn,
    #theta_center=theta_center,
    #theta_scale=theta_scale,
    #stab_margin=1e-5,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #base_seed=2,
    #candidate_npos=80,
    #disk_b=20.0,
    #disk_eigs_tol=1e-8,
    #disk_nev=60,
    #disk_eps_probe=1e-6,
    #disk_axis_warn=1e-3,
    #verbose=True,
#)

#soln = pygranso(
    #var_spec=var_spec,
    #combined_fn=comb_fn,
    #user_opts=opts,
#)

#best_feas = best_feasible_from_history(comb_fn.history)

#if IS_WORLD_ROOT:
    #print("\n=== Best feasible dynamic iterate (WithKy, r=4) ===")
    #if best_feas is None:
        #print("No feasible iterate found.")
    #else:
        #print("f         =", best_feas["f"])
        #print("alpha     =", best_feas["alpha"])
        #print("c         =", best_feas["c"])
        #print("omega*    =", best_feas["omega_star"])
        #print("theta*    =", best_feas["theta"])
        #print("certified =", best_feas["certified"])
        #print("cert_info =", best_feas["cert_info"])

#########################################################################################################################
#FEEDFORWARD
#########################################################################################################################

xa_ff = 1.03  # actuator
xs_ff = -0.98  # upstream sensor

b_ff, _ = build_actuator_vector(xa_ff, sigma_gauss, V, phi, bc, is_free)

s_ff, _ = build_sensor_state_vector(xs_ff, sigma_gauss, V, is_free)
c_ff = sensor_output_vector(M, s_ff)   # important: c = M s


#cl_dyn_ff = ClosedLoopDynamicCompanionSISO(
    #A=A, M=M, b=b_ff, c=c_ff,
    #r=4,
    #optimize_mc_gain=False,
    #fixed_mc_gain=1.0,
#)

#a_center = np.array([4.4, 16.3, 18.6, 7.7], dtype=float)
#n_center = np.zeros(4, dtype=float)
#K_center = float(0.0)

#theta_center = np.concatenate([a_center, n_center, np.array([K_center])])

#theta_scale = np.array([
    #0, 0.0, 0.0, 0.0,   # freeze a
    #0.025,0.025,0.025,0.025,   # freeze n
    #0.00            # free only K
#], dtype=float)

#var_spec_dyn = {"z": [cl_dyn_ff.num_params, 1]}

#opts_dyn0 = pygransoStruct()
#opts_dyn0.torch_device = device
#opts_dyn0.double_precision = True
#opts_dyn0.globalAD = False
#opts_dyn0.maxit = 20
#opts_dyn0.print_frequency = 1
#opts_dyn0.print_level = 1
#opts_dyn0.quadprog_info_msg = False
#opts_dyn0.x0 = torch.zeros((cl_dyn_ff.num_params, 1), device=device, dtype=dtype)

#comb_fn_dyn0 = make_pygranso_combined_fn_dynamic_strictly_proper(
    #cl_dyn=cl_dyn_ff,
    #theta_center=theta_center,
    #theta_scale=theta_scale,
    #stab_margin=1e-5,
    #num_guess=8,
    #wmin=1e-6,
    #wmax=2e1,
    #base_seed=2,
    #candidate_npos=80,
    #disk_b=20.0,
    #disk_eigs_tol=1e-8,
    #disk_nev=60,
    #disk_eps_probe=1e-6,
    #disk_axis_warn=1e-3,
    #verbose=True,
#)

#soln_dyn0 = pygranso(
    #var_spec=var_spec_dyn,
    #combined_fn=comb_fn_dyn0,
    #user_opts=opts_dyn0,
#)

#best_dyn0 = best_feasible_from_history(comb_fn_dyn0.history)

#########################################################################################################################
# Controller order performance test
#########################################################################################################################

import numpy as np
import torch
import matplotlib.pyplot as plt

def run_static_for_placement(
    A, M, b, c,
    a_center,
    device,
    dtype,
    stab_margin=1e-5,
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
    maxit=40,
    verbose=False,
):
    r = len(a_center)

    cl_dyn = ClosedLoopDynamicCompanionSISO(
        A=A, M=M, b=b, c=c,
        r=r,
        optimize_mc_gain=False,
        fixed_mc_gain=1.0,
    )

    n_center = np.zeros(r, dtype=np.complex128)
    K_center = 0.0
    theta_center = cl_dyn.pack_theta(
        np.asarray(a_center, dtype=np.complex128),
        n_center,
        Kstat=K_center,
    )

    # freeze complex a and n, optimize only real K
    theta_scale = cl_dyn.pack_theta(
        np.zeros(r, dtype=np.complex128),
        np.zeros(r, dtype=np.complex128),
        Kstat=2.5e-3,
    )

    var_spec = {"z": [cl_dyn.num_params, 1]}

    opts = pygransoStruct()
    opts.torch_device = device
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = maxit
    opts.print_frequency = 1
    opts.print_level = (1 if (verbose and IS_WORLD_ROOT) else 0)
    opts.quadprog_info_msg = False
    opts.x0 = torch.zeros((cl_dyn.num_params, 1), device=device, dtype=dtype)

    comb_fn = make_pygranso_combined_fn_dynamic_strictly_proper(
        cl_dyn=cl_dyn,
        theta_center=theta_center,
        theta_scale=theta_scale,
        stab_margin=stab_margin,
        num_guess=num_guess,
        wmin=wmin,
        wmax=wmax,
        base_seed=base_seed,
        candidate_npos=candidate_npos,
        disk_b=disk_b,
        disk_eigs_tol=disk_eigs_tol,
        disk_nev=disk_nev,
        disk_eps_probe=disk_eps_probe,
        disk_axis_warn=disk_axis_warn,
        verbose=verbose,
    )

    pygranso(var_spec=var_spec, combined_fn=comb_fn, user_opts=opts)
    best = best_feasible_from_history(comb_fn.history)
    return best, cl_dyn


def run_dynamic_for_order(
    A, M, b, c,
    r,
    a_center,
    K_seed,
    device,
    dtype,
    n_scale=2.5e-2,
    stab_margin=1e-5,
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
    maxit=40,
    verbose=False,
):
    cl_dyn = ClosedLoopDynamicCompanionSISO(
        A=A, M=M, b=b, c=c,
        r=r,
        optimize_mc_gain=False,
        fixed_mc_gain=1.0,
    )

    n_center = np.zeros(r, dtype=np.complex128)
    theta_center = cl_dyn.pack_theta(
        np.asarray(a_center, dtype=np.complex128),
        n_center,
        Kstat=K_seed,
    )

    # freeze complex a and real K, optimize n in both real and imaginary directions
    theta_scale = cl_dyn.pack_theta(
        np.zeros(r, dtype=np.complex128),                 # a frozen
        n_scale * (1.0 + 1.0j) * np.ones(r, dtype=np.complex128),  # complex n free
        Kstat=0.0,                                        # K frozen
    )

    var_spec = {"z": [cl_dyn.num_params, 1]}

    opts = pygransoStruct()
    opts.torch_device = device
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = maxit
    opts.print_frequency = 1
    opts.print_level = (1 if (verbose and IS_WORLD_ROOT) else 0)
    opts.quadprog_info_msg = False
    opts.x0 = torch.zeros((cl_dyn.num_params, 1), device=device, dtype=dtype)

    comb_fn = make_pygranso_combined_fn_dynamic_strictly_proper(
        cl_dyn=cl_dyn,
        theta_center=theta_center,
        theta_scale=theta_scale,
        stab_margin=stab_margin,
        num_guess=num_guess,
        wmin=wmin,
        wmax=wmax,
        base_seed=base_seed,
        candidate_npos=candidate_npos,
        disk_b=disk_b,
        disk_eigs_tol=disk_eigs_tol,
        disk_nev=disk_nev,
        disk_eps_probe=disk_eps_probe,
        disk_axis_warn=disk_axis_warn,
        verbose=verbose,
    )

    pygranso(var_spec=var_spec, combined_fn=comb_fn, user_opts=opts)
    best = best_feasible_from_history(comb_fn.history)
    return best


def extract_best_obj(best_dict):
    # adjust to your actual structure
    return float(best_dict["f"])


def extract_best_theta(best_dict):
    # adjust to your actual structure
    return np.asarray(best_dict["theta"], dtype=float)


def sweep_orders(
    orders,
    A, M, b, c,
    a_center_by_order,
    device,
    dtype,
    n_scale=2.5e-2,
    seeds=(2, 3, 4),
    verbose=False,
):
    results = []

    for r in orders:
        a_center = np.asarray(a_center_by_order[r], dtype=float)

        # stage 1: best static for this order
        best_static_list = []
        for seed in seeds:
            best_static, _ = run_static_for_placement(
                A=A, M=M, b=b, c=c,
                a_center=a_center,
                device=device,
                dtype=dtype,
                base_seed=seed,
                verbose=verbose,
            )
            best_static_list.append(best_static)

        best_static = min(best_static_list, key=extract_best_obj)
        f_static = extract_best_obj(best_static)
        theta_static = extract_best_theta(best_static)
        K_seed = float(theta_static[-1])

        # stage 2: best dynamic with n free, K fixed at best static
        best_dynamic_list = []
        for seed in seeds:
            best_dyn = run_dynamic_for_order(
                A=A, M=M, b=b, c=c,
                r=r,
                a_center=a_center,
                K_seed=K_seed,
                device=device,
                dtype=dtype,
                n_scale=n_scale,
                base_seed=seed,
                verbose=verbose,
            )
            best_dynamic_list.append(best_dyn)

        best_dynamic = min(best_dynamic_list, key=extract_best_obj)
        f_dynamic = extract_best_obj(best_dynamic)
        theta_dynamic = extract_best_theta(best_dynamic)

        improvement = (f_static - f_dynamic) / f_static

        results.append({
            "r": r,
            "f_static": f_static,
            "f_dynamic": f_dynamic,
            "improvement": improvement,
            "K_static": K_seed,
            "theta_dynamic": theta_dynamic.copy(),
        })

        print(
            f"r={r:2d}  "
            f"f_static={f_static:12.6f}  "
            f"f_dynamic={f_dynamic:12.6f}  "
            f"improvement={100*improvement:7.3f}%"
        )

    return results


def plot_order_sweep(results):
    orders = np.array([d["r"] for d in results], dtype=int)
    f_static = np.array([d["f_static"] for d in results], dtype=float)
    f_dynamic = np.array([d["f_dynamic"] for d in results], dtype=float)
    improvement = 100.0 * np.array([d["improvement"] for d in results], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(orders, f_static, marker="o", label="static")
    plt.plot(orders, f_dynamic, marker="o", label="dynamic")
    plt.xlabel("controller order r")
    plt.ylabel("best objective")
    plt.title("Best objective vs controller order")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(orders, improvement, marker="o")
    plt.xlabel("controller order r")
    plt.ylabel("improvement over static (%)")
    plt.title("Dynamic benefit vs controller order")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#orders = [1, 2, 3, 4, 5, 6]
#a_center_by_order = {
    #1: [4.4],
    #2: [4.4, 16.3],
    #3: [4.4, 16.3, 18.6],
    #4: [4.4, 16.3, 18.6, 7.7],
    #5: [4.4, 16.3, 18.6, 7.7, 10.0],
    #6: [4.4, 16.3, 18.6, 7.7, 10.0, 12.0],
#}

#results = sweep_orders(
    #orders=orders,
    #A=A, M=M, b=b_ff, c=c_ff,
    #a_center_by_order=a_center_by_order,
    #device=device,
    #dtype=dtype,
    #n_scale=2.5e-2,
    #seeds=(2, 3, 4),
    #verbose=False,
#)

#plot_order_sweep(results)


def gaussian_center_derivative_function(V, x0, sigma):
    """
    Derivative wrt Gaussian center x0:
        d/dx0 exp(-(x-x0)^2/(2 sigma^2))
      = ((x-x0)/sigma^2) exp(-(x-x0)^2/(2 sigma^2))
    """
    dg_fun = fem.Function(V)
    x_dofs = V.tabulate_dof_coordinates()[:, 0]

    g = np.exp(-0.5 * ((x_dofs - x0) / sigma)**2)
    dg = ((x_dofs - x0) / sigma**2) * g

    dg_fun.x.array[:] = dg
    dg_fun.x.scatter_forward()
    return dg_fun


def build_actuator_vector_and_deriv(xa, sigma, V, phi, bc, is_free):
    g_fun = gaussian_function(V, xa, sigma)
    dg_fun = gaussian_center_derivative_function(V, xa, sigma)

    b_full = fem.petsc.assemble_vector(fem.form(g_fun * ufl.conj(phi) * ufl.dx))
    b_full.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                       mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b_full, [bc])
    b_red = restrict_vec_to_is(b_full, is_free)

    db_full = fem.petsc.assemble_vector(fem.form(dg_fun * ufl.conj(phi) * ufl.dx))
    db_full.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                        mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(db_full, [bc])
    db_red = restrict_vec_to_is(db_full, is_free)

    return b_red, db_red, g_fun, dg_fun


def build_sensor_output_and_deriv(xs, sigma, V, M, is_free):
    s_fun = gaussian_function(V, xs, sigma)
    ds_fun = gaussian_center_derivative_function(V, xs, sigma)

    s_full = s_fun.x.petsc_vec.copy()
    s_full.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
    s_red = restrict_vec_to_is(s_full, is_free)

    ds_full = ds_fun.x.petsc_vec.copy()
    ds_full.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)
    ds_red = restrict_vec_to_is(ds_full, is_free)

    c_red = M.createVecLeft()
    M.mult(s_red, c_red)

    dc_red = M.createVecLeft()
    M.mult(ds_red, dc_red)

    return c_red, dc_red, s_red, ds_red


def petsc_norm(v):
    return v.norm(PETSc.NormType.NORM_2)

def check_actuator_derivative(xa, eps=1e-5):
    b0, db, *_ = build_actuator_vector_and_deriv(
        xa, sigma_gauss, V, phi, bc, is_free
    )

    bp, *_ = build_actuator_vector(
        xa + eps, sigma_gauss, V, phi, bc, is_free
    )

    bm, *_ = build_actuator_vector(
        xa - eps, sigma_gauss, V, phi, bc, is_free
    )

    fd = bp.copy()
    fd.axpy(-1.0, bm)
    fd.scale(1.0 / (2.0 * eps))

    err = fd.copy()
    err.axpy(-1.0, db)

    print("actuator derivative relative error:",
          petsc_norm(err) / max(1e-14, petsc_norm(db)))


def check_sensor_derivative(xs, eps=1e-5):
    c0, dc, *_ = build_sensor_output_and_deriv(
        xs, sigma_gauss, V, M, is_free
    )

    sp, _ = build_sensor_state_vector(xs + eps, sigma_gauss, V, is_free)
    cp = sensor_output_vector(M, sp)

    sm, _ = build_sensor_state_vector(xs - eps, sigma_gauss, V, is_free)
    cm = sensor_output_vector(M, sm)

    fd = cp.copy()
    fd.axpy(-1.0, cm)
    fd.scale(1.0 / (2.0 * eps))

    err = fd.copy()
    err.axpy(-1.0, dc)

    print("sensor derivative relative error:",
          petsc_norm(err) / max(1e-14, petsc_norm(dc)))
    

if _env_flag("RUN_DERIVATIVE_CHECKS", False):
    check_actuator_derivative(xa=1.03, eps=1e-5)
    check_sensor_derivative(xs=-0.98, eps=1e-5)
else:
    _root_print("Skipping derivative checks; set RUN_DERIVATIVE_CHECKS=1 to enable.")

xa_ff = 1.03
xs_ff = -0.98
K_center = -1.89522178
r = 4
theta_center = np.concatenate([
    np.array([1.03, -0.98]),                 # xa, xs
    np.real(a_center), np.imag(a_center),    # complex a coordinates
    np.real(n_center), np.imag(n_center),    # complex n coordinates
    np.array([K_center]),
])

theta_scale = np.concatenate([
    np.array([0.0, 0.0]),                    # freeze xa, xs
    np.zeros(r), np.zeros(r),                # freeze Re(a), Im(a)
    0.025 * np.ones(r),                     # free Re(n)
    0.025 * np.ones(r),                     # free Im(n)
    np.array([0.0]),                         # freeze K
])

#########################################################################################################################
# FEEDFORWARD PLACEMENT + CONTROLLER TEST
#
# Tests:
#   1. For each (xa, xs), optimize static gain K.
#   2. Then freeze K at the best static value and optimize dynamic numerator n.
#   3. Search over actuator/sensor placement.
#
# This does NOT require rewriting your existing PyGRANSO combined function.
#########################################################################################################################

import time
import numpy as np
import torch
import matplotlib.pyplot as plt


#########################################################################################################################
# Small utilities
#########################################################################################################################

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def extract_best_obj(best):
    if best is None:
        return np.inf
    return safe_float(best.get("f", np.inf), np.inf)


def extract_best_theta(best):
    if best is None:
        return None
    return np.asarray(best["theta"], dtype=float)


def make_cldyn_for_placement(
    xa,
    xs,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    r,
):
    """
    Build b,c for one actuator/sensor placement and return closed-loop object.
    """
    b_red, _ = build_actuator_vector(xa, sigma_gauss, V, phi, bc, is_free)

    s_red, _ = build_sensor_state_vector(xs, sigma_gauss, V, is_free)
    c_red = sensor_output_vector(M, s_red)

    cl_dyn = ClosedLoopDynamicCompanionSISO(
        A=A,
        M=M,
        b=b_red,
        c=c_red,
        r=r,
        optimize_mc_gain=False,
        fixed_mc_gain=1.0,
    )

    return cl_dyn


def run_pygranso_dynamic_problem(
    cl_dyn,
    theta_center,
    theta_scale,
    device,
    dtype,
    stab_margin=1e-5,
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
    maxit=15,
    print_level=1,
    verbose_combined=True,
):
    """
    Common PyGRANSO wrapper.
    """
    var_spec = {"z": [cl_dyn.num_params, 1]}

    opts = pygransoStruct()
    opts.torch_device = device
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = maxit
    opts.print_frequency = 1
    opts.print_level = (int(print_level) if IS_WORLD_ROOT else 0)
    opts.quadprog_info_msg = False
    opts.x0 = torch.zeros((cl_dyn.num_params, 1), device=device, dtype=dtype)

    comb_fn = make_pygranso_combined_fn_dynamic_strictly_proper(
        cl_dyn=cl_dyn,
        theta_center=theta_center,
        theta_scale=theta_scale,
        stab_margin=stab_margin,
        num_guess=num_guess,
        wmin=wmin,
        wmax=wmax,
        base_seed=base_seed,
        candidate_npos=candidate_npos,
        disk_b=disk_b,
        disk_eigs_tol=disk_eigs_tol,
        disk_nev=disk_nev,
        disk_eps_probe=disk_eps_probe,
        disk_axis_warn=disk_axis_warn,
        verbose=verbose_combined,
    )

    try:
        pygranso(
            var_spec=var_spec,
            combined_fn=comb_fn,
            user_opts=opts,
        )
        best = best_feasible_from_history(comb_fn.history)
    except Exception as e:
        print(f"    PyGRANSO failed: {repr(e)}")
        best = None
        comb_fn = None

    return best, comb_fn


#########################################################################################################################
# Controller optimization at one fixed placement
#########################################################################################################################

def optimize_static_K_for_placement(
    xa,
    xs,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    a_center,
    device,
    dtype,
    K_center=0.0,
    K_scale=2.5e-3,
    maxit=15,
    base_seed=2,
    print_level=1,
):
    """
    Static controller test:
        freeze a, freeze n, optimize only K.
    """
    a_center = np.asarray(a_center, dtype=float)
    r = len(a_center)

    cl_dyn = make_cldyn_for_placement(
        xa=xa,
        xs=xs,
        A=A,
        M=M,
        V=V,
        phi=phi,
        bc=bc,
        is_free=is_free,
        sigma_gauss=sigma_gauss,
        r=r,
    )

    n_center = np.zeros(r, dtype=np.complex128)

    theta_center = cl_dyn.pack_theta(
        np.asarray(a_center, dtype=np.complex128),
        n_center,
        Kstat=K_center,
    )

    theta_scale = cl_dyn.pack_theta(
        np.zeros(r, dtype=np.complex128),  # freeze complex a
        np.zeros(r, dtype=np.complex128),  # freeze complex n
        Kstat=K_scale,                     # free real K only
    )

    best, comb_fn = run_pygranso_dynamic_problem(
        cl_dyn=cl_dyn,
        theta_center=theta_center,
        theta_scale=theta_scale,
        device=device,
        dtype=dtype,
        maxit=maxit,
        base_seed=base_seed,
        print_level=print_level,
        verbose_combined=(print_level > 0),
    )

    return best, comb_fn


def optimize_dynamic_n_for_placement(
    xa,
    xs,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    a_center,
    K_seed,
    device,
    dtype,
    n_scale=2.5e-2,
    maxit=15,
    base_seed=2,
    print_level=1,
):
    """
    Dynamic controller test:
        freeze a,
        freeze K at static optimum,
        optimize only n.
    """
    a_center = np.asarray(a_center, dtype=float)
    r = len(a_center)

    cl_dyn = make_cldyn_for_placement(
        xa=xa,
        xs=xs,
        A=A,
        M=M,
        V=V,
        phi=phi,
        bc=bc,
        is_free=is_free,
        sigma_gauss=sigma_gauss,
        r=r,
    )

    n_center = np.zeros(r, dtype=np.complex128)

    theta_center = cl_dyn.pack_theta(
        np.asarray(a_center, dtype=np.complex128),
        n_center,
        Kstat=K_seed,
    )

    theta_scale = cl_dyn.pack_theta(
        np.zeros(r, dtype=np.complex128),                         # freeze complex a
        n_scale * (1.0 + 1.0j) * np.ones(r, dtype=np.complex128),  # free complex n
        Kstat=0.0,                                                # freeze K
    )

    best, comb_fn = run_pygranso_dynamic_problem(
        cl_dyn=cl_dyn,
        theta_center=theta_center,
        theta_scale=theta_scale,
        device=device,
        dtype=dtype,
        maxit=maxit,
        base_seed=base_seed,
        print_level=print_level,
        verbose_combined=(print_level > 0),
    )

    return best, comb_fn


def optimize_dynamic_K_and_n_for_placement(
    xa,
    xs,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    a_center,
    K_seed,
    device,
    dtype,
    n_scale=2.5e-2,
    K_scale=5e-2,
    maxit=15,
    base_seed=2,
    print_level=1,
):
    """
    Optional refinement:
        freeze a,
        optimize n and K together.

    Use this only after the n-only test is behaving.
    """
    a_center = np.asarray(a_center, dtype=float)
    r = len(a_center)

    cl_dyn = make_cldyn_for_placement(
        xa=xa,
        xs=xs,
        A=A,
        M=M,
        V=V,
        phi=phi,
        bc=bc,
        is_free=is_free,
        sigma_gauss=sigma_gauss,
        r=r,
    )

    n_center = np.zeros(r, dtype=np.complex128)

    theta_center = cl_dyn.pack_theta(
        np.asarray(a_center, dtype=np.complex128),
        n_center,
        Kstat=K_seed,
    )

    theta_scale = cl_dyn.pack_theta(
        np.zeros(r, dtype=np.complex128),                         # freeze complex a
        n_scale * (1.0 + 1.0j) * np.ones(r, dtype=np.complex128),  # free complex n
        Kstat=K_scale,                                            # free K a little
    )

    best, comb_fn = run_pygranso_dynamic_problem(
        cl_dyn=cl_dyn,
        theta_center=theta_center,
        theta_scale=theta_scale,
        device=device,
        dtype=dtype,
        maxit=maxit,
        base_seed=base_seed,
        print_level=print_level,
        verbose_combined=(print_level > 0),
    )

    return best, comb_fn


#########################################################################################################################
# Full test at one placement
#########################################################################################################################

def evaluate_placement(
    xa,
    xs,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    a_center,
    device,
    dtype,
    seeds=(2, ),
    n_scale=2.5e-2,
    K_scale_static=2.5e-3,
    run_Kn_refine=False,
    K_scale_refine=5e-2,
    maxit_static=15,
    maxit_dynamic=15,
    print_level=1,
):
    """
    For one placement:
      1. Optimize static K for several seeds.
      2. Pick best static K.
      3. Optimize dynamic n for several seeds.
      4. Optionally optimize K+n starting from static K.
    """
    t0 = time.time()

    static_bests = []

    for seed in seeds:
        best_static, _ = optimize_static_K_for_placement(
            xa=xa,
            xs=xs,
            A=A,
            M=M,
            V=V,
            phi=phi,
            bc=bc,
            is_free=is_free,
            sigma_gauss=sigma_gauss,
            a_center=a_center,
            device=device,
            dtype=dtype,
            K_center=0.0,
            K_scale=K_scale_static,
            maxit=maxit_static,
            base_seed=seed,
            print_level=print_level,
        )
        if best_static is not None:
            static_bests.append(best_static)

    if len(static_bests) == 0:
        return {
            "xa": xa,
            "xs": xs,
            "ok": False,
            "reason": "static_failed",
            "f_static": np.inf,
            "f_dynamic": np.inf,
            "f_Kn": np.inf,
            "K_static": np.nan,
            "theta_static": None,
            "theta_dynamic": None,
            "theta_Kn": None,
            "time_sec": time.time() - t0,
        }

    best_static = min(static_bests, key=extract_best_obj)
    f_static = extract_best_obj(best_static)
    theta_static = extract_best_theta(best_static)
    K_static = float(theta_static[-1])

    dynamic_bests = []

    for seed in seeds:
        best_dynamic, _ = optimize_dynamic_n_for_placement(
            xa=xa,
            xs=xs,
            A=A,
            M=M,
            V=V,
            phi=phi,
            bc=bc,
            is_free=is_free,
            sigma_gauss=sigma_gauss,
            a_center=a_center,
            K_seed=K_static,
            device=device,
            dtype=dtype,
            n_scale=n_scale,
            maxit=maxit_dynamic,
            base_seed=seed,
            print_level=print_level,
        )
        if best_dynamic is not None:
            dynamic_bests.append(best_dynamic)

    if len(dynamic_bests) == 0:
        return {
            "xa": xa,
            "xs": xs,
            "ok": False,
            "reason": "dynamic_failed",
            "f_static": f_static,
            "f_dynamic": np.inf,
            "f_Kn": np.inf,
            "K_static": K_static,
            "theta_static": theta_static,
            "theta_dynamic": None,
            "theta_Kn": None,
            "time_sec": time.time() - t0,
        }

    best_dynamic = min(dynamic_bests, key=extract_best_obj)
    f_dynamic = extract_best_obj(best_dynamic)
    theta_dynamic = extract_best_theta(best_dynamic)

    best_Kn = None
    f_Kn = np.nan
    theta_Kn = None

    if run_Kn_refine:
        Kn_bests = []

        for seed in seeds:
            best_tmp, _ = optimize_dynamic_K_and_n_for_placement(
                xa=xa,
                xs=xs,
                A=A,
                M=M,
                V=V,
                phi=phi,
                bc=bc,
                is_free=is_free,
                sigma_gauss=sigma_gauss,
                a_center=a_center,
                K_seed=K_static,
                device=device,
                dtype=dtype,
                n_scale=n_scale,
                K_scale=K_scale_refine,
                maxit=maxit_dynamic,
                base_seed=seed,
                print_level=print_level,
            )
            if best_tmp is not None:
                Kn_bests.append(best_tmp)

        if len(Kn_bests) > 0:
            best_Kn = min(Kn_bests, key=extract_best_obj)
            f_Kn = extract_best_obj(best_Kn)
            theta_Kn = extract_best_theta(best_Kn)

    improvement_dynamic = (f_static - f_dynamic) / f_static if np.isfinite(f_static) else np.nan
    improvement_Kn = (f_static - f_Kn) / f_static if np.isfinite(f_Kn) and np.isfinite(f_static) else np.nan

    return {
        "xa": xa,
        "xs": xs,
        "ok": True,
        "reason": "ok",
        "f_static": f_static,
        "f_dynamic": f_dynamic,
        "f_Kn": f_Kn,
        "improvement_dynamic": improvement_dynamic,
        "improvement_Kn": improvement_Kn,
        "K_static": K_static,
        "theta_static": theta_static,
        "theta_dynamic": theta_dynamic,
        "theta_Kn": theta_Kn,
        "time_sec": time.time() - t0,
    }


#########################################################################################################################
# Placement search
#########################################################################################################################

def random_placement_search(
    xa0,
    xs0,
    radius,
    n_trials,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    a_center,
    device,
    dtype,
    seeds=(2,),
    n_scale=2.5e-2,
    run_Kn_refine=False,
    rng_seed=123,
    maxit_static=15,
    maxit_dynamic=15,
):
    """
    Random local placement search around (xa0, xs0).

    This is usually the best first test.
    """
    rng = np.random.default_rng(rng_seed)

    candidates = [(float(xa0), float(xs0))]

    for _ in range(n_trials):
        xa = xa0 + radius * rng.uniform(-1.0, 1.0)
        xs = xs0 + radius * rng.uniform(-1.0, 1.0)
        candidates.append((float(xa), float(xs)))

    results = []

    for k, (xa, xs) in enumerate(candidates):
        print("\n" + "=" * 100)
        print(f"PLACEMENT TRIAL {k:03d}/{len(candidates)-1:03d}: xa={xa:+.6f}, xs={xs:+.6f}")
        print("=" * 100)

        out = evaluate_placement(
            xa=xa,
            xs=xs,
            A=A,
            M=M,
            V=V,
            phi=phi,
            bc=bc,
            is_free=is_free,
            sigma_gauss=sigma_gauss,
            a_center=a_center,
            device=device,
            dtype=dtype,
            seeds=seeds,
            n_scale=n_scale,
            run_Kn_refine=run_Kn_refine,
            maxit_static=maxit_static,
            maxit_dynamic=maxit_dynamic,
            print_level=1,
        )

        results.append(out)

        print(
            f"ok={out['ok']}  "
            f"f_static={out['f_static']:.6e}  "
            f"f_dynamic={out['f_dynamic']:.6e}  "
            f"improvement={100*out.get('improvement_dynamic', np.nan):.3f}%  "
            f"K_static={out['K_static']:+.6e}  "
            f"time={out['time_sec']:.1f}s"
        )

        good = [d for d in results if d["ok"] and np.isfinite(d["f_dynamic"])]
        if len(good) > 0:
            current_best = min(good, key=lambda d: d["f_dynamic"])
            print(
                f"current best: xa={current_best['xa']:+.6f}, "
                f"xs={current_best['xs']:+.6f}, "
                f"f_dynamic={current_best['f_dynamic']:.6e}"
            )

    return results


def coordinate_refine_placement(
    start_result,
    steps,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    a_center,
    device,
    dtype,
    seeds=(2,),
    n_scale=2.5e-2,
    run_Kn_refine=False,
    maxit_static=15,
    maxit_dynamic=15,
):
    """
    Simple coordinate refinement after random search.
    Tries +/- step in xa and xs and accepts improvements.
    """
    if start_result is None or not start_result["ok"]:
        raise ValueError("start_result must be a valid result dictionary.")

    best = dict(start_result)
    all_results = [best]

    for step in steps:
        print("\n" + "#" * 100)
        print(f"COORDINATE REFINEMENT STEP = {step}")
        print("#" * 100)

        improved = True

        while improved:
            improved = False

            trial_points = [
                (best["xa"] + step, best["xs"]),
                (best["xa"] - step, best["xs"]),
                (best["xa"], best["xs"] + step),
                (best["xa"], best["xs"] - step),
            ]

            for xa, xs in trial_points:
                print("\n" + "-" * 100)
                print(f"refine trial: xa={xa:+.6f}, xs={xs:+.6f}")
                print("-" * 100)

                out = evaluate_placement(
                    xa=xa,
                    xs=xs,
                    A=A,
                    M=M,
                    V=V,
                    phi=phi,
                    bc=bc,
                    is_free=is_free,
                    sigma_gauss=sigma_gauss,
                    a_center=a_center,
                    device=device,
                    dtype=dtype,
                    seeds=seeds,
                    n_scale=n_scale,
                    run_Kn_refine=run_Kn_refine,
                    maxit_static=maxit_static,
                    maxit_dynamic=maxit_dynamic,
                    print_level=1,
                )

                all_results.append(out)

                print(
                    f"ok={out['ok']}  "
                    f"f_static={out['f_static']:.6e}  "
                    f"f_dynamic={out['f_dynamic']:.6e}  "
                    f"improvement={100*out.get('improvement_dynamic', np.nan):.3f}%  "
                    f"K_static={out['K_static']:+.6e}"
                )

                if out["ok"] and out["f_dynamic"] < best["f_dynamic"]:
                    print("ACCEPTED")
                    best = dict(out)
                    improved = True

        print(
            f"best after step {step}: xa={best['xa']:+.6f}, "
            f"xs={best['xs']:+.6f}, "
            f"f_dynamic={best['f_dynamic']:.6e}"
        )

    return best, all_results


#########################################################################################################################
# Plotting / saving
#########################################################################################################################

def results_to_array(results):
    rows = []

    for d in results:
        rows.append([
            d.get("xa", np.nan),
            d.get("xs", np.nan),
            d.get("ok", False),
            d.get("f_static", np.nan),
            d.get("f_dynamic", np.nan),
            d.get("f_Kn", np.nan),
            d.get("improvement_dynamic", np.nan),
            d.get("improvement_Kn", np.nan),
            d.get("K_static", np.nan),
            d.get("time_sec", np.nan),
        ])

    return np.asarray(rows, dtype=object)


def save_results_csv(results, filename="placement_search_results.csv"):
    arr = results_to_array(results)

    header = (
        "xa,xs,ok,f_static,f_dynamic,f_Kn,"
        "improvement_dynamic,improvement_Kn,K_static,time_sec"
    )

    np.savetxt(
        filename,
        arr,
        delimiter=",",
        header=header,
        comments="",
        fmt="%s",
    )

    print(f"Saved {filename}")


def plot_placement_results(results):
    good = [d for d in results if d.get("ok", False) and np.isfinite(d.get("f_dynamic", np.inf))]

    if len(good) == 0:
        print("No good results to plot.")
        return

    xa = np.array([d["xa"] for d in good], dtype=float)
    xs = np.array([d["xs"] for d in good], dtype=float)
    f_static = np.array([d["f_static"] for d in good], dtype=float)
    f_dynamic = np.array([d["f_dynamic"] for d in good], dtype=float)
    improvement = 100.0 * np.array([d["improvement_dynamic"] for d in good], dtype=float)

    best_idx = int(np.argmin(f_dynamic))

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(xa, xs, c=f_dynamic, s=70)
    plt.scatter([xa[best_idx]], [xs[best_idx]], marker="*", s=250)
    plt.xlabel("actuator position xa")
    plt.ylabel("sensor position xs")
    plt.title("Placement search: dynamic objective")
    plt.colorbar(sc, label="f_dynamic")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(good)), f_static, marker="o", label="static")
    plt.plot(np.arange(len(good)), f_dynamic, marker="o", label="dynamic")
    plt.xlabel("accepted trial index")
    plt.ylabel("objective")
    plt.title("Static vs dynamic objective over placement trials")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(good)), improvement, marker="o")
    plt.xlabel("accepted trial index")
    plt.ylabel("dynamic improvement over static (%)")
    plt.title("Dynamic benefit over placement trials")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nBEST PLACEMENT")
    print(f"xa          = {xa[best_idx]:+.8f}")
    print(f"xs          = {xs[best_idx]:+.8f}")
    print(f"f_static    = {f_static[best_idx]:.8e}")
    print(f"f_dynamic   = {f_dynamic[best_idx]:.8e}")
    print(f"improvement = {improvement[best_idx]:.4f}%")


#########################################################################################################################
# Actual run
#########################################################################################################################

## Your current interesting feedforward placement.
#xa0 = 1.03
#xs0 = -0.98

## Current controller order.
#r = 4
#a_center = np.array([4.4, 16.3, 18.6, 7.7], dtype=float)

## Use short runs first. Then increase.
#seeds = (2,)

## First evaluate the current placement.
#baseline = evaluate_placement(
    #xa=xa0,
    #xs=xs0,
    #A=A,
    #M=M,
    #V=V,
    #phi=phi,
    #bc=bc,
    #is_free=is_free,
    #sigma_gauss=sigma_gauss,
    #a_center=a_center,
    #device=device,
    #dtype=dtype,
    #seeds=seeds,
    #n_scale=2.5e-2,
    #run_Kn_refine=False,
    #maxit_static=15,
    #maxit_dynamic=15,
    #print_level=1,
#)

#print("\n" + "=" * 100)
#print("BASELINE RESULT")
#print("=" * 100)
#print(f"xa={baseline['xa']:+.6f}, xs={baseline['xs']:+.6f}")
#print(f"f_static  = {baseline['f_static']:.8e}")
#print(f"f_dynamic = {baseline['f_dynamic']:.8e}")
#print(f"K_static  = {baseline['K_static']:+.8e}")
#print(f"improvement = {100*baseline.get('improvement_dynamic', np.nan):.4f}%")

## Random local placement search.
#random_results = random_placement_search(
    #xa0=xa0,
    #xs0=xs0,
    #radius=0.35,          # start local; increase later if this is stable
    #n_trials=8,          # increase to 25-50 for a real figure
    #A=A,
    #M=M,
    #V=V,
    #phi=phi,
    #bc=bc,
    #is_free=is_free,
    #sigma_gauss=sigma_gauss,
    #a_center=a_center,
    #device=device,
    #dtype=dtype,
    #seeds=seeds,
    #n_scale=2.5e-2,
    #run_Kn_refine=False,
    #rng_seed=123,
    #maxit_static=15,
    #maxit_dynamic=15,
#)

#all_results = [baseline] + random_results

#good = [d for d in all_results if d.get("ok", False) and np.isfinite(d.get("f_dynamic", np.inf))]
#best_random = min(good, key=lambda d: d["f_dynamic"])

#print("\n" + "=" * 100)
#print("BEST AFTER RANDOM SEARCH")
#print("=" * 100)
#print(f"xa={best_random['xa']:+.8f}, xs={best_random['xs']:+.8f}")
#print(f"f_static  = {best_random['f_static']:.8e}")
#print(f"f_dynamic = {best_random['f_dynamic']:.8e}")
#print(f"K_static  = {best_random['K_static']:+.8e}")
#print(f"improvement = {100*best_random.get('improvement_dynamic', np.nan):.4f}%")

## Optional coordinate refinement.
#best_refined, refine_results = coordinate_refine_placement(
    #start_result=best_random,
    #steps=[0.10, 0.05, 0.025],
    #A=A,
    #M=M,
    #V=V,
    #phi=phi,
    #bc=bc,
    #is_free=is_free,
    #sigma_gauss=sigma_gauss,
    #a_center=a_center,
    #device=device,
    #dtype=dtype,
    #seeds=seeds,
    #n_scale=2.5e-2,
    #run_Kn_refine=False,
    #maxit_static=15,
    #maxit_dynamic=15,
#)

#all_results = all_results + refine_results

#print("\n" + "=" * 100)
#print("FINAL BEST PLACEMENT")
#print("=" * 100)
#print(f"xa={best_refined['xa']:+.8f}, xs={best_refined['xs']:+.8f}")
#print(f"f_static  = {best_refined['f_static']:.8e}")
#print(f"f_dynamic = {best_refined['f_dynamic']:.8e}")
#print(f"K_static  = {best_refined['K_static']:+.8e}")
#print(f"theta_dynamic = {best_refined['theta_dynamic']}")
#print(f"improvement = {100*best_refined.get('improvement_dynamic', np.nan):.4f}%")

#plot_placement_results(all_results)



#########################################################################################################################
# JOINT FEEDFORWARD PLACEMENT + DYNAMIC NUMERATOR OPTIMIZATION
#
# Joint variables:
#     z = [eta_a, eta_s, z_n0, ..., z_n{r-1}]
#
# Physical variables:
#     xa = tanh_to_interval(eta_a, xa_bounds)
#     xs = tanh_to_interval(eta_s, xs_bounds)
#     n  = n_center + n_scale * z_n
#
# Frozen:
#     a = a_center
#     K = K_fixed
#
# Controller class used:
#     ClosedLoopDynamicCompanionSISO
#
# This is the "with K" dynamic controller, but K is fixed.
#########################################################################################################################

import numpy as np
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


#########################################################################################################################
# Smooth bounded placement map
#########################################################################################################################

def tanh_to_interval(eta, lo, hi):
    """
    Smoothly maps eta in R to x in (lo, hi).
    Also returns dx/deta.
    """
    eta = float(eta)
    lo = float(lo)
    hi = float(hi)

    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)

    t = np.tanh(eta)
    x = mid + rad * t
    dx_deta = rad * (1.0 - t * t)

    return x, dx_deta


def interval_to_tanh(x, lo, hi, eps=1e-12):
    """
    Inverse of tanh_to_interval, useful for initializing eta from xa or xs.
    """
    x = float(x)
    lo = float(lo)
    hi = float(hi)

    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)

    y = (x - mid) / rad
    y = np.clip(y, -1.0 + eps, 1.0 - eps)

    return np.arctanh(y)


#########################################################################################################################
# Joint combined function
#########################################################################################################################

def make_pygranso_combined_fn_joint_placement_dynamic_n(
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    a_center,
    K_center,
    K_scale,
    n_center,
    n_scale,
    xa_bounds,
    xs_bounds,
    K_bounds,
    device,
    dtype,
    stab_margin=1e-5,
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
    placement_fd=2.5e-3,
    stab_penalty_weight=1e5,
    verbose=True,
):
    """
    Joint PyGRANSO combined function.

    Optimizes:
        xa, xs, n

    Free variables:
        z = [eta_a, eta_s, z_n]

    Frozen:
        a = a_center
        K = K_fixed

    The placement derivatives are computed by centered finite differences.
    The n-derivatives use your existing analytic dfdtheta/dcdtheta from the inner oracle.
    """

    a_center = np.asarray(a_center, dtype=np.complex128).reshape(-1)
    n_center = np.asarray(n_center, dtype=np.complex128).reshape(-1)
    n_scale = np.asarray(n_scale, dtype=float).reshape(-1)

    r = len(a_center)

    if n_center.size != r:
        raise ValueError(f"n_center must have length r={r}, got {n_center.size}")

    if n_scale.size == 1:
        n_scale = float(n_scale[0]) * np.ones(r, dtype=float)

    if n_scale.size != r:
        raise ValueError(f"n_scale must be scalar or length r={r}, got {n_scale.size}")

    history = []
    state = {"calls": 0}

    def build_cl_and_theta(xa, xs, nvec, K):
        cl_dyn = make_cldyn_for_placement(
            xa=xa,
            xs=xs,
            A=A,
            M=M,
            V=V,
            phi=phi,
            bc=bc,
            is_free=is_free,
            sigma_gauss=sigma_gauss,
            r=r,
        )

        theta_ctrl = cl_dyn.pack_theta(
            a_center,
            np.asarray(nvec, dtype=np.complex128).reshape(r),
            Kstat=float(K),
        )

        return cl_dyn, theta_ctrl


    def oracle_at(xa, xs, nvec, K, seed):
        cl_dyn, theta_ctrl = build_cl_and_theta(xa, xs, nvec, K)

        out = cl_dyn.objective_constraint_oracle_newton_disk(
            theta_ctrl,
            stab_margin=stab_margin,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
            candidate_npos=candidate_npos,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        return out, cl_dyn, theta_ctrl
    def finite_difference_placement(xa, xs, nvec, K, seed):
        xa_lo, xa_hi = xa_bounds
        xs_lo, xs_hi = xs_bounds

        hxa = float(placement_fd) * max(1.0, abs(float(xa)))
        hxs = float(placement_fd) * max(1.0, abs(float(xs)))

        xa_p = min(float(xa) + hxa, xa_hi)
        xa_m = max(float(xa) - hxa, xa_lo)

        xs_p = min(float(xs) + hxs, xs_hi)
        xs_m = max(float(xs) - hxs, xs_lo)

        if xa_p == xa_m:
            dfd_xa = 0.0
            dcd_xa = 0.0
        else:
            out_p, _, _ = oracle_at(xa_p, xs, nvec, K, seed)
            out_m, _, _ = oracle_at(xa_m, xs, nvec, K, seed)

            dfd_xa = (float(out_p["f"]) - float(out_m["f"])) / (xa_p - xa_m)
            dcd_xa = (float(out_p["c"]) - float(out_m["c"])) / (xa_p - xa_m)

        if xs_p == xs_m:
            dfd_xs = 0.0
            dcd_xs = 0.0
        else:
            out_p, _, _ = oracle_at(xa, xs_p, nvec, K, seed)
            out_m, _, _ = oracle_at(xa, xs_m, nvec, K, seed)

            dfd_xs = (float(out_p["f"]) - float(out_m["f"])) / (xs_p - xs_m)
            dcd_xs = (float(out_p["c"]) - float(out_m["c"])) / (xs_p - xs_m)

        return dfd_xa, dfd_xs, dcd_xa, dcd_xs
    def combined_fn(X_struct):
        state["calls"] += 1

        z_torch = X_struct.z
        z = z_torch.detach().cpu().numpy().reshape(-1)

        expected = 2 + r + 1
        if z.size != expected:
            raise ValueError(f"z must have length {expected}, got {z.size}")

        eta_a = z[0]
        eta_s = z[1]
        z_n = z[2:2+r]
        eta_K = z[2+r]

        xa, dxa_deta = tanh_to_interval(eta_a, xa_bounds[0], xa_bounds[1])
        xs, dxs_deta = tanh_to_interval(eta_s, xs_bounds[0], xs_bounds[1])

        nvec = n_center + n_scale * z_n

        K, dK_deta = tanh_to_interval(eta_K, K_bounds[0], K_bounds[1])
        # Use one seed consistently for base and finite-difference probes.
        # This keeps the finite-difference gradient less noisy.
        seed_this = base_seed


        out0, cl_dyn, theta_ctrl = oracle_at(
            xa=xa,
            xs=xs,
            nvec=nvec,
            K=K,
            seed=seed_this,
        )
        f_val = float(out0["f"])
        c_val = float(out0["c"])

        g_theta_f = np.asarray(out0["dfdtheta"], dtype=float).reshape(-1)
        g_theta_c = np.asarray(out0["dcdtheta"], dtype=float).reshape(-1)

        expected_theta_len = 4 * r + 1
        if g_theta_f.size != expected_theta_len:
            raise ValueError(
                f"Expected inner dfdtheta length {expected_theta_len}, got {g_theta_f.size}"
            )

        if g_theta_c.size != expected_theta_len:
            raise ValueError(
                f"Expected inner dcdtheta length {expected_theta_len}, got {g_theta_c.size}"
            )

        # This older wrapper only optimizes real n directions.
        # New controller gradient layout is [Re(a), Im(a), Re(n), Im(n), K].
        g_n_f = g_theta_f[2*r:3*r]
        g_n_c = g_theta_c[2*r:3*r]

        g_K_f = g_theta_f[-1]
        g_K_c = g_theta_c[-1]
        dfd_xa, dfd_xs, dcd_xa, dcd_xs = finite_difference_placement(
            xa=xa,
            xs=xs,
            nvec=nvec,
            K=K,
            seed=seed_this,
        )

        df_dz = np.zeros_like(z, dtype=float)
        dc_dz = np.zeros_like(z, dtype=float)

        # placement variables
        df_dz[0] = dfd_xa * dxa_deta
        df_dz[1] = dfd_xs * dxs_deta

        dc_dz[0] = dcd_xa * dxa_deta
        dc_dz[1] = dcd_xs * dxs_deta

        # dynamic numerator variables
        df_dz[2:2+r] = n_scale * g_n_f
        dc_dz[2:2+r] = n_scale * g_n_c

        # bounded static gain variable
        df_dz[2+r] = dK_deta * g_K_f
        dc_dz[2+r] = dK_deta * g_K_c
        # ------------------------------------------------------------------
        # soft stability penalty.
        #
        # constraint convention:
        #     c = alpha + stab_margin <= 0
        #
        # we still pass c as an inequality constraint to pygranso, but we also
        # penalize infeasible trial points in the objective so the line search
        # is not attracted to unstable low-hinf points.
        # ------------------------------------------------------------------

        c_plus = max(c_val, 0.0)

        f_raw = f_val
        df_raw_dz = df_dz.copy()

        f_return = f_raw + stab_penalty_weight * c_plus**2
        df_return_dz = df_raw_dz.copy()

        if c_val > 0.0:
            df_return_dz += 2.0 * stab_penalty_weight * c_val * dc_dz

        rec = {
            "z": z.copy(),
            "xa": float(xa),
            "xs": float(xs),
            "a": a_center.copy(),
            "n": nvec.copy(),
            "K": float(K),
            "theta_ctrl": theta_ctrl.copy(),

            "f": f_raw,
            "f_raw": f_raw,
            "f_return": f_return,
            "stab_penalty": stab_penalty_weight * c_plus**2,

            "alpha": float(out0["alpha"]),
            "c": c_val,
            "omega_star": float(out0["omega_star"]),
            "certified": bool(out0["hinf_info"]["certified"]),
            "cert_info": out0["hinf_info"]["cert_info"],
        }
        history.append(rec)
        if verbose and IS_WORLD_ROOT:
            print(
                f"[joint placement+n+boundedK] call={state['calls']:03d}  "
                f"xa={xa:+.6f}  xs={xs:+.6f}  "
                f"f_raw={f_raw:.6e}  "
                f"f_return={f_return:.6e}  "
                f"pen={stab_penalty_weight * c_plus**2:.3e}  "
                f"omega*={out0['omega_star']:+.6e}  "
                f"alpha={out0['alpha']:+.6e}  "
                f"c={c_val:+.6e}  "
                f"K={K:+.6e}  "
                f"cert={out0['hinf_info']['certified']}"
            )
        dev = z_torch.device
        dt = z_torch.dtype

        f = f_return
        f_grad = torch.tensor(df_return_dz.reshape(-1, 1), device=dev, dtype=dt)

        ci = torch.tensor([[c_val]], device=dev, dtype=dt)
        ci_grad = torch.tensor(dc_dz.reshape(-1, 1), device=dev, dtype=dt)

        return [f, f_grad, ci, ci_grad, None, None]

    combined_fn.history = history
    return combined_fn


#########################################################################################################################
# Actual joint run
#########################################################################################################################

device = torch.device("cpu")
dtype = torch.double

xa_init = 0.0
xs_init = 0.0
K_center = 0.0
K_scale = 0.05
K_bounds = (-5.0, 5.0)
K_init = 0.0   # or previous best K, clipped inside bounds

eta_K0 = interval_to_tanh(K_init, K_bounds[0], K_bounds[1])
r = 4
a_center = np.array([4.4, 16.3, 18.6, 7.7], dtype=float)

# If you do not have a previous dynamic theta, start from zeros.
n_center = np.zeros(r, dtype=float)

# Local placement bounds around current best.
# Keep this local at first. You can increase radius later.
placement_radius = 0.25


xa_bounds = (
    max(x_min + 3.0 * sigma_gauss, xa_init - placement_radius),
    min(x_max - 3.0 * sigma_gauss, xa_init + placement_radius),
)

xs_bounds = (
    max(x_min + 3.0 * sigma_gauss, xs_init - placement_radius),
    min(x_max - 3.0 * sigma_gauss, xs_init + placement_radius),
)

# Since n_center is already near a good dynamic controller, use a moderate local scale.
n_scale = 0.25 * np.ones(r, dtype=float)

# Initialize eta so that xa_init and xs_init are exactly represented.
eta_a0 = interval_to_tanh(xa_init, xa_bounds[0], xa_bounds[1])
eta_s0 = interval_to_tanh(xs_init, xs_bounds[0], xs_bounds[1])

z0 = np.zeros(2 + r + 1, dtype=float)

z0[0] = eta_a0
z0[1] = eta_s0
z0[2:2+r] = 0.0
z0[2+r] = eta_K0

comb_joint = make_pygranso_combined_fn_joint_placement_dynamic_n(
    A=A,
    M=M,
    V=V,
    phi=phi,
    bc=bc,
    is_free=is_free,
    sigma_gauss=sigma_gauss,
    a_center=a_center,
    K_center=K_center,
    K_scale=K_scale,
    K_bounds = (-5.0, 5.0),
    n_center=n_center,
    n_scale=n_scale,
    xa_bounds=xa_bounds,
    xs_bounds=xs_bounds,
    device=device,
    dtype=dtype,
    stab_margin=1e-5,
    num_guess=6,              # reduce for joint search
    wmin=1e-6,
    wmax=2e1,
    base_seed=2,
    candidate_npos=60,        # reduce for joint search
    disk_b=20.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
    placement_fd=2.5e-3,
    verbose=True,
)

#var_spec_joint = {"z": [2 + r + 1, 1]}

#opts_joint = pygransoStruct()
#opts_joint.torch_device = device
#opts_joint.double_precision = True
#opts_joint.globalAD = False
#opts_joint.maxit = 10              # start small; increase after it behaves
#opts_joint.print_frequency = 1
#opts_joint.print_level = 1
#opts_joint.quadprog_info_msg = False

#opts_joint.x0 = torch.tensor(
    #z0.reshape(-1, 1),
    #device=device,
    #dtype=dtype,
#)

#soln_joint = pygranso(
    #var_spec=var_spec_joint,
    #combined_fn=comb_joint,
    #user_opts=opts_joint,
#)

#best_joint = best_feasible_from_history(comb_joint.history)

#if IS_WORLD_ROOT:
    #print("\n" + "=" * 100)
    #print("BEST FEASIBLE JOINT PLACEMENT + DYNAMIC-N RESULT")
    #print("=" * 100)

    #if best_joint is None:
        #print("No feasible joint iterate found.")
    #else:
        #print(f"f          = {best_joint['f']:.10e}")
        #print(f"alpha      = {best_joint['alpha']:+.10e}")
        #print(f"c          = {best_joint['c']:+.10e}")
        #print(f"omega*     = {best_joint['omega_star']:+.10e}")
        #print(f"xa*        = {best_joint['xa']:+.10f}")
        #print(f"xs*        = {best_joint['xs']:+.10f}")
        #print(f"K fixed    = {best_joint['K']:+.10e}")
        #print(f"a fixed    = {best_joint['a']}")
        #print(f"n*         = {best_joint['n']}")
        #print(f"theta_ctrl = {best_joint['theta_ctrl']}")
        #print(f"certified  = {best_joint['certified']}")
        #print(f"cert_info  = {best_joint['cert_info']}")

def plot_joint_history(history):
    good = [h for h in history if np.isfinite(h["f"])]

    if len(good) == 0:
        print("No finite history to plot.")
        return

    f = np.array([h["f"] for h in good], dtype=float)
    xa = np.array([h["xa"] for h in good], dtype=float)
    xs = np.array([h["xs"] for h in good], dtype=float)
    c = np.array([h["c"] for h in good], dtype=float)

    k = np.arange(len(good))

    plt.figure(figsize=(7, 4))
    plt.semilogy(k, f, marker="o")
    plt.xlabel("oracle call")
    plt.ylabel("objective")
    plt.title("Joint optimization objective history")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(k, xa, marker="o", label="xa")
    plt.plot(k, xs, marker="o", label="xs")
    plt.xlabel("oracle call")
    plt.ylabel("placement")
    plt.title("Placement history")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(k, c, marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("oracle call")
    plt.ylabel("constraint c")
    plt.title("Stability constraint history")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#plot_joint_history(comb_joint.history)

#########################################################################################################################
# POST-PROCESSING: BODE, POLES, ZEROS, CANCELLATION CHECKS
#########################################################################################################################

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.signal as sig
import matplotlib.pyplot as plt
from mpi4py import MPI


#########################################################################################################################
# Choose controller / placement to analyze
#########################################################################################################################



## Example: bounded-K result
#xa_star = 0.0
#xs_star = 0.0

#r = 4
#a_star = np.array([4.4, 16.3, 18.6, 7.7], dtype=float)
#n_star = np.array([12.6259559, -13.72495015, -5.26515133, 5.72344306], dtype=float)
#K_star = -5.0

#theta_ctrl_star = np.concatenate([a_star, n_star, np.array([K_star])])


#print("\n=== Design being analyzed ===")
#print("xa =", xa_star)
#print("xs =", xs_star)
#print("a  =", a_star)
#print("n  =", n_star)
#print("K  =", K_star)
#print("theta_ctrl =", theta_ctrl_star)


##########################################################################################################################
## Build closed-loop object at this placement
##########################################################################################################################

#cl_star = make_cldyn_for_placement(
    #xa=xa_star,
    #xs=xs_star,
    #A=A,
    #M=M,
    #V=V,
    #phi=phi,
    #bc=bc,
    #is_free=is_free,
    #sigma_gauss=sigma_gauss,
    #r=r,
#)

##########################################################################################################################
## Singular-value Bode plot: open-loop vs closed-loop resolvent gain
##########################################################################################################################

#def plot_resolvent_bode_open_closed(
    #A,
    #M,
    #cl_dyn,
    #theta_ctrl,
    #wmin=1e-4,
    #wmax=2e1,
    #nw=250,
#):
    #"""
    #Plots:
        #||G_open(i omega)|| and ||G_closed(i omega)||
    #for both positive and negative frequencies.

    #Uses your existing gain_of_omega and cl_dyn.gain_of_omega methods.
    #"""
    #wp = np.logspace(np.log10(wmin), np.log10(wmax), nw)

    #g_open_pos = np.empty_like(wp)
    #g_open_neg = np.empty_like(wp)
    #g_cl_pos = np.empty_like(wp)
    #g_cl_neg = np.empty_like(wp)

    #for k, w in enumerate(wp):
        #g_open_pos[k] = gain_of_omega(A, M, float(+w))
        #g_open_neg[k] = gain_of_omega(A, M, float(-w))

        #g_cl_pos[k] = cl_dyn.gain_of_omega(theta_ctrl, float(+w))
        #g_cl_neg[k] = cl_dyn.gain_of_omega(theta_ctrl, float(-w))

        #if MPI.COMM_WORLD.rank == 0 and (k % max(1, nw // 10) == 0):
            #print(f"Bode sample {k:4d}/{nw}: omega={w:.3e}")

    #if IS_WORLD_ROOT:
        #plt.figure(figsize=(8, 5))

        #plt.loglog(wp, g_open_pos, "--", label="open loop, +ω")
        #plt.loglog(wp, g_open_neg, "--", label="open loop, -ω")
        #plt.loglog(wp, g_cl_pos, label="closed loop, +ω")
        #plt.loglog(wp, g_cl_neg, label="closed loop, -ω")

        #i_best_pos = int(np.argmax(g_cl_pos))
        #i_best_neg = int(np.argmax(g_cl_neg))

        #plt.scatter([wp[i_best_pos]], [g_cl_pos[i_best_pos]], s=40)
        #plt.scatter([wp[i_best_neg]], [g_cl_neg[i_best_neg]], s=40)

        #plt.xlabel(r"$|\omega|$")
        #plt.ylabel(r"$\sigma_{\max}(G(i\omega))$")
        #plt.title("Open-loop vs closed-loop resolvent gain")
        #plt.grid(True, which="both", linestyle="--", alpha=0.4)
        #plt.legend()
        #plt.tight_layout()
        #plt.show()

        #print("\n=== Closed-loop peak over sampled grid ===")
        #print("+omega branch: omega =", +wp[i_best_pos], "gain =", g_cl_pos[i_best_pos])
        #print("-omega branch: omega =", -wp[i_best_neg], "gain =", g_cl_neg[i_best_neg])

    #return wp, g_open_pos, g_open_neg, g_cl_pos, g_cl_neg


#wp, g_open_pos, g_open_neg, g_cl_pos, g_cl_neg = plot_resolvent_bode_open_closed(
    #A=A,
    #M=M,
    #cl_dyn=cl_star,
    #theta_ctrl=theta_ctrl_star,
    #wmin=1e-4,
    #wmax=2e1,
    #nw=250,
#)

##########################################################################################################################
## Controller transfer function and Bode plot
##########################################################################################################################

#def companion_L(a):
    #"""
    #L(a) in controllable canonical form:

        #[ 0  1  0  ... 0 ]
        #[ 0  0  1  ... 0 ]
        #[ .             . ]
        #[ -a0 -a1 ... -a_{r-1} ]
    #"""
    #a = np.asarray(a, dtype=np.complex128).reshape(-1)
    #r = len(a)

    #L = np.zeros((r, r), dtype=np.complex128)

    #if r > 1:
        #for j in range(r - 1):
            #L[j, j + 1] = 1.0

    #L[-1, :] = -a
    #return L


#def controller_transfer_eval(a, n, K, omega, mc_gain=1.0):
    #"""
    #Evaluate Kc(i omega) = K + n^H (i omega I - L)^(-1) mc_gain e_r.
    #"""
    #a = np.asarray(a, dtype=np.complex128).reshape(-1)
    #n = np.asarray(n, dtype=np.complex128).reshape(-1)
    #r = len(a)

    #L = companion_L(a)
    #er = np.zeros(r, dtype=np.complex128)
    #er[-1] = 1.0

    #s = 1j * float(omega)

    #x = la.solve(s * np.eye(r, dtype=np.complex128) - L, mc_gain * er)
    #val = complex(K) + np.vdot(n, x)   # vdot conjugates n
    #return val


#def controller_zpk(a, n, K, mc_gain=1.0):
    #"""
    #Compute controller zeros, poles, gain using scipy.signal.ss2zpk.

    #State-space:
        #xdot = L x + mc_gain e_r y
        #u    = n^H x + K y
    #"""
    #a = np.asarray(a, dtype=np.complex128).reshape(-1)
    #n = np.asarray(n, dtype=np.complex128).reshape(-1)
    #r = len(a)

    #L = companion_L(a)

    #Bc = np.zeros((r, 1), dtype=np.complex128)
    #Bc[-1, 0] = mc_gain

    #Cc = np.conjugate(n).reshape(1, r)
    #Dk = np.array([[complex(K)]], dtype=np.complex128)

    #z, p, k = sig.ss2zpk(L, Bc, Cc, Dk)
    #return np.asarray(z), np.asarray(p), k


#def plot_controller_bode(a, n, K, wmin=1e-4, wmax=2e2, nw=600):
    #wp = np.logspace(np.log10(wmin), np.log10(wmax), nw)

    #H = np.array([controller_transfer_eval(a, n, K, w) for w in wp])
    #mag = np.abs(H)
    #phase = np.unwrap(np.angle(H)) * 180.0 / np.pi

    #plt.figure(figsize=(8, 5))
    #plt.loglog(wp, mag)
    #plt.xlabel(r"$\omega$")
    #plt.ylabel(r"$|K_c(i\omega)|$")
    #plt.title("Controller magnitude")
    #plt.grid(True, which="both", linestyle="--", alpha=0.4)
    #plt.tight_layout()
    #plt.show()

    #plt.figure(figsize=(8, 4))
    #plt.semilogx(wp, phase)
    #plt.xlabel(r"$\omega$")
    #plt.ylabel("phase [deg]")
    #plt.title("Controller phase")
    #plt.grid(True, which="both", linestyle="--", alpha=0.4)
    #plt.tight_layout()
    #plt.show()

    #return wp, H, mag, phase


#z_ctrl, p_ctrl, k_ctrl = controller_zpk(a_star, n_star, K_star)

#print("\n=== Controller poles and zeros ===")
#print("controller poles:")
#print(p_ctrl)
#print("controller zeros:")
#print(z_ctrl)
#print("controller gain:")
#print(k_ctrl)

#wp_ctrl, H_ctrl, mag_ctrl, phase_ctrl = plot_controller_bode(
    #a=a_star,
    #n=n_star,
    #K=K_star,
    #wmin=1e-4,
    #wmax=2e2,
    #nw=600,
#)

##########################################################################################################################
## PETSc -> SciPy conversion and pole computation
##########################################################################################################################

#def petsc_mat_to_csr(P):
    #"""
    #Convert a sequential PETSc AIJ matrix to scipy CSR.
    #This is meant for post-processing / plotting.

    #If you run with multiple MPI ranks, use one rank or adapt this to gather.
    #"""
    #P.assemble()
    #ai, aj, av = P.getValuesCSR()
    #shape = P.getSize()
    #return sp.csr_matrix((av, aj, ai), shape=shape)


#def generalized_poles_from_petsc(A_petsc, M_petsc):
    #"""
    #Compute all generalized eigenvalues:
        #A v = lambda M v

    #Uses dense scipy.linalg.eig, fine for moderate FEM sizes.
    #"""
    #A_sp = petsc_mat_to_csr(A_petsc)
    #M_sp = petsc_mat_to_csr(M_petsc)

    #A_dense = A_sp.toarray()
    #M_dense = M_sp.toarray()

    #lam = la.eig(A_dense, M_dense, right=False)

    #lam = lam[np.isfinite(lam)]
    #return lam


#def closed_loop_poles_from_cldyn(cl_dyn, theta_ctrl):
    #"""
    #Compute poles of augmented descriptor closed-loop system:
        #M_aug xdot = A_cl x
    #"""
    #Acl_petsc = cl_dyn.Acl(theta_ctrl)

    #Acl_sp = petsc_mat_to_csr(Acl_petsc)
    #M_sp = petsc_mat_to_csr(cl_dyn.M)

    #n = M_sp.shape[0]
    #r = cl_dyn.r

    #M_aug_sp = sp.block_diag(
        #(M_sp, sp.eye(r, dtype=np.complex128, format="csr")),
        #format="csr",
    #)

    #lam = la.eig(Acl_sp.toarray(), M_aug_sp.toarray(), right=False)
    #lam = lam[np.isfinite(lam)]
    #return lam


#plant_poles = generalized_poles_from_petsc(A, M)
#closed_loop_poles = closed_loop_poles_from_cldyn(cl_star, theta_ctrl_star)

#print("\n=== Dominant plant poles ===")
#for lam in plant_poles[np.argsort(np.real(plant_poles))[::-1]][:12]:
    #print(f"{lam.real:+.6e} {lam.imag:+.6e}j")

#print("\n=== Dominant closed-loop poles ===")
#for lam in closed_loop_poles[np.argsort(np.real(closed_loop_poles))[::-1]][:12]:
    #print(f"{lam.real:+.6e} {lam.imag:+.6e}j")

##########################################################################################################################
## Pole-zero map
##########################################################################################################################

#def plot_pole_zero_map(
    #plant_poles,
    #closed_loop_poles,
    #controller_poles,
    #controller_zeros,
    #real_window=(-1.0, 0.2),
    #imag_window=(-3.0, 3.0),
#):
    #plt.figure(figsize=(8, 6))

    #plt.scatter(
        #np.real(plant_poles),
        #np.imag(plant_poles),
        #marker="x",
        #s=35,
        #label="plant poles",
        #alpha=0.5,
    #)

    #plt.scatter(
        #np.real(closed_loop_poles),
        #np.imag(closed_loop_poles),
        #marker="o",
        #s=35,
        #facecolors="none",
        #label="closed-loop poles",
    #)

    #plt.scatter(
        #np.real(controller_poles),
        #np.imag(controller_poles),
        #marker="^",
        #s=80,
        #label="controller poles",
    #)

    #plt.scatter(
        #np.real(controller_zeros),
        #np.imag(controller_zeros),
        #marker="s",
        #s=80,
        #label="controller zeros",
    #)

    #plt.axvline(0.0, linestyle="--", linewidth=1.0)
    #plt.xlabel("Real part")
    #plt.ylabel("Imaginary part")
    #plt.title("Pole-zero map")
    #plt.grid(True, linestyle="--", alpha=0.4)
    #plt.legend()

    #if real_window is not None:
        #plt.xlim(real_window)
    #if imag_window is not None:
        #plt.ylim(imag_window)

    #plt.tight_layout()
    #plt.show()


#plot_pole_zero_map(
    #plant_poles=plant_poles,
    #closed_loop_poles=closed_loop_poles,
    #controller_poles=p_ctrl,
    #controller_zeros=z_ctrl,
    #real_window=(-1.0, 0.2),
    #imag_window=(-3.0, 3.0),
#)

##########################################################################################################################
## Near-cancellation reports
##########################################################################################################################

#def nearest_match_report(
    #A_points,
    #B_points,
    #A_name="A",
    #B_name="B",
    #rel_tol=1e-2,
    #abs_tol=1e-3,
    #max_print=20,
#):
    #"""
    #For every point in A_points, find nearest point in B_points.
    #Report pairs satisfying:
        #|a-b| <= abs_tol + rel_tol * max(1, |b|)
    #"""
    #A_points = np.asarray(A_points, dtype=np.complex128).reshape(-1)
    #B_points = np.asarray(B_points, dtype=np.complex128).reshape(-1)

    #rows = []

    #for a in A_points:
        #if len(B_points) == 0:
            #continue

        #d = np.abs(a - B_points)
        #j = int(np.argmin(d))
        #b = B_points[j]
        #dist = float(d[j])
        #thresh = abs_tol + rel_tol * max(1.0, abs(b))

        #if dist <= thresh:
            #rows.append((a, b, dist, thresh))

    #rows = sorted(rows, key=lambda t: t[2])

    #print("\n" + "=" * 100)
    #print(f"Near-cancellation check: {A_name} near {B_name}")
    #print(f"rel_tol={rel_tol}, abs_tol={abs_tol}")
    #print("=" * 100)

    #if len(rows) == 0:
        #print("No near matches found.")
        #return rows

    #for a, b, dist, thresh in rows[:max_print]:
        #print(
            #f"{A_name}: {a.real:+.6e} {a.imag:+.6e}j   "
            #f"{B_name}: {b.real:+.6e} {b.imag:+.6e}j   "
            #f"dist={dist:.3e}  threshold={thresh:.3e}"
        #)

    #return rows


## Focus on dynamically relevant plant poles near the imaginary axis.
#plant_poles_near = plant_poles[np.real(plant_poles) > -5.0]

## 1. Controller zeros near plant poles: possible plant/controller pole cancellation.
#cancel_z_plant = nearest_match_report(
    #A_points=z_ctrl,
    #B_points=plant_poles_near,
    #A_name="controller zero",
    #B_name="plant pole",
    #rel_tol=1e-2,
    #abs_tol=1e-3,
    #max_print=20,
#)

## 2. Controller zeros near controller poles: possible internal controller pole-zero cancellation.
#cancel_z_pctrl = nearest_match_report(
    #A_points=z_ctrl,
    #B_points=p_ctrl,
    #A_name="controller zero",
    #B_name="controller pole",
    #rel_tol=1e-2,
    #abs_tol=1e-3,
    #max_print=20,
#)

## 3. Closed-loop poles near controller poles: shows whether controller modes survive in closed loop.
#match_cl_pctrl = nearest_match_report(
    #A_points=p_ctrl,
    #B_points=closed_loop_poles,
    #A_name="controller pole",
    #B_name="closed-loop pole",
    #rel_tol=1e-2,
    #abs_tol=1e-3,
    #max_print=20,
#)

##########################################################################################################################
## Compact summaries
##########################################################################################################################

#def print_top_poles(label, poles, nshow=15):
    #poles = np.asarray(poles)
    #poles = poles[np.isfinite(poles)]
    #poles = poles[np.argsort(np.real(poles))[::-1]]

    #print("\n" + "=" * 80)
    #print(label)
    #print("=" * 80)

    #for k, lam in enumerate(poles[:nshow]):
        #print(
            #f"{k:02d}: "
            #f"lambda = {lam.real:+.8e} {lam.imag:+.8e}j, "
            #f"freq = {abs(lam.imag):.8e}, "
            #f"growth = {lam.real:+.8e}"
        #)


#print_top_poles("Dominant plant poles", plant_poles, nshow=15)
#print_top_poles("Dominant closed-loop poles", closed_loop_poles, nshow=15)

#print("\nController poles:")
#for p in p_ctrl:
    #print(f"{p.real:+.8e} {p.imag:+.8e}j")

#print("\nController zeros:")
#for z in z_ctrl:
    #print(f"{z.real:+.8e} {z.imag:+.8e}j")


##########################################################################################################################
## POST-PROCESSING TEST: UNBOUNDED-K DESIGN, K ≈ -140
##########################################################################################################################

#import numpy as np

## Unbounded-K joint result from your previous run
#xa_unbounded = -0.6771503053
#xs_unbounded = +0.6286729155

#r = 4

#a_unbounded = np.array([0.0,0.0,0.0,0.0], dtype=float)

#n_unbounded = np.array([
    #202.02634267,
   #-219.61038411,
    #-84.24696092,
     #91.57977909,
#], dtype=float)

#K_unbounded = -140.69722847

#theta_ctrl_unbounded = np.concatenate([
    #a_unbounded,
    #n_unbounded,
    #np.array([K_unbounded], dtype=float),
#])

#print("\n" + "=" * 100)
#print("UNBOUNDED-K DESIGN BEING ANALYZED")
#print("=" * 100)
#print("xa =", xa_unbounded)
#print("xs =", xs_unbounded)
#print("a  =", a_unbounded)
#print("n  =", n_unbounded)
#print("K  =", K_unbounded)
#print("theta_ctrl =", theta_ctrl_unbounded)


## Build closed-loop object at this placement
#cl_unbounded = make_cldyn_for_placement(
    #xa=xa_unbounded,
    #xs=xs_unbounded,
    #A=A,
    #M=M,
    #V=V,
    #phi=phi,
    #bc=bc,
    #is_free=is_free,
    #sigma_gauss=sigma_gauss,
    #r=r,
#)

#wp_unbounded, g_open_pos_unbounded, g_open_neg_unbounded, g_cl_pos_unbounded, g_cl_neg_unbounded = (
    #plot_resolvent_bode_open_closed(
        #A=A,
        #M=M,
        #cl_dyn=cl_unbounded,
        #theta_ctrl=theta_ctrl_unbounded,
        #wmin=1e-4,
        #wmax=1e4,
        #nw=250,
    #)
#)

#z_ctrl_unbounded, p_ctrl_unbounded, k_ctrl_unbounded = controller_zpk(
    #a_unbounded,
    #n_unbounded,
    #K_unbounded,
#)

#print("\n=== Unbounded-K controller poles and zeros ===")
#print("controller poles:")
#print(p_ctrl_unbounded)
#print("controller zeros:")
#print(z_ctrl_unbounded)
#print("controller gain:")
#print(k_ctrl_unbounded)

#wp_ctrl_unbounded, H_ctrl_unbounded, mag_ctrl_unbounded, phase_ctrl_unbounded = plot_controller_bode(
    #a=a_unbounded,
    #n=n_unbounded,
    #K=K_unbounded,
    #wmin=1e-4,
    #wmax=1e4,
    #nw=600,
#)

#closed_loop_poles_unbounded = closed_loop_poles_from_cldyn(
    #cl_unbounded,
    #theta_ctrl_unbounded,
#)

#print_top_poles("Dominant plant poles", plant_poles, nshow=15)
#print_top_poles("Dominant closed-loop poles, unbounded K", closed_loop_poles_unbounded, nshow=15)

#print("\nUnbounded-K controller poles:")
#for p in p_ctrl_unbounded:
    #print(f"{p.real:+.8e} {p.imag:+.8e}j")

#print("\nUnbounded-K controller zeros:")
#for z in z_ctrl_unbounded:
    #print(f"{z.real:+.8e} {z.imag:+.8e}j")

#plot_pole_zero_map(
    #plant_poles=plant_poles,
    #closed_loop_poles=closed_loop_poles_unbounded,
    #controller_poles=p_ctrl_unbounded,
    #controller_zeros=z_ctrl_unbounded,
    #real_window=(-5.0, 0.5),
    #imag_window=(-4.0, 4.0),
#)

#plant_poles_near = plant_poles[np.real(plant_poles) > -5.0]

#cancel_z_plant_unbounded = nearest_match_report(
    #A_points=z_ctrl_unbounded,
    #B_points=plant_poles_near,
    #A_name="controller zero",
    #B_name="plant pole",
    #rel_tol=1e-2,
    #abs_tol=1e-3,
    #max_print=20,
#)

#cancel_z_pctrl_unbounded = nearest_match_report(
    #A_points=z_ctrl_unbounded,
    #B_points=p_ctrl_unbounded,
    #A_name="controller zero",
    #B_name="controller pole",
    #rel_tol=1e-2,
    #abs_tol=1e-3,
    #max_print=20,
#)

#match_cl_pctrl_unbounded = nearest_match_report(
    #A_points=p_ctrl_unbounded,
    #B_points=closed_loop_poles_unbounded,
    #A_name="controller pole",
    #B_name="closed-loop pole",
    #rel_tol=1e-2,
    #abs_tol=1e-3,
    #max_print=20,
#)

##########################################################################################################################
## Optional comparison: bounded K vs unbounded K
##########################################################################################################################

## Only run this if you still have the bounded arrays from the previous test:
## wp, g_cl_pos, g_cl_neg correspond to bounded-K result.

#try:
    #import matplotlib.pyplot as plt

    #plt.figure(figsize=(8, 5))

    #plt.loglog(wp, g_cl_pos, label="bounded K, +ω")
    #plt.loglog(wp, g_cl_neg, label="bounded K, -ω")

    #plt.loglog(wp_unbounded, g_cl_pos_unbounded, "--", label="unbounded K, +ω")
    #plt.loglog(wp_unbounded, g_cl_neg_unbounded, "--", label="unbounded K, -ω")

    #plt.xlabel(r"$|\omega|$")
    #plt.ylabel(r"$\sigma_{\max}(G(i\omega))$")
    #plt.title("Closed-loop resolvent gain: bounded vs unbounded K")
    #plt.grid(True, which="both", linestyle="--", alpha=0.4)
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

#except NameError:
    #print("Bounded-K arrays not found. Skipping overlay.")

#def summarize_design(label, poles_cl, z_ctrl, p_ctrl, wp, g_pos, g_neg, K, n):
    #i_pos = int(np.argmax(g_pos))
    #i_neg = int(np.argmax(g_neg))

    #if g_pos[i_pos] >= g_neg[i_neg]:
        #peak_w = +wp[i_pos]
        #peak_g = g_pos[i_pos]
    #else:
        #peak_w = -wp[i_neg]
        #peak_g = g_neg[i_neg]

    #alpha_cl = np.max(np.real(poles_cl))

    #print("\n" + "=" * 100)
    #print(label)
    #print("=" * 100)
    #print(f"sampled peak gain = {peak_g:.8e}")
    #print(f"sampled peak omega = {peak_w:+.8e}")
    #print(f"closed-loop alpha = {alpha_cl:+.8e}")
    #print(f"K = {K:+.8e}")
    #print(f"||n||_2 = {np.linalg.norm(n):.8e}")
    #print(f"controller poles = {p_ctrl}")
    #print(f"controller zeros = {z_ctrl}")


#summarize_design(
    #label="UNBOUNDED-K DESIGN SUMMARY",
    #poles_cl=closed_loop_poles_unbounded,
    #z_ctrl=z_ctrl_unbounded,
    #p_ctrl=p_ctrl_unbounded,
    #wp=wp_unbounded,
    #g_pos=g_cl_pos_unbounded,
    #g_neg=g_cl_neg_unbounded,
    #K=K_unbounded,
    #n=n_unbounded,
#)

"""
Joint sensor/actuator placement + controller-parameter sweep for yo.py.

Append this block near the end of yo.py, after these already exist:
  - build_actuator_vector, build_sensor_state_vector, sensor_output_vector
  - ClosedLoopSOFF
  - ClosedLoopDynamicCompanionSISO
  - gain_of_omega, rightmost_eig_realpart or SLEPc imports
  - pygranso, pygransoStruct, torch, PETSc, MPI, matplotlib

It runs four cases for r = 4:
  1. K = 0, a = 0, n = 0; optimize placement only.
  2. K free, a = 0, n = 0; optimize placement + static K.
  3. K = 0, a and n free; optimize placement + dynamic parameters.
  4. K free, a and n free; optimize placement + dynamic parameters.

Important modeling note:
  When a = n = 0, the dynamic controller states are disconnected/marginal.
  Therefore cases 1-2 are evaluated with the static SISO closed-loop class
  ClosedLoopSOFF, not the augmented dynamic class. This avoids artificial zero
  controller poles contaminating the stability constraint.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from mpi4py import MPI


# -----------------------------------------------------------------------------
# Smooth bounded maps for placement and optionally K.
# -----------------------------------------------------------------------------

def _tanh_to_interval(eta, lo, hi):
    eta = float(eta)
    lo = float(lo)
    hi = float(hi)
    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    t = np.tanh(eta)
    return mid + rad * t, rad * (1.0 - t * t)


def _interval_to_tanh(x, lo, hi, eps=1e-12):
    x = float(x)
    lo = float(lo)
    hi = float(hi)
    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    y = (x - mid) / rad
    y = np.clip(y, -1.0 + eps, 1.0 - eps)
    return float(np.arctanh(y))


def _clip_step(x, h, bounds):
    lo, hi = bounds
    xp = min(float(x) + float(h), float(hi))
    xm = max(float(x) - float(h), float(lo))
    return xp, xm


# -----------------------------------------------------------------------------
# Build static closed-loop at one placement.
# Uses sign +1 to match ClosedLoopDynamicCompanionSISO, where Acl = A + b K c^H.
# -----------------------------------------------------------------------------

def make_static_cl_for_placement(
    xa,
    xs,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    feedback_sign=+1.0,
):
    b_red, _ = build_actuator_vector(xa, sigma_gauss, V, phi, bc, is_free)
    s_red, _ = build_sensor_state_vector(xs, sigma_gauss, V, is_free)
    c_red = sensor_output_vector(M, s_red)
    return ClosedLoopSOFF(A, M, b_red, c_red, feedback_sign=feedback_sign)


# -----------------------------------------------------------------------------
# Hard-gated oracles.
# The key change is: compute spectral abscissa first. If c = alpha+margin > 0,
# skip Hinf/certification and return a large stability objective. This prevents
# the optimizer from following unstable points into bad local minima.
# -----------------------------------------------------------------------------

def static_oracle_hard_stability(
    cl,
    K,
    stab_margin=1e-5,
    stab_buffer=5e-5,
    stab_penalty_weight=1e8,
    unstable_objective=1e6,
    num_guess=6,
    wmin=1e-6,
    wmax=2e1,
    seed=2,
    disk_b=20.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
):
    alpha, dcdK = cl.spectral_abscissa_and_grad(float(K))
    c_val = float(alpha + stab_margin)

    # Hard gate: unstable trial point. Do not compute Hinf.
    if c_val > 0.0:
        f_return = unstable_objective + stab_penalty_weight * c_val**2
        df_return_dK = 2.0 * stab_penalty_weight * c_val * dcdK
        return {
            "f_raw": np.inf,
            "f_return": float(f_return),
            "df_ctrl": np.array([df_return_dK], dtype=float),
            "c": c_val,
            "dc_ctrl": np.array([dcdK], dtype=float),
            "alpha": float(alpha),
            "omega_star": np.nan,
            "certified": False,
            "cert_info": {"reason": "hard_stability_gate"},
            "skipped_hinf": True,
        }

    h = cl.hinf_value_grad_newton_disk(
        float(K),
        num_guess=num_guess,
        wmin=wmin,
        wmax=wmax,
        seed=seed,
        disk_b=disk_b,
        disk_eigs_tol=disk_eigs_tol,
        disk_nev=disk_nev,
        disk_eps_probe=disk_eps_probe,
        disk_axis_warn=disk_axis_warn,
    )

    f_raw = float(h["gamma_star"])
    df_raw_dK = float(h["dgamma_dk"])

    # Soft buffer on top of the hard constraint to keep iterates away from alpha=0.
    c_buf = max(c_val + stab_buffer, 0.0)
    f_return = f_raw + stab_penalty_weight * c_buf**2
    df_return_dK = df_raw_dK
    if c_buf > 0.0:
        df_return_dK += 2.0 * stab_penalty_weight * c_buf * dcdK

    _progress_print(f"[oracle] Hinf oracle done: f_raw={f_raw:.6e}, omega*={float(h['omega_star']):+.6e}")
    return {
        "f_raw": f_raw,
        "f_return": float(f_return),
        "df_ctrl": np.array([df_return_dK], dtype=float),
        "c": c_val,
        "dc_ctrl": np.array([dcdK], dtype=float),
        "alpha": float(alpha),
        "omega_star": float(h["omega_star"]),
        "certified": bool(h["certified"]),
        "cert_info": h["cert_info"],
        "skipped_hinf": False,
    }


def dynamic_oracle_hard_stability(
    cl_dyn,
    theta,
    stab_margin=1e-5,
    stab_buffer=5e-5,
    stab_penalty_weight=1e8,
    unstable_objective=1e6,
    num_guess=6,
    wmin=1e-6,
    wmax=2e1,
    seed=2,
    candidate_npos=60,
    disk_b=20.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
):
    theta = np.asarray(theta, dtype=float).reshape(-1)

    _progress_print("[oracle] spectral-abscissa check start")
    alpha, dc_dtheta = cl_dyn.spectral_abscissa_and_grad(theta)
    dc_dtheta = np.asarray(dc_dtheta, dtype=float).reshape(-1)
    c_val = float(alpha + stab_margin)
    _progress_print(f"[oracle] alpha={float(alpha):+.6e}, c={c_val:+.6e}")

    # Hard gate: unstable trial point. Do not compute Hinf.
    if c_val > 0.0:
        f_return = unstable_objective + stab_penalty_weight * c_val**2
        df_return = 2.0 * stab_penalty_weight * c_val * dc_dtheta
        return {
            "f_raw": np.inf,
            "f_return": float(f_return),
            "df_ctrl": df_return,
            "c": c_val,
            "dc_ctrl": dc_dtheta,
            "alpha": float(alpha),
            "omega_star": np.nan,
            "certified": False,
            "cert_info": {"reason": "hard_stability_gate"},
            "skipped_hinf": True,
        }

    _progress_print("[oracle] stable enough; computing Hinf oracle")
    h = cl_dyn.hinf_value_grad_newton_disk(
        theta,
        num_guess=num_guess,
        wmin=wmin,
        wmax=wmax,
        seed=seed,
        candidate_npos=candidate_npos,
        disk_b=disk_b,
        disk_eigs_tol=disk_eigs_tol,
        disk_nev=disk_nev,
        disk_eps_probe=disk_eps_probe,
        disk_axis_warn=disk_axis_warn,
    )

    f_raw = float(h["gamma_star"])
    df_raw = np.asarray(h["grad"], dtype=float).reshape(-1)

    # Soft buffer on top of the hard constraint.
    c_buf = max(c_val + stab_buffer, 0.0)
    f_return = f_raw + stab_penalty_weight * c_buf**2
    df_return = df_raw.copy()
    if c_buf > 0.0:
        df_return += 2.0 * stab_penalty_weight * c_buf * dc_dtheta

    _progress_print(f"[oracle] Hinf oracle done: f_raw={f_raw:.6e}, omega*={float(h['omega_star']):+.6e}")
    return {
        "f_raw": f_raw,
        "f_return": float(f_return),
        "df_ctrl": df_return,
        "c": c_val,
        "dc_ctrl": dc_dtheta,
        "alpha": float(alpha),
        "omega_star": float(h["omega_star"]),
        "certified": bool(h["certified"]),
        "cert_info": h["cert_info"],
        "skipped_hinf": False,
    }


# -----------------------------------------------------------------------------
# Case configuration and parameter unpacking.
# -----------------------------------------------------------------------------

def make_param_sweep_cases(r, a_free_center, n_free_center=None, K_init=0.0):
    if n_free_center is None:
        n_free_center = np.zeros(r, dtype=float)

    a_free_center = np.asarray(a_free_center, dtype=np.complex128).reshape(r)
    n_free_center = np.asarray(n_free_center, dtype=np.complex128).reshape(r)

    return [
        {
            "name": "P_only__K0_a0_n0",
            "mode": "static",
            "free_a": False,
            "free_n": False,
            "free_K": False,
            "a_center": np.zeros(r, dtype=float),
            "n_center": np.zeros(r, dtype=float),
            "K_center": 0.0,
        },
        {
            "name": "P_plus_K__Kfree_a0_n0",
            "mode": "static",
            "free_a": False,
            "free_n": False,
            "free_K": True,
            "a_center": np.zeros(r, dtype=float),
            "n_center": np.zeros(r, dtype=float),
            "K_center": float(K_init),
        },
        {
            "name": "P_plus_dyn__K0_afree_nfree",
            "mode": "dynamic",
            "free_a": True,
            "free_n": True,
            "free_K": False,
            "a_center": a_free_center.copy(),
            "n_center": n_free_center.copy(),
            "K_center": 0.0,
        },
        {
            "name": "P_plus_dyn_plus_K__Kfree_afree_nfree",
            "mode": "dynamic",
            "free_a": True,
            "free_n": True,
            "free_K": True,
            "a_center": a_free_center.copy(),
            "n_center": n_free_center.copy(),
            "K_center": float(K_init),
        },
    ]


def _case_var_slices(case, r):
    # z = [eta_xa, eta_xs, optional Re/Im(a), optional Re/Im(n), optional eta_K]
    # PyGRANSO variables are real; complex controller coefficients use paired
    # real/imaginary coordinates.
    idx = 2
    out = {
        "xa": 0, "xs": 1,
        "a_re": None, "a_im": None,
        "n_re": None, "n_im": None,
        "K": None,
    }
    if case["free_a"]:
        out["a_re"] = slice(idx, idx + r); idx += r
        out["a_im"] = slice(idx, idx + r); idx += r
    if case["free_n"]:
        out["n_re"] = slice(idx, idx + r); idx += r
        out["n_im"] = slice(idx, idx + r); idx += r
    if case["free_K"]:
        out["K"] = idx
        idx += 1
    out["nvar"] = idx
    return out


def _unpack_case_z(z, case, r, xa_bounds, xs_bounds, K_bounds, a_scale, n_scale):
    z = np.asarray(z, dtype=float).reshape(-1)
    sl = _case_var_slices(case, r)

    xa, dxa_deta = _tanh_to_interval(z[sl["xa"]], xa_bounds[0], xa_bounds[1])
    xs, dxs_deta = _tanh_to_interval(z[sl["xs"]], xs_bounds[0], xs_bounds[1])

    a = np.asarray(case["a_center"], dtype=np.complex128).reshape(r).copy()
    nvec = np.asarray(case["n_center"], dtype=np.complex128).reshape(r).copy()
    K = float(case["K_center"])

    nvar = sl["nvar"]
    da_re_dz = np.zeros((r, nvar), dtype=float)
    da_im_dz = np.zeros((r, nvar), dtype=float)
    dn_re_dz = np.zeros((r, nvar), dtype=float)
    dn_im_dz = np.zeros((r, nvar), dtype=float)
    dK_dz = np.zeros(nvar, dtype=float)

    if case["free_a"]:
        a_scale_arr = np.asarray(a_scale, dtype=float)
        if a_scale_arr.size == 1:
            a_scale_arr = float(a_scale_arr) * np.ones(r, dtype=float)
        a = a + a_scale_arr * (z[sl["a_re"]] + 1j * z[sl["a_im"]])
        for j in range(r):
            da_re_dz[j, sl["a_re"].start + j] = a_scale_arr[j]
            da_im_dz[j, sl["a_im"].start + j] = a_scale_arr[j]

    if case["free_n"]:
        n_scale_arr = np.asarray(n_scale, dtype=float)
        if n_scale_arr.size == 1:
            n_scale_arr = float(n_scale_arr) * np.ones(r, dtype=float)
        nvec = nvec + n_scale_arr * (z[sl["n_re"]] + 1j * z[sl["n_im"]])
        for j in range(r):
            dn_re_dz[j, sl["n_re"].start + j] = n_scale_arr[j]
            dn_im_dz[j, sl["n_im"].start + j] = n_scale_arr[j]

    if case["free_K"]:
        K, dK_deta = _tanh_to_interval(z[sl["K"]], K_bounds[0], K_bounds[1])
        dK_dz[sl["K"]] = dK_deta

    return {
        "xa": float(xa),
        "xs": float(xs),
        "a": a,
        "n": nvec,
        "K": float(K),
        "dxa_dz": {sl["xa"]: dxa_deta},
        "dxs_dz": {sl["xs"]: dxs_deta},
        "da_re_dz": da_re_dz,
        "da_im_dz": da_im_dz,
        "dn_re_dz": dn_re_dz,
        "dn_im_dz": dn_im_dz,
        # Backward-compatible aliases for code that only reads real directions.
        "da_dz": da_re_dz,
        "dn_dz": dn_re_dz,
        "dK_dz": dK_dz,
        "nvar": nvar,
    }


def _initial_z_for_case(case, r, xa_init, xs_init, xa_bounds, xs_bounds, K_bounds):
    sl = _case_var_slices(case, r)
    z0 = np.zeros(sl["nvar"], dtype=float)
    z0[sl["xa"]] = _interval_to_tanh(xa_init, xa_bounds[0], xa_bounds[1])
    z0[sl["xs"]] = _interval_to_tanh(xs_init, xs_bounds[0], xs_bounds[1])
    if case["free_K"]:
        K0 = np.clip(float(case["K_center"]), K_bounds[0] + 1e-12, K_bounds[1] - 1e-12)
        z0[sl["K"]] = _interval_to_tanh(K0, K_bounds[0], K_bounds[1])
    return z0


# -----------------------------------------------------------------------------
# Main combined function for one case.
# Placement gradients are finite differences. Controller gradients are analytic
# through your existing static/dynamic oracles.
# -----------------------------------------------------------------------------

def make_pygranso_combined_fn_joint_case(
    case,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    r,
    xa_bounds,
    xs_bounds,
    K_bounds=(-5.0, 5.0),
    a_scale=0.25,
    n_scale=2.5e-2,
    placement_fd=2.5e-3,
    stab_margin=1e-5,
    stab_buffer=5e-5,
    stab_penalty_weight=1e8,
    unstable_objective=1e6,
    num_guess=6,
    wmin=1e-6,
    wmax=2e1,
    base_seed=2,
    candidate_npos=60,
    disk_b=20.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
    verbose=True,
):
    history = []
    state = {"calls": 0}
    cache = {}

    sl = _case_var_slices(case, r)
    nvar = sl["nvar"]

    def cache_key(xa, xs, a, nvec, K):
        # Rounding prevents tiny FD noise from defeating the cache.
        return (
            round(float(xa), 12),
            round(float(xs), 12),
            tuple(np.round(np.r_[np.real(a), np.imag(a)], 12)),
            tuple(np.round(np.r_[np.real(nvec), np.imag(nvec)], 12)),
            round(float(K), 12),
        )

    def eval_physical(xa, xs, a, nvec, K, seed):
        key = cache_key(xa, xs, a, nvec, K)
        if key in cache:
            return cache[key]

        if case["mode"] == "static":
            cl = make_static_cl_for_placement(
                xa=xa,
                xs=xs,
                A=A,
                M=M,
                V=V,
                phi=phi,
                bc=bc,
                is_free=is_free,
                sigma_gauss=sigma_gauss,
                feedback_sign=+1.0,
            )
            out = static_oracle_hard_stability(
                cl,
                K=K,
                stab_margin=stab_margin,
                stab_buffer=stab_buffer,
                stab_penalty_weight=stab_penalty_weight,
                unstable_objective=unstable_objective,
                num_guess=num_guess,
                wmin=wmin,
                wmax=wmax,
                seed=seed,
                disk_b=disk_b,
                disk_eigs_tol=disk_eigs_tol,
                disk_nev=disk_nev,
                disk_eps_probe=disk_eps_probe,
                disk_axis_warn=disk_axis_warn,
            )
        else:
            cl_dyn = make_cldyn_for_placement(
                xa=xa,
                xs=xs,
                A=A,
                M=M,
                V=V,
                phi=phi,
                bc=bc,
                is_free=is_free,
                sigma_gauss=sigma_gauss,
                r=r,
            )
            theta = cl_dyn.pack_theta(a, nvec, Kstat=K)
            out = dynamic_oracle_hard_stability(
                cl_dyn,
                theta=theta,
                stab_margin=stab_margin,
                stab_buffer=stab_buffer,
                stab_penalty_weight=stab_penalty_weight,
                unstable_objective=unstable_objective,
                num_guess=num_guess,
                wmin=wmin,
                wmax=wmax,
                seed=seed,
                candidate_npos=candidate_npos,
                disk_b=disk_b,
                disk_eigs_tol=disk_eigs_tol,
                disk_nev=disk_nev,
                disk_eps_probe=disk_eps_probe,
                disk_axis_warn=disk_axis_warn,
            )

        cache[key] = out
        return out

    def fd_placement_grad(xa, xs, a, nvec, K, seed):
        hxa = float(placement_fd) * max(1.0, abs(float(xa)))
        hxs = float(placement_fd) * max(1.0, abs(float(xs)))

        xa_p, xa_m = _clip_step(xa, hxa, xa_bounds)
        xs_p, xs_m = _clip_step(xs, hxs, xs_bounds)

        if xa_p == xa_m:
            dfd_xa = 0.0
            dcd_xa = 0.0
        else:
            op = eval_physical(xa_p, xs, a, nvec, K, seed)
            om = eval_physical(xa_m, xs, a, nvec, K, seed)
            dfd_xa = (float(op["f_return"]) - float(om["f_return"])) / (xa_p - xa_m)
            dcd_xa = (float(op["c"]) - float(om["c"])) / (xa_p - xa_m)

        if xs_p == xs_m:
            dfd_xs = 0.0
            dcd_xs = 0.0
        else:
            op = eval_physical(xa, xs_p, a, nvec, K, seed)
            om = eval_physical(xa, xs_m, a, nvec, K, seed)
            dfd_xs = (float(op["f_return"]) - float(om["f_return"])) / (xs_p - xs_m)
            dcd_xs = (float(op["c"]) - float(om["c"])) / (xs_p - xs_m)

        return float(dfd_xa), float(dfd_xs), float(dcd_xa), float(dcd_xs)

    def combined_fn(X_struct):
        state["calls"] += 1
        z_torch = X_struct.z
        z = z_torch.detach().cpu().numpy().reshape(-1)
        if z.size != nvar:
            raise ValueError(f"Expected z length {nvar}, got {z.size}")

        p = _unpack_case_z(
            z,
            case=case,
            r=r,
            xa_bounds=xa_bounds,
            xs_bounds=xs_bounds,
            K_bounds=K_bounds,
            a_scale=a_scale,
            n_scale=n_scale,
        )

        seed_this = base_seed  # fixed seed gives less noisy finite differences
        out0 = eval_physical(p["xa"], p["xs"], p["a"], p["n"], p["K"], seed_this)

        f_return = float(out0["f_return"])
        c_val = float(out0["c"])

        df_dz = np.zeros(nvar, dtype=float)
        dc_dz = np.zeros(nvar, dtype=float)

        dfd_xa, dfd_xs, dcd_xa, dcd_xs = fd_placement_grad(
            p["xa"], p["xs"], p["a"], p["n"], p["K"], seed_this
        )

        df_dz[0] = dfd_xa * p["dxa_dz"][0]
        df_dz[1] = dfd_xs * p["dxs_dz"][1]
        dc_dz[0] = dcd_xa * p["dxa_dz"][0]
        dc_dz[1] = dcd_xs * p["dxs_dz"][1]

        if case["mode"] == "static":
            # Static ctrl gradient is [d/dK].
            df_ctrl = np.asarray(out0["df_ctrl"], dtype=float).reshape(1)
            dc_ctrl = np.asarray(out0["dc_ctrl"], dtype=float).reshape(1)
            df_dz += df_ctrl[0] * p["dK_dz"]
            dc_dz += dc_ctrl[0] * p["dK_dz"]
        else:
            # Dynamic ctrl gradient is
            # [d/dRe(a), d/dIm(a), d/dRe(n), d/dIm(n), d/dK].
            df_ctrl = np.asarray(out0["df_ctrl"], dtype=float).reshape(4 * r + 1)
            dc_ctrl = np.asarray(out0["dc_ctrl"], dtype=float).reshape(4 * r + 1)

            gfa_re = df_ctrl[0:r]
            gfa_im = df_ctrl[r:2*r]
            gfn_re = df_ctrl[2*r:3*r]
            gfn_im = df_ctrl[3*r:4*r]
            gfK = df_ctrl[-1]

            gca_re = dc_ctrl[0:r]
            gca_im = dc_ctrl[r:2*r]
            gcn_re = dc_ctrl[2*r:3*r]
            gcn_im = dc_ctrl[3*r:4*r]
            gcK = dc_ctrl[-1]

            df_dz += p["da_re_dz"].T @ gfa_re
            df_dz += p["da_im_dz"].T @ gfa_im
            df_dz += p["dn_re_dz"].T @ gfn_re
            df_dz += p["dn_im_dz"].T @ gfn_im
            df_dz += gfK * p["dK_dz"]

            dc_dz += p["da_re_dz"].T @ gca_re
            dc_dz += p["da_im_dz"].T @ gca_im
            dc_dz += p["dn_re_dz"].T @ gcn_re
            dc_dz += p["dn_im_dz"].T @ gcn_im
            dc_dz += gcK * p["dK_dz"]

        rec = {
            "case": case["name"],
            "call": state["calls"],
            "z": z.copy(),
            "xa": p["xa"],
            "xs": p["xs"],
            "a": p["a"].copy(),
            "n": p["n"].copy(),
            "K": p["K"],
            "f": float(out0["f_raw"]),
            "f_raw": float(out0["f_raw"]),
            "f_return": f_return,
            "alpha": float(out0["alpha"]),
            "c": c_val,
            "omega_star": float(out0["omega_star"]) if np.isfinite(out0["omega_star"]) else np.nan,
            "certified": bool(out0["certified"]),
            "cert_info": out0["cert_info"],
            "skipped_hinf": bool(out0["skipped_hinf"]),
        }
        history.append(rec)

        if verbose and IS_WORLD_ROOT:
            print(
                f"[{case['name']}] call={state['calls']:03d}  "
                f"xa={p['xa']:+.6f} xs={p['xs']:+.6f}  "
                f"f_raw={out0['f_raw']:.6e} f_ret={f_return:.6e}  "
                f"alpha={out0['alpha']:+.6e} c={c_val:+.6e}  "
                f"K={p['K']:+.6e} skipped={out0['skipped_hinf']} cert={out0['certified']}"
            )

        dev = z_torch.device
        dt = z_torch.dtype
        f_grad = torch.tensor(df_dz.reshape(-1, 1), device=dev, dtype=dt)
        ci = torch.tensor([[c_val]], device=dev, dtype=dt)
        ci_grad = torch.tensor(dc_dz.reshape(-1, 1), device=dev, dtype=dt)
        return [f_return, f_grad, ci, ci_grad, None, None]

    combined_fn.history = history
    combined_fn.case = case
    combined_fn.nvar = nvar
    return combined_fn


def best_feasible_joint_history(history):
    feasible = [h for h in history if h["c"] <= 0.0 and np.isfinite(h["f_raw"])]
    if len(feasible) == 0:
        return None
    return min(feasible, key=lambda h: h["f_raw"])


# -----------------------------------------------------------------------------
# Run one case / run all four cases.
# -----------------------------------------------------------------------------

def run_joint_placement_case(
    case,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    r,
    xa_init,
    xs_init,
    xa_bounds,
    xs_bounds,
    K_bounds=(-5.0, 5.0),
    a_scale=0.25,
    n_scale=2.5e-2,
    placement_fd=2.5e-3,
    maxit=25,
    base_seed=2,
    print_level=1,
    verbose=True,
    **oracle_kwargs,
):
    device = oracle_kwargs.pop("device", torch.device("cpu"))
    dtype = oracle_kwargs.pop("dtype", torch.double)

    z0 = _initial_z_for_case(case, r, xa_init, xs_init, xa_bounds, xs_bounds, K_bounds)

    comb_fn = make_pygranso_combined_fn_joint_case(
        case=case,
        A=A,
        M=M,
        V=V,
        phi=phi,
        bc=bc,
        is_free=is_free,
        sigma_gauss=sigma_gauss,
        r=r,
        xa_bounds=xa_bounds,
        xs_bounds=xs_bounds,
        K_bounds=K_bounds,
        a_scale=a_scale,
        n_scale=n_scale,
        placement_fd=placement_fd,
        base_seed=base_seed,
        verbose=verbose,
        **oracle_kwargs,
    )

    opts = pygransoStruct()
    opts.torch_device = device
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = maxit
    opts.print_frequency = 1
    opts.print_level = (int(print_level) if IS_WORLD_ROOT else 0)
    opts.quadprog_info_msg = False
    opts.x0 = torch.tensor(z0.reshape(-1, 1), device=device, dtype=dtype)

    var_spec = {"z": [comb_fn.nvar, 1]}
    t0 = time.time()
    try:
        pygranso(var_spec=var_spec, combined_fn=comb_fn, user_opts=opts)
    except Exception as e:
        if IS_WORLD_ROOT:
            print(f"[{case['name']}] PyGRANSO stopped/failed: {repr(e)}")

    best = best_feasible_joint_history(comb_fn.history)
    out = {
        "case": case,
        "case_name": case["name"],
        "best": best,
        "history": comb_fn.history,
        "elapsed_sec": time.time() - t0,
    }

    if IS_WORLD_ROOT:
        print("\n" + "=" * 88)
        print(f"CASE DONE: {case['name']}")
        print(f"elapsed = {out['elapsed_sec']:.1f} s")
        if best is None:
            print("No feasible iterate found.")
        else:
            print(f"best f       = {best['f_raw']:.10e}")
            print(f"best alpha   = {best['alpha']:+.10e}")
            print(f"best c       = {best['c']:+.10e}")
            print(f"best xa/xs   = {best['xa']:+.10f}, {best['xs']:+.10f}")
            print(f"best K       = {best['K']:+.10e}")
            print(f"best a       = {best['a']}")
            print(f"best n       = {best['n']}")
        print("=" * 88 + "\n")

    return out


def run_all_four_parametrizations(
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    x_min,
    x_max,
    r=4,
    xa_init=1.03,
    xs_init=-0.98,
    placement_radius=0.35,
    K_init=0.0,
    K_bounds=(-5.0, 5.0),
    a_free_center=None,
    n_free_center=None,
    a_scale=0.25,
    n_scale=2.5e-2,
    maxit=25,
    base_seed=2,
    print_level=1,
    verbose=True,
    **oracle_kwargs,
):
    if a_free_center is None:
        # Stable warm start for free-a cases. Starting at a=0 creates marginal
        # controller poles in the augmented model, so it is a poor initial point.
        a_free_center = np.array([4.4, 16.3, 18.6, 7.7], dtype=float)
    if n_free_center is None:
        n_free_center = np.zeros(r, dtype=float)

    xa_bounds = (
        max(float(x_min) + 3.0 * float(sigma_gauss), float(xa_init) - float(placement_radius)),
        min(float(x_max) - 3.0 * float(sigma_gauss), float(xa_init) + float(placement_radius)),
    )
    xs_bounds = (
        max(float(x_min) + 3.0 * float(sigma_gauss), float(xs_init) - float(placement_radius)),
        min(float(x_max) - 3.0 * float(sigma_gauss), float(xs_init) + float(placement_radius)),
    )

    cases = make_param_sweep_cases(
        r=r,
        a_free_center=a_free_center,
        n_free_center=n_free_center,
        K_init=K_init,
    )

    runs = []
    for k, case in enumerate(cases):
        if IS_WORLD_ROOT:
            print("\n" + "#" * 100)
            print(f"RUNNING CASE {k+1}/4: {case['name']}")
            print("#" * 100)

        run = run_joint_placement_case(
            case=case,
            A=A,
            M=M,
            V=V,
            phi=phi,
            bc=bc,
            is_free=is_free,
            sigma_gauss=sigma_gauss,
            r=r,
            xa_init=xa_init,
            xs_init=xs_init,
            xa_bounds=xa_bounds,
            xs_bounds=xs_bounds,
            K_bounds=K_bounds,
            a_scale=a_scale,
            n_scale=n_scale,
            maxit=maxit,
            base_seed=base_seed + 100 * k,
            print_level=print_level,
            verbose=verbose,
            **oracle_kwargs,
        )
        runs.append(run)

    return runs


# -----------------------------------------------------------------------------
# Plotting helpers.
# -----------------------------------------------------------------------------

def plot_joint_param_sweep_histories(runs):
    if not IS_WORLD_ROOT:
        return

    plt.figure(figsize=(8, 5))
    for run in runs:
        hist = [h for h in run["history"] if np.isfinite(h["f_return"])]
        if not hist:
            continue
        y = np.array([h["f_return"] for h in hist], dtype=float)
        plt.semilogy(np.arange(len(y)), y, marker="o", label=run["case_name"])
    plt.xlabel("oracle call")
    plt.ylabel("objective returned to PyGRANSO")
    plt.title("Objective history with stability gate")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for run in runs:
        hist = run["history"]
        if not hist:
            continue
        c = np.array([h["c"] for h in hist], dtype=float)
        plt.plot(np.arange(len(c)), c, marker="o", label=run["case_name"])
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("oracle call")
    plt.ylabel("stability constraint c = alpha + margin")
    plt.title("Stability constraint history")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    for run in runs:
        hist = [h for h in run["history"] if np.isfinite(h["f_raw"])]
        if not hist:
            continue
        xa = np.array([h["xa"] for h in hist], dtype=float)
        xs = np.array([h["xs"] for h in hist], dtype=float)
        plt.plot(xa, xs, marker="o", label=run["case_name"])
    plt.xlabel("actuator position xa")
    plt.ylabel("sensor position xs")
    plt.title("Placement trajectories")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    names = []
    vals = []
    for run in runs:
        names.append(run["case_name"].replace("__", "\n"))
        vals.append(np.nan if run["best"] is None else run["best"]["f_raw"])
    vals = np.array(vals, dtype=float)

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), names, rotation=20, ha="right", fontsize=8)
    plt.ylabel("best feasible Hinf")
    plt.title("Best feasible objective by parametrization")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def controller_transfer_response(a, nvec, K, ws):
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    nvec = np.asarray(nvec, dtype=np.complex128).reshape(-1)
    r = len(a)
    if r == 0 or np.allclose(nvec, 0.0):
        return np.full_like(np.asarray(ws, dtype=float), fill_value=complex(K), dtype=np.complex128)

    L = np.zeros((r, r), dtype=np.complex128)
    for i in range(r - 1):
        L[i, i + 1] = 1.0
    L[-1, :] = -a
    er = np.zeros(r, dtype=np.complex128)
    er[-1] = 1.0

    vals = []
    I = np.eye(r, dtype=np.complex128)
    for w in ws:
        x = np.linalg.solve(1j * float(w) * I - L, er)
        vals.append(complex(K) + np.vdot(nvec, x))
    return np.asarray(vals, dtype=np.complex128)


def plot_controller_laws(runs, wmin=1e-4, wmax=2e1, nw=200):
    if not IS_WORLD_ROOT:
        return

    ws = np.logspace(np.log10(wmin), np.log10(wmax), nw)

    plt.figure(figsize=(8, 5))
    for run in runs:
        best = run["best"]
        if best is None:
            continue
        Hk = controller_transfer_response(best["a"], best["n"], best["K"], ws)
        plt.loglog(ws, np.abs(Hk), label=run["case_name"])
    plt.xlabel("omega")
    plt.ylabel("|controller(i omega)|")
    plt.title("Controller-law magnitude")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for run in runs:
        best = run["best"]
        if best is None:
            continue
        Hk = controller_transfer_response(best["a"], best["n"], best["K"], ws)
        plt.semilogx(ws, np.unwrap(np.angle(Hk)), label=run["case_name"])
    plt.xlabel("omega")
    plt.ylabel("unwrapped phase [rad]")
    plt.title("Controller-law phase")
    plt.grid(True, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_best_closed_loop_gain_curves(
    runs,
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    wmin=1e-4,
    wmax=2e1,
    nw=80,
):
    """
    This plot is expensive because each point is a gain solve.
    Use a small nw first.
    """
    wp = np.logspace(np.log10(wmin), np.log10(wmax), nw)
    data = []

    for run in runs:
        best = run["best"]
        if best is None:
            continue

        if run["case"]["mode"] == "static":
            cl = make_static_cl_for_placement(
                best["xa"], best["xs"], A, M, V, phi, bc, is_free, sigma_gauss, feedback_sign=+1.0
            )
            vals = np.array([max(cl.gain_of_omega(best["K"], +w), cl.gain_of_omega(best["K"], -w)) for w in wp])
        else:
            r = len(best["a"])
            cl = make_cldyn_for_placement(
                best["xa"], best["xs"], A, M, V, phi, bc, is_free, sigma_gauss, r=r
            )
            theta = cl_dyn.pack_theta(best["a"], best["n"], Kstat=best["K"])
            vals = np.array([max(cl.gain_of_omega(theta, +w), cl.gain_of_omega(theta, -w)) for w in wp])

        data.append((run["case_name"], vals))

    if IS_WORLD_ROOT:
        plt.figure(figsize=(8, 5))
        for name, vals in data:
            plt.loglog(wp, vals, label=name)
        plt.xlabel("omega")
        plt.ylabel("max(||G(i omega)||, ||G(-i omega)||)")
        plt.title("Best closed-loop gain curves")
        plt.grid(True, which="both")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Robust mixed-sensitivity PyGRANSO run: placement + dynamic controller + bounded K
# -----------------------------------------------------------------------------
# This block replaces the earlier four-case sweep.  It optimizes one dynamic
# SISO controller with variables
#     z = [eta_xa, eta_xs, a_1..a_r, n_1..n_r, eta_K]
# where xa, xs and K are bounded through tanh maps.  The objective remains the
# nonsmooth signed Hinf closed-loop resolvent from your Newton/disk oracle, while
# robustness is imposed through sampled mixed-sensitivity constraints:
#     alpha + margin <= 0
#     ||S||_inf <= MS_MAX,  S = 1/(1-L)
#     ||T||_inf <= MT_MAX,  T = L/(1-L)
#     ||K S||_inf <= KS_MAX
# using the positive-feedback convention A_cl = A + b K(s) c^H, so L = P K(s)
# and the return difference is 1 - L.
# -----------------------------------------------------------------------------

import csv
from scipy.optimize import minimize_scalar


def _rprint(*args, **kwargs):
    if IS_WORLD_ROOT:
        print(*args, **kwargs)


def _robust_signed_log_grid(wmin=1e-4, wmax=5e1, npos=45):
    wp = np.logspace(np.log10(float(wmin)), np.log10(float(wmax)), int(npos))
    return np.concatenate([-wp[::-1], np.array([0.0]), wp])


def _robust_plant_tf_scalar(A_use, M_use, b_vec, c_vec, omega):
    S = build_S(A_use, M_use, float(omega))
    ksp = make_lu_ksp(S)
    xsol = S.createVecRight()
    ksp.solve(b_vec, xsol)
    return c_vec.dot(xsol)  # c^H (i omega M - A)^-1 b


def _robust_controller_tf_dynamic(a, nvec, K, omega):
    s = 1j * float(omega)
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    nvec = np.asarray(nvec, dtype=np.complex128).reshape(-1)
    K = float(K)
    r = a.size
    if r == 0:
        return complex(K)
    Lc = np.zeros((r, r), dtype=np.complex128)
    for j in range(r - 1):
        Lc[j, j + 1] = 1.0
    Lc[-1, :] = -a
    er = np.zeros(r, dtype=np.complex128)
    er[-1] = 1.0
    try:
        xk = np.linalg.solve(s * np.eye(r, dtype=np.complex128) - Lc, er)
        return complex(K + np.vdot(nvec, xk))
    except np.linalg.LinAlgError:
        return complex(np.nan + 1j * np.nan)


def _robust_loop_metrics_dynamic(A_use, M_use, b_vec, c_vec, a, nvec, K,
                                 wmin=1e-4, wmax=5e1, npos=45,
                                 csv_name=None):
    ws = _robust_signed_log_grid(wmin=wmin, wmax=wmax, npos=npos)

    def _eval_one(w):
        w = float(w)
        P = _robust_plant_tf_scalar(A_use, M_use, b_vec, c_vec, w)
        Hk = _robust_controller_tf_dynamic(a, nvec, K, w)
        Lloop = P * Hk
        den = 1.0 - Lloop
        if not np.isfinite(den.real) or not np.isfinite(den.imag) or abs(den) < 1e-300:
            absS = absT = absKS = np.inf
            abs_return = 0.0
        else:
            Sval = 1.0 / den
            Tval = Lloop / den
            KSval = Hk / den
            absS = float(abs(Sval))
            absT = float(abs(Tval))
            absKS = float(abs(KSval))
            abs_return = float(abs(den))
        absL = float(abs(Lloop)) if np.isfinite(Lloop.real) and np.isfinite(Lloop.imag) else np.inf
        return (w, absS, absT, absKS, abs_return, absL, complex(P), complex(Hk))

    rows = mpi_task_map(_eval_one, list(ws), label="robust_loop_metrics_dynamic")

    Ms = Mt = Mks = 0.0
    w_Ms = w_Mt = w_Mks = np.nan
    min_return = np.inf
    w_min_return = np.nan
    peak_L = 0.0
    w_peak_L = np.nan

    for w, absS, absT, absKS, abs_return, absL, P, Hk in rows:
        if absS > Ms:
            Ms, w_Ms = absS, float(w)
        if absT > Mt:
            Mt, w_Mt = absT, float(w)
        if absKS > Mks:
            Mks, w_Mks = absKS, float(w)
        if abs_return < min_return:
            min_return, w_min_return = abs_return, float(w)
        if absL > peak_L:
            peak_L, w_peak_L = absL, float(w)

    if csv_name is not None and IS_WORLD_ROOT:
        with open(csv_name, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["omega", "absS", "absT", "absKS", "abs_1_minus_L", "absL",
                         "P_real", "P_imag", "Hk_real", "Hk_imag"])
            for w, absS, absT, absKS, ret, absL, P, Hk in rows:
                wr.writerow([w, absS, absT, absKS, ret, absL, P.real, P.imag, Hk.real, Hk.imag])

    return {
        "Ms": float(Ms), "omega_Ms": float(w_Ms),
        "Mt": float(Mt), "omega_Mt": float(w_Mt),
        "KS": float(Mks), "omega_KS": float(w_Mks),
        "min_return": float(min_return), "omega_min_return": float(w_min_return),
        "peak_L": float(peak_L), "omega_peak_L": float(w_peak_L),
        "rows": rows,
    }

def _robust_unpack_z(z, r, xa_bounds, xs_bounds, K_bounds,
                     a_center, n_center, a_scale, n_scale):
    z = np.asarray(z, dtype=float).reshape(-1)
    r = int(r)
    # z = [eta_xa, eta_xs, z_Re(a), z_Im(a), z_Re(n), z_Im(n), eta_K]
    nvar = 2 + 4 * r + 1
    if z.size != nvar:
        raise ValueError(f"Expected z size {nvar}, got {z.size}")
    xa, dxa = _tanh_to_interval(z[0], xa_bounds[0], xa_bounds[1])
    xs, dxs = _tanh_to_interval(z[1], xs_bounds[0], xs_bounds[1])
    a_center = np.asarray(a_center, dtype=np.complex128).reshape(r)
    n_center = np.asarray(n_center, dtype=np.complex128).reshape(r)
    a_scale_arr = np.asarray(a_scale, dtype=float)
    n_scale_arr = np.asarray(n_scale, dtype=float)
    if a_scale_arr.size == 1:
        a_scale_arr = float(a_scale_arr) * np.ones(r, dtype=float)
    if n_scale_arr.size == 1:
        n_scale_arr = float(n_scale_arr) * np.ones(r, dtype=float)

    ia_re = slice(2, 2 + r)
    ia_im = slice(2 + r, 2 + 2 * r)
    in_re = slice(2 + 2 * r, 2 + 3 * r)
    in_im = slice(2 + 3 * r, 2 + 4 * r)

    a = a_center + a_scale_arr * (z[ia_re] + 1j * z[ia_im])
    nvec = n_center + n_scale_arr * (z[in_re] + 1j * z[in_im])
    K, dK = _tanh_to_interval(z[-1], K_bounds[0], K_bounds[1])
    return {
        "xa": float(xa), "xs": float(xs), "a": a, "n": nvec, "K": float(K),
        "dxa_dz": float(dxa), "dxs_dz": float(dxs), "dK_dz": float(dK),
        "a_scale": a_scale_arr, "n_scale": n_scale_arr,
        "idx_a_re": ia_re, "idx_a_im": ia_im,
        "idx_n_re": in_re, "idx_n_im": in_im,
    }


def _robust_initial_z(r, xa_init, xs_init, K_init, xa_bounds, xs_bounds, K_bounds):
    z0 = np.zeros(2 + 4 * int(r) + 1, dtype=float)
    z0[0] = _interval_to_tanh(xa_init, xa_bounds[0], xa_bounds[1])
    z0[1] = _interval_to_tanh(xs_init, xs_bounds[0], xs_bounds[1])
    K0 = np.clip(float(K_init), K_bounds[0] + 1e-12, K_bounds[1] - 1e-12)
    z0[-1] = _interval_to_tanh(K0, K_bounds[0], K_bounds[1])
    return z0


def make_pygranso_combined_fn_robust_dynamic_placement(
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    r=4,
    xa_bounds=(-6.0, 4.0),
    xs_bounds=(-4.0, 6.0),
    K_bounds=(-5.0, 5.0),
    a_center=np.array([4.4, 16.3, 18.6, 7.7], dtype=float),
    n_center=np.zeros(4, dtype=float),
    a_scale=0.5,
    n_scale=0.10,
    placement_fd=2.5e-3,
    z_fd=1e-3,
    stab_margin=1e-3,
    stab_buffer=5e-4,
    stab_penalty_weight=1e8,
    unstable_objective=1e6,
    num_guess=1,
    wmin=1e-4,
    wmax=5e1,
    candidate_npos=50,
    disk_b=50.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
    loop_wmin=1e-4,
    loop_wmax=5e1,
    loop_npos=45,
    MS_MAX=2.0,
    MT_MAX=2.0,
    KS_MAX=50.0,
    base_seed=2,
    verbose=True,
):
    r = int(r)
    nvar = 2 + 4 * r + 1
    history = []
    cache_oracle = {}
    cache_loop = {}
    state = {"calls": 0}

    def key_from_z(z):
        return tuple(np.round(np.asarray(z, dtype=float).reshape(-1), 10))

    def eval_oracle_from_z(z):
        key = key_from_z(z)
        if key in cache_oracle:
            return cache_oracle[key]
        p = _robust_unpack_z(z, r, xa_bounds, xs_bounds, K_bounds, a_center, n_center, a_scale, n_scale)
        cl_dyn = make_cldyn_for_placement(
            xa=p["xa"], xs=p["xs"], A=A, M=M, V=V, phi=phi, bc=bc,
            is_free=is_free, sigma_gauss=sigma_gauss, r=r,
        )
        theta = cl_dyn.pack_theta(p["a"], p["n"], Kstat=p["K"])
        out = dynamic_oracle_hard_stability(
            cl_dyn,
            theta=theta,
            stab_margin=stab_margin,
            stab_buffer=stab_buffer,
            stab_penalty_weight=stab_penalty_weight,
            unstable_objective=unstable_objective,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=base_seed,
            candidate_npos=candidate_npos,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )
        out["p"] = p
        cache_oracle[key] = out
        return out

    def eval_loop_from_z(z):
        key = key_from_z(z)
        if key in cache_loop:
            return cache_loop[key]
        p = _robust_unpack_z(z, r, xa_bounds, xs_bounds, K_bounds, a_center, n_center, a_scale, n_scale)
        b_red, _ = build_actuator_vector(p["xa"], sigma_gauss, V, phi, bc, is_free)
        s_red, _ = build_sensor_state_vector(p["xs"], sigma_gauss, V, is_free)
        c_red = sensor_output_vector(M, s_red)
        loop = _robust_loop_metrics_dynamic(
            A, M, b_red, c_red, p["a"], p["n"], p["K"],
            wmin=loop_wmin, wmax=loop_wmax, npos=loop_npos,
        )
        cache_loop[key] = loop
        return loop

    def loop_constraints_from_z(z):
        loop = eval_loop_from_z(z)
        return np.array([
            loop["Ms"] / float(MS_MAX) - 1.0,
            loop["Mt"] / float(MT_MAX) - 1.0,
            loop["KS"] / float(KS_MAX) - 1.0,
        ], dtype=float)

    def fd_loop_constraint_grad(z, ci_loop_base):
        z = np.asarray(z, dtype=float).reshape(-1)
        g = np.zeros((nvar, 3), dtype=float)
        for j in range(nvar):
            h = float(z_fd) * max(1.0, abs(z[j]))
            zp = z.copy(); zm = z.copy()
            zp[j] += h; zm[j] -= h
            cp = loop_constraints_from_z(zp)
            cm = loop_constraints_from_z(zm)
            g[j, :] = (cp - cm) / (2.0 * h)
        return g

    def placement_fd_grad_f_c(z, p):
        hxa = float(placement_fd) * max(1.0, abs(p["xa"]))
        hxs = float(placement_fd) * max(1.0, abs(p["xs"]))
        xa_p, xa_m = _clip_step(p["xa"], hxa, xa_bounds)
        xs_p, xs_m = _clip_step(p["xs"], hxs, xs_bounds)

        def z_with_xa(xaval):
            zz = np.asarray(z, dtype=float).copy()
            zz[0] = _interval_to_tanh(float(xaval), xa_bounds[0], xa_bounds[1])
            return zz
        def z_with_xs(xsval):
            zz = np.asarray(z, dtype=float).copy()
            zz[1] = _interval_to_tanh(float(xsval), xs_bounds[0], xs_bounds[1])
            return zz

        if xa_p == xa_m:
            dfd_xa = dcd_xa = 0.0
        else:
            op = eval_oracle_from_z(z_with_xa(xa_p))
            om = eval_oracle_from_z(z_with_xa(xa_m))
            dfd_xa = (float(op["f_return"]) - float(om["f_return"])) / (xa_p - xa_m)
            dcd_xa = (float(op["c"]) - float(om["c"])) / (xa_p - xa_m)

        if xs_p == xs_m:
            dfd_xs = dcd_xs = 0.0
        else:
            op = eval_oracle_from_z(z_with_xs(xs_p))
            om = eval_oracle_from_z(z_with_xs(xs_m))
            dfd_xs = (float(op["f_return"]) - float(om["f_return"])) / (xs_p - xs_m)
            dcd_xs = (float(op["c"]) - float(om["c"])) / (xs_p - xs_m)
        return dfd_xa, dfd_xs, dcd_xa, dcd_xs

    def combined_fn(X_struct):
        state["calls"] += 1
        z_torch = X_struct.z
        z = z_torch.detach().cpu().numpy().reshape(-1)
        _progress_print(f"[hinf-r{r}] combined_fn call={state['calls']:03d}: base oracle start")
        out = eval_oracle_from_z(z)
        _progress_print(f"[hinf-r{r}] combined_fn call={state['calls']:03d}: base oracle done; FD gradient start")
        p = out["p"]
        loop = eval_loop_from_z(z)
        ci_loop = np.array([
            loop["Ms"] / float(MS_MAX) - 1.0,
            loop["Mt"] / float(MT_MAX) - 1.0,
            loop["KS"] / float(KS_MAX) - 1.0,
        ], dtype=float)

        f_return = float(out["f_return"])
        c_stab = float(out["c"])

        # Objective/stability gradients: placement by FD, controller params by existing oracle.
        df_dz = np.zeros(nvar, dtype=float)
        dcstab_dz = np.zeros(nvar, dtype=float)
        dfd_xa, dfd_xs, dcd_xa, dcd_xs = placement_fd_grad_f_c(z, p)
        _progress_print(f"[hinf-r{r}] combined_fn call={state['calls']:03d}: FD gradient done")
        df_dz[0] = dfd_xa * p["dxa_dz"]
        df_dz[1] = dfd_xs * p["dxs_dz"]
        dcstab_dz[0] = dcd_xa * p["dxa_dz"]
        dcstab_dz[1] = dcd_xs * p["dxs_dz"]

        df_ctrl = np.asarray(out["df_ctrl"], dtype=float).reshape(4 * r + 1)
        dc_ctrl = np.asarray(out["dc_ctrl"], dtype=float).reshape(4 * r + 1)
        # theta = [Re(a), Im(a), Re(n), Im(n), K]
        df_dz[p["idx_a_re"]] += df_ctrl[0:r] * p["a_scale"]
        df_dz[p["idx_a_im"]] += df_ctrl[r:2*r] * p["a_scale"]
        df_dz[p["idx_n_re"]] += df_ctrl[2*r:3*r] * p["n_scale"]
        df_dz[p["idx_n_im"]] += df_ctrl[3*r:4*r] * p["n_scale"]
        df_dz[-1] += df_ctrl[-1] * p["dK_dz"]

        dcstab_dz[p["idx_a_re"]] += dc_ctrl[0:r] * p["a_scale"]
        dcstab_dz[p["idx_a_im"]] += dc_ctrl[r:2*r] * p["a_scale"]
        dcstab_dz[p["idx_n_re"]] += dc_ctrl[2*r:3*r] * p["n_scale"]
        dcstab_dz[p["idx_n_im"]] += dc_ctrl[3*r:4*r] * p["n_scale"]
        dcstab_dz[-1] += dc_ctrl[-1] * p["dK_dz"]

        # Loop-metric constraints are nonsmooth; use finite-difference gradients.
        g_loop = fd_loop_constraint_grad(z, ci_loop)

        ci_vals = np.concatenate([[c_stab], ci_loop])
        ci_grad = np.column_stack([dcstab_dz, g_loop])  # nvar x 4

        rec = {
            "call": state["calls"], "z": z.copy(),
            "xa": p["xa"], "xs": p["xs"], "a": p["a"].copy(), "n": p["n"].copy(), "K": p["K"],
            "f_raw": float(out["f_raw"]), "f_return": f_return,
            "alpha": float(out["alpha"]), "c_stab": c_stab,
            "Ms": loop["Ms"], "Mt": loop["Mt"], "KS": loop["KS"],
            "c_Ms": ci_loop[0], "c_Mt": ci_loop[1], "c_KS": ci_loop[2],
            "omega_star": float(out["omega_star"]) if np.isfinite(out["omega_star"]) else np.nan,
            "omega_Ms": loop["omega_Ms"], "omega_Mt": loop["omega_Mt"], "omega_KS": loop["omega_KS"],
            "skipped_hinf": bool(out["skipped_hinf"]),
            "certified": bool(out["certified"]),
        }
        history.append(rec)

        if verbose and IS_WORLD_ROOT:
            print(
                f"[robust-dyn] call={state['calls']:03d} "
                f"f={f_return:.6e} raw={out['f_raw']:.6e} alpha={out['alpha']:+.3e} "
                f"xa={p['xa']:+.4f} xs={p['xs']:+.4f} K={p['K']:+.3f} "
                f"Ms={loop['Ms']:.3f}/{MS_MAX:g} Mt={loop['Mt']:.3f}/{MT_MAX:g} "
                f"KS={loop['KS']:.3f}/{KS_MAX:g} "
                f"ci=[{c_stab:+.2e},{ci_loop[0]:+.2e},{ci_loop[1]:+.2e},{ci_loop[2]:+.2e}]"
            )

        dev = z_torch.device
        dt = z_torch.dtype
        f_grad = torch.tensor(df_dz.reshape(-1, 1), device=dev, dtype=dt)
        ci = torch.tensor(ci_vals.reshape(-1, 1), device=dev, dtype=dt)
        ci_grad_t = torch.tensor(ci_grad, device=dev, dtype=dt)
        return [f_return, f_grad, ci, ci_grad_t, None, None]

    combined_fn.history = history
    combined_fn.nvar = nvar
    combined_fn.targets = {"MS_MAX": MS_MAX, "MT_MAX": MT_MAX, "KS_MAX": KS_MAX}
    combined_fn.bounds = {"xa_bounds": xa_bounds, "xs_bounds": xs_bounds, "K_bounds": K_bounds}
    return combined_fn


def _best_feasible_robust(history):
    feas = []
    for h in history:
        if not np.isfinite(h["f_raw"]):
            continue
        if h["c_stab"] <= 0.0 and h["c_Ms"] <= 0.0 and h["c_Mt"] <= 0.0 and h["c_KS"] <= 0.0:
            feas.append(h)
    if len(feas) == 0:
        return None
    return min(feas, key=lambda h: h["f_raw"])


def run_robust_dynamic_placement_pygranso(
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    x_min,
    x_max,
    r=4,
    xa_init=-1.0,
    xs_init=1.0,
    placement_radius=5.0,
    K_init=0.0,
    K_bounds=(-5.0, 5.0),
    a_center=np.array([4.4, 16.3, 18.6, 7.7], dtype=float),
    n_center=np.zeros(4, dtype=float),
    a_scale=0.5,
    n_scale=0.10,
    maxit=20,
    print_level=1,
    verbose=True,
    **kwargs,
):
    xa_bounds = (
        max(float(x_min) + 3.0 * float(sigma_gauss), float(xa_init) - float(placement_radius)),
        min(float(x_max) - 3.0 * float(sigma_gauss), float(xa_init) + float(placement_radius)),
    )
    xs_bounds = (
        max(float(x_min) + 3.0 * float(sigma_gauss), float(xs_init) - float(placement_radius)),
        min(float(x_max) - 3.0 * float(sigma_gauss), float(xs_init) + float(placement_radius)),
    )

    comb_fn = make_pygranso_combined_fn_robust_dynamic_placement(
        A=A, M=M, V=V, phi=phi, bc=bc, is_free=is_free, sigma_gauss=sigma_gauss,
        r=r, xa_bounds=xa_bounds, xs_bounds=xs_bounds, K_bounds=K_bounds,
        a_center=a_center, n_center=n_center, a_scale=a_scale, n_scale=n_scale,
        verbose=verbose, **kwargs,
    )

    z0 = _robust_initial_z(r, xa_init, xs_init, K_init, xa_bounds, xs_bounds, K_bounds)
    opts = pygransoStruct()
    opts.torch_device = torch.device("cpu")
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = int(maxit)
    opts.print_frequency = 1
    opts.print_level = (int(print_level) if IS_WORLD_ROOT else 0)
    opts.quadprog_info_msg = False
    opts.x0 = torch.tensor(z0.reshape(-1, 1), device=opts.torch_device, dtype=torch.double)
    var_spec = {"z": [comb_fn.nvar, 1]}

    t0 = time.time()
    try:
        pygranso(var_spec=var_spec, combined_fn=comb_fn, user_opts=opts)
    except Exception as e:
        _rprint(f"[robust-dyn] PyGRANSO stopped/failed: {repr(e)}")
    elapsed = time.time() - t0
    best = _best_feasible_robust(comb_fn.history)

    _rprint("\n" + "=" * 96)
    _rprint("ROBUST DYNAMIC PLACEMENT RUN DONE")
    _rprint(f"elapsed = {elapsed:.1f} s")
    _rprint(f"constraints: alpha+margin<=0, Ms<={comb_fn.targets['MS_MAX']}, "
            f"Mt<={comb_fn.targets['MT_MAX']}, KS<={comb_fn.targets['KS_MAX']}")
    if best is None:
        _rprint("No feasible robust iterate found. Try loosening MS_MAX/MT_MAX/KS_MAX or increasing maxit.")
    else:
        _rprint(f"best f       = {best['f_raw']:.10e}")
        _rprint(f"best alpha   = {best['alpha']:+.10e}")
        _rprint(f"best xa/xs   = {best['xa']:+.10f}, {best['xs']:+.10f}")
        _rprint(f"best K       = {best['K']:+.10e}")
        _rprint(f"best a       = {best['a']}")
        _rprint(f"best n       = {best['n']}")
        _rprint(f"best Ms/Mt/KS= {best['Ms']:.6e}, {best['Mt']:.6e}, {best['KS']:.6e}")
        _rprint(f"omegas       = Hinf {best['omega_star']:+.6e}, "
                f"Ms {best['omega_Ms']:+.6e}, Mt {best['omega_Mt']:+.6e}, KS {best['omega_KS']:+.6e}")
    _rprint("=" * 96 + "\n")

    if IS_WORLD_ROOT:
        with open("robust_dynamic_pygranso_history.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["call", "f_raw", "f_return", "alpha", "xa", "xs", "K", "Ms", "Mt", "KS",
                         "c_stab", "c_Ms", "c_Mt", "c_KS", "omega_star", "omega_Ms", "omega_Mt", "omega_KS"])
            for h in comb_fn.history:
                wr.writerow([h["call"], h["f_raw"], h["f_return"], h["alpha"], h["xa"], h["xs"], h["K"],
                             h["Ms"], h["Mt"], h["KS"], h["c_stab"], h["c_Ms"], h["c_Mt"], h["c_KS"],
                             h["omega_star"], h["omega_Ms"], h["omega_Mt"], h["omega_KS"]])
        if best is not None:
            with open("robust_dynamic_pygranso_best.txt", "w") as f:
                f.write(f"best f = {best['f_raw']:.12e}\n")
                f.write(f"best alpha = {best['alpha']:.12e}\n")
                f.write(f"best xa = {best['xa']:.12e}\n")
                f.write(f"best xs = {best['xs']:.12e}\n")
                f.write(f"best K = {best['K']:.12e}\n")
                f.write("best a = " + np.array2string(best['a'], precision=12) + "\n")
                f.write("best n = " + np.array2string(best['n'], precision=12) + "\n")
                f.write(f"best Ms = {best['Ms']:.12e}\n")
                f.write(f"best Mt = {best['Mt']:.12e}\n")
                f.write(f"best KS = {best['KS']:.12e}\n")

    return {"best": best, "history": comb_fn.history, "elapsed_sec": elapsed, "combined_fn": comb_fn}



# =============================================================================
# Hinf-only dynamic placement optimization, order r=6.
# No robustness-metric constraints are imposed in the optimization.
# Constraint: alpha + stab_margin <= 0 only.
# K remains bounded in K_bounds, default (-5, 5).
# =============================================================================

def make_pygranso_combined_fn_hinf_dynamic_placement(
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    r=6,
    xa_bounds=(-6.0, 4.0),
    xs_bounds=(-4.0, 6.0),
    K_bounds=(-5.0, 5.0),
    a_center=None,
    n_center=None,
    a_scale=0.5,
    n_scale=0.10,
    placement_fd=2.5e-3,
    stab_margin=1e-3,
    stab_buffer=5e-4,
    stab_penalty_weight=1e8,
    unstable_objective=1e6,
    num_guess=1,
    wmin=1e-4,
    wmax=5e1,
    candidate_npos=50,
    disk_b=50.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
    base_seed=2,
    verbose=True,
):
    """
    PyGRANSO combined_fn for the Hinf-only version.

    Variables z = [eta_xa, eta_xs, z_a[0:r], z_n[0:r], eta_K].
    Physical parameters:
        xa, xs via tanh maps to bounded intervals.
        K      via tanh map to K_bounds.
        a      = a_center + a_scale*z_a.
        n      = n_center + n_scale*z_n.

    Objective: signed-frequency Hinf(q -> q) from dynamic_oracle_hard_stability.
    Constraint: alpha + stab_margin <= 0 only.
    """
    r = int(r)
    if a_center is None:
        # Default stable 6th-order denominator.  These are the old r=4 poles
        # (-0.5, -1, -2.2, -4) plus two faster stable poles (-6, -8).
        poles = -np.array([0.5, 1.0, 2.2, 4.0, 6.0, 8.0], dtype=float)[:r]
        a_center = np.poly(poles)[1:][::-1]
    if n_center is None:
        n_center = np.zeros(r, dtype=float)
    a_center = np.asarray(a_center, dtype=np.complex128).reshape(r)
    n_center = np.asarray(n_center, dtype=np.complex128).reshape(r)

    nvar = 2 + 4 * r + 1
    history = []
    cache_oracle = {}
    state = {"calls": 0}

    def key_from_z(z):
        return tuple(np.round(np.asarray(z, dtype=float).reshape(-1), 10))

    def eval_oracle_from_z(z):
        key = key_from_z(z)
        if key in cache_oracle:
            return cache_oracle[key]
        p = _robust_unpack_z(z, r, xa_bounds, xs_bounds, K_bounds,
                             a_center, n_center, a_scale, n_scale)
        cl_dyn = make_cldyn_for_placement(
            xa=p["xa"], xs=p["xs"], A=A, M=M, V=V, phi=phi, bc=bc,
            is_free=is_free, sigma_gauss=sigma_gauss, r=r,
        )
        theta = cl_dyn.pack_theta(p["a"], p["n"], Kstat=p["K"])
        out = dynamic_oracle_hard_stability(
            cl_dyn,
            theta=theta,
            stab_margin=stab_margin,
            stab_buffer=stab_buffer,
            stab_penalty_weight=stab_penalty_weight,
            unstable_objective=unstable_objective,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=base_seed,
            candidate_npos=candidate_npos,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )
        out["p"] = p
        cache_oracle[key] = out
        return out

    def placement_fd_grad_f_c(z, p):
        hxa = float(placement_fd) * max(1.0, abs(p["xa"]))
        hxs = float(placement_fd) * max(1.0, abs(p["xs"]))
        xa_p, xa_m = _clip_step(p["xa"], hxa, xa_bounds)
        xs_p, xs_m = _clip_step(p["xs"], hxs, xs_bounds)

        def z_with_xa(xaval):
            zz = np.asarray(z, dtype=float).copy()
            zz[0] = _interval_to_tanh(float(xaval), xa_bounds[0], xa_bounds[1])
            return zz

        def z_with_xs(xsval):
            zz = np.asarray(z, dtype=float).copy()
            zz[1] = _interval_to_tanh(float(xsval), xs_bounds[0], xs_bounds[1])
            return zz

        tasks = []
        if xa_p != xa_m:
            tasks.append(("xa_p", z_with_xa(xa_p)))
            tasks.append(("xa_m", z_with_xa(xa_m)))
        if xs_p != xs_m:
            tasks.append(("xs_p", z_with_xs(xs_p)))
            tasks.append(("xs_m", z_with_xs(xs_m)))

        def _eval_fd_task(item):
            name, zz = item
            return name, eval_oracle_from_z(zz)

        # Optional outer-level FD parallelism.  By default, frequency/disk loops
        # inside each oracle use the task groups; set MPI_PARALLEL_FD=1 to spend
        # groups on the +/- placement finite-difference oracles instead.
        _progress_print(f"[hinf-r{r}] placement FD: {len(tasks)} oracle task(s), parallel_fd={_env_flag('MPI_PARALLEL_FD', False)}")
        if tasks and _env_flag("MPI_PARALLEL_FD", False):
            task_results = mpi_task_map(_eval_fd_task, tasks, label="placement_fd_oracles")
        else:
            task_results = [_eval_fd_task(t) for t in tasks]
        out_by_name = dict(task_results)

        if xa_p == xa_m:
            dfd_xa = dcd_xa = 0.0
        else:
            op = out_by_name["xa_p"]
            om = out_by_name["xa_m"]
            dfd_xa = (float(op["f_return"]) - float(om["f_return"])) / (xa_p - xa_m)
            dcd_xa = (float(op["c"]) - float(om["c"])) / (xa_p - xa_m)

        if xs_p == xs_m:
            dfd_xs = dcd_xs = 0.0
        else:
            op = out_by_name["xs_p"]
            om = out_by_name["xs_m"]
            dfd_xs = (float(op["f_return"]) - float(om["f_return"])) / (xs_p - xs_m)
            dcd_xs = (float(op["c"]) - float(om["c"])) / (xs_p - xs_m)
        return dfd_xa, dfd_xs, dcd_xa, dcd_xs

    def combined_fn(X_struct):
        state["calls"] += 1
        z_torch = X_struct.z
        z = z_torch.detach().cpu().numpy().reshape(-1)
        _progress_print(f"[hinf-r{r}] combined_fn call={state['calls']:03d}: base oracle start")
        out = eval_oracle_from_z(z)
        _progress_print(f"[hinf-r{r}] combined_fn call={state['calls']:03d}: base oracle done; FD gradient start")
        p = out["p"]

        f_return = float(out["f_return"])
        c_stab = float(out["c"])

        # Objective/stability gradients: placement by finite difference;
        # controller parameters by existing oracle derivatives.
        df_dz = np.zeros(nvar, dtype=float)
        dcstab_dz = np.zeros(nvar, dtype=float)
        dfd_xa, dfd_xs, dcd_xa, dcd_xs = placement_fd_grad_f_c(z, p)
        _progress_print(f"[hinf-r{r}] combined_fn call={state['calls']:03d}: FD gradient done")
        df_dz[0] = dfd_xa * p["dxa_dz"]
        df_dz[1] = dfd_xs * p["dxs_dz"]
        dcstab_dz[0] = dcd_xa * p["dxa_dz"]
        dcstab_dz[1] = dcd_xs * p["dxs_dz"]

        df_ctrl = np.asarray(out["df_ctrl"], dtype=float).reshape(4 * r + 1)
        dc_ctrl = np.asarray(out["dc_ctrl"], dtype=float).reshape(4 * r + 1)
        # theta = [Re(a), Im(a), Re(n), Im(n), K]
        df_dz[p["idx_a_re"]] += df_ctrl[0:r] * p["a_scale"]
        df_dz[p["idx_a_im"]] += df_ctrl[r:2*r] * p["a_scale"]
        df_dz[p["idx_n_re"]] += df_ctrl[2*r:3*r] * p["n_scale"]
        df_dz[p["idx_n_im"]] += df_ctrl[3*r:4*r] * p["n_scale"]
        df_dz[-1] += df_ctrl[-1] * p["dK_dz"]

        dcstab_dz[p["idx_a_re"]] += dc_ctrl[0:r] * p["a_scale"]
        dcstab_dz[p["idx_a_im"]] += dc_ctrl[r:2*r] * p["a_scale"]
        dcstab_dz[p["idx_n_re"]] += dc_ctrl[2*r:3*r] * p["n_scale"]
        dcstab_dz[p["idx_n_im"]] += dc_ctrl[3*r:4*r] * p["n_scale"]
        dcstab_dz[-1] += dc_ctrl[-1] * p["dK_dz"]

        ci_vals = np.array([c_stab], dtype=float)
        ci_grad = dcstab_dz.reshape(-1, 1)

        rec = {
            "call": state["calls"], "z": z.copy(),
            "xa": p["xa"], "xs": p["xs"], "a": p["a"].copy(),
            "n": p["n"].copy(), "K": p["K"],
            "f_raw": float(out["f_raw"]), "f_return": f_return,
            "alpha": float(out["alpha"]), "c_stab": c_stab,
            "omega_star": float(out["omega_star"]) if np.isfinite(out["omega_star"]) else np.nan,
            "skipped_hinf": bool(out["skipped_hinf"]),
            "certified": bool(out["certified"]),
        }
        history.append(rec)

        if verbose and IS_WORLD_ROOT:
            print(
                f"[hinf-r{r}] call={state['calls']:03d} "
                f"f={f_return:.6e} raw={out['f_raw']:.6e} alpha={out['alpha']:+.3e} "
                f"xa={p['xa']:+.4f} xs={p['xs']:+.4f} K={p['K']:+.3f} "
                f"c_stab={c_stab:+.2e} omega={rec['omega_star']:+.6e}"
            )

        dev = z_torch.device
        dt = z_torch.dtype
        f_grad = torch.tensor(df_dz.reshape(-1, 1), device=dev, dtype=dt)
        ci = torch.tensor(ci_vals.reshape(-1, 1), device=dev, dtype=dt)
        ci_grad_t = torch.tensor(ci_grad, device=dev, dtype=dt)
        return [f_return, f_grad, ci, ci_grad_t, None, None]

    combined_fn.history = history
    combined_fn.nvar = nvar
    combined_fn.bounds = {"xa_bounds": xa_bounds, "xs_bounds": xs_bounds, "K_bounds": K_bounds}
    return combined_fn


def _best_feasible_hinf(history):
    feas = []
    for h in history:
        if not np.isfinite(h["f_raw"]):
            continue
        if h["c_stab"] <= 0.0:
            feas.append(h)
    if len(feas) == 0:
        return None
    return min(feas, key=lambda h: h["f_raw"])


def default_stable_a_center(r):
    """Stable companion-form denominator coefficients for order r."""
    base_poles = np.array([-0.5, -1.0, -2.2, -4.0, -6.0, -8.0,
                           -10.0, -12.0, -15.0, -18.0], dtype=float)
    if r > base_poles.size:
        extra = -np.linspace(20.0, 20.0 + 2.0 * (r - base_poles.size - 1),
                             r - base_poles.size)
        poles = np.concatenate([base_poles, extra])
    else:
        poles = base_poles[:r]
    return np.poly(poles)[1:][::-1]


def run_hinf_only_dynamic_placement_pygranso(
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    x_min,
    x_max,
    r=6,
    xa_init=-0.70,
    xs_init=0.70,
    placement_radius=5.0,
    K_init=-4.5,
    K_bounds=(-5.0, 5.0),
    a_center=None,
    n_center=None,
    a_scale=0.5,
    n_scale=0.20,
    maxit=25,
    print_level=1,
    verbose=True,
    **kwargs,
):
    if a_center is None:
        a_center = default_stable_a_center(r)
    if n_center is None:
        n_center = np.zeros(r, dtype=float)
    a_center = np.asarray(a_center, dtype=np.complex128).reshape(r)
    n_center = np.asarray(n_center, dtype=np.complex128).reshape(r)

    xa_bounds = (
        max(float(x_min) + 3.0 * float(sigma_gauss), float(xa_init) - float(placement_radius)),
        min(float(x_max) - 3.0 * float(sigma_gauss), float(xa_init) + float(placement_radius)),
    )
    xs_bounds = (
        max(float(x_min) + 3.0 * float(sigma_gauss), float(xs_init) - float(placement_radius)),
        min(float(x_max) - 3.0 * float(sigma_gauss), float(xs_init) + float(placement_radius)),
    )

    comb_fn = make_pygranso_combined_fn_hinf_dynamic_placement(
        A=A, M=M, V=V, phi=phi, bc=bc, is_free=is_free, sigma_gauss=sigma_gauss,
        r=r, xa_bounds=xa_bounds, xs_bounds=xs_bounds, K_bounds=K_bounds,
        a_center=a_center, n_center=n_center, a_scale=a_scale, n_scale=n_scale,
        verbose=verbose, **kwargs,
    )

    z0 = _robust_initial_z(r, xa_init, xs_init, K_init, xa_bounds, xs_bounds, K_bounds)
    opts = pygransoStruct()
    opts.torch_device = torch.device("cpu")
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = int(maxit)
    opts.print_frequency = 1
    opts.print_level = (int(print_level) if IS_WORLD_ROOT else 0)
    opts.quadprog_info_msg = False
    opts.x0 = torch.tensor(z0.reshape(-1, 1), device=opts.torch_device, dtype=torch.double)
    var_spec = {"z": [comb_fn.nvar, 1]}

    _rprint(f"[hinf-r{r}] starting PyGRANSO: nvar={comb_fn.nvar}, maxit={int(maxit)}, disk_cert={_env_flag('HINF_USE_DISK_CERT', False)}, MPI_PARALLEL_FD={_env_flag('MPI_PARALLEL_FD', False)}")
    t0 = time.time()
    try:
        pygranso(var_spec=var_spec, combined_fn=comb_fn, user_opts=opts)
    except Exception as e:
        _rprint(f"[hinf-r{r}] PyGRANSO stopped/failed: {repr(e)}")
    elapsed = time.time() - t0
    best = _best_feasible_hinf(comb_fn.history)

    _rprint("\n" + "=" * 96)
    _rprint(f"HINF-ONLY DYNAMIC PLACEMENT RUN DONE, r={r}")
    _rprint(f"elapsed = {elapsed:.1f} s")
    _rprint("constraints: alpha+margin<=0 only; no Ms/Mt/KS constraints")
    _rprint(f"K_bounds = ({K_bounds[0]}, {K_bounds[1]})")
    if best is None:
        _rprint("No feasible iterate found. Try different initial K/xa/xs or lower stab_margin.")
    else:
        _rprint(f"best f       = {best['f_raw']:.10e}")
        _rprint(f"best alpha   = {best['alpha']:+.10e}")
        _rprint(f"best xa/xs   = {best['xa']:+.10f}, {best['xs']:+.10f}")
        _rprint(f"best K       = {best['K']:+.10e}")
        _rprint(f"best a       = {best['a']}")
        _rprint(f"best n       = {best['n']}")
        _rprint(f"omega        = Hinf {best['omega_star']:+.6e}")
    _rprint("=" * 96 + "\n")

    if IS_WORLD_ROOT:
        with open("hinf_only_r6_pygranso_history.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["call", "f_raw", "f_return", "alpha", "xa", "xs", "K",
                         "c_stab", "omega_star", "skipped_hinf", "certified"])
            for h in comb_fn.history:
                wr.writerow([h["call"], h["f_raw"], h["f_return"], h["alpha"], h["xa"], h["xs"],
                             h["K"], h["c_stab"], h["omega_star"], h["skipped_hinf"], h["certified"]])
        if best is not None:
            with open("hinf_only_r6_pygranso_best.txt", "w") as f:
                f.write(f"best f = {best['f_raw']:.12e}\n")
                f.write(f"best alpha = {best['alpha']:.12e}\n")
                f.write(f"best xa = {best['xa']:.12e}\n")
                f.write(f"best xs = {best['xs']:.12e}\n")
                f.write(f"best K = {best['K']:.12e}\n")
                f.write("best a = " + np.array2string(best['a'], precision=12) + "\n")
                f.write("best n = " + np.array2string(best['n'], precision=12) + "\n")
                f.write(f"best omega_star = {best['omega_star']:.12e}\n")

    return {"best": best, "history": comb_fn.history, "elapsed_sec": elapsed, "combined_fn": comb_fn}



# =============================================================================
# Hinf-only dynamic controller optimization, r=4, OPEN-LOOP start, NO K.
# =============================================================================
# This block assumes the helper functions/classes above are already defined:
#   ClosedLoopDynamicCompanionSISO, make_cldyn_for_placement,
#   dynamic_oracle_hard_stability, default_stable_a_center,
#   _robust_loop_metrics_dynamic, build_actuator_vector,
#   build_sensor_state_vector, sensor_output_vector, _rprint.
#
# Variables optimized by PyGRANSO:
#   z = [z_a[0:r], z_n[0:r]]
# Physical controller parameters:
#   a = a_center + a_scale*z_a
#   n = n_center + n_scale*z_n
#   K = 0 fixed  (no static feedthrough)
#   mc_gain = 1 fixed inside ClosedLoopDynamicCompanionSISO
#
# Open-loop start:
#   n_center = 0 and K = 0, so u = 0 initially.
# =============================================================================

def make_pygranso_combined_fn_hinf_r4_noK_openloop(
    cl_dyn,
    r=4,
    a_center=None,
    n_center=None,
    a_scale=1.0,
    n_scale=1.0,
    K_fixed=0.0,
    stab_margin=1e-3,
    stab_buffer=5e-4,
    stab_penalty_weight=1e8,
    unstable_objective=1e6,
    num_guess=1,
    wmin=1e-4,
    wmax=5e1,
    candidate_npos=50,
    disk_b=50.0,
    disk_eigs_tol=1e-8,
    disk_nev=60,
    disk_eps_probe=1e-6,
    disk_axis_warn=1e-3,
    base_seed=2,
    verbose=True,
):
    r = int(r)
    if a_center is None:
        a_center = default_stable_a_center(r)
    if n_center is None:
        n_center = np.zeros(r, dtype=np.complex128)
    # The dynamic controller supports complex coefficients.  PyGRANSO still
    # optimizes real variables, so represent a and n by independent real and
    # imaginary coordinates.
    a_center = np.asarray(a_center, dtype=np.complex128).reshape(r)
    n_center = np.asarray(n_center, dtype=np.complex128).reshape(r)

    nvar = 4 * r
    history = []
    cache = {}
    state = {"calls": 0}

    def unpack_z(z):
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.size != nvar:
            raise ValueError(f"z must have length {nvar} = 4*r for complex a/n, got {z.size}")

        za_re = z[0:r]
        za_im = z[r:2 * r]
        zn_re = z[2 * r:3 * r]
        zn_im = z[3 * r:4 * r]

        a = a_center + float(a_scale) * (za_re + 1j * za_im)
        nvec = n_center + float(n_scale) * (zn_re + 1j * zn_im)

        # ClosedLoopDynamicCompanionSISO.pack_theta returns
        # [Re(a), Im(a), Re(n), Im(n), K] when optimize_mc_gain=False.
        theta_full = cl_dyn.pack_theta(a, nvec, Kstat=float(K_fixed))
        return a, nvec, theta_full

    def key_from_z(z):
        return tuple(np.round(np.asarray(z, dtype=float).reshape(-1), 10))

    def eval_oracle_from_z(z):
        key = key_from_z(z)
        if key in cache:
            return cache[key]
        a, nvec, theta_full = unpack_z(z)
        out = dynamic_oracle_hard_stability(
            cl_dyn,
            theta=theta_full,
            stab_margin=stab_margin,
            stab_buffer=stab_buffer,
            stab_penalty_weight=stab_penalty_weight,
            unstable_objective=unstable_objective,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=base_seed,
            candidate_npos=candidate_npos,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )
        out["a"] = a.copy()
        out["n"] = nvec.copy()
        out["theta_full"] = theta_full.copy()
        cache[key] = out
        return out

    def combined_fn(X_struct):
        state["calls"] += 1
        z_torch = X_struct.z
        z = z_torch.detach().cpu().numpy().reshape(-1)
        out = eval_oracle_from_z(z)
        a = out["a"]
        nvec = out["n"]

        f_return = float(out["f_return"])
        c_stab = float(out["c"])

        # dynamic_oracle returns gradients with respect to
        # theta_full = [Re(a), Im(a), Re(n), Im(n), K].
        # We freeze K, so we keep only the first 4*r controller coordinates.
        expected_grad_len = cl_dyn.num_params
        df_ctrl = np.asarray(out["df_ctrl"], dtype=float).reshape(expected_grad_len)
        dc_ctrl = np.asarray(out["dc_ctrl"], dtype=float).reshape(expected_grad_len)

        if expected_grad_len != 4 * r + 1:
            raise ValueError(
                f"Expected cl_dyn.num_params={4 * r + 1} for complex a/n plus fixed K, "
                f"got {expected_grad_len}"
            )

        df_dz = np.zeros(nvar, dtype=float)
        dc_dz = np.zeros(nvar, dtype=float)

        # z = [za_re, za_im, zn_re, zn_im], while theta_full uses the same
        # ordering followed by K.  Scale the chain rule for a/n; skip K.
        df_dz[0:r] = df_ctrl[0:r] * float(a_scale)
        df_dz[r:2 * r] = df_ctrl[r:2 * r] * float(a_scale)
        df_dz[2 * r:3 * r] = df_ctrl[2 * r:3 * r] * float(n_scale)
        df_dz[3 * r:4 * r] = df_ctrl[3 * r:4 * r] * float(n_scale)

        dc_dz[0:r] = dc_ctrl[0:r] * float(a_scale)
        dc_dz[r:2 * r] = dc_ctrl[r:2 * r] * float(a_scale)
        dc_dz[2 * r:3 * r] = dc_ctrl[2 * r:3 * r] * float(n_scale)
        dc_dz[3 * r:4 * r] = dc_ctrl[3 * r:4 * r] * float(n_scale)

        rec = {
            "call": state["calls"],
            "z": z.copy(),
            "a": a.copy(),
            "n": nvec.copy(),
            "K": float(K_fixed),
            "f_raw": float(out["f_raw"]),
            "f_return": f_return,
            "alpha": float(out["alpha"]),
            "c_stab": c_stab,
            "omega_star": float(out["omega_star"]) if np.isfinite(out["omega_star"]) else np.nan,
            "skipped_hinf": bool(out["skipped_hinf"]),
            "certified": bool(out["certified"]),
        }
        history.append(rec)

        if verbose and MPI.COMM_WORLD.rank == 0:
            _rprint(
                f"[hinf-r{r}-noK] call={state['calls']:03d} "
                f"f={f_return:.6e} raw={out['f_raw']:.6e} alpha={out['alpha']:+.3e} "
                f"c_stab={c_stab:+.2e} omega={rec['omega_star']:+.6e} "
                f"||n||={np.linalg.norm(nvec):.3e}"
            )

        dev = z_torch.device
        dt = z_torch.dtype
        f_grad = torch.tensor(df_dz.reshape(-1, 1), device=dev, dtype=dt)
        ci = torch.tensor(np.array([c_stab], dtype=float).reshape(-1, 1), device=dev, dtype=dt)
        ci_grad = torch.tensor(dc_dz.reshape(-1, 1), device=dev, dtype=dt)
        return [f_return, f_grad, ci, ci_grad, None, None]

    combined_fn.history = history
    combined_fn.nvar = nvar
    combined_fn.unpack_z = unpack_z
    return combined_fn


def _best_feasible_hinf_r4_noK(history):
    feas = []
    for h in history:
        if np.isfinite(h["f_raw"]) and h["c_stab"] <= 0.0:
            feas.append(h)
    if len(feas) == 0:
        return None
    return min(feas, key=lambda h: h["f_raw"])


def run_hinf_r4_noK_openloop_pygranso(
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    r=4,
    xa_fixed=-0.98,
    xs_fixed=0.95,
    a_center=None,
    n_center=None,
    a_scale=1.0,
    n_scale=1.0,
    K_fixed=0.0,
    maxit=100,
    print_level=1,
    verbose=True,
    **kwargs,
):
    if a_center is None:
        a_center = default_stable_a_center(r)
    if n_center is None:
        n_center = np.zeros(r, dtype=np.complex128)
    a_center = np.asarray(a_center, dtype=np.complex128).reshape(r)
    n_center = np.asarray(n_center, dtype=np.complex128).reshape(r)

    cl_dyn = make_cldyn_for_placement(
        xa=float(xa_fixed), xs=float(xs_fixed), A=A, M=M, V=V, phi=phi,
        bc=bc, is_free=is_free, sigma_gauss=sigma_gauss, r=r,
    )

    comb_fn = make_pygranso_combined_fn_hinf_r4_noK_openloop(
        cl_dyn=cl_dyn,
        r=r,
        a_center=a_center,
        n_center=n_center,
        a_scale=a_scale,
        n_scale=n_scale,
        K_fixed=K_fixed,
        verbose=verbose,
        **kwargs,
    )

    # Open-loop start: z=0 -> a=a_center, n=n_center=0, K=0.
    z0 = np.zeros(comb_fn.nvar, dtype=float)

    opts = pygransoStruct()
    opts.torch_device = torch.device("cpu")
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = int(maxit)
    opts.print_frequency = 1
    opts.print_level = int(print_level)
    opts.quadprog_info_msg = False
    opts.x0 = torch.tensor(z0.reshape(-1, 1), device=opts.torch_device, dtype=torch.double)
    var_spec = {"z": [comb_fn.nvar, 1]}

    _rprint("\n" + "#" * 100)
    _rprint("HINF-ONLY r=4, NO-K, OPEN-LOOP START")
    _rprint(f"fixed placement: xa={float(xa_fixed):+.10f}, xs={float(xs_fixed):+.10f}")
    _rprint("variables: Re(a), Im(a), Re(n), Im(n) only; K fixed to 0; mc_gain fixed to 1")
    _rprint(f"a_center={a_center}")
    _rprint(f"n_center={n_center}")
    _rprint("#" * 100 + "\n")

    t0 = time.time()
    try:
        pygranso(var_spec=var_spec, combined_fn=comb_fn, user_opts=opts)
    except Exception as e:
        _rprint(f"[hinf-r{r}-noK] PyGRANSO stopped/failed: {repr(e)}")
    elapsed = time.time() - t0
    best = _best_feasible_hinf_r4_noK(comb_fn.history)

    _rprint("\n" + "=" * 96)
    _rprint(f"HINF-ONLY NO-K OPEN-LOOP-START RUN DONE, r={r}")
    _rprint(f"elapsed = {elapsed:.1f} s")
    _rprint("constraints: alpha+margin<=0 only; no placement movement; no K")
    if best is None:
        _rprint("No feasible iterate found.")
    else:
        _rprint(f"best f       = {best['f_raw']:.10e}")
        _rprint(f"best alpha   = {best['alpha']:+.10e}")
        _rprint(f"fixed xa/xs  = {float(xa_fixed):+.10f}, {float(xs_fixed):+.10f}")
        _rprint(f"fixed K      = {K_fixed:+.10e}")
        _rprint(f"best a       = {best['a']}")
        _rprint(f"best n       = {best['n']}")
        _rprint(f"omega        = Hinf {best['omega_star']:+.6e}")
    _rprint("=" * 96 + "\n")

    if MPI.COMM_WORLD.rank == 0:
        with open("hinf_r4_noK_openloop_history.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["call", "f_raw", "f_return", "alpha", "c_stab", "omega_star", "norm_n", "skipped_hinf", "certified"])
            for h in comb_fn.history:
                wr.writerow([h["call"], h["f_raw"], h["f_return"], h["alpha"], h["c_stab"],
                             h["omega_star"], float(np.linalg.norm(h["n"])), h["skipped_hinf"], h["certified"]])
        if best is not None:
            with open("hinf_r4_noK_openloop_best.txt", "w") as f:
                f.write(f"best f = {best['f_raw']:.12e}\n")
                f.write(f"best alpha = {best['alpha']:.12e}\n")
                f.write(f"fixed xa = {float(xa_fixed):.12e}\n")
                f.write(f"fixed xs = {float(xs_fixed):.12e}\n")
                f.write(f"fixed K = {float(K_fixed):.12e}\n")
                f.write("best a = " + np.array2string(best["a"], precision=12) + "\n")
                f.write("best n = " + np.array2string(best["n"], precision=12) + "\n")
                f.write(f"best omega_star = {best['omega_star']:.12e}\n")

    return {"best": best, "history": comb_fn.history, "elapsed_sec": elapsed, "combined_fn": comb_fn}


# =============================================================================
# TRUE CONTROLLER MULTISTART: Hinf-only, r=4, complex no-K, open-loop/near-open-loop starts.
# =============================================================================
# This replaces the single-start execution block.  It runs several independent
# PyGRANSO trajectories from different controller initial conditions.
#
# Important: if n=0 exactly, the controller is disconnected from the plant and
# the denominator a is initially irrelevant.  Therefore start 0 is the exact
# open-loop start, while starts 1..N use small random complex n seeds and stable
# random complex denominator poles.  No LQG warm start is used.
# =============================================================================


def _rng_complex_unit_vector(rng, r):
    v = rng.standard_normal(r) + 1j * rng.standard_normal(r)
    nv = np.linalg.norm(v)
    if nv == 0.0:
        v[0] = 1.0 + 0.0j
        nv = 1.0
    return v / nv


def _random_stable_complex_denominator(r, rng, pole_jitter=0.35, imag_max=2.0):
    """Return companion denominator coefficients from stable random complex poles.

    Coefficient ordering matches default_stable_a_center(r):
        D(s) = s^r + a[-1] s^(r-1) + ... + a[1] s + a[0]
    so we return np.poly(poles)[1:][::-1].
    """
    base_poles = np.array([-0.5, -1.0, -2.2, -4.0, -6.0, -8.0,
                           -10.0, -12.0], dtype=float)[:r]
    # Keep real parts negative and reasonably close to the usual stable poles.
    mag = np.abs(base_poles)
    re = -mag * np.exp(float(pole_jitter) * rng.standard_normal(r))
    im = float(imag_max) * rng.standard_normal(r)
    poles = re + 1j * im
    return np.poly(poles)[1:][::-1].astype(np.complex128)


def _make_r4_noK_multistart_centers(
    nstarts,
    r=4,
    seed=123,
    include_exact_openloop=True,
    n_norm_min=0.05,
    n_norm_max=5.0,
    pole_jitter=0.35,
    pole_imag_max=2.0,
):
    rng = np.random.default_rng(int(seed))
    starts = []

    if include_exact_openloop:
        starts.append({
            "start_id": 0,
            "seed": int(seed),
            "label": "exact_open_loop",
            "a_center": np.asarray(default_stable_a_center(r), dtype=np.complex128),
            "n_center": np.zeros(r, dtype=np.complex128),
            "n_norm_target": 0.0,
        })

    next_id = len(starts)
    for j in range(next_id, int(nstarts)):
        a0 = _random_stable_complex_denominator(
            r, rng, pole_jitter=pole_jitter, imag_max=pole_imag_max
        )
        if float(n_norm_min) <= 0.0 or float(n_norm_max) <= 0.0:
            nmag = 0.0
        else:
            lo = np.log(float(n_norm_min))
            hi = np.log(float(n_norm_max))
            nmag = float(np.exp(rng.uniform(lo, hi)))
        n0 = nmag * _rng_complex_unit_vector(rng, r)
        starts.append({
            "start_id": j,
            "seed": int(seed) + 1009 * j,
            "label": "random_near_open_loop",
            "a_center": a0,
            "n_center": n0,
            "n_norm_target": nmag,
        })
    return starts


def _run_one_hinf_r4_noK_controller_start(
    cl_dyn,
    start,
    r=4,
    a_scale=1.0,
    n_scale=1.0,
    K_fixed=0.0,
    maxit=40,
    print_level=0,
    verbose=True,
    **oracle_kwargs,
):
    sid = int(start["start_id"])
    a_center = np.asarray(start["a_center"], dtype=np.complex128).reshape(r)
    n_center = np.asarray(start["n_center"], dtype=np.complex128).reshape(r)

    comb_fn = make_pygranso_combined_fn_hinf_r4_noK_openloop(
        cl_dyn=cl_dyn,
        r=r,
        a_center=a_center,
        n_center=n_center,
        a_scale=a_scale,
        n_scale=n_scale,
        K_fixed=K_fixed,
        verbose=verbose,
        base_seed=int(start["seed"]),
        **oracle_kwargs,
    )

    # z=0 means: a=a_center, n=n_center.  For start 0 this is exact open loop.
    z0 = np.zeros(comb_fn.nvar, dtype=float)

    opts = pygransoStruct()
    opts.torch_device = torch.device("cpu")
    opts.double_precision = True
    opts.globalAD = False
    opts.maxit = int(maxit)
    opts.print_frequency = 1
    opts.print_level = int(print_level) if IS_WORLD_ROOT else 0
    opts.quadprog_info_msg = False
    opts.x0 = torch.tensor(z0.reshape(-1, 1), device=opts.torch_device, dtype=torch.double)
    var_spec = {"z": [comb_fn.nvar, 1]}

    _rprint("\n" + "=" * 96)
    _rprint(f"MULTISTART {sid:03d}: {start['label']}")
    _rprint(f"  seed             = {int(start['seed'])}")
    _rprint(f"  ||n_center||      = {np.linalg.norm(n_center):.6e}")
    _rprint(f"  a_center          = {np.array2string(a_center, precision=6)}")
    _rprint(f"  n_center          = {np.array2string(n_center, precision=6)}")
    _rprint("=" * 96)

    t0 = time.time()
    try:
        pygranso(var_spec=var_spec, combined_fn=comb_fn, user_opts=opts)
        status = "ok"
    except Exception as e:
        status = f"exception: {repr(e)}"
        _rprint(f"[multistart {sid:03d}] PyGRANSO stopped/failed: {repr(e)}")
    elapsed = time.time() - t0

    best = _best_feasible_hinf_r4_noK(comb_fn.history)

    # Write per-start history on world root.
    if IS_WORLD_ROOT:
        hist_name = f"hinf_r4_noK_multistart_{sid:03d}_history.csv"
        with open(hist_name, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["call", "f_raw", "f_return", "alpha", "c_stab", "omega_star",
                         "norm_n", "skipped_hinf", "certified"])
            for h in comb_fn.history:
                wr.writerow([h["call"], h["f_raw"], h["f_return"], h["alpha"], h["c_stab"],
                             h["omega_star"], float(np.linalg.norm(h["n"])),
                             h["skipped_hinf"], h["certified"]])

        if best is not None:
            with open(f"hinf_r4_noK_multistart_{sid:03d}_best.txt", "w") as f:
                f.write(f"start_id = {sid}\n")
                f.write(f"label = {start['label']}\n")
                f.write(f"seed = {int(start['seed'])}\n")
                f.write(f"best f = {best['f_raw']:.12e}\n")
                f.write(f"best alpha = {best['alpha']:.12e}\n")
                f.write("best a = " + np.array2string(best["a"], precision=12) + "\n")
                f.write("best n = " + np.array2string(best["n"], precision=12) + "\n")
                f.write(f"best omega_star = {best['omega_star']:.12e}\n")

    _rprint(f"[multistart {sid:03d}] elapsed={elapsed:.1f}s status={status}")
    if best is None:
        _rprint(f"[multistart {sid:03d}] no feasible iterate found")
    else:
        _rprint(f"[multistart {sid:03d}] best f={best['f_raw']:.10e} "
                f"alpha={best['alpha']:+.10e} omega={best['omega_star']:+.6e} "
                f"||n||={np.linalg.norm(best['n']):.6e}")

    return {
        "start": start,
        "best": best,
        "history": comb_fn.history,
        "elapsed_sec": elapsed,
        "status": status,
        "combined_fn": comb_fn,
    }


def run_hinf_r4_noK_true_multistart_openloop(
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    r=4,
    xa_fixed=-0.98,
    xs_fixed=0.95,
    nstarts=12,
    master_seed=123,
    include_exact_openloop=True,
    n_norm_min=0.05,
    n_norm_max=5.0,
    pole_jitter=0.35,
    pole_imag_max=2.0,
    a_scale=1.0,
    n_scale=1.0,
    K_fixed=0.0,
    maxit=40,
    print_level=0,
    verbose=True,
    **oracle_kwargs,
):
    r = int(r)
    starts = _make_r4_noK_multistart_centers(
        nstarts=int(nstarts),
        r=r,
        seed=int(master_seed),
        include_exact_openloop=bool(include_exact_openloop),
        n_norm_min=float(n_norm_min),
        n_norm_max=float(n_norm_max),
        pole_jitter=float(pole_jitter),
        pole_imag_max=float(pole_imag_max),
    )

    cl_dyn = make_cldyn_for_placement(
        xa=float(xa_fixed), xs=float(xs_fixed), A=A, M=M, V=V, phi=phi,
        bc=bc, is_free=is_free, sigma_gauss=sigma_gauss, r=r,
    )

    _rprint("\n" + "#" * 100)
    _rprint("TRUE CONTROLLER MULTISTART: HINF-ONLY r=4, COMPLEX NO-K")
    _rprint(f"fixed placement: xa={float(xa_fixed):+.10f}, xs={float(xs_fixed):+.10f}")
    _rprint(f"number of controller starts = {len(starts)}")
    _rprint("start 0 is exact open loop if HINF_MS_INCLUDE_EXACT_OPENLOOP=1")
    _rprint("other starts use random stable complex a_center and small random complex n_center")
    _rprint("no LQG warm start, no K, no placement movement")
    _rprint("#" * 100 + "\n")

    results = []
    total_t0 = time.time()
    for start in starts:
        res = _run_one_hinf_r4_noK_controller_start(
            cl_dyn=cl_dyn,
            start=start,
            r=r,
            a_scale=a_scale,
            n_scale=n_scale,
            K_fixed=K_fixed,
            maxit=maxit,
            print_level=print_level,
            verbose=verbose,
            **oracle_kwargs,
        )
        results.append(res)
    total_elapsed = time.time() - total_t0

    feasible = [res for res in results if res["best"] is not None]
    best_res = None if not feasible else min(feasible, key=lambda res: res["best"]["f_raw"])

    if IS_WORLD_ROOT:
        with open("hinf_r4_noK_true_multistart_summary.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["start_id", "label", "seed", "status", "elapsed_sec", "init_norm_n",
                         "best_f", "best_alpha", "best_omega", "best_norm_n"])
            for res in results:
                st = res["start"]
                b = res["best"]
                if b is None:
                    wr.writerow([st["start_id"], st["label"], st["seed"], res["status"],
                                 res["elapsed_sec"], float(np.linalg.norm(st["n_center"])),
                                 np.inf, np.nan, np.nan, np.nan])
                else:
                    wr.writerow([st["start_id"], st["label"], st["seed"], res["status"],
                                 res["elapsed_sec"], float(np.linalg.norm(st["n_center"])),
                                 b["f_raw"], b["alpha"], b["omega_star"],
                                 float(np.linalg.norm(b["n"]))])

        if best_res is not None:
            b = best_res["best"]
            st = best_res["start"]
            with open("hinf_r4_noK_true_multistart_global_best.txt", "w") as f:
                f.write(f"best start_id = {st['start_id']}\n")
                f.write(f"best label = {st['label']}\n")
                f.write(f"best seed = {int(st['seed'])}\n")
                f.write(f"best f = {b['f_raw']:.12e}\n")
                f.write(f"best alpha = {b['alpha']:.12e}\n")
                f.write(f"fixed xa = {float(xa_fixed):.12e}\n")
                f.write(f"fixed xs = {float(xs_fixed):.12e}\n")
                f.write(f"fixed K = {float(K_fixed):.12e}\n")
                f.write("best a = " + np.array2string(b["a"], precision=12) + "\n")
                f.write("best n = " + np.array2string(b["n"], precision=12) + "\n")
                f.write(f"best omega_star = {b['omega_star']:.12e}\n")

    _rprint("\n" + "#" * 100)
    _rprint("TRUE MULTISTART DONE")
    _rprint(f"total elapsed = {total_elapsed:.1f} s")
    if best_res is None:
        _rprint("No feasible iterate found in any start.")
    else:
        b = best_res["best"]
        st = best_res["start"]
        _rprint(f"global best start_id = {st['start_id']} ({st['label']})")
        _rprint(f"global best f       = {b['f_raw']:.10e}")
        _rprint(f"global best alpha   = {b['alpha']:+.10e}")
        _rprint(f"global best omega   = {b['omega_star']:+.6e}")
        _rprint(f"global best ||n||   = {np.linalg.norm(b['n']):.6e}")
        _rprint(f"global best a       = {b['a']}")
        _rprint(f"global best n       = {b['n']}")
    _rprint("#" * 100 + "\n")

    # Run loop diagnostics only for the global best.
    if best_res is not None:
        b = best_res["best"]
        b_vec, _ = build_actuator_vector(float(xa_fixed), sigma_gauss, V, phi, bc, is_free)
        s_vec_best, _ = build_sensor_state_vector(float(xs_fixed), sigma_gauss, V, is_free)
        c_vec_best = sensor_output_vector(M, s_vec_best)
        fine_loop = _robust_loop_metrics_dynamic(
            A, M, b_vec, c_vec_best, b["a"], b["n"], 0.0,
            wmin=float(os.environ.get("HINF_MS_LOOP_WMIN", "1e-5")),
            wmax=float(os.environ.get("HINF_MS_LOOP_WMAX", "5e1")),
            npos=int(os.environ.get("HINF_MS_LOOP_NPOS", "160")),
            csv_name="hinf_r4_noK_true_multistart_global_best_loop_fine.csv",
        )
        _rprint("Diagnostic loop check for global best Hinf-only r=4 no-K multistart design:")
        _rprint(f"  Ms={fine_loop['Ms']:.8e} at omega={fine_loop['omega_Ms']:+.8e}")
        _rprint(f"  Mt={fine_loop['Mt']:.8e} at omega={fine_loop['omega_Mt']:+.8e}")
        _rprint(f"  KS={fine_loop['KS']:.8e} at omega={fine_loop['omega_KS']:+.8e}")
        _rprint(f"  min |1-L|={fine_loop['min_return']:.8e} at omega={fine_loop['omega_min_return']:+.8e}")

    return {"results": results, "best_result": best_res, "elapsed_sec": total_elapsed}



# =============================================================================
# NONZERO-n SCREENED MULTISTART: Hinf-only, r=4, complex no-K.
# =============================================================================
# Purpose:
#   Avoid the exactly-degenerate open-loop start n=0.  We sample stable
#   denominator coefficients a and random complex numerator directions n_dir,
#   line-search the numerator magnitude rho, keep the best stable nonzero seeds,
#   then optionally run PyGRANSO from those nonzero centers.
#
# This uses no LQG warm start and keeps K fixed to 0.
# =============================================================================


def _env_float_list(name, default_values):
    raw = os.environ.get(name, None)
    if raw is None or str(raw).strip() == "":
        return list(default_values)
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _nonzero_seed_denominators(r, num_a, rng, include_default=True,
                               pole_jitter=0.65, pole_imag_max=4.0):
    dens = []
    if include_default:
        dens.append((0, "default_real", np.asarray(default_stable_a_center(r), dtype=np.complex128)))
    start = len(dens)
    for k in range(start, int(num_a)):
        a0 = _random_stable_complex_denominator(
            r, rng, pole_jitter=float(pole_jitter), imag_max=float(pole_imag_max)
        )
        dens.append((k, f"random_a_{k:03d}", a0))
    return dens


def _make_nonzero_screen_tasks(
    r=4,
    seed=123,
    num_a=4,
    dirs_per_a=6,
    rho_values=None,
    include_default_a=True,
    pole_jitter=0.65,
    pole_imag_max=4.0,
):
    rng = np.random.default_rng(int(seed))
    if rho_values is None:
        rho_values = np.logspace(-3, 5, 17)
    rho_values = np.asarray(rho_values, dtype=float).reshape(-1)

    denominators = _nonzero_seed_denominators(
        r=r,
        num_a=int(num_a),
        rng=rng,
        include_default=bool(include_default_a),
        pole_jitter=float(pole_jitter),
        pole_imag_max=float(pole_imag_max),
    )

    tasks = []
    task_id = 0
    for a_id, a_label, a0 in denominators:
        for d_id in range(int(dirs_per_a)):
            n_dir = _rng_complex_unit_vector(rng, r)
            for rho in rho_values:
                if float(rho) <= 0.0:
                    continue
                tasks.append({
                    "task_id": task_id,
                    "a_id": int(a_id),
                    "a_label": str(a_label),
                    "dir_id": int(d_id),
                    "rho": float(rho),
                    "a": np.asarray(a0, dtype=np.complex128),
                    "n_dir": np.asarray(n_dir, dtype=np.complex128),
                })
                task_id += 1
    return tasks


def _eval_nonzero_seed_task(args):
    cl_dyn, task, oracle_kwargs = args
    a0 = np.asarray(task["a"], dtype=np.complex128).reshape(cl_dyn.r)
    n0 = float(task["rho"]) * np.asarray(task["n_dir"], dtype=np.complex128).reshape(cl_dyn.r)
    theta = cl_dyn.pack_theta(a0, n0)

    rec = {
        "task_id": int(task["task_id"]),
        "a_id": int(task["a_id"]),
        "a_label": str(task["a_label"]),
        "dir_id": int(task["dir_id"]),
        "rho": float(task["rho"]),
        "norm_n": float(np.linalg.norm(n0)),
        "a_norm": float(np.linalg.norm(a0)),
        "status": "ok",
        "f_raw": np.inf,
        "f_return": np.inf,
        "alpha": np.nan,
        "c_stab": np.nan,
        "omega_star": np.nan,
        "skipped_hinf": True,
        "a": a0,
        "n": n0,
    }
    try:
        out = dynamic_oracle_hard_stability(cl_dyn, theta, **oracle_kwargs)
        rec.update({
            "f_raw": float(out["f_raw"]),
            "f_return": float(out["f_return"]),
            "alpha": float(out["alpha"]),
            "c_stab": float(out["c"]),
            "omega_star": float(out["omega_star"]) if np.isfinite(out["omega_star"]) else np.nan,
            "skipped_hinf": bool(out["skipped_hinf"]),
        })
    except Exception as exc:
        rec["status"] = "exception: " + repr(exc)
    return rec


def _screen_nonzero_seeds(
    cl_dyn,
    r=4,
    seed=123,
    num_a=4,
    dirs_per_a=6,
    rho_values=None,
    include_default_a=True,
    pole_jitter=0.65,
    pole_imag_max=4.0,
    oracle_kwargs=None,
    csv_name="nonzero_n_seed_screen.csv",
):
    if oracle_kwargs is None:
        oracle_kwargs = {}
    tasks = _make_nonzero_screen_tasks(
        r=r,
        seed=int(seed),
        num_a=int(num_a),
        dirs_per_a=int(dirs_per_a),
        rho_values=rho_values,
        include_default_a=bool(include_default_a),
        pole_jitter=float(pole_jitter),
        pole_imag_max=float(pole_imag_max),
    )
    _rprint("\n" + "#" * 100)
    _rprint("NONZERO-n SEED SCREEN")
    _rprint(f"tasks = {len(tasks)} = num_a({num_a}) * dirs_per_a({dirs_per_a}) * rho_count({len(rho_values)})")
    _rprint("K fixed to 0, no LQG, no exact n=0 starts")
    _rprint("#" * 100 + "\n")

    packed_tasks = [(cl_dyn, t, oracle_kwargs) for t in tasks]
    results = mpi_task_map(_eval_nonzero_seed_task, packed_tasks,
                           label="nonzero_n_seed_screen",
                           use_parallel=_env_flag("NONZERO_SCREEN_PARALLEL", True))

    # Sort feasible finite raw-Hinf seeds by objective.
    finite = [r0 for r0 in results if np.isfinite(r0["f_raw"]) and float(r0["c_stab"]) <= 0.0]
    finite.sort(key=lambda z: float(z["f_raw"]))

    if IS_WORLD_ROOT:
        with open(csv_name, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["task_id", "a_id", "a_label", "dir_id", "rho", "norm_n", "a_norm",
                         "status", "f_raw", "f_return", "alpha", "c_stab", "omega_star", "skipped_hinf"])
            for r0 in results:
                wr.writerow([r0["task_id"], r0["a_id"], r0["a_label"], r0["dir_id"],
                             r0["rho"], r0["norm_n"], r0["a_norm"], r0["status"],
                             r0["f_raw"], r0["f_return"], r0["alpha"], r0["c_stab"],
                             r0["omega_star"], r0["skipped_hinf"]])

        with open("nonzero_n_seed_screen_top.txt", "w") as f:
            f.write(f"num finite feasible seeds = {len(finite)}\n")
            for j, r0 in enumerate(finite[:20]):
                f.write("\n")
                f.write(f"rank {j}\n")
                f.write(f"task_id = {r0['task_id']}\n")
                f.write(f"a_label = {r0['a_label']}\n")
                f.write(f"dir_id = {r0['dir_id']}\n")
                f.write(f"rho = {r0['rho']:.12e}\n")
                f.write(f"f_raw = {r0['f_raw']:.12e}\n")
                f.write(f"alpha = {r0['alpha']:.12e}\n")
                f.write(f"omega_star = {r0['omega_star']:.12e}\n")
                f.write("a = " + np.array2string(r0["a"], precision=12) + "\n")
                f.write("n = " + np.array2string(r0["n"], precision=12) + "\n")

    _rprint("NONZERO-n SEED SCREEN DONE")
    _rprint(f"finite feasible seeds = {len(finite)} / {len(results)}")
    if finite:
        b = finite[0]
        _rprint(f"best screened seed: f={b['f_raw']:.10e}, alpha={b['alpha']:+.4e}, "
                f"rho={b['rho']:.3e}, ||n||={b['norm_n']:.3e}, "
                f"a_label={b['a_label']}, dir={b['dir_id']}, omega={b['omega_star']:+.6e}")
    _rprint("#" * 100 + "\n")
    return finite, results


def _make_pygranso_starts_from_screened(finite, top=5, seed=123):
    starts = []
    # Deduplicate lightly by (a_id, dir_id), keeping the best rho for each direction,
    # so PyGRANSO does not spend all starts on the same line-search direction.
    used = set()
    for r0 in finite:
        key = (int(r0["a_id"]), int(r0["dir_id"]))
        if key in used:
            continue
        used.add(key)
        sid = len(starts)
        starts.append({
            "start_id": sid,
            "seed": int(seed) + 1009 * sid,
            "label": f"screened_nonzero_a{r0['a_id']}_d{r0['dir_id']}_rho{r0['rho']:.3e}",
            "a_center": np.asarray(r0["a"], dtype=np.complex128),
            "n_center": np.asarray(r0["n"], dtype=np.complex128),
            "n_norm_target": float(np.linalg.norm(r0["n"])),
            "screen_f": float(r0["f_raw"]),
            "screen_alpha": float(r0["alpha"]),
            "screen_omega": float(r0["omega_star"]),
        })
        if len(starts) >= int(top):
            break
    return starts


def run_hinf_r4_noK_nonzero_screened_multistart(
    A,
    M,
    V,
    phi,
    bc,
    is_free,
    sigma_gauss,
    r=4,
    xa_fixed=-0.98,
    xs_fixed=0.95,
    screen_seed=123,
    num_a=4,
    dirs_per_a=6,
    rho_values=None,
    include_default_a=True,
    pole_jitter=0.65,
    pole_imag_max=4.0,
    opt_top=5,
    run_pygranso=True,
    a_scale=1.0,
    n_scale=1.0,
    maxit=40,
    print_level=0,
    verbose=True,
    **oracle_kwargs,
):
    r = int(r)
    cl_dyn = make_cldyn_for_placement(
        xa=float(xa_fixed), xs=float(xs_fixed), A=A, M=M, V=V, phi=phi,
        bc=bc, is_free=is_free, sigma_gauss=sigma_gauss, r=r,
    )

    finite, all_screen = _screen_nonzero_seeds(
        cl_dyn=cl_dyn,
        r=r,
        seed=int(screen_seed),
        num_a=int(num_a),
        dirs_per_a=int(dirs_per_a),
        rho_values=rho_values,
        include_default_a=bool(include_default_a),
        pole_jitter=float(pole_jitter),
        pole_imag_max=float(pole_imag_max),
        oracle_kwargs=oracle_kwargs,
        csv_name="nonzero_n_seed_screen.csv",
    )

    starts = _make_pygranso_starts_from_screened(finite, top=int(opt_top), seed=int(screen_seed))
    if not run_pygranso or len(starts) == 0:
        _rprint("Skipping PyGRANSO stage. Set NONZERO_RUN_PYGRANSO=1 and ensure screened feasible seeds exist.")
        return {"screened": finite, "all_screen": all_screen, "results": [], "best_result": None}

    _rprint("\n" + "#" * 100)
    _rprint("PYGRANSO FROM SCREENED NONZERO-n SEEDS")
    _rprint(f"starts = {len(starts)}")
    _rprint("No exact n=0 start; K fixed to 0; placement fixed")
    _rprint("#" * 100 + "\n")

    results = []
    total_t0 = time.time()
    for st in starts:
        _rprint(f"screen seed for start {st['start_id']:03d}: "
                f"screen_f={st['screen_f']:.6e}, alpha={st['screen_alpha']:+.3e}, "
                f"||n||={np.linalg.norm(st['n_center']):.3e}, label={st['label']}")
        res = _run_one_hinf_r4_noK_controller_start(
            cl_dyn=cl_dyn,
            start=st,
            r=r,
            a_scale=float(a_scale),
            n_scale=float(n_scale),
            K_fixed=0.0,
            maxit=int(maxit),
            print_level=int(print_level),
            verbose=bool(verbose),
            **oracle_kwargs,
        )
        results.append(res)
    total_elapsed = time.time() - total_t0

    feasible = [res for res in results if res["best"] is not None]
    best_res = None if not feasible else min(feasible, key=lambda res: res["best"]["f_raw"])

    if IS_WORLD_ROOT:
        with open("nonzero_n_screened_multistart_summary.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["start_id", "label", "seed", "status", "screen_f", "screen_alpha", "screen_omega",
                         "init_norm_n", "best_f", "best_alpha", "best_omega", "best_norm_n"])
            for res in results:
                st = res["start"]
                b = res["best"]
                if b is None:
                    wr.writerow([st["start_id"], st["label"], st["seed"], res["status"],
                                 st.get("screen_f", np.nan), st.get("screen_alpha", np.nan), st.get("screen_omega", np.nan),
                                 float(np.linalg.norm(st["n_center"])), np.inf, np.nan, np.nan, np.nan])
                else:
                    wr.writerow([st["start_id"], st["label"], st["seed"], res["status"],
                                 st.get("screen_f", np.nan), st.get("screen_alpha", np.nan), st.get("screen_omega", np.nan),
                                 float(np.linalg.norm(st["n_center"])), b["f_raw"], b["alpha"], b["omega_star"],
                                 float(np.linalg.norm(b["n"]))])

        if best_res is not None:
            b = best_res["best"]
            st = best_res["start"]
            with open("nonzero_n_screened_multistart_global_best.txt", "w") as f:
                f.write(f"best start_id = {st['start_id']}\n")
                f.write(f"best label = {st['label']}\n")
                f.write(f"best seed = {int(st['seed'])}\n")
                f.write(f"best f = {b['f_raw']:.12e}\n")
                f.write(f"best alpha = {b['alpha']:.12e}\n")
                f.write(f"fixed xa = {float(xa_fixed):.12e}\n")
                f.write(f"fixed xs = {float(xs_fixed):.12e}\n")
                f.write("fixed K = 0.000000000000e+00\n")
                f.write("best a = " + np.array2string(b["a"], precision=12) + "\n")
                f.write("best n = " + np.array2string(b["n"], precision=12) + "\n")
                f.write(f"best omega_star = {b['omega_star']:.12e}\n")

    _rprint("\n" + "#" * 100)
    _rprint("NONZERO-n SCREENED MULTISTART DONE")
    _rprint(f"PyGRANSO elapsed = {total_elapsed:.1f} s")
    if best_res is None:
        _rprint("No feasible PyGRANSO iterate found.")
    else:
        b = best_res["best"]
        st = best_res["start"]
        _rprint(f"global best start_id = {st['start_id']} ({st['label']})")
        _rprint(f"global best f       = {b['f_raw']:.10e}")
        _rprint(f"global best alpha   = {b['alpha']:+.10e}")
        _rprint(f"global best omega   = {b['omega_star']:+.6e}")
        _rprint(f"global best ||n||   = {np.linalg.norm(b['n']):.6e}")
        _rprint(f"global best a       = {b['a']}")
        _rprint(f"global best n       = {b['n']}")
    _rprint("#" * 100 + "\n")

    return {"screened": finite, "all_screen": all_screen, "results": results, "best_result": best_res}


# -----------------------------------------------------------------------------
# Execute screened nonzero-n multistart.
# -----------------------------------------------------------------------------
R4_NZ_XA = float(os.environ.get("HINF_R4_NOK_XA", "-0.98"))
R4_NZ_XS = float(os.environ.get("HINF_R4_NOK_XS", "0.95"))

if "NONZERO_RHO_VALUES" in os.environ:
    _rho_values = np.array(_env_float_list("NONZERO_RHO_VALUES", []), dtype=float)
else:
    _rho_values = np.logspace(
        float(os.environ.get("NONZERO_RHO_LOG10_MIN", "-4")),
        float(os.environ.get("NONZERO_RHO_LOG10_MAX", "6")),
        int(os.environ.get("NONZERO_RHO_COUNT", "21")),
    )

nonzero_n_screened_multistart_run = run_hinf_r4_noK_nonzero_screened_multistart(
    A=A,
    M=M,
    V=V,
    phi=phi,
    bc=bc,
    is_free=is_free,
    sigma_gauss=sigma_gauss,
    r=4,
    xa_fixed=R4_NZ_XA,
    xs_fixed=R4_NZ_XS,
    screen_seed=int(os.environ.get("NONZERO_SEED", "123")),
    num_a=int(os.environ.get("NONZERO_NUM_A", "4")),
    dirs_per_a=int(os.environ.get("NONZERO_DIRS_PER_A", "6")),
    rho_values=_rho_values,
    include_default_a=_env_flag("NONZERO_INCLUDE_DEFAULT_A", True),
    pole_jitter=float(os.environ.get("NONZERO_POLE_JITTER", "0.65")),
    pole_imag_max=float(os.environ.get("NONZERO_POLE_IMAG_MAX", "4.0")),
    opt_top=int(os.environ.get("NONZERO_OPT_TOP", "5")),
    run_pygranso=_env_flag("NONZERO_RUN_PYGRANSO", False),
    a_scale=float(os.environ.get("NONZERO_A_SCALE", "1.0")),
    n_scale=float(os.environ.get("NONZERO_N_SCALE", "1.0")),
    maxit=int(os.environ.get("HINF_MAXIT", "40")),
    print_level=int(os.environ.get("NONZERO_PRINT_LEVEL", "0")),
    verbose=_env_flag("NONZERO_VERBOSE", True),
    stab_margin=float(os.environ.get("HINF_STAB_MARGIN", "1e-3")),
    stab_buffer=float(os.environ.get("HINF_STAB_BUFFER", "5e-4")),
    stab_penalty_weight=float(os.environ.get("HINF_STAB_PENALTY", "1e8")),
    unstable_objective=float(os.environ.get("HINF_UNSTABLE_OBJECTIVE", "1e6")),
    num_guess=int(os.environ.get("NONZERO_SCREEN_NUM_GUESS", os.environ.get("HINF_NUM_GUESS", "1"))),
    wmin=float(os.environ.get("HINF_WMIN", "1e-4")),
    wmax=float(os.environ.get("HINF_WMAX", "5e1")),
    candidate_npos=int(os.environ.get("NONZERO_SCREEN_CANDIDATE_NPOS", os.environ.get("HINF_CANDIDATE_NPOS", "30"))),
    disk_b=float(os.environ.get("HINF_DISK_B", "50.0")),
    disk_eigs_tol=float(os.environ.get("HINF_DISK_EIGS_TOL", "1e-8")),
    disk_nev=int(os.environ.get("HINF_DISK_NEV", "60")),
    disk_eps_probe=float(os.environ.get("HINF_DISK_EPS_PROBE", "1e-6")),
    disk_axis_warn=float(os.environ.get("HINF_DISK_AXIS_WARN", "1e-3")),
)