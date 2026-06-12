# import librairies
import os
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

os.environ["PATH"] = "/opt/tools/texlive/2021/bin/x86_64-linux:" + os.environ["PATH"]
plt.style.use("/stck/vbasle/Public/mystyle.mplstyle")
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

    if IS_WORLD_ROOT:
        print(
            f"[INSIDE PENCIL ENTRY] received theta={theta:.12g}",
            flush=True,
        )

    s0 = 1j * theta

    if IS_WORLD_ROOT:
        print(
            f"[INSIDE PENCIL SHIFT] s0={s0.real:+.4e}{s0.imag:+.12e}j",
            flush=True,
        )
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
    for lam in lambdas[:min(len(lambdas), 8)]:
        w = float(lam.imag)

        for h in (-eps_eigs, 0.0, eps_eigs):
            g = gain_of_omega(A, M, w + h)
            bisect_log.append((w + h, g))

            if g > gamma:
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

        g_theta = gain_of_omega(A,M,theta)

        probe_ws = np.linspace(lo, hi, 21)
        probe_gs = [gain_of_omega(A, M, float(w)) for w in probe_ws]
        jmax = int(np.argmax(probe_gs))

        if IS_WORLD_ROOT:
            print(
                f"[interval probe] lo={lo:.12g}, hi={hi:.12g}, "
                f"best_w={probe_ws[jmax]:.12g}, "
                f"best_g={probe_gs[jmax]:.12g}, "
                f"gamma={gamma:.12g}, "
                f"violates={probe_gs[jmax] > gamma}",
                flush=True,
            )

        if probe_gs[jmax] > gamma:
            return True
                # Compute lambdas the eigenvalues of sN - Mgam close to i*theta

        if IS_WORLD_ROOT:
                print(
                    f"\n[VISIT] gamma={gamma:.12g}, "
                    f"lo={lo:.12g}, hi={hi:.12g}, "
                    f"theta={theta:.12g}, "
                    f"side={'NEG' if theta < 0 else 'POS'}, "
                    f"gain(theta)={g_theta:.12g}, "
                    f"violates={g_theta > gamma}, "
                    f"n_intervals_left={len(intervals)}",
                    flush=True,
                )

        if g_theta > gamma:
            return True
        lambdas, nconv = eigs_close_to_shift_pencil(
            N, Mgam, theta, nev=nev, tol=eigs_tol
        )
        # Compute lambdas the eigenvalues of sN - Mgam close to i*theta
        t0 = time.perf_counter()
        lambdas, nconv = eigs_close_to_shift_pencil(N, Mgam, theta, nev=nev, tol=eigs_tol)
        lambdas, nconv = eigs_close_to_shift_pencil(N, Mgam, theta, nev=nev, tol=eigs_tol)

        if IS_WORLD_ROOT:
            print(f"[pencil] theta={theta:.12g}, nconv={nconv}", flush=True)
            for j, lam in enumerate(lambdas[:8]):
                print(
                    f"  lam[{j}] = {lam.real:+.4e} {lam.imag:+.12e}j "
                    f"dist_to_shift={abs(lam - 1j*theta):.4e}",
                    flush=True,
                )
        timer.add("shiftinvert", time.perf_counter() - t0)
        if nconv == 0 or lambdas.size == 0:
            intervals += split_by_removed_middle(lo, hi, theta, r=0.0, min_progress=0.5*min_interval)
            print("Nconv == 0 and here is the interval", intervals)
            continue

        shift = 1j*theta    
        r = float(np.min(np.abs(lambdas - shift)))

        for lam in lambdas[:min(len(lambdas), 8)]:
            w = float(lam.imag)
            g = gain_of_omega(A, M, w)

            if IS_WORLD_ROOT:
                print(
                    f"[candidate probe] theta={theta:.12g}, "
                    f"lam={lam.real:+.4e}{lam.imag:+.12e}j, "
                    f"gain(Im lam)={g:.12g}, gamma={gamma:.12g}, "
                    f"violates={g > gamma}",
                    flush=True,
                )

            if g > gamma:
                return True
        if not np.isfinite(r):
            intervals += split_by_removed_middle(lo,hi, theta, r = 0.0, min_progress=0.5*min_interval)
            print("if not n.isfinite print here and thats the interval", intervals)
            continue

        if r <= 0.5*(hi - lo):
            intervals += split_by_removed_middle(lo, hi, theta, r=r, min_progress=0.5*min_interval)
            print("r < 0.5*(hi - lo)", intervals)

        g_neg_peak = gain_of_omega(A, M, -0.6450884089267692)
        g_pos_peak = gain_of_omega(A, M, 0.6450884089267692)

        if IS_WORLD_ROOT:
            print(
                f"[known peaks] gamma={gamma:.12g}, "
                f"gain(-0.645)={g_neg_peak:.12g}, "
                f"gain(+0.645)={g_pos_peak:.12g}",
                flush=True,
            )
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



# =============================================================================
# H-infinity explanation plots
# Paste this AFTER your current code, or keep it in a separate file and import it
# after A, M, gain_of_omega, build_N_from_M, build_Mgamma, and
# eigs_close_to_shift_pencil have been defined.
#
# To enable automatically at the end of this script, run for example:
#   MAKE_HINF_EXPLAIN_PLOTS=1 HINF_EXPLAIN_TRACE=1 python multistartfixed.py
# =============================================================================

def hinf_root_mkdir(outdir):
    """Check if this process is the root process"""
    
    if IS_WORLD_ROOT:
        os.makedirs(outdir, exist_ok=True)
    WORLD.Barrier()


def symmetric_log_frequency_grid(omega_min=1e-6, omega_max=5e1, number_of_points=250, include_zero=True):
    """Frequency grid that shows negative and positive frequencies on one axis."""

    omega_min = float(omega_min)
    omega_max = float(omega_max)
    number_of_points = int(number_of_points) # determines how many points are generated between wmin and wmax on the positive side

    # Check if 0 < omega_min and omega_max 
    if omega_min <= 0.0 or omega_max <= omega_min:
        raise ValueError("Need 0 < wmin < wmax.")
    
    # Generate frequencies
    frequencies = np.logspace(np.log10(omega_min), np.log10(omega_max), number_of_points)

    # Add zero with concatenate
    if include_zero:
        return np.concatenate((-frequencies[::-1], [0.0], frequencies))
    
    return np.concatenate((-frequencies[::-1], frequencies))

def compute_gain(omega):
    omega = float(omega)
    gain = float(gain_of_omega(A, M, omega))
    return omega, gain

def compute_openloop_gain_sweep(A, M, omega_min=1e-6, omega_max=5e1, number_of_points=250):
    """Compute ||G(i omega)||_inf on a frequency grid"""
    
    frequencies = symmetric_log_frequency_grid(omega_min=omega_min, omega_max=omega_max, number_of_points=number_of_points)

    # Evaluate all frequencies with MPI
    omega_and_gain = mpi_task_map(compute_gain, frequencies, label="hinf_plot_gain_sweep", use_parallel=True)

    # Sort frequencies from lowest to highest
    omega_and_gain = sorted(omega_and_gain, key=lambda p: p[0])
    frequencies = np.array([p[0] for p in omega_and_gain], dtype=float)
    gains = np.array([p[1] for p in omega_and_gain], dtype=float)

    return frequencies, gains

def plot_hinf_gain_curve(frequencies, gains, gamma_star=None, outdir="/stck/eclement/Control Of NavierStokes/plots", basename="01_gain_curve"):
    """Plot the resolvent gain curve and mark the sampled peak and optional gamma*."""
    if not IS_WORLD_ROOT:
        return None
    
    hinf_root_mkdir(outdir)

    gains = np.asarray(gains, dtype=float) 
    frequencies = np.asarray(frequencies, dtype=float)
    finite = np.isfinite(gains) # Test element-wise for finiteness (not infinity and not Not a Number).
    frequencies_plot = frequencies[finite]
    gains_plot = gains[finite]

    # H_inf norm
    maximum_gain_value_index = int(np.argmax(gains_plot))
    omega_peak = float(frequencies_plot[maximum_gain_value_index])
    gain_peak = float(gains_plot[maximum_gain_value_index])

    # Create a figure
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(frequencies_plot, gains_plot, linewidth=1.0, label=r"sampled $\|G(i\omega)\|$") # plot de gain curve
    ax.scatter([omega_peak], [gain_peak], s=55, zorder=5, label=fr"sampled peak: $\omega={omega_peak:.3g}$") #zorder = top of the curve, s = point size

    # Draw gamma_star if available
    if gamma_star is not None and np.isfinite(gamma_star):
        ax.axhline(float(gamma_star), linestyle="--", linewidth=2.0, label=fr"certified/estimated $\gamma_\star={float(gamma_star):.4g}$") # horizontal line

    if "bisect_log" in globals() and len(bisect_log) > 0:
        log = np.array([(float(omega), float(gain)) for (omega, gain) in bisect_log if np.isfinite(omega) and np.isfinite(gain)], dtype=float)
        if log.size:
            ax.scatter(log[:, 0], log[:, 1], s=55, alpha=0.7, label="oracle checks")

    # Symlog to include 0
    # linthresh = max(np.min(np.abs(frequencies_plot[frequencies_plot != 0.0])) if np.any(frequencies_plot != 0.0) else 1e-6, 1e-12)
    #ax.set_xscale("symlog", linthresh=5.0)
    ax.set_yscale("log")
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel(r"gain $\|G(i\omega)\|$")
    ax.set_title(r"H$\infty$ norm as the maximum frequency response gain")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize = 15)
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    fig.savefig(png, dpi=250)
     
    plt.close(fig)

    return {"png": png, "omega_peak": omega_peak, "gain_peak": gain_peak}

def plot_interval_exclusion_trace_complex_plane(
    trace,
    outdir,
    basename,
):
    """Plot Boyd interval exclusions on the actual complex lambda-plane."""

    if not IS_WORLD_ROOT:
        return None

    hinf_root_mkdir(outdir)

    events = trace.get("events", [])
    if len(events) == 0:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 9.0))

    # Complex-plane axes
    ax.axvline(0.0, linestyle="--", linewidth=1.5, label=r"imaginary axis")
    ax.axhline(0.0, linestyle=":", linewidth=1.0, label=r"real axis")

    max_r = 0.0

    for e in events:
        theta = float(e["theta"])
        lo = float(e["lo"])
        hi = float(e["hi"])

        # Current interval on the imaginary axis
        ax.plot(
            [0.0, 0.0],
            [lo, hi],
            linewidth=2.0,
            alpha=0.35,
        )

        # Shift point i theta
        ax.scatter(
            [0.0],
            [theta],
            s=28,
            zorder=4,
        )

        # Exclusion disk
        r = float(e.get("r", np.nan))
        if np.isfinite(r) and r > 0.0:
            circle = plt.Circle(
                (0.0, theta),
                r,
                fill=False,
                alpha=0.35,
                linewidth=1.5,
            )
            ax.add_patch(circle)
            max_r = max(max_r, r)

        # Excluded interval on the imaginary axis
        removed_lo = float(e.get("removed_lo", np.nan))
        removed_hi = float(e.get("removed_hi", np.nan))

        if np.isfinite(removed_lo) and np.isfinite(removed_hi):
            ax.plot(
                [0.0, 0.0],
                [removed_lo, removed_hi],
                linewidth=7.0,
                alpha=0.45,
            )
        nearest_real = e.get("nearest_real", np.nan)
        nearest_imag = e.get("nearest_imag", np.nan)

        if np.isfinite(nearest_real) and np.isfinite(nearest_imag):
            ax.scatter(
                [nearest_real],
                [nearest_imag],
                s=35,
                marker="x",
                zorder=5,
                label="nearest eigenvalue" if e["visit"] == 0 else None,
            )

            ax.plot(
                [0.0, nearest_real],
                [theta, nearest_imag],
                linewidth=1.0,
                alpha=0.5,
                label=r"radius $r$" if e["visit"] == 0 else None,
    )
        # Optional visit label
        ax.text(
            0.02 * max(1.0, max_r),
            theta,
            str(int(e["visit"])),
            fontsize=8,
            va="center",
        )

    b = float(trace.get("b", 50.0))
    xpad = max(1e-6, 1.1 * max_r)

    ax.set_xlim(-xpad, xpad)
    ax.set_ylim(-b, b)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r'$\Re(\lambda)$')
    ax.set_ylabel(r'$\Im(\lambda)$')
    ax.set_title(
        fr"Boyd imaginary-axis exclusion search for trial $\gamma={trace['gamma']:.4g}$"
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    fig.savefig(png, dpi=250)
    plt.close(fig)


    return {"png": png}
def plot_inner_check_trace(trace, outdir, basename="02_inner_check_trace"):
    """
    Plot the interval-exclusion trace for the inner H-infinity check.

    Shows which frequency intervals were visited and which parts were removed.
    """
    if not IS_WORLD_ROOT:
        return None

    hinf_root_mkdir(outdir)

    events = trace.get("events", [])
    if len(events) == 0:
        return None
    

    gamma = trace.get("gamma", None)
    violates = trace.get("violates", None)
    reason = trace.get("reason", "")

    fig_height = max(4.0, 0.28 * len(events) + 1.5)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))

    for k, event in enumerate(events):
        y = k

        lo = float(event["lo"])
        hi = float(event["hi"])
        theta = float(event["theta"])

        removed_lo = event.get("removed_lo", np.nan)
        removed_hi = event.get("removed_hi", np.nan)
        action = event.get("action", "")

        # Full interval being inspected.
        ax.hlines(
            y,
            lo,
            hi,
            linewidth=1.5,
            alpha=0.5,
            label="visited interval" if k == 0 else None,
        )

        # Midpoint where the eigenvalue problem was shifted.
        ax.scatter(
            [theta],
            [y],
            s=25,
            zorder=5,
            label=r"shift point $\theta$" if k == 0 else None,
        )

        # Removed/excluded interval.
        if np.isfinite(removed_lo) and np.isfinite(removed_hi):
            ax.hlines(
                y,
                float(removed_lo),
                float(removed_hi),
                linewidth=5.0,
                alpha=0.8,
                label="excluded interval" if k == 0 else None,
            )

        # Optional annotation of what happened.
        ax.text(
            hi,
            y,
            "  " + action,
            va="center",
            fontsize=8,
            alpha=0.8,
        )

    # Mark the actual violating frequency, if gamma failed.
    if violates and "hit_omega" in trace:
        hit_omega = float(trace["hit_omega"])
        ax.axvline(
            hit_omega,
            linestyle="--",
            linewidth=2.0,
            label=fr"violation at $\omega={hit_omega:.3g}$",
        )

    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel("interval visit")
    ax.set_title(
        fr"Inner check interval exclusion trace, $\gamma={gamma:.4g}$"
        if gamma is not None
        else "Inner check interval exclusion trace"
    )

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    ax.invert_yaxis()

    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    fig.savefig(png, dpi=250)
    plt.close(fig)

    print("YO")

    return {
        "png": png,
        "violates": violates,
        "reason": reason,
    }
def trace_inner_check_openloop(A, M, gamma, b=10.0, eigs_tol=1e-8, nev=60, eps_eigs=1e-6, max_visits=200):
    """
    Gamma is a constant that acts as a horizontal line. we want to know whether gamma crosses the gain curve. We pick gamma = (gamma_low + gamma_high) / 2 
    because we are doing a bisection. If gamma_low = 8 and gamma_high = 10 we test gamma = 9. If the Hamiltonian has imaginary eigenvalues then gamma_low becomes
    that gamma else gamma_high becomses the new gamma.
    Until gamma is small enough.

    boyd says if you have an eigenvalue with just the imaginary part then gamma_low changes. 
    So basically we look at an interval, take its midpoint, on the imaginary axis, we take the radius and we're like okay withing that disk no imaginary eigenvalues, 
    then we split it and look at other parts of the imaginary axis that havent been excluded yet. The radius is the distance to the closest eig
    """
    gamma = float(gamma)
    b = float(b)
    min_interval = float(eigs_tol)

    # Build the pencil matrices
    N = build_N_from_M(M)
    Mgam = build_Mgamma(A, M, gamma)
    N = N.convert("aij"); N.assemble()
    Mgam = Mgam.convert("aij"); Mgam.assemble()

    # Low-frequency guard check (could be high frequency too)
    guard = []
    for omega in [0.0, -1e-12, 1e-12, -1e-9, 1e-9, -1e-6, 1e-6, -1e-5, 1e-5]:
        gain = float(gain_of_omega(A, M, omega))
        guard.append({"omega": float(omega), "gain": gain})
        if gain > gamma:
            return {
                "gamma": gamma,
                "b": b,
                "violates": True,
                "reason": "low_frequency_gain_exceeds_gamma",
                "hit_omega": float(omega),
                "hit_gain": gain,
                "guard": guard,
                "events": [],
            }

    intervals = [(-b, 0.0), (0.0, b)]
    events = []
    visit = 0

    # Removes intervals that are empty or too small
    while intervals and visit < max_visits:
        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        intervals.sort(key=lambda interval: interval[1] - interval[0], reverse=True)
        lo, hi = intervals.pop(0)
        theta = 0.5 * (lo + hi)

        lambdas, nconv = eigs_close_to_shift_pencil(N, Mgam, theta, nev=nev, tol=eigs_tol)
        event = {
            "visit": visit,
            "lo": float(lo),
            "hi": float(hi),
            "theta": float(theta),
            "nconv": int(nconv),
            "r": np.nan,
            "nearest_real": np.nan,
            "nearest_imag": np.nan,
            "removed_lo": np.nan,
            "removed_hi": np.nan,
            "action": "no_eigenvalues_returned",
        }

        # Once a region of frequencies has been certified as safe or already checked, 
        # remove it from the search interval and continue checking the remaining pieces.
        if nconv == 0 or lambdas.size == 0:
            remaining = split_by_removed_middle(lo, hi, theta, r=0.0, min_progress=0.5 * min_interval)
            event["removed_lo"] = theta
            event["removed_hi"] = theta
            event["remaining_after"] = len(intervals) + len(remaining)
            events.append(event)
            intervals += remaining
            visit += 1
            continue
        
        # Each iteration we repeatedly choose different thera values from the remaining intervals
        shift = 1j * theta
        lambdas = np.array(lambdas, dtype=complex)
        lambdas = lambdas[np.argsort(np.abs(lambdas - shift))]
        nearest = lambdas[0]
        r = float(abs(nearest - shift))

        event.update({
            "r": r,
            "nearest_real": float(np.real(nearest)),
            "nearest_imag": float(np.imag(nearest)),
        })

        # If an eigenvalue is essentially on the imaginary axis, probe the gain nearby.
        # This is the visual reason gamma is too low.
        for lam in lambdas[:min(len(lambdas), 6)]:
            if abs(lam.real) <= 1e-8:
                for h in (-eps_eigs, 0.0, eps_eigs):
                    w_probe = float(lam.imag + h)
                    g_probe = float(gain_of_omega(A, M, w_probe))
                    if g_probe > gamma:
                        event["action"] = "imaginary_axis_hit_gamma_too_low"
                        events.append(event)
                        return {
                            "gamma": gamma,
                            "b": b,
                            "violates": True,
                            "reason": "imaginary_axis_hit_gamma_too_low",
                            "hit_omega": w_probe,
                            "hit_gain": g_probe,
                            "guard": guard,
                            "events": events,
                        }

        if np.isfinite(r) and r <= 0.5 * (hi - lo):
            remaining = split_by_removed_middle(lo, hi, theta, r=r, min_progress=0.5 * min_interval)
            event["removed_lo"] = float(max(lo, theta - max(r, 0.5 * min_interval)))
            event["removed_hi"] = float(min(hi, theta + max(r, 0.5 * min_interval)))
            event["action"] = "removed_certified_disk"
            event["remaining_after"] = len(intervals) + len(remaining)
            intervals += remaining
        else:
            event["removed_lo"] = float(lo)
            event["removed_hi"] = float(hi)
            event["action"] = "whole_interval_excluded"
            event["remaining_after"] = len(intervals)

        events.append(event)
        visit += 1

    return {
        "gamma": gamma,
        "b": b,
        "violates": False,
        "reason": "intervals_exhausted_or_visit_limit",
        "guard": guard,
        "events": events,
    }
def plot_interval_exclusion_trace(trace, outdir="hinf_explain_plots", basename="02_interval_exclusion_trace"):
    """Plot interval bisection/exclusion visits from trace_inner_check_openloop."""

    if not IS_WORLD_ROOT:
        return None

    hinf_root_mkdir(outdir)
    # Event contains the interval visit history
    events = trace.get("events", [])
    if len(events) == 0:
        return None

    # Figure taller if more interval visits
    fig_h = max(4.5, min(12.0, 0.22 * len(events) + 3.0))
    fig, ax = plt.subplots(figsize=(9.0, fig_h))

    for e in events:
        y = int(e["visit"])
        ax.hlines(y, e["lo"], e["hi"], linewidth=3.0, label="tested interval" if y == 0 else None)
        ax.scatter([e["theta"]], [y], s=18, zorder=4, color="orange", label=r"shift $i\theta$" if y == 0 else None)
        if np.isfinite(e.get("removed_lo", np.nan)) and np.isfinite(e.get("removed_hi", np.nan)):
            ax.hlines(y + 0.18, e["removed_lo"], e["removed_hi"], linewidth=7.0,
                      alpha=0.35, label="excluded region" if y == 0 else None)
        nearest_real = e.get("nearest_real", np.nan)
        nearest_imag = e.get("nearest_imag", np.nan)

        if np.isfinite(nearest_real) and np.isfinite(nearest_imag):
            ax.scatter(
            [e["nearest_imag"]],
            [y],
            marker="x",
            s=35,
            label="Im(nearest eigenvalue)" if y == 0 else None,
        )

            ax.plot(
                [e["theta"], nearest_imag],
                [y, y],
                linewidth=1.0,
                alpha=0.5,
                label=r"imaginary-axis projection of $r$" if y == 0 else None,
            )
        
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel("interval visit")
    ax.set_title(fr"Boyd disk/certificate interval search for trial $\gamma={trace['gamma']:.4g}$")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    plot_interval_exclusion_trace_complex_plane(trace=trace, outdir=outdir, basename="lol")
    
    return {"png": png}


def plot_pencil_eigs_near_shift_openloop(A, M, gamma, theta=0.0, nev=60, eigs_tol=1e-8,
                                         outdir="hinf_explain_plots",
                                         basename="03_pencil_eigs_near_shift"):
    """
    Plot eigenvalues of sN - M_gamma close to i*theta. The visual message:
    when gamma crosses the H-inf norm, this pencil touches the imaginary axis.
    """
    gamma = float(gamma)
    theta = float(theta)

    N = build_N_from_M(M)
    Mgam = build_Mgamma(A, M, gamma)
    N = N.convert("aij"); N.assemble()
    Mgam = Mgam.convert("aij"); Mgam.assemble()
    lambdas, nconv = eigs_close_to_shift_pencil(N, Mgam, theta, nev=nev, tol=eigs_tol)

    if not IS_WORLD_ROOT:
        return None

    hinf_root_mkdir(outdir)
    lambdas = np.array(lambdas, dtype=complex)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if lambdas.size:
        ax.scatter(lambdas.real, lambdas.imag, s=42, label="nearby pencil eigenvalues")
    ax.axvline(0.0, linestyle="--", linewidth=1.5, label="imaginary axis")
    ax.axhline(theta, linestyle=":", linewidth=1.5, label=fr"target $\theta={theta:.3g}$")
    ax.scatter([0.0], [theta], marker="x", s=80, label=r"shift $i\theta$")
    ax.set_xlabel(r'$\Re(\lambda)$')
    ax.set_ylabel(r'$\Im(\lambda)$')
    ax.set_title(fr"Hamiltonian/Boyd pencil near $i\theta$ for $\gamma={gamma:.4g}$")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    suffix = f"_theta_{theta:+.3e}".replace("+", "p").replace("-", "m")
    png = os.path.join(outdir, f"{basename}{suffix}.png")

    fig.savefig(png, dpi=250)

    plt.close(fig)



    return {"png": png, "nconv": int(nconv)}


def plot_hinf_algorithm_flowchart(outdir="hinf_explain_plots", basename="00_algorithm_flowchart"):
    """Small visual summary of the H-inf norm certification algorithm."""
    if not IS_WORLD_ROOT:
        return None

    hinf_root_mkdir(outdir)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.axis("off")

    boxes = [
        (0.05, 0.70, "1. Choose trial level $\\gamma$\nIs the peak gain above this?"),
        (0.31, 0.70, "2. Build resolvent test\n$S=i\\omega M-A$\n$\\|G(i\\omega)\\|$ from largest eigenvalue"),
        (0.58, 0.70, "3. Build Boyd pencil\n$sN-M_\\gamma$"),
        (0.31, 0.28, "4. Search frequency intervals\nshift near $i\\theta$ and remove\ncertified safe disks"),
        (0.58, 0.28, "5a. Imaginary-axis hit\n$\\Rightarrow \\gamma$ is too low"),
        (0.80, 0.28, "5b. All intervals removed\n$\\Rightarrow \\gamma$ is high enough"),
    ]

    for x, y, txt in boxes:
        ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.45", alpha=0.16))

    arrows = [
        ((0.25, 0.70), (0.30, 0.70)),
        ((0.52, 0.70), (0.57, 0.70)),
        ((0.67, 0.63), (0.45, 0.38)),
        ((0.50, 0.28), (0.57, 0.28)),
        ((0.72, 0.28), (0.79, 0.28)),
        ((0.42, 0.36), (0.16, 0.64)),
    ]
    for xy0, xy1 in arrows:
        ax.annotate("", xy=xy1, xytext=xy0, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", linewidth=1.6))

    ax.text(0.05, 0.08,
            r"Outer bisection repeats this test until $\gamma_{high}-\gamma_{low}$ is small; "
            r"the result is $\|G\|_\infty$.",
            transform=ax.transAxes, fontsize=12)

    fig.tight_layout()
    png = os.path.join(outdir, f"{basename}.png")
    fig.savefig(png, dpi=250)
    plt.close(fig)
    return {"png": png}


def make_hinf_explanation_plots_openloop(A, M, gamma_star=None, wmin=1e-6, wmax=5e1,
                                         npos=250, outdir="hinf_explain_plots",
                                         do_trace=True, trace_gamma_factor=1.01,
                                         eigs_tol=1e-8, nev=60):
    """
    Main convenience function. It crea1e-80.81e-61e-61e-31221+e5e16
      00_algorithm_flowchart
      01_gain_curve
      02_interval_exclusion_trace, optional and more expensive
      03_pencil_eigs_near_shift
    """
    paths = {}
    flow = plot_hinf_algorithm_flowchart(outdir=outdir)
    if IS_WORLD_ROOT:
        paths["flowchart"] = flow

    ws, vals = compute_openloop_gain_sweep(A, M, omega_min=wmin, omega_max=wmax, number_of_points=npos)
    sampled_peak = float(np.nanmax(vals))
    w_peak = float(ws[int(np.nanargmax(vals))])

    gamma_for_plot = float(gamma_star) if gamma_star is not None else sampled_peak
    gain_paths = plot_hinf_gain_curve(ws, vals, gamma_star=gamma_for_plot, outdir=outdir)
    if IS_WORLD_ROOT:
        paths["gain_curve"] = gain_paths

    gamma_for_certificate = float(gamma_star) if gamma_star is not None else trace_gamma_factor * sampled_peak

    if do_trace:
        trace = trace_inner_check_openloop(A, M, gamma_for_certificate, b=wmax,
                                           eigs_tol=eigs_tol, nev=nev)
        trace_paths = plot_interval_exclusion_trace(trace, outdir=outdir)
        if IS_WORLD_ROOT:
            paths["interval_trace"] = trace_paths
            paths["trace_summary"] = {
                "gamma": trace["gamma"],
                "violates": trace["violates"],
                "reason": trace["reason"],
                "num_events": len(trace.get("events", [])),
            }

    eig_paths = plot_pencil_eigs_near_shift_openloop(A, M, gamma_for_certificate,
                                                     theta=w_peak, nev=nev,
                                                     eigs_tol=eigs_tol,
                                                     outdir=outdir)
    
    if IS_WORLD_ROOT:
        paths["pencil_eigs"] = eig_paths
        paths["sampled_peak"] = {"omega": w_peak, "gain": sampled_peak}
        print("\nH-infinity explanation plots written to:", outdir, flush=True)
        for key, value in paths.items():
            print(" ", key, "->", value, flush=True)

    return paths if IS_WORLD_ROOT else None


# Optional automatic execution. This is OFF by default so your expensive
# multistart run does not always generate extra figures.
make_hinf_explanation_plots_openloop(
    A, M,
    gamma_star=None,
    wmin=float(os.environ.get("HINF_EXPLAIN_WMIN", "1e-1")),
    wmax=float(os.environ.get("HINF_EXPLAIN_WMAX", "1e1")),
    npos=int(os.environ.get("HINF_EXPLAIN_NPOS", "500")),
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    do_trace=_env_flag("HINF_EXPLAIN_TRACE", True),
    eigs_tol=float(os.environ.get("HINF_EXPLAIN_EIGS_TOL", "1e-8")),
    nev=int(os.environ.get("HINF_EXPLAIN_NEV", "60")),
)

def outer_bisection_with_trace(A, M, tol_gamma=1e-8, tol_imag=1e-10, max_bisect=100):
    """
    Same as outer_bisection, but records the history of gamma_low, gamma_high,
    and the tested trial gamma values.
    """
    gamma_low = 0.0
    gamma_high = 1e-6
    history = []

    # First expand gamma_high until it is no longer too low.
    expand_it = 0
    while inner_check(gamma_high, A, M, tol_imag=tol_imag):
        history.append({
            "phase": "bracket",
            "iter": expand_it,
            "gamma_low": float(gamma_low),
            "gamma_high": float(gamma_high),
            "gamma_trial": float(gamma_high),
            "violates": True,   # gamma_high still too low
        })
        gamma_low = gamma_high
        gamma_high *= 2.0
        expand_it += 1

    # Record the first gamma_high that passes
    history.append({
        "phase": "bracket",
        "iter": expand_it,
        "gamma_low": float(gamma_low),
        "gamma_high": float(gamma_high),
        "gamma_trial": float(gamma_high),
        "violates": False,
    })

    # Standard outer bisection
    for it in range(max_bisect):
        gamma = 0.5 * (gamma_low + gamma_high)
        violates = bool(inner_check(gamma, A, M, tol_imag=tol_imag))

        history.append({
            "phase": "bisect",
            "iter": it,
            "gamma_low": float(gamma_low),
            "gamma_high": float(gamma_high),
            "gamma_trial": float(gamma),
            "violates": violates,
        })

        if violates:
            gamma_low = gamma
        else:
            gamma_high = gamma

        if gamma_high - gamma_low <= tol_gamma * max(1.0, gamma_high):
            break

    return {
        "gamma_star": float(gamma_high),
        "gamma_low": float(gamma_low),
        "gamma_high": float(gamma_high),
        "history": history,
    }

def plot_outer_bisection_real_data(
    A,
    M,
    bisect_info=None,
    omega_min=1e-6,
    omega_max=5e1,
    number_of_points=400,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="04_outer_bisection_real_data",
):
    """
    Plot the OUTER bisection using the real sampled gain curve.
    IMPORTANT: uses the full signed frequency sweep, not only positive omega.
    """
    if not IS_WORLD_ROOT:
        return None

    hinf_root_mkdir(outdir)

    if bisect_info is None:
        bisect_info = outer_bisection_with_trace(A, M)

    gamma_star = float(bisect_info["gamma_star"])
    gamma_low_final = float(bisect_info["gamma_low"])
    gamma_high_final = float(bisect_info["gamma_high"])
    history = bisect_info["history"]

    # Full signed frequency sweep: DO NOT discard negative frequencies.
    frequencies, gains = compute_openloop_gain_sweep(
        A, M,
        omega_min=omega_min,
        omega_max=omega_max,
        number_of_points=number_of_points,
    )

    frequencies = np.asarray(frequencies, dtype=float)
    gains = np.asarray(gains, dtype=float)

    finite = np.isfinite(gains)
    frequencies_plot = frequencies[finite]
    gains_plot = gains[finite]

    if gains_plot.size == 0:
        return None

    peak_idx = int(np.argmax(gains_plot))
    omega_peak = float(frequencies_plot[peak_idx])
    gain_peak = float(gains_plot[peak_idx])

    # symlog x-axis so we can show negative, zero, and positive omega nicely
    nonzero = np.abs(frequencies_plot[np.abs(frequencies_plot) > 0.0])
    linthresh = max(float(np.min(nonzero)) if nonzero.size else 1e-6, 1e-12)

    # History arrays
    n_hist = len(history)
    xhist = np.arange(n_hist, dtype=int)

    gamma_low_hist = np.array([h["gamma_low"] for h in history], dtype=float)
    gamma_high_hist = np.array([h["gamma_high"] for h in history], dtype=float)
    gamma_trial_hist = np.array([h["gamma_trial"] for h in history], dtype=float)
    violates_hist = np.array([bool(h["violates"]) for h in history], dtype=bool)
    phase_hist = [h["phase"] for h in history]

    n_bracket = sum(1 for p in phase_hist if p == "bracket")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # ------------------------------------------------------------------
    # Left panel: real gain curve + trial gamma levels
    # ------------------------------------------------------------------
    ax1.plot(
        frequencies_plot,
        gains_plot,
        linewidth=2.0,
        label=r"sampled $\|G(i\omega)\|$"
    )

    ax1.scatter(
        [omega_peak],
        [gain_peak],
        s=55,
        zorder=5,
        label=fr"sampled peak: $\omega={omega_peak:.3g}$"
    )

    # Draw all tested gamma levels faintly
    for h in history:
        ax1.axhline(
            h["gamma_trial"],
            linewidth=0.8,
            alpha=0.15,
            color="gray",
        )

    # Highlight final bracket
    ax1.axhline(
        gamma_low_final,
        linestyle="--",
        linewidth=2.0,
        label=fr"final $\gamma_{{low}}={gamma_low_final:.4g}$"
    )
    ax1.axhline(
        gamma_high_final,
        linestyle="-",
        linewidth=2.2,
        label=fr"final $\gamma_{{high}}={gamma_high_final:.4g}$"
    )

    ax1.set_xscale("symlog", linthresh=linthresh)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"frequency $\omega$")
    ax1.set_ylabel(r"gain $\|G(i\omega)\|$")
    ax1.set_title(r"Outer bisection shown on the real gain curve")
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=9)

    # ------------------------------------------------------------------
    # Right panel: gamma bracket shrinking by iteration
    # ------------------------------------------------------------------
    ax2.fill_between(
        xhist,
        gamma_low_hist,
        gamma_high_hist,
        alpha=0.18,
        label=r"current bracket $[\gamma_{low}, \gamma_{high}]$"
    )

    ax2.plot(xhist, gamma_low_hist, linewidth=1.8, label=r"$\gamma_{low}$")
    ax2.plot(xhist, gamma_high_hist, linewidth=1.8, label=r"$\gamma_{high}$")
    ax2.scatter(xhist, gamma_trial_hist, s=22, zorder=4, label=r"trial $\gamma$")

    # Mark which trials were too low / high enough
    if np.any(violates_hist):
        ax2.scatter(
            xhist[violates_hist],
            gamma_trial_hist[violates_hist],
            marker="x",
            s=40,
            zorder=5,
            label=r"trial too low"
        )
    if np.any(~violates_hist):
        ax2.scatter(
            xhist[~violates_hist],
            gamma_trial_hist[~violates_hist],
            marker="o",
            s=20,
            facecolors="none",
            zorder=5,
            label=r"trial high enough"
        )

    if 0 < n_bracket < n_hist:
        ax2.axvline(
            n_bracket - 0.5,
            linestyle=":",
            linewidth=1.5,
            label="start of bisection"
        )

    ax2.set_yscale("log")
    ax2.set_xlabel("outer iteration")
    ax2.set_ylabel(r"gain level $\gamma$")
    ax2.set_title(r"Outer bisection convergence")
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)
    ax2.legend(loc="best", fontsize=9)

    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    return {
        "png": png,
        "pdf": pdf,
        "gamma_star": gamma_star,
        "gamma_low": gamma_low_final,
        "gamma_high": gamma_high_final,
        "omega_peak": omega_peak,
        "gain_peak": gain_peak,
    }

bisect_info = outer_bisection_with_trace(A, M)

outer_plot = plot_outer_bisection_real_data(
    A,
    M,
    bisect_info=bisect_info,
    omega_min=1e-1,
    omega_max=1e1,
    number_of_points=500,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
)

if IS_WORLD_ROOT:
    print("outer plot written to:", outer_plot)


from scipy.optimize import minimize_scalar

def local_max_gain(A, M, lo=-0.655, hi=-0.635):
    res = minimize_scalar(
        lambda w: -gain_of_omega(A, M, float(w)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-9}
    )

    w_star = float(res.x)
    g_star = float(-res.fun)




# =============================================================================
# Boyd H-infinity visual diagnostics
# Paste after your current Boyd / gain functions.
# Produces:
#   05_boyd_gamma_convergence.png
#   06_boyd_bracket_width.png
#   07_boyd_frequency_search.png
#   08_boyd_interval_focus.png
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt


def _root_only():
    return "IS_WORLD_ROOT" not in globals() or IS_WORLD_ROOT


def _ensure_dir(outdir):
    if _root_only():
        os.makedirs(outdir, exist_ok=True)
    if "WORLD" in globals():
        WORLD.Barrier()


def boyd_inner_check_visual(
    A,
    M,
    gamma,
    b=10.0,
    eigs_tol=1e-6,
    nev=16,
    max_visits=200,
    probe_points=21,
    probe_as_oracle=True,
    candidate_count=8,
    eps_candidate=1e-5,
):
    """
    Visual version of the Boyd inner test.

    Returns
    -------
    violates : bool
        True  means gamma is too low: max gain > gamma.
        False means this inner test certified / did not find a violation.

    trace : dict
        Contains all interval visits, midpoint gains, nearest pencil eigenvalues,
        interval-probe data, and candidate eigenvalue probes.

    Important:
    - probe_as_oracle=True uses interval sampling to detect narrow peaks.
      This is useful for explaining convergence to the peak.
    - For a purer Boyd-only trace, set probe_as_oracle=False.
    """

    gamma = float(gamma)
    b = float(b)
    min_interval = float(eigs_tol)

    N = build_N_from_M(M)
    Mgam = build_Mgamma(A, M, gamma)

    N = N.convert("aij")
    N.assemble()
    Mgam = Mgam.convert("aij")
    Mgam.assemble()

    trace = {
        "gamma": gamma,
        "b": b,
        "violates": False,
        "reason": "not_finished",
        "hit_omega": np.nan,
        "hit_gain": np.nan,
        "events": [],
    }

    # Low-frequency guard. Keep it visible in the trace.
    guard_ws = [0.0, -1e-12, 1e-12, -1e-9, 1e-9, -1e-6, 1e-6]
    guard = []
    for w in guard_ws:
        g = float(gain_of_omega(A, M, float(w)))
        guard.append({"omega": float(w), "gain": g})
        if g > gamma:
            trace.update({
                "violates": True,
                "reason": "low_frequency_guard",
                "hit_omega": float(w),
                "hit_gain": g,
                "guard": guard,
            })
            return True, trace

    trace["guard"] = guard

    intervals = [(-b, 0.0), (0.0, b)]
    visit = 0

    while intervals and visit < max_visits:
        intervals = prune_intervals(intervals, min_interval)
        if not intervals:
            break

        # Boyd-style: inspect the largest remaining interval first.
        intervals.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        lo, hi = intervals.pop(0)

        if hi <= lo:
            continue

        theta = 0.5 * (lo + hi)
        g_theta = float(gain_of_omega(A, M, theta))

        event = {
            "visit": int(visit),
            "lo": float(lo),
            "hi": float(hi),
            "width": float(hi - lo),
            "theta": float(theta),
            "gain_theta": float(g_theta),
            "nconv": 0,
            "r": np.nan,
            "removed_lo": np.nan,
            "removed_hi": np.nan,
            "nearest_real": np.nan,
            "nearest_imag": np.nan,
            "probe_best_omega": np.nan,
            "probe_best_gain": np.nan,
            "candidate_best_omega": np.nan,
            "candidate_best_gain": np.nan,
            "action": "",
            "remaining_after": np.nan,
        }

        # Diagnostic sampling inside the interval.
        # This is what makes the plot show convergence toward the actual peak.
        if probe_points is not None and probe_points > 1:
            probe_ws = np.linspace(lo, hi, int(probe_points))
            probe_gs = np.array(
                [float(gain_of_omega(A, M, float(w))) for w in probe_ws],
                dtype=float,
            )
            jmax = int(np.nanargmax(probe_gs))
            event["probe_best_omega"] = float(probe_ws[jmax])
            event["probe_best_gain"] = float(probe_gs[jmax])

            if probe_as_oracle and probe_gs[jmax] > gamma:
                event["action"] = "interval_probe_found_gain_above_gamma"
                trace["events"].append(event)
                trace.update({
                    "violates": True,
                    "reason": event["action"],
                    "hit_omega": float(probe_ws[jmax]),
                    "hit_gain": float(probe_gs[jmax]),
                })
                return True, trace

        # Midpoint violation check.
        if g_theta > gamma:
            event["action"] = "midpoint_gain_above_gamma"
            trace["events"].append(event)
            trace.update({
                "violates": True,
                "reason": event["action"],
                "hit_omega": float(theta),
                "hit_gain": float(g_theta),
            })
            return True, trace

        # Boyd pencil solve near i*theta.
        lambdas, nconv = eigs_close_to_shift_pencil(
            N, Mgam, theta, nev=nev, tol=eigs_tol
        )

        event["nconv"] = int(nconv)

        if nconv == 0 or lambdas.size == 0:
            # No spectral info. Remove only a tiny midpoint neighborhood.
            remaining = split_by_removed_middle(
                lo, hi, theta, r=0.0, min_progress=0.5 * min_interval
            )
            event["removed_lo"] = float(theta)
            event["removed_hi"] = float(theta)
            event["action"] = "no_eigenvalues_returned"
            event["remaining_after"] = int(len(intervals) + len(remaining))
            intervals += remaining
            trace["events"].append(event)
            visit += 1
            continue

        shift = 1j * theta
        lambdas = np.asarray(lambdas, dtype=complex)
        lambdas = lambdas[np.argsort(np.abs(lambdas - shift))]

        nearest = lambdas[0]
        r = float(abs(nearest - shift))

        event["r"] = r
        event["nearest_real"] = float(nearest.real)
        event["nearest_imag"] = float(nearest.imag)

        # Probe the frequencies suggested by the nearest pencil eigenvalues.
        # This is a useful visual bridge between the pencil and the gain peak.
        best_candidate_w = np.nan
        best_candidate_g = -np.inf

        for lam in lambdas[:min(len(lambdas), candidate_count)]:
            wc = float(lam.imag)
            for h in (-eps_candidate, 0.0, eps_candidate):
                w_probe = wc + float(h)
                g_probe = float(gain_of_omega(A, M, w_probe))

                if g_probe > best_candidate_g:
                    best_candidate_g = g_probe
                    best_candidate_w = w_probe

                if g_probe > gamma:
                    event["candidate_best_omega"] = float(best_candidate_w)
                    event["candidate_best_gain"] = float(best_candidate_g)
                    event["action"] = "candidate_eigenvalue_probe_above_gamma"
                    trace["events"].append(event)
                    trace.update({
                        "violates": True,
                        "reason": event["action"],
                        "hit_omega": float(w_probe),
                        "hit_gain": float(g_probe),
                    })
                    return True, trace

        if np.isfinite(best_candidate_g):
            event["candidate_best_omega"] = float(best_candidate_w)
            event["candidate_best_gain"] = float(best_candidate_g)

        # Boyd exclusion: remove the disk intersection with the imaginary axis.
        if np.isfinite(r) and r > 0.0 and r <= 0.5 * (hi - lo):
            r_eff = max(r, 0.5 * min_interval)
            removed_lo = max(lo, theta - r_eff)
            removed_hi = min(hi, theta + r_eff)

            remaining = split_by_removed_middle(
                lo, hi, theta, r=r, min_progress=0.5 * min_interval
            )

            event["removed_lo"] = float(removed_lo)
            event["removed_hi"] = float(removed_hi)
            event["action"] = "removed_certified_disk"
            event["remaining_after"] = int(len(intervals) + len(remaining))
            intervals += remaining
        else:
            # If the nearest eigenvalue is farther than half the interval,
            # the whole interval is safe.
            event["removed_lo"] = float(lo)
            event["removed_hi"] = float(hi)
            event["action"] = "whole_interval_excluded"
            event["remaining_after"] = int(len(intervals))

        trace["events"].append(event)
        visit += 1

    trace.update({
        "violates": False,
        "reason": "intervals_exhausted_or_visit_limit",
    })
    return False, trace


def outer_bisection_with_visual_trace(
    A,
    M,
    tol_gamma=1e-8,
    tol_imag=1e-10,
    max_bisect=80,
    b=10.0,
    eigs_tol=1e-6,
    nev=16,
    probe_points=21,
    probe_as_oracle=True,
):
    """
    Outer bisection with full history.

    Convention:
    - violates=True  => gamma is too small, so gamma_low moves up.
    - violates=False => gamma is high enough, so gamma_high moves down.
    """

    gamma_low = 0.0
    gamma_high = 1e-6

    history = []
    traces = []

    # Bracketing phase.
    k = 0
    while True:
        violates, trace = boyd_inner_check_visual(
            A, M, gamma_high,
            b=b,
            eigs_tol=eigs_tol,
            nev=nev,
            probe_points=probe_points,
            probe_as_oracle=probe_as_oracle,
        )

        history.append({
            "global_iter": len(history),
            "phase": "bracket",
            "iter": k,
            "gamma_trial": float(gamma_high),
            "gamma_low_before": float(gamma_low),
            "gamma_high_before": float(gamma_high),
            "violates": bool(violates),
            "reason": trace["reason"],
            "hit_omega": float(trace.get("hit_omega", np.nan)),
            "hit_gain": float(trace.get("hit_gain", np.nan)),
        })
        traces.append(trace)

        if not violates:
            break

        gamma_low = gamma_high
        gamma_high *= 2.0
        k += 1

    # Bisection phase.
    for k in range(max_bisect):
        gamma = 0.5 * (gamma_low + gamma_high)

        violates, trace = boyd_inner_check_visual(
            A, M, gamma,
            b=b,
            eigs_tol=eigs_tol,
            nev=nev,
            probe_points=probe_points,
            probe_as_oracle=probe_as_oracle,
        )

        gamma_low_before = gamma_low
        gamma_high_before = gamma_high

        if violates:
            gamma_low = gamma
        else:
            gamma_high = gamma

        history.append({
            "global_iter": len(history),
            "phase": "bisect",
            "iter": k,
            "gamma_trial": float(gamma),
            "gamma_low_before": float(gamma_low_before),
            "gamma_high_before": float(gamma_high_before),
            "gamma_low_after": float(gamma_low),
            "gamma_high_after": float(gamma_high),
            "bracket_width": float(gamma_high - gamma_low),
            "violates": bool(violates),
            "reason": trace["reason"],
            "hit_omega": float(trace.get("hit_omega", np.nan)),
            "hit_gain": float(trace.get("hit_gain", np.nan)),
        })
        traces.append(trace)

        if gamma_high - gamma_low <= tol_gamma * max(1.0, gamma_high):
            break

    result = {
        "gamma_star": float(gamma_high),
        "gamma_low": float(gamma_low),
        "gamma_high": float(gamma_high),
        "history": history,
        "traces": traces,
        "last_trace": traces[-1] if traces else None,
    }

    if _root_only():
        print(
            "[Boyd visual] "
            f"gamma_star={result['gamma_star']:.15g}, "
            f"gamma_low={result['gamma_low']:.15g}, "
            f"gamma_high={result['gamma_high']:.15g}, "
            f"n_outer={len(history)}",
            flush=True,
        )

    return result


def plot_boyd_gamma_convergence(result, outdir, basename="05_boyd_gamma_convergence"):
    """Plot gamma_low, gamma_high, and trial gamma during the outer bisection."""

    if not _root_only():
        return None

    _ensure_dir(outdir)

    hist = result["history"]
    xs = np.arange(len(hist))

    gamma_trial = np.array([h["gamma_trial"] for h in hist], dtype=float)
    gamma_low = np.array([
        h.get("gamma_low_after", h.get("gamma_low_before", np.nan))
        for h in hist
    ], dtype=float)
    gamma_high = np.array([
        h.get("gamma_high_after", h.get("gamma_high_before", np.nan))
        for h in hist
    ], dtype=float)
    violates = np.array([h["violates"] for h in hist], dtype=bool)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    ax.plot(xs, gamma_low, marker=".", label=r"$\gamma_{low}$")
    ax.plot(xs, gamma_high, marker=".", label=r"$\gamma_{high}$")
    ax.scatter(xs[violates], gamma_trial[violates], marker="x", s=55, label=r"trial too low")
    ax.scatter(xs[~violates], gamma_trial[~violates], marker="o", s=35, label=r"trial certified")

    ax.axhline(result["gamma_star"], linestyle="--", linewidth=1.5,
               label=fr"final $\gamma_\star={result['gamma_star']:.6g}$")

    ax.set_xlabel("outer iteration")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(r"Outer bisection convergence of $\gamma$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    return {"png": png, "pdf": pdf}


def plot_boyd_bracket_width(result, outdir, basename="06_boyd_bracket_width"):
    """Plot gamma_high - gamma_low on a log scale."""

    if not _root_only():
        return None

    _ensure_dir(outdir)

    hist = [h for h in result["history"] if "bracket_width" in h]
    if len(hist) == 0:
        return None

    xs = np.arange(len(hist))
    widths = np.array([h["bracket_width"] for h in hist], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.semilogy(xs, widths, marker=".")
    ax.set_xlabel("bisection iteration")
    ax.set_ylabel(r"$\gamma_{high}-\gamma_{low}$")
    ax.set_title("Bisection bracket width")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    return {"png": png, "pdf": pdf}


def _events_from_selected_traces(result, mode="all"):
    """
    Collect inner events from traces.

    mode:
      - "all": all outer iterations, can be busy.
      - "last": only final trace.
      - "last_violating": last trace where gamma was too low.
    """
    traces = result.get("traces", [])
    history = result.get("history", [])

    if mode == "last":
        return result["last_trace"].get("events", []) if result.get("last_trace") else []

    if mode == "last_violating":
        for h, tr in zip(reversed(history), reversed(traces)):
            if h.get("violates", False):
                return tr.get("events", [])
        return []

    events = []
    for tr in traces:
        events.extend(tr.get("events", []))
    return events


def plot_boyd_frequency_search(
    result,
    outdir,
    frequencies=None,
    gains=None,
    true_omega=None,
    true_gain=None,
    mode="all",
    basename="07_boyd_frequency_search",
):
    """
    Overlay the frequency response with all Boyd midpoint/probe frequencies.

    For a clean presentation:
      mode="last_violating" often shows the final search around the peak.
      mode="all" shows the whole algorithm.
    """

    if not _root_only():
        return None

    _ensure_dir(outdir)

    events = _events_from_selected_traces(result, mode=mode)
    if len(events) == 0:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    if frequencies is not None and gains is not None:
        frequencies = np.asarray(frequencies, dtype=float)
        gains = np.asarray(gains, dtype=float)
        finite = np.isfinite(frequencies) & np.isfinite(gains)
        ax.plot(frequencies[finite], gains[finite], linewidth=1.2,
                label=r"sampled $\|G(i\omega)\|$")

    theta = np.array([e["theta"] for e in events], dtype=float)
    g_theta = np.array([e["gain_theta"] for e in events], dtype=float)

    ax.scatter(theta, g_theta, s=28, marker="o", label="Boyd midpoint checks")

    probe_w = np.array([e.get("probe_best_omega", np.nan) for e in events], dtype=float)
    probe_g = np.array([e.get("probe_best_gain", np.nan) for e in events], dtype=float)
    ok_probe = np.isfinite(probe_w) & np.isfinite(probe_g)
    if np.any(ok_probe):
        ax.scatter(probe_w[ok_probe], probe_g[ok_probe], s=45, marker="x",
                   label="best interval probes")

    cand_w = np.array([e.get("candidate_best_omega", np.nan) for e in events], dtype=float)
    cand_g = np.array([e.get("candidate_best_gain", np.nan) for e in events], dtype=float)
    ok_cand = np.isfinite(cand_w) & np.isfinite(cand_g)
    if np.any(ok_cand):
        ax.scatter(cand_w[ok_cand], cand_g[ok_cand], s=35, marker="+",
                   label="eigenvalue-imag probes")

    ax.axhline(result["gamma_star"], linestyle="--", linewidth=1.5,
               label=fr"final $\gamma_\star={result['gamma_star']:.6g}$")

    if true_omega is not None and true_gain is not None:
        ax.scatter([true_omega], [true_gain], s=75, zorder=6,
                   label=fr"optimized peak $\omega={true_omega:.4g}$")

    ax.set_yscale("log")
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel(r"gain $\|G(i\omega)\|$")
    ax.set_title("How Boyd's checks move toward the gain peak")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    return {"png": png, "pdf": pdf}


def plot_boyd_interval_focus(
    result,
    outdir,
    mode="last_violating",
    basename="08_boyd_interval_focus",
):
    """
    Plot interval visits as horizontal bars.
    This shows how the inner algorithm removes certified regions and focuses near the peak.
    """

    if not _root_only():
        return None

    _ensure_dir(outdir)

    events = _events_from_selected_traces(result, mode=mode)
    if len(events) == 0:
        return None

    fig_h = max(4.5, min(14.0, 0.26 * len(events) + 2.5))
    fig, ax = plt.subplots(figsize=(10.0, fig_h))

    for k, e in enumerate(events):
        y = k
        lo = float(e["lo"])
        hi = float(e["hi"])
        theta = float(e["theta"])

        ax.hlines(y, lo, hi, linewidth=2.0, alpha=0.65,
                  label="visited interval" if k == 0 else None)
        ax.scatter([theta], [y], s=28, zorder=5,
                   label=r"shift $\theta$" if k == 0 else None)

        removed_lo = e.get("removed_lo", np.nan)
        removed_hi = e.get("removed_hi", np.nan)
        if np.isfinite(removed_lo) and np.isfinite(removed_hi):
            ax.hlines(y + 0.18, removed_lo, removed_hi, linewidth=6.0, alpha=0.45,
                      label="excluded by disk" if k == 0 else None)

        probe_w = e.get("probe_best_omega", np.nan)
        if np.isfinite(probe_w):
            ax.scatter([probe_w], [y - 0.18], marker="x", s=35,
                       label="best sampled point in interval" if k == 0 else None)

        ax.text(hi, y, "  " + str(e.get("action", "")),
                va="center", fontsize=8, alpha=0.8)

    ax.axvline(0.0, linestyle="--", linewidth=1.0)

    # Optional: zoom automatically around the active intervals if they are narrow.
    all_los = np.array([e["lo"] for e in events], dtype=float)
    all_his = np.array([e["hi"] for e in events], dtype=float)
    if np.nanmax(all_his) - np.nanmin(all_los) < 4.0:
        pad = 0.05 * (np.nanmax(all_his) - np.nanmin(all_los) + 1e-12)
        ax.set_xlim(np.nanmin(all_los) - pad, np.nanmax(all_his) + pad)

    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel("inner visit")
    ax.set_title("Boyd inner interval exclusion / focus toward the peak")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    ax.invert_yaxis()
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    return {"png": png, "pdf": pdf}


def make_boyd_visual_report(
    A,
    M,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    b=10.0,
    wmin=1e-1,
    wmax=1e1,
    npos=500,
    true_omega=None,
    true_gain=None,
    tol_gamma=1e-8,
    eigs_tol=1e-6,
    nev=16,
    max_bisect=80,
    probe_points=21,
    probe_as_oracle=True,
):
    """
    Convenience function: run traced Boyd outer bisection and make the plots.

    Set true_omega and true_gain to your optimized local max:
        true_omega = -0.645763204247214
        true_gain  = 7201.04835813522
    """

    _ensure_dir(outdir)

    result = outer_bisection_with_visual_trace(
        A,
        M,
        tol_gamma=tol_gamma,
        max_bisect=max_bisect,
        b=b,
        eigs_tol=eigs_tol,
        nev=nev,
        probe_points=probe_points,
        probe_as_oracle=probe_as_oracle,
    )

    # Use your existing sweep helper if it exists.
    frequencies = None
    gains = None
    if "compute_openloop_gain_sweep" in globals():
        frequencies, gains = compute_openloop_gain_sweep(
            A, M,
            omega_min=wmin,
            omega_max=wmax,
            number_of_points=npos,
        )

    paths = {}
    paths["gamma_convergence"] = plot_boyd_gamma_convergence(result, outdir)
    paths["bracket_width"] = plot_boyd_bracket_width(result, outdir)

    paths["frequency_search_all"] = plot_boyd_frequency_search(
        result,
        outdir,
        frequencies=frequencies,
        gains=gains,
        true_omega=true_omega,
        true_gain=true_gain,
        mode="all",
        basename="07_boyd_frequency_search_all",
    )

    paths["frequency_search_last_violating"] = plot_boyd_frequency_search(
        result,
        outdir,
        frequencies=frequencies,
        gains=gains,
        true_omega=true_omega,
        true_gain=true_gain,
        mode="last_violating",
        basename="07_boyd_frequency_search_last_violating",
    )

    paths["interval_focus"] = plot_boyd_interval_focus(
        result,
        outdir,
        mode="last_violating",
        basename="08_boyd_interval_focus_last_violating",
    )

    if _root_only():
        print("\nBoyd visual report written to:", outdir, flush=True)
        print("gamma_star:", result["gamma_star"], flush=True)
        for k, v in paths.items():
            print(" ", k, "->", v, flush=True)

    return result, paths


result, paths = make_boyd_visual_report(
    A,
    M,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    b=10.0,
    wmin=1e-1,
    wmax=1e1,
    npos=500,
    true_omega=-0.645763204247214,
    true_gain=7201.04835813522,
    tol_gamma=1e-8,
    eigs_tol=1e-6,
    nev=16,
    max_bisect=80,
    probe_points=21,
    probe_as_oracle=True,
)


# =============================================================================
# Boyd gamma bisection plot
# Paste this block after outer_bisection(...), once inner_check, gain_of_omega,
# mpi_task_map, and the MPI/root globals have been defined.
# =============================================================================

def _boyd_plot_is_root():
    return ("IS_WORLD_ROOT" not in globals()) or IS_WORLD_ROOT


def _boyd_plot_barrier():
    if "WORLD" in globals():
        WORLD.Barrier()


def _boyd_plot_mkdir(outdir):
    if _boyd_plot_is_root():
        os.makedirs(outdir, exist_ok=True)
    _boyd_plot_barrier()


def _boyd_signed_log_grid(omega_min=1e-1, omega_max=1e1, npos=300):
    omega_min = float(omega_min)
    omega_max = float(omega_max)
    npos = int(npos)

    if omega_min <= 0.0 or omega_max <= omega_min:
        raise ValueError("Need 0 < omega_min < omega_max.")

    wp = np.logspace(np.log10(omega_min), np.log10(omega_max), npos)
    return np.concatenate((-wp[::-1], [0.0], wp))


def _boyd_gain_sweep_for_plot(A, M, omega_min=1e-1, omega_max=1e1, npos=300):
    """Collective over WORLD if mpi_task_map is available."""

    omegas = _boyd_signed_log_grid(omega_min, omega_max, npos)

    def _one_gain(w):
        w = float(w)
        return w, float(gain_of_omega(A, M, w))

    if "mpi_task_map" in globals():
        pairs = mpi_task_map(
            _one_gain,
            omegas,
            label="boyd_gamma_plot_gain_sweep",
            use_parallel=True,
        )
    else:
        pairs = [_one_gain(w) for w in omegas]

    pairs = sorted(pairs, key=lambda p: p[0])
    ws = np.array([p[0] for p in pairs], dtype=float)
    gs = np.array([p[1] for p in pairs], dtype=float)
    return ws, gs


def outer_bisection_gamma_history(
    A,
    M,
    tol_gamma=1e-8,
    tol_imag=1e-10,
    max_bisect=12,
    max_bracket=80,
):
    """
    Same logic as outer_bisection(...), but records every gamma tested.

    Convention:
      violates=True  -> gamma was too low, so gamma_low moves up.
      violates=False -> gamma was high enough, so gamma_high moves down.
    """

    gamma_low = 0.0
    gamma_high = 1e-6
    history = []

    # 1) Bracket the norm: increase gamma_high until the inner test passes.
    for k in range(int(max_bracket)):
        gamma_low_before = gamma_low
        gamma_high_before = gamma_high

        violates = bool(inner_check(gamma_high, A, M, tol_imag=tol_imag))

        if violates:
            gamma_low = gamma_high
            gamma_high = 2.0 * gamma_high

        history.append({
            "global_iter": len(history),
            "phase": "bracket",
            "iter": k,
            "gamma_trial": float(gamma_high_before),
            "gamma_low_before": float(gamma_low_before),
            "gamma_high_before": float(gamma_high_before),
            "gamma_low_after": float(gamma_low),
            "gamma_high_after": float(gamma_high),
            "bracket_width": float(gamma_high - gamma_low),
            "violates": bool(violates),
        })

        if not violates:
            break
    else:
        raise RuntimeError("Could not bracket the Hinf norm; increase max_bracket.")

    # 2) Bisection: squeeze gamma_low and gamma_high.
    for k in range(int(max_bisect)):
        gamma_low_before = gamma_low
        gamma_high_before = gamma_high
        gamma_trial = 0.5 * (gamma_low + gamma_high)

        violates = bool(inner_check(gamma_trial, A, M, tol_imag=tol_imag))

        if violates:
            gamma_low = gamma_trial
        else:
            gamma_high = gamma_trial

        history.append({
            "global_iter": len(history),
            "phase": "bisect",
            "iter": k,
            "gamma_trial": float(gamma_trial),
            "gamma_low_before": float(gamma_low_before),
            "gamma_high_before": float(gamma_high_before),
            "gamma_low_after": float(gamma_low),
            "gamma_high_after": float(gamma_high),
            "bracket_width": float(gamma_high - gamma_low),
            "violates": bool(violates),
        })

        if gamma_high - gamma_low <= tol_gamma * max(1.0, gamma_high):
            break

    return {
        "gamma_star": float(gamma_high),
        "gamma_low": float(gamma_low),
        "gamma_high": float(gamma_high),
        "history": history,
    }


def plot_boyd_gamma_history_on_gain_curve(
    A,
    M,
    result=None,
    omega_min=1e-1,
    omega_max=1e1,
    npos=300,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="boyd_gamma_iterations",
    max_gamma_lines=10,
):
    """
    Makes a two-panel plot:
      left  = sampled gain curve with trial gamma levels
      right = gamma_low/gamma_high bracket over Boyd outer iterations

    Important MPI rule:
      all ranks enter this function;
      only root writes the PNG/PDF.
    """

    if result is None:
        result = outer_bisection_gamma_history(A, M)

    ws, gs = _boyd_gain_sweep_for_plot(A, M, omega_min, omega_max, npos)

    _boyd_plot_mkdir(outdir)

    if not _boyd_plot_is_root():
        return result, None

    hist = result["history"]

    finite = np.isfinite(ws) & np.isfinite(gs)
    ws_plot = ws[finite]
    gs_plot = gs[finite]

    peak_idx = int(np.nanargmax(gs_plot))
    omega_peak = float(ws_plot[peak_idx])
    gain_peak = float(gs_plot[peak_idx])

    x = np.arange(len(hist))
    gamma_trial = np.array([h["gamma_trial"] for h in hist], dtype=float)
    gamma_low = np.array([h["gamma_low_after"] for h in hist], dtype=float)
    gamma_high = np.array([h["gamma_high_after"] for h in hist], dtype=float)
    violates = np.array([h["violates"] for h in hist], dtype=bool)
    phases = [h["phase"] for h in hist]

    # For symlog, pick a sensible linear window around zero.
    nz = np.abs(ws_plot[np.abs(ws_plot) > 0.0])
    linthresh = max(float(np.min(nz)) if nz.size else 1e-6, 1e-12)

    fig, (ax_gain, ax_gamma) = plt.subplots(1, 2, figsize=(14.0, 5.5))

    # -------------------------------------------------------------------------
    # Left: actual sampled frequency response with trial gamma horizontal lines.
    # -------------------------------------------------------------------------
    ax_gain.plot(ws_plot, gs_plot, linewidth=1.8, label=r"sampled $\|G(i\omega)\|$")
    ax_gain.scatter([omega_peak], [gain_peak], s=55, zorder=5,
                    label=fr"sampled peak: $\omega={omega_peak:.4g}$")

    # Draw only the last few gamma trials so the picture stays readable.
    start = max(0, len(hist) - int(max_gamma_lines))
    for j, h in enumerate(hist[start:], start=start):
        alpha = 0.15 + 0.55 * (j - start + 1) / max(1, len(hist[start:]))
        ax_gain.axhline(h["gamma_trial"], linewidth=1.0, alpha=alpha)

    ax_gain.axhline(result["gamma_low"], linestyle="--", linewidth=2.0,
                    label=fr"final $\gamma_{{low}}={result['gamma_low']:.5g}$")
    ax_gain.axhline(result["gamma_high"], linestyle="-", linewidth=2.0,
                    label=fr"final $\gamma_{{high}}={result['gamma_high']:.5g}$")

    ax_gain.set_xscale("symlog", linthresh=linthresh)
    ax_gain.set_yscale("log")
    ax_gain.set_xlabel(r"frequency $\omega$")
    ax_gain.set_ylabel(r"gain $\|G(i\omega)\|$")
    ax_gain.set_title(r"Trial $\gamma$ levels on the gain curve")
    ax_gain.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_gain.legend(loc="best", fontsize=9)

    # -------------------------------------------------------------------------
    # Right: gamma bracket shrinkage.
    # -------------------------------------------------------------------------
    ax_gamma.fill_between(x, gamma_low, gamma_high, alpha=0.18,
                          label=r"bracket $[\gamma_{low},\gamma_{high}]$")
    ax_gamma.plot(x, gamma_low, marker=".", label=r"$\gamma_{low}$")
    ax_gamma.plot(x, gamma_high, marker=".", label=r"$\gamma_{high}$")

    ax_gamma.scatter(x[violates], gamma_trial[violates], marker="x", s=55,
                     label=r"trial too low: $\gamma < \|G\|_\infty$")
    ax_gamma.scatter(x[~violates], gamma_trial[~violates], marker="o", s=35,
                     facecolors="none",
                     label=r"trial high enough: $\gamma \geq \|G\|_\infty$")

    n_bracket = sum(p == "bracket" for p in phases)
    if 0 < n_bracket < len(hist):
        ax_gamma.axvline(n_bracket - 0.5, linestyle=":", linewidth=1.5,
                         label="start bisection")

    ax_gamma.axhline(result["gamma_high"], linestyle="--", linewidth=1.3,
                     label=fr"returned $\gamma_\star={result['gamma_high']:.5g}$")

    ax_gamma.set_yscale("log")
    ax_gamma.set_xlabel("outer iteration")
    ax_gamma.set_ylabel(r"$\gamma$")
    ax_gamma.set_title(r"Boyd outer bisection: how $\gamma$ moves")
    ax_gamma.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_gamma.legend(loc="best", fontsize=8)

    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    paths = {
        "png": png,
        "pdf": pdf,
        "gamma_star": result["gamma_high"],
        "gamma_low": result["gamma_low"],
        "omega_peak_sampled": omega_peak,
        "gain_peak_sampled": gain_peak,
        "n_gamma_tests": len(hist),
    }

    print("\nBoyd gamma plot written to:", paths, flush=True)
    return result, paths


# Optional run switch. This prevents your expensive script from always plotting.
if 1==1:
    boyd_gamma_result = outer_bisection_gamma_history(
        A,
        M,
        tol_gamma=float(os.environ.get("BOYD_GAMMA_TOL", "1e-8")),
        tol_imag=float(os.environ.get("BOYD_GAMMA_TOL_IMAG", "1e-10")),
        max_bisect=int(os.environ.get("BOYD_GAMMA_MAX_BISECT", "12")),
        max_bracket=int(os.environ.get("BOYD_GAMMA_MAX_BRACKET", "80")),
    )

    boyd_gamma_result, boyd_gamma_paths = plot_boyd_gamma_history_on_gain_curve(
        A,
        M,
        result=boyd_gamma_result,
        omega_min=float(os.environ.get("BOYD_GAMMA_WMIN", "1e-1")),
        omega_max=float(os.environ.get("BOYD_GAMMA_WMAX", "1e1")),
        npos=int(os.environ.get("BOYD_GAMMA_NPOS", "300")),
        outdir=os.environ.get("BOYD_GAMMA_OUTDIR", "/stck/eclement/Control Of NavierStokes/plots"),
        basename=os.environ.get("BOYD_GAMMA_BASENAME", "boyd_gamma_iterations"),
        max_gamma_lines=int(os.environ.get("BOYD_GAMMA_MAX_LINES", "10")),
    )


# =============================================================================
# Simple Boyd gamma iteration plot
# Looks like the ChatGPT "Boyd-style gamma bisection" chart:
#   x-axis: outer iteration
#   curves: test gamma, lower bound, upper bound, estimated Hinf norm
#
# Paste this AFTER whichever function produces a result dictionary with
# result["history"], for example:
#   - outer_bisection_gamma_history(...)
#   - outer_bisection_with_visual_trace(...)
# =============================================================================

def _simple_boyd_is_root():
    return ("IS_WORLD_ROOT" not in globals()) or bool(IS_WORLD_ROOT)


def _simple_boyd_barrier():
    if "WORLD" in globals():
        WORLD.Barrier()


def _simple_boyd_ensure_dir(outdir):
    if _simple_boyd_is_root():
        os.makedirs(outdir, exist_ok=True)
    _simple_boyd_barrier()


def _simple_boyd_get_after_value(h, after_key, before_key=None):
    """Prefer *_after values, but fall back to *_before for bracketing rows."""
    if after_key in h and np.isfinite(float(h[after_key])):
        return float(h[after_key])
    if before_key is not None and before_key in h and np.isfinite(float(h[before_key])):
        return float(h[before_key])
    return np.nan


def plot_boyd_gamma_like_chatgpt(
    result,
    true_hinf=None,
    only_bisection=True,
    max_points=12,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="boyd_gamma_like_chatgpt",
):
    """
    Plot the simple gamma bisection history.

    Parameters
    ----------
    result : dict
        Must contain result["history"]. This is produced by either
        outer_bisection_gamma_history(...) or outer_bisection_with_visual_trace(...).

    true_hinf : float or None
        If you know a trusted Hinf estimate, pass it here.
        If None, the plot uses result["gamma_star"] or result["gamma_high"] and labels
        it as an estimate.

    only_bisection : bool
        True gives the clean chart like the example: just bisection iterations.
        False includes the initial bracketing/doubling tests too.

    max_points : int or None
        Keep the plot readable by showing the first max_points iterations.
        Set to None to show everything.
    """

    if result is None or "history" not in result:
        raise ValueError("result must be a dictionary containing result['history'].")

    hist_all = list(result["history"])
    if only_bisection:
        hist = [h for h in hist_all if h.get("phase", "bisect") == "bisect"]
    else:
        hist = hist_all

    if max_points is not None:
        hist = hist[:int(max_points)]

    if len(hist) == 0:
        raise ValueError("No history entries to plot. Try only_bisection=False.")

    # Use the returned upper bound as the final certified/estimated Hinf norm.
    if true_hinf is None:
        true_hinf = result.get("gamma_star", result.get("gamma_high", np.nan))
        true_label = r"estimated $\|G\|_\infty$"
    else:
        true_label = r"true / reference $\|G\|_\infty$"

    xs = np.arange(1, len(hist) + 1)

    gamma_trial = np.array([float(h["gamma_trial"]) for h in hist], dtype=float)
    gamma_low = np.array([
        _simple_boyd_get_after_value(h, "gamma_low_after", "gamma_low_before")
        for h in hist
    ], dtype=float)
    gamma_high = np.array([
        _simple_boyd_get_after_value(h, "gamma_high_after", "gamma_high_before")
        for h in hist
    ], dtype=float)
    hinf_line = np.full_like(gamma_trial, float(true_hinf), dtype=float)
    violates = np.array([bool(h.get("violates", False)) for h in hist], dtype=bool)

    _simple_boyd_ensure_dir(outdir)

    if not _simple_boyd_is_root():
        return None

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    # Main lines, intentionally simple like the ChatGPT chart.
    ax.plot(xs, gamma_trial, marker="o", linewidth=2.0, label=r"test $\gamma$")
    ax.plot(xs, gamma_low, marker=".", linewidth=2.0, label=r"lower bound")
    ax.plot(xs, gamma_high, marker=".", linewidth=2.0, label=r"upper bound")
    ax.plot(xs, hinf_line, linestyle="--", linewidth=2.0, label=true_label)

    # Make pass/fail visually obvious.
    if np.any(violates):
        ax.scatter(
            xs[violates],
            gamma_trial[violates],
            marker="x",
            s=80,
            zorder=5,
            label=r"too low: $\gamma < \|G\|_\infty$",
        )

    if np.any(~violates):
        ax.scatter(
            xs[~violates],
            gamma_trial[~violates],
            marker="o",
            s=70,
            facecolors="none",
            zorder=5,
            label=r"high enough: $\gamma \geq \|G\|_\infty$",
        )

    # Small labels above test points: fail/pass.
    for x, y, bad in zip(xs, gamma_trial, violates):
        ax.annotate(
            "low" if bad else "OK",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            alpha=0.85,
        )

    # If all values are positive, log scale usually makes bisection easier to see.
    all_y = np.concatenate([gamma_trial, gamma_low, gamma_high, hinf_line])
    all_y = all_y[np.isfinite(all_y)]
    if all_y.size and np.all(all_y > 0.0) and np.nanmax(all_y) / np.nanmin(all_y) > 20.0:
        ax.set_yscale("log")

    ax.set_xlabel("Boyd outer iteration")
    ax.set_ylabel(r"gain level $\gamma$")
    ax.set_title(r"Boyd-style $\gamma$ bisection for the H$\infty$ norm")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    info = {
        "png": png,
        "pdf": pdf,
        "n_points": int(len(hist)),
        "hinf_reference": float(true_hinf),
        "gamma_low_final": float(result.get("gamma_low", np.nan)),
        "gamma_high_final": float(result.get("gamma_high", result.get("gamma_star", np.nan))),
    }

    print("[Boyd simple gamma plot] written to:", info, flush=True)
    return info


# -----------------------------------------------------------------------------
# Optional switch: run this plot automatically.
#
# Usage examples:
#
#   MAKE_BOYD_SIMPLE_GAMMA_PLOT=1 mpirun -n 4 python your_script.py
#
#   MAKE_BOYD_SIMPLE_GAMMA_PLOT=1 BOYD_SIMPLE_POINTS=8 mpirun -n 4 python your_script.py
#
# If a history result already exists in globals(), this reuses it.
# Otherwise it computes one collectively using the available tracing function.
# -----------------------------------------------------------------------------
if 1==1:
    if "boyd_gamma_result" in globals():
        _simple_result = boyd_gamma_result
    elif "bisect_info" in globals():
        _simple_result = bisect_info
    elif "outer_bisection_with_visual_trace" in globals():
        _simple_result = outer_bisection_with_visual_trace(
            A,
            M,
            tol_gamma=float(os.environ.get("BOYD_SIMPLE_TOL", "1e-8")),
            tol_imag=float(os.environ.get("BOYD_SIMPLE_TOL_IMAG", "1e-10")),
            max_bisect=int(os.environ.get("BOYD_SIMPLE_MAX_BISECT", "12")),
            b=float(os.environ.get("BOYD_SIMPLE_B", "10.0")),
            eigs_tol=float(os.environ.get("BOYD_SIMPLE_EIGS_TOL", "1e-6")),
            nev=int(os.environ.get("BOYD_SIMPLE_NEV", "16")),
        )
    elif "outer_bisection_gamma_history" in globals():
        _simple_result = outer_bisection_gamma_history(
            A,
            M,
            tol_gamma=float(os.environ.get("BOYD_SIMPLE_TOL", "1e-8")),
            tol_imag=float(os.environ.get("BOYD_SIMPLE_TOL_IMAG", "1e-10")),
            max_bisect=int(os.environ.get("BOYD_SIMPLE_MAX_BISECT", "12")),
        )
    else:
        raise RuntimeError(
            "Need either boyd_gamma_result, bisect_info, "
            "outer_bisection_with_visual_trace, or outer_bisection_gamma_history."
        )

    boyd_simple_gamma_plot = plot_boyd_gamma_like_chatgpt(
        _simple_result,
        true_hinf=None,
        only_bisection=_env_flag("BOYD_SIMPLE_ONLY_BISECTION", True),
        max_points=int(os.environ.get("BOYD_SIMPLE_POINTS", "8")),
        outdir=os.environ.get("BOYD_SIMPLE_OUTDIR", "/stck/eclement/Control Of NavierStokes/plots"),
        basename=os.environ.get("BOYD_SIMPLE_BASENAME", "boyd_gamma_like_chatgpt"),
    )

# =============================================================================
# Newton / eigenvalue peak-search explanation plots
#
# Produces:
#   09_newton_paths_on_gain_curve.png/pdf
#   10_newton_convergence.png/pdf
#   11_newton_quadratic_step.png/pdf
#   12_boyd_pencil_touch_at_newton_peak.png/pdf
#
# Run with:
#   MAKE_NEWTON_PEAK_PLOTS=1 python plot.py
# or under MPI as usual.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt


def _newton_plot_is_root():
    return ("IS_WORLD_ROOT" not in globals()) or IS_WORLD_ROOT


def _newton_plot_barrier():
    if "WORLD" in globals():
        WORLD.Barrier()


def _newton_plot_mkdir(outdir):
    if _newton_plot_is_root():
        os.makedirs(outdir, exist_ok=True)
    _newton_plot_barrier()


def _newton_env_flag(name, default=False):
    if "_env_flag" in globals():
        return _env_flag(name, default)
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def deterministic_signed_initial_omegas(wmin=1e-1, wmax=1e1, num_per_side=4):
    """
    Nice deterministic starts for a presentation figure.
    This shows Newton starting from both sides of the gain curve.
    """
    wp = np.geomspace(float(wmin), float(wmax), int(num_per_side))
    return np.concatenate((-wp[::-1], wp))

from multistartfixed import (get_lambda_dot, run_hinf_r4_noK_true_multistart_openloop, build_sensor_state_vector, sensor_output_vector)
def local_newton_peak_with_trace(
    A,
    M,
    omega0,
    max_it=20,
    tol_w=1e-8,
    tol_lamdot=1e-8,
    trust_radius=0.25,
    backtrack=0.5,
    min_step=1e-10,
):
    """
    Same idea as local_newton_peak(...), but records every Newton step.

    Newton is applied to lambda(omega) = ||G(i omega)||^2.
    The peak condition is lambda'(omega) = 0, with lambda''(omega) < 0.
    """

    omega = float(omega0)

    g, gdot, lam, lamdot, lamddot = get_lambda_dot(A, M, omega)
    best_omega = float(omega)
    best_gain = float(g)

    records = []

    for it in range(int(max_it)):
        rec = {
            "iter": int(it),
            "omega": float(omega),
            "gain": float(g),
            "lambda": float(np.real(lam)),
            "lambda_dot": float(np.real(lamdot)),
            "lambda_ddot": float(np.real(lamddot)),
            "status": "running",
        }

        # Converged to a local maximum of lambda = gain^2.
        if abs(lamdot) <= tol_lamdot * max(1.0, abs(lam)) and lamddot < 0.0:
            rec["status"] = "converged"
            records.append(rec)
            return {
                "omega0": float(omega0),
                "omega_star": float(omega),
                "gain_star": float(g),
                "status": "converged",
                "records": records,
            }

        # Newton step for lambda'(omega)=0.
        if np.isfinite(lamddot) and abs(lamddot) > 1e-14 and lamddot < 0.0:
            step_raw = -lamdot / lamddot
            step_kind = "newton"
        else:
            # Fallback: move in the uphill direction if the local curvature
            # is not yet useful.
            step_raw = trust_radius * np.sign(lamdot) if lamdot != 0 else 0.0
            step_kind = "fallback"

        step = float(np.clip(np.real(step_raw), -trust_radius, trust_radius))

        rec["step_raw"] = float(np.real(step_raw))
        rec["step"] = float(step)
        rec["step_kind"] = step_kind

        accepted = False
        alpha = 1.0
        omega_trial = omega
        g_trial = g

        while alpha >= min_step:
            omega_trial = float(omega + alpha * step)
            g_trial = float(gain_of_omega(A, M, omega_trial))

            # Accept only uphill moves. This makes the figure easy to explain:
            # each accepted dot is at least as high as the previous one.
            if np.isfinite(g_trial) and g_trial >= g:
                g_new, gdot_new, lam_new, lamdot_new, lamddot_new = get_lambda_dot(
                    A, M, omega_trial
                )

                omega = omega_trial
                g = float(g_new)
                gdot = float(gdot_new)
                lam = lam_new
                lamdot = lamdot_new
                lamddot = lamddot_new

                if g > best_gain:
                    best_omega = float(omega)
                    best_gain = float(g)

                accepted = True
                break

            alpha *= backtrack

        rec["accepted"] = bool(accepted)
        rec["alpha"] = float(alpha)
        rec["omega_trial"] = float(omega_trial)
        rec["gain_trial"] = float(g_trial)

        if not accepted:
            rec["status"] = "stalled"
            records.append(rec)
            return {
                "omega0": float(omega0),
                "omega_star": float(best_omega),
                "gain_star": float(best_gain),
                "status": "stalled",
                "records": records,
            }

        records.append(rec)

        if abs(alpha * step) <= tol_w * max(1.0, abs(omega)):
            records.append({
                "iter": int(it + 1),
                "omega": float(omega),
                "gain": float(g),
                "lambda": float(np.real(lam)),
                "lambda_dot": float(np.real(lamdot)),
                "lambda_ddot": float(np.real(lamddot)),
                "status": "small_step",
            })
            return {
                "omega0": float(omega0),
                "omega_star": float(omega),
                "gain_star": float(g),
                "status": "small_step",
                "records": records,
            }

    records.append({
        "iter": int(max_it),
        "omega": float(omega),
        "gain": float(g),
        "lambda": float(np.real(lam)),
        "lambda_dot": float(np.real(lamdot)),
        "lambda_ddot": float(np.real(lamddot)),
        "status": "max_it",
    })

    return {
        "omega0": float(omega0),
        "omega_star": float(best_omega),
        "gain_star": float(best_gain),
        "status": "max_it",
        "records": records,
    }


def compute_newton_peak_traces(A, M, initial_omegas):
    """
    Compute traced Newton runs from several starting frequencies.
    Uses mpi_task_map if available.
    """

    initial_omegas = [float(w) for w in initial_omegas]

    def _one_trace(w0):
        return local_newton_peak_with_trace(A, M, w0)

    if "mpi_task_map" in globals():
        traces = mpi_task_map(
            _one_trace,
            initial_omegas,
            label="newton_peak_trace",
            use_parallel=True,
        )
    else:
        traces = [_one_trace(w0) for w0 in initial_omegas]

    traces = sorted(traces, key=lambda tr: tr["gain_star"], reverse=True)
    return traces


def _trace_arrays(trace):
    recs = trace["records"]
    it = np.array([r["iter"] for r in recs], dtype=int)
    w = np.array([r["omega"] for r in recs], dtype=float)
    g = np.array([r["gain"] for r in recs], dtype=float)
    return it, w, g


def plot_newton_paths_on_gain_curve(
    frequencies,
    gains,
    traces,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="09_newton_paths_on_gain_curve",
    max_traces=6,
):
    """
    Plot the gain curve and overlay the actual Newton iterates.
    """

    _newton_plot_mkdir(outdir)
    if not _newton_plot_is_root():
        return None

    frequencies = np.asarray(frequencies, dtype=float)
    gains = np.asarray(gains, dtype=float)
    finite = np.isfinite(frequencies) & np.isfinite(gains)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    ax.plot(
        frequencies[finite],
        gains[finite],
        linewidth=1.4,
        label=r"sampled $\|G(i\omega)\|$",
    )

    shown = traces[:int(max_traces)]

    for j, tr in enumerate(shown):
        _, w, g = _trace_arrays(tr)

        label = (
            fr"Newton path from $\omega_0={tr['omega0']:.3g}$"
            if j == 0 else None
        )

        ax.plot(
            w,
            g,
            marker="o",
            markersize=4,
            linewidth=1.2,
            alpha=0.85,
            label=label,
        )

        # Small arrows showing direction of movement.
        for k in range(len(w) - 1):
            ax.annotate(
                "",
                xy=(w[k + 1], g[k + 1]),
                xytext=(w[k], g[k]),
                arrowprops=dict(arrowstyle="->", linewidth=0.9, alpha=0.55),
            )

        ax.scatter([w[0]], [g[0]], marker="s", s=42, zorder=5)
        ax.scatter([w[-1]], [g[-1]], marker="*", s=95, zorder=6)

    best = traces[0]
    ax.scatter(
        [best["omega_star"]],
        [best["gain_star"]],
        marker="*",
        s=150,
        zorder=7,
        label=fr"best Newton peak: $\omega={best['omega_star']:.6g}$",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel(r"gain $\|G(i\omega)\|$")
    ax.set_title("Newton peak search: a few gain evaluations move to the maximum")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    return {"png": png, "pdf": pdf}


def plot_newton_convergence(
    traces,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="10_newton_convergence",
    max_traces=6,
):
    """
    Show why Newton feels faster than bisection:
    the frequency error and gain gap collapse in very few iterations.
    """

    _newton_plot_mkdir(outdir)
    if not _newton_plot_is_root():
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    for j, tr in enumerate(traces[:int(max_traces)]):
        it, w, g = _trace_arrays(tr)

        w_final = float(w[-1])
        g_final = float(np.nanmax(g))

        omega_error = np.maximum(np.abs(w - w_final), 1e-16)
        gain_gap = np.maximum(g_final - g, 1e-16)

        label = fr"$\omega_0={tr['omega0']:.3g}$"

        ax1.semilogy(it, omega_error, marker="o", linewidth=1.2, label=label)
        ax2.semilogy(it, gain_gap, marker="o", linewidth=1.2, label=label)

    ax1.set_xlabel("Newton iteration")
    ax1.set_ylabel(r"$|\omega_k-\omega_\star|$")
    ax1.set_title("Frequency error")
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)

    ax2.set_xlabel("Newton iteration")
    ax2.set_ylabel(r"$\|G(i\omega_\star)\|-\|G(i\omega_k)\|$")
    ax2.set_title("Gain gap")
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)

    ax2.legend(loc="best", fontsize=8)
    fig.suptitle(r"Newton convergence to the $H_\infty$ peak", y=1.02)
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {"png": png, "pdf": pdf}


def _collect_gain_curve_on_grid(A, M, grid):
    grid = [float(w) for w in grid]

    def _one(w):
        return w, float(gain_of_omega(A, M, w))

    if "mpi_task_map" in globals():
        pairs = mpi_task_map(
            _one,
            grid,
            label="newton_quadratic_step_gain_grid",
            use_parallel=True,
        )
    else:
        pairs = [_one(w) for w in grid]

    pairs = sorted(pairs, key=lambda p: p[0])
    return (
        np.array([p[0] for p in pairs], dtype=float),
        np.array([p[1] for p in pairs], dtype=float),
    )


def plot_newton_quadratic_step(
    A,
    M,
    trace,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="11_newton_quadratic_step",
    ngrid=120,
):
    """
    Show one Newton step geometrically.

    We fit the local quadratic model to lambda(omega)=gain^2:
        lambda_q(omega)
        = lambda_k + lambda'_k d + 1/2 lambda''_k d^2.

    The top of this quadratic gives the next Newton frequency.
    """

    # Pick the first record where a true Newton step was used and accepted.
    chosen = None
    for r in trace["records"]:
        if r.get("step_kind") == "newton" and r.get("accepted", False):
            chosen = r
            break

    if chosen is None:
        chosen = trace["records"][0]

    omega_k = float(chosen["omega"])
    gain_k = float(chosen["gain"])
    lam_k = float(chosen["lambda"])
    lamdot_k = float(chosen["lambda_dot"])
    lamddot_k = float(chosen["lambda_ddot"])

    omega_next = float(chosen.get("omega_trial", omega_k))
    gain_next = float(chosen.get("gain_trial", gain_k))

    radius = max(2.5 * abs(omega_next - omega_k), 0.05)
    grid = np.linspace(omega_k - radius, omega_k + radius, int(ngrid))

    # This must be computed collectively if running under MPI.
    ws, actual_gain = _collect_gain_curve_on_grid(A, M, grid)

    d = ws - omega_k
    lambda_quad = lam_k + lamdot_k * d + 0.5 * lamddot_k * d**2
    gain_quad = np.sqrt(np.maximum(lambda_quad, 0.0))

    _newton_plot_mkdir(outdir)
    if not _newton_plot_is_root():
        return None

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    ax.plot(ws, actual_gain, linewidth=2.0, label=r"actual gain $\|G(i\omega)\|$")
    ax.plot(ws, gain_quad, linestyle="--", linewidth=2.0,
            label=r"local quadratic model from eigenvalue derivatives")

    ax.scatter([omega_k], [gain_k], s=70, zorder=5,
               label=fr"current iterate $\omega_k={omega_k:.5g}$")
    ax.scatter([omega_next], [gain_next], marker="*", s=130, zorder=6,
               label=fr"Newton update $\omega_{{k+1}}={omega_next:.5g}$")

    ax.annotate(
        "",
        xy=(omega_next, gain_next),
        xytext=(omega_k, gain_k),
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel(r"gain $\|G(i\omega)\|$")
    ax.set_title("One Newton step: use eigenvalue derivatives to jump toward the peak")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)

    return {"png": png, "pdf": pdf}


def plot_boyd_pencil_touch_at_newton_peak(
    A,
    M,
    omega_star,
    gamma_star,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="12_boyd_pencil_touch_at_newton_peak",
    gamma_factors=(0.98, 1.0, 1.02),
    nev=60,
    eigs_tol=1e-8,
):
    """
    Eigenvalue explanation plot.

    At gamma = ||G||_infty, the Boyd/Hamiltonian pencil has an eigenvalue
    on or very close to the imaginary axis at lambda = i omega_star.
    """

    omega_star = float(omega_star)
    gamma_star = float(gamma_star)

    N = build_N_from_M(M)
    N = N.convert("aij")
    N.assemble()

    eig_data = []

    for fac in gamma_factors:
        gamma = float(fac) * gamma_star

        Mgam = build_Mgamma(A, M, gamma)
        Mgam = Mgam.convert("aij")
        Mgam.assemble()

        lambdas, nconv = eigs_close_to_shift_pencil(
            N,
            Mgam,
            theta=omega_star,
            nev=nev,
            tol=eigs_tol,
        )

        lambdas = np.asarray(lambdas, dtype=complex)
        eig_data.append({
            "factor": float(fac),
            "gamma": float(gamma),
            "lambdas": lambdas,
            "nconv": int(nconv),
        })

    _newton_plot_mkdir(outdir)
    if not _newton_plot_is_root():
        return None

    fig, axes = plt.subplots(
        1,
        len(eig_data),
        figsize=(5.0 * len(eig_data), 5.2),
        sharey=True,
    )

    if len(eig_data) == 1:
        axes = [axes]

    # Common axis limits around the shift.
    all_re = []
    all_im_shifted = []

    for item in eig_data:
        lam = item["lambdas"]
        if lam.size:
            all_re.extend(np.real(lam))
            all_im_shifted.extend(np.imag(lam) - omega_star)

    if len(all_re) == 0:
        xlim = (-1.0, 1.0)
        ylim = (-1.0, 1.0)
    else:
        max_re = max(1e-10, np.nanmax(np.abs(all_re)))
        max_im = max(1e-10, np.nanmax(np.abs(all_im_shifted)))
        xlim = (-1.15 * max_re, 1.15 * max_re)
        ylim = (-1.15 * max_im, 1.15 * max_im)

    for ax, item in zip(axes, eig_data):
        lam = item["lambdas"]

        if lam.size:
            x = np.real(lam)
            y = np.imag(lam) - omega_star
            ax.scatter(x, y, s=36, label="nearby pencil eigenvalues")

            j = int(np.argmin(np.abs(lam - 1j * omega_star)))
            ax.scatter(
                [np.real(lam[j])],
                [np.imag(lam[j]) - omega_star],
                marker="x",
                s=95,
                zorder=6,
                label="closest to shift",
            )

        ax.axvline(0.0, linestyle="--", linewidth=1.5,
                   label="imaginary axis")
        ax.axhline(0.0, linestyle=":", linewidth=1.5,
                   label=r"$\Im(\lambda)=\omega_\star$")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\Re(\lambda)$")
        ax.grid(True, linestyle="--", alpha=0.30)

        fac = item["factor"]
        ax.set_title(
            fr"$\gamma={fac:.2f}\,\gamma_\star$"
            "\n"
            fr"nconv={item['nconv']}"
        )

    axes[0].set_ylabel(r"$\Im(\lambda)-\omega_\star$")

    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="best", fontsize=8)

    fig.suptitle(
        fr"Boyd pencil near the Newton peak $\omega_\star={omega_star:.6g}$",
        y=1.02,
    )
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {"png": png, "pdf": pdf}


def make_newton_peak_explanation_plots(
    A,
    M,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    wmin=1e-1,
    wmax=1e1,
    npos=500,
    num_per_side=4,
    eigs_tol=1e-8,
    nev=60,
):
    """
    Convenience wrapper for all Newton/eigenvalue explanation plots.
    """

    _newton_plot_mkdir(outdir)

    initial_omegas = deterministic_signed_initial_omegas(
        wmin=wmin,
        wmax=wmax,
        num_per_side=num_per_side,
    )

    traces = compute_newton_peak_traces(A, M, initial_omegas)
    best = traces[0]

    # Reuse your existing sweep helper if available.
    if "compute_openloop_gain_sweep" in globals():
        frequencies, gains = compute_openloop_gain_sweep(
            A,
            M,
            omega_min=wmin,
            omega_max=wmax,
            number_of_points=npos,
        )
    else:
        frequencies = symmetric_log_frequency_grid(
            omega_min=wmin,
            omega_max=wmax,
            number_of_points=npos,
        )
        frequencies, gains = _collect_gain_curve_on_grid(A, M, frequencies)

    paths = {}
    paths["newton_paths"] = plot_newton_paths_on_gain_curve(
        frequencies,
        gains,
        traces,
        outdir=outdir,
    )
    paths["newton_convergence"] = plot_newton_convergence(
        traces,
        outdir=outdir,
    )
    paths["newton_quadratic_step"] = plot_newton_quadratic_step(
        A,
        M,
        best,
        outdir=outdir,
    )
    paths["boyd_pencil_touch"] = plot_boyd_pencil_touch_at_newton_peak(
        A,
        M,
        omega_star=best["omega_star"],
        gamma_star=best["gain_star"],
        outdir=outdir,
        eigs_tol=eigs_tol,
        nev=nev,
    )

    if _newton_plot_is_root():
        print("\nNewton peak explanation plots written to:", outdir, flush=True)
        print("omega_star:", best["omega_star"], flush=True)
        print("gamma_star:", best["gain_star"], flush=True)
        for key, value in paths.items():
            print(" ", key, "->", value, flush=True)

    return {
        "omega_star": best["omega_star"],
        "gamma_star": best["gain_star"],
        "traces": traces,
        "paths": paths,
    }


# Optional automatic execution. Off by default.
if 1 == 1:
    newton_peak_report = make_newton_peak_explanation_plots(
        A,
        M,
        outdir="/stck/eclement/Control Of NavierStokes/plots",
        wmin=float(os.environ.get("NEWTON_PEAK_WMIN", "1e-1")),
        wmax=float(os.environ.get("NEWTON_PEAK_WMAX", "1e1")),
        npos=int(os.environ.get("NEWTON_PEAK_NPOS", "500")),
        num_per_side=int(os.environ.get("NEWTON_PEAK_NUM_PER_SIDE", "4")),
        eigs_tol=float(os.environ.get("NEWTON_PEAK_EIGS_TOL", "1e-8")),
        nev=int(os.environ.get("NEWTON_PEAK_NEV", "60")),
    )


# =============================================================================
# Nonsmooth controller-optimization explanation plots
#
# Cheap plots (CSV-based, no new optimization):
#   13_nonsmooth_multistart_summary.png/pdf
#   14_nonsmooth_best_start_history.png/pdf
#   15_nonsmooth_top_starts_overlay.png/pdf
#   16_nonsmooth_active_frequency_and_stability.png/pdf
#
# Optional extra plots (small number of oracle evaluations, but NO multistart):
#   17_nonsmooth_slice_openloop_to_best.png/pdf
#   18_nonsmooth_envelope_along_slice.png/pdf
#
# Run with:
#   MAKE_NONSMOOTH_CONTROLLER_PLOTS=1 python plot.py
#
# Optional:
#   MAKE_NONSMOOTH_SLICE_PLOTS=1 python plot.py
#
# IMPORTANT:
#   Keep RUN_TRUE_MULTISTART=0 when only making plots.
# =============================================================================

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# basic helpers
# -----------------------------------------------------------------------------

def _ns_is_root():
    return ("IS_WORLD_ROOT" not in globals()) or IS_WORLD_ROOT


def _ns_barrier():
    if "WORLD" in globals():
        WORLD.Barrier()


def _ns_mkdir(outdir):
    if _ns_is_root():
        os.makedirs(outdir, exist_ok=True)
    _ns_barrier()


def _ns_env_flag(name, default=False):
    if "_env_flag" in globals():
        return _env_flag(name, default)
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _ns_savefig(fig, outdir, basename):
    _ns_mkdir(outdir)
    if not _ns_is_root():
        return None
    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _read_csv_dicts(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _load_multistart_summary(csv_name="hinf_r4_noK_true_multistart_summary.csv"):
    if not os.path.exists(csv_name):
        return []
    rows = _read_csv_dicts(csv_name)
    out = []
    for r in rows:
        out.append({
            "start_id": int(r["start_id"]),
            "label": r["label"],
            "seed": int(float(r["seed"])),
            "status": r["status"],
            "elapsed_sec": _safe_float(r["elapsed_sec"]),
            "init_norm_n": _safe_float(r["init_norm_n"]),
            "best_f": _safe_float(r["best_f"]),
            "best_alpha": _safe_float(r["best_alpha"]),
            "best_omega": _safe_float(r["best_omega"]),
            "best_norm_n": _safe_float(r["best_norm_n"]),
        })
    return out


def _load_one_history(start_id):
    path = f"hinf_r4_noK_multistart_{int(start_id):03d}_history.csv"
    if not os.path.exists(path):
        return []
    rows = _read_csv_dicts(path)
    out = []
    for r in rows:
        out.append({
            "call": int(r["call"]),
            "f_raw": _safe_float(r["f_raw"]),
            "f_return": _safe_float(r["f_return"]),
            "alpha": _safe_float(r["alpha"]),
            "c_stab": _safe_float(r["c_stab"]),
            "omega_star": _safe_float(r["omega_star"]),
            "norm_n": _safe_float(r["norm_n"]),
            "skipped_hinf": str(r["skipped_hinf"]).strip().lower() in {"true", "1", "yes"},
            "certified": str(r["certified"]).strip().lower() in {"true", "1", "yes"},
        })
    return out


def _best_start_from_summary(summary):
    feas = [r for r in summary if np.isfinite(r["best_f"])]
    if not feas:
        return None
    return min(feas, key=lambda r: r["best_f"])


def _top_starts_from_summary(summary, ntop=5):
    feas = [r for r in summary if np.isfinite(r["best_f"])]
    feas = sorted(feas, key=lambda r: r["best_f"])
    return feas[:int(ntop)]


# -----------------------------------------------------------------------------
# Plot 1: multistart summary
# -----------------------------------------------------------------------------

def plot_nonsmooth_multistart_summary(
    csv_name="hinf_r4_noK_true_multistart_summary.csv",
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="13_nonsmooth_multistart_summary",
):
    summary = _load_multistart_summary(csv_name)
    if not summary or not _ns_is_root():
        return None

    start_id = np.array([r["start_id"] for r in summary], dtype=int)
    best_f = np.array([r["best_f"] for r in summary], dtype=float)
    elapsed = np.array([r["elapsed_sec"] for r in summary], dtype=float)
    init_norm_n = np.array([r["init_norm_n"] for r in summary], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.2, 7.4), sharex=True)

    finite = np.isfinite(best_f)
    ax1.scatter(start_id[finite], best_f[finite], s=45, label="feasible best value")
    bad = ~finite
    if np.any(bad):
        ybad = np.full(np.sum(bad), np.nanmax(best_f[finite]) if np.any(finite) else 1.0)
        ax1.scatter(start_id[bad], ybad, marker="x", s=45, label="no feasible point")

    best = _best_start_from_summary(summary)
    if best is not None:
        ax1.scatter(
            [best["start_id"]],
            [best["best_f"]],
            marker="*",
            s=180,
            zorder=6,
            label=fr"global best start {best['start_id']}",
        )

    ax1.set_yscale("log")
    ax1.set_ylabel(r"best feasible $H_\infty$")
    ax1.set_title("Multistart summary: different initial controllers reach different local minima")
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=9)

    ax2.plot(start_id, elapsed, marker="o", linewidth=1.2, label="elapsed time")
    ax2_t = ax2.twinx()
    ax2_t.plot(start_id, init_norm_n, linestyle="--", marker="s", linewidth=1.0,
               label=r"initial $\|n\|$")

    ax2.set_xlabel("start id")
    ax2.set_ylabel("elapsed time [s]")
    ax2_t.set_ylabel(r"initial $\|n\|$")
    ax2.set_title("Runtime and size of the initial controller perturbation")
    ax2.grid(True, linestyle="--", alpha=0.35)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

    fig.tight_layout()
    return _ns_savefig(fig, outdir, basename)


# -----------------------------------------------------------------------------
# Plot 2: one best-start history
# -----------------------------------------------------------------------------

def plot_nonsmooth_best_start_history(
    csv_name="hinf_r4_noK_true_multistart_summary.csv",
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="14_nonsmooth_best_start_history",
):
    summary = _load_multistart_summary(csv_name)
    best = _best_start_from_summary(summary)
    if best is None:
        return None

    hist = _load_one_history(best["start_id"])
    if not hist or not _ns_is_root():
        return None

    k = np.array([h["call"] for h in hist], dtype=int)
    f_raw = np.array([h["f_raw"] for h in hist], dtype=float)
    f_ret = np.array([h["f_return"] for h in hist], dtype=float)
    c_stab = np.array([h["c_stab"] for h in hist], dtype=float)
    skipped = np.array([h["skipped_hinf"] for h in hist], dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(10.4, 9.0), sharex=True)

    axes[0].semilogy(k, f_ret, marker="o", linewidth=1.2, label="objective returned to PyGRANSO")
    good = np.isfinite(f_raw)
    axes[0].semilogy(k[good], f_raw[good], marker="s", linewidth=1.0,
                     label=r"raw $H_\infty$ objective")
    if np.any(skipped):
        axes[0].scatter(k[skipped], f_ret[skipped], marker="x", s=50,
                        label=r"unstable: hard gate, no $H_\infty$ evaluation")
    axes[0].set_ylabel("objective")
    axes[0].set_title(
        fr"Best start history (start {best['start_id']}): objective, gate, and feasibility"
    )
    axes[0].grid(True, which="both", linestyle="--", alpha=0.35)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(k, c_stab, marker="o", linewidth=1.2)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.2)
    axes[1].set_ylabel(r"$c_{\mathrm{stab}} = \alpha + \mathrm{margin}$")
    axes[1].set_title("Hard stability gate")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    penalty = f_ret.copy()
    penalty[good] = f_ret[good] - f_raw[good]
    penalty[~np.isfinite(penalty)] = np.nan
    axes[2].semilogy(k, np.maximum(np.abs(penalty), 1e-16), marker="o", linewidth=1.2)
    axes[2].set_xlabel("oracle call")
    axes[2].set_ylabel(r"$|f_{\mathrm{return}} - f_{\mathrm{raw}}|$")
    axes[2].set_title("How much the stability penalty / gate changes the returned objective")
    axes[2].grid(True, which="both", linestyle="--", alpha=0.35)

    fig.tight_layout()
    return _ns_savefig(fig, outdir, basename)


# -----------------------------------------------------------------------------
# Plot 3: top-start overlays
# -----------------------------------------------------------------------------

def plot_nonsmooth_top_starts_overlay(
    csv_name="hinf_r4_noK_true_multistart_summary.csv",
    ntop=5,
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="15_nonsmooth_top_starts_overlay",
):
    summary = _load_multistart_summary(csv_name)
    top = _top_starts_from_summary(summary, ntop=ntop)
    if not top or not _ns_is_root():
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.4, 8.2), sharex=False)

    for row in top:
        hist = _load_one_history(row["start_id"])
        if not hist:
            continue
        k = np.array([h["call"] for h in hist], dtype=int)
        f_ret = np.array([h["f_return"] for h in hist], dtype=float)
        alpha = np.array([h["alpha"] for h in hist], dtype=float)

        ax1.semilogy(k, f_ret, marker="o", linewidth=1.2,
                     label=fr"start {row['start_id']}: best={row['best_f']:.3e}")
        ax2.plot(k, alpha, marker="o", linewidth=1.2,
                 label=fr"start {row['start_id']}")

    ax1.set_xlabel("oracle call")
    ax1.set_ylabel("returned objective")
    ax1.set_title("Top multistart trajectories: same problem, different local behavior")
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=8)

    ax2.axhline(0.0, linestyle="--", linewidth=1.2)
    ax2.set_xlabel("oracle call")
    ax2.set_ylabel(r"spectral abscissa $\alpha$")
    ax2.set_title("Some starts stay stable, others approach the gate differently")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return _ns_savefig(fig, outdir, basename)


# -----------------------------------------------------------------------------
# Plot 4: active frequency + stability for the best start
# -----------------------------------------------------------------------------

def plot_nonsmooth_active_frequency_and_stability(
    csv_name="hinf_r4_noK_true_multistart_summary.csv",
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="16_nonsmooth_active_frequency_and_stability",
):
    summary = _load_multistart_summary(csv_name)
    best = _best_start_from_summary(summary)
    if best is None:
        return None

    hist = _load_one_history(best["start_id"])
    if not hist or not _ns_is_root():
        return None

    k = np.array([h["call"] for h in hist], dtype=int)
    omega = np.array([h["omega_star"] for h in hist], dtype=float)
    alpha = np.array([h["alpha"] for h in hist], dtype=float)
    skipped = np.array([h["skipped_hinf"] for h in hist], dtype=bool)
    good = ~skipped & np.isfinite(omega)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.4, 7.0), sharex=True)

    ax1.plot(k, alpha, marker="o", linewidth=1.2)
    ax1.axhline(0.0, linestyle="--", linewidth=1.2)
    ax1.set_ylabel(r"$\alpha$")
    ax1.set_title("Stability and active peak frequency along the best-start run")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2.plot(k[good], omega[good], marker="o", linewidth=1.2, label=r"active $\omega_\star$")
    if np.any(skipped):
        ax2.scatter(k[skipped], np.zeros(np.sum(skipped)), marker="x", s=45,
                    label=r"no $\omega_\star$ because $H_\infty$ was skipped")
    ax2.set_xlabel("oracle call")
    ax2.set_ylabel(r"$\omega_\star$")
    ax2.set_title("Jumps in the active peak frequency are a visible source of nonsmoothness")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return _ns_savefig(fig, outdir, basename)

from multistartfixed import (make_cldyn_for_placement, default_stable_a_center, dynamic_oracle_hard_stability, run_hinf_r4_noK_true_multistart_openloop, build_actuator_vector)
# -----------------------------------------------------------------------------
# optional: local slice around the best controller
# -----------------------------------------------------------------------------

def _get_best_run_object():
    if "hinf_r4_noK_multistart_run" not in globals():
        return None, None, None

    run = run_hinf_r4_noK_true_multistart_openloop
    if run is None or run.get("best_result", None) is None:
        return None, None, None

    best_res = run["best_result"]
    best = best_res["best"]
    if best is None:
        return None, None, None

    return run, best_res, best


def _make_best_cldyn_for_slice(r=4):
    xa_fixed = float(os.environ.get("HINF_R4_NOK_XA", "-0.98"))
    xs_fixed = float(os.environ.get("HINF_R4_NOK_XS", "0.95"))
    return make_cldyn_for_placement(
        xa=xa_fixed,
        xs=xs_fixed,
        A=A,
        M=M,
        V=V,
        phi=phi,
        bc=bc,
        is_free=is_free,
        sigma_gauss=sigma_gauss,
        r=r,
    )


def evaluate_nonsmooth_slice_openloop_to_best(
    nslice=21,
    tmin=-0.15,
    tmax=1.15,
    r=4,
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
):
    run, best_res, best = _get_best_run_object()
    if best is None:
        return None

    cl_dyn = _make_best_cldyn_for_slice(r=r)

    a_best = np.asarray(best["a"], dtype=np.complex128).reshape(r)
    n_best = np.asarray(best["n"], dtype=np.complex128).reshape(r)

    a_open = np.asarray(default_stable_a_center(r), dtype=np.complex128).reshape(r)
    n_open = np.zeros(r, dtype=np.complex128)

    ts = np.linspace(float(tmin), float(tmax), int(nslice))

    rows = []
    for t in ts:
        a_t = a_open + t * (a_best - a_open)
        n_t = n_open + t * (n_best - n_open)
        theta_t = cl_dyn.pack_theta(a_t, n_t, Kstat=0.0)

        out = dynamic_oracle_hard_stability(
            cl_dyn,
            theta=theta_t,
            stab_margin=stab_margin,
            stab_buffer=stab_buffer,
            stab_penalty_weight=stab_penalty_weight,
            unstable_objective=unstable_objective,
            num_guess=num_guess,
            wmin=wmin,
            wmax=wmax,
            seed=2,
            candidate_npos=candidate_npos,
            disk_b=disk_b,
            disk_eigs_tol=disk_eigs_tol,
            disk_nev=disk_nev,
            disk_eps_probe=disk_eps_probe,
            disk_axis_warn=disk_axis_warn,
        )

        rows.append({
            "t": float(t),
            "a": a_t.copy(),
            "n": n_t.copy(),
            "theta": theta_t.copy(),
            "f_raw": float(out["f_raw"]),
            "f_return": float(out["f_return"]),
            "alpha": float(out["alpha"]),
            "c": float(out["c"]),
            "omega_star": float(out["omega_star"]) if np.isfinite(out["omega_star"]) else np.nan,
            "skipped_hinf": bool(out["skipped_hinf"]),
        })

    return {
        "rows": rows,
        "a_open": a_open,
        "n_open": n_open,
        "a_best": a_best,
        "n_best": n_best,
    }


def plot_nonsmooth_slice_openloop_to_best(
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="17_nonsmooth_slice_openloop_to_best",
    nslice=21,
):
    report = evaluate_nonsmooth_slice_openloop_to_best(nslice=nslice)
    if report is None or not _ns_is_root():
        return None

    rows = report["rows"]
    t = np.array([r["t"] for r in rows], dtype=float)
    f_raw = np.array([r["f_raw"] for r in rows], dtype=float)
    f_ret = np.array([r["f_return"] for r in rows], dtype=float)
    c = np.array([r["c"] for r in rows], dtype=float)
    omega = np.array([r["omega_star"] for r in rows], dtype=float)
    skipped = np.array([r["skipped_hinf"] for r in rows], dtype=bool)
    good = np.isfinite(f_raw)

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 9.4), sharex=True)

    axes[0].plot(t, f_ret, marker="o", linewidth=1.3, label="returned objective")
    axes[0].plot(t[good], f_raw[good], marker="s", linewidth=1.1, label=r"raw $H_\infty$")
    if np.any(skipped):
        axes[0].scatter(t[skipped], f_ret[skipped], marker="x", s=45,
                        label=r"hard stability gate active")
    axes[0].axvline(0.0, linestyle="--", linewidth=1.1)
    axes[0].axvline(1.0, linestyle="--", linewidth=1.1)
    axes[0].set_ylabel("objective")
    axes[0].set_title("One-dimensional slice from open loop to the best controller")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.35)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(t, c, marker="o", linewidth=1.3)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.1)
    axes[1].axvline(0.0, linestyle="--", linewidth=1.1)
    axes[1].axvline(1.0, linestyle="--", linewidth=1.1)
    axes[1].set_ylabel(r"$c_{\mathrm{stab}}$")
    axes[1].set_title("Crossing the stability boundary changes the objective definition")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].plot(t[~skipped & np.isfinite(omega)], omega[~skipped & np.isfinite(omega)],
                 marker="o", linewidth=1.3)
    axes[2].axvline(0.0, linestyle="--", linewidth=1.1)
    axes[2].axvline(1.0, linestyle="--", linewidth=1.1)
    axes[2].set_xlabel(r"slice parameter $t$  (0=open loop, 1=best controller)")
    axes[2].set_ylabel(r"$\omega_\star$")
    axes[2].set_title("Jumps in the active frequency reveal nonsmooth behavior")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    return _ns_savefig(fig, outdir, basename)


# -----------------------------------------------------------------------------
# optional: upper-envelope explanation along the slice
# -----------------------------------------------------------------------------

def plot_nonsmooth_envelope_along_slice(
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="18_nonsmooth_envelope_along_slice",
    nslice=21,
    nbranch=4,
):
    report = evaluate_nonsmooth_slice_openloop_to_best(nslice=nslice)
    if report is None or not _ns_is_root():
        return None

    rows = report["rows"]
    cl_dyn = _make_best_cldyn_for_slice(r=4)

    t = np.array([r["t"] for r in rows], dtype=float)
    f_raw = np.array([r["f_raw"] for r in rows], dtype=float)
    skipped = np.array([r["skipped_hinf"] for r in rows], dtype=bool)
    omega_star = np.array([r["omega_star"] for r in rows], dtype=float)

    # pick a few distinct active frequencies from the slice
    ws = []
    for w in omega_star[np.isfinite(omega_star)]:
        if len(ws) == 0 or np.min(np.abs(np.array(ws) - w)) > 2e-2:
            ws.append(float(w))
    ws = ws[:int(nbranch)]
    if len(ws) == 0:
        return None

    branch_vals = []
    for wfix in ws:
        vals = []
        for row in rows:
            theta = row["theta"]
            Acl = cl_dyn.Acl(theta)
            Ecl = cl_dyn.Ecl()
            vals.append(float(gain_of_omega(Acl, Ecl, float(wfix))))
        branch_vals.append(np.array(vals, dtype=float))

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), sharex=True)

    for j, (wfix, vals) in enumerate(zip(ws, branch_vals)):
        axes[0].plot(t, vals, marker="o", linewidth=1.1,
                     label=fr"fixed $\omega={wfix:.3g}$")

    good = np.isfinite(f_raw)
    axes[0].plot(t[good], f_raw[good], color="black", linewidth=2.2,
                 label=r"true raw $H_\infty$ objective")
    axes[0].set_ylabel(r"$\|G(i\omega)\|$")
    axes[0].set_title(
        r"Nonsmoothness comes from taking the maximum over frequency-dependent branches"
    )
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(t[~skipped & np.isfinite(omega_star)], omega_star[~skipped & np.isfinite(omega_star)],
                 marker="o", linewidth=1.3)
    axes[1].set_xlabel(r"slice parameter $t$")
    axes[1].set_ylabel(r"active $\omega_\star$")
    axes[1].set_title("When the maximizing frequency changes, the active branch changes")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    return _ns_savefig(fig, outdir, basename)


# -----------------------------------------------------------------------------
# master wrapper
# -----------------------------------------------------------------------------

def make_nonsmooth_controller_explanation_plots(
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    make_slice_plots=False,
):
    paths = {}

    paths["multistart_summary"] = plot_nonsmooth_multistart_summary(outdir=outdir)
    paths["best_start_history"] = plot_nonsmooth_best_start_history(outdir=outdir)
    paths["top_starts_overlay"] = plot_nonsmooth_top_starts_overlay(outdir=outdir)
    paths["active_frequency"] = plot_nonsmooth_active_frequency_and_stability(outdir=outdir)

    if make_slice_plots:
        paths["slice_openloop_to_best"] = plot_nonsmooth_slice_openloop_to_best(outdir=outdir)
        paths["envelope_along_slice"] = plot_nonsmooth_envelope_along_slice(outdir=outdir)
    else:
        paths["slice_openloop_to_best"] = None
        paths["envelope_along_slice"] = None

    if _ns_is_root():
        print("\nNonsmooth controller-optimization plots written to:", outdir, flush=True)
        for key, value in paths.items():
            print(" ", key, "->", value, flush=True)

    return paths


# -----------------------------------------------------------------------------
# optional auto-run
# -----------------------------------------------------------------------------

if 1==1:
    nonsmooth_controller_plot_paths = make_nonsmooth_controller_explanation_plots(
        outdir="/stck/eclement/Control Of NavierStokes/plots",
        make_slice_plots=_ns_env_flag("MAKE_NONSMOOTH_SLICE_PLOTS", False),
    )

# =============================================================================
# Extra visual plots:
#
# 19_noise_response_controller_switch:
#   time-domain noisy response, with a vertical line where the controller is
#   switched on.
#
# 20_frequency_response_flattening:
#   closed-loop gain curves at several PyGRANSO iterates, showing the peak
#   being flattened.
#
# Run:
#   RUN_TRUE_MULTISTART=0 MAKE_SWITCH_AND_FLATTENING_PLOTS=1 python plot.py
#
# Optional:
#   SWITCH_T_ON=40 SWITCH_T_FINAL=100 SWITCH_DT=0.02 python plot.py
# =============================================================================

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------

def _sw_is_root():
    return ("IS_WORLD_ROOT" not in globals()) or IS_WORLD_ROOT


def _sw_barrier():
    if "WORLD" in globals():
        WORLD.Barrier()


def _sw_env_flag(name, default=False):
    if "_env_flag" in globals():
        return _env_flag(name, default)
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _sw_mkdir(outdir):
    if _sw_is_root():
        os.makedirs(outdir, exist_ok=True)
    _sw_barrier()


def _sw_savefig(fig, outdir, basename):
    _sw_mkdir(outdir)
    if not _sw_is_root():
        return None
    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def _petsc_mat_to_csr(P):
    """
    Convert a PETSc matrix to scipy CSR.

    This is intended for the plotting/simulation helper, not for large
    production computations. It assumes the FE communicator has one rank.
    """
    if "FE_SIZE" in globals() and FE_SIZE != 1:
        raise RuntimeError(
            "The time-domain scipy plot assumes FE_SIZE=1. "
            "Run with MPI_TASK_GROUP_SIZE=1 or use a serial plotting run."
        )

    Q = P.convert("aij")
    Q.assemble()
    indptr, indices, data = Q.getValuesCSR()
    return sps.csr_matrix((data, indices, indptr), shape=Q.getSize())


def _petsc_vec_to_numpy(v):
    arr = v.getArray(readonly=True)
    return np.asarray(arr, dtype=np.complex128).copy()


def _complex_array_from_best_txt(text, key, expected_len=None):
    """
    Parse lines like:
        best a = [-77.6 -17.6j  -89.7+138.9j ...]
    allowing line wrapping inside the brackets.
    """
    start = text.find(key)
    if start < 0:
        return None

    b0 = text.find("[", start)
    if b0 < 0:
        return None

    depth = 0
    b1 = None
    for j in range(b0, len(text)):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                b1 = j
                break

    if b1 is None:
        return None

    inside = text[b0 + 1:b1].replace("\n", " ")

    # Match "real imagj", where imag has its own sign.
    pat = re.compile(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"\s*"
        r"([+-]\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)j"
    )

    vals = []
    for m in pat.finditer(inside):
        re_part = float(m.group(1))
        im_part = float(m.group(2).replace(" ", ""))
        vals.append(re_part + 1j * im_part)

    vals = np.array(vals, dtype=np.complex128)

    if expected_len is not None and vals.size != int(expected_len):
        print(f"[warning] parsed {vals.size} entries for {key}, expected {expected_len}")

    return vals


def _get_best_controller_a_n(r=4):
    """
    Prefer the in-memory multistart object. If not available, fall back to the
    global-best txt file produced by your multistart.
    """
    if "hinf_r4_noK_multistart_run" in globals():
        run = run_hinf_r4_noK_true_multistart_openloop
        if run is not None and run.get("best_result", None) is not None:
            b = run["best_result"]["best"]
            if b is not None:
                return (
                    np.asarray(b["a"], dtype=np.complex128).reshape(r),
                    np.asarray(b["n"], dtype=np.complex128).reshape(r),
                    "in_memory_best",
                )

    txt_name = "hinf_r4_noK_true_multistart_global_best.txt"
    if os.path.exists(txt_name):
        text = open(txt_name, "r").read()
        a = _complex_array_from_best_txt(text, "best a", expected_len=r)
        nvec = _complex_array_from_best_txt(text, "best n", expected_len=r)
        if a is not None and nvec is not None:
            return a.reshape(r), nvec.reshape(r), txt_name

    raise RuntimeError(
        "Could not find best controller. Need either hinf_r4_noK_multistart_run "
        "in memory or hinf_r4_noK_true_multistart_global_best.txt on disk."
    )


def _make_cldyn_for_switch_plot(r=4):
    xa_fixed = float(os.environ.get("HINF_R4_NOK_XA", "-0.98"))
    xs_fixed = float(os.environ.get("HINF_R4_NOK_XS", "0.95"))

    return make_cldyn_for_placement(
        xa=xa_fixed,
        xs=xs_fixed,
        A=A,
        M=M,
        V=V,
        phi=phi,
        bc=bc,
        is_free=is_free,
        sigma_gauss=sigma_gauss,
        r=r,
    )


def _build_sensor_output_vector_for_switch_plot(xs_fixed=None):
    if xs_fixed is None:
        xs_fixed = float(os.environ.get("HINF_R4_NOK_XS", "0.95"))

    s_red, _ = build_sensor_state_vector(
        float(xs_fixed),
        sigma_gauss,
        V,
        is_free,
    )
    c_vec = sensor_output_vector(M, s_red)
    return _petsc_vec_to_numpy(c_vec)


def _make_random_disturbance_vector(n, seed=7):
    rng = np.random.default_rng(int(seed))
    bw = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    bw = bw / (np.linalg.norm(bw) + 1e-30)
    return bw.astype(np.complex128)


# -----------------------------------------------------------------------------
# Plot 19: noisy time response, controller switched on at t_on
# -----------------------------------------------------------------------------

def simulate_noise_response_with_controller_switch(
    r=4,
    t_final=100.0,
    dt=0.02,
    t_on=40.0,
    noise_amp=1.0,
    noise_tau=0.25,
    noise_seed=4,
):
    """
    Simulate:

        M qdot = A q + B_w eta(t),                         t < t_on

        E_cl z_dot = A_cl(theta_star) z + B_w,cl eta(t),    t >= t_on

    where z = [q; controller state].

    The controller state is initialized to zero when the controller is switched on.
    This gives a clean visual explanation: first the plant is disturbed in open loop,
    then the optimized controller is activated.
    """

    a_best, n_best, source = _get_best_controller_a_n(r=r)
    cl_dyn = _make_cldyn_for_switch_plot(r=r)

    theta_best = cl_dyn.pack_theta(a_best, n_best, Kstat=0.0)

    A_sp = _petsc_mat_to_csr(A)
    M_sp = _petsc_mat_to_csr(M)

    Acl_sp = _petsc_mat_to_csr(cl_dyn.Acl(theta_best))
    Ecl_sp = _petsc_mat_to_csr(cl_dyn.Ecl())

    nplant = M_sp.shape[0]
    ncl = Ecl_sp.shape[0]

    c = _build_sensor_output_vector_for_switch_plot()
    bw_q = _make_random_disturbance_vector(nplant, seed=noise_seed)
    bw_cl = np.zeros(ncl, dtype=np.complex128)
    bw_cl[:nplant] = bw_q

    Nt = int(np.round(float(t_final) / float(dt))) + 1
    t = np.linspace(0.0, float(t_final), Nt)

    # Smooth-ish random input, more visual than pure white noise.
    rng = np.random.default_rng(int(noise_seed))
    eta = np.zeros(Nt, dtype=float)
    rho = np.exp(-float(dt) / float(noise_tau))
    for k in range(1, Nt):
        eta[k] = rho * eta[k - 1] + np.sqrt(max(1.0 - rho**2, 0.0)) * rng.standard_normal()
    eta *= float(noise_amp)

    # Implicit Euler matrices.
    L_open = spla.factorized((M_sp - float(dt) * A_sp).tocsc())
    L_cl = spla.factorized((Ecl_sp - float(dt) * Acl_sp).tocsc())

    q = np.zeros(nplant, dtype=np.complex128)
    z = None

    y = np.zeros(Nt, dtype=np.complex128)
    qnorm = np.zeros(Nt, dtype=float)
    control_on = np.zeros(Nt, dtype=bool)

    switched = False

    for k in range(Nt):
        y[k] = np.vdot(c, q)
        qnorm[k] = np.sqrt(np.real(np.vdot(q, M_sp @ q)))

        if k == Nt - 1:
            break

        if t[k] < float(t_on):
            rhs = M_sp @ q + float(dt) * bw_q * eta[k]
            q = L_open(rhs)
        else:
            if not switched:
                z = np.zeros(ncl, dtype=np.complex128)
                z[:nplant] = q
                switched = True

            rhs = Ecl_sp @ z + float(dt) * bw_cl * eta[k]
            z = L_cl(rhs)
            q = z[:nplant]
            control_on[k + 1] = True

    return {
        "t": t,
        "eta": eta,
        "y": y,
        "abs_y": np.abs(y),
        "qnorm": qnorm,
        "t_on": float(t_on),
        "a_best": a_best,
        "n_best": n_best,
        "source": source,
    }


def plot_noise_response_controller_switch(
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="19_noise_response_controller_switch",
    r=4,
    t_final=100.0,
    dt=0.02,
    t_on=40.0,
    noise_amp=1.0,
    noise_tau=0.25,
    noise_seed=4,
):
    data = simulate_noise_response_with_controller_switch(
        r=r,
        t_final=t_final,
        dt=dt,
        t_on=t_on,
        noise_amp=noise_amp,
        noise_tau=noise_tau,
        noise_seed=noise_seed,
    )

    if not _sw_is_root():
        return None

    t = data["t"]
    eta = data["eta"]
    abs_y = data["abs_y"]
    qnorm = data["qnorm"]
    t_on = data["t_on"]

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 9.2), sharex=True)

    axes[0].plot(t, eta, linewidth=1.0)
    axes[0].axvline(t_on, linestyle="--", linewidth=1.5)
    axes[0].set_ylabel(r"disturbance $\eta(t)$")
    axes[0].set_title("Noisy forcing is applied first, then the optimized controller is switched on")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].semilogy(t, np.maximum(abs_y, 1e-16), linewidth=1.4)
    axes[1].axvline(t_on, linestyle="--", linewidth=1.5,
                    label=r"controller on: $t=t_{\mathrm{on}}$")
    axes[1].set_ylabel(r"sensor amplitude $|y(t)|$")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.35)
    axes[1].legend(loc="best", fontsize=9)

    axes[2].semilogy(t, np.maximum(qnorm, 1e-16), linewidth=1.4)
    axes[2].axvline(t_on, linestyle="--", linewidth=1.5)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel(r"state energy $\|q(t)\|_M$")
    axes[2].grid(True, which="both", linestyle="--", alpha=0.35)

    eq_text = (
        r"$M\dot q=Aq+B_w\eta(t)$,  $t<t_{\mathrm{on}}$" "\n"
        r"$E_{\mathrm{cl}}\dot z=A_{\mathrm{cl}}(\theta_\star)z+B_{w,\mathrm{cl}}\eta(t)$,  "
        r"$t\geq t_{\mathrm{on}}$"
    )

    axes[1].text(
        0.02,
        0.05,
        eq_text,
        transform=axes[1].transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    fig.tight_layout()
    return _sw_savefig(fig, outdir, basename)


# -----------------------------------------------------------------------------
# Plot 20: frequency-response curves flattening over optimizer iterations
# -----------------------------------------------------------------------------

def _select_history_snapshots(hist, nsnap=6):
    """
    Select a few stable, finite-Hinf iterates from one PyGRANSO history.
    """
    good = [
        h for h in hist
        if (not h.get("skipped_hinf", False))
        and np.isfinite(h.get("f_raw", np.nan))
        and ("a" in h)
        and ("n" in h)
    ]

    if len(good) == 0:
        return []

    if len(good) <= int(nsnap):
        return good

    idx = np.unique(np.round(np.linspace(0, len(good) - 1, int(nsnap))).astype(int))
    return [good[i] for i in idx]


def _get_best_start_history_with_coefficients(nsnap=6):
    """
    Returns selected iteration records containing a and n.

    This requires the in-memory hinf_r4_noK_multistart_run object. The old CSV
    does not store a and n, so it cannot reconstruct per-iteration curves.
    """
    if "hinf_r4_noK_multistart_run" not in globals():
        return []

    run = run_hinf_r4_noK_true_multistart_openloop
    if run is None or run.get("best_result", None) is None:
        return []

    hist = run["best_result"].get("history", [])
    return _select_history_snapshots(hist, nsnap=nsnap)


def _signed_frequency_grid(wmin=1e-3, wmax=5e1, npos=160):
    if "symmetric_log_frequency_grid" in globals():
        return symmetric_log_frequency_grid(wmin, wmax, npos)

    wp = np.geomspace(float(wmin), float(wmax), int(npos) // 2)
    return np.concatenate((-wp[::-1], wp))


def _gain_curve_for_theta(cl_dyn, theta, ws):
    """
    Compute gain curve for a closed-loop controller theta.
    This is the expensive part, so keep ws modest.
    """
    Acl = cl_dyn.Acl(theta)
    Ecl = cl_dyn.Ecl()

    tasks = [float(w) for w in ws]

    def _one(w):
        return w, float(gain_of_omega(Acl, Ecl, w))

    if "mpi_task_map" in globals():
        pairs = mpi_task_map(
            _one,
            tasks,
            label="flattening_gain_curve",
            use_parallel=True,
        )
    else:
        pairs = [_one(w) for w in tasks]

    pairs = sorted(pairs, key=lambda p: p[0])
    return (
        np.array([p[0] for p in pairs], dtype=float),
        np.array([p[1] for p in pairs], dtype=float),
    )


def plot_frequency_response_flattening_over_iterations(
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="20_frequency_response_flattening",
    r=4,
    nsnap=6,
    wmin=1e-3,
    wmax=5e1,
    npos=160,
):
    """
    Show how the closed-loop frequency-response curve gets flattened over
    PyGRANSO iterations.

    Needs in-memory history with coefficients a and n.
    """
    snapshots = _get_best_start_history_with_coefficients(nsnap=nsnap)

    if len(snapshots) == 0:
        if _sw_is_root():
            print(
                "\n[flattening plot skipped]\n"
                "I could not find per-iteration a/n coefficients in memory.\n"
                "The existing history CSV does not store them, so the full curves cannot be reconstructed.\n"
                "To make this plot, run it in the same script execution as the multistart, or save a/n in the history.\n",
                flush=True,
            )
        return None

    cl_dyn = _make_cldyn_for_switch_plot(r=r)
    ws = _signed_frequency_grid(wmin=wmin, wmax=wmax, npos=npos)

    curves = []

    for h in snapshots:
        a = np.asarray(h["a"], dtype=np.complex128).reshape(r)
        nvec = np.asarray(h["n"], dtype=np.complex128).reshape(r)
        theta = cl_dyn.pack_theta(a, nvec, Kstat=0.0)

        w_curve, g_curve = _gain_curve_for_theta(cl_dyn, theta, ws)

        curves.append({
            "call": int(h["call"]),
            "f_raw": float(h["f_raw"]),
            "omega_star": float(h["omega_star"]),
            "w": w_curve,
            "g": g_curve,
        })

    if not _sw_is_root():
        return None

    fig, ax = plt.subplots(figsize=(10.6, 6.2))

    for j, cdat in enumerate(curves):
        label = fr"call {cdat['call']}, peak $\approx {cdat['f_raw']:.2e}$"
        lw = 1.1 if j < len(curves) - 1 else 2.4
        alpha = 0.65 if j < len(curves) - 1 else 1.0

        ax.plot(
            cdat["w"],
            cdat["g"],
            linewidth=lw,
            alpha=alpha,
            label=label,
        )

        if np.isfinite(cdat["omega_star"]):
            ax.scatter(
                [cdat["omega_star"]],
                [cdat["f_raw"]],
                marker="o" if j < len(curves) - 1 else "*",
                s=45 if j < len(curves) - 1 else 140,
                zorder=5,
            )

    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_yscale("log")
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel(r"closed-loop gain $\|G_{\mathrm{cl}}(i\omega)\|$")
    ax.set_title(r"Optimizer flattens the active $H_\infty$ peak over iterations")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)

    note = (
        r"$H_\infty=\max_\omega \|G_{\mathrm{cl}}(i\omega)\|$" "\n"
        r"Optimization reduces and flattens the upper envelope."
    )
    ax.text(
        0.02,
        0.04,
        note,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    fig.tight_layout()
    return _sw_savefig(fig, outdir, basename)


# -----------------------------------------------------------------------------
# master wrapper
# -----------------------------------------------------------------------------

def make_switch_and_flattening_plots(
    outdir="/stck/eclement/Control Of NavierStokes/plots",
):
    paths = {}

    paths["noise_switch"] = plot_noise_response_controller_switch(
        outdir=outdir,
        r=int(os.environ.get("SWITCH_R", "4")),
        t_final=float(os.environ.get("SWITCH_T_FINAL", "100")),
        dt=float(os.environ.get("SWITCH_DT", "0.02")),
        t_on=float(os.environ.get("SWITCH_T_ON", "40")),
        noise_amp=float(os.environ.get("SWITCH_NOISE_AMP", "1.0")),
        noise_tau=float(os.environ.get("SWITCH_NOISE_TAU", "0.25")),
        noise_seed=int(os.environ.get("SWITCH_NOISE_SEED", "4")),
    )

    paths["flattening"] = plot_frequency_response_flattening_over_iterations(
        outdir=outdir,
        r=int(os.environ.get("SWITCH_R", "4")),
        nsnap=int(os.environ.get("FLATTEN_NSNA", "6")),
        wmin=float(os.environ.get("FLATTEN_WMIN", "1e-3")),
        wmax=float(os.environ.get("FLATTEN_WMAX", "5e1")),
        npos=int(os.environ.get("FLATTEN_NPOS", "160")),
    )

    if _sw_is_root():
        print("\nSwitch/flattening plots written to:", outdir, flush=True)
        for k, v in paths.items():
            print(" ", k, "->", v, flush=True)

    return paths


if 1==1:
    switch_and_flattening_paths = make_switch_and_flattening_plots(
        outdir="/stck/eclement/Control Of NavierStokes/plots",
    )



# =============================================================================
# Two extra visual plots:
#
# 19_noise_switch_controller_response:
#   time-domain disturbance/noise response; vertical line = controller ON.
#
# 20_gain_curve_flattening:
#   closed-loop gain curves being flattened.
#   Uses actual PyGRANSO iterates if hinf_r4_noK_multistart_run is in memory.
#   Otherwise falls back to open-loop -> best-controller continuation.
#
# Run:
#   RUN_TRUE_MULTISTART=0 MAKE_SWITCH_RESPONSE_PLOT=1 python plot.py
#   RUN_TRUE_MULTISTART=0 MAKE_FLATTENING_PLOT=1 python plot.py
# =============================================================================

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc


def _extra_is_root():
    return ("IS_WORLD_ROOT" not in globals()) or bool(IS_WORLD_ROOT)


def _extra_barrier():
    if "WORLD" in globals():
        WORLD.Barrier()


def _extra_env_flag(name, default=False):
    if "_env_flag" in globals():
        return _env_flag(name, default)
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _extra_mkdir(outdir):
    if _extra_is_root():
        os.makedirs(outdir, exist_ok=True)
    _extra_barrier()


def _extra_savefig(fig, outdir, basename):
    _extra_mkdir(outdir)
    if not _extra_is_root():
        return None

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def _extra_make_lu_ksp(S):
    if "make_lu_ksp" in globals():
        return make_lu_ksp(S)

    ksp = PETSc.KSP().create(S.comm)
    ksp.setOperators(S)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorShift(shift_type=PETSc.Mat.FactorShiftType.NONZERO, amount=1e-12)
    ksp.setUp()
    ksp.setErrorIfNotConverged(True)
    return ksp


def _parse_float_from_text(text, key, default=np.nan):
    num = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    m = re.search(re.escape(key) + r"\s*=\s*" + num, text)
    return float(m.group(1)) if m else default


def _parse_complex_vector_between(text, key, next_key):
    m = re.search(
        re.escape(key) + r"\s*=\s*(\[[\s\S]*?\])\s*" + re.escape(next_key),
        text,
    )
    if not m:
        raise RuntimeError(f"Could not parse '{key}' from global best file.")

    block = m.group(1)

    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    pairs = re.findall(rf"({num})\s*({num})j", block)

    vals = []
    for re_part, im_part in pairs:
        vals.append(float(re_part) + 1j * float(im_part.replace(" ", "")))

    if len(vals) == 0:
        raise RuntimeError(f"Found '{key}', but no complex entries were parsed.")

    return np.asarray(vals, dtype=np.complex128)


def load_best_noK_controller(
    txt_name="hinf_r4_noK_true_multistart_global_best.txt",
    r=4,
):
    """
    Loads the best no-K controller from memory if possible, otherwise from the
    global-best txt file written by the multistart run.
    """

    # Fast path: use the in-memory multistart result if it exists.
    if "hinf_r4_noK_multistart_run" in globals():
        run = run_hinf_r4_noK_true_multistart_openloop
        if run is not None and run.get("best_result", None) is not None:
            best = run["best_result"]["best"]
            if best is not None:
                return {
                    "a": np.asarray(best["a"], dtype=np.complex128).reshape(r),
                    "n": np.asarray(best["n"], dtype=np.complex128).reshape(r),
                    "K": float(best.get("K", 0.0)),
                    "xa": float(os.environ.get("HINF_R4_NOK_XA", "-0.98")),
                    "xs": float(os.environ.get("HINF_R4_NOK_XS", "0.95")),
                    "source": "in-memory hinf_r4_noK_multistart_run",
                }

    if not os.path.exists(txt_name):
        raise RuntimeError(
            f"Could not find {txt_name}. Either keep the multistart result in "
            "memory or run this in the directory where the global-best txt file exists."
        )

    with open(txt_name, "r") as f:
        text = f.read()

    a = _parse_complex_vector_between(text, "best a", "best n").reshape(r)
    nvec = _parse_complex_vector_between(text, "best n", "best omega_star").reshape(r)

    return {
        "a": a,
        "n": nvec,
        "K": _parse_float_from_text(text, "fixed K", default=0.0),
        "xa": _parse_float_from_text(text, "fixed xa", default=float(os.environ.get("HINF_R4_NOK_XA", "-0.98"))),
        "xs": _parse_float_from_text(text, "fixed xs", default=float(os.environ.get("HINF_R4_NOK_XS", "0.95"))),
        "source": txt_name,
    }


def companion_L_from_a(a):
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    r = len(a)
    L = np.zeros((r, r), dtype=np.complex128)
    for i in range(r - 1):
        L[i, i + 1] = 1.0
    L[-1, :] = -a
    return L


def build_best_input_output_vectors(best):
    b_vec, _ = build_actuator_vector(
        float(best["xa"]),
        sigma_gauss,
        V,
        phi,
        bc,
        is_free,
    )

    s_vec_best, _ = build_sensor_state_vector(
        float(best["xs"]),
        sigma_gauss,
        V,
        is_free,
    )

    c_vec = sensor_output_vector(M, s_vec_best)
    return b_vec, c_vec


def moving_rms(x, window):
    x = np.asarray(x, dtype=float)
    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.sqrt(np.convolve(x**2, kernel, mode="same"))


def simulate_noise_with_controller_switch(
    A,
    M,
    best,
    t_final=80.0,
    t_on=30.0,
    dt=0.02,
    noise_amp=1.0,
    noise_smooth_time=0.20,
    rms_window_time=2.0,
    seed=3,
):
    """
    Simulates

        M qdot = A q + b [d(t) + u(t)]
        y      = c^H q

    with controller off before t_on and controller on after t_on:

        u = 0,                                  t < t_on
        xKdot = L(a) xK + e_r y,
        u = n^H xK + K y,                       t >= t_on

    The plant time stepping is linearly implicit:

        (M - dt A) q_{k+1} = M q_k + dt b [d_k + u_k].
    """

    a = np.asarray(best["a"], dtype=np.complex128).reshape(-1)
    nvec = np.asarray(best["n"], dtype=np.complex128).reshape(-1)
    K = float(best.get("K", 0.0))
    r = len(a)

    b_vec, c_vec = build_best_input_output_vectors(best)

    L = companion_L_from_a(a)
    er = np.zeros(r, dtype=np.complex128)
    er[-1] = 1.0

    nt = int(np.ceil(float(t_final) / float(dt))) + 1
    ts = np.arange(nt, dtype=float) * float(dt)

    rng = np.random.default_rng(int(seed))
    d = rng.normal(size=nt)

    smooth_n = max(1, int(round(float(noise_smooth_time) / float(dt))))
    if smooth_n > 1:
        d = np.convolve(d, np.ones(smooth_n) / smooth_n, mode="same")

    d = float(noise_amp) * d / (np.std(d) + 1e-30)

    S = M.copy()
    S.axpy(-float(dt), A)
    S.assemble()
    ksp = _extra_make_lu_ksp(S)

    q = M.createVecRight()
    q.set(0.0)

    q_next = q.duplicate()
    rhs = q.duplicate()
    xK = np.zeros(r, dtype=np.complex128)

    y_hist = np.zeros(nt, dtype=np.complex128)
    u_hist = np.zeros(nt, dtype=np.complex128)
    on_hist = np.zeros(nt, dtype=bool)

    for k, t in enumerate(ts):
        y = c_vec.dot(q)  # c^H q

        if t >= float(t_on):
            on_hist[k] = True
            u = np.vdot(nvec, xK) + K * y
            xdot = L @ xK + er * y
            xK = xK + float(dt) * xdot
        else:
            u = 0.0 + 0.0j

        M.mult(q, rhs)
        rhs.axpy(float(dt) * (complex(d[k]) + complex(u)), b_vec)

        ksp.solve(rhs, q_next)
        q_next.copy(q)

        y_hist[k] = c_vec.dot(q)
        u_hist[k] = u

    win = max(1, int(round(float(rms_window_time) / float(dt))))

    return {
        "t": ts,
        "d": d,
        "y": y_hist,
        "u": u_hist,
        "y_abs": np.abs(y_hist),
        "y_rms": moving_rms(np.abs(y_hist), win),
        "u_abs": np.abs(u_hist),
        "on": on_hist,
        "t_on": float(t_on),
        "dt": float(dt),
        "best": best,
    }


def plot_noise_switch_controller_response(
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="19_noise_switch_controller_response",
    t_final=80.0,
    t_on=30.0,
    dt=0.02,
    noise_amp=1.0,
    noise_smooth_time=0.20,
    rms_window_time=2.0,
    seed=3,
):
    best = load_best_noK_controller(r=4)

    sim = simulate_noise_with_controller_switch(
        A,
        M,
        best,
        t_final=t_final,
        t_on=t_on,
        dt=dt,
        noise_amp=noise_amp,
        noise_smooth_time=noise_smooth_time,
        rms_window_time=rms_window_time,
        seed=seed,
    )

    _extra_mkdir(outdir)
    if not _extra_is_root():
        return sim, None

    t = sim["t"]
    d = sim["d"]
    y_abs = sim["y_abs"]
    y_rms = sim["y_rms"]
    u_abs = sim["u_abs"]

    pre = (t > 0.25 * t_on) & (t < t_on)
    post = t > (t_on + 0.25 * (t_final - t_on))

    pre_rms = float(np.mean(y_rms[pre])) if np.any(pre) else np.nan
    post_rms = float(np.mean(y_rms[post])) if np.any(post) else np.nan

    fig = plt.figure(figsize=(11.0, 8.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.95, 1.0, 1.35, 1.0])

    ax_eq = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1])
    ax_y = fig.add_subplot(gs[2], sharex=ax_d)
    ax_u = fig.add_subplot(gs[3], sharex=ax_d)

    ax_eq.axis("off")
    eq_text = (
        r"$M\dot q(t)=Aq(t)+b\,[d(t)+u(t)],\qquad y(t)=c^{H}q(t)$" "\n"
        r"$u(t)=0,\quad t<t_{\rm on}$" "\n"
        r"$\dot x_K(t)=L(a)x_K(t)+e_r y(t),\qquad "
        r"u(t)=n^{H}x_K(t)+K y(t),\quad t\ge t_{\rm on}$"
    )
    ax_eq.text(
        0.02,
        0.50,
        eq_text,
        transform=ax_eq.transAxes,
        va="center",
        ha="left",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.45", alpha=0.10),
    )

    ax_d.plot(t, d, linewidth=0.9)
    ax_d.axvline(t_on, linestyle="--", linewidth=1.5,
                 label=r"controller on at $t_{\rm on}$")
    ax_d.set_ylabel(r"noise $d(t)$")
    ax_d.set_title("Disturbance injected in the actuator/input channel")
    ax_d.grid(True, linestyle="--", alpha=0.35)
    ax_d.legend(loc="best", fontsize=9)

    ax_y.plot(t, y_abs, linewidth=0.75, alpha=0.45,
              label=r"instantaneous $|y(t)|$")
    ax_y.plot(t, y_rms, linewidth=2.0,
              label=fr"moving RMS, before={pre_rms:.3e}, after={post_rms:.3e}")
    ax_y.axvline(t_on, linestyle="--", linewidth=1.5)
    ax_y.set_yscale("log")
    ax_y.set_ylabel(r"measured response")
    ax_y.set_title("The controller is switched on after the transient has developed")
    ax_y.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_y.legend(loc="best", fontsize=9)

    ax_u.plot(t, u_abs, linewidth=1.2)
    ax_u.axvline(t_on, linestyle="--", linewidth=1.5)
    ax_u.set_xlabel("time")
    ax_u.set_ylabel(r"$|u(t)|$")
    ax_u.set_title("Control effort after switch-on")
    ax_u.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(r"Time-domain noise response: open loop first, optimized controller after switch-on", y=0.995)
    fig.tight_layout()

    paths = _extra_savefig(fig, outdir, basename)
    return sim, paths


# -----------------------------------------------------------------------------
# Gain curve flattening plot
# -----------------------------------------------------------------------------

def _extra_signed_log_grid(wmin=1e-3, wmax=5e1, npos=80):
    wp = np.logspace(np.log10(float(wmin)), np.log10(float(wmax)), int(npos))
    return np.concatenate((-wp[::-1], [0.0], wp))


def _make_cldyn_for_best(best, r=4):
    return make_cldyn_for_placement(
        xa=float(best["xa"]),
        xs=float(best["xs"]),
        A=A,
        M=M,
        V=V,
        phi=phi,
        bc=bc,
        is_free=is_free,
        sigma_gauss=sigma_gauss,
        r=r,
    )


def _gain_curve_for_theta(cl_dyn, theta, ws):
    Acl = cl_dyn.Acl(theta)
    Ecl = cl_dyn.Ecl()

    vals = []
    for w in ws:
        vals.append(float(gain_of_omega(Acl, Ecl, float(w))))

    return np.asarray(vals, dtype=float)


def _actual_history_if_available(r=4):
    if "hinf_r4_noK_multistart_run" not in globals():
        return None

    run = run_hinf_r4_noK_true_multistart_openloop
    if run is None or run.get("best_result", None) is None:
        return None

    hist = run["best_result"].get("history", [])
    out = []
    for h in hist:
        if (
            "a" in h and "n" in h
            and np.isfinite(h.get("f_raw", np.nan))
            and not bool(h.get("skipped_hinf", False))
        ):
            out.append(h)

    return out if len(out) > 0 else None


def plot_controller_gain_curve_flattening(
    outdir="/stck/eclement/Control Of NavierStokes/plots",
    basename="20_gain_curve_flattening",
    wmin=1e-3,
    wmax=5e1,
    npos=70,
    ncurves=6,
    use_actual_history_if_available=True,
):
    """
    If actual in-memory PyGRANSO history exists:
        plot true gain curves at selected oracle calls.

    Otherwise:
        plot a no-rerun visual continuation from open loop to the best controller:
            a(t) = a_open + t(a_best-a_open)
            n(t) = 0      + t(n_best-0)
    """

    best = load_best_noK_controller(r=4)
    cl_dyn = _make_cldyn_for_best(best, r=4)
    ws = _extra_signed_log_grid(wmin=wmin, wmax=wmax, npos=npos)

    actual_hist = None
    if bool(use_actual_history_if_available):
        actual_hist = _actual_history_if_available(r=4)

    curves = []

    if actual_hist is not None:
        idx = np.linspace(0, len(actual_hist) - 1, min(int(ncurves), len(actual_hist)))
        idx = np.unique(np.round(idx).astype(int))

        for j in idx:
            h = actual_hist[int(j)]
            theta = cl_dyn.pack_theta(h["a"], h["n"], Kstat=float(h.get("K", 0.0)))
            vals = _gain_curve_for_theta(cl_dyn, theta, ws)
            curves.append({
                "label": fr"call {int(h['call'])}, $f={float(h['f_raw']):.3g}$",
                "vals": vals,
                "kind": "actual",
            })

        mode = "actual PyGRANSO iterates"

    else:
        a_best = np.asarray(best["a"], dtype=np.complex128).reshape(4)
        n_best = np.asarray(best["n"], dtype=np.complex128).reshape(4)

        a_open = np.asarray(default_stable_a_center(4), dtype=np.complex128).reshape(4)
        n_open = np.zeros(4, dtype=np.complex128)

        ts = np.linspace(0.0, 1.0, int(ncurves))

        for t in ts:
            a_t = a_open + t * (a_best - a_open)
            n_t = n_open + t * (n_best - n_open)
            theta = cl_dyn.pack_theta(a_t, n_t, Kstat=float(best.get("K", 0.0)))
            vals = _gain_curve_for_theta(cl_dyn, theta, ws)
            curves.append({
                "label": fr"$s={t:.2f}$",
                "vals": vals,
                "kind": "continuation",
            })

        mode = "open-loop to best-controller continuation"

    _extra_mkdir(outdir)
    if not _extra_is_root():
        return {"mode": mode, "curves": curves}, None

    finite_ws = np.asarray(ws, dtype=float)

    nz = np.abs(finite_ws[np.abs(finite_ws) > 0.0])
    linthresh = max(float(np.min(nz)) if nz.size else 1e-6, 1e-12)

    fig, ax = plt.subplots(figsize=(10.8, 5.9))

    for j, c in enumerate(curves):
        vals = c["vals"]
        lw = 1.0 + 1.3 * j / max(1, len(curves) - 1)
        alpha = 0.45 + 0.50 * j / max(1, len(curves) - 1)
        ax.plot(finite_ws, vals, linewidth=lw, alpha=alpha, label=c["label"])

        kmax = int(np.nanargmax(vals))
        ax.scatter([finite_ws[kmax]], [vals[kmax]], s=35, zorder=5)

    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("log")
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel(r"closed-loop gain $\|G_{\rm cl}(i\omega)\|$")
    ax.set_title("Gain curve being flattened during controller improvement")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)

    subtitle = (
        r"Actual selected PyGRANSO iterates"
        if mode == "actual PyGRANSO iterates"
        else r"No-rerun visual continuation from open loop to saved best controller"
    )
    fig.suptitle(subtitle, y=0.995)
    fig.tight_layout()

    paths = _extra_savefig(fig, outdir, basename)
    return {"mode": mode, "curves": curves}, paths


# -----------------------------------------------------------------------------
# Auto-run flags
# -----------------------------------------------------------------------------

if 1 ==1:
    switch_response_report, switch_response_paths = plot_noise_switch_controller_response(
        outdir="/stck/eclement/Control Of NavierStokes/plots",
        t_final=float(os.environ.get("SWITCH_TFINAL", "80.0")),
        t_on=float(os.environ.get("SWITCH_TON", "30.0")),
        dt=float(os.environ.get("SWITCH_DT", "0.02")),
        noise_amp=float(os.environ.get("SWITCH_NOISE_AMP", "1.0")),
        noise_smooth_time=float(os.environ.get("SWITCH_NOISE_SMOOTH", "0.20")),
        rms_window_time=float(os.environ.get("SWITCH_RMS_WINDOW", "2.0")),
        seed=int(os.environ.get("SWITCH_SEED", "3")),
    )
    if _extra_is_root():
        print("Switch-response plot:", switch_response_paths, flush=True)


    flattening_report, flattening_paths = plot_controller_gain_curve_flattening(
        outdir="/stck/eclement/Control Of NavierStokes/plots",
        wmin=float(os.environ.get("FLAT_WMIN", "1e-3")),
        wmax=float(os.environ.get("FLAT_WMAX", "5e1")),
        npos=int(os.environ.get("FLAT_NPOS", "70")),
        ncurves=int(os.environ.get("FLAT_NCURVES", "6")),
        use_actual_history_if_available=_extra_env_flag("FLAT_USE_ACTUAL_HISTORY", True),
    )
    if _extra_is_root():
        print("Flattening plot mode:", flattening_report["mode"], flush=True)
        print("Flattening plot:", flattening_paths, flush=True)