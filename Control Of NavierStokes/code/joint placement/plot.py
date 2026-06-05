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

def trace_inner_check_openloop(A, M, gamma, b=10.0, eigs_tol=1e-8, nev=60, eps_eigs=1e-6, max_visits=200):
    """
    Gamma is a constant that acts as a horizontal line. we want to know whether gamma crosses the gain curve. We pick gamma = (gamma_low + gamma_high) / 2 
    because we are doing a bisection. If gamma_low = 8 and gamma_high = 10 we test gamma = 9. If the Hamiltonian has imaginary eigenvalues then gamma_low becomes that gamma else gamma_high becomses the new gamma.
    Until gamma is small enough.
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


def plot_interval_exclusion_trace(trace, outdir="hinf_explain_plots",
                                  basename="02_interval_exclusion_trace"):
    """Plot interval bisection/exclusion visits from trace_inner_check_openloop."""
    if not IS_WORLD_ROOT:
        return None

    hinf_root_mkdir(outdir)
    events = trace.get("events", [])
    if len(events) == 0:
        return None

    fig_h = max(4.5, min(12.0, 0.22 * len(events) + 3.0))
    fig, ax = plt.subplots(figsize=(9.0, fig_h))

    for e in events:
        y = int(e["visit"])
        ax.hlines(y, e["lo"], e["hi"], linewidth=3.0, label="tested interval" if y == 0 else None)
        ax.scatter([e["theta"]], [y], s=28, zorder=4, label=r"shift $i\theta$" if y == 0 else None)
        if np.isfinite(e.get("removed_lo", np.nan)) and np.isfinite(e.get("removed_hi", np.nan)):
            ax.hlines(y + 0.18, e["removed_lo"], e["removed_hi"], linewidth=7.0,
                      alpha=0.35, label="excluded region" if y == 0 else None)

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel("interval visit")
    ax.set_title(fr"Boyd disk/certificate interval search for trial $\gamma={trace['gamma']:.4g}$")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    ax.invert_yaxis()
    fig.tight_layout()

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    csv_path = os.path.join(outdir, f"{basename}_data.csv")
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    plt.close(fig)


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
    ax.set_xlabel(r"$\operatorname{Re}(\lambda)$")
    ax.set_ylabel(r"$\operatorname{Im}(\lambda)$")
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
    Main convenience function. It creates:
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