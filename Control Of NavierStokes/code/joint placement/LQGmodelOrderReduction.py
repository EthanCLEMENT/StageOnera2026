####################################################################################################
# Balanced-truncation study for the full Rowley--Chen LQG controller.
#
# Paste this block near the end of yo.py AFTER the FE/PETSc matrices and helper functions exist,
# and BEFORE any expensive PyGRANSO run you do not want to repeat.
#
# Purpose:
#   1) Build the full-order Rowley--Chen LQG controller on the current FE model.
#   2) Balanced-truncate the CONTROLLER K(s), not the plant.
#   3) Close each reduced K_r(s) around the original FE plant.
#   4) Compare alpha, signed-frequency Hinf(q->q) in the same M-energy norm as yo.py,
#      plus Ms/Mt/KS loop metrics.
#
# IMPORTANT:
#   The dense CARE + dense Lyapunov balanced truncation part is still performed once on WORLD rank 0.
#   The expensive closed-loop validation is MPI-parallelized when this block is run in an environment
#   where the FE/PETSc matrices are local to each MPI rank (i.e. A.comm.getSize() == 1 on every rank).
#   This is the same "task group size = 1" layout used in the MPI-parallel yo.py/par.py variants.
####################################################################################################

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
import os
from mpi4py import MPI

WORLD = MPI.COMM_WORLD
WORLD_RANK = WORLD.Get_rank()
WORLD_SIZE = WORLD.Get_size()

BT_GROUP_SIZE = int(os.environ.get("BT_GROUP_SIZE", "1"))
BT_GROUP_SIZE = max(1, min(BT_GROUP_SIZE, WORLD_SIZE))
BT_GROUP_ID = WORLD_RANK // BT_GROUP_SIZE
BT_NUM_GROUPS = (WORLD_SIZE + BT_GROUP_SIZE - 1) // BT_GROUP_SIZE

FE_COMM = WORLD.Split(color=BT_GROUP_ID, key=WORLD_RANK)
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

        if self.comm.rank == 0:
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

prof = Profiler(MPI.COMM_WORLD)
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
        if self.comm.rank == 0:
            total = sum(out.values())
            print("\n=== Timing summary (max over ranks) ===")
            for k in out:
                print(f"{k:12s}: {out[k]:.6f} s  ({100*out[k]/(total+1e-30):.1f}%)")
            print(f"{'total':12s}: {total:.6f} s")
        return out

timer = SectionTimer(MPI.COMM_WORLD)

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

domain = mesh.create_interval(MPI.COMM_WORLD, Nx,[x_min, x_max])
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

    if M.comm.rank == 0:
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
    if M.comm.rank == 0:
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
    if N.comm.rank == 0:
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

# if A.comm.rank == 0:
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
# if A.comm.rank == 0:
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

import csv as _bt_csv
from dataclasses import dataclass as _bt_dataclass
import os as _bt_os
import numpy as _bt_np
import scipy.linalg as _bt_la
import scipy.optimize as _bt_opt

try:
    from mpi4py import MPI as _bt_MPI
except Exception:  # pragma: no cover - keeps plain serial imports usable
    _bt_MPI = None


# -----------------------------------------------------------------------------------------------
# MPI helpers
# -----------------------------------------------------------------------------------------------
if _bt_MPI is not None:
    _BT_WORLD = _bt_MPI.COMM_WORLD
    _BT_WORLD_RANK = _BT_WORLD.Get_rank()
    _BT_WORLD_SIZE = _BT_WORLD.Get_size()
else:
    _BT_WORLD = None
    _BT_WORLD_RANK = 0
    _BT_WORLD_SIZE = 1


def _bt_env_flag(name, default=False):
    val = _bt_os.environ.get(name, None)
    if val is None:
        return bool(default)
    return str(val).strip().lower() not in ("0", "false", "no", "off", "")


def _bt_env_int(name, default):
    try:
        return int(_bt_os.environ.get(name, default))
    except Exception:
        return int(default)


def _bt_is_world_root():
    return _BT_WORLD_RANK == 0


# BT_GROUP_SIZE controls frequency/sample parallelism within each controller order.
# With BT_GROUP_SIZE=1, ranks split controller orders.  With BT_GROUP_SIZE>1, groups split
# controller orders, and ranks inside each group split frequency grids for that order.
_BT_REQUESTED_GROUP_SIZE = max(1, _bt_env_int("BT_GROUP_SIZE", 1))
_BT_GROUP_SIZE = max(1, min(_BT_REQUESTED_GROUP_SIZE, _BT_WORLD_SIZE))
_BT_GROUP_ID = _BT_WORLD_RANK // _BT_GROUP_SIZE
_BT_NUM_GROUPS = (_BT_WORLD_SIZE + _BT_GROUP_SIZE - 1) // _BT_GROUP_SIZE
if _bt_MPI is not None:
    _BT_GROUP_COMM = _BT_WORLD.Split(color=_BT_GROUP_ID, key=_BT_WORLD_RANK)
    _BT_GROUP_RANK = _BT_GROUP_COMM.Get_rank()
    _BT_GROUP_ACTUAL_SIZE = _BT_GROUP_COMM.Get_size()
else:
    _BT_GROUP_COMM = None
    _BT_GROUP_RANK = 0
    _BT_GROUP_ACTUAL_SIZE = 1


def _bt_is_group_root():
    return _BT_GROUP_RANK == 0


def _bt_can_split_work():
    """True when PETSc objects are local per rank, so MPI ranks may do independent solves."""
    if _BT_WORLD_SIZE <= 1:
        return False
    if "A" not in globals():
        return False
    try:
        return A.comm.getSize() == 1
    except Exception:
        return False


def _bt_check_parallel_layout_or_raise():
    """Fail early for the dangerous case: WORLD-size MPI with distributed A/M.

    The dense CARE conversion and the task-parallel validation both require each worker rank to
    own a complete local copy of A/M.  If A was assembled on MPI.COMM_WORLD, PETSc/SLEPc calls are
    collective and cannot be split independently by frequency/order here.
    """
    if _BT_WORLD_SIZE > 1:
        try:
            petsc_size = A.comm.getSize()
        except Exception:
            petsc_size = None
        if petsc_size != 1:
            raise RuntimeError(
                "MPI-parallel balanced-reduction validation requires A.comm.getSize() == 1 on "
                "each rank.  Build the FE/PETSc matrices on a per-rank/task communicator "
                "(for example MPI_TASK_GROUP_SIZE=1 / FE_COMM=COMM_SELF style), or run this "
                "append block serially.  Refusing to split work because the current A/M appear "
                "to be distributed PETSc objects."
            )


def _bt_progress(msg, *, group=False):
    if not _bt_env_flag("BT_PROGRESS", default=False):
        return
    if group:
        if _bt_is_group_root():
            print(f"[BT group {_BT_GROUP_ID}/{_BT_NUM_GROUPS}] {msg}", flush=True)
    else:
        if _bt_is_world_root():
            print(f"[BT mpi] {msg}", flush=True)


def _bt_group_allgather(obj):
    if _bt_can_split_work() and _BT_GROUP_ACTUAL_SIZE > 1:
        return _BT_GROUP_COMM.allgather(obj)
    return [obj]


def _bt_world_bcast(obj, root=0):
    if _BT_WORLD_SIZE > 1:
        return _BT_WORLD.bcast(obj, root=root)
    return obj


def _bt_world_allgather(obj):
    if _BT_WORLD_SIZE > 1:
        return _BT_WORLD.allgather(obj)
    return [obj]


def _bt_eval_scalar_grid_group(ws, eval_func, label="grid"):
    """Evaluate scalar function on a grid, splitting points inside the current BT group."""
    ws = _bt_np.asarray(ws, dtype=float)
    if not (_bt_can_split_work() and _BT_GROUP_ACTUAL_SIZE > 1):
        vals = []
        for i, w in enumerate(ws):
            if _bt_env_flag("BT_PROGRESS", default=False) and _bt_is_group_root():
                if i == 0 or (i + 1) == len(ws) or (i + 1) % max(1, len(ws) // 10) == 0:
                    print(f"[BT {label}] {i+1}/{len(ws)}", flush=True)
            vals.append(float(eval_func(float(w))))
        return _bt_np.asarray(vals, dtype=float)

    local = []
    errors = []
    for i, w in enumerate(ws):
        if i % _BT_GROUP_ACTUAL_SIZE != _BT_GROUP_RANK:
            continue
        try:
            if _bt_env_flag("BT_PROGRESS", default=False):
                print(f"[BT group {_BT_GROUP_ID} rank {_BT_GROUP_RANK}] {label}: point {i+1}/{len(ws)} w={float(w):+.6e}", flush=True)
            local.append((i, float(eval_func(float(w)))))
        except Exception as e:
            errors.append((i, repr(e)))
            local.append((i, _bt_np.nan))

    chunks = _BT_GROUP_COMM.allgather(local)
    err_chunks = _BT_GROUP_COMM.allgather(errors)
    flat_errors = [e for chunk in err_chunks for e in chunk]
    if flat_errors:
        raise RuntimeError(f"Parallel grid evaluation failed in {label}: {flat_errors[:3]}")

    vals = _bt_np.empty(len(ws), dtype=float)
    vals.fill(_bt_np.nan)
    for chunk in chunks:
        for i, val in chunk:
            vals[int(i)] = float(val)
    if _bt_np.any(~_bt_np.isfinite(vals)):
        bad = _bt_np.where(~_bt_np.isfinite(vals))[0][:10]
        raise RuntimeError(f"Parallel grid evaluation left missing/non-finite values in {label}: {bad}")
    return vals

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
    if comm.rank == 0:
        print("\n[TEST] basic")
        print("A size:", (nA, mA), "type:", A.getType(), "||A||F:", mat_norm(A))
        print("M size:", (nM, mM), "type:", M.getType(), "||M||F:", mat_norm(M))
    assert (nA == mA == nM == mM), "A and M must be square and same size"
    assert np.isfinite(mat_norm(A)) and np.isfinite(mat_norm(M)), "Non-finite matrix norm"

def test_mass_hermitian(M, tol=1e-10):
    ok, rel = is_hermitian_mat(M, tol=tol)
    if M.comm.rank == 0:
        print("\n[TEST] M Hermitian:", ok, "rel_err:", rel)
    # This should be essentially Hermitian for standard FE mass matrices
    if not ok and M.comm.rank == 0:
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
        if comm.rank == 0:
            print(f"[TEST] M positivity trial {k}: v^H M v = {q}")
    if comm.rank == 0:
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
        if comm.rank == 0:
            print(f"\n[TEST] adjoint identity trial {k}: rel_err = {rel}")
        if rel > tol:
            if comm.rank == 0:
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

    if comm.rank == 0:
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

    if comm.rank == 0:
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

    if comm.rank == 0:
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
        if comm.rank == 0:
            print("rel_diff(Mgam(1,1), -A^H) =", br_rel)
        assert br_rel < 1e-10, "Hamiltonian bottom-right block is not -A^H (very likely your current bug)."
    else:
        if comm.rank == 0:
            print("Mgam is not nest type; skipping explicit BR block check.")

    return True
def restrict_vec_to_is(v_full: PETSc.Vec, iset: PETSc.IS) -> PETSc.Vec:
    """
    Return a standalone copy of the subvector v_full[iset].
    """
    subv = v_full.getSubVector(iset)
    out = subv.copy()
    v_full.restoreSubVector(iset, subv)
    return out

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

# -----------------------------------------------------------------------------------------------
# Small helpers / fallbacks
# -----------------------------------------------------------------------------------------------
def _bt_require_serial_petsc(obj, name="this balanced-truncation LQG block"):
    comm = obj.comm if hasattr(obj, "comm") else PETSc.COMM_WORLD
    if comm.getSize() != 1:
        raise RuntimeError(
            f"{name} requires a local serial PETSc object on this MPI worker because it forms "
            f"dense CARE/Lyapunov matrices or dense NumPy arrays.  If using MPI, build A/M on "
            f"per-rank/task communicators so A.comm.getSize() == 1; otherwise run with one rank."
        )


def _bt_petsc_mat_to_numpy_serial(P: PETSc.Mat) -> _bt_np.ndarray:
    _bt_require_serial_petsc(P, "_bt_petsc_mat_to_numpy_serial")
    P = P.convert("aij"); P.assemble()
    n, m = P.getSize()
    rows = _bt_np.arange(n, dtype=PETSc.IntType)
    cols = _bt_np.arange(m, dtype=PETSc.IntType)
    return _bt_np.asarray(P.getValues(rows, cols), dtype=_bt_np.complex128)


def _bt_vec_to_numpy_serial(v: PETSc.Vec) -> _bt_np.ndarray:
    if 'gather_full_vec_numpy' in globals():
        return _bt_np.asarray(gather_full_vec_numpy(v), dtype=_bt_np.complex128).reshape(-1)
    _bt_require_serial_petsc(v, "_bt_vec_to_numpy_serial")
    return _bt_np.asarray(v.getArray(readonly=True), dtype=_bt_np.complex128).copy().reshape(-1)


def _bt_signed_grid(wmin=1e-4, wmax=50.0, ngrid=301):
    wpos = _bt_np.logspace(_bt_np.log10(float(wmin)), _bt_np.log10(float(wmax)), int(ngrid))
    return _bt_np.concatenate((-wpos[::-1], _bt_np.array([0.0]), wpos))


def _bt_make_dense_mat(comm, A_np):
    if 'make_dense_small_mat' in globals():
        return make_dense_small_mat(comm, _bt_np.asarray(A_np, dtype=_bt_np.complex128))
    A_np = _bt_np.asarray(A_np, dtype=_bt_np.complex128)
    n, m = A_np.shape
    P = PETSc.Mat().createAIJ([n, m], comm=comm)
    P.setUp()
    rows = _bt_np.arange(n, dtype=PETSc.IntType)
    cols = _bt_np.arange(m, dtype=PETSc.IntType)
    P.setValues(rows, cols, A_np)
    P.assemble()
    return P


def _bt_build_nested_aij(blocks, sizes, comm):
    if 'build_nested_aij' in globals():
        return build_nested_aij(blocks, sizes, comm)
    isrows = _block_is(comm, sizes)
    iscols = _block_is(comm, sizes)
    Nmat = PETSc.Mat().createNest(blocks, isrows=isrows, iscols=iscols, comm=comm)
    Nmat.assemble()
    Aij = Nmat.convert("aij")
    Aij.assemble()
    return Aij


def _bt_set_subvector(vbig, iset, vsmall):
    if 'set_subvector' in globals():
        set_subvector(vbig, iset, vsmall)
        return
    # serial fallback
    idx = iset.getIndices()
    vals = vsmall.getArray(readonly=True)
    vbig.setValues(idx, vals)
    vbig.assemble()


def _bt_split_vec_by_is(vbig, iset):
    if 'split_vec_by_is' in globals():
        return split_vec_by_is(vbig, iset)
    sub = vbig.getSubVector(iset)
    out = sub.copy()
    vbig.restoreSubVector(iset, sub)
    return out


def _bt_make_rect_outer_vec_row(left: PETSc.Vec, row: _bt_np.ndarray, tol=1e-14) -> PETSc.Mat:
    """Return left * row^H.  To build left*C, pass row=conj(C)."""
    if 'make_rect_outer_vec_row' in globals():
        return make_rect_outer_vec_row(left, row, tol=tol)
    _bt_require_serial_petsc(left, "_bt_make_rect_outer_vec_row")
    l = _bt_vec_to_numpy_serial(left)
    row = _bt_np.asarray(row, dtype=_bt_np.complex128).reshape(-1)
    A_np = _bt_np.outer(l, _bt_np.conjugate(row))
    A_np[_bt_np.abs(A_np) < tol] = 0.0
    return _bt_make_dense_mat(left.comm, A_np)


def _bt_make_rect_outer_col_vecH(col: _bt_np.ndarray, right: PETSc.Vec, tol=1e-14) -> PETSc.Mat:
    """Return col * right^H."""
    if 'make_rect_outer_col_vecH' in globals():
        return make_rect_outer_col_vecH(col, right, tol=tol)
    _bt_require_serial_petsc(right, "_bt_make_rect_outer_col_vecH")
    c = _bt_np.asarray(col, dtype=_bt_np.complex128).reshape(-1)
    r = _bt_vec_to_numpy_serial(right)
    A_np = _bt_np.outer(c, _bt_np.conjugate(r))
    A_np[_bt_np.abs(A_np) < tol] = 0.0
    return _bt_make_dense_mat(right.comm, A_np)


def _bt_build_actuator_and_sensor(xa, xs):
    """Return reduced actuator vector b and sensor-output vector c with y=c^H q."""
    if all(name in globals() for name in ['build_actuator_vector', 'build_sensor_state_vector', 'sensor_output_vector']):
        b_red, _ = build_actuator_vector(float(xa), sigma_gauss, V, phi, bc, is_free)
        s_red, _ = build_sensor_state_vector(float(xs), sigma_gauss, V, is_free)
        c_red = sensor_output_vector(M, s_red)
        return b_red, c_red

    # Fallback for the original FE definitions.
    _bt_require_serial_petsc(A, "_bt_build_actuator_and_sensor fallback")
    b_fun = gaussian_function(V, float(xa), sigma_gauss)
    s_fun = gaussian_function(V, float(xs), sigma_gauss)

    b_full = fem.petsc.assemble_vector(fem.form(b_fun * ufl.conj(phi) * ufl.dx))
    b_full.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b_full, [bc])

    s_full = s_fun.x.petsc_vec.copy()
    s_full.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    b_sub = b_full.getSubVector(is_free); b_red = b_sub.copy(); b_full.restoreSubVector(is_free, b_sub)
    s_sub = s_full.getSubVector(is_free); s_red = s_sub.copy(); s_full.restoreSubVector(is_free, s_sub)

    c_red = M.createVecRight()
    M.mult(s_red, c_red)     # y = s^H M q = c^H q, with c = M s for Hermitian M
    return b_red, c_red


def _bt_siso_tf(P, K):
    Lloop = P * K
    den = 1.0 - Lloop       # positive-feedback convention u=K(s)y, same as yo.py blocks
    return Lloop, 1.0 / den, Lloop / den, K / den, den


def _bt_sqrt_psd_factor(W, reltol=1e-12):
    """Return R such that W ~= R R^H, dropping tiny negative/noisy eigendirections."""
    W = 0.5 * (W + W.conjugate().T)
    vals, vecs = _bt_la.eigh(W)
    vmax = max(float(_bt_np.max(vals.real)), 0.0)
    if vmax <= 0.0:
        raise RuntimeError("Gramian appears non-positive; cannot form square-root factor.")
    keep = vals.real > reltol * vmax
    vals_keep = _bt_np.maximum(vals.real[keep], 0.0)
    vecs_keep = vecs[:, keep]
    return vecs_keep * _bt_np.sqrt(vals_keep)[None, :]


# -----------------------------------------------------------------------------------------------
# Data and full/reduced controller design
# -----------------------------------------------------------------------------------------------
@_bt_dataclass
class BTControllerData:
    label: str
    xa: float
    xs: float
    beta: float
    vhalf: float
    Ar: _bt_np.ndarray       # controller state matrix, r x r
    Br: _bt_np.ndarray       # controller input column, r
    Cr: _bt_np.ndarray       # controller output row, r ; u = Cr xr + D y
    D: complex
    hsv: object = None
    error_bound: object = None

    @property
    def order(self):
        return int(self.Ar.shape[0])


def bt_design_full_rowley_chen_lqg(xa=-1.03, xs=0.98, beta=7.0, vhalf=2.0e-4, label="LQG_full"):
    """Build full-order LQG controller K(s) = -F (sI-Ak)^-1 L on the current FE model."""
    _bt_require_serial_petsc(A, "bt_design_full_rowley_chen_lqg")
    b_red, c_red = _bt_build_actuator_and_sensor(xa, xs)

    Md = _bt_petsc_mat_to_numpy_serial(M)
    Ad = _bt_petsc_mat_to_numpy_serial(A)
    bd = _bt_vec_to_numpy_serial(b_red).reshape(-1)
    cd = _bt_vec_to_numpy_serial(c_red).reshape(-1)
    n = Md.shape[0]

    Astd = _bt_la.solve(Md, Ad, assume_a='her')
    Bstd = _bt_la.solve(Md, bd.reshape(n, 1), assume_a='her').reshape(n)
    C = _bt_np.conjugate(cd).reshape(1, n)     # y = c^H q = C q

    Q = (float(beta) ** 2) * Md
    Q = 0.5 * (Q + Q.conjugate().T)
    R = _bt_np.array([[1.0]], dtype=_bt_np.complex128)
    Vnoise = _bt_np.array([[float(vhalf) ** 2]], dtype=_bt_np.complex128)

    if A.comm.rank == 0:
        print("\nDesigning full Rowley--Chen LQG controller for balanced reduction")
        print(f"  xa={xa:+.10f}, xs={xs:+.10f}, beta={beta}, V^(1/2)={vhalf}")
        print(f"  dense CARE order n={n}")

    X = _bt_la.solve_continuous_are(Astd, Bstd.reshape(n, 1), Q, R)
    Y = _bt_la.solve_continuous_are(Astd.conjugate().T, C.conjugate().T,
                                    _bt_np.eye(n, dtype=_bt_np.complex128), Vnoise)
    X = 0.5 * (X + X.conjugate().T)
    Y = 0.5 * (Y + Y.conjugate().T)

    F = (Bstd.conjugate().reshape(1, n) @ X).reshape(n)
    Lgain = (Y @ C.conjugate().T / Vnoise[0, 0]).reshape(n)
    Ak = Astd - _bt_np.outer(Bstd, F) - _bt_np.outer(Lgain, C.reshape(n))

    # Controller realization: xk_dot = Ak xk + L y, u = -F xk
    full = BTControllerData(
        label=label,
        xa=float(xa), xs=float(xs), beta=float(beta), vhalf=float(vhalf),
        Ar=_bt_np.asarray(Ak, dtype=_bt_np.complex128),
        Br=_bt_np.asarray(Lgain, dtype=_bt_np.complex128),
        Cr=_bt_np.asarray(-F, dtype=_bt_np.complex128),
        D=0.0 + 0.0j,
        hsv=None,
        error_bound=None,
    )
    return full, b_red, c_red


def bt_balanced_truncation_family(full: BTControllerData, orders, gramian_reltol=1e-12):
    """Square-root balanced truncation of the CONTROLLER transfer K(s)."""
    Ak = _bt_np.asarray(full.Ar, dtype=_bt_np.complex128)
    Bk = _bt_np.asarray(full.Br, dtype=_bt_np.complex128).reshape(-1, 1)
    Ck = _bt_np.asarray(full.Cr, dtype=_bt_np.complex128).reshape(1, -1)
    n = Ak.shape[0]

    if A.comm.rank == 0:
        print("\nBalanced truncation of full LQG controller")
        print(f"  controller order = {n}")
        print("  solving controller controllability/observability Lyapunov equations...")

    # A Wc + Wc A^H + B B^H = 0, A^H Wo + Wo A + C^H C = 0
    Wc = _bt_la.solve_continuous_lyapunov(Ak, -(Bk @ Bk.conjugate().T))
    Wo = _bt_la.solve_continuous_lyapunov(Ak.conjugate().T, -(Ck.conjugate().T @ Ck))
    Wc = 0.5 * (Wc + Wc.conjugate().T)
    Wo = 0.5 * (Wo + Wo.conjugate().T)

    Rc = _bt_sqrt_psd_factor(Wc, reltol=gramian_reltol)
    Ro = _bt_sqrt_psd_factor(Wo, reltol=gramian_reltol)

    U, hsv, Vh = _bt_la.svd(Ro.conjugate().T @ Rc, full_matrices=False)
    hsv = _bt_np.asarray(hsv, dtype=float)
    V = Vh.conjugate().T

    s_safe = _bt_np.maximum(hsv, _bt_np.finfo(float).eps)
    Tbal = Rc @ V @ _bt_np.diag(1.0 / _bt_np.sqrt(s_safe))
    Tinv = _bt_np.diag(1.0 / _bt_np.sqrt(s_safe)) @ U.conjugate().T @ Ro.conjugate().T

    # Balanced full realization, then truncate leading states.
    Ab = Tinv @ Ak @ Tbal
    Bb = Tinv @ Bk
    Cb = Ck @ Tbal

    orders = [int(r) for r in orders if int(r) > 0]
    orders = sorted(set([min(r, len(hsv), n) for r in orders] + [n]))

    family = []
    for r in orders:
        discarded = hsv[r:] if r < len(hsv) else _bt_np.array([], dtype=float)
        err_bound = float(2.0 * _bt_np.sum(discarded))
        fam = BTControllerData(
            label=f"{full.label}_BT_r{r}",
            xa=full.xa, xs=full.xs, beta=full.beta, vhalf=full.vhalf,
            Ar=_bt_np.asarray(Ab[:r, :r], dtype=_bt_np.complex128),
            Br=_bt_np.asarray(Bb[:r, 0], dtype=_bt_np.complex128),
            Cr=_bt_np.asarray(Cb[0, :r], dtype=_bt_np.complex128),
            D=full.D,
            hsv=hsv.copy(),
            error_bound=err_bound,
        )
        family.append(fam)

    return family, hsv


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


# -----------------------------------------------------------------------------------------------
# Closed-loop evaluation for a generic reduced controller
# -----------------------------------------------------------------------------------------------
class _BTQtoQCtx:
    def __init__(self, parent, S_aug, ksp_aug):
        self.parent = parent
        self.S_aug = S_aug
        self.ksp_aug = ksp_aug
        self.u_aug = S_aug.createVecRight()

    def mult(self, A_shell, x_q, y_q):
        rhs = self.parent.qvec_to_aug(x_q, self.S_aug)
        self.ksp_aug.solve(rhs, self.u_aug)
        Zu = self.parent.apply_Z(self.u_aug)
        z_aug = self.parent.solve_H_via_transpose(self.ksp_aug, self.S_aug, Zu)
        zq, _ = self.parent.split_aug_vec(z_aug)
        zq.copy(y_q)


class BTReducedControllerEnergy:
    """Close a full/reduced dynamic controller around the original FE plant and evaluate q->q energy gain."""
    def __init__(self, Aplant, Mplant, b_actual, c_actual, data: BTControllerData):
        self.A = Aplant
        self.M = Mplant
        self.b = b_actual.copy()
        self.c = c_actual.copy()
        self.data = data
        self.comm = Aplant.comm
        self.n = Aplant.getSize()[0]
        self.r = data.order
        self.is_q, self.is_k = _block_is(self.comm, [self.n, self.r])
        self.Ik = eye_scaled(self.comm, self.r, 1.0, like_mat=Aplant)
        self.zero_r_r = PETSc.Mat().createAIJ([self.r, self.r], comm=self.comm)
        self.zero_r_r.setUp(); self.zero_r_r.assemble()
        self._Acl = None
        self._Ecl = None
        self._Z = None

    def Ecl(self):
        if self._Ecl is None:
            self._Ecl = _bt_build_nested_aij([[self.M, None], [None, self.Ik]], [self.n, self.r], self.comm)
        return self._Ecl

    def Z(self):
        if self._Z is None:
            self._Z = _bt_build_nested_aij([[self.M, None], [None, self.zero_r_r]], [self.n, self.r], self.comm)
        return self._Z

    def Acl(self):
        if self._Acl is None:
            # With D=0: M qdot = A q + b * Cr xk,   xk_dot = Ar xk + Br c^H q.
            UR = _bt_make_rect_outer_vec_row(self.b, _bt_np.conjugate(self.data.Cr))  # b * Cr
            LL = _bt_make_rect_outer_col_vecH(self.data.Br, self.c)                  # Br * c^H
            LR = _bt_make_dense_mat(self.comm, self.data.Ar)
            self._Acl = _bt_build_nested_aij([[self.A, UR], [LL, LR]], [self.n, self.r], self.comm)
        return self._Acl

    def qvec_to_aug(self, qvec: PETSc.Vec, template: PETSc.Mat) -> PETSc.Vec:
        aug = template.createVecRight()
        aug.set(0.0)
        _bt_set_subvector(aug, self.is_q, qvec)
        return aug

    def split_aug_vec(self, v_aug: PETSc.Vec):
        return _bt_split_vec_by_is(v_aug, self.is_q), _bt_split_vec_by_is(v_aug, self.is_k)

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
        Zu = self.Z().createVecRight()
        Zu.set(0.0)
        Muq = uq.duplicate(); self.M.mult(uq, Muq)
        _bt_set_subvector(Zu, self.is_q, Muq)
        return Zu

    def build_S_aug(self, omega: float):
        S = self.Acl().copy()
        S.scale(-1.0)
        S.axpy(1j * float(omega), self.Ecl())
        S.assemble()
        return S

    def gain_of_omega(self, omega: float, eps_tol=1e-8, max_it=400, ncv=50) -> float:
        S_aug = self.build_S_aug(float(omega))
        ksp_aug = make_lu_ksp(S_aug)
        Tq = PETSc.Mat().createPython([self.n, self.n], comm=self.comm)
        Tq.setPythonContext(_BTQtoQCtx(self, S_aug, ksp_aug))
        Tq.setUp(); Tq.assemble()
        eps = SLEPc.EPS().create(comm=self.comm)
        eps.setOperators(Tq, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        eps.setDimensions(nev=1, ncv=ncv)
        eps.setTolerances(eps_tol, max_it)
        eps.setFromOptions()
        eps.solve()
        if eps.getConverged() < 1:
            raise RuntimeError(f"GHEP did not converge for reduced-controller gain at omega={omega}")
        return float(_bt_np.sqrt(max(float(_bt_np.real(eps.getEigenvalue(0))), 0.0)))

    def spectral_abscissa(self, nev=6, tol=1e-8, max_it=500) -> float:
        vals = rightmost_eig_realpart(self.Acl(), self.Ecl(), nev=nev, tol=tol, max_it=max_it)
        return float(_bt_np.max(_bt_np.real(vals)))

    def plant_tf(self, omega: float) -> complex:
        S = self.A.copy(); S.scale(-1.0); S.axpy(1j * float(omega), self.M); S.assemble()
        ksp = make_lu_ksp(S)
        x = S.createVecRight()
        ksp.solve(self.b, x)
        return complex(self.c.dot(x))

    def controller_tf(self, omega: float) -> complex:
        r = self.r
        X = _bt_la.solve(1j * float(omega) * _bt_np.eye(r, dtype=_bt_np.complex128) - self.data.Ar,
                         self.data.Br.reshape(r, 1), assume_a='gen')
        return complex((self.data.Cr.reshape(1, r) @ X)[0, 0] + self.data.D)

    def loop_metrics_at_omega(self, omega: float):
        P = self.plant_tf(float(omega))
        Kc = self.controller_tf(float(omega))
        Lloop, Sval, Tval, KSval, den = _bt_siso_tf(P, Kc)
        return {"omega": float(omega), "P": P, "K": Kc, "L": Lloop, "S": Sval,
                "T": Tval, "KS": KSval, "den": den}


def bt_energy_hinf_signed(cl: BTReducedControllerEnergy, wmin=1e-4, wmax=50.0, ngrid=201,
                          refine=True, csv_path=None, parallel_grid=True):
    ws = _bt_signed_grid(wmin=wmin, wmax=wmax, ngrid=ngrid)

    _bt_progress(f"{cl.data.label}: Hinf signed grid with {len(ws)} frequencies", group=True)
    if parallel_grid:
        vals = _bt_eval_scalar_grid_group(
            ws,
            lambda w: cl.gain_of_omega(float(w)),
            label=f"{cl.data.label} Hinf grid",
        )
    else:
        vals = _bt_np.asarray([cl.gain_of_omega(float(w)) for w in ws], dtype=float)

    idx = int(_bt_np.nanargmax(vals))
    best_w = float(ws[idx]); best_g = float(vals[idx])

    if refine and 0 < idx < len(ws) - 1:
        lo, hi = sorted([float(ws[idx - 1]), float(ws[idx + 1])])
        try:
            if parallel_grid and _bt_can_split_work() and _BT_GROUP_ACTUAL_SIZE > 1:
                # scipy.minimize_scalar is inherently sequential; in MPI mode use a deterministic
                # refined mini-grid so the expensive gain calls are still split across ranks.
                nref = max(9, _bt_env_int("BT_REFINE_POINTS", 41))
                ws_ref = _bt_np.linspace(lo, hi, nref)
                vals_ref = _bt_eval_scalar_grid_group(
                    ws_ref,
                    lambda om: cl.gain_of_omega(float(om)),
                    label=f"{cl.data.label} Hinf refine",
                )
                j = int(_bt_np.nanargmax(vals_ref))
                if float(vals_ref[j]) >= best_g:
                    best_g = float(vals_ref[j]); best_w = float(ws_ref[j])
            else:
                res = _bt_opt.minimize_scalar(lambda om: -cl.gain_of_omega(float(om)),
                                              bounds=(lo, hi), method='bounded',
                                              options={"xatol": 1e-5, "maxiter": 35})
                if res.success and -float(res.fun) >= best_g:
                    best_g = -float(res.fun); best_w = float(res.x)
        except Exception as e:
            if _bt_is_group_root():
                print(f"  [warning] Hinf refinement failed for {cl.data.label}:", repr(e), flush=True)

    neg = ws < 0; pos = ws > 0
    dc_val = float(cl.gain_of_omega(0.0))
    out = {
        "Hinf": best_g, "omega": best_w,
        "neg_peak": float(_bt_np.max(vals[neg])) if _bt_np.any(neg) else _bt_np.nan,
        "neg_omega": float(ws[neg][_bt_np.argmax(vals[neg])]) if _bt_np.any(neg) else _bt_np.nan,
        "pos_peak": float(_bt_np.max(vals[pos])) if _bt_np.any(pos) else _bt_np.nan,
        "pos_omega": float(ws[pos][_bt_np.argmax(vals[pos])]) if _bt_np.any(pos) else _bt_np.nan,
        "dc": dc_val,
    }

    if csv_path and _bt_is_group_root():
        with open(csv_path, "w", newline="") as f:
            wr = _bt_csv.writer(f); wr.writerow(["omega", "sigma_q_to_q"])
            for w, g in zip(ws, vals):
                wr.writerow([float(w), float(g)])
    return out

def bt_loop_robustness_metrics(cl: BTReducedControllerEnergy, wmin=1e-4, wmax=50.0, ngrid=301,
                               csv_path=None):
    ws = _bt_signed_grid(wmin=wmin, wmax=wmax, ngrid=ngrid)
    _bt_progress(f"{cl.data.label}: loop metrics grid with {len(ws)} frequencies", group=True)

    def _one_row(w):
        m = cl.loop_metrics_at_omega(float(w))
        return [float(w), abs(m["L"]), abs(m["S"]), abs(m["T"]), abs(m["KS"]), abs(m["den"])]

    if _bt_can_split_work() and _BT_GROUP_ACTUAL_SIZE > 1:
        local = []
        errors = []
        for i, w in enumerate(ws):
            if i % _BT_GROUP_ACTUAL_SIZE != _BT_GROUP_RANK:
                continue
            try:
                if _bt_env_flag("BT_PROGRESS", default=False):
                    print(f"[BT group {_BT_GROUP_ID} rank {_BT_GROUP_RANK}] {cl.data.label} loop point {i+1}/{len(ws)}", flush=True)
                local.append((i, _one_row(float(w))))
            except Exception as e:
                errors.append((i, repr(e)))
        chunks = _BT_GROUP_COMM.allgather(local)
        err_chunks = _BT_GROUP_COMM.allgather(errors)
        flat_errors = [e for chunk in err_chunks for e in chunk]
        if flat_errors:
            raise RuntimeError(f"Parallel loop-metric evaluation failed for {cl.data.label}: {flat_errors[:3]}")
        rows_by_i = {}
        for chunk in chunks:
            for i, row in chunk:
                rows_by_i[int(i)] = row
        rows = [rows_by_i[i] for i in range(len(ws))]
    else:
        rows = []
        for i, w in enumerate(ws):
            if _bt_env_flag("BT_PROGRESS", default=False) and _bt_is_group_root():
                if i == 0 or (i + 1) == len(ws) or (i + 1) % max(1, len(ws) // 10) == 0:
                    print(f"[BT {cl.data.label} loop] {i+1}/{len(ws)}", flush=True)
            rows.append(_one_row(float(w)))

    arr = _bt_np.asarray(rows, dtype=float)
    L_vals = arr[:, 1]; Ms_vals = arr[:, 2]; Mt_vals = arr[:, 3]
    KS_vals = arr[:, 4]; den_vals = arr[:, 5]
    iMs = int(_bt_np.nanargmax(Ms_vals)); iMt = int(_bt_np.nanargmax(Mt_vals))
    iKS = int(_bt_np.nanargmax(KS_vals)); idn = int(_bt_np.nanargmin(den_vals)); iL = int(_bt_np.nanargmax(L_vals))
    out = {
        "Ms": float(Ms_vals[iMs]), "omega_Ms": float(ws[iMs]),
        "Mt": float(Mt_vals[iMt]), "omega_Mt": float(ws[iMt]),
        "KS": float(KS_vals[iKS]), "omega_KS": float(ws[iKS]),
        "modulus_margin": float(1.0 / Ms_vals[iMs]),
        "multiplicative_radius": float(1.0 / Mt_vals[iMt]),
        "additive_output_radius": float(1.0 / KS_vals[iKS]),
        "min_abs_1_minus_L": float(den_vals[idn]), "omega_min_den": float(ws[idn]),
        "peak_abs_L": float(L_vals[iL]), "omega_peak_L": float(ws[iL]),
    }
    if csv_path and _bt_is_group_root():
        with open(csv_path, "w", newline="") as f:
            wr = _bt_csv.writer(f); wr.writerow(["omega", "absL", "absS", "absT", "absKS", "abs_1_minus_L"])
            wr.writerows(rows)
    return out

def bt_rebuild_actual_b_c(xa, xs):
    return _bt_build_actuator_and_sensor(xa, xs)


def bt_fixed_controller_jitter(data: BTControllerData, sigmas=(0.05, 0.1, 0.25, 0.5), nsamp=20,
                               seed=123, wmin=1e-4, wmax=20.0, ngrid=81):
    rows = []
    for sx in sigmas:
        # Pre-generate identical jitter samples on all ranks for deterministic MPI/serial behavior.
        rng = _bt_np.random.default_rng(seed + int(round(1000000 * float(sx))))
        samples = [
            (
                data.xa + float(sx) * rng.standard_normal(),
                data.xs + float(sx) * rng.standard_normal(),
            )
            for _ in range(int(nsamp))
        ]

        local = []
        if _bt_can_split_work() and _BT_GROUP_ACTUAL_SIZE > 1:
            iterator = [(j, p) for j, p in enumerate(samples) if j % _BT_GROUP_ACTUAL_SIZE == _BT_GROUP_RANK]
        else:
            iterator = list(enumerate(samples))

        for j, (xa_j, xs_j) in iterator:
            b_j, c_j = bt_rebuild_actual_b_c(xa_j, xs_j)
            clj = BTReducedControllerEnergy(A, M, b_j, c_j, data)
            try:
                alpha = clj.spectral_abscissa()
            except Exception:
                alpha = _bt_np.inf
            if alpha < 0:
                try:
                    h = bt_energy_hinf_signed(clj, wmin=wmin, wmax=wmax, ngrid=ngrid, refine=False,
                                              parallel_grid=False)
                    hinf = float(h["Hinf"])
                except Exception:
                    hinf = _bt_np.inf
            else:
                hinf = _bt_np.inf
            local.append((j, float(alpha), float(hinf)))

        chunks = _bt_group_allgather(local)
        all_results = [item for chunk in chunks for item in chunk]
        all_results.sort(key=lambda x: x[0])
        alphas = _bt_np.asarray([x[1] for x in all_results], dtype=float)
        hinfs = _bt_np.asarray([x[2] for x in all_results], dtype=float)

        stable = _bt_np.isfinite(alphas) & (alphas < 0.0)
        finite_h = hinfs[_bt_np.isfinite(hinfs)]
        rows.append({
            "sigma_x": float(sx),
            "stable_frac": float(_bt_np.mean(stable)),
            "worst_alpha": float(_bt_np.nanmax(alphas)),
            "median_Hinf": float(_bt_np.median(finite_h)) if finite_h.size else _bt_np.inf,
            "worst_Hinf": float(_bt_np.max(finite_h)) if finite_h.size and _bt_np.all(_bt_np.isfinite(hinfs)) else _bt_np.inf,
        })
    return rows


# -----------------------------------------------------------------------------------------------
# Main sweep driver
# -----------------------------------------------------------------------------------------------
def run_lqg_balanced_reduction_study(
    xa=-1.03,
    xs=0.98,
    beta=7.0,
    vhalf=2.0e-4,
    orders=(2, 4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 80, 120, 160),
    wmin=1e-4,
    wmax=50.0,
    ngrid_hinf=161,
    ngrid_loop=241,
    do_jitter=False,
    jitter_nsamp=20,
):
    _bt_check_parallel_layout_or_raise()
    can_parallel = _bt_can_split_work()

    if _bt_is_world_root():
        print("\n" + "#" * 110)
        print("BALANCED-TRUNCATED ROWLEY--CHEN LQG CONTROLLER STUDY")
        print("Metric: signed-frequency Hinf q->q with FE M-energy input/output norm, same as yo.py")
        print(f"placement: xa={xa:+.10f}, xs={xs:+.10f}; beta={beta}; V^(1/2)={vhalf}")
        print(f"MPI layout: WORLD_SIZE={_BT_WORLD_SIZE}, BT_GROUP_SIZE={_BT_GROUP_ACTUAL_SIZE}, "
              f"NUM_ORDER_GROUPS={_BT_NUM_GROUPS}, can_split_work={can_parallel}")
        print("#" * 110, flush=True)

    # Dense CARE + Lyapunov balanced truncation is done exactly once on WORLD rank 0.
    # The resulting small NumPy controller family is then broadcast to worker ranks.
    if can_parallel:
        if _bt_is_world_root():
            full, _, _ = bt_design_full_rowley_chen_lqg(xa=xa, xs=xs, beta=beta, vhalf=vhalf, label="RC_LQG")
            family, hsv = bt_balanced_truncation_family(full, orders)
            payload = (family, hsv)
        else:
            payload = None
        family, hsv = _bt_world_bcast(payload, root=0)
        # Each worker rank builds its local PETSc b/c vectors for validation.
        b_nom, c_nom = bt_rebuild_actual_b_c(xa, xs)
    else:
        full, b_nom, c_nom = bt_design_full_rowley_chen_lqg(xa=xa, xs=xs, beta=beta, vhalf=vhalf, label="RC_LQG")
        family, hsv = bt_balanced_truncation_family(full, orders)

    if _bt_is_world_root():
        with open("lqg_controller_hsv.csv", "w", newline="") as f:
            wr = _bt_csv.writer(f); wr.writerow(["index", "hsv"])
            for i, s in enumerate(hsv, start=1):
                wr.writerow([i, float(s)])
        print("\nFirst 30 controller Hankel singular values:")
        for i, s in enumerate(hsv[:30], start=1):
            print(f"  {i:4d}: {s:.6e}")
        print("HSV CSV written: lqg_controller_hsv.csv", flush=True)

    local_summary = []

    for order_index, data in enumerate(family):
        owner_group = (order_index % _BT_NUM_GROUPS) if can_parallel else 0
        if can_parallel and _BT_GROUP_ID != owner_group:
            continue

        _bt_progress(f"starting order r={data.order} assigned to group {owner_group}", group=True)

        cl = BTReducedControllerEnergy(A, M, b_nom, c_nom, data)
        try:
            alpha = cl.spectral_abscissa()
        except Exception as e:
            alpha = _bt_np.inf
            if _bt_is_group_root():
                print(f"[r={data.order}] spectral abscissa failed: {repr(e)}", flush=True)

        if alpha < 0.0:
            h = bt_energy_hinf_signed(cl, wmin=wmin, wmax=wmax, ngrid=ngrid_hinf, refine=True,
                                      csv_path=f"lqg_BT_r{data.order}_energy_hinf.csv")
            lm = bt_loop_robustness_metrics(cl, wmin=wmin, wmax=wmax, ngrid=ngrid_loop,
                                            csv_path=f"lqg_BT_r{data.order}_loop_metrics.csv")
        else:
            h = {"Hinf": _bt_np.inf, "omega": _bt_np.nan, "neg_peak": _bt_np.nan,
                 "neg_omega": _bt_np.nan, "pos_peak": _bt_np.nan, "pos_omega": _bt_np.nan, "dc": _bt_np.nan}
            lm = {"Ms": _bt_np.inf, "omega_Ms": _bt_np.nan, "Mt": _bt_np.inf, "omega_Mt": _bt_np.nan,
                  "KS": _bt_np.inf, "omega_KS": _bt_np.nan, "modulus_margin": 0.0,
                  "multiplicative_radius": 0.0, "additive_output_radius": 0.0,
                  "min_abs_1_minus_L": 0.0, "omega_min_den": _bt_np.nan,
                  "peak_abs_L": _bt_np.nan, "omega_peak_L": _bt_np.nan}

        jit = None
        if do_jitter and alpha < 0.0:
            jit = bt_fixed_controller_jitter(data, nsamp=jitter_nsamp)

        row = {
            "order": data.order,
            "alpha": float(alpha),
            "gap": float(-alpha) if _bt_np.isfinite(alpha) else -_bt_np.inf,
            "Hinf_energy": float(h["Hinf"]),
            "omega_Hinf": float(h["omega"]),
            "Ms": float(lm["Ms"]),
            "Mt": float(lm["Mt"]),
            "KS": float(lm["KS"]),
            "min_abs_1_minus_L": float(lm["min_abs_1_minus_L"]),
            "controller_error_bound_2sum_tail_HSV": float(data.error_bound) if data.error_bound is not None else 0.0,
            "stable_frac_sigma_0p25": _bt_np.nan,
        }
        if jit is not None:
            for jr in jit:
                if abs(jr["sigma_x"] - 0.25) < 1e-12:
                    row["stable_frac_sigma_0p25"] = jr["stable_frac"]
                    break

        if _bt_is_group_root():
            local_summary.append(row)
            print("\n" + "=" * 94)
            print(f"Reduced LQG controller order r={data.order}")
            print(f"  alpha=max Re(lambda)          = {alpha:+.10e}")
            print(f"  Hinf_energy(q->q)             = {h['Hinf']:.10e} at omega={h['omega']:+.10e}")
            print(f"  Ms/Mt/KS                      = {lm['Ms']:.6e}, {lm['Mt']:.6e}, {lm['KS']:.6e}")
            print(f"  min |1-L|                     = {lm['min_abs_1_minus_L']:.6e}")
            print(f"  BT controller error bound     = {row['controller_error_bound_2sum_tail_HSV']:.6e}")
            if jit is not None:
                print("  placement jitter:")
                for jr in jit:
                    print(f"    sigma={jr['sigma_x']:.3g}: stable_frac={jr['stable_frac']:.3f}, "
                          f"worst_alpha={jr['worst_alpha']:+.3e}, median_Hinf={jr['median_Hinf']:.3e}")
            print("=" * 94, flush=True)

    gathered = _bt_world_allgather(local_summary)
    summary = [row for chunk in gathered for row in chunk]
    summary.sort(key=lambda r: int(r["order"]))

    if _bt_is_world_root():
        with open("lqg_balanced_reduction_summary.csv", "w", newline="") as f:
            keys = list(summary[0].keys()) if summary else []
            wr = _bt_csv.DictWriter(f, fieldnames=keys)
            if keys:
                wr.writeheader(); wr.writerows(summary)
        print("\nCSV written: lqg_balanced_reduction_summary.csv")
        print("Per-order CSVs written: lqg_BT_r<order>_energy_hinf.csv and lqg_BT_r<order>_loop_metrics.csv", flush=True)

    return summary, hsv


# Set this to False if you only want to import/define functions.
RUN_LQG_BALANCED_REDUCTION_STUDY = _bt_env_flag("RUN_LQG_BALANCED_REDUCTION_STUDY", default=True)

if RUN_LQG_BALANCED_REDUCTION_STUDY:
    run_lqg_balanced_reduction_study(
        # For your current yo.py model with mu0=0.38, (-0.98,0.95) is the subcritical-like placement.
        # Use (-1.03,0.98) if you want the paper's supercritical SISO placement.
        xa=-0.98,
        xs=0.95,
        beta=7.0,
        vhalf=2.0e-4,
        orders=(2, 4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 80, 120, 160),
        wmin=1e-4,
        wmax=50.0,
        ngrid_hinf=161,
        ngrid_loop=241,
        do_jitter=False,
        jitter_nsamp=20,
    )