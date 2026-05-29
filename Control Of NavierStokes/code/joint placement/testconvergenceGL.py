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

import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from petsc4py import PETSc
from dolfinx import fem
import ufl

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
Nx = 5000 # resolution

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

import numpy as np
from petsc4py import PETSc

def bc_local_dofs(bc):
    """
    Return local dof indices (1D np.int32 array) from a dolfinx DirichletBC,
    handling versions where bc.dof_indices() returns a tuple.
    """
    di = bc.dof_indices()

    # Many versions return (dofs, something_else)
    if isinstance(di, (tuple, list)):
        di = di[0]

    di = np.asarray(di, dtype=np.int32).ravel()
    return di


def free_dof_is(V, bc, comm):
    """
    Build a PETSc IS of global free dofs (excluding Dirichlet constrained dofs),
    assuming scalar CG space (bs=1) OR block size handled via index_map_bs.
    """
    imap = V.dofmap.index_map
    bs   = V.dofmap.index_map_bs

    n_local = imap.size_local * bs
    start   = imap.local_range[0] * bs

    bc_dofs_local = bc_local_dofs(bc)

    mask = np.ones(n_local, dtype=bool)
    mask[bc_dofs_local] = False

    free_local = np.nonzero(mask)[0].astype(np.int32)
    free_global = (free_local + start).astype(np.int32)

    return PETSc.IS().createGeneral(free_global, comm=comm)
is_free = free_dof_is(V, bc, MPI.COMM_WORLD)
A_free  = A.createSubMatrix(is_free, is_free); A_free.assemble()
M_free  = M.createSubMatrix(is_free, is_free); M_free.assemble()

E = SLEPc.EPS().create(MPI.COMM_WORLD) # Eigenvalue problem solver = EPS 
E.setOperators(A_free, M_free)
E.setProblemType(SLEPc.EPS.ProblemType.GNHEP) # GNHEP = generalized non hermitian eigenvalue problem
E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
E.setDimensions(nev=20, ncv=120)     # much bigger subspace
E.setTolerances(1e-10, 800)          # more iterations

E.solve()
nconv = E.getConverged()

print("requested nev =", E.getDimensions()[0])
print("nconv =", E.getConverged())
print("its =", E.getIterationNumber())
print("reason =", E.getConvergedReason())


for i in range(nconv):
    vr, _ = A_free.createVecs()
    vi, _ = A_free.createVecs()
    lam = E.getEigenpair(i, vr, vi)
    print(lam)


import numpy as np

# ---------- High-precision analytic eigenvalues ----------
def analytic_eigs_GL(mu_0, mu_2, c_u, U, c_d, k, mp_dps=80):
    """
    Analytic eigenvalues:
      lambda_n = (mu_0 - c_u^2) - nu^2/(4*gamma) - (n+1/2) h
      h = sqrt(-2*mu_2*gamma)
      gamma = 1 + i*c_d
      nu    = U + 2 i*c_u
    Uses mpmath for high precision, returns numpy complex array length k.
    """
    try:
        import mpmath as mp
        mp.mp.dps = mp_dps

        mu0 = mp.mpf(mu_0)
        mu2 = mp.mpf(mu_2)
        cu  = mp.mpf(c_u)
        U_  = mp.mpf(U)
        cd  = mp.mpf(c_d)

        gamma = mp.mpc(1, cd)              # 1 + i*c_d
        nu    = mp.mpc(U_, 2*cu)           # U + 2 i*c_u

        h  = mp.sqrt(-2*mu2*gamma)         # principal branch
        lam0 = (mu0 - cu**2) - (nu**2)/(4*gamma)

        out = []
        for n in range(k):
            lamn = lam0 - (mp.mpf(n) + mp.mpf('0.5'))*h
            out.append(complex(lamn.real, lamn.imag))
        return np.array(out, dtype=complex)

    except ImportError:
        # fallback: double precision numpy (still ~1e-15-ish)
        gamma = 1.0 + 1j*c_d
        nu    = U + 2j*c_u
        h     = np.sqrt(-2.0*mu_2*gamma)
        lam0  = (mu_0 - c_u**2) - (nu**2)/(4.0*gamma)
        n = np.arange(k, dtype=float)
        return lam0 - (n + 0.5)*h


# ---------- Matching: nearest neighbor in complex plane ----------
def match_nearest(ana, disc):
    """
    Greedy nearest-neighbor matching:
      for each analytic eigenvalue, pick closest remaining discrete eigenvalue.
    Returns list of tuples: (n, lam_ana, lam_disc, abs_err, rel_err)
    """
    disc_remaining = list(disc)
    rows = []
    for n, la in enumerate(ana):
        if not disc_remaining:
            break
        j = int(np.argmin([abs(ld - la) for ld in disc_remaining]))
        ld = disc_remaining.pop(j)
        abs_err = abs(ld - la)
        rel_err = abs_err / max(1.0, abs(la))
        rows.append((n, la, ld, abs_err, rel_err))
    return rows


# ---------- Put this after you compute your EPS eigenvalues ----------
# Collect discrete eigenvalues in a Python list
lam_disc = []
for i in range(nconv):
    vr, _ = A_free.createVecs()
    vi, _ = A_free.createVecs()
    lam = E.getEigenpair(i, vr, vi)
    lam_disc.append(lam)
lam_disc = np.array(lam_disc, dtype=complex)

# Sort discrete eigenvalues by real part descending (rightmost first)
lam_disc = lam_disc[np.argsort(np.real(lam_disc))[::-1]]

# Compute analytic eigenvalues (same count as discrete)
lam_ana = analytic_eigs_GL(mu_0, mu_2, c_u, U, c_d, k=len(lam_disc), mp_dps=80)

# Match and print errors
pairs = match_nearest(lam_ana, lam_disc)
print("\nCompare analytic vs discrete (nearest-neighbor matched):")
print("   n |     Re(ana)        Im(ana)    |     Re(disc)       Im(disc)   |  abs_err     rel_err")
print("-----+----------------------------------------------------------------+-----------------------")

for (n, la, ld, ae, re) in pairs[:20]:
    print(f"{n:4d} | {la.real:12.6f} {la.imag:12.6f} | {ld.real:12.6f} {ld.imag:12.6f} | {ae:9.3e}  {re:9.3e}")

abs_errs = np.array([p[3] for p in pairs], dtype=float)
rel_errs = np.array([p[4] for p in pairs], dtype=float)

print("\nSummary on first 20 matched:")
print("max abs err:", abs_errs[:20].max())
print("max rel err:", rel_errs[:20].max())
print("median rel err:", np.median(rel_errs[:20]))

def one_eig_near(Aop, Mop, target, nev=6, tol=1e-12, max_it=400):
    eps = SLEPc.EPS().create(comm=Aop.comm)
    eps.setOperators(Aop, Mop)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target)

    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(target)

    eps.setDimensions(nev, ncv=max(4*nev, 40))
    eps.setTolerances(tol, max_it)
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        return None

    vals = np.array([eps.getEigenvalue(i) for i in range(nconv)], dtype=complex)
    j = np.argmin(np.abs(vals - target))
    return vals[j]
n_check = 8  # only first 8 modes
errs = []
for n in range(n_check):
    target = lam_ana[n]
    lam_d = one_eig_near(A_free, M_free, target)
    ae = abs(lam_d - target)
    re = ae / (abs(target) + 1e-30)
    errs.append(re)
print("max rel err over first", n_check, "modes:", max(errs))
