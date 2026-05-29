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
Nx = 400 # resolution

domain = mesh.create_interval(MPI.COMM_WORLD, Nx,[x_min, x_max])
x = ufl.SpatialCoordinate(domain)
gamma = fem.Constant(domain, 1 + 1j*c_d) # diffusion coefficient
nu = fem.Constant(domain, U + 2j*c_u) # advection coefficient 
mu_x = (mu_0 - c_u**2) + (mu_2*x[0]**2)/2 # region of amplification

# if mu_0 - c_u^2 < 0 then the flow is stable everywhere

V = fem.functionspace(domain, ("CG", 4)) # continous Galerkin, x ordre
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

##########################################################################################################

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

def T(M: PETSc.Mat) -> PETSc.Mat:

    MT = MT.hermitianTranspose()

    MT = M.copy()      
    MT.assemble()
    return MT

def build_S(A: PETSc.Mat, M: PETSc.Mat, omega: float) -> PETSc.Mat:
    s = 1j * omega
    S = A.copy()
    S.scale(-1.0)      # -A
    S.axpy(s, M)       # -A + s M  = iω M - A
    S.assemble()
    return S

def make_lu_ksp(mat: PETSc.Mat) -> PETSc.KSP:
    ksp = PETSc.KSP().create(mat.comm)
    ksp.setOperators(mat)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    ksp.setUp()
    ksp.setErrorIfNotConverged(True)
    return ksp

class AomegaCtx:
    def __init__(self, S: PETSc.Mat, M: PETSc.Mat):
        self.M = M

        # Forward solve: S u = x
        self.ksp_fwd = make_lu_ksp(S)

        # Adjoint solve: S^H y = v
        Sh = T(S)
        Sh.assemble()
        self.ksp_adj = make_lu_ksp(Sh)

        # work vectors
        self.u = M.createVecRight()   # u = S^{-1} x
        self.v = M.createVecLeft()   # v = M u

        self.tmp = S.createVecRight()

    def mult(self, A, x, y):
        # u = S^{-1} x
        self.ksp_fwd.solve(x, self.u)
        # v = M u
        self.M.mult(self.u, self.v)
        # y = (S^H)^{-1} v
        self.ksp_adj.solve(self.v, y)

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
def sigma_max_energy_resolvent(A: PETSc.Mat, M: PETSc.Mat, omega: float,
                              eps_tol=1e-8, max_it=200, ncv=40) -> float:

    Aop = make_Aomega_shell(A, M, omega)
    eps = SLEPc.EPS().create(comm=M.comm)
    eps.setOperators(Aop, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)  
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setDimensions(nev=1, ncv=ncv)
    eps.setTolerances(eps_tol, max_it)

    eps.solve()
    nconv = eps.getConverged()
    if nconv < 1:
        raise RuntimeError("EPS did not converge for sigma_max at omega=%g" % omega)

    vr = M.createVecRight()
    vi = M.createVecRight()
    lam = eps.getEigenpair(0, vr, vi)
    rq = rayleigh_quotient_check(Aop, M, vr)

    if M.comm.rank == 0:
        print("lam from EPS =", lam)
        print("rq - lam =", rq - lam)
        print("rel diff =", abs(rq - lam) / (abs(lam) + 1e-30))

    Ay = vr.duplicate(); Aop.mult(vr, Ay)
    My = vr.duplicate(); M.mult(vr, My)
    nm = np.sqrt(np.real(vr.dot(My)))
    vr.scale(1.0 / (nm + 1e-30))

    Ay = vr.duplicate(); Aop.mult(vr, Ay)
    My = vr.duplicate(); M.mult(vr, My)
    r = Ay.copy(); r.axpy(-lam, My)

    rel = r.norm() / (Ay.norm() + abs(lam)*My.norm() + 1e-30)
    print("manual relative residual =", rel)

    r = Ay.copy()
    r.axpy(-lam, My)   # r = Ay - lam*My

    rel = r.norm() / (Ay.norm() + abs(lam)*My.norm() + 1e-30)
    # Build S and its KSP exactly how the shell does it (LU)
    S = build_S(A, M, omega).convert("aij"); S.assemble()
    ksp_S = make_lu_ksp(S)

    energy_identity_check(S, ksp_S, M, Aop, vr)

    if M.comm.rank == 0:
        print("lambda =", lam)
        print("relative residual =", rel)

    lam = eps.getEigenvalue(0)
    lam = float(np.real(lam))
    lam = max(lam, 0.0)

    print("lam ", lam)
    print("lam.imag", lam.imag)
    print("lam.real0"),lam.real
    print("abs(imag(lam))", abs(lam.imag))
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

S = build_S(A, M, 1.0)
sigma = sigma_max_energy_resolvent(A,M,omega=1e-6)
print(sigma)

