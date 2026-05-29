import torch
from torch import linalg as LA
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
import numpy as np

device = torch.device('cpu')  # or 'cuda'
dtype = torch.double
A = np.array([
    [-0.493,   0.201, -17.775,  9.810],
    [-4.005, -30.616,  -1.598,  0.000],
    [-3.690, -32.385,  -3.185,  0.000],
    [ 0.000,   1.000,   0.000,  0.000],
], dtype=float)

B = np.array([
    [  1.915,   0.000],
    [153.149,   0.000],
    [161.248,   0.000],
    [  0.000,   0.000],
], dtype=float)

C = np.eye(4, dtype=float)

# Initial guess
K0 = np.zeros((2, 4), dtype=float)
# given numpy A,B,C -> torch tensors
A = torch.tensor(A, device=device, dtype=dtype)
B = torch.tensor(B, device=device, dtype=dtype)
C = torch.tensor(C, device=device, dtype=dtype)

m = B.shape[1]   # K rows
p = C.shape[0]   # K cols

var_spec = {"K": [m, p]}

def user_fn(K_struct, A, B, C):
    K = K_struct.K
    M = A + B @ K @ C
    eigvals = LA.eigvals(M)      # complex
    f = torch.max(eigvals.real)  # spectral abscissa

    ci = None  # no inequality constraints
    ce = None  # no equality constraints
    return [f, ci, ce]

comb_fn = lambda K_struct: user_fn(K_struct, A, B, C)

opts = pygransoStruct()
opts.torch_device = device
opts.maxit = 500
opts.print_frequency = 10
opts.x0 = torch.zeros(m*p, 1, device=device, dtype=dtype)  # vectorized init

soln = pygranso(var_spec=var_spec, combined_fn=comb_fn, user_opts=opts)

K_star = soln.final.x.reshape(m, p)  # check exact field name in your version
print(K_star)
Ac_final = A + B @ K_star @ C
lambda_final = np.linalg.eigvals(Ac_final)
print("Final closed-loop eigenvalues:")
print(lambda_final)
print(f"Final spectral abscissa = {np.max(np.real(lambda_final)):.6g}")