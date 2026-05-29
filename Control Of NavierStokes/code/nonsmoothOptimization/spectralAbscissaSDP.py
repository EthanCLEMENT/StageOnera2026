import numpy as np
import scipy.linalg as la
import cvxpy as cp


def uniquetol_rows(mat: np.ndarray, tol: float):
    """
    Group rows with an absolute tolerance, similar to MATLAB's uniquetol for rows.

    Returns:
      unique_rows: (k, d) array of representative unique rows
      group_id: (nrows,) int array mapping each input row -> 1..k (MATLAB/Julia style)
    """
    nrows, d = mat.shape
    unique_rows = np.empty((0, d), dtype=float)
    group_id = np.zeros(nrows, dtype=int)

    for i in range(nrows):
        row = mat[i, :]
        found = False
        for j in range(unique_rows.shape[0]):
            if np.allclose(row, unique_rows[j, :], atol=tol, rtol=0.0):
                group_id[i] = j + 1  # 1-based group id
                found = True
                break
        if not found:
            unique_rows = np.vstack([unique_rows, row[None, :]])
            group_id[i] = unique_rows.shape[0]  # new group (1-based)
    return unique_rows, group_id

def sdp_step_direction(B, C, delta, c_vec, U_list, V_list):
    # B: (n,m), C: (p,n)
    # U_list[j]: (n,rj) left basis, V_list[j]: (n,rj) right basis with U^H V = I
    # c_vec[j] = Re(mu_j) - alpha <= 0
    n, m = B.shape
    p, _ = C.shape
    J = len(U_list)

    H = cp.Variable((m, p))     # real step in K-space
    t = cp.Variable()           # epigraph variable
    s = cp.Variable(J)          # upper bounds on lambda_max terms

    constraints = []

    for j in range(J):
        Uj = U_list[j]          # (n,r)
        Vj = V_list[j]          # (n,r)
        r = Vj.shape[1]

        # S_j(H) = U^H B H C V   (r x r), complex affine in H
        S = Uj.conj().T @ B @ H @ C @ Vj

        # Hermitian part: (S + S^H)/2
        SymS = 0.5 * (S + cp.conj(cp.transpose(S)))

        # s_j I - SymS >= 0  enforces s_j >= lambda_max(SymS)
        constraints += [s[j] * np.eye(r) - SymS >> 0]

        # t >= c_j + s_j
        constraints += [t >= c_vec[j] + s[j]]

    obj = cp.Minimize(t + 0.5 * delta * cp.sum_squares(H))
    prob = cp.Problem(obj, constraints)

    prob.solve(solver=cp.SCS, verbose=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDP failed: {prob.status}")

    theta = prob.value
    return H.value, theta, t.value, s.value

def nonsmooth_stabilize(
    A, B, C, K0,
    rho=0.5, delta=1.0,
    epsTheta=1e-6, epsAlpha=1e-6, epsK=1e-6,
    beta=1e-6, maxIter=100
):
    """
    Minimizes the spectral abscissa of A + B*K*C using a first–order descent–type algorithm.

    Inputs:
      A: (n,n)
      B: (n,m)
      C: (p,n)
      K0: (m,p)

    Outputs:
      K: final controller
      history: dict with keys "K", "alphaval", "theta"
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    K = np.array(K0, dtype=float).copy()

    n = A.shape[0]
    m, p = K.shape

    history = {"K": [K.copy()], "alphaval": [], "theta": []}

    groupTol = 1e-4
    alpha_trial = np.inf

    for it in range(1, maxIter + 1):
        Acl = A + B @ K @ C

        # left/right eigenvectors
        eigvals, VL, VR = la.eig(Acl, left=True, right=True)
        V = VR
        U_all = VL
        reEig = np.real(eigvals)

        alphaVal = np.max(reEig)
        minRe = np.min(reEig)
        history["alphaval"].append(float(alphaVal))

        # near-active set
        active_mask = (alphaVal - reEig) <= rho * (alphaVal - minRe)
        active_idx = np.where(active_mask)[0]
        active_eigs = eigvals[active_idx]

        # group near-equal eigenvalues
        temp = np.column_stack([np.real(active_eigs), np.imag(active_eigs)])
        unique_temp, group_id = uniquetol_rows(temp.astype(float), groupTol)
        unique_active = unique_temp[:, 0] + 1j * unique_temp[:, 1]
        num_groups = len(unique_active)

        U_list, V_list = [], []
        c_vec = []

        for i_group in range(1, num_groups + 1):
            group_indices = active_idx[group_id == i_group]
            r = len(group_indices)

            Vj = V[:, group_indices]            # (n,r)
            Uj = U_all[:, group_indices]        # (n,r)

            # enforce biorthonormality: Uj^H Vj = I
            M = Uj.conj().T @ Vj
            Uj = Uj @ np.linalg.inv(M).conj().T

            U_list.append(Uj)
            V_list.append(Vj)

            mu_j = unique_active[i_group - 1]
            c_vec.append(float(np.real(mu_j) - alphaVal))  # <= 0

        c_vec = np.array(c_vec, dtype=float)

        H_dir, theta_val, t_val, s_vals = sdp_step_direction(B, C, delta, c_vec, U_list, V_list)
        history["theta"].append(float(theta_val))

        # dDir is the support-function value in direction H_dir
        dDir = float(np.max(np.array(s_vals).reshape(-1)))
        if dDir >= 0:
            print("WARNING: non-descent direction (dDir >= 0). Consider enlarging J_rho or tightening grouping.")

        if theta_val >= -epsTheta:
            print(f"Stopping: theta >= -epsTheta (theta = {theta_val:e}) at iteration {it}.")
            break

        print(f"Iter {it:3d}: alpha = {alphaVal:e}, theta = {theta_val:e}, d = {dDir:e}")

        # Armijo backtracking line search
        t_step = 1.0
        maxLSiter = 100
        found_step = False

        for _ in range(maxLSiter):
            K_trial = K + t_step * H_dir
            Acl_trial = A + B @ K_trial @ C
            eig_trial = la.eigvals(Acl_trial)
            alpha_trial = float(np.max(np.real(eig_trial)))

            if alpha_trial <= alphaVal + beta * t_step * theta_val:
                found_step = True
                break
            t_step *= 0.5

        if not found_step:
            print(f"WARNING: Line search failed at iteration {it}.")
            break

        # Update controller
        K = K + t_step * H_dir
        history["K"].append(K.copy())

    print(f"Algorithm terminated after {it} iterations. Final spectral abscissa: {alphaVal:e}")
    return K, history


def main():
    # Lateral flight dynamics X8
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
    #A = np.array([[0,1],[-1,0]])
    #B = np.array([[0],[1]])
    #C = np.array([[0,1]])
    #K0 = np.array([[0.0]]) 

    # A matrix (9x9)
    #A = np.array([
        #[-0.00702,  0.06339,  0.00518, -0.55566, -0.06112,  0,        0.00712, -0.00566,  0],
        #[ 0.01654, -0.38892,  1.00570,  0.00591, -0.04632,  0,        0.01554,  0.04018,  0],
        #[ 0.00061,  0.35210, -0.47381,  0,        1.78620,  0,       -0.00061, -0.03638,  0],
        #[ 0,        0,        1,        0,        0,        0,        0,        0,        0],
        #[ 0,        0,        0,        0,       20,       20,        0,        0,        0],
        #[ 0,        0,        0,        0,        0,      -30,        0,        0,        0],
        #[ 0,        0,        0,        0,        0,        0,       -0.55454,  0,        0],
        #[ 0,        0,        0,        0,        0,        0,        0,       -0.55456,  0.00555],
        #[ 0,        0,        0,        0,        0,        0,        0,       -0.00555, -0.55454]
    #], dtype=float)

    ## Full B matrix (9x5)
    #Bfull = np.array([
        #[ 0,       0,        0,        0,    0],
        #[ 0,       0,        0,        0,    0],
        #[ 0,       0,        0,        0,    0],
        #[ 0,       0,        0,        0,    0],
        #[ 0,       0,        0,        0,    0],
        #[30,       0,        0,        1,    0],
        #[ 0,    1.0531,      0,        0,    0],
        #[ 0,       0,     1.2898,      0,    0],
        #[ 0,       0,   -54.5140,      0,    0]
    #], dtype=float)

    ## Elevator channel (9x1)
    #Bu = Bfull[:, [0]]   # keep as column vector (9x1)

    ## Output matrix Cy (2x9)
    #Cy = np.array([
        #[0.00500, 0.11679, -0.00172, 0, -0.91413, 0, -0.00800, -0.01207, 0],
        #[0,       0,        1,       0,  0,       0,  0,        0,       0]
    #], dtype=float)

    #C = Cy
    #B = Bu

    ## Initial static output feedback gain (1x2)
    #K0 = np.zeros((1, 2))

    # Options
    rho = 0.2
    delta = 1.0
    epsTheta = 1e-5
    epsAlpha = 1e-6
    epsK = 1e-6
    beta = 1e-4
    maxIter = 1000

    K_final, history = nonsmooth_stabilize(
        A, B, C, K0,
        rho=rho, delta=delta,
        epsTheta=epsTheta, epsAlpha=epsAlpha, epsK=epsK,
        beta=beta, maxIter=maxIter
    )

    print("\nFinal controller gain K =")
    print(K_final)

    Ac_final = A + B @ K_final @ C
    lambda_final = la.eigvals(Ac_final)
    print("Final closed-loop eigenvalues:")
    print(lambda_final)
    print(f"Final spectral abscissa = {np.max(np.real(lambda_final)):.6g}")


if __name__ == "__main__":
    main()
