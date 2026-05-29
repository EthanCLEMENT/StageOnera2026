import numpy as np
import scipy.linalg as la
import cvxpy as cp

from numpy.linalg import eig, norm

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

def nonsmooth_stabilize(A, B, C, K0,rho=0.0, delta=1.0, epsTheta=1e-6, epsAlpha=1e-6, epsK=1e-6,beta=0.5, maxIter=100):
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

    # state-space system
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    K = np.array(K0, dtype=float).copy()

    n = A.shape[0]
    m, p = K.shape

    history = {"K": [K.copy()], "alphaval": [], "theta": []}

    groupTol = 1e-6
    alpha_trial = np.inf

    for it in range(1, maxIter + 1):
        Acl = A + B @ K @ C
        eigvals, V = la.eig(Acl)  
        reEig = np.real(eigvals)

        alphaVal = np.max(reEig)
        minRe = np.min(reEig)
        history["alphaval"].append(float(alphaVal))

        active_mask = (alphaVal - reEig) <= rho * (alphaVal - minRe)
        active_idx = np.where(active_mask)[0]
        active_eigs = eigvals[active_idx]

        # Group nearly equal eigenvalues by [real imag] rows with tolerance
        temp = np.column_stack([np.real(active_eigs), np.imag(active_eigs)])
        unique_temp, group_id = uniquetol_rows(temp.astype(float), groupTol)
        unique_active = unique_temp[:, 0] + 1j * unique_temp[:, 1]
        num_groups = len(unique_active)

        L = la.solve(V, np.eye(n)) 

        P = np.zeros((m * p, num_groups), dtype=float)
        c_vec = np.zeros(num_groups, dtype=float)

        for i_group in range(1, num_groups + 1):  # 1-based group id values
            group_indices = active_idx[group_id == i_group]
            n_group = len(group_indices)

            Vj = V[:, group_indices]                       # (n, n_group)
            Uj = L[group_indices, :].conj().T              # (n, n_group)
            Yj = (1.0 / n_group) * np.eye(n_group)         # (n_group, n_group)

            # Julia uses Vj' meaning conjugate-transpose
            phi_group = np.real(C @ Vj @ Yj @ Uj.conj().T @ B ).T  # (m, p)

            print("Phi group : ")
            print(phi_group)

            # Julia vec() is column-major; match with order='F'
            P[:, i_group - 1] = phi_group.reshape(-1, order='F')
            mu_j = unique_active[i_group - 1]
            c_vec[i_group - 1] = float(np.real(mu_j) - alphaVal)

        # QP: min -c^T tau + (1/(2*delta)) tau^T Q tau  s.t. tau>=0, sum tau=1
        Q = P.T @ P  # PSD Gram matrix

        tau = cp.Variable(num_groups, nonneg=True)
        objective = cp.Minimize(-c_vec @ tau + (1.0/(2.0*delta)) * cp.sum_squares(P @ tau))

        constraints = [cp.sum(tau) == 1]
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=True)
        except Exception as e:
            print(f"WARNING: QP solver exception at iteration {it}: {e}")
            break

        if prob.status not in ("optimal", "optimal_inaccurate"):
            print(f"WARNING: QP solver did not converge at iteration {it}. status={prob.status}")
            break

        tau_opt = np.array(tau.value).reshape(-1)

        # theta value
        quad_term = float(tau_opt @ (Q @ tau_opt))
        theta_val = -prob.value 
        history["theta"].append(float(theta_val))

        # Optimality check
        #if theta_val >= -epsTheta:
            #print(f"Stopping: theta >= -epsTheta (theta = {theta_val:e}) at iteration {it}.")
            #break

        # Descent direction
        H_vec = P @ tau_opt
        H_dir = -(1.0 / delta) * H_vec.reshape((m, p), order='F')
        # --- Directional derivative for Armijo: use ONLY the true active set J(K) ---

        actTol = 1e-8  # you can tune this (often <= groupTol is fine)

        # True active eigenvalues among *all* eigenvalues of Acl:
        true_active_mask_all = (alphaVal - reEig) <= actTol
        true_active_idx_all = np.where(true_active_mask_all)[0]

        # Map those back to your enriched-index list, because group_id was built on active_idx
        # (active_idx is your enriched set indices).
        true_active_in_enriched_mask = np.isin(active_idx, true_active_idx_all)
        true_active_group_ids = np.unique(group_id[true_active_in_enriched_mask])

        # If numerical issues yield empty active groups, fall back to all groups
        if true_active_group_ids.size == 0:
            active_group_cols = np.arange(num_groups)  # 0..num_groups-1
        else:
            # group_id is 1-based; convert to 0-based column indices
            active_group_cols = true_active_group_ids - 1

        # Compute dDir over active groups only
        dvals = np.empty(active_group_cols.size, dtype=float)
        for k, col in enumerate(active_group_cols):
            phi_group = P[:, col].reshape((m, p), order='F')
            dvals[k] = float(np.sum(phi_group * H_dir))  # Frobenius inner product

        dDir = float(np.max(dvals))


        print(f"Iter {it:3d}: alpha = {alphaVal:e}, theta = {theta_val:e}, d = {dDir:e}")

        # Armijo backtracking line search
        t = 1.0
        maxLSiter = 100
        found_step = False

        for _ in range(maxLSiter):
            K_trial = K + t * H_dir
            Acl_trial = A + B @ K_trial @ C
            eig_trial = la.eigvals(Acl_trial)
            alpha_trial = float(np.max(np.real(eig_trial)))

            if alpha_trial <= alphaVal + beta * t * dDir:
                found_step = True
                break
            t *= 0.5

        if not found_step:
            print(f"WARNING: Line search failed at iteration {it}.")
            break

        # Update controller
        K = K + t * H_dir
        history["K"].append(K.copy())

    print(f"Algorithm terminated after {it} iterations. Final spectral abscissa: {alpha_trial:e}")
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
    # Define the original matrix K0
    K0 = np.array([[ 0.0324,  0.0276,  0.0752, -0.2656],
        [ 0.0000,  0.0000,  0.0000,  0.0000]])
    #A = np.array([[0,1],[-1,0]])
    #B = np.array([[0],[1]])
    #C = np.array([[0,1])
    #K0 = np.array([[0.0]]) 

## A matrix (9x9)
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
    #Cy = np.eye(9)

    #C = Cy
    #B = Bu

## Initial static output feedback gain (1x2)
    #K0 = np.zeros((1, 9))

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
