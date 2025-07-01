import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.linalg import expm

## Implementation
def ArchakovHansen(n: int, v: np.ndarray, eps: float=1e-15, kmax: int = 100000) -> np.ndarray:
    x = np.ones(n)
    A = np.zeros((n, n))
    np.fill_diagonal(A, x)
    i = np.tril_indices(n, -1)
    A[i] = v
    A = A + A.T - np.diag(np.diag(A))
    dist = np.inf
    k = 0
    while dist > eps:
        y = x - np.log(np.diag(expm(A)))
        dist = np.linalg.norm(x-y)
        x = y
        np.fill_diagonal(A, x)
        k += 1
        if k>kmax:
            print('Error: max number of iterations reached')
            return np.zeros(n,n)
    C = expm(A)
    C = (C+C.T)/2
    return C


def estimate_copula_corr_kendall(returns: np.ndarray) -> np.ndarray:
    """
    Estimate the copula correlation matrix (Omega) for a t-copula
    using Kendall's tau moment matching.

    Parameters:
    -----------
    return : numpy.ndarray
        2D array-like structure (n_samples x n_assets)

    Returns:
    --------
    omega : np.ndarray
        Estimated t-copula correlation matrix (d x d)
    """
    d = returns.shape[1]
    tau_matrix = np.zeros((d, d))

    # Compute pairwise Kendall's tau
    for i in range(d):
        for j in range(i, d):
            tau, _ = kendalltau(returns[:, i], returns[:, j])
            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau  # Symmetric

    # Convert to copula correlation matrix using sin(pi/2 * tau)
    omega = np.sin((np.pi / 2) * tau_matrix)

    return omega

