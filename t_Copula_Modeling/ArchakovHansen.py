import numpy as np
from scipy.linalg import expm

## Implementation
def ArchakovHansen(n: int, v: np.ndarray, eps: float, kmax: int = 100000) -> np.ndarray:
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