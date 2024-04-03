__all__ = ['ROU_cholesky',
           ]

import numpy as np

def ROU_cholesky(L, v, alpha=1, beta=1):
    """
    Perform a rank-one update of the Cholesky decomposition of a matrix.

    Parameters
    ----------
    L : ndarray
        The lower triangular Cholesky factor of the matrix A.
    alpha : float
        The scalar multiplier for the matrix. Must be non-negative.
    beta : float
        The scalar multiplier for the outer product of v. Must be non-negative.
    v : ndarray
        The vector used for the rank-one update.

    Returns
    ----------
    L_prime : ndarray
        The updated lower triangular Cholesky factor of the matrix
         \tilde{A} = alpha * A + beta * v * v^T.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
    2. Krause Oswin, Igel ChristianA, 2015, 
        More Efficient Rank-one Covariance Matrix Update for Evolution Strategies,
        https://christian-igel.github.io/paper/AMERCMAUfES.pdf

    Example
    ----------
    >>> L = np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1]])
    >>> alpha = 2
    >>> beta = 3
    >>> v = np.array([1, 2, 3])
    >>> L_prime = ROU_cholesky(L, v, alpha, beta)
    >>> print(L_prime)
    """

    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    
    n = L.shape[0]
    L, x = np.sqrt(alpha) * L, np.sqrt(beta) * v
    for k in range(n):
        r = np.sqrt(L[k, k]**2 + x[k]**2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k < n - 1:
            L[(k+1):n, k] = (L[(k+1):n, k] + s * x[(k+1):n]) / c
            x[(k+1):n] = c * x[(k+1):n] - s * L[(k+1):n, k]
    return L