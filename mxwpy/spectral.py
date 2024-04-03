__all__ = ['JacobiP', 'Jacobi_Gauss', 'Jacobi_Gauss_Lobatto', 
           'HermiteP', 'HermiteF', 'Hermite_Gauss', 'mapped_Jacobi_Gauss',
           'glue1D', 'glue_pts_1D',
           ]

"""
spectral.py

This module provides functions for working with spectral methods.
The implementation is mainly based on the book: 
    Shen, J., Tang, T. & Wang, L.-L. Spectral Methods: Algorithms, 
    Analysis and Applications. vol. 41 (Springer Science & Business 
    Media, 2011).
    https://link.springer.com/book/10.1007/978-3-540-71041-7
"""

import numpy as np
from scipy.special import roots_hermite, gamma
from scipy.sparse import diags, eye, lil_matrix, csr_matrix
from scipy.linalg import eigh, block_diag
from sympy import symbols, sqrt, atanh, tanh, sinh, log, lambdify, diff

def JacobiP(x, alpha, beta, N):
    """
    This function evaluates the orthonormal Jacobi polynomial of order 
    up to N with parameters alpha and beta at points x.
    
    Parameters
    ----------
    x : array
        Points at which the Jacobi polynomial is to be computed.
    alpha : float
        The alpha parameter of the Jacobi polynomial. Must be greater than -1.
    beta : float
        The beta parameter of the Jacobi polynomial. Must be greater than -1.
    N : int
        The order of the Jacobi polynomial.

    Returns
    ----------
    PL: ndarray, shape (N + 1, len(x))
        The N-th row of PL is the values of orthonormal Jacobi 
        polynomial J_{N}^{alpha, beta}(x) / sqrt(gamma_{N}^{alpha, beta}).

    References:
    ----------
    1. Spectral Method P74
    2. Code-reproduction/Poisson-GPU.ipynb
    """

    xp = x.copy()
    if len(xp.shape) == 2 and xp.shape[1] == 1:
        xp = xp.T
    PL = np.zeros((N + 1, max(xp.shape)))
    gamma0 = np.power(2, alpha + beta + 1) * gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 2)
    PL[0] = 1.0 / np.sqrt(gamma0)
    if N == 0:
        return PL.T
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)
    aold = 2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        anew = 2 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) / (h1 + 1) / (h1 + 3))
        bnew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2)
        PL[i + 1] = 1 / anew * (-aold * PL[i - 1] + (xp - bnew) * PL[i])
        aold = anew
    return PL

def Jacobi_Gauss(alpha, beta, N):
    """
    This function computes the Gauss Jacobi quadrature first order 
    derivative matrix, nodes and weights of Jacobi polynomial J_{N}^{alpha, beta}.

    Parameters
    ----------
    alpha : float
        The alpha parameter of the Gauss Jacobi quadrature. alpha > -1.
    beta : float
        The beta parameter of the Gauss Jacobi quadrature. beta > -1.
        If alpha = beta = 0, the Jacobi polynomial is Legendre polynomial.
    N : int
        The order of the Gauss Jacobi quadrature.

    Returns
    ----------
    D: ndarray, shape (N, N)
        The first order derivative matrix of the Gauss Jacobi quadrature.
    r: ndarray, shape (N,)
        The Gauss Jacobi quadrature nodes.
    w: ndarray, shape (N,)
        The Gauss Jacobi quadrature weights.

    References
    ----------
    1. Spectral Method P84
    """

    if N == 1:
        D = np.zeros((1,1))
        r = np.array([-(alpha - beta) / (alpha + beta + 2)])
        w = np.array([2])
        return D, r, w
    
    h1 = 2.0 * np.arange(N) + alpha + beta
    h11, h12, h13 = h1 + 1, h1 + 2, h1 + 3
    h2 = 1.0 * np.arange(1, N)
    # Adjust h1, h11, h12 values based on alpha and beta
    # to avoid division by zero
    if abs(alpha + beta) < 10 * np.finfo(float).eps:
        h1[0] = 1.0
    elif abs(alpha + beta + 1) < 10 * np.finfo(float).eps:
        h11[0] = 1.0
    elif abs(alpha + beta + 2) < 10 * np.finfo(float).eps:
        h1[1], h12[0] = 1.0, 1.0

    # equation (3.142) symmetric tridiagonal matrix A_{N+1}
    A = diags(0.5 * (beta**2 - alpha**2) / h12 / h1).toarray() + \
        diags(2 / h12[:-1] * np.sqrt(h2 * (h2 + alpha + beta) * \
        (h2 + alpha) * (h2 + beta) / h11[:-1] / h13[:-1]), 1).toarray()

    r, V = eigh(A + A.T)
    # equation (3.144)
    w = np.power(V[0, :], 2) * np.power(2, alpha + beta + 1) * \
        gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 2)
    
    l = JacobiP(r, alpha + 1, alpha + 1, N - 1)[-1]
    # construct first order JG derivative matrix D by equation (3.164)
    Distance = r[:, None] - r[None, :] + np.eye(N)
    D = l[:, None] / l[None, :] / Distance
    np.fill_diagonal(D, 0)

    # for program stability, we force row sum of D to be 0
    # which ensure the derivative of constants to be zero matrix
    np.fill_diagonal(D, -np.sum(D, axis=1))
    return D, r, w


def Jacobi_Gauss_Lobatto(alpha, beta, N):
    """
    This function computes the Gauss-Lobatto quadrature first order
    derivative matrix, nodes and weights of Jacobi polynomial J_{N}^{alpha, beta}.
    The nodes are {-1, 1, zeros of dx(J_N^{alpha, beta}(x))}

    Parameters
    ----------
    alpha : float
        The alpha parameter of the Gauss-Lobatto quadrature. alpha > -1.
    beta : float
        The beta parameter of the Gauss-Lobatto quadrature. beta > -1.
        If alpha = beta = 0, the Jacobi polynomial is Legendre polynomial.
    N : int
        The order of the Gauss-Lobatto quadrature.

    Returns
    ----------
    D: ndarray, shape (N + 1, N + 1)
        The first order derivative matrix of the Gauss-Lobatto quadrature.
    r: ndarray, shape (N + 1,)
        The Gauss-Lobatto quadrature nodes.
    w: ndarray, shape (N + 1,)
        The Gauss-Lobatto quadrature weights.

    References
    ----------
    1. Spectral Method P83 
    2. Code-reproduction/Poisson-GPU.ipynb
    """

    r = np.zeros((N + 1,))
    r[0], r[-1] = -1.0, 1.0
    w = np.zeros((N + 1,))
    w[0] = (beta + 1) * gamma(beta + 1)**2
    w[-1] = (alpha + 1) * gamma(alpha + 1)**2
    
    if N > 1:
        # dx(J_N^{alpha, beta}(x)) = C(alpha, beta, N) * J_{N-1}^{alpha+1, beta+1}(x)
        # thus have same zeros
        r[1:-1] = Jacobi_Gauss(alpha + 1, beta + 1, N - 1)[1]
        # equ(3.139)
        cd = 2**(alpha + beta + 1) * gamma(N) / gamma(N + alpha + beta + 2)
        md = gamma(N + alpha + 1) / gamma(N + beta + 1)
        w[0] *= cd * md
        w[-1] *= cd / md
        w[1:-1] = (2 * N + alpha + beta + 1) / (1 - r[1:-1]**2)**2 / (N-1) / (N + alpha + beta + 2) / JacobiP(r[1:-1], alpha + 2, beta + 2, N - 2)[-1]**2
        
    # construct first order LGL derivative matrix D
    Distance = r[:, None] - r[None, :] + np.eye(N + 1)
    omega = np.prod(Distance, axis=1)
    # equation (3.75)
    D = diags(omega) @ (1 / Distance) @ diags(1 / omega)
    # for program stability, we force row sum of D to be 0
    # which ensure the derivative of constants to be zero matrix
    np.fill_diagonal(D, 0)
    np.fill_diagonal(D, -np.sum(D, axis=1))
    return D, r, w


def HermiteP(x, N, normalized=False, return_full=True):
    """
    Evaluate orthognormal Hermite polynomial of degree N at x by recurrence relation.

    Parameters
    ----------
    x : ndarray
        The input array.
    N : int
        The degree of Hermite polynomial.
    normalized : bool, optional
        Whether to normalize PL[N] to have L2 norm 1. Default is False.
    return_full : bool, optional
        Whether to return the full PL array. Default is True.

    Returns
    ----------
    PL[N] or PL : ndarray, shape (N + 1, len(x))
        The value of Hermite polynomial of degree N at x, or the full PL array if return_full is True.

    References
    ----------
    Spectral Method P254
    """
    
    xp = x.copy()
    if len(xp.shape) == 2 and xp.shape[1] == 1:
        xp = xp.T
    PL = np.ones((N + 1, max(xp.shape)))
    if normalized:
        Norm = np.zeros(N + 1)
        Norm[0] = np.linalg.norm(PL[0])
    if N == 0:
        return PL[0] / Norm[0] if normalized else PL[0]
    PL[1] = 2 * xp
    if normalized:
        Norm[1] = np.linalg.norm(PL[1])
    if N == 1 and normalized:
        PL[0] /= Norm[0]
    for i in range(1, N):
        PL[i + 1] = 2 * xp * PL[i] - 2 * i * PL[i - 1]
        if normalized:
            if i == 1:
                PL[0] /= Norm[0]
            PL[i] /= Norm[i]
            PL[i + 1] /= Norm[i]
            Norm[i + 1] = np.linalg.norm(PL[i + 1])
    if normalized:
        PL[N] /= Norm[N]
    return PL if return_full else PL[N]


def HermiteF(x, N, return_full=True):
    """
    Evaluate modified Hermite Function of degree N at x by recurrence relation.

    Parameters
    ----------
    x : ndarray
        The input array.
    N : int
        The degree of Hermite polynomial.
    return_full : bool, optional
        Whether to return the full PL array. Default is True.

    Returns
    ----------
    PL[N] or PL : ndarray, shape (N + 1, len(x))
        The value of modified Hermite polynomial of degree N at x, or the full PL array if return_full is True.

    References
    ----------
    Spectral Method P256
    """

    xp = x.copy()
    if len(xp.shape) == 2 and xp.shape[1] == 1:
        xp = xp.T
    PL = np.ones((N + 1, max(xp.shape)))
    PL[0] = np.exp(-xp ** 2 / 2) / np.pi ** 0.25    # underflow may occur for large r
    if N == 0:
        return PL[0]
    PL[1] = np.sqrt(2) * xp * PL[0]
    for i in range(1, N):
        PL[i + 1] = np.sqrt(2 / (i + 1)) * xp * PL[i] - np.sqrt(i / (i + 1)) * PL[i - 1]
    return PL if return_full else PL[N]


def Hermite_Gauss(N, c=1 / np.sqrt(2)):
    """
    Generate the Hermite-Gauss(HG) quadrature points r and weights w w.r.t Hermite function.

    Parameters
    ----------
    N : int
        Number of points, underflow will occur for N > 740 with default c.
    c : float, optional
        The decay factor for v = exp(-(cx)^2) * u(x). Default is 1 / sqrt(2).
        If c = 0, it degenerates to Hermite-ploynomial case.

    Returns
    ----------
    D : ndarray, shape (N, N)
        The first order derivative matrix of Hermite-Gauss quadrature points. 
    r : ndarray, shape (N,)
        The Hermite-Gauss quadrature points.
    w : ndarray, shape (N,)
        The weights of Hermite-Gauss quadrature points.

    References
    ----------
    Spectral Method P261
    """

    r, w = roots_hermite(N)
    if c != 0:
        w = 1 / (HermiteF(r, N - 1, False)**2 * N)  # equ(7.81), modified weights for Hermite function
    H = HermiteP(r, N - 1, True, False) * np.exp(-(c * r) ** 2)  # equ(7.93), underflow may occur for large r
    dis = r[:, np.newaxis] - r[np.newaxis, :]
    np.fill_diagonal(dis, 1)
    D = H[:, np.newaxis] / H[np.newaxis, :] / dis
    np.fill_diagonal(D, r * (1 - 2 * c**2))
    return D, r, w



def mapped_Jacobi_Gauss(alpha, beta, N, sf = 1, mapping = 'alg'):
    """
    Mapped Jacobi-Gauss quadrature points, weights and first order derivative matrix.
    The mapping is defined by the function y = map(r, s), where r is the original Jacobi-Gauss quadrature points,
    and s is the scaling factor of the mapping.

    Parameters
    ----------
    alpha : float
        The alpha parameter of the Jacobi polynomial. alpha > -1.
    beta : float
        The beta parameter of the Jacobi polynomial. beta > -1.
        If alpha = beta = 0, the Jacobi polynomial is Legendre polynomial.
    N : int
        The order of the Jacobi polynomial.
    sf : float, optional
        The scaling factor of the mapping. Default is 1.
    mapping : str, optional
        The mapping function. Default is 'alg', which means using the algebraic mapping (cf. equ(7.159)).
        Other options are 'log' and 'exp'.

    Returns
    ----------
    D : ndarray, shape (N, N)
        The first order derivative matrix of the mapped Jacobi-Gauss quadrature.
    r : ndarray, shape (N,)
        The mapped Jacobi-Gauss quadrature nodes.
    w : ndarray, shape (N,)
        The modified mapped Jacobi-Gauss quadrature weights.

    References
    ----------
    1. Spectral Method P280, P286
    """

    y, s = symbols('y s')

    if mapping == 'alg':
        map_expr = s * y / sqrt(1 - y**2)
        diff_map_expr = diff(map_expr, y)
    elif mapping == 'log':
        map_expr = s * atanh(y)
        diff_map_expr = diff(map_expr, y)
    elif mapping == 'exp':
        map_expr = sinh(s * y)
        diff_map_expr = diff(map_expr, y)
    else:
        raise ValueError("Invalid mapping function. Please choose from 'alg', 'log' and 'exp'.")

    map = lambdify((y, s), map_expr, "numpy")
    diff_map = lambdify((y, s), diff_map_expr, "numpy")

    D, r, w = Jacobi_Gauss(alpha, beta, N)
    omega = (1 - r)**alpha * (1 + r)**beta
    diff_g = diff_map(r, sf)
    D, r, w = D / diff_g[:, None], map(r, sf), w / omega * diff_g
    return D, r, w




def glue1D(interval, Ncell, D, r, w, end_pts = False):
    """
    Glue the differential matrix D, Gauss points r, and quadrature weights w on each cell together.

    Parameters
    ----------
    interval : list or tuple, length 2
        The left and right edge of the domain.
    Ncell : int
        The number of cells.
    D, r, w : ndarray
        The differential matrix, Gauss points, and quadrature weights on the reference cell [-1, 1],
        which can be generated by `spectral.Jacobi_Gauss`, `spectral.Jacobi_Gauss_Lobatto`.
    end_pts : bool, optional
        Whether the end points -1, 1 are included in r or not. The default is False.
        E.g., if `spectral.Jacobi_Gauss_Lobatto` is used, the end points are included in r.

    Returns
    ----------
    D, r, w : ndarray
        The differential matrix D, Gauss points r, and quadrature weights w on the interval.
        if end_pts is False, shape (Ncell * Np, Ncell * Np), (Ncell * Np,), (Ncell * Np,)
        if end_pts is True, shape (Ncell * (Np - 1) + 1, Ncell * (Np - 1) + 1), (Ncell * (Np - 1) + 1,), (Ncell * (Np - 1) + 1,)
    """

    Np = len(r)
    D, r, w = D.copy(), r.copy(), w.copy()
    left, right = interval
    Len = (right - left) / Ncell / 2
    r = (r + 1) * Len + left
    r = r[None, :].repeat(Ncell, axis=0)
    r = r + np.arange(Ncell)[:, None] * Len * 2
    D_blocks = [D / Len] * Ncell
    D, r, w = block_diag(*D_blocks), r.reshape(-1), np.tile(w, Ncell) * Len
    if end_pts:
        Glue = lil_matrix(np.zeros((Ncell * (Np - 1) + 1, Ncell * Np)))
        for j in range(Ncell):
            rowStart, colStart = j * (Np - 1), j * Np
            Glue[rowStart:rowStart+Np, colStart:colStart+Np] = eye(Np)
        D, Glue = csr_matrix(D), Glue.tocsr()
        D = (Glue @ D @ Glue.T).toarray()
        r = np.delete(r, np.arange(Np-1, len(r)-1, Np))
        w = Glue @ w
    return D, r, w

def glue_pts_1D(interval, Ncell, r, end_pts = False):
    """
    Glue the points r on each cell together.

    Parameters
    ----------
    interval : list or tuple, length 2
        The left and right edge of the domain.
    Ncell : int
        The number of cells.
    r : ndarray
        The collocation points on the reference cell [-1, 1],
    end_pts : bool, optional
        Whether the end points -1, 1 are included in r or not. The default is False.

    Returns
    ----------
    r : ndarray
        The collocation points r on the interval.
        if end_pts is False, shape (Ncell * Np,)
        if end_pts is True, shape (Ncell * (Np - 1) + 1,)
    """

    Np = len(r)
    r = r.copy()
    left, right = interval
    Len = (right - left) / Ncell / 2
    r = (r + 1) * Len + left
    r = r[None, :].repeat(Ncell, axis=0)
    r = r + np.arange(Ncell)[:, None] * Len * 2
    r = r.reshape(-1)
    if end_pts:
        r = np.delete(r, np.arange(Np-1, len(r)-1, Np))
    return r