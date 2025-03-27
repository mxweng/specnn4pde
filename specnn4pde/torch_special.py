__all__ = ['JacobiP_torch', 'BesselI', 'BesselK', 
           ]

"""
torch_special.py

This module implements some special functions using PyTorch.
"""

import torch


def JacobiP_torch(x, alpha, beta, N):
    """
    This function evaluates the Jacobi polynomial of order 
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
        The N-th row of PL is the values of Jacobi polynomial 
        J_{N}^{alpha, beta}(x) / sqrt(gamma_{N}^{alpha, beta}).
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor.")
    
    x = x.flatten()
    PL = torch.zeros((N + 1, len(x)), dtype=x.dtype, device=x.device)
    PL[0] = 1.0
    if N == 0:
        return PL
    
    PL[1, :] = ((alpha + beta + 2) * x + alpha - beta) / 2
    for n in range(1, N):
        a_n = (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2) / (2 * (n + 1) * (n + alpha + beta + 1))
        b_n = ((beta**2 - alpha**2) * (2 * n + alpha + beta + 1)) / \
                (2 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta))
        c_n = (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2) / \
                ((n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta))
        PL[n + 1, :] = (a_n * x - b_n) * PL[n, :] - c_n * PL[n - 1, :]
    return PL


def besseli_smallx(v, x):
    """
    Calculate I_v(x) using series expansion and adaptive truncation to ensure accuracy and efficiency.
    Only suitable for x < max(10, 2 * (v + 1)).

    Referrence: 
        https://live.boost.org/doc/libs/1_87_0/libs/math/doc/html/math_toolkit/bessel/mbessel.html
    
    Parameters
    ----------
    v : float
        Order of the Bessel function.
    x : tensor
        Points at which the Bessel function is to be computed.

    Returns
    -------
    I_v(x) : tensor
        Values of the Bessel function at the given points.
    """
    term = (x / 2) ** v / torch.exp(torch.lgamma(torch.tensor(v + 1, device=x.device, dtype=x.dtype)))
    sum_series = term.clone()
    tol = torch.finfo(x.dtype).eps
    k = 1

    while True:
        term *= (x / 2) ** 2 / (k * (k + v))
        sum_series += term
        k += 1
        indicator = torch.abs(term / sum_series)
        indicator[torch.isnan(indicator)] = 0
        if torch.max(indicator) <= tol:
            break
    # print(k, torch.max(torch.abs(term[mask] / sum_series[mask])))
    return sum_series


def BesselI_smallx(n, x, upto = False):
    """
    This function evaluates the modified Bessel function of the first kind I_n(x).
    Suitable for x < max(10, 2 * (v + 1)).

    Parameters
    ----------
    n : int
        The order of the Bessel function.
    x : tensor
        Points at which the Bessel function is to be computed.
    upto : bool, optional, default False
        If True, return I_0(x), I_1(x), ..., I_n(x).

    Returns
    ----------
    I : tensor
        If upto is True, I is a tensor of shape (n+1, n+1, x.numel()) where I[k] is I_k(x).
        If upto is False, I = I_n(x) mantains the shape of x.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor.")
    
    shape = x.shape
    n, x = abs(n), x.flatten()
    I = torch.zeros((n+3, len(x)), dtype=x.dtype, device=x.device)
    
    I[-1] = besseli_smallx(n+2, x)
    I[-2] = besseli_smallx(n+1, x)
    to = -1 if upto else n-1
    for k in range(n, to, -1):
        I[k] = 2 * (k + 1) / x * I[k+1] + I[k+2]

    return I[:n+1] if upto else I[n].reshape(shape)


def BesselI_largex(n, x, upto = False):
    """
    This function evaluates the modified Bessel function of the first kind I_n(x).
    Suitable for x > max(10, 2 * (v + 1)).

    Parameters
    ----------
    n : int
        The order of the Bessel function.
    x : tensor
        Points at which the Bessel function is to be computed.
    upto : bool, optional, default False
        If True, return I_0(x), I_1(x), ..., I_n(x).

    Returns
    ----------
    I : tensor
        If upto is True, I is a tensor of shape (n+1, n+1, x.numel()) where I[k] is I_k(x).
        If upto is False, I = I_n(x) mantains the shape of x.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor.")
    
    shape = x.shape
    n, x = abs(n), x.flatten()
    I = torch.zeros((n+1, len(x)), dtype=x.dtype, device=x.device)

    I[0] = torch.special.i0(x)
    if n == 0:
        return I if upto else I[-1].reshape(shape)

    I[1] = torch.special.i1(x)
    for k in range(2, n + 1):
        I[k] = - 2 * (k - 1) / x * I[k-1] + I[k-2]

    return I if upto else I[-1].reshape(shape)


def BesselI(n, x, upto = False):
    """
    This function evaluates the modified Bessel function of the first kind I_n(x).

    Parameters
    ----------
    n : int
        The order of the Bessel function.
    x : tensor
        Points at which the Bessel function is to be computed.
    upto : bool, optional, default False
        If True, return I_0(x), I_1(x), ..., I_n(x).

    Returns
    ----------
    I : tensor
        If upto is True, I is a tensor of shape (n+1, x.numel()) where I[k] is I_k(x).
        If upto is False, I = I_n(x) mantains the shape of x.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor.")
    
    shape = x.shape
    n, x = abs(n), x.flatten()
    mask = x < max(10, 2 * (n + 1))
    
    if upto:
        I = torch.zeros((n+1, len(x)), dtype=x.dtype, device=x.device)
        I[:, mask] = BesselI_smallx(n, x[mask], upto)
        I[:, ~mask] = BesselI_largex(n, x[~mask], upto)
    else:
        I = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        I[mask] = BesselI_smallx(n, x[mask], upto)
        I[~mask] = BesselI_largex(n, x[~mask], upto)
        
    return I if upto else I.reshape(shape)


def BesselK(n, x, upto = False):
    """
    This function evaluates the modified Bessel function of the second kind K_n(x).

    Parameters
    ----------
    n : int
        The order of the Bessel function.
    x : tensor
        Points at which the Bessel function is to be computed.
    upto : bool, optional, default False
        If True, return K_0(x), K_1(x), ..., K_n(x).

    Returns
    ----------
    K : tensor
        If upto is True, K is a tensor of shape (n+1, x.numel()) where K[k] is K_k(x).
        If upto is False, K = K_n(x) mantains the shape of x.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor.")
    
    shape = x.shape
    n, x = abs(n), x.flatten()
    K = torch.zeros((n+1, len(x)), dtype=x.dtype, device=x.device)

    K[0] = torch.special.modified_bessel_k0(x)
    if n == 0:
        return K if upto else K[-1].reshape(shape)

    K[1] = torch.special.modified_bessel_k1(x)
    for k in range(2, n + 1):
        K[k] = 2 * (k - 1) / x * K[k-1] + K[k-2]

    return K if upto else K[-1].reshape(shape)

