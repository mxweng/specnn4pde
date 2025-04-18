"""
linalg.py

This module implements some linear algebra algorithms using NumPy.

Brief description:
    - ROU_cholesky: Perform a rank-one update of the Cholesky decomposition of a matrix.
    - Butcher_step: Perform a single step of a Runge-Kutta method using a Butcher tableau.
    - Butcher_table: Store the Butcher tableau of a Runge-Kutta method.

For more information, see the documentation of each function.
"""

__all__ = ['ROU_cholesky', 'Butcher_step', 'Butcher_table',
           ]

import numpy as np

def ROU_cholesky(L, v, alpha=1, beta=1):
    """
    Perform a rank-one update of the Cholesky decomposition of a matrix.
    The complexity of the rank-one update is O(n^2), where n is the size of the matrix.

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



def Butcher_step(butcher_table: np.ndarray, f, xn: float, yn: float, h: float = 0.1):
    """
    Perform a single step of a Runge-Kutta method using a Butcher tableau.

    The Butcher tableau is a matrix used to describe the coefficients of Runge-Kutta methods,
    which are used for the numerical solution of ordinary differential equations.

    ** This function is only for explicit Runge-Kutta methods, and only applicable to 1D problem.**

    Parameters
    ----------
    butcher_table : ndarray
        The Butcher tableau for the Runge-Kutta method. The last row contains the weights (b),
        and the first column of the other rows contains the nodes (c). The rest of the matrix (A)
        contains the coefficients.
    f : function
        The function defining the differential equation dy/dx = f(x, y).
    xn : float
        The current value of x.
    yn : float
        The current value of y.
    h : float, optional
        The step size. Default is 0.1.

    Returns
    ----------
    yn_next : float
        The estimated value of y at x + h.

    References
    ----------
    1. https://zhuanlan.zhihu.com/p/408015963
    2. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods

    Example
    ----------
    >>> # Example of a Butcher tableau for the Euler method
    >>> # ODE: dy/dx = y - 2x/y, y(0) = 1, x in [0, 5]
    >>> 
    >>> table = Butcher_table('forward_euler')
    >>> f = lambda x, y: y - 2 * x / y
    >>> 
    >>> N = 50
    >>> xx = np.linspace(0, 5, N + 1, endpoint=True)
    >>> yy = np.zeros_like(xx)
    >>> yy[0] = 1
    >>> for i in range(N):
    >>>     yy[i + 1] = Butcher_step(table, f, xx[i], yy[i], h=xx[i + 1] - xx[i])
    """
    A = butcher_table[:-1, 1:]
    c = butcher_table[:-1, 0]
    b = butcher_table[-1, 1:]
    k = np.zeros_like(b)
    for i, A_row in enumerate(A):
        k[i] = f(xn + c[i] * h, yn + sum(A_row * k) * h)
    return yn + sum(b * k) * h

def Butcher_table(type: str):
    """
    Generates the Butcher tableau for various Runge-Kutta methods.

    The Butcher tableau is a systematic way to represent the coefficients of Runge-Kutta (RK) methods,
    which are used for the numerical solution of ordinary differential equations (ODEs). This function
    returns the Butcher tableau for a specified RK method.

    Parameters
    ----------
    type : str
        The type of Runge-Kutta method for which to generate the Butcher tableau. Supported types include:
        - 'forward_euler': Explicit Euler method
        - 'backward_euler': Implicit Euler method
        - 'CN2': Crank-Nicolson method (2nd order)
        - 'heun': Heun's method
        - 'SSPRK3': Third-order Strong Stability Preserving Runge-Kutta method
        - 'RK4': Classic fourth-order Runge-Kutta method
        - 'RK5': Fifth-order Runge-Kutta method

    Returns
    ----------
    table : ndarray
        The Butcher tableau for the specified Runge-Kutta method. The last row of the tableau contains
        the weights for the linear combination of slopes (k values), the first column (excluding the last row)
        contains the c coefficients (time fractions), and the rest of the matrix contains the a coefficients
        (weights for the slopes in the linear combinations).

    References
    ----------
    1. https://zhuanlan.zhihu.com/p/408015963
    2. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
    """

    if type == 'forward_euler':
        table = np.array([[0., 0.],
                        [0., 1.]])
    elif type == 'backward_euler':
        table = np.array([[1., 1.],
                        [0., 1.]])
    elif type == 'CN2':
        table = np.array([[0., 0., 0.],
                        [1., 0.5, 0.5],
                        [0., 0.5, 0.5]])
    elif type == 'heun':
        table = np.array([[0., 0., 0.],
                        [1., 1., 0.],
                        [0., 0.5, 0.5]])
    elif type == 'SSPRK3':
        table = np.array([[0., 0., 0., 0.],
                        [1., 1., 0., 0.],
                        [0.5, 0.25, 0.25, 0],
                        [0, 1 / 6, 1 / 6, 2 / 3]])
    elif type == 'RK4':
        table = np.array([[0., 0., 0., 0., 0.],
                        [0.5, 0.5, 0., 0., 0],
                        [0.5, 0., 0.5, 0, 0],
                        [1, 0, 0, 1, 0],
                        [0, 1 / 6, 1 / 3, 1 / 3, 1 / 6]])
    elif type == 'RK5':
        table = np.array([[0., 0., 0., 0., 0., 0, 0],
                    [0.25, 0.25, 0., 0., 0, 0, 0],
                    [3 / 8, 3 / 32., 9 / 32, 0, 0, 0, 0],
                    [12 / 13, 1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
                    [1, 439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
                    [1 / 2, -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
                    [0, 16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]])
    else:
        raise ValueError(f"Invalid Butcher type: {type}!")
    
    return table