__all__ = ['gradients', 'Jacobian', 'partial_derivative', 'partial_derivative_vector', 
           'meshgrid_to_matrix', 'gen_collo',
           ]

import torch
from torch.autograd.functional import jacobian


def gradients(u, x, order=1, retain_graph=False):
    """
    Compute the gradients for d dimensional function. It only supports two kinds of functions:

    1. scalar function f(x1, x2, ..., xd)
        The first order gradients are [df/dx1, df/dx2, ..., df/dxd]
    2. vector function like F(x1,x2,...,xd) = [f1(x1), f2(x2), ..., fd(xd)]
        The first order gradients are [df1/dx1, df2/dx2, ..., dfn/dxd]

    Higher order gradients are also supported.

    !!! For functions like F(x1, ..., xd) = [f1(x1, ..., xd), ..., fd(x1, ..., xd)], 
        use `partial_derivative_vector` instead.

    Parameters
    ----------
    u : tensor
        The values of the function at the point x.
    x : Tensor, shape (n, d)
        The point at which to compute the gradients, where n is the number of points and d is the dimension.
    order : int, optional
        The order of the gradients. The default is 1.
    retain_graph : bool, optional
        Whether to retain the computational graph for further computation. Defaults to False.

    Returns
    ----------
    grads: list of tensors of shape like x
        The gradients up to the order. grads[i] is the (i+1)-th order gradients.

    Example
    ----------
    >>> def f(x):
    ...     return x**2
    >>> x = torch.tensor([[1.0, 2], [3, 4], [5, 6]], requires_grad=True)
    >>> u = f(x)
    >>> gradients(u, x, 2)
    [tensor([[ 2.,  4.],
             [ 6.,  8.],
             [10., 12.]]),
     tensor([[2., 2.],
             [2., 2.],
             [2., 2.]])]
    """

    grads = [torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]]
    for _ in range(1, order):
        grads.append(torch.autograd.grad(grads[-1], x, grad_outputs=torch.ones_like(grads[-1]), create_graph=True)[0])
    # clean the computational graph of the gradients to save memory
    if not retain_graph:
        grads = [g.detach() for g in grads]
    return grads


def Jacobian(f, x, order=1, create_graph=False):
    """
    Compute the Jacobian of a vector function.
    But only support univariate function. For multivariate vector function, 
    use `partial_derivative_vector` instead.

    !!! Not suitale for high order derivatives, because
        creating computational graphs will consume a lot of time and memory.

    Parameters
    ----------
    f : function
        The function to compute the Jacobian for.
    x : Tensor
        The point at which to compute the Jacobian.
    order : int, optional
        The order of the derivative. Defaults to 1.
    create_graph : bool, optional
        Whether to create a computational graph. Defaults to False.

    Returns
    ----------
    Tensor
        The Jacobian of the function at the given point.

    Example
    ----------
    >>> def f(x):
    ...     return torch.cat([x, x**2], dim=1)
    >>> x = torch.tensor([[1.0], [2], [3]])
    >>> Jacobian(f, x)
    tensor([[1., 2.],
            [1., 4.],
            [1., 6.]])
    """

    def _f(*args, **kwargs):
        return f(*args, **kwargs).sum(dim=0)
    if order == 0:
        return f(x) if create_graph else f(x).detach()
    elif order == 1:
        return jacobian(_f, x, create_graph).squeeze(2).T
    else:
        def _jacobian(x):
            return jacobian(_f, x, True).squeeze(2).T
        return Jacobian(_jacobian, x, order-1, create_graph)

def partial_derivative(F, X, Alpha, create_graph=False):
    """
    Compute the partial derivative for vector-valued function but inefficient, 
    use `partial_derivative_vector` instead.

    !!! Not suitale for high order derivatives, because
        creating computational graphs will consume a lot of time and memory.    

    Parameters
    ----------
    F : function
        The function to compute the partial derivative for.
    X : Tensor
        The points at which to compute the partial derivative.
    Alpha : list
        The order of the derivative for each dimension.
    create_graph : bool, optional
        Whether to create a computational graph. Defaults to False.

    Returns
    ----------
    Tensor
        The partial derivative of the function at the given points.
    """

    if len(Alpha) == 1:
        return Jacobian(F, X, Alpha[0], create_graph)
    else:
        X_perfix, x_last = X[:, :-1], X[:, -1:]
        def _f(x):
            def _F(X):
                return F(torch.cat([X, x], dim=1))
            return partial_derivative(_F, X_perfix, Alpha[:-1], True)
        return Jacobian(_f, x_last, Alpha[-1], create_graph)

def partial_derivative_vector(F, X, Alpha, create_graph=False, batch_size=[15000,1]):
    """
    Compute the partial derivatives for vector-valued function 
    F(x1, x2, ..., xn) = [f1(x1, x2, ..., xn), f2(x1, x2, ..., xn), ..., fk(x1, x2, ..., xn)],
    return [\partial^\alpha f1, \partial^\alpha f2, ..., \partial^\alpha fn].

    !!! Not suitale for high order derivatives, because
        creating computational graphs will consume a lot of time and memory.

    Parameters
    ----------
    F : function
        The function to compute the partial derivatives for.
    X : tensor, shape (N, d)
        The points at which to compute the partial derivatives, where N is the number of points and d is the dimension.
    Alpha : list
        The order of the derivative for each dimension.
    create_graph : bool, optional
        Whether to create a computational graph. Defaults to False.
    batch_size : list, optional
        The batch size for computing the partial derivatives. 
        batch_size[0] is the number of points in each batch, 
        and batch_size[1] is the number of functions in each batch.
        Defaults to [15000, 1].

    Returns
    ----------
    Tensor, shape (k, N, d)
        The partial derivatives of the function at the given points.

    Example
    ----------
    >>> def F(X):
    ...     return X.prod(dim=1).unsqueeze(dim=1).repeat(1, 2)
    >>> X = torch.tensor([[1.0, 2], [3, 4]])
    >>> partial_derivative_vector(F, X, Alpha = [1, 1])
    tensor([[1., 1.],
            [1., 1.]])
    """

    out_dim = F(X[:1]).shape[1]
    res = []
    for i in range(0, X.shape[0], batch_size[0]):
        res_sub = []
        for j in range(0, out_dim, batch_size[1]):
            def _F(X): return F(X)[:, j:j+batch_size[1]]
            res_sub.append(partial_derivative(_F, X[i:i+batch_size[0]], Alpha, create_graph))
        res.append(torch.cat(res_sub, dim=1))
    return torch.cat(res, dim=0)


def meshgrid_to_matrix(inputs, indexing='xy'):
    """
    Convert the meshgrid to matrix.

    Parameters
    ----------
    inputs : list of iterables, length d
        The grid points in each dimension.
    indexing : str, optional
        The indexing of the meshgrid. The default is 'xy'.
        The options are 'xy' and 'ij', the same as numpy.meshgrid and torch.meshgrid.

    Returns
    ----------
    tensor, shape ( n1*n2*...*nd, d)
        The matrix of the grid points, ni is the number of grid points in the i-th dimension.

    Example
    ----------
    >>> x = torch.linspace(1, 2, 3)
    >>> y = torch.linspace(4, 5, 3)
    >>> meshgrid_to_matrix([x, y], indexing='xy')
    tensor([[1.0000, 4.0000],
            [1.5000, 4.0000],
            [2.0000, 4.0000],
            [1.0000, 4.5000],
            [1.5000, 4.5000],
            [2.0000, 4.5000],
            [1.0000, 5.0000],
            [1.5000, 5.0000],
            [2.0000, 5.0000]])
    >>> meshgrid_to_matrix([x, y], indexing='ij')
    tensor([[1.0000, 4.0000],
            [1.0000, 4.5000],
            [1.0000, 5.0000],
            [1.5000, 4.0000],
            [1.5000, 4.5000],
            [1.5000, 5.0000],
            [2.0000, 4.0000],
            [2.0000, 4.5000],
            [2.0000, 5.0000]])
    """

    Co = torch.meshgrid(*inputs, indexing=indexing)
    return torch.cat([c.reshape(-1,1) for c in Co], dim=1)


def gen_collo(Domain, grids, temporal = False, corner = True):
    """
    Generate the collocation points for the PDE problem on regular domain.
    If Domain and grids are provided, the uniform grids will be generated automatically as G.
    If Domain and grids are not provided, G should be provided.

    Parameters
    ----------
    Domain : list of list, optional
        The domain of the problem. eg. [[t_min, x1_min, x2_min, ...], [t_max, x1_max, x2_max, ...]]
    grids : list, optional
        The number of collocations in each dimension. eg. [N_t, N_x1, N_x2, ...]
    temporal : bool, optional
        If the problem is temporal. The default is False.
    corner : bool, optional
        If the collocation points include the corner points. The default is True.
    G : list of tensor, optional
        The tensors in the list are the collocation points in each dimension.
        If Domain and grids are not provided, G should be provided.
    
    Returns
    -------
    collo_rs : tensor
        The collocation points in the interior of the domain.
    collo_ic : tensor, optional
        If temporal is set as True. The collocation points on the initial condition.
    collo_bc : tensor
        The collocation points on the boundary condition.

    Example
    ----------
    >>> domian = [[0, 0, 1], [2, 3, 4]]
    >>> grids = [3, 4, 5]
    >>> gen_collo(domian, grids)
    (tensor([[1.0000, 1.0000, 1.7500],
             [1.0000, 1.0000, 2.5000],
             [1.0000, 1.0000, 3.2500],
             [1.0000, 2.0000, 1.7500],
             [1.0000, 2.0000, 2.5000],
             [1.0000, 2.0000, 3.2500]]),
     tensor([[0.0000, 0.0000, 1.0000],
             [0.0000, 0.0000, 1.7500],
             [0.0000, 0.0000, 2.5000],
             [0.0000, 0.0000, 3.2500],
             [0.0000, 0.0000, 4.0000],
             [2.0000, 0.0000, 1.0000],
             ......
             [1.0000, 1.0000, 4.0000],
             [1.0000, 2.0000, 1.0000],
             [1.0000, 2.0000, 4.0000]]))
    """

    if G is None:
        dim = len(Domain[0])
        if len(grids) != dim:
            if len(grids) == 1:
                Warning("The number of grids is set as the same for all dimensions.")
                grids = grids * dim
            else:
                raise ValueError("The length of grids should be equal to the dimension of the domain.")
        G = [torch.linspace(l, r, n) for l, r, n in zip(*(Domain + [grids]))]
    dim = len(G)
    if temporal:
        G_rs = [G[0][1:]] + [G[i][1:-1] for i in range(1, dim)]
        G_ic = [G[0][0]] + G[1:]
        collo_rs = meshgrid_to_matrix(G_rs)
        collo_ic = meshgrid_to_matrix(G_ic)
        collo_bc = []
        for i in range(1, dim):
            G_bc = [G[0]]
            for j in range(1, dim):
                if j < i:
                    G_bc.append(G[j][1:-1])
                elif j == i:
                    G_bc.append(G[j][[0,-1]])
                else:
                    G_bc.append(G[j] if corner else G[j][1:-1])
            collo_bc.append(meshgrid_to_matrix(G_bc))
        collo_bc = torch.cat(collo_bc, dim=0)
        return collo_rs, collo_ic, collo_bc
    else:
        G_rs = [G[i][1:-1] for i in range(dim)]
        collo_rs = meshgrid_to_matrix(G_rs)
        collo_bc = []
        for i in range(dim):
            G_bc = []
            for j in range(dim):
                if j < i:
                    G_bc.append(G[j][1:-1])
                elif j == i:
                    G_bc.append(G[j][[0,-1]])
                else:
                    G_bc.append(G[j] if corner else G[j][1:-1])
            collo_bc.append(meshgrid_to_matrix(G_bc))
        collo_bc = torch.cat(collo_bc, dim=0)
        return collo_rs, collo_bc