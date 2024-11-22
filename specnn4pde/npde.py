__all__ = ['gradients', 'Jacobian', 'partial_derivative', 'partial_derivative_vector', 
           'meshgrid_to_matrix', 'gen_collo', 'frequency_analysis',
           'Domain', 
           ]

import numpy as np
import torch
from torch.autograd.functional import jacobian
from typing import Optional, Union
import matplotlib.pyplot as plt

from .spectral import Jacobi_Gauss, Jacobi_Gauss_Lobatto


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

    return torch.stack(Co, dim=-1).reshape(-1, len(inputs))


def gen_collo(Domain = [], grids = [], temporal = False, corner = True, G = None, indexing = 'xy',
              dtype = torch.float32, device: Optional[Union[torch.device, str]] = 'cpu'):
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
        If the collocation points include the corner points of the domain. The default is True.
    G : list of tensor, optional
        The tensors in the list are the collocation points in each dimension.
        If Domain and grids are not provided, G should be provided.
    indexing : str, optional
        The indexing of the meshgrid. The default is 'xy'. The options are 'xy' and 'ij'.
    dtype : torch.dtype, optional
        The data type of the collocation points. The default is torch.float32.
    device : str, optional
        The device of the collocation points. The default is 'cpu'.
    
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
                grids = grids * dim
            else:
                raise ValueError("The length of grids should be equal to the dimension of the domain.")
        G = [torch.linspace(l, r, n, dtype=dtype, device=device) for l, r, n in zip(*(Domain + [grids]))]
    else:
        G = [torch.as_tensor(g, dtype=dtype, device=device) for g in G]
    dim = len(G)
    if temporal:
        G_rs = [G[0][1:]] + [G[i][1:-1] for i in range(1, dim)]
        G_ic = [G[0][0]] + G[1:]
        collo_rs = meshgrid_to_matrix(G_rs, indexing)
        collo_ic = meshgrid_to_matrix(G_ic, indexing)
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
            collo_bc.append(meshgrid_to_matrix(G_bc, indexing))
        collo_bc = torch.cat(collo_bc, dim=0)
        return collo_rs, collo_ic, collo_bc
    else:
        G_rs = [G[i][1:-1] for i in range(dim)]
        collo_rs = meshgrid_to_matrix(G_rs, indexing)
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
            collo_bc.append(meshgrid_to_matrix(G_bc, indexing))
        collo_bc = torch.cat(collo_bc, dim=0)
        return collo_rs, collo_bc
    


def frequency_analysis(domain, func = None, x = None, Func_val = None, grid = 66, device='cpu', dtype=torch.float):
    """
    Compute the main frequency and its amplitude of the function on the domain.

    Args
    -----------
    domain (list): 
            The domain of the function, e.g. [[0,0], [1,1]] for the unit square
    func (callable):
            The function to be analyzed. If None, Func_val should be provided.
    x (tensor):
            The input tensor of the function, If Func_val is provided, x can be ignored.
            If Func_val is None, x will be generated automatically if it is not provided.
    Func_val (tensor):
            The evaluation of the function on the uniform grid of the domain.
            If None, func should be provided.
    grid (int):
            The number of points in each dimension
    device (str):
            The device of the tensors, e.g. 'cpu' or 'cuda'
    dtype (torch.dtype):
            The data type of the tensors

    Returns
    -----------
    main_freq (tensor): 
            The main frequency of the function
    amplitude (tensor):
            The amplitude of the main frequency
    fft (tensor):
            The Fourier transform of the function

    Example
    -----------
    >>> domain = [[0, 1, 2, 3], [3, 3, 4, 4]]
    >>> freq = 2 * torch.pi * torch.tensor([1.33, 2.5, 3., 4.]).view(-1, 1)
    >>> freq = freq.to('cuda')
    >>> func = lambda x: 9 * torch.sin(x @ freq)
    >>> main_freq, amplitude = frequency_analysis(domain, func = func, grid = 100, device = 'cuda')[:2]
    >>> print("Main Frequency: ", main_freq)
    >>> print("Amplitude: ", amplitude)
    Main Frequency:  tensor([1.3333, 2.5000, 3.0000, 4.0000], device='cuda:0')
    Amplitude:  tensor(8.8474, device='cuda:0')
    """

    # check the input
    if func is None and Func_val is None:
        raise ValueError('Either func or Func_val should be provided.')

    dim = len(domain[0])
    dom_tensor = torch.tensor(domain, device=device, dtype=dtype)

    # evaluate the function
    if Func_val is None:
        if x is None:
            x, _ = gen_collo(domain, [grid+2]*dim, corner = False, indexing='ij', device=device, dtype=dtype)
        Func_val = func(x).reshape(*[grid] * dim)

    # compute the Fourier transform and remove the symmetric part
    fft = torch.fft.fftn(Func_val)
    slices = [slice(0, n // 2) for n in fft.shape]    
    fft = torch.abs(fft[slices]) 
    fft = 2 * fft / (grid)**dim

    # find the main frequency and its amplitude
    main_freq = np.unravel_index(torch.argmax(fft).item(), fft.shape)
    amplitude = fft[main_freq]
    main_freq = torch.tensor(main_freq, device=device, dtype=dtype) / (dom_tensor[1] - dom_tensor[0])

    return main_freq, amplitude, fft



class Domain:
    """
    Generate the domain, grids, and collocation points for the PDE problem.

    Remarks
    -----------------
    For the complex region, the following methods should be implemented in the subclass:
        is_inside(x, strict=True):
                Check if the point x is inside the domain.
                Should return a tensor of bool, shape (x.shape[0], 1)
        bndry_collo(grids, type='equidistant', corner=False, indexing='ij'):
                Generate the boundary collocation points
        clip(ax, img):
                Covering or clipping the image.

    Attributes
    -----------------
    list (list): 
            The domain of the problem, e.g. [[0,0], [1,1]] for the unit square
    tensor (tensor):
            The domain tensor, shape (2, dim)
    complex (bool):
            Whether the domain is a complex region
    dim (int):
            The dimension of the problem
    width (tensor):
            The width of the domain, shape (dim,)
    center (tensor):
            The center of the domain, shape (dim,)
    area (tensor or float):
            The area of the domain, for complex region, it should be given manually
    dtype (torch.dtype):
            The data type of the tensors
    device (str):
            The device of the tensors, e.g. 'cpu' or 'cuda

    Methods
    -----------------
    is_inside(x, strict=True):
            Check if the point x is inside the domain.
    clip(ax, img):
            Covering or clipping the image.
    gen_G(grids, type='equidistant', bndry_skip=1e-3):
            Generate the grids for each dimension
    int_collo(grids, type='equidistant', bndry_skip=1e-3, indexing='ij'):
            Generate the interior collocation points
    bndry_collo(grids, type='equidistant', corner=False, indexing='ij'):
            Generate the boundary collocation points
    plot_grid(grids, bndry_skip=0, indexing='ij'):
            Generate the grids for plotting
    show_collo(collo, s=1, figsize=None):
            Show the collocation points
    """

    def __init__(self, domain, complex=False, area=None, dtype=torch.float32, device='cpu'):
        """
        Initialize the domain class.

        Args
        -----------------
        domain (list): 
                The domain of the problem, e.g. [[0,0], [1,1]] for the unit square
                For complex region, it should be the clousure of the domain
        complex (bool):
                Whether the domain is a complex region
        area (tensor or float):
            The area of the domain, for complex region, it should be given manually
        dtype (torch.dtype):
                The data type of the tensors
        device (str):
                The device of the tensors, e.g. 'cpu' or 'cuda'        
        """
        
        self.list = domain
        self.tensor = torch.as_tensor(domain, dtype=dtype, device=device)
        self.complex = complex

        self.dim = len(domain[0])
        self.width = self.tensor[1] - self.tensor[0]
        self.center = self.tensor.mean(dim=0)

        if complex:
            if area is None:
                raise ValueError('The area should be provided for the complex region.')
            self.area = area
        else:
            self.area = torch.prod(self.width)

        self.dtype = dtype
        self.device = device
    
    def __repr__(self):
        return f'Domain({self.list})'
    
    def __check_grids__(self, grids):
        """
        Check the grids for the domain.
        """
        
        if len(grids) != self.dim:
            if len(grids) == 1:
                grids = grids * self.dim
            else:
                raise ValueError("The length of grids should be equal to the dimension of the domain.")
        
        return grids
    
    def is_inside(self, x, strict=True):
        """
        Check if the point x is inside the domain for complex region.
        Need to be implemented in the subclass for different regions.

        Args
        -----------------
        x (tensor): 
                The point to be checked, shape (N_pts, dim)
        strict (bool):
                Whether to exclude the boundary points.

        Returns
        -----------------
        inside (tensor of bool): 
                Whether the point is inside the domain, shape (N_pts, 1)
        """

        return torch.ones(x.shape[0], 1, dtype=torch.bool, device=self.device)
    
    def clip(self, ax, img):
        """
        Covering or clipping the image.
        To be implemented in the subclass.

        Args
        ----------------
        ax (Axes):
                The axes to plot the image
        img (AxesImage):
                The image to be clipped

        Sample codes
        ----------------
        >>> # 1. Covering
        >>> # Fill the area enclosed by the circle with white color
        >>> t = np.linspace(0, 2 * np.pi, 100)
        >>> ax.fill(np.cos(t), np.sin(t), 'w')

        >>> # 2. Clipping
        >>> # Clip the image with a polygon
        >>> verts = np.array([[0.,0],[0,-1],[1,-1],[1,1],[-1,1],[-1,0]])
        >>> polygon = Polygon(verts, closed=True, color='none', alpha=0.5)
        >>> ax.add_patch(polygon)
        >>> img.set_clip_path(polygon)
        """
        pass

    def gen_G(self, grids, type = 'equidistant', bndry_skip = 1e-3):
        """
        Generate the grids for each dimension.

        Args
        -----------------
        see method `int_collo` for the arguments

        Returns
        -----------------
        G (list): 
                The grids for each dimension
        """

        if type in {'uniform', 'equidistant'}:
            # equidistant sampling
            G = [torch.linspace(l+bndry_skip, r-bndry_skip, n, dtype=self.dtype, device=self.device) 
                 for l, r, n in zip(*(self.list + [grids]))]
            
        elif type == 'gauss':
            # Legendre Gauss points
            G = [(torch.as_tensor(Jacobi_Gauss(0, 0, n)[1], device=self.device, dtype=self.dtype) + 1) / 2 * (r - l) + l
                 for l, r, n in zip(*(self.list + [grids]))]
            
        elif type == 'gauss_lobatto':
            # Legendre Gauss Lobatto points
            G = [(torch.as_tensor(Jacobi_Gauss_Lobatto(0, 0, n-1)[1], device=self.device, dtype=self.dtype) + 1) / 2 * (r - l) + l
                 for l, r, n in zip(*(self.list + [grids]))]
            
        return G

    def int_collo(self, grids, type = 'equidistant', bndry_skip = 1e-3, indexing = 'ij'):
        """
        Generate the interior collocation points.

        Args
        -----------------
        grids (list): 
                The number of grids for each dimension
        type (str):
                The type of the grids, including the following options:
                'uniform' or 'equidistant': equidistant sampling, boundry points included or excluded depending on `bndry_skip`.
                'gauss': Legendre Gauss points, excluding the boundary points.
                'gauss_lobatto': Legendre Gauss Lobatto points, including the boundary points.
                'random': uniform distribution sampling, excluding the boundary points.
        bndry_skip (float):
                The boundary skip for the grids, only valid when `type` is 'uniform' or 'equidistant'.
                If bndry_skip = 0, the boundary points are included. Otherwise, the boundary points are excluded.
        indexing (str):
                The indexing of the grids, 'ij' or 'xy'

        Returns
        -----------------
        collo (tensor): 
                The interior collocation points, shape (N_pts, dim)
                For complex regions, points outside the domain are filtered out, so the number of points returned is uncertain, except for the `random` type.
        """

        grids = self.__check_grids__(grids)

        if type in {'uniform', 'equidistant', 'gauss', 'gauss_lobatto'}:
            G = self.gen_G(grids, type, bndry_skip)

        elif type == 'random':
            # uniform distribution sampling
            # collo = torch.rand([torch.prod(torch.tensor(grids)), self.dim], dtype=self.dtype, device=self.device)
            # collo = collo * self.width + self.tensor[0]
            collo = self.rejection_sampling(torch.prod(torch.tensor(grids)), self.is_inside, 1, random_candidates=True)

        else:
            raise ValueError('Invalid collocation type!')
        
        if type != 'random':
            Co = torch.meshgrid(*G, indexing=indexing)
            collo = torch.stack(Co, dim=-1).reshape(-1, self.dim)

            # filter the points outside the domain
            if self.complex:
                mask = self.is_inside(collo)
                collo = collo[mask.squeeze()]

        return collo
    
    def bndry_collo(self, grids, type='equidistant', corner=False, indexing = 'ij'):
        """
        Generate the boundary collocation points.

        Args
        -----------------
        corner (bool): 
                Whether to include the corner points
        see method `int_collo` for the other arguments

        Returns
        -----------------
        collo_bc (tensor): 
                The boundary collocation points, shape (N_pts, dim)
        """

        grids = self.__check_grids__(grids)

        if type in {'uniform', 'equidistant', 'gauss_lobatto'}:
            G = self.gen_G(grids, type=type, bndry_skip=0)
            
            collo_bc = []
            for i in range(self.dim):
                G_bc = []
                for j in range(self.dim):
                    if j < i:
                        G_bc.append(G[j][1:-1])
                    elif j == i:
                        G_bc.append(G[j][[0,-1]])
                    else:
                        G_bc.append(G[j] if corner else G[j][1:-1])
                collo_bc.append(meshgrid_to_matrix(G_bc, indexing))
            collo_bc = torch.cat(collo_bc, dim=0)
        
        elif type == 'random':
            # TODO: random boundary collocation is under development
            raise ValueError('Random boundary collocation is under development!')

        else:
            raise ValueError('Invalid collocation type!')

        return collo_bc
    
    def plot_grid(self, grids, bndry_skip = 0, indexing = 'ij'):
        """
        Generate the equidistant grids for plotting. Only support 1D, 2D and 3D.

        Args
        -----------------
        see method `int_collo` for other arguments

        Returns
        -----------------
        grids (list): 
                The grids for plotting
        Co (tuple of tensor):
                The meshgrid for plotting, shape (dim, *grids)
        """

        grids = self.__check_grids__(grids)

        G = [torch.linspace(l+bndry_skip, r-bndry_skip, n, dtype=self.dtype, device=self.device) 
                 for l, r, n in zip(*(self.list + [grids]))]

        Co = torch.meshgrid(*G, indexing=indexing)
        collo = torch.stack(Co, dim=-1).reshape(-1, self.dim)

        return collo, Co
    
    def rejection_sampling(self, num_samples, target_dist, M, min_batch=100, max_batch=10000, random_candidates=False, grids=None, bndry=False):
        """
        Perform rejection sampling to draw samples from a target distribution.
        
        Parameters
        -----------------
        - num_samples (int): The number of samples to generate.
        - target_dist (callable): The target distribution function p(x). It should take a tensor of 
                                  shape (N, dim) as input and return a tensor of shape (N, 1).
        - domain (list): The rectangular domain from which to sample, as [(x_min, x_max), (y_min, y_max)].
        - M (float): An upper bound on the target distribution within the domain.
        - min_batch (int): The minimum batch size for sampling.
        - max_batch (int): The maximum batch size for sampling.
        - random_candidates (bool): Whether to sample the candidate points randomly.
                                    If True, the candidate points will be generated from the uniform distribution strictly 
                                    within the interior of the domain, excluding the boundary. 
                                    If False, see the following two arguments.
        - grids (list): Only valid when `random_candidates` is False.
                        The number of grids for each dimension, the candidate points will be generated uniformly from the grids. 
                        If None, the grids will be set as max(1000, 2*\sqrt[dim]{num_samples}) for each dimension.
        - bndry (bool): Only valid when `random_candidates` is False.
                        When True, the candidate points will not be generated on the boundary.

        Returns
        -----------------
        - samples: Array of samples generated from the target distribution.
        """

        if grids is None:
            grids = [max(1000, 2 * int(num_samples ** (1 / self.dim)))] * self.dim
        grids = torch.tensor(grids, device=self.device)
        
        samples = torch.rand(0, self.dim, device=self.device, dtype=self.dtype)
        
        while samples.shape[0] < num_samples:
            batch_size = min(max(2 * (num_samples - samples.shape[0]), min_batch), max_batch)

            # Step 1: Sample candidate points from the uniform distribution over the domain
            if random_candidates:
                x_candidate = torch.rand(batch_size, self.dim, device=self.device) * self.width + self.tensor[0]
            else:
                if bndry:
                    x_candidate = torch.stack([torch.randint(0, g+1, (batch_size,), device=self.device) for g in grids], dim=1)
                else:
                    x_candidate = torch.stack([torch.randint(1, g, (num_samples,), device=self.device) for g in grids], dim=1)

                x_candidate = x_candidate * self.width / grids + self.tensor[0]

            # Step 2: Evaluate the target distribution at the candidate point
            p_xy = abs(target_dist(x_candidate).to(self.dtype))
            # filter the points outside the domain
            if self.complex:
                mask = self.is_inside(x_candidate)
                p_xy = p_xy * mask
            
            # Step 3: Generate a uniform random number u ~ U(0, M)
            u = torch.empty_like(p_xy).uniform_(0, M)
            
            # Step 4: Accept or reject the candidate sample
            mask = (u < p_xy).squeeze()
            # unique the samples, note that the returned samples are sorted
            samples = torch.unique(torch.cat([samples, x_candidate[mask]], dim=0), dim=0)

        random_indices = torch.randperm(samples.size(0))[:num_samples]
        return samples[random_indices]

    
    def show_collo(self, collo, s=1, figsize=None):
        """
        Show the collocation points.

        Args
        -----------------
        collo (tensor): 
                The collocation points, shape (N_pts, dim)
        s (int):
                The size of the points
        figsize (tuple):
                The size of the figure
        """

        if figsize is None:
            figsize = (5, 1) if self.dim == 1 else (5, 5)

        if self.dim == 1:
            fig = plt.figure(figsize=figsize)
            plt.scatter(collo, torch.zeros_like(collo), s=s)
        elif self.dim == 2:
            fig = plt.figure(figsize=figsize)
            plt.scatter(collo[:, 0], collo[:, 1], s=s)
        elif self.dim == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(collo[:, 0], collo[:, 1], collo[:, 2], s=s)
        else:
            raise ValueError('Only support 1D, 2D and 3D plotting!')
        plt.show()



class Domain_circle:
    """
    Generate the domain, grids, and collocation points for circle domain.

    Remarks
    -----------------
    For the complex region, the following methods should be implemented in the subclass:
        is_inside(x, strict=True):
                Check if the point x is inside the domain.
                Should return a tensor of bool, shape (x.shape[0], 1)
        bndry_collo(grids, type='equidistant', corner=False, indexing='ij'):
                Generate the boundary collocation points
        clip(ax, img):
                Covering or clipping the image.

    Attributes
    -----------------
    list (list): 
            The domain of the problem in polar coordinates,
            e.g. [[0,0], [1,2*pi]] for the unit circle
    tensor (tensor):
            The domain tensor, shape (2, dim)
    complex (bool):
            Whether the domain is a complex region
    dim (int):
            The dimension of the problem
    center (tensor):
            The center of the domain, shape (dim,)
    radius (float):
            The radius of the circle
    area (tensor or float):
            The area of the domain, for complex region, it should be given manually
    dtype (torch.dtype):
            The data type of the tensors
    device (str):
            The device of the tensors, e.g. 'cpu' or 'cuda

    Methods
    -----------------
    is_inside(x, strict=True):
            Check if the point x is inside the domain.
    clip(ax, img):
            Covering or clipping the image.
    gen_G(grids, type='equidistant', bndry_skip=1e-3):
            Generate the grids for each dimension
    int_collo(grids, type='equidistant', bndry_skip=1e-3, indexing='ij'):
            Generate the interior collocation points
    bndry_collo(grids, type='equidistant', corner=False, indexing='ij'):
            Generate the boundary collocation points
    plot_grid(grids, bndry_skip=0, indexing='ij'):
            Generate the grids for plotting
    show_collo(collo, s=1, figsize=None):
            Show the collocation points
    """

    def __init__(self, domain, complex=False, area=None, center=(0,0), dtype=torch.float32, device='cpu'):
        """
        Initialize the circle domain class.

        Args
        -----------------
        domain (list): 
                The domain of the problem, e.g. [[0,0], [1, 2*pi]] for the unit circle
        complex (bool):
                Whether the domain is a complex region
        area (float or float):
                The area of the domain, for complex region, it should be given manually
        center (tuple):
                The center of the circle
        dtype (torch.dtype):
                The data type of the tensors
        device (str):
                The device of the tensors, e.g. 'cpu' or 'cuda'        
        """
        
        self.list = domain
        self.tensor = torch.as_tensor(domain, dtype=dtype, device=device)
        self.complex = complex

        self.dim = len(domain[0])
        self.center = torch.tensor(center, dtype=dtype, device=device)
        self.radius = domain[1][0]

        if complex:
            if area is None:
                raise ValueError('The area should be provided for the complex region.')
            self.area = area
        else:
            self.area = self.area = np.pi * self.radius ** 2

        self.dtype = dtype
        self.device = device
    
    def __repr__(self):
        return f'Domain({self.list})'
    
    def __check_grids__(self, grids):
        """
        Check the grids for the domain.
        """
        
        if len(grids) != self.dim:
            if len(grids) == 1:
                grids = grids * self.dim
            else:
                raise ValueError("The length of grids should be equal to the dimension of the domain.")
        
        return grids
    
    def is_inside(self, x, strict=True):
        """
        Check if the input samples are within the domain.

        Args
        -----------------
        x (tensor): Input samples, shape (N_pts, dim)
        strict (bool): Whether to exclude the boundary

        Returns
        -----------------
        result (tensor): Shape (N_pts, 1), values are 1 if within the domain, otherwise 0
        """

        r = x.norm(dim=1, keepdim=True)
        return (r < self.radius) if strict else (r <= self.radius)
    
    def clip(self, ax, img):
        """
        Covering or clipping the image.
        To be implemented in the subclass.

        Args
        ----------------
        ax (Axes):
                The axes to plot the image
        img (AxesImage):
                The image to be clipped

        Sample codes
        ----------------
        >>> # 1. Covering
        >>> # Fill the area enclosed by the circle with white color
        >>> t = np.linspace(0, 2 * np.pi, 100)
        >>> ax.fill(np.cos(t), np.sin(t), 'w')

        >>> # 2. Clipping
        >>> # Clip the image with a polygon
        >>> verts = np.array([[0.,0],[0,-1],[1,-1],[1,1],[-1,1],[-1,0]])
        >>> polygon = Polygon(verts, closed=True, color='none', alpha=0.5)
        >>> ax.add_patch(polygon)
        >>> img.set_clip_path(polygon)
        """
        pass

    def gen_G(self, grids, type = 'equidistant', bndry_skip = 1e-3):
        """
        Generate the grids for each dimension.
        In radial direction, the point 0 are excluded, even if `bndry_skip` is 0.

        Args
        -----------------
        see method `int_collo` for the arguments

        Returns
        -----------------
        G (list): 
                The grids for each dimension
        """

        if type in {'uniform', 'equidistant'}:
            # equidistant sampling
            if bndry_skip == 0:
                G = [torch.linspace(l+bndry_skip, r-bndry_skip, n+1, dtype=self.dtype, device=self.device) 
                 for l, r, n in zip(*(self.list + [grids]))]
                G[0] = G[0][1:]     # exclude the origin point (0,0)
                G[1] = G[1][:-1]    # exclude the boundary point 2*pi, which is the same as 0
            else:
                G = [torch.linspace(l+bndry_skip, r-bndry_skip, n, dtype=self.dtype, device=self.device) 
                 for l, r, n in zip(*(self.list + [grids]))]
            
        elif type == 'gauss':
            # Legendre Gauss points
            G = [(torch.as_tensor(Jacobi_Gauss(0, 0, n)[1], device=self.device, dtype=self.dtype) + 1) / 2 * (r - l) + l
                 for l, r, n in zip(*(self.list + [grids]))]
            
        elif type == 'gauss_lobatto':
            # Legendre Gauss Lobatto points
            G = [(torch.as_tensor(Jacobi_Gauss_Lobatto(0, 0, n)[1], device=self.device, dtype=self.dtype) + 1) / 2 * (r - l) + l
                 for l, r, n in zip(*(self.list + [grids]))]
            G[0] = G[0][1:]     # exclude the origin point (0,0)
            G[1] = G[1][:-1]    # exclude the boundary point 2*pi, which is the same as 0
            
        return G

    def int_collo(self, grids, type = 'equidistant', bndry_skip = 1e-3, indexing = 'ij'):
        """
        Generate the interior collocation points.

        Args
        -----------------
        grids (list): 
                The number of grids for each dimension
        type (str):
                The type of the grids, including the following options:
                'uniform' or 'equidistant': equidistant sampling, boundry points included or excluded depending on `bndry_skip`.
                                            note that the origin point (0,0) is excluded anyway
                'gauss': Legendre Gauss points, excluding the boundary points.
                'gauss_lobatto': Legendre Gauss Lobatto points, including the boundary points.
                !!!!!!!!!! the above three are distributions in (r, theta) coordinates
                'random': uniform distribution sampling, excluding the boundary points.
        bndry_skip (float):
                The boundary skip for the grids, only valid when `type` is 'uniform' or 'equidistant'.
                If bndry_skip = 0, the boundary points are included. Otherwise, the boundary points are excluded.
        indexing (str):
                The indexing of the grids, 'ij' or 'xy'

        Returns
        -----------------
        collo (tensor): 
                The interior collocation points, shape (N_pts, dim)
                For complex regions, points outside the domain are filtered out, so the number of points returned is uncertain, except for the `random` type.
        """

        grids = self.__check_grids__(grids)

        if type in {'uniform', 'equidistant', 'gauss', 'gauss_lobatto'}:
            G = self.gen_G(grids, type, bndry_skip)

        elif type == 'random':
            # uniform distribution sampling
            num_samples = torch.prod(torch.tensor(grids))
            collo = self.rejection_sampling(num_samples, self.is_inside, 1, random_candidates=True)                

        else:
            raise ValueError('Invalid collocation type!')
        
        if type != 'random':
            Co = torch.meshgrid(*G, indexing=indexing)
            # transform the polar coordinates to Cartesian coordinates
            Co_Cartesian = (Co[0] * torch.cos(Co[1]), Co[0] * torch.sin(Co[1]))
            collo = torch.stack(Co_Cartesian, dim=-1).reshape(-1, self.dim)

            # filter the points outside the domain
            if self.complex:
                mask = self.is_inside(collo)
                collo = collo[mask.squeeze()]

        return collo
    
    
    def bndry_collo(self, grids, type='equidistant', **kwargs):
        """
        Generate the boundary collocation points.

        Args
        -----------------
        grids (int):
                The number of grids
        see method `int_collo` for the other arguments

        Returns
        -----------------
        collo_bc (tensor): 
                The boundary collocation points, shape (N_pts, dim)
        """

        if type in {'uniform', 'equidistant'}:
            theta = torch.linspace(self.list[0][1], self.list[1][1], grids + 1, dtype=self.dtype, device=self.device)[:-1]
        
        elif type == 'random':
            # TODO: random boundary collocation is under development
            raise ValueError('Random boundary collocation is under development!')

        else:
            raise ValueError('Invalid collocation type!')
        
        collo_bc = self.radius * torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
        return collo_bc
    
    def plot_grid(self, grids, bndry_skip = 0, indexing = 'ij'):
        """
        Generate the equidistant grids for plotting. Only support 1D, 2D and 3D.

        Args
        -----------------
        see method `int_collo` for the arguments

        Returns
        -----------------
        grids (list): 
                The grids for plotting
        Co_Cartesian (tensor):
                The meshgrid points in Cartesian coordinates, shape (dim, *grids)
        Co (tuple of tensor):
                The meshgrid points in polar coordinates, shape (dim, *grids)
        """

        grids = self.__check_grids__(grids)

        G = [torch.linspace(l+bndry_skip, r-bndry_skip, n, dtype=self.dtype, device=self.device) 
                 for l, r, n in zip(*(self.list + [grids]))]

        Co = torch.meshgrid(*G, indexing=indexing)
        # transform the polar coordinates to Cartesian coordinates
        Co_Cartesian = (Co[0] * torch.cos(Co[1]), Co[0] * torch.sin(Co[1]))
        collo = torch.stack(Co_Cartesian, dim=-1).reshape(-1, self.dim)

        return collo, Co_Cartesian, Co
    

    
    def rejection_sampling(self, num_samples, target_dist, M, min_batch=100, max_batch=10000, random_candidates=False, grids=None, bndry=False):
        """
        Perform rejection sampling to draw samples from a target distribution.
        
        Parameters
        -----------------
        - num_samples (int): The number of samples to generate.
        - target_dist (callable): The target distribution function p(x). It should take a tensor of 
                                  shape (N, dim) as input and return a tensor of shape (N,).
        - domain (list): The rectangular domain from which to sample, as [(x_min, x_max), (y_min, y_max)].
        - M (float): An upper bound on the target distribution within the domain.
        - min_batch (int): The minimum batch size for sampling.
        - max_batch (int): The maximum batch size for sampling.
        - random_candidates (bool): Whether to sample the candidate points randomly.
                                    If True, the candidate points will be generated from the uniform distribution strictly 
                                    within the interior of the domain, excluding the boundary. 
                                    If False, see the following two arguments.
        - grids (list): Only valid when `random_candidates` is False.
                        The number of grids for each dimension, the candidate points will be generated uniformly from the grids. 
                        If None, the grids will be set as max(1000, 2*\sqrt[dim]{num_samples}) for each dimension.
        - bndry (bool): Only valid when `random_candidates` is False.
                        When True, the candidate points will not be generated on the boundary.

        Returns
        -----------------
        - samples: Array of samples generated from the target distribution.
        """

        if grids is None:
            grids = [max(1000, 2 * int(num_samples ** (1 / self.dim)))] * self.dim
        grids = torch.tensor(grids, device=self.device)
        
        samples = torch.rand(0, self.dim, device=self.device, dtype=self.dtype)
        
        while samples.shape[0] < num_samples:
            batch_size = min(max(2 * (num_samples - samples.shape[0]), min_batch), max_batch)

            # Step 1: Sample candidate points from the uniform distribution over the domain
            if random_candidates:
                x_candidate = (torch.rand(batch_size, self.dim, device=self.device, dtype=self.dtype) * 2 - 1) * self.radius
            else:
                if bndry:
                    x_candidate = torch.stack([torch.randint(0, g+1, (batch_size,), device=self.device, dtype=self.dtype) for g in grids], dim=1)
                else:
                    x_candidate = torch.stack([torch.randint(1, g, (num_samples,), device=self.device, dtype=self.dtype) for g in grids], dim=1)

                x_candidate = (x_candidate / grids * 2 - 1) * self.radius

            # Step 2: Evaluate the target distribution at the candidate point
            # and check if the candidate points are within the domain
            p_xy = abs(target_dist(x_candidate).to(self.dtype))
            # filter the points outside the domain
            if self.complex:
                mask = self.is_inside(x_candidate)
                p_xy = p_xy * mask

            # Step 3: Generate a uniform random number u ~ U(0, M)
            u = torch.empty_like(p_xy).uniform_(0, M)
            
            # Step 4: Accept or reject the candidate sample
            mask = (u < p_xy).squeeze()
            # unique the samples, note that the returned samples are sorted
            samples = torch.unique(torch.cat([samples, x_candidate[mask]], dim=0), dim=0)

        random_indices = torch.randperm(samples.size(0))[:num_samples]
        return samples[random_indices]

    
    def show_collo(self, collo, s=1, figsize=None):
        """
        Show the collocation points.

        Args
        -----------------
        collo (tensor): 
                The collocation points, shape (N_pts, dim)
        s (int):
                The size of the points
        figsize (tuple):
                The size of the figure
        """

        if figsize is None:
            figsize = (5, 1) if self.dim == 1 else (5, 5)

        if self.dim == 2:
            fig = plt.figure(figsize=figsize)
            plt.scatter(collo[:, 0], collo[:, 1], s=s)
            plt.xlim(-self.radius*1.05, self.radius*1.05)
            plt.ylim(-self.radius*1.05, self.radius*1.05)
        # elif self.dim == 3:
        #     fig = plt.figure(figsize=figsize)
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter(collo[:, 0], collo[:, 1], collo[:, 2], s=s)
        else:
            raise ValueError('Only support 2D plotting!')
        plt.show()
