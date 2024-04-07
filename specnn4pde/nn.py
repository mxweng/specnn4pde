__all__ = ['setup_seed', 'seed_worker',
              'RFMNet',
           ]

import random, os
import torch
from torch import nn
import numpy as np


def setup_seed(seed, Pytorch=True, Random=False, Numpy=False, Hash=False):
    """
    Set the seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed number.
    Pytorch : bool, optional
        If set the seed for Pytorch. The default is True.
    Random : bool, optional
        If set the seed for module random. The default is False.
    Numpy : bool, optional
        If set the seed for module numpy. The default is False.
    Hash : bool, optional
        If set the seed for hash. The default is False. If your code depends on the iteration order of dictionaries or collections, you may need to set this to True.
    """

    if Pytorch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if Random:
        random.seed(seed)
    if Numpy:
        np.random.seed(seed)
    if Hash:
        os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    """
    Fix seed for DataLoaders for reproducibility. 

    References:
    1. https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    2. [set_seed 会破坏随机性，官方 worker_init_fn 无法解决] https://zhuanlan.zhihu.com/p/618639620

    Example:
    ----------
    # set_seed will break randomness in DataLoader, see references 2
    >>> g = torch.Generator()
    >>> rank = torch.distributed.get_rank()
    >>> g.manual_seed(3407 + rank)  # given generator based on process rank to avoid same random number 
    >>>
    >>> DataLoader(
    >>>     train_dataset,
    >>>     batch_size=batch_size,
    >>>     shuffle=True,
    >>>     num_workers=num_workers,
    >>>     generator = g,
    >>>     worker_init_fn=seed_worker,     # redundant if torch>=1.9
    >>> )
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class RFMNet(nn.Module):
    """
    The random feature method (RFM) neural network for solving PDEs.
    
    Parameters
    ----------
    X_n : (N, d) tensor
        The coordinates of the centers of the local regions. N is the number of local regions.
    R_n : (N, d) tensor
        The radii of the local regions.
    d : int, optional
        The dimension of the input. The default is 1.
    seed : int, optional
        The seed number. The default is 0.
    J_n : int, optional
        The number of local/global basis (random features) around each local/global center 
        points. The default is 50.
    R : float, optional
        The range of the initial values of the weights. The default is 1.
    glb_num : int, optional
        The number of global center points of global basis (random features). The default is 0.
    trunc_func : str, optional
        The truncation function. The default is 'psi_b'. 'psi_a' and 'psi_b' are available. 
        See references [1;(2.6)-(2.7)] for definitions.
    active_func : function, optional
        The activation function. The default is torch.tanh.
    device : str, optional
        The device to use. The default is 'cpu'.
    dtype : torch.dtype, optional
        The data type. The default is torch.float64. 
        torch.float16, torch.float32, and torch.float64 are available.

    Attributes
    ----------
    M_p : int
        The number of center points including local and global center points.
    X_n, R_n, J_n, d, R, glb_num, trunc_func, active_func, device, dtype : see Parameters.
    K : (M_p, J_n, d) tensor
        The weights of the basis (random features), which are randomly initialized and fixed.
    b : (M_p, J_n) tensor
        The biases of the basis (random features), which are randomly initialized and fixed.
    U : (M_p, J_n) tensor
        The weights of the output layer to be trained.

    Methods
    -------
    __init__(self, X_n, R_n, d, seed, J_n, R, glb_num, trunc_func, active_func, device, dtype)
        Initialize the RFM neural network. See Parameters.
    forward(X) -> (N, 1) tensor
        The forward pass of the RFM neural network.
    hidden_layer_outputs(X) -> (N, M_p * J_n) tensor
        Returns the outputs of all basis (random features) in the hidden layer.
    normalize_coordinates(X) -> (N, M_p, d) tensor
        Normalize the coordinates of the input w.r.t the center points. X -> (X - X_n) / R_n.
    psi_a(X_tilde) -> (N, M_p) tensor
        The truncation function. See references [1;(2.6)] for definition.
    psi_b(X_tilde) -> (N, M_p) tensor
        The truncation function. See references [1;(2.7)] for definition.

    References
    ----------
    1. 2022, Jingrun Chen, Xurong Chi, Weinan E, Zhouwang Yang, JML.
        https://global-sci.org/intro/article_detail/jml/21029.html
        Bridging Traditional and Machine Learning-Based Algorithms for Solving PDEs: The Random Feature Method.
    """
    def __init__(self, X_n, R_n, d = 1, seed = 0, J_n = 50, R = 1, glb_num = 0, trunc_func = 'psi_b', active_func = torch.tanh, device='cpu', dtype=torch.float64):
        super(RFMNet, self).__init__()
        self.M_p = len(X_n)
        self.X_n = X_n.to(device)
        self.R_n = R_n.to(device)
        self.J_n = J_n
        self.d = d
        self.R = R
        self.glb_num = glb_num
        self.trunc_func = self.psi_a if trunc_func == 'psi_a' else self.psi_b
        self.active_func = active_func
        self.device = device
        self.dtype = dtype
        setup_seed(seed)
        # fixed parameters
        if R == None:
            self.K = torch.zeros(self.M_p, self.J_n, d, device=device)
            self.b = torch.zeros(self.M_p, self.J_n, device=device)
            nn.init.xavier_uniform_(self.K)
            nn.init.xavier_uniform_(self.b)
        else:
            self.K = torch.rand(self.M_p, self.J_n, d, device=device) * 2 * R - R
            self.b = torch.rand(self.M_p, self.J_n, device=device) * 2 * R - R
        # parameters to be trained
        self.U = nn.Parameter(torch.rand(self.M_p, self.J_n, device=device))

        if dtype == torch.float16:
            self = self.half()
            self.K = self.K.half()
            self.b = self.b.half()
        elif dtype == torch.float32:
            self = self.float()
            self.K = self.K.float()
            self.b = self.b.float()
        elif dtype == torch.float64:
            self = self.double()
            self.K = self.K.double()
            self.b = self.b.double()

    def forward(self, X):
        X = X.to(self.device, dtype=self.dtype)
        X_tilde = self.normalize_coordinates(X)
        u_M = torch.einsum('njd,nsd->snj', self.K, X_tilde) + self.b
        if self.glb_num:
            Psi1 = self.trunc_func(X_tilde[:-self.glb_num])
            Psi = torch.cat([Psi1, torch.ones(self.glb_num, Psi1.shape[1], device=self.device, dtype=self.dtype)], dim=0)
        else:
            Psi = self.trunc_func(X_tilde)
        u_M = Psi * torch.einsum('nj,snj->ns', self.U, self.active_func(u_M))
        return u_M.sum(dim=0).reshape(-1, 1)        
    
    def hidden_layer_outputs(self, X):
        X = X.to(self.device, dtype=self.dtype)
        X_tilde = self.normalize_coordinates(X)
        H = torch.einsum('njd,nsd->snj', self.K, X_tilde) + self.b
        if self.glb_num:
            Psi1 = self.trunc_func(X_tilde[:-self.glb_num])
            Psi = torch.cat([Psi1, torch.ones(self.glb_num, Psi1.shape[1], device=self.device, dtype=self.dtype)], dim=0)
        else:
            Psi = self.trunc_func(X_tilde)
        H = torch.einsum('ns,snj->snj', Psi, self.active_func(H))
        return H.reshape(len(H), -1)
    
    def normalize_coordinates(self, X):
        X_n = self.X_n.unsqueeze(1).repeat(1, X.shape[0], 1)
        R_n = self.R_n.unsqueeze(1).repeat(1, X.shape[0], 1)
        return (X - X_n) / R_n

    # truncation function
    def psi_a(self, X_tilde):
        condition1 = (X_tilde >= -1) & (X_tilde < 1)
        if self.dtype == torch.float64:
            return condition1.double().prod(dim=2)
        else:
            return condition1.float().prod(dim=2)

    def psi_b(self, X_tilde):
        condition1 = (X_tilde >= -5/4) & (X_tilde < -3/4)
        condition2 = (X_tilde >= -3/4) & (X_tilde < 3/4)
        condition3 = (X_tilde >= 3/4) & (X_tilde < 5/4)
        if self.dtype == torch.float64:
            res = condition1.double() * (1 + torch.sin(2 * torch.pi * X_tilde)) / 2 \
            + condition2.double() \
            + condition3.double() * (1 - torch.sin(2 * torch.pi * X_tilde)) / 2
        else:
            res = condition1.float() * (1 + torch.sin(2 * torch.pi * X_tilde)) / 2 \
            + condition2.float() \
            + condition3.float() * (1 - torch.sin(2 * torch.pi * X_tilde)) / 2
        return res.prod(dim=2)