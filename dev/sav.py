__all__ = ['RSAV']
# To be add
# 1. vanilla SAV
# 2. Restart SAV: \hat{r}_n correction
# 3. Relaxed SAV (RSAV)
# 4. adaptive RASV (A-RSAV)
# 5. Element-wise RSAV (E-RSAV)

"""
sav.py

This module provides optimizers based on the Scalar Auxilary Variable (SAV) method.

Reference:
1. Liu, X.; Shen, J.; Zhang, X. 
    An Efficient and Robust SAV Based Algorithm for Discrete Gradient Systems Arising from Optimizations. 
    arXiv May 10, 2023. http://arxiv.org/abs/2301.02942
2. Zhang, S.; Zhang, J.; Shen, J.; Lin, G. 
    An Element-Wise RSAV Algorithm for Unconstrained Optimization Problems. 
    arXiv September 7, 2023. http://arxiv.org/abs/2309.04013
3. Ma, Z.; Mao, Z.; Shen, J. 
    Efficient and Stable SAV-Based Methods for Gradient Flows Arising from Deep Learning. 
    JCP 2024, 505, 112911. https://doi.org/10.1016/j.jcp.2024.112911
"""

import torch
from torch.optim import Optimizer
import numpy as np
from autograd import grad, hessian

class RSAV(Optimizer):
    r"""Implements RSAV algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    For further details regarding the algorithm we refer to
    Liu, X.; Shen, J.; Zhang, X. 
    An Efficient and Robust SAV Based Algorithm for Discrete Gradient Systems Arising from Optimizations. 
    arXiv May 10, 2023. http://arxiv.org/abs/2301.02942

    Args:


    """

    def __init__(self,
                 params, 
                 init_loss, 
                 lr, 
                 lr_max=100,
                 lr_min=0.01, 
                 C=0, 
                 opL='trival', 
                 eta=0.99, 
                 rho=1.1, 
                 gamma=0.9, 
                 adaptive=True,
                 tol=1e-6):
        
        modified_energy = torch.sqrt(init_loss + C)
        defaults = dict(r=modified_energy, ME=modified_energy, lr=lr, lr_max=lr_max, lr_min=lr_min, C=C, opL=opL, eta=eta, rho=rho, gamma=gamma, adaptive=adaptive, tol=tol)
        super(RSAV, self).__init__(params, defaults)
        self.loss = init_loss

    def step(self, closure=None, k=0):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            # adaptive step size according to the modified energy
            indicator = group['r'] / group['ME']
            # indicator = 1 - abs(indicator - 1)
            if not group['adaptive']:
                pass
            elif indicator < group['gamma'] and group['lr'] > group['lr_min']:
                group['lr'] = max(indicator * group['lr'], group['lr_min'])
            else:
                group['lr'] = group['rho'] * group['lr']
           
            # set the operator L
            if group['opL'] == "trival":
                # flatten the group parameters and gradients
                self.loss.backward()
                r_tilde = sum([p.grad.norm()**2 for p in group['params']]) / (group['ME']**2)
                r_tilde = group['r'] / (1 + group['lr'] * r_tilde / 2)
                for p in group['params']:
                    p.data -= group['lr'] * r_tilde * p.grad / group['ME']
            elif group['opL'] == "diag_hessian":
                # flatten the group parameters and gradients
                params_flatten = torch.cat([p.view(-1) for p in group['params']])
                grad1 = torch.autograd.grad(self.loss, group['params'], create_graph=True)
                grad_flatten = torch.cat([p.view(-1) for p in grad1])
                grad2 = []
                # calculate the diagonal of Hessian matrix
                # here we've calculate the whole Hessian matrix
                # TODO: calculate the diagonal of Hessian matrix without calculating the whole Hessian matrix
                for (g, x) in zip(grad1, group['params']):
                    hessian = torch.zeros_like(x)
                    for index in np.ndindex(*g.shape):
                        hessian[index] = (torch.autograd.grad(g[index], x, retain_graph=True)[0][index])
                    grad2.append(hessian.detach_())
                diag_L = torch.cat([p.view(-1) for p in grad2])
                # update params
                g = grad_flatten / group['ME']
                hat_g = (1 / (group['lr'] * diag_L + 1)) * g
                r_tilde = group['r'] / (1 + group['lr'] * torch.dot(g, hat_g) / 2)
                params_flatten = params_flatten - group['lr'] * r_tilde * hat_g

                # update group['params']
                start = 0
                for p in group['params']:
                    end = start + p.numel()
                    p.data = params_flatten[start:end].view(p.shape)
                    start = end
            else:
                raise ValueError("Invalid value for opL: {}".format(group['opL']))

            # update modefied energy (ME) and scalar auxilary variable (r)
            self.loss = closure()
            group['ME'] = torch.sqrt(self.loss + group['C'])
            if group['ME'] != r_tilde:
                xi = (group['ME'] - torch.sqrt((1-group['eta']) * r_tilde**2 + group['eta'] * group['r']**2 + (1-group['eta']) * (r_tilde-group['r'])**2)) / (group['ME'] - r_tilde)
                xi = max(xi, 0)
            else:
                xi = 0
            group['r'] = xi * r_tilde + (1 - xi) * group['ME']
            
        return self.loss.item()