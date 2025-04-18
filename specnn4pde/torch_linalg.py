"""
torch_linalg.py

This module provides linear algebra utilities implemented with PyTorch.

Brief description:
    - incremental_qr: Incrementally update reduced QR decomposition when adding a new column.
    - omp: Orthogonal Matching Pursuit (OMP) using QR decomposition for sparse signal recovery.

For more information, see the documentation of each function.
"""

__all__ = ['incremental_qr', 'omp',
           ]

import torch

def incremental_qr(Q, R, a_new):
    """
    Incrementally update reduced QR decomposition when adding a new column.

    Args:
        Q (torch.Tensor): Existing orthogonal matrix (m, k).
        R (torch.Tensor): Existing upper triangular matrix (k, k).
        a_new (torch.Tensor): New column to add (m,).

    Returns:
        Q_new (torch.Tensor): Updated orthogonal matrix (m, k+1).
        R_new (torch.Tensor): Updated upper triangular matrix (k+1, k+1).
    """
    # Project a_new onto the existing Q
    q_proj = Q.T @ a_new
    r_new = a_new - Q @ q_proj

    # Normalize r_new to create a new orthogonal vector
    r_norm = torch.norm(r_new)
    if r_norm > torch.finfo(Q.dtype).resolution:  # Avoid division by zero
        r_new = r_new / r_norm
        Q_new = torch.cat([Q, r_new.unsqueeze(1)], dim=1)
        R_new = torch.cat([
            torch.cat([R, q_proj.unsqueeze(1)], dim=1),
            torch.cat([torch.zeros(1, R.shape[1], device=Q.device), r_norm.view(1, 1)], dim=1)
        ], dim=0)
    else:
        # If r_new is zero, no new orthogonal vector is added
        Q_new = Q
        R_new = torch.cat([R, q_proj.unsqueeze(1)], dim=1)

    return Q_new, R_new

def omp(A, b, max_atoms=10, tol=1e-6):
    """
    Orthogonal Matching Pursuit using QR decomposition (incremental version).

    Args:
        A (torch.Tensor): Dictionary matrix of shape (m, n).
        b (torch.Tensor): Observation vector of shape (m,) or (m, 1).
        max_atoms (int): Maximum number of atoms to select.
        tol (float): Tolerance for residual norm to stop early.

    Returns:
        x (torch.Tensor): Sparse solution vector of shape (n,).
        support (list): Indices of selected atoms.

    Example:
        >>> device = 'cuda'
        >>> features, basis = 200, 600
        >>> A = torch.randn(features, basis).to(dtype=torch.float64, device=device)
        >>> x_true = torch.zeros(basis).to(dtype=torch.float64, device=device)
        >>> x_true[torch.randperm(basis)[:5]] = torch.randn(5).to(dtype=torch.float64, device=device)
        >>> b = A @ x_true
        >>> x_est, support = omp(A, b, max_atoms=500, tol=1e-14)
        >>> print(f"Recover L2 error: {(A @ x_est - b).norm():.4e}")
        >>> print(f"Selected basis no.: {(x_est.abs() > 0).sum().item()}")
    """
    m, n = A.shape
    b = b.view(-1)  # ensure shape (m,)
    residual = b.clone()
    support = []

    for i in range(max_atoms):
        # Step 1: Correlation
        corr = torch.abs(A.T @ residual)
        corr[support] = -1  # mask already selected
        idx = torch.argmax(corr).item()

        # Step 2: Update support
        support.append(idx)
        A_support = A[:, support]  # (m, k)

        # Step 3: Solve least squares problem
        if i == 0:
            Q_mat, R_mat = torch.linalg.qr(A_support, mode='reduced')
        else:
            Q_mat, R_mat = incremental_qr(Q_mat, R_mat, A_support[:, -1])
        x_support = torch.linalg.solve(R_mat, Q_mat.T @ b)

        residual = b - A_support @ x_support
        # Early stopping
        if torch.norm(residual) < tol:
            break

    # Final sparse solution
    x = torch.zeros(n, dtype=A.dtype, device=A.device)
    x[support] = x_support
    return x, support