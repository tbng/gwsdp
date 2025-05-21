"""
Utility functions for Gromov-Wasserstein optimal transport.

This module provides utilities for computing cost tensors, evaluating Gromov-Wasserstein
distances, and analyzing solutions. It includes both NumPy and PyTorch implementations
to support different computational backends.
"""

import torch
import numpy as np


def is_symmetric(matrix):
    """Check if a matrix is symmetric.

    Args:
        matrix: Input matrix to check

    Returns:
        bool: True if the matrix is equal to its transpose, False otherwise
    """
    return np.array_equal(matrix, matrix.T)


def cost_tensor_numpy(D1, D2):
    """
    Efficiently compute the cost tensor for Gromov-Wasserstein distance using NumPy broadcasting.

    Computes the 4D tensor L with elements L[i,j,k,l] = (D1[i,k] - D2[j,l])^2.
    This is the core tensor needed for Gromov-Wasserstein distance computations.

    Args:
        D1: Source distance matrix (m x m)
        D2: Target distance matrix (n x n)

    Returns:
        4D cost tensor (m x n x m x n)
    """
    # Expand dimensions to enable broadcasting
    D1_expanded = D1[:, np.newaxis, :, np.newaxis]
    D2_expanded = D2[np.newaxis, :, np.newaxis, :]

    # Calculate (D1[i,k] - D2[j,l])^2 using broadcasting
    return (D1_expanded - D2_expanded) ** 2



def cost_tensor_pytorch(D1: torch.Tensor, D2: torch.Tensor) -> torch.Tensor:
    """
    Efficiently compute the cost tensor for Gromov-Wasserstein distance using PyTorch broadcasting.

    PyTorch implementation of the 4D tensor L with elements L[i,j,k,l] = (D1[i,k] - D2[j,l])^2.

    Args:
        D1: Source distance matrix (m x m)
        D2: Target distance matrix (n x n)

    Returns:
        4D cost tensor (m x n x m x n)
    """
    # Expand dimensions to enable broadcasting
    D1_expanded = D1[:, None, :, None]
    D2_expanded = D2[None, :, None, :]

    # Calculate (D1[i,k] - D2[j,l])^2 using broadcasting
    return (D1_expanded - D2_expanded) ** 2


def gw_from_cost_pytorch(D1: torch.Tensor, D2: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gromov-Wasserstein distance using PyTorch tensors.

    PyTorch implementation of the Gromov-Wasserstein distance calculation.

    Args:
        D1: Source distance matrix (m x m)
        D2: Target distance matrix (n x n)
        pi: Transport plan (m x n)

    Returns:
        Gromov-Wasserstein distance value as a PyTorch tensor
    """
    # calculation from definition of GW
    L = cost_tensor_pytorch(D1, D2)
    T = torch.einsum('ijkl,kl->ij', L, pi)
    return torch.sum(T * pi)


def gw_from_cost(D1, D2, pi):
    """Compute the Gromov-Wasserstein distance using NumPy arrays.

    This function calculates the Gromov-Wasserstein distance between two metric spaces
    represented by their distance matrices D1 and D2, given a transport plan pi.

    Args:
        D1: Source distance matrix (m x m)
        D2: Target distance matrix (n x n)
        pi: Transport plan (m x n)

    Returns:
        float: Gromov-Wasserstein distance value
    """
    # calculation from definition of GW
    L = cost_tensor_numpy(D1, D2)
    T = np.einsum('ijkl,kl->ij', L, pi)
    return np.sum(T * pi)


def fgw_from_cost(M, D1, D2, pi, alpha):
    """Compute the fused Gromov-Wasserstein distance.

    This function calculates the fused Gromov-Wasserstein distance, which combines
    the structural distance (GW term) with feature similarity (OT term).

    Args:
        M: Feature cost matrix (m x n)
        D1: Source distance matrix (m x m)
        D2: Target distance matrix (n x n)
        pi: Transport plan (m x n)
        alpha: Weight parameter between 0 and 1 (0: pure OT, 1: pure GW)

    Returns:
        float: Fused Gromov-Wasserstein distance value
    """
    # calculation from definition of GW
    L = cost_tensor_numpy(D1, D2)
    m, n = pi.shape
    T = np.einsum('ijkl,kl->ij', L, pi)
    return alpha * np.sum(T * pi) + (1 - alpha) * np.sum(M * pi)


def optimality_gap(D1, D2, pi_sdp, pi_star):
    """Compute the optimality gap between an SDP solution and a ground truth solution.

    This function calculates the ratio between the GW distance achieved by the SDP solver
    and the GW distance achieved by a reference solution, typically a known optimal plan.

    Args:
        D1: Source distance matrix (m x m)
        D2: Target distance matrix (n x n)
        pi_sdp: Transport plan from SDP solver (m x n)
        pi_star: Reference transport plan (m x n)

    Returns:
        float: Optimality gap ratio (1.0 means equal performance)
    """
    return gw_from_cost(D1, D2, pi_sdp) / gw_from_cost(D1, D2, pi_star)


def approximation_gap(D1, D2, P, pi_sdp):
    """Compute the approximation gap between the lifted solution and the projected solution.

    This function calculates the ratio between the GW distance achieved by the projected plan pi_sdp
    and the objective value achieved by the lifted matrix P in the SDP formulation.

    Args:
        D1: Source distance matrix (m x m)
        D2: Target distance matrix (n x n)
        P: Lifted transport matrix from SDP solution (mn x mn)
        pi_sdp: Projected transport plan from SDP solution (m x n)

    Returns:
        float: Approximation gap ratio
    """
    m, n = pi_sdp.shape
    L = cost_tensor_numpy(D1, D2).reshape(m * n, m * n)
    return gw_from_cost(D1, D2, pi_sdp) / np.sum(L * P)


def sinkhorn_scaling(A, max_iters=1e3, tol=1e-12):
    """Perform Sinkhorn-Knopp matrix scaling to normalize rows and columns.

    This function takes a nonnegative matrix A and normalizes it to obtain a matrix M
    where the sum of each row and the sum of each column equals 1. This is achieved
    through an iterative scaling process also known as the Sinkhorn-Knopp algorithm.

    Args:
        A: Input nonnegative matrix
        max_iters: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Matrix with normalized rows and columns
    """
    iter = 0
    c = 1. / np.sum(A, 0)
    r = 1. / (A @ c)
    while iter < max_iters:
        iter += 1
        if np.max(np.abs(r @ A * c - 1)) <= tol:
            break
        c = 1. / (r @ A)
        r = 1. / (A @ c)
    return A * np.outer(r, c) / A.shape[0]
