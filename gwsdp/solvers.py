"""
SDP-based solvers for Gromov-Wasserstein optimal transport problems.

This module provides implementations of semidefinite programming (SDP) approaches
to solve Gromov-Wasserstein (GW) and fused Gromov-Wasserstein (FGW) problems.
These solvers find optimal transport plans that minimize the discrepancy between
the structure of two metric spaces, possibly with additional feature information.
"""

import gc

import cvxpy as cp
import numpy as np
from scipy import sparse
from tqdm import tqdm

from gwsdp.utils import (cost_tensor_numpy, gw_from_cost,
                         sinkhorn_scaling)


def solve_gw_sdp(C1, C2, p, q, solver='SCS', max_iters=10000,
                 verbose=False, tol=1e-5):
    """Solve the Gromov-Wasserstein optimal transport problem using semidefinite programming.

    This function solves the following optimization problem:
    min_{Pi} <L, P> s.t. Pi is a coupling between p and q, and P satisfies the necessary constraints.

    Args:
        C1: First distance/cost matrix (m x m)
        C2: Second distance/cost matrix (n x n)
        p: Source distribution (m-dimensional vector)
        q: Target distribution (n-dimensional vector)
        solver: SDP solver to use ('SCS' or 'mosek')
        max_iters: Maximum number of iterations for the solver
        verbose: Whether to display solver progress
        tol: Tolerance for solver convergence

    Returns:
        Pi: Optimal transport plan (m x n matrix)
        P: Lifted transport matrix (mn x mn matrix)
        value: Optimal objective value
    """
    # Problem size
    m = p.size
    n = q.size

    # Flatten the cost tensor
    L = cost_tensor_numpy(C1, C2).transpose(1, 0, 3, 2).reshape(m * n, m * n)

    # Convert to sparse matrix if sparsity ratio is smaller than 0.5
    if np.sum(L != 0) / L.size <= 0.5:
        L = sparse.csr_matrix(L)

    # Initialize problem
    Pi = cp.Variable((m, n))  # the transportation matrix
    P = cp.Variable((m * n, m * n))

    M = cp.bmat([
        [P, cp.reshape(cp.vec(Pi), (m * n, 1))],
        [cp.reshape(cp.vec(Pi), (1, m * n)), np.ones((1, 1))]
    ])

    constraints = [
        M >> 0,
        Pi >= 0,
        P >= 0,
        Pi @ np.ones(n) == p,
        Pi.T @ np.ones(m) == q,
    ]

    # marg constraint
    ais = np.tile(np.eye(m), n)
    bjs = np.repeat(np.eye(n), m, axis=1)

    for i in range(m):
        constraints += [
            P @ ais[i] == p[i] * cp.vec(Pi)
        ]

    for j in range(n):
        constraints += [
            P @ bjs[j] == q[j] * cp.vec(Pi)
        ]

    # solve the problem
    prob = cp.Problem(cp.Minimize(cp.trace(L @ P.T)), constraints)
    if solver == 'scs':
        prob.solve(solver=cp.SCS, verbose=verbose,
                   max_iters=max_iters, eps=tol)
    elif solver == 'mosek':
        prob.solve(solver=cp.MOSEK, verbose=verbose,
                   mosek_params={'MSK_IPAR_BI_MAX_ITERATIONS': max_iters,
                                 'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iters,
                                 'MSK_IPAR_SIM_MAX_ITERATIONS': max_iters, })
    else:
        raise TypeError('Solver is not supported.')

    return Pi.value, P.value, prob.value


def solve_fused_gw_sdp(M, C1, C2, p, q, alpha=0.5, max_iters=10000, tol=1e-5,
                       verbose=False, solver='scs'):
    """Solve the fused Gromov-Wasserstein optimal transport problem using semidefinite programming.

    This function solves a weighted combination of standard OT and GW:
    min_{Pi} (1-alpha)*<M,Pi> + alpha*<L,P> s.t. Pi is a coupling between p and q

    The fused approach combines structural similarity (GW term) with feature similarity (OT term).

    Args:
        M: Cost matrix for feature distances (m x n)
        C1: First structure matrix (m x m)
        C2: Second structure matrix (n x n)
        p: Source distribution (m-dimensional vector)
        q: Target distribution (n-dimensional vector)
        alpha: Weight parameter between 0 and 1 (0: pure OT, 1: pure GW)
        max_iters: Maximum number of iterations for the solver
        tol: Tolerance for solver convergence
        verbose: Whether to display solver progress
        solver: SDP solver to use ('scs' or 'mosek')

    Returns:
        Pi: Optimal transport plan (m x n matrix)
        P: Lifted transport matrix (mn x mn matrix)
        value: Optimal objective value
    """
    # Problem size
    m = p.size
    n = q.size

    # Flatten the cost tensor
    L = cost_tensor_numpy(C1, C2).transpose(1, 0, 3, 2).reshape(m * n, m * n)

    # Convert to sparse matrix if sparsity ratio is smaller than 0.5
    if np.sum(L != 0) / L.size <= 0.5:
        L = sparse.csr_matrix(L)

    # Initialize problem
    Pi = cp.Variable((m, n))  # the transportation matrix
    P = cp.Variable((m * n, m * n))

    M = cp.bmat([
        [P, cp.reshape(cp.vec(Pi), (m * n, 1))],
        [cp.reshape(cp.vec(Pi), (1, m * n)), np.ones((1, 1))]
    ])

    constraints = [
        M >> 0,
        Pi >= 0,
        P >= 0,
        Pi @ np.ones(n) == p,
        Pi.T @ np.ones(m) == q,
    ]

    # marg constraint
    ais = np.tile(np.eye(m), n)
    bjs = np.repeat(np.eye(n), m, axis=1)

    for i in range(m):
        constraints += [
            P @ ais[i] == p[i] * cp.vec(Pi)
        ]

    for j in range(n):
        constraints += [
            P @ bjs[j] == q[j] * cp.vec(Pi)
        ]

    prob = cp.Problem(cp.Minimize(
        (1 - alpha) * cp.trace(M.T @ Pi) + alpha * cp.trace(L @ P.T)), constraints)
    if solver == 'scs':
        prob.solve(solver=cp.SCS, verbose=verbose,
                   max_iters=max_iters, eps=tol)
    elif solver == 'mosek':
        prob.solve(solver=cp.MOSEK, verbose=verbose,
                   mosek_params={'MSK_IPAR_BI_MAX_ITERATIONS': max_iters,
                                 'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iters,
                                 'MSK_IPAR_SIM_MAX_ITERATIONS': max_iters,
                                 },
                   )
    else:
        raise TypeError('Solver is not supported.')

    return Pi.value, P.value, prob.value
