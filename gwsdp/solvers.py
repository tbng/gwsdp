"""Semidefinite relaxations of Gromov-Wasserstein and fused GW transport."""

import numpy as np
from scipy import sparse

from gwsdp.utils import cost_tensor_numpy


def solve_gw_sdp(C1, C2, p, q, solver="SCS", max_iters=10000, verbose=False, tol=1e-5):
    """Solve the squared-loss GW relaxation between structures C1 and C2.

    p and q are marginal mass vectors. Choose 'scs', 'mosek', or 'rinnal'.
    Returns (Pi, P, value), where P lifts the column-major vectorization of
    the transport plan Pi and value is the objective evaluated on P.
    """
    return _solve(C1, C2, p, q, None, 1.0, solver, max_iters, verbose, tol)


def solve_fused_gw_sdp(
    M, C1, C2, p, q, alpha=0.5, max_iters=10000, tol=1e-5, verbose=False, solver="scs"
):
    """Solve (1-alpha) <M, Pi> + alpha <L, P> with the GW lift constraints.

    M is the rectangular feature cost and L is the squared structural loss.
    Solver choices and return values match solve_gw_sdp.
    """
    return _solve(C1, C2, p, q, M, alpha, solver, max_iters, verbose, tol)


def _solve(C1, C2, p, q, feature_cost, alpha, solver, max_iters, verbose, tol):
    solver = solver.lower()
    if solver == "rinnal":
        from gwsdp._rinnal import solve_rinnal

        return solve_rinnal(
            C1,
            C2,
            p,
            q,
            feature_cost=feature_cost,
            alpha=alpha,
            tol=tol,
            max_iters=max_iters,
            verbose=verbose,
        )
    if solver not in ("scs", "mosek"):
        raise TypeError("Solver is not supported.")

    import cvxpy as cp

    m, n = len(p), len(q)
    d = m * n
    loss = cost_tensor_numpy(C1, C2).transpose(1, 0, 3, 2).reshape(d, d)
    if np.count_nonzero(loss) <= loss.size / 2:
        loss = sparse.csr_matrix(loss)

    pi = cp.Variable((m, n))
    moment = cp.Variable((d, d), symmetric=True)
    x = cp.vec(pi, order="F")
    column = cp.reshape(x, (d, 1), order="F")
    lift = cp.bmat([[moment, column], [column.T, np.ones((1, 1))]])
    constraints = [
        lift >> 0,
        pi >= 0,
        moment >= 0,
        cp.sum(pi, axis=1) == p,
        cp.sum(pi, axis=0) == q,
    ]
    source = np.tile(np.eye(m), n)
    target = np.repeat(np.eye(n), m, axis=1)
    for row, mass in zip(source, p):
        constraints.append(moment @ row == mass * x)
    for row, mass in zip(target, q):
        constraints.append(moment @ row == mass * x)

    objective = alpha * cp.sum(cp.multiply(loss, moment))
    if feature_cost is not None:
        objective += (1 - alpha) * cp.sum(cp.multiply(feature_cost, pi))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    if solver == "scs":
        problem.solve(solver=cp.SCS, verbose=verbose, max_iters=max_iters, eps=tol)
    else:
        problem.solve(
            solver=cp.MOSEK,
            verbose=verbose,
            mosek_params={
                "MSK_IPAR_BI_MAX_ITERATIONS": max_iters,
                "MSK_IPAR_INTPNT_MAX_ITERATIONS": max_iters,
                "MSK_IPAR_SIM_MAX_ITERATIONS": max_iters,
            },
        )
    return pi.value, moment.value, problem.value
