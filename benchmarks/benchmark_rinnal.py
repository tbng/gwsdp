"""Reproduce one GWSDP simulation per process, with a JSON record on stdout.

From the checkout, run e.g. `python -m benchmarks.benchmark_rinnal --size 6 6`.
The Gaussian experiment uses the repository's 2D/3D point-cloud generator.
The two SBM experiments follow sections 4.2 and 4.3 of arXiv:2312.14572v3.
The scaled isometry relabels a cloud and multiplies its distances by 1.5.
Run SCS separately on the same size and seed to compare time and peak RSS.
"""

import argparse
import json
import sys
from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

from gwsdp.generate_dataset import generate_sample, generate_sample_two


def make_instance(case, m, n, seed):
    # The upstream functions also use NumPy's global RNG (target cloud / features).
    # Give it a separate reproducible stream from the source cloud / graph seed.
    np.random.seed(np.random.SeedSequence([seed, 1]).generate_state(1)[0])
    if case == "replicated-blocks":
        # Uniform lifts of the two-point and five-point discrete metrics.
        # The SDP attains 3/10, strictly below the best coupling's 17/50.
        if m % 2 or n % 5:
            raise ValueError(
                "Replicated blocks require an even source size and target size divisible by 5"
            )
        rng = np.random.default_rng(seed)
        source = rng.permutation(np.repeat(np.arange(2), m // 2))
        target = rng.permutation(np.repeat(np.arange(5), n // 5))
        C = (source[:, None] != source).astype(float)
        D = (target[:, None] != target).astype(float)
        return C, D, np.full(m, 1 / m), np.full(n, 1 / n), None, 1.0
    if case in ("gaussian", "scaled-isometry"):
        C, D, p, q, *_ = generate_sample(m, n, seed=seed)
        C, D = C.astype(float), D.astype(float)
        if case == "scaled-isometry":
            if m != n:
                raise ValueError(
                    "A scaled isometry requires equal source and target sizes"
                )
            perm = np.random.default_rng(np.random.SeedSequence([seed, 2])).permutation(
                n
            )
            D = 1.5 * C[np.ix_(perm, perm)]
        return C, D, p, q, None, 1.0

    if m % 3 or n % 2:
        raise ValueError("SBM source size must be divisible by 3 and target size by 2")
    # Paper orientation is three blocks -> two; all between-block probabilities are .1.
    p3 = [[1, 0.1, 0.1], [0.1, 0.95, 0.1], [0.1, 0.1, 0.9]]
    D, C, q, p, G, F, *_ = generate_sample_two(n, m, seed=seed, p3=p3)
    if case == "sbm-gw":
        return C * cdist(F, F), D * cdist(G, G), p, q, None, 1.0

    C, D = (shortest_path(A, directed=False, unweighted=True) for A in (C, D))
    if not (np.isfinite(C).all() and np.isfinite(D).all()):
        raise ValueError(
            "Selected SBM seed produced a disconnected graph; choose another seed"
        )
    return C, D, p, q, cdist(F, G), 0.8


def solve_conic(C, D, p, q, M, alpha, tol, max_iters, solver):
    """Solve the full convex lift independently, without the factorization."""
    m, n = len(p), len(q)
    d = m * n
    Y = cp.Variable((d + 1, d + 1), symmetric=True)
    # Stacking [-b, A] gives all original and lifted marginal equalities at once.
    A = sparse.vstack(
        [
            sparse.kron(np.ones((1, n)), sparse.eye(m)),
            sparse.kron(sparse.eye(n), np.ones((1, m))),
        ]
    )
    face = sparse.hstack([-np.r_[p, q][:, None], A], format="csr")
    loss = (C[None, :, None, :] - D[:, None, :, None]).reshape(d, d) ** 2
    objective = alpha * cp.sum(cp.multiply(loss, Y[1:, 1:]))
    if M is not None:
        objective += (1 - alpha) * (M.ravel(order="F") @ Y[0, 1:])
    problem = cp.Problem(
        cp.Minimize(objective), [Y >> 0, Y >= 0, Y[0, 0] == 1, face @ Y == 0]
    )
    if solver == "scs":
        problem.solve(solver="SCS", eps=tol, max_iters=max_iters)
    else:
        problem.solve(
            solver="CLARABEL",
            max_iter=max_iters,
            tol_feas=tol,
            tol_gap_abs=tol,
            tol_gap_rel=tol,
        )
    if Y.value is None:
        raise RuntimeError(f"{solver} returned no primal iterate ({problem.status})")
    return (
        Y.value[0, 1:].reshape((m, n), order="F"),
        Y.value[1:, 1:],
        problem.value,
        {
            "status": problem.status,
            "converged": problem.status == cp.OPTIMAL,
            "iterations": problem.solver_stats.num_iters,
        },
    )


def cost_of_plan(T, C, D, M, alpha):
    # Contraction of squared loss with T tensor T, using only 2D arrays.
    a, b = T.sum(axis=1), T.sum(axis=0)
    gw = a @ (C**2) @ a + b @ (D**2) @ b - 2 * np.sum(T * (C @ T @ D.T))
    return alpha * gw + ((1 - alpha) * np.sum(M * T) if M is not None else 0.0)


def diagnostics(C, D, p, q, M, alpha, pi, P):
    m, n = pi.shape
    x = pi.ravel(order="F")
    moments = P.reshape(m * n, n, m)
    marginal_error = max(abs(pi.sum(axis=1) - p).max(), abs(pi.sum(axis=0) - q).max())
    lifted_error = max(
        abs(moments.sum(axis=1) - np.outer(x, p)).max(),
        abs(moments.sum(axis=2) - np.outer(x, q)).max(),
    )
    upper = cost_of_plan(np.outer(p, q), C, D, M, alpha)
    result = {
        "plan_objective": cost_of_plan(pi, C, D, M, alpha),
        "min_plan": pi.min(),
        "min_lift": P.min(),
        "marginal_residual": marginal_error,
        "lifted_marginal_residual": lifted_error,
        "feasible_upper_bound": upper,
    }
    if m == n:  # All benchmark measures are uniform.
        # Conditional on matching (i,j), each second-moment row is a coupling
        # of the two uniform distance profiles. Their 1D W2 cost bounds that row.
        H = alpha * cdist(np.sort(C, axis=1), np.sort(D, axis=1), "sqeuclidean") / n
        if M is not None:
            H += (1 - alpha) * M
        result["profile_lower_bound"] = H[linear_sum_assignment(H)].mean()
        permutation = np.zeros_like(pi)
        permutation[linear_sum_assignment(-pi)] = 1 / n
        result["feasible_upper_bound"] = min(
            upper, cost_of_plan(permutation, C, D, M, alpha)
        )
    return result


def peak_rss_mib():
    """Process high-water RSS through solver return, including imports and input."""
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2**20 if sys.platform == "darwin" else 2**10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[
            "gaussian",
            "scaled-isometry",
            "replicated-blocks",
            "sbm-gw",
            "sbm-fgw",
        ],
        default="gaussian",
    )
    parser.add_argument("--size", type=int, nargs=2, default=(6, 6), metavar=("M", "N"))
    parser.add_argument("--seed", type=int, help="Default 0 for Gaussian, 5 for SBM")
    parser.add_argument(
        "--solver", choices=["rinnal", "scs", "clarabel"], default="rinnal"
    )
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max-iters", type=int, default=10000)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = 5 if args.case.startswith("sbm") else 0
    C, D, p, q, M, alpha = make_instance(args.case, *args.size, args.seed)
    if args.solver == "rinnal":
        from gwsdp import solve_rinnal
    start = perf_counter()
    if args.solver == "rinnal":
        pi, P, value, info = solve_rinnal(
            C,
            D,
            p,
            q,
            feature_cost=M,
            alpha=alpha,
            tol=args.tol,
            max_iters=args.max_iters,
            return_info=True,
        )
    else:
        pi, P, value, info = solve_conic(
            C, D, p, q, M, alpha, args.tol, args.max_iters, args.solver
        )
    elapsed = perf_counter() - start
    result = {
        **vars(args),
        "alpha": alpha,
        "wall_seconds": elapsed,
        "peak_rss_mib": peak_rss_mib(),
        "sdp_objective": value,
        **info,
        **diagnostics(C, D, p, q, M, alpha, pi, P),
    }
    if args.case == "scaled-isometry":
        # Cauchy-Schwarz bounds E[C_ik D_jl] for any feasible lift with
        # fixed pair marginals. The planted permutation attains equality.
        result["analytic_reference"] = (1.5 - 1) ** 2 * np.mean(C**2)
    elif args.case == "replicated-blocks":
        result["analytic_reference"] = 0.3
    # A stopped solve may have no finite dual bound or gap.
    for key, val in result.items():
        if isinstance(val, (float, np.floating)) and not np.isfinite(val):
            result[key] = None
    print(json.dumps(result, default=lambda x: x.item(), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
