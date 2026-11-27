"""Dual warm start from one-dimensional transport between distance profiles."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _assignment_slack(cost):
    """Recover nonnegative reduced costs of the optimal uniform assignment."""
    rows, perm = linear_sum_assignment(cost)
    matched = cost[rows, perm]
    # In matched-column order, v[j] - v[i] <= c[i,perm[j]] - c[i,perm[i]].
    # Bellman-Ford from a zero-cost super-source gives feasible potentials.
    edges = cost[:, perm] - matched[:, None]
    v = np.zeros_like(matched)
    for _ in range(len(v) - 1):
        updated = np.minimum(v, np.min(v[:, None] + edges, axis=0))
        if np.array_equal(updated, v):
            break
        v = updated
    u = matched - v
    return np.maximum(cost - u[:, None] - v[np.argsort(perm)], 0.0), matched.sum()


def profile_multiplier(C1, C2, p, q, feature, alpha, cost_scale):
    """Build an orthant multiplier in the same units as the normalized cost.

    For equal uniform measures, each conditional second-moment row transports
    two equally weighted distance profiles. Sorting solves that 1D problem.
    Its dual slacks, together with the outer assignment slack, leave a constant
    linear objective on the transport face. Return (multiplier, lower_bound)
    for supported inputs and None otherwise.
    """
    n = len(p)
    if (
        len(q) != n
        or p[0] <= 0
        or not np.all(p == p[0])
        or not np.all(q == p[0])
        or not np.array_equal(C1, C1.T)
        or not np.array_equal(C2, C2.T)
    ):
        return None

    source = C1 - C1[0, 0]
    target = C2 - C1[0, 0]
    source_order = np.argsort(source, axis=1)
    target_order = np.argsort(target, axis=1)
    x = np.take_along_axis(source, source_order, axis=1)
    y = np.take_along_axis(target, target_order, axis=1)

    # For sorted costs -2 alpha x[k] y[l], adjacent inequalities sandwich
    # successive differences of the source potential. Their midpoint is
    # feasible; equality on matched ranks gives the target potential.
    a = np.zeros((n, n, n))
    steps = -alpha * np.diff(x, axis=1)[:, None, :] * (y[:, :-1] + y[:, 1:])
    a[:, :, 1:] = np.cumsum(steps, axis=2)
    b = -2 * alpha * x[:, None, :] * y[None, :, :] - a
    # Squared distances differ from the cross term by row and column
    # potentials. These preserve assignment slack and avoid cancellation
    # in the lower bound when the two structures nearly match.
    profiles = alpha * p[0] * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    profiles += (1 - alpha) * feature
    outer, assignment_cost = _assignment_slack(profiles)
    lower_bound = p[0] * assignment_cost
    a = np.take_along_axis(a, np.argsort(source_order, axis=1)[:, None, :], axis=2)
    b = np.take_along_axis(b, np.argsort(target_order, axis=1)[None, :, :], axis=2)

    # Fill the symmetrized conditional slacks one target block at a time.
    # Each temporary has n^3 entries, while the returned multiplier has n^4.
    size = n * n
    W = np.empty((size + 1, size + 1))
    W[0] = W[:, 0] = 0.0
    # P @ ones = mass * vec(plan), so the outer slack on the plan can
    # instead be split symmetrically between rows and columns of P.
    outer = outer / (2 * p.sum())
    for j in range(n):
        cross = -2 * alpha * source[:, :, None] * target[j]
        forward = a[:, j, :, None] + b[:, j, None, :]
        reverse = a.transpose(2, 0, 1) + b[None, :, :, j]
        block = np.maximum(cross - 0.5 * (forward + reverse), 0.0)
        block += outer[:, j, None, None] + outer[None, :, :]
        W[1 + j * n : 1 + (j + 1) * n, 1:] = (
            block.transpose(0, 2, 1).reshape(n, -1) / cost_scale
        )
    return W, float(lower_bound)
