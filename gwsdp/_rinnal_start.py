"""Feasible initialization for equally weighted GW matching problems."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def initial_plan(C, D, p, q, feature, alpha):
    """Match distance profiles and improve the assignment by pairwise swaps."""
    mass = p.sum()
    n = len(p)
    if n != len(q) or not (np.all(p == p[0]) and np.all(q == q[0])):
        return np.outer(p, q) / mass

    weight = p[0]
    rows = np.arange(n)
    profile = cdist(np.sort(C, axis=1), np.sort(D, axis=1), "sqeuclidean") * weight
    _, best = linear_sum_assignment(alpha * profile + (1 - alpha) * feature)
    source_diff = C.diagonal()[:, None] + C.diagonal() - C - C.T
    rng = np.random.default_rng(0)
    best_value = np.inf

    # Each swap is a rank-one change of the plan. Its exact quadratic cost
    # is the directional derivative plus the second-order term below. The
    # same formula covers directed structures and the linear feature cost.
    for attempt in range(64):
        perm = best.copy()
        if attempt % 2:
            subset = rng.choice(n, size=max(2, n // 4), replace=False)
            perm[subset] = rng.permutation(perm[subset])
        elif attempt:
            perm = rng.permutation(n)
        for _ in range(4 * n):
            target = D[np.ix_(perm, perm)]
            derivative = -2 * alpha * weight * (C @ target.T + C.T @ target)
            derivative += (1 - alpha) * feature[:, perm]
            first = (
                derivative
                + derivative.T
                - derivative.diagonal()[:, None]
                - derivative.diagonal()
            )
            target_diff = (
                target.diagonal()[:, None] + target.diagonal() - target - target.T
            )
            changes = weight * first - 2 * alpha * weight**2 * source_diff * target_diff
            i, j = np.unravel_index(changes.argmin(), changes.shape)
            if changes[i, j] >= -np.finfo(float).eps * max(1.0, abs(changes).max()):
                break
            perm[[i, j]] = perm[[j, i]]
        value = alpha * weight**2 * np.sum((C - D[np.ix_(perm, perm)]) ** 2)
        value += (1 - alpha) * weight * np.sum(feature[rows, perm])
        if value < best_value:
            best, best_value = perm, value
    plan = np.zeros((n, n))
    plan[rows, best] = weight
    return plan
