"""Refine the uniform distance-profile bound using the symmetry of the lift."""

import numpy as np

from ._rinnal_profile import _assignment_slack


def refine_profile_multiplier(multiplier, lower_bound, p, cost_scale, sweeps):
    """Update a uniform assignment multiplier in place, preserving weak duality.

    For each pair (i,j), its slice of P has row and column sums p[0]*pi[i,j].
    Subtracting assignment dual potentials from that slice's cost transfers
    their value to a linear cost on pi. A second assignment splits this linear
    cost into a constant lower bound and a nonnegative remainder. Symmetry of
    P allows the remaining pair costs to be redistributed before another sweep.
    """
    n = len(p)
    remaining = multiplier[1:, 1:]
    conditional_cost = np.empty(n * n)
    for _ in range(sweeps):
        for row, costs in enumerate(remaining):
            slack, assignment_cost = _assignment_slack(costs.reshape(n, n))
            costs[:] = slack.ravel()
            conditional_cost[row] = p[0] * assignment_cost

        outer_slack, assignment_cost = _assignment_slack(conditional_cost.reshape(n, n))
        lower_bound += cost_scale * p[0] * assignment_cost
        # Each row of P sums to mass*pi[i,j].
        remaining += outer_slack.reshape(-1, 1) / p.sum()
        # Symmetrize in blocks to avoid a second array of size n**4.
        for start in range(0, n * n, n):
            stop = start + n
            block = remaining[start:stop, start:stop]
            block[:] = 0.5 * (block + block.T)
            upper = remaining[start:stop, stop:]
            lower = remaining[stop:, start:stop]
            upper[:] = 0.5 * (upper + lower.T)
            lower[:] = upper.T
    return multiplier, float(lower_bound)
