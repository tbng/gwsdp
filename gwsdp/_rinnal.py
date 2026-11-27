"""RiNNAL / RiNNAL+ specialized to the continuous GWSDP transport face."""

import warnings
from numbers import Integral

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment, minimize
from scipy.sparse.linalg import LinearOperator

from ._rinnal_bounds import refine_profile_multiplier
from ._rinnal_face import TransportFace
from ._rinnal_profile import profile_multiplier
from ._rinnal_start import initial_plan
from ._rinnal_projection import (
    append_negative_direction,
    least_eigenpair,
    project_factor,
)


class _GWCost:
    """Apply the lifted cost via Kronecker products, in vec_F order."""

    def __init__(self, C1, C2, p, q, feature, alpha, tol=1e-5):
        # The fixed marginals make the two squared-distance terms constant.
        # Subtracting a common offset from C1,C2 preserves every pairwise loss.
        self.original = (C1, C2)
        self.C1 = C1 - C1[0, 0]
        self.C2 = C2 - C1[0, 0]
        self.alpha = alpha
        self.direct_threshold = min(1e-8, 0.05 * tol)
        self.reduced_matrix = None
        self.offset = alpha * (p @ (self.C1**2) @ p + q @ (self.C2**2) @ q)
        self.feature = 0.5 * (1 - alpha) * feature.ravel(order="F")
        magnitude = max(
            2 * alpha * np.max(abs(self.C1)) * np.max(abs(self.C2)),
            np.max(abs(self.feature)),
        )
        self.scale = magnitude if magnitude > 0 else 1.0
        self.symmetric = np.array_equal(self.C1, self.C1.T) and np.array_equal(
            self.C2, self.C2.T
        )

        # Forward-error scale for the expanded constant-minus-cross form.
        # When costs nearly match, that form can lose the whole GW loss.
        mass = p.sum()
        self.roundoff_scale = (
            16
            * (len(p) + len(q))
            * np.finfo(float).eps
            * (abs(self.offset) + self.scale * (mass + mass**2))
        )

    def _direct_value(self, x, active, lift=None):
        """Contract the original squared differences in bounded-size blocks."""
        C1, C2 = self.original
        i, j = active % len(C1), active // len(C1)
        weights = x[active]
        value = 2 * (self.feature @ x)
        if self.alpha == 0:
            return float(value)
        block = max(1, 262144 // len(active))
        for start in range(0, len(active), block):
            rows = slice(start, start + block)
            loss = C1[np.ix_(i[rows], i)] - C2[np.ix_(j[rows], j)]
            np.square(loss, out=loss)
            if lift is None:
                term = weights[rows] @ (loss @ weights)
            else:
                term = np.vdot(loss, lift[rows])
            value += self.alpha * term
        return float(value)

    def plan_value(self, plan):
        """Evaluate a rank-one feasible upper bound, preserving small losses."""
        x = plan.ravel(order="F")
        active = np.flatnonzero(x)
        # A scaled permutation has only n active entries: direct evaluation
        # costs O(n²) and avoids subtracting two nearly equal large costs.
        if len(active) <= 2 * sum(plan.shape):
            return self._direct_value(x, active)
        full = np.r_[1.0, x][:, None]
        value = float(self.offset + self.scale * np.vdot(full, self.product(full)))
        if self.roundoff_scale > self.direct_threshold * (1 + abs(value)):
            return self._direct_value(x, active)
        return value

    def moment_value(self, primal, full, moment):
        value = float(self.offset + self.scale * primal)
        if self.roundoff_scale > self.direct_threshold * (1 + abs(value)):
            x = full[1:, 0]
            return self._direct_value(x, np.arange(len(x)), moment[1:, 1:])
        return value

    def product(self, full):
        matrices = full[1:].T.reshape(-1, len(self.C2), len(self.C1))
        cross = self.C2 @ matrices @ self.C1.T
        if self.symmetric:
            cross *= -2 * self.alpha
        else:
            cross = -self.alpha * (cross + self.C2.T @ matrices @ self.C1)
        bottom = cross.reshape(len(matrices), -1).T + self.feature[:, None] * full[0]
        return np.vstack((self.feature @ full[1:], bottom)) / self.scale


def _input_array(value, name, ndim):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite nonempty {ndim}-dimensional array")
    return array


def _factor_solve(face, cost, factor, multiplier, penalty, tol, inner_limit):
    """Minimize the shifted orthant AL on the factor's affine slice."""
    rank = factor.shape[1]

    def objective(coordinates):
        candidate = np.vstack((factor[0], coordinates.reshape(-1, rank)))
        full = face.expand(candidate)
        cost_product = cost.product(full)
        negative = full @ full.T
        negative *= penalty
        negative -= multiplier
        # P is entrywise nonnegative and P @ 1 = mass*x, so x >= 0
        # follows. Omitting its redundant AL term improves conditioning.
        negative[0, :] = 0
        negative[:, 0] = 0
        np.minimum(negative, 0, out=negative)
        value = np.vdot(full, cost_product) + np.vdot(negative, negative) / (
            2 * penalty
        )
        gradient = face.reduce(cost_product + negative @ full)
        return value, 2 * gradient[1:].ravel()

    result = minimize(
        objective,
        factor[1:].ravel(),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": inner_limit,
            "maxls": 35,
            "gtol": max(1e-12, 0.05 * tol),
            "ftol": 4 * np.finfo(float).eps,
            "maxcor": 10,
        },
    )
    return np.vstack((factor[0], result.x.reshape(-1, rank)))


def _lifted_state(face, cost, factor, multiplier, penalty):
    full = face.expand(factor)
    moment = full @ full.T
    new_multiplier = np.maximum(multiplier - penalty * moment, 0)
    new_multiplier[0, :] = 0
    new_multiplier[:, 0] = 0
    cost_product = cost.product(full)
    value = np.vdot(full, cost_product)
    lagrangian = value - np.vdot(new_multiplier, moment)

    def gradient(columns):
        expanded = face.expand(columns)
        return face.reduce(cost.product(expanded) - new_multiplier @ expanded)

    def dense_gradient():
        # Form B.T @ W @ B via two separable contractions. Multiplying the
        # dense multiplier by an explicit basis would require cubic work.
        if cost.reduced_matrix is None:
            basis = face.expand(np.eye(face.dimension))
            cost.reduced_matrix = face.reduce(cost.product(basis))
        return cost.reduced_matrix - face.reduce(face.reduce(new_multiplier).T).T

    return full, moment, new_multiplier, value, lagrangian, gradient, dense_gradient


def _dual_residual(face, gradient, lagrangian, trace_budget, dense_slack=None):
    """Test weak duality on a diagonally rescaled, trace-bounded PSD face.

    For every feasible reduced S, S00=c² and tr(S[1:,1:]) <= trace_budget.
    In the diagonally rescaled coordinates the trace is at most two.
    """
    dimension = face.dimension
    scale = np.full(dimension, np.sqrt(trace_budget))
    scale[0] = face.first_norm
    intercept = lagrangian / face.first_norm**2

    def slack_matrix(columns):
        scaled = scale[:, None] * columns
        product = gradient(scaled)
        product[0] -= intercept * scaled[0]
        return scale[:, None] * product

    if dense_slack is not None:
        slack = scale[:, None] * dense_slack * scale
        slack[0, 0] -= lagrangian
        slack = (slack + slack.T) * 0.5
        eigenvalue, directions = eigh(slack, subset_by_index=[0, 0])
        return eigenvalue[0], scale * directions[:, 0]
    operator = LinearOperator(
        (dimension, dimension),
        dtype=float,
        matvec=lambda v: slack_matrix(v.reshape(-1, 1)).ravel(),
    )
    eigenvalue, direction = least_eigenpair(operator, tol=1e-5)
    return eigenvalue, scale * direction


def _round_plan(raw_plan, p, q, uniform_square):
    """Construct a feasible coupling for an upper bound on the lifted SDP."""
    if uniform_square:
        plan = np.zeros_like(raw_plan)
        plan[linear_sum_assignment(-raw_plan)] = p[0]
        return plan

    # Altschuler et al. (2017), Round: scale rows and columns down, then
    # distribute the remaining mass using the outer product of the deficits.
    plan = np.maximum(raw_plan, 0)
    plan *= (p / np.maximum(p, plan.sum(axis=1)))[:, None]
    plan *= q / np.maximum(q, plan.sum(axis=0))
    rows = np.maximum(p - plan.sum(axis=1), 0)
    cols = np.maximum(q - plan.sum(axis=0), 0)
    if cols.sum() > 0:
        plan += np.outer(rows, cols / cols.sum())
    return plan


def _rank_one_result(plan, value, lower_bound, iteration):
    """Return a feasible plan whose rank-one lift closes the bound gap."""
    x = plan.ravel(order="F")
    gap = float(abs(value - lower_bound) / (1 + abs(value)))
    info = dict(
        iterations=iteration,
        converged=True,
        rank=1,
        primal_residual=0.0,
        lower_bound=float(lower_bound),
        upper_bound=value,
        relative_gap=gap,
        bound_gap=gap,
    )
    return plan, np.outer(x, x), value, info


def _active_solve(C1, C2, p, q, feature, alpha, tol, max_iters, verbose):
    face = TransportFace(p, q)
    cost = _GWCost(C1, C2, p, q, feature, alpha, tol)
    mass = p.sum()
    small_face = face.dimension <= 128
    # On a nearly isometric input at large scale, the stable distance-profile
    # certificate can resolve a loss below the precision of the cross term.
    use_profile = (
        not small_face or cost.roundoff_scale > 0.1 * tol
    ) and face.dimension > 1
    coupling = (
        np.outer(p, q) / mass
        if not use_profile
        else initial_plan(C1, C2, p, q, feature, alpha)
    )
    factor = face.reduce(np.r_[1.0, coupling.ravel(order="F")][:, None])
    factor[0, 0] = face.first_norm
    best_plan = coupling
    upper_bound = cost.plan_value(coupling)
    uniform_square = len(p) == len(q) and np.all(p == p[0]) and np.all(q == p[0])
    if face.dimension == 1:
        return _rank_one_result(coupling, upper_bound, upper_bound, 0)

    size = len(p) * len(q)
    # P_kk <= min(p_i,q_j) x_k, so trace(P) <= mass min(max(p),max(q)).
    trace_budget = mass * min(p.max(), q.max()) - face.origin @ face.origin
    trace_budget = max(trace_budget, np.finfo(float).eps * mass**2)
    lower_bound = (1 - alpha) * mass * feature.min()
    profile = (
        None
        if not use_profile
        else profile_multiplier(C1, C2, p, q, feature, alpha, cost.scale)
    )
    if profile is None:
        multiplier = np.zeros((1 + size, 1 + size))
    else:
        multiplier, profile_bound = profile
        lower_bound = max(lower_bound, profile_bound)
        if abs(upper_bound - lower_bound) <= tol * (1 + abs(upper_bound)):
            return _rank_one_result(coupling, upper_bound, lower_bound, 0)
        if cost.roundoff_scale <= 0.1 * tol * (1 + abs(profile_bound)):
            multiplier, refined_bound = refine_profile_multiplier(
                multiplier, profile_bound, p, cost.scale, sweeps=8
            )
            lower_bound = max(lower_bound, refined_bound)
        if abs(upper_bound - lower_bound) <= tol * (1 + abs(upper_bound)):
            return _rank_one_result(coupling, upper_bound, lower_bound, 0)
    del profile
    penalty = max(1.0, size / mass**2)
    previous_residual = np.inf
    estimate = -np.inf
    scaled_tol = tol / cost.scale
    inner_limit = 200

    for iteration in range(1, max_iters + 1):
        factor = _factor_solve(
            face, cost, factor, multiplier, penalty, scaled_tol, inner_limit
        )
        full, moment, new_multiplier, primal, lagrangian, gradient, dense_gradient = (
            _lifted_state(face, cost, factor, multiplier, penalty)
        )
        value = cost.moment_value(primal, full, moment)
        violation = np.linalg.norm(np.minimum(moment, 0))
        residual = max(violation, np.linalg.norm(new_multiplier - multiplier) / penalty)
        verify = small_face or iteration == max_iters or iteration % 25 == 0
        dense = dense_gradient() if verify else None
        eigenvalue, direction = _dual_residual(
            face, gradient, lagrangian, trace_budget, dense
        )
        candidate = cost.offset + cost.scale * (lagrangian + 2 * min(0.0, eigenvalue))
        estimate = candidate
        # The spectral certificate is formed in the expanded cost units.
        # If their cancellation exceeds the accuracy of this bound, retain
        # the independently evaluated lower bound (e.g. distance profiles).
        reliable = cost.roundoff_scale <= 0.1 * tol * (1 + abs(candidate))
        if verify and reliable:
            lower_bound = max(lower_bound, candidate)
        elif (
            not verify
            and reliable
            and violation <= tol
            and (abs(value - candidate) <= tol * (1 + abs(value)))
        ):
            # An iterative Ritz value can overestimate the least eigenvalue.
            # Confirm convergence with a full-spectrum lower endpoint.
            eigenvalue, direction = _dual_residual(
                face, gradient, lagrangian, trace_budget, dense_gradient()
            )
            candidate = cost.offset + cost.scale * (
                lagrangian + 2 * min(0.0, eigenvalue)
            )
            if cost.roundoff_scale <= 0.1 * tol * (1 + abs(candidate)):
                lower_bound = max(lower_bound, candidate)
        raw_plan = full[1:, 0].reshape(coupling.shape, order="F")
        rounded = _round_plan(raw_plan, p, q, uniform_square)
        rounded_value = cost.plan_value(rounded)
        if rounded_value < upper_bound:
            best_plan, upper_bound = rounded, rounded_value
        bound_gap = abs(upper_bound - lower_bound) / (1 + abs(upper_bound))
        gap = abs(value - lower_bound) / (1 + abs(value))
        converged = violation <= tol and gap <= tol
        if not small_face and bound_gap <= tol:
            return _rank_one_result(best_plan, upper_bound, lower_bound, iteration)
        if verbose:
            print(
                f"RiNNAL {iteration:4d}  rank {factor.shape[1]:3d}  "
                f"objective {value:.8g}  primal {violation:.2e}  "
                f"gap estimate {abs(value - estimate) / (1 + abs(value)):.2e}",
                flush=True,
            )
        if converged or iteration == max_iters:
            break
        # Correct spurious stationary points of the fixed-rank AL subproblem.
        # The reduced gradient is penalty-Lipschitz because the basis is
        # isometric and entrywise projection is nonexpansive.
        # Let descent settle at the new rank before adding further columns.
        rank_update = small_face or (iteration - 1) % 5 == 0
        if rank_update and eigenvalue < -0.1 * scaled_tol * (1 + abs(value)):
            if small_face:
                trial = factor @ factor.T - dense / penalty
                factor = project_factor(trial, face.first_norm)
            else:
                factor = append_negative_direction(
                    factor, direction, lambda v: gradient(v[:, None]).ravel(), penalty
                )
        if factor.shape[1] > 2:
            # Drop numerically null covariance directions without rotating
            # the first column that represents the transport plan.
            vectors, singular, _ = np.linalg.svd(factor[1:, 1:], full_matrices=False)
            # On larger faces compress weak directions to control width.
            # Tighten the cutoff with the requested solution accuracy.
            cutoff = 1e-7 if small_face else min(0.02, 8 * np.sqrt(tol))
            keep = singular > cutoff * singular[0]
            if keep.sum() + 1 < factor.shape[1]:
                tail = np.vstack(
                    (np.zeros(keep.sum()), vectors[:, keep] * singular[keep])
                )
                factor = np.column_stack((factor[:, 0], tail))
        multiplier = new_multiplier
        if small_face and iteration >= 10:
            inner_limit = 700
        if (
            residual > max(tol * 0.05, 0.75 * previous_residual)
            and gap < 200 * residual * cost.scale
        ):
            penalty = min(1.5 * penalty, 1e8 * max(1.0, size / mass**2))
        elif residual < min(1e-3, 1e-3 * gap / cost.scale):
            penalty = max(0.01, penalty / 1.5)
        previous_residual = residual
        # Release the previous lift and closures before the next inner solve.
        full = moment = gradient = dense_gradient = dense = None

    info = dict(
        iterations=iteration,
        converged=bool(converged),
        rank=factor.shape[1],
        primal_residual=float(violation),
        lower_bound=float(lower_bound),
        relative_gap=float(gap),
        upper_bound=upper_bound,
        bound_gap=float(bound_gap),
    )
    if not converged:
        warnings.warn(
            f"RiNNAL stopped after {iteration} iterations before convergence "
            f"(primal residual {violation:.3g}, relative gap {gap:.3g})",
            RuntimeWarning,
            stacklevel=3,
        )
    return full[1:, 0].reshape(coupling.shape, order="F"), moment[1:, 1:], value, info


def solve_rinnal(
    C1,
    C2,
    p,
    q,
    *,
    feature_cost=None,
    alpha=1.0,
    tol=1e-5,
    max_iters=10000,
    verbose=False,
    return_info=False,
):
    """Solve the GW or fused-GW doubly nonnegative relaxation in factor space.

    max_iters counts outer multiplier updates. When return_info is true, a
    fourth result gives the stopping diagnostics for the lifted SDP.
    """
    # Marginals formed independently in float32 (or float16) can differ at
    # their input precision, even when both are normalized probabilities.
    mass_rtol = 1e-12
    for marginal in (p, q):
        dtype = np.asarray(marginal).dtype
        if np.issubdtype(dtype, np.floating):
            mass_rtol = max(mass_rtol, 3 * np.finfo(dtype).eps)
    p = _input_array(p, "p", 1)
    q = _input_array(q, "q", 1)
    C1 = _input_array(C1, "C1", 2)
    C2 = _input_array(C2, "C2", 2)
    if C1.shape != (len(p), len(p)) or C2.shape != (len(q), len(q)):
        raise ValueError("C1 and C2 must be square and match the marginal sizes")
    source_mass, target_mass = p.sum(), q.sum()
    if (
        np.any(p < 0)
        or np.any(q < 0)
        or min(source_mass, target_mass) <= 0
        or not np.isfinite([source_mass, target_mass]).all()
    ):
        raise ValueError("marginals must be nonnegative and have positive mass")
    if not np.isclose(source_mass, target_mass, rtol=mass_rtol, atol=0):
        raise ValueError("marginals must have equal total mass")
    if source_mass != target_mass:
        q = q * (source_mass / target_mass)
    if not (np.isfinite(alpha) and 0 <= alpha <= 1):
        raise ValueError("alpha must be between zero and one")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError("tol must be finite and positive")
    if (
        isinstance(max_iters, bool)
        or not isinstance(max_iters, Integral)
        or max_iters <= 0
    ):
        raise ValueError("max_iters must be a positive integer")
    feature = (
        np.zeros((len(p), len(q)))
        if feature_cost is None
        else _input_array(feature_cost, "feature_cost", 2)
    )
    if feature.shape != (len(p), len(q)):
        raise ValueError("feature_cost must have shape (len(p), len(q))")

    rows, cols = np.flatnonzero(p), np.flatnonzero(q)
    # A zero marginal forces the corresponding rows and columns of the lift
    # to vanish. Removing them also restores relative strict feasibility.
    plan, lift, value, info = _active_solve(
        C1[np.ix_(rows, rows)],
        C2[np.ix_(cols, cols)],
        p[rows],
        q[cols],
        feature[np.ix_(rows, cols)],
        alpha,
        tol,
        max_iters,
        verbose,
    )
    if len(rows) != len(p) or len(cols) != len(q):
        original = np.zeros((len(p), len(q)))
        original[np.ix_(rows, cols)] = plan
        active = (rows[:, None] + len(p) * cols).ravel(order="F")
        expanded = np.zeros((original.size, original.size))
        expanded[np.ix_(active, active)] = lift
        plan, lift = original, expanded
    return (plan, lift, value, info) if return_info else (plan, lift, value)
