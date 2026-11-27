import cvxpy as cp
import numpy as np
import pytest

from gwsdp import solve_fused_gw_sdp, solve_gw_sdp


pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def lifted_data(C1, C2):
    """Construct the objective and marginal map using explicit (source, target) pairs."""
    m, n = len(C1), len(C2)
    i = np.tile(np.arange(m), n)
    j = np.repeat(np.arange(n), m)
    loss = (C1[np.ix_(i, i)] - C2[np.ix_(j, j)]) ** 2
    marginal_map = np.vstack([np.eye(m)[i].T, np.eye(n)[j].T])
    return loss, marginal_map


def reference_value(C1, C2, p, q, M, alpha):
    """Full convex SDP, with no transport-face or low-rank parametrization."""
    loss, A = lifted_data(C1, C2)
    d = len(loss)
    Y = cp.Variable((d + 1, d + 1), symmetric=True)
    b = np.r_[p, q]
    constraints = [Y >> 0, Y >= 0, Y[0, 0] == 1]
    for row, marginal in zip(A, b):
        constraints.append(row @ Y[1:, :] == marginal * Y[0, :])
    objective = alpha * cp.sum(cp.multiply(loss, Y[1:, 1:]))
    objective += (1 - alpha) * (M.ravel(order="F") @ Y[1:, 0])
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver="SCS", eps=1e-8, max_iters=100000)
    assert problem.status == cp.OPTIMAL
    return problem.value


def check_solution(C1, C2, p, q, result, M=None, alpha=1.0, atol=1e-5):
    pi, P, value = result
    m, n = len(p), len(q)
    assert pi.shape == (m, n)
    assert P.shape == (m * n, m * n)
    x = pi.ravel(order="F")
    Y = np.block([[np.ones((1, 1)), x[None, :]], [x[:, None], P]])
    loss, A = lifted_data(C1, C2)
    b = np.r_[p, q]
    np.testing.assert_allclose(A @ x, b, atol=atol, rtol=0)
    np.testing.assert_allclose(A @ P, np.outer(b, x), atol=atol, rtol=0)
    np.testing.assert_allclose(Y, Y.T, atol=atol, rtol=0)
    assert np.linalg.eigvalsh(Y)[0] >= -atol
    assert Y.min() >= -atol
    objective = alpha * np.sum(loss * P)
    if M is not None:
        objective += (1 - alpha) * np.sum(M * pi)
    assert value == pytest.approx(objective, abs=1e-10)


def test_two_point_rank_escape():
    # Starting at the independent coupling gives a stationary rank-one cost of 1.5.
    C = np.ones((2, 2)) - np.eye(2)
    p = np.full(2, 0.5)
    result = solve_gw_sdp(C, 2 * C, p, p, solver="rinnal", tol=1e-7)
    check_solution(C, 2 * C, p, p, result)
    assert result[2] == pytest.approx(0.5, abs=2e-6)


def test_sdp_with_strict_rank_one_gap():
    # The exact lifted optimum is 1/6. The best coupling has GW cost 5/18.
    C1, C2 = [np.ones((n, n)) - np.eye(n) for n in (2, 3)]
    p, q = np.full(2, 1 / 2), np.full(3, 1 / 3)
    result = solve_gw_sdp(C1, C2, p, q, solver="rinnal", tol=1e-7)
    check_solution(C1, C2, p, q, result)
    assert result[2] == pytest.approx(1 / 6, abs=2e-6)
    x = result[0].ravel(order="F")
    assert np.linalg.norm(result[1] - np.outer(x, x)) > 1e-3


@pytest.fixture
def rectangular():
    C1 = np.array([[0.0, 1.4], [1.4, 0.0]])
    C2 = np.abs(np.subtract.outer([0.0, 0.7, 1.9], [0.0, 0.7, 1.9]))
    p, q = np.array([0.6, 0.4]), np.array([0.2, 0.45, 0.35])
    M = np.array([[1.0, 0.0, 1.6], [0.3, 1.0, 0.0]])
    return C1, C2, p, q, M


@pytest.mark.parametrize("alpha", [0.0, 0.4, 1.0])
def test_fused_rectangular_against_full_sdp(rectangular, alpha):
    C1, C2, p, q, M = rectangular
    expected = reference_value(C1, C2, p, q, M, alpha)
    result = solve_fused_gw_sdp(M, C1, C2, p, q, alpha=alpha, solver="rinnal", tol=1e-6)
    check_solution(C1, C2, p, q, result, M, alpha)
    assert result[2] == pytest.approx(expected, abs=1e-5)


def test_asymmetric_structure_costs(rectangular):
    C1, C2, p, q, M = rectangular
    C1[0, 1], C2[2, 1] = 0.5, 0.4
    expected = reference_value(C1, C2, p, q, M, 1.0)
    result = solve_gw_sdp(C1, C2, p, q, solver="rinnal", tol=1e-6)
    check_solution(C1, C2, p, q, result)
    assert result[2] == pytest.approx(expected, abs=1e-5)


def test_diagnostics_and_approximation_gap(rectangular):
    from gwsdp import approximation_gap, solve_rinnal

    C1, C2, p, q, M = rectangular
    tol = 1e-6
    pi, P, value, info = solve_rinnal(C1, C2, p, q, tol=tol, return_info=True)
    optimum = reference_value(C1, C2, p, q, M, 1.0)
    assert info["converged"]
    assert info["primal_residual"] <= tol
    assert info["lower_bound"] <= optimum + 1e-7
    assert abs(value - info["lower_bound"]) / (1 + abs(value)) <= tol
    loss, _ = lifted_data(C1, C2)
    x = pi.ravel(order="F")
    assert approximation_gap(C1, C2, P, pi) == pytest.approx((x @ loss @ x) / value)


def test_existing_scs_entry_points(rectangular):
    C1, C2, p, q, M = rectangular
    gw = solve_gw_sdp(C1, C2, p, q, tol=1e-6)
    assert gw[2] == pytest.approx(reference_value(C1, C2, p, q, M, 1.0), abs=1e-5)
    alpha = 0.4
    fused = solve_fused_gw_sdp(M, C1, C2, p, q, alpha=alpha, solver="scs", tol=1e-6)
    assert fused[2] == pytest.approx(reference_value(C1, C2, p, q, M, alpha), abs=1e-5)


@pytest.mark.parametrize("p", [[0.6, 0.0, 0.4], [1.0]])
def test_zero_marginals_and_singletons(p):
    p, q = np.array(p), np.array([0.3, 0.7, 0.0])
    C1 = np.abs(np.subtract.outer(np.arange(len(p)), np.arange(len(p))))
    C2 = np.abs(np.subtract.outer([0, 1, 3], [0, 1, 3]))
    expected = reference_value(C1, C2, p, q, np.zeros((len(p), len(q))), 1.0)
    result = solve_gw_sdp(C1, C2, p, q, solver="rinnal", tol=1e-6)
    check_solution(C1, C2, p, q, result)
    assert result[2] == pytest.approx(expected, abs=1e-5)


def test_iteration_limit_warns(rectangular):
    C1, C2, p, q, _ = rectangular
    with pytest.warns(RuntimeWarning):
        result = solve_gw_sdp(C1, C2, p, q, solver="rinnal", max_iters=1, tol=1e-12)
    assert np.isfinite(result[2])


@pytest.mark.parametrize("p", [[1.0], [0.4, 0.4], [1.1, -0.1]])
def test_invalid_marginals(rectangular, p):
    C1, C2, _, q, _ = rectangular
    with pytest.raises(ValueError):
        solve_gw_sdp(C1, C2, np.array(p), q, solver="rinnal")
