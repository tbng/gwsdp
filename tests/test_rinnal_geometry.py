"""Geometry and dual certificates used by the GW-specialized RiNNAL solver."""

import numpy as np
import pytest

from gwsdp import solve_rinnal
from gwsdp._rinnal_bounds import refine_profile_multiplier
from gwsdp._rinnal_face import TransportFace
from gwsdp._rinnal_profile import profile_multiplier
from gwsdp._rinnal_projection import project_factor


pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def test_transport_face_isometry_adjoint_and_vector_order():
    p = np.array([0.6, 0.25, 0.15])
    q = np.array([0.1, 0.4, 0.3, 0.2])
    rng = np.random.default_rng(4)
    face = TransportFace(p, q)
    coords = rng.normal(size=(face.dimension, 5))
    probe = rng.normal(size=(1 + p.size * q.size, 5))
    full = face.expand(coords)
    np.testing.assert_allclose(face.reduce(full), coords, atol=1e-13)
    assert np.vdot(full, probe) == pytest.approx(np.vdot(coords, face.reduce(probe)))
    for column in full.T:
        plan = column[1:].reshape((p.size, q.size), order="F")
        np.testing.assert_allclose(plan.sum(axis=1), column[0] * p, atol=1e-13)
        np.testing.assert_allclose(plan.sum(axis=0), column[0] * q, atol=1e-13)


def test_projection_onto_psd_with_fixed_first_entry():
    # Construct an instance with a known exact projection from its KKT conditions.
    factor = np.array([[1.5, 0.0], [1.0, 1.0], [0.2, -0.5]])
    optimum = factor @ factor.T
    null = np.cross(factor[:, 0], factor[:, 1])
    trial = optimum - 2 * np.outer(null, null)
    trial[0, 0] += 4.0
    actual = project_factor(trial, factor[0, 0])
    np.testing.assert_allclose(actual[0], [1.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(actual @ actual.T, optimum, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.6, 1.0])
def test_profile_dual_identity_on_transport_face(alpha):
    n, scale = 4, 2.7
    p = np.full(n, 0.65)  # total mass 2.6
    rng = np.random.default_rng(2)
    C1, C2 = [(A + A.T) / 2 for A in rng.normal(size=(2, n, n))]
    feature = rng.normal(size=(n, n))
    source, target = np.tile(np.arange(n), n), np.repeat(np.arange(n), n)
    loss = (C1[np.ix_(source, source)] - C2[np.ix_(target, target)]) ** 2

    plan = p[0] * np.eye(n)[:, rng.permutation(n)]
    x = plan.ravel(order="F")
    column = np.r_[1.0, x]
    rank_one = np.outer(column, column)
    U = rng.normal(size=(n, n))
    U -= U.mean(axis=0, keepdims=True)
    U -= U.mean(axis=1, keepdims=True)
    tangent = np.r_[0.0, U.ravel(order="F")]
    rank_two = rank_one + np.outer(tangent, tangent)

    W, bound = profile_multiplier(C1, C2, p, p, feature, alpha, scale)
    refined, stronger = refine_profile_multiplier(W.copy(), bound, p, scale, sweeps=2)
    assert stronger >= bound - 1e-12
    if alpha > 0:
        assert stronger > bound + 0.1
    for multiplier, lower in ((W, bound), (refined, stronger)):
        assert multiplier.min() >= -1e-12
        np.testing.assert_allclose(multiplier.T, multiplier, atol=1e-12)
        np.testing.assert_allclose(multiplier[0], 0, atol=1e-12)
        for Y in (rank_one, rank_two):
            objective = alpha * np.sum(loss * Y[1:, 1:]) + (1 - alpha) * np.sum(
                feature * plan
            )
            assert objective - scale * np.sum(multiplier * Y) == pytest.approx(
                lower, abs=1e-10
            )


def test_large_face_rank_escape_with_replicated_targets():
    # Equal mass on each of two target classes gives the 2x2 analytic optimum.
    # Replication retains the optimum and increases the face dimension to 132.
    C = np.ones((2, 2)) - np.eye(2)
    labels = np.arange(132) % 2
    D = 2 * C[np.ix_(labels, labels)]
    p = np.full(2, 0.5)
    q = np.full(len(labels), 1 / len(labels))
    pi, P, value, info = solve_rinnal(
        C, D, p, q, tol=1e-5, max_iters=100, return_info=True
    )
    assert info["converged"]
    assert info["iterations"] > 1
    assert value == pytest.approx(0.5, abs=1e-5)
    assert info["lower_bound"] <= 0.5 + 1e-8
    assert P.min() >= -1e-5
    np.testing.assert_allclose(pi.sum(axis=1), p, atol=1e-10)
    np.testing.assert_allclose(pi.sum(axis=0), q, atol=1e-10)


def test_square_warm_start_returns_a_matching_permutation_and_bounds():
    # After relabeling, every structural difference is 0.5. Its mean is fixed
    # by the marginals, so Jensen gives the matching SDP lower bound of 0.25.
    n = 13
    rng = np.random.default_rng(6)
    perm = rng.permutation(n)
    A = rng.normal(size=(n, n))
    C = (A + A.T) / 2
    D = C[np.ix_(perm, perm)] + 0.5
    p = np.full(n, 1 / n)
    pi, P, value, info = solve_rinnal(C, D, p, p, tol=1e-7, return_info=True)
    np.testing.assert_allclose(pi[perm, np.arange(n)], p, atol=1e-9)
    x = pi.ravel(order="F")
    np.testing.assert_allclose(P, np.outer(x, x), atol=1e-9)
    assert info["converged"]
    assert info["lower_bound"] == pytest.approx(0.25, abs=1e-7)
    assert info["upper_bound"] == pytest.approx(0.25, abs=1e-7)
    assert value == pytest.approx(0.25, abs=1e-7)


def test_float32_marginals_with_different_rounded_totals():
    p = np.full(3, 1 / 3, dtype=np.float32)
    q = np.full(2, 0.5, dtype=np.float32)
    assert p.astype(float).sum() != q.astype(float).sum()
    pi, _, value = solve_rinnal(np.zeros((3, 3)), np.zeros((2, 2)), p, q)
    np.testing.assert_allclose(pi.sum(axis=1), p, atol=1e-12)
    np.testing.assert_allclose(pi.sum(axis=0), q, atol=1e-7)
    assert value == 0.0


@pytest.mark.parametrize("n", [2, 13])
def test_close_large_distances_preserve_lifted_cost_and_bounds(n):
    # Both off-diagonal events have probability (n-1)/n. A permutation aligns
    # them, attaining the minimum loss of 1 on each shared off-diagonal pair.
    expected = (n - 1) / n
    off_diagonal = 1 - np.eye(n)
    C = 1e8 * off_diagonal
    D = (1e8 + 1) * off_diagonal
    p = np.full(n, 1 / n)
    pi, P, value, info = solve_rinnal(
        C, D, p, p, tol=1e-7, max_iters=20, return_info=True
    )
    assert info["converged"]
    assert 0 <= info["lower_bound"] <= expected + 1e-8
    assert info["upper_bound"] == pytest.approx(expected, abs=1e-7, rel=0)
    assert value == pytest.approx(expected, abs=1e-7, rel=0)
    i, j = np.tile(np.arange(n), n), np.repeat(np.arange(n), n)
    loss = (C[np.ix_(i, i)] - D[np.ix_(j, j)]) ** 2
    # Independent evaluation subtracts distances before squaring.
    assert np.sum(loss * P) == pytest.approx(value, abs=1e-7, rel=0)
    np.testing.assert_allclose(pi.sum(axis=0), p, atol=1e-12)
    np.testing.assert_allclose(pi.sum(axis=1), p, atol=1e-12)
