"""Lifted projected-gradient correction for the continuous RiNNAL+ subproblem."""

import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import (
    ArpackError,
    LinearOperator,
    aslinearoperator,
    eigsh,
    lobpcg,
)


def project_factor(trial, first_norm):
    """Factor the Frobenius projection onto {S >= 0 (PSD), S[0, 0] = c²}.

    The constraint multiplier changes only the (0, 0) entry before the PSD
    projection. The corrected factor has first row (c, 0, ...), the gauge
    used by the subsequent low-rank minimization.
    """
    target = first_norm**2
    shifted = (trial + trial.T) * 0.5
    # All feasible points have the same first entry, so trial[0, 0] has no
    # effect on the projection. At shift zero the PSD part has entry >= c².
    shifted[0, 0] = target

    def eigensystem(shift):
        shifted[0, 0] = target - shift
        return np.linalg.eigh(shifted)

    def residual(shift):
        values, vectors = eigensystem(shift)
        return (vectors[0] ** 2) @ np.maximum(values, 0.0) - target

    values, vectors = eigensystem(0.0)
    eps = np.finfo(float).eps
    error = (vectors[0] ** 2) @ np.maximum(values, 0.0) - target
    if error > 8 * eps * target:
        # The first entry of the PSD part decreases to zero as the shift
        # grows. Only a scalar normalization must be solved on this face.
        upper = max(target, error, np.linalg.norm(shifted[1:, 0]))
        while residual(upper) > 0:
            upper *= 2.0
        shift = brentq(residual, 0.0, upper, xtol=4 * eps * target)
        values, vectors = eigensystem(shift)

    cutoff = len(values) * eps * values[-1]
    keep = (values > cutoff) | (vectors[0] ** 2 * values > eps * target)
    factor = vectors[:, keep] * np.sqrt(values[keep])

    # A Householder reflection sends the first row to a coordinate axis
    # without changing the Gram matrix. Choose its sign to avoid cancellation.
    axis = factor[0].copy()
    sign = 1.0 if axis[0] >= 0 else -1.0
    axis[0] += sign * np.linalg.norm(axis)
    axis /= np.linalg.norm(axis)
    factor -= 2.0 * np.outer(factor @ axis, axis)
    factor[:, 0] *= -sign
    factor[0] = 0.0
    factor[0, 0] = first_norm
    return factor


def least_eigenpair(operator, tol, v0=None):
    """Estimate the lowest eigenpair of a symmetric face operator by Lanczos.

    The Rayleigh quotient can identify negative curvature. A positive Ritz
    value alone does not certify that the full operator is positive semidefinite.
    """
    operator = aslinearoperator(operator)
    n = operator.shape[0]
    if n == 1:
        vector = np.ones(1)
        return float(operator.matvec(vector)[0]), vector

    noise = np.random.default_rng(0).standard_normal(n)
    noise /= np.linalg.norm(noise)
    if v0 is None:
        start = noise
    else:
        # Keep a component outside the previous Ritz space when the operator
        # changes and a different eigenspace becomes the leftmost one.
        start = v0 + 0.01 * noise
        start /= np.linalg.norm(start)

    # An identity shift leaves the eigenvectors unchanged and provides an
    # absolute tolerance near zero for ARPACK's relative stopping criterion.
    shifted = LinearOperator(
        (n, n), matvec=lambda x: operator.matvec(x) + x, dtype=float
    )
    try:
        _, vectors = eigsh(
            shifted, k=1, which="SA", v0=start, tol=tol, ncv=min(n, 20), maxiter=18
        )
    except ArpackError:
        # LOBPCG provides a Rayleigh direction even at its iteration limit.
        # The caller independently verifies an apparent convergence.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"scipy\.")
            _, vectors = lobpcg(
                operator, start[:, None], largest=False, tol=tol, maxiter=40
            )
    vector = vectors[:, 0]
    vector /= np.linalg.norm(vector)
    return float(vector @ operator.matvec(vector)), vector


def append_negative_direction(factor, direction, gradient_product, penalty):
    """Take a feasible rank-increasing RiNNAL step for a negative Ritz direction."""
    # Eliminate the intercept using a current factor column. At fixed-rank
    # stationarity the dual slack annihilates that column.
    tangent = direction - (direction[0] / factor[0, 0]) * factor[:, 0]
    length = np.linalg.norm(tangent)
    if length <= np.finfo(float).eps * np.linalg.norm(direction):
        return factor
    tangent /= length
    curvature = float(tangent @ gradient_product(tangent))
    if curvature >= 0:
        return factor
    # Along S+t vvᵀ, the AL is bounded above by its value at S plus
    # t vᵀ∇f(S)v + penalty*t²/2, since the face basis is orthonormal.
    return np.column_stack((factor, np.sqrt(-curvature / penalty) * tangent))
