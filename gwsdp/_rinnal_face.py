"""Orthonormal coordinates for the face fixed by the transport marginals."""

import numpy as np
from scipy.linalg import helmert


class TransportFace:
    """Apply the face basis and its transpose without forming a Kronecker matrix.

    Each column of a lifted factor has bottom block vec_F(T). Its row and
    column sums equal its first entry times p and q. The minimum-norm T for
    first entry one is orthogonal to every matrix with zero marginal sums.
    """

    def __init__(self, p, q):
        self.m, self.n = len(p), len(q)
        origin = p[:, None] / self.n + q[None, :] / self.m - p.sum() / (self.m * self.n)
        self.origin = origin.ravel(order="F")
        self.first_norm = np.sqrt(1.0 + self.origin @ self.origin)
        self.dimension = 1 + (self.m - 1) * (self.n - 1)
        self._source = helmert(self.m)
        self._target = helmert(self.n)

    def expand(self, reduced):
        """Return B @ reduced, including the first (homogenizing) row."""
        k = reduced.shape[1]
        top = reduced[:1] / self.first_norm
        # B's remaining columns are kron(H_target.T, H_source.T). Leading
        # batch dimension keeps columns independent and source index fastest.
        directions = reduced[1:].T.reshape(k, self.n - 1, self.m - 1)
        matrices = self._target.T @ directions @ self._source
        bottom = matrices.reshape(k, self.n * self.m).T + self.origin[:, None] * top
        return np.vstack((top, bottom))

    def reduce(self, full):
        """Return B.T @ full. Reduce and expand are adjoints."""
        k = full.shape[1]
        top = (full[0] + self.origin @ full[1:]) / self.first_norm
        matrices = full[1:].T.reshape(k, self.n, self.m)
        directions = self._target @ matrices @ self._source.T
        return np.vstack((top, directions.reshape(k, self.dimension - 1).T))
