"""
Residual-shell upper bound on the Wasserstein-1 distance.

Implements the bucket-based residual matching algorithm from the
ORC-bound paper for computing an upper bound on W_1(mu, nu).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, NamedTuple


class UpperBoundResult(NamedTuple):
    """Result of the residual-shell W1 upper bound computation."""

    ub: float
    """Upper bound on W_1(mu, nu)."""

    m_r: np.ndarray
    """Mass assigned to each distance bucket r."""

    residual_mass: float
    """Remaining unmatched mass R_l."""

    rbar: float
    """Effective distance for the residual mass."""


def residual_shell_upper_bound(
    mu_x: Dict[int, float],
    mu_y: Dict[int, float],
    D: np.ndarray,
    idx: Dict[int, int],
    l: int = 2,
    tol: float = 1e-12,
    rbar_mode: str = "local-max",
) -> Tuple[float, np.ndarray, float, float]:
    """
    Compute the residual-shell upper bound on W_1(mu_x, mu_y).

    The algorithm performs bucket-based matching between the supports
    of the two measures, grouping pairs by their graph distance.

    Parameters
    ----------
    mu_x : Dict[int, float]
        Measure at node x, as a dict of {node: mass}.
    mu_y : Dict[int, float]
        Measure at node y, as a dict of {node: mass}.
    D : np.ndarray
        Precomputed distance matrix from :func:`all_pairs_shortest_path_matrix_cutoff`.
    idx : Dict[int, int]
        Node-to-index mapping.
    l : int, default=2
        Maximum shell distance to consider. Pairs at distance > l
        are assigned to the residual shell.
    tol : float, default=1e-12
        Pruning threshold. Masses below this are ignored.
    rbar_mode : str, default="local-max"
        How to compute the residual distance rbar:
        - ``"local-max"``: max distance between residual supports.
        - ``"global"``: max finite distance in the matrix.

    Returns
    -------
    ub : float
        Upper bound on the 1-Wasserstein distance.
    m_r : np.ndarray
        Mass assigned to each bucket r=0..l.
    residual_mass : float
        Remaining unmatched mass.
    rbar : float
        Effective residual distance.

    Raises
    ------
    RuntimeError
        If no finite residual distances are found (cutoff too small).

    Examples
    --------
    >>> import numpy as np
    >>> D = np.array([[0., 1., 2.],
    ...               [1., 0., 1.],
    ...               [2., 1., 0.]], dtype=float)
    >>> idx = {0: 0, 1: 1, 2: 2}
    >>> mu_x = {0: 0.5, 1: 0.5}
    >>> mu_y = {1: 0.5, 2: 0.5}
    >>> ub, m_r, Rl, rbar = residual_shell_upper_bound(mu_x, mu_y, D, idx, l=2)
    >>> ub >= 0
    True
    """
    # Filter by tolerance
    U = [u for u, m in mu_x.items() if m > tol]
    W = [v for v, m in mu_y.items() if m > tol]

    a = {u: mu_x[u] for u in U}
    b = {v: mu_y[v] for v in W}

    # Build distance buckets
    buckets = [[] for _ in range(l + 1)]
    for u in U:
        iu = idx[u]
        for v in W:
            d = D[iu, idx[v]]
            if np.isfinite(d) and d <= l:
                buckets[int(d)].append((u, v))

    m_r = np.zeros(l + 1, dtype=np.float64)

    # Residual matching within each bucket
    for r in range(l + 1):
        for (u, v) in buckets[r]:
            delta = min(a[u], b[v])
            if delta > tol:
                a[u] -= delta
                b[v] -= delta
                m_r[r] += delta

    # Remaining residual mass
    Rl = sum(a.values())

    if Rl <= tol:
        rbar = 0.0
    else:
        if rbar_mode == "local-max":
            RU = [u for u in U if a[u] > tol]
            RV = [v for v in W if b[v] > tol]

            finite_residual_distances = [
                D[idx[u], idx[v]]
                for u in RU for v in RV
                if np.isfinite(D[idx[u], idx[v]])
            ]

            if len(finite_residual_distances) == 0:
                raise RuntimeError(
                    "No finite residual distances found. "
                    "The shortest-path cutoff may be too small."
                )

            rbar = max(finite_residual_distances)
        else:
            rbar = float(np.max(D[np.isfinite(D)]))

    ub = float(np.dot(np.arange(l + 1), m_r) + rbar * Rl)
    return ub, m_r, Rl, rbar
