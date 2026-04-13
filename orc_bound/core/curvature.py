"""
Residual-shell Ricci curvature approximation.

Computes lower bounds on Ollivier-Ricci Curvature (ORC) using the residual-shell
Wasserstein-1 upper bound with k-hop lazy random walk measures.

The main entry point is :func:`residual_shell_ricci_approximation`.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

from orc_bound.core.measures import build_lazy_measures_k
from orc_bound.core.distance import all_pairs_shortest_path_matrix_cutoff
from orc_bound.core.upper_bound import residual_shell_upper_bound
from orc_bound.utils.parallel import _cpu_count


def _compute_edge_curvature(
    u: int,
    v: int,
    mu: dict,
    D: np.ndarray,
    idx: dict,
    l_eff: int,
    tol: float,
    rbar_mode: str,
) -> Tuple[int, int, float]:
    """
    Compute curvature lower bound for a single edge (worker function).

    Parameters
    ----------
    u, v : int
        Edge endpoints.
    mu : dict
        Node-to-measure dict from build_lazy_measures_k.
    D, idx, l_eff, tol, rbar_mode
        As in residual_shell_upper_bound.

    Returns
    -------
    iu, iv, kappa_lb : tuple
        Row index, column index, and curvature value.
    """
    ub_w1, _, _, _ = residual_shell_upper_bound(
        mu[u], mu[v], D, idx,
        l=l_eff, tol=tol, rbar_mode=rbar_mode
    )

    d_uv = D[idx[u], idx[v]]
    if d_uv <= 0 or not np.isfinite(d_uv):
        return idx[u], idx[v], 0.0  # shouldn't happen for edges

    kappa_lb = 1.0 - ub_w1 / d_uv
    return idx[u], idx[v], float(kappa_lb)


def residual_shell_ricci_approximation(
    graph: nx.Graph,
    num_nodes: int,
    k: int = 1,
    alpha_lazy: float = 0.0,
    l_shell: int = 3,
    rbar_mode: str = "local-max",
    tol: float = 1e-12,
    symmetric: bool = False,
    n_jobs: Optional[int] = None,
) -> csr_matrix:
    """
    Compute lower bounds on Ollivier-Ricci Curvature (ORC) via residual-shell W1 upper bounds.

    The ORC lower bound for an edge (u, v) is:

        kappa_lb(u, v) = 1 - W_1(mu_u, mu_v) / dist(u, v)

    where mu_u is the k-hop lazy random walk measure at u and W_1 is
    approximated using the residual-shell bucket matching algorithm.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.
    num_nodes : int
        Number of nodes. Must match ``len(graph.nodes())``.
    k : int, default=1
        Number of random walk steps (k-hop neighborhood).
        Higher k gives a more global measure.
    alpha_lazy : float, default=0.0
        Lazy mixing parameter in [0, 1]. Controls the weight of the
        stationary distribution.
    l_shell : int, default=3
        Maximum shell distance for bucket matching. The effective l
        is ``min(l_shell, 2*k+1)``.
    rbar_mode : str, default="local-max"
        Method for residual distance estimation. Options: ``"local-max"``,
        ``"global"``.
    tol : float, default=1e-12
        Numerical tolerance for mass pruning.
    symmetric : bool, default=False
        If True, populate both (u,v) and (v,u) entries.
    n_jobs : int or None, default=None
        Number of parallel workers for edge curvature computation.
        - ``None`` or ``0``: use all available CPU cores.
        - Positive int: use exactly that many workers.
        - ``-1``: use all cores minus one.

    Returns
    -------
    curvature_matrix : scipy.sparse.csr_matrix
        Sparse matrix of shape (num_nodes, num_nodes). Entry (i, j) is the
        ORC lower bound for the edge from node i to node j. Non-edge entries
        are zero.

    Notes
    -----
    The distance cutoff is automatically set to ``2*k + 1``, which is the
    maximum possible distance between any two points in the supports of
    mu_u^(k) and mu_v^(k) when both are centered at u and v respectively.

    Multi-threading: Each edge's curvature is computed independently, so
    the computation parallelizes naturally across edges.

    Examples
    --------
    >>> import networkx as nx
    >>> from orc_bound import residual_shell_ricci_approximation
    >>> G = nx.watts_strogatz_graph(80, 6, 0.2, seed=0)
    >>> C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=2)
    >>> C.nnz  # number of non-zero entries (edges)
    240

    Parallel execution:

    >>> C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=5, n_jobs=8)
    """
    dist_cutoff = 2 * k + 1
    l_eff = min(l_shell, dist_cutoff)

    if num_nodes != len(graph.nodes()):
        raise ValueError(
            f"num_nodes={num_nodes} does not match graph node count "
            f"={len(graph.nodes())}."
        )

    # Precompute distance matrix (single-threaded, but fast)
    nodes, idx, D = all_pairs_shortest_path_matrix_cutoff(graph, cutoff=dist_cutoff)

    # Build k-hop measures (single-threaded, uses BLAS internally)
    mu = build_lazy_measures_k(graph, alpha_lazy, k)

    # Determine number of workers
    n_workers = _cpu_count(n_jobs)

    edges = list(graph.edges())

    if n_workers <= 1:
        # Sequential fallback
        row, col, data = [], [], []
        for u, v in edges:
            iu, iv, klb = _compute_edge_curvature(
                u, v, mu, D, idx, l_eff, tol, rbar_mode
            )
            row.append(iu)
            col.append(iv)
            data.append(klb)
            if symmetric:
                row.append(iv)
                col.append(iu)
                data.append(klb)
    else:
        # Parallel edge processing
        row, col, data = [], [], []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _compute_edge_curvature,
                    u, v, mu, D, idx, l_eff, tol, rbar_mode
                ): (u, v)
                for u, v in edges
            }
            for future in as_completed(futures):
                iu, iv, klb = future.result()
                row.append(iu)
                col.append(iv)
                data.append(klb)
                if symmetric:
                    row.append(iv)
                    col.append(iu)
                    data.append(klb)

    curvature_matrix = csr_matrix(
        (data, (row, col)),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    )
    return curvature_matrix
