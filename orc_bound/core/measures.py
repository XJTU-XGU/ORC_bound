"""
k-hop lazy random walk measures.

mu_x^(k) = alpha * delta_x + (1 - alpha) * (k-step random walk distribution)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Dict


def build_lazy_measures_k(
    G: nx.Graph,
    alpha_lazy: float,
    k: int,
) -> Dict[int, Dict[int, float]]:
    """
    Build k-hop lazy random walk probability measures for all nodes.

    For each node x, the measure is:
        mu_x^(k)[y] = alpha_lazy * delta_x(y) + (1 - alpha_lazy) * P^k(x, y)

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    alpha_lazy : float
        Lazy mixing parameter in [0, 1]. Higher values weight the
        stationary distribution more heavily.
    k : int
        Number of random walk steps (k-hop neighborhood).

    Returns
    -------
    mu_dict : Dict[int, Dict[int, float]]
        Dictionary mapping each node to its mass dictionary, where
        keys are neighbor nodes and values are probabilities.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)
    >>> mu = build_lazy_measures_k(G, alpha_lazy=0.0, k=2)
    >>> abs(sum(mu[0].values()) - 1.0) < 1e-10
    True
    """
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    # Build transition matrix P
    P = np.zeros((n, n), dtype=np.float64)
    for u in nodes:
        i = idx[u]
        nbrs = list(G.neighbors(u))
        if len(nbrs) == 0:
            P[i, i] = 1.0
        else:
            p = 1.0 / len(nbrs)
            for v in nbrs:
                P[i, idx[v]] = p

    # Compute P^k via matrix power
    Pk = np.linalg.matrix_power(P, k)

    mu_dict: Dict[int, Dict[int, float]] = {}
    for u in nodes:
        i = idx[u]
        d: Dict[int, float] = {}

        # Delta mass at u
        d[u] = alpha_lazy

        # k-step random walk mass
        for v in nodes:
            mass = (1.0 - alpha_lazy) * Pk[i, idx[v]]
            if mass > 1e-14:
                d[v] = d.get(v, 0.0) + mass

        mu_dict[u] = d

    return mu_dict
