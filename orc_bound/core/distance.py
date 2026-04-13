"""
Truncated all-pairs shortest path matrix.

Computes pairwise shortest path distances up to a given cutoff,
which avoids full APSP on large graphs.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple


def all_pairs_shortest_path_matrix_cutoff(
    G: nx.Graph,
    cutoff: int,
) -> Tuple[List[int], Dict[int, int], np.ndarray]:
    """
    Compute the all-pairs shortest path distance matrix up to a cutoff.

    Distances larger than the cutoff are stored as ``np.inf``.
    This is significantly faster than full APSP when the cutoff is small
    relative to the graph diameter.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    cutoff : int
        Maximum distance to compute. Distances beyond this are ``inf``.

    Returns
    -------
    nodes : List[int]
        List of node IDs in the same order as the index map.
    idx : Dict[int, int]
        Mapping from node ID to matrix row/column index.
    D : np.ndarray
        Distance matrix of shape (n, n) where n = |nodes|.
        ``D[i, j]`` is the shortest path distance from nodes[i] to nodes[j],
        or ``np.inf`` if the distance exceeds the cutoff.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)
    >>> nodes, idx, D = all_pairs_shortest_path_matrix_cutoff(G, cutoff=3)
    >>> D[idx[0], idx[4]]
    4.0
    >>> D[idx[0], idx[2]]
    2.0
    """
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for u in nodes:
        iu = idx[u]
        dmap = nx.single_source_shortest_path_length(G, u, cutoff=cutoff)
        for v, d in dmap.items():
            D[iu, idx[v]] = float(d)

    return nodes, idx, D
