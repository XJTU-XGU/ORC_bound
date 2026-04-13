"""
ORC-Bound: Ollivier-Ricci Curvature Lower Bounds via Residual-Shell Measures

A Python package for computing lower bounds on Ollivier-Ricci Curvature (ORC)
using k-hop lazy random walk measures and residual-shell Wasserstein-1 upper bounds.

Features
--------
- Residual-shell Ricci curvature approximation with k-hop support
- Parallel edge processing via multi-threading
- Truncated all-pairs shortest path computation
- Sparse matrix output for memory efficiency

Example
-------
>>> import networkx as nx
>>> from orc_bound import residual_shell_ricci_approximation
>>> G = nx.watts_strogatz_graph(100, 6, 0.2, seed=0)
>>> C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=2, n_jobs=4)
"""

__version__ = "0.1.0"
__author__ = "ORC-Bound Team"

from orc_bound.core.measures import build_lazy_measures_k
from orc_bound.core.distance import all_pairs_shortest_path_matrix_cutoff
from orc_bound.core.upper_bound import residual_shell_upper_bound
from orc_bound.core.curvature import residual_shell_ricci_approximation

__all__ = [
    "build_lazy_measures_k",
    "all_pairs_shortest_path_matrix_cutoff",
    "residual_shell_upper_bound",
    "residual_shell_ricci_approximation",
]
