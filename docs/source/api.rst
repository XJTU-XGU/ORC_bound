API Reference
=============

Main Function
------------

.. autofunction:: orc_bound.residual_shell_ricci_approximation

Core Functions
--------------

.. autofunction:: orc_bound.build_lazy_measures_k

.. autofunction:: orc_bound.all_pairs_shortest_path_matrix_cutoff

.. autofunction:: orc_bound.residual_shell_upper_bound

Return Type
-----------

All functions that compute a curvature matrix return a
``scipy.sparse.csr_matrix`` of shape ``(num_nodes, num_nodes)``.
Non-zero entries correspond to graph edges; their values are the
ORC lower bounds.

.. code-block:: python

    from scipy.sparse import csr_matrix
    C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=2)

    # Average curvature over all edges
    avg_curv = C.sum() / C.nnz

    # Per-edge curvature: iterate over non-zero entries
    for i, j, v in zip(C.row, C.col, C.data):
        print(f"Edge ({i}, {j}): κ_lb = {v:.4f}")

    # Access a specific edge
    i, j = 0, 1
    kappa_ij = C[i, j]

Utility Functions
-----------------

.. autofunction:: orc_bound.utils.parallel._cpu_count

.. autofunction:: orc_bound.utils.search.search_orc_literature

.. autofunction:: orc_bound.utils.search.search_general

Parameter Reference
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``graph``
     - —
     - Input NetworkX graph. Must be unweighted.
   * - ``num_nodes``
     - —
     - Number of nodes. Must equal ``len(graph.nodes())``.
   * - ``k``
     - 1
     - Number of random walk steps (k-hop neighborhood). Higher k gives more global measure.
   * - ``alpha_lazy``
     - 0.0
     - Lazy mixing parameter in [0, 1]. ``0`` = pure random walk; ``1`` = trivial delta measure.
   * - ``l_shell``
     - 3
     - Maximum shell distance for bucket matching. Effective value is ``min(l_shell, 2*k+1)``.
   * - ``rbar_mode``
     - ``"local-max"``
     - Residual distance mode: ``"local-max"`` or ``"global"``.
   * - ``tol``
     - 1e-12
     - Pruning threshold. Masses below this are ignored.
   * - ``symmetric``
     - False
     - If True, populate both (u,v) and (v,u) entries.
   * - ``n_jobs``
     - None
     - Number of parallel workers. ``None/0`` = all cores; ``-1`` = all minus one.
