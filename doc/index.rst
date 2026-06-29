ORC-Bound Documentation
=======================

.. image:: https://img.shields.io/badge/version-0.2.2-blue.svg
   :target: https://github.com/XJTU-XGU/ORC_bound
   :alt: version 0.2.2

``orc_bound`` computes lower bounds for Ollivier-Ricci curvature on graph
edges. The current package centers on the C++/OpenMP accelerated
``residual_shell_based_orc_bound`` helper, which builds k-hop lazy random-walk
measures, applies the residual-shell Wasserstein-1 upper bound, and returns
curvature arrays plus a sparse matrix for the requested edges.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   algorithm
   api
   examples
   release_notes

Quick Example
-------------

.. code-block:: python

    import networkx as nx
    from orc_bound import residual_shell_based_orc_bound

    graph = nx.karate_club_graph()
    edges = list(graph.edges())

    result = residual_shell_based_orc_bound(
        graph,
        k_hop=1,
        alpha_lazy=0.4,
        l_shell=2,
        num_threads=4,
    )

    edge_to_curvature = dict(zip(edges, result["curvatures"]))
    print(edge_to_curvature[edges[0]])

What Changed From the Older API
-------------------------------

Older documentation used ``residual_shell_ricci_approximation`` and described
a direct sparse-matrix return value. The current C++ package uses
``residual_shell_based_orc_bound`` and returns a dictionary. The most commonly
used fields are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``curvatures``
     - One curvature lower-bound value per computed edge.
   * - ``w1_upper_bounds``
     - Residual-shell upper bounds on Wasserstein-1 distance.
   * - ``curvature_matrix``
     - Symmetric SciPy CSR matrix populated on computed undirected edges.
   * - ``edge_indices``
     - Integer edge array using the internal contiguous node order.
   * - ``nodes`` and ``node_index``
     - Mapping between original NetworkX node labels and internal indices.
   * - ``distances``
     - Dense unweighted hop-distance matrix used by the residual-shell kernel.

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
