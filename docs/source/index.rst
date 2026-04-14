ORC-Bound Documentation
========================

.. image:: https://img.shields.io/badge/version-0.1.1-blue.svg
   :target: https://github.com/XJTU-XGU/ORC_bound

**ORC-Bound** provides lower bound algorithms for Ollivier–Ricci Curvature (ORC)
on graph edges, using residual-shell Wasserstein-1 upper bounds with k-hop lazy
random walk measures.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   algorithm
   api

Quick Start
-----------

.. code-block:: python

    import networkx as nx
    from orc_bound import residual_shell_ricci_approximation

    G = nx.watts_strogatz_graph(200, 6, 0.2, seed=0)
    C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=2, n_jobs=4)
    print(f"Average curvature: {C.sum() / C.nnz:.4f}")

Basic Usage
-----------

Compute curvature for all edges of a graph and inspect the results:

.. code-block:: python

    import networkx as nx
    from orc_bound import residual_shell_ricci_approximation

    G = nx.karate_club_graph()
    C = residual_shell_ricci_approximation(G, G.number_of_nodes())

    # Average curvature over all edges
    print(f"Average: {C.sum() / C.nnz:.4f}")

    # All edge curvatures
    for u, v in G.edges():
        print(f"Edge ({u}, {v}): {C[u, v]:.4f}")

Choosing k — Number of Random Walk Steps
-----------------------------------------

The parameter ``k`` controls how "global" the random walk measure is.
Larger ``k`` spreads mass further from the source node.

.. code-block:: python

    # k=1: local measure (neighbors only)
    C1 = residual_shell_ricci_approximation(G, n, k=1)

    # k=5: semi-global measure
    C5 = residual_shell_ricci_approximation(G, n, k=5)

    # k=20: near-uniform measure
    C20 = residual_shell_ricci_approximation(G, n, k=20)

Lazy Mixing Parameter alpha
----------------------------

The ``alpha_lazy`` parameter controls the weight of the Dirac delta at the source node.
Higher alpha keeps more mass locally.

.. code-block:: python

    # alpha=0: pure random walk (mass spreads fully)
    C0 = residual_shell_ricci_approximation(G, n, k=3, alpha_lazy=0.0)

    # alpha=0.5: balanced
    C5 = residual_shell_ricci_approximation(G, n, k=3, alpha_lazy=0.5)

    # alpha=1: trivial delta measure (kappa = 0 for all edges)
    C1 = residual_shell_ricci_approximation(G, n, k=3, alpha_lazy=1.0)

Multi-Threaded Parallel Computation
-------------------------------------

The most expensive step — computing curvature for each edge — is fully parallelizable.
Use ``n_jobs`` to control the number of worker threads.

.. code-block:: python

    # Use all available CPU cores (default)
    C = residual_shell_ricci_approximation(G, n, k=10)

    # Use exactly 8 threads
    C = residual_shell_ricci_approximation(G, n, k=10, n_jobs=8)

    # Use all but one core
    C = residual_shell_ricci_approximation(G, n, k=10, n_jobs=-1)

Interpreting the Result Matrix
------------------------------

The return value is a ``scipy.sparse.csr_matrix``. Non-zero entries correspond to graph edges.

.. code-block:: python

    # Basic statistics
    print(f"Shape: {C.shape}")           # (n, n)
    print(f"Non-zeros: {C.nnz}")        # equals number of edges
    print(f"Average: {C.sum()/C.nnz:.4f}")
    print(f"Min: {C.data.min():.4f}")
    print(f"Max: {C.data.max():.4f}")

    # Get all non-zero entries
    for i in range(C.nnz):
        u, v = C.row[i], C.col[i]
        print(f"Edge ({u}, {v}): {C.data[i]:.4f}")

    # Extract as dense array
    dense = C.toarray()

Different Graph Types
---------------------

.. code-block:: python

    import networkx as nx
    from orc_bound import residual_shell_ricci_approximation

    n = 200

    # Watts-Strogatz (small-world)
    G_sw = nx.watts_strogatz_graph(n, 8, 0.1, seed=0)
    C_sw = residual_shell_ricci_approximation(G_sw, n, k=3)
    print(f"Watts-Strogatz avg: {C_sw.sum()/C_sw.nnz:.4f}")

    # Barabási–Albert (scale-free)
    G_sf = nx.barabasi_albert_graph(n, 3, seed=0)
    C_sf = residual_shell_ricci_approximation(G_sf, n, k=3)
    print(f"Barabási–Albert avg: {C_sf.sum()/C_sf.nnz:.4f}")

    # 2D Grid (regular lattice)
    G_lat = nx.grid_2d_graph(10, 10)
    G_lat = nx.convert_node_labels_to_integers(G_lat)
    C_lat = residual_shell_ricci_approximation(G_lat, G_lat.number_of_nodes(), k=2)
    print(f"Lattice avg: {C_lat.sum()/C_lat.nnz:.4f}")

    # Erdős–Rényi (random)
    G_er = nx.erdos_renyi_graph(n, 0.05, seed=0)
    C_er = residual_shell_ricci_approximation(G_er, n, k=3)
    print(f"Erdős–Rényi avg: {C_er.sum()/C_er.nnz:.4f}")

Using Core Functions Directly
-----------------------------

For fine-grained control, call the core functions step by step:

.. code-block:: python

    import networkx as nx
    from orc_bound import (
        build_lazy_measures_k,
        all_pairs_shortest_path_matrix_cutoff,
        residual_shell_upper_bound,
    )

    G = nx.path_graph(20)
    n = G.number_of_nodes()

    # Step 1: Build k-hop measures
    mu = build_lazy_measures_k(G, alpha_lazy=0.0, k=3)

    # Step 2: Precompute distance matrix (cutoff = 2*k + 1)
    cutoff = 2 * 3 + 1
    nodes, idx, D = all_pairs_shortest_path_matrix_cutoff(G, cutoff=cutoff)

    # Step 3: Compute W1 upper bound for a specific edge
    ub, m_r, Rl, rbar = residual_shell_upper_bound(
        mu[0], mu[1], D, idx,
        l=3,
        rbar_mode="local-max",
        tol=1e-12,
    )
    print(f"W1 upper bound: {ub:.4f}")
    print(f"Mass per bucket: {m_r}")
    print(f"Residual mass: {Rl:.4f}, residual distance: {rbar:.1f}")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
