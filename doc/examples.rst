Examples
========

Basic Graph
-----------

.. code-block:: python

    import networkx as nx
    from orc_bound import residual_shell_based_orc_bound

    graph = nx.path_graph(6)
    edges = list(graph.edges())

    result = residual_shell_based_orc_bound(
        graph,
        k_hop=1,
        alpha_lazy=0.4,
        l_shell=2,
        num_threads=2,
    )

    for edge, curvature, w1 in zip(
        edges,
        result["curvatures"],
        result["w1_upper_bounds"],
    ):
        print(edge, curvature, w1)

Custom Node Labels
------------------

The package maps NetworkX labels to contiguous integer indices internally.

.. code-block:: python

    import networkx as nx
    from orc_bound import residual_shell_based_orc_bound

    graph = nx.Graph()
    graph.add_edges_from([
        ("alice", "bob"),
        ("bob", "carol"),
        ("carol", "dave"),
    ])

    result = residual_shell_based_orc_bound(graph)
    print(result["nodes"])
    print(result["node_index"])

Weighted Measures
-----------------

Use ``weight_attr`` to choose which edge attribute controls random-walk
transition probabilities.

.. code-block:: python

    import networkx as nx
    from orc_bound import residual_shell_based_orc_bound

    graph = nx.Graph()
    graph.add_edge(0, 1, feature_weight=2.0)
    graph.add_edge(1, 2, feature_weight=1.0)
    graph.add_edge(2, 3, feature_weight=4.0)
    graph.add_edge(3, 0, feature_weight=1.0)

    result = residual_shell_based_orc_bound(
        graph,
        k_hop=2,
        alpha_lazy=0.4,
        l_shell=3,
        weight_attr="feature_weight",
    )

    print(result["curvatures"])

Exact W1 Comparison
-------------------

``exact_w1_ot`` uses the optional POT package and is intended for small
baselines, not large production runs.

.. code-block:: python

    import networkx as nx
    from orc_bound import (
        all_pairs_unweighted_shortest_path_matrix,
        build_m_hop_weighted_lazy_measures,
        exact_w1_ot,
        residual_shell_based_orc_bound,
    )

    graph = nx.cycle_graph(6)
    nodes, idx, distances = all_pairs_unweighted_shortest_path_matrix(graph)
    measures = build_m_hop_weighted_lazy_measures(
        graph,
        k_hop=1,
        alpha_lazy=0.4,
        nodes=nodes,
        idx=idx,
    )

    exact = exact_w1_ot(measures[0], measures[1], distances, idx)
    bound = residual_shell_based_orc_bound(graph, edges=[(0, 1)])

    print("exact W1:", exact)
    print("upper bound:", bound["w1_upper_bounds"][0])

Facebook Dataset Script
-----------------------

The repository includes ``examples/facebook_dataset_test.py``. It loads a
``Facebook.mat`` dataset, computes residual-shell curvature bounds, prints
summary statistics, and can save per-edge results:

.. code-block:: bash

    python examples/facebook_dataset_test.py \
        --dataset ../datasets/Facebook.mat \
        --k-hop 1 \
        --l-shell 2 \
        --threads 8 \
        --output facebook_curvatures.csv
