Quick Start
===========

Compute Curvature Bounds
------------------------

Use ``residual_shell_based_orc_bound`` for the normal end-to-end workflow:

.. code-block:: python

    import networkx as nx
    from orc_bound import residual_shell_based_orc_bound

    graph = nx.cycle_graph(8)
    edges = list(graph.edges())

    result = residual_shell_based_orc_bound(
        graph,
        k_hop=1,
        alpha_lazy=0.4,
        l_shell=2,
        num_threads=0,
    )

    for edge, curvature in zip(edges, result["curvatures"]):
        print(edge, curvature)

The order of ``result["curvatures"]`` follows the computed edge list. If you do
not pass ``edges``, that list is ``list(graph.edges())``.

Weighted Random-Walk Measures
-----------------------------

Edge weights affect the transition matrix used to build the local measures.
They do not affect the ground distance used by transport; the residual-shell
kernel uses unweighted hop distances.

.. code-block:: python

    import networkx as nx
    from orc_bound import residual_shell_based_orc_bound

    graph = nx.Graph()
    graph.add_edge("a", "b", similarity=3.0)
    graph.add_edge("b", "c", similarity=1.0)
    graph.add_edge("c", "a", similarity=2.0)

    result = residual_shell_based_orc_bound(
        graph,
        k_hop=1,
        alpha_lazy=0.4,
        l_shell=2,
        weight_attr="similarity",
    )

    print(result["curvatures"])

Missing values for ``weight_attr`` are treated as ``1.0``. Zero-weight edges
remain in the graph but receive no transition probability mass. Negative
weights raise ``ValueError``.

Computing a Subset of Edges
---------------------------

Pass ``edges`` when only part of the graph should be evaluated:

.. code-block:: python

    target_edges = [(0, 1), (2, 3)]

    result = residual_shell_based_orc_bound(
        graph,
        edges=target_edges,
        k_hop=1,
        l_shell=2,
    )

    edge_to_curvature = dict(zip(target_edges, result["curvatures"]))

Distance Cutoffs
----------------

By default, ``distance_cutoff="auto"`` uses ``2 * k_hop + 1``. This is enough
for adjacent-edge k-hop supports in the usual residual-shell calculation and
keeps the distance preprocessing local. Use ``distance_cutoff=None`` to compute
full all-pairs hop distances, or pass a non-negative integer cutoff.

Threading
---------

``num_threads`` controls the OpenMP edge loop in the C++ kernel:

.. code-block:: python

    # Use the OpenMP default thread count.
    result = residual_shell_based_orc_bound(graph, num_threads=0)

    # Use exactly 8 threads.
    result = residual_shell_based_orc_bound(graph, num_threads=8)

If the extension is built without OpenMP support, the kernel runs with one
thread and reports ``result["num_threads"] == 1``.
