API Reference
=============

Main Entry Point
----------------

.. autofunction:: orc_bound.residual_shell_based_orc_bound

Parameters
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``graph``
     - required
     - NetworkX graph. Node labels may be any hashable Python object.
   * - ``k_hop``
     - ``1``
     - Random-walk step count used in ``P^k``. ``0`` is allowed.
   * - ``alpha_lazy``
     - ``0.4``
     - Mass kept on the source node. Must be in ``[0, 1]``.
   * - ``l_shell``
     - ``2``
     - Largest explicit distance shell used before residual mass is charged.
   * - ``rbar_mode``
     - ``"local-max"``
     - Residual distance mode: ``"local-max"`` or ``"global-diam"``.
   * - ``tol``
     - ``1e-12``
     - Numerical tolerance used by the C++ kernel.
   * - ``measure_tol``
     - ``1e-15``
     - Tiny probability masses below this threshold are dropped while building measures.
   * - ``weight_attr``
     - ``"weight"``
     - Edge attribute used as transition weight.
   * - ``num_threads``
     - ``0``
     - OpenMP thread count. ``0`` uses the OpenMP default.
   * - ``progress_interval``
     - ``0``
     - Print C++ progress every N computed edges. ``0`` disables progress output.
   * - ``distance_cutoff``
     - ``"auto"``
     - ``"auto"``, ``None``, or a non-negative integer.
   * - ``nodes``
     - ``None``
     - Optional explicit node order.
   * - ``edges``
     - ``None``
     - Optional edge iterable. If omitted, all graph edges are computed.

Return Dictionary
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Description
   * - ``curvatures``
     - NumPy array of curvature lower bounds.
   * - ``w1_upper_bounds``
     - NumPy array of residual-shell Wasserstein-1 upper bounds.
   * - ``curvature_matrix``
     - Symmetric SciPy CSR matrix containing curvature values.
   * - ``edge_indices``
     - Internal integer edge array with shape ``(num_edges, 2)``.
   * - ``nodes``
     - Node order used by the computation.
   * - ``node_index``
     - Mapping from original node labels to internal integer indices.
   * - ``distances``
     - Dense unweighted hop-distance matrix used by the kernel.
   * - ``distance_cutoff``
     - Resolved cutoff passed to shortest-path preprocessing.
   * - ``measures``
     - Python mapping of k-hop lazy measures.
   * - ``elapsed_seconds``
     - C++ kernel wall time.
   * - ``python_total_seconds``
     - Python-side time around the residual-shell compute call.
   * - ``edge_time_seconds``
     - Per-edge kernel timing array.
   * - ``sum_edge_time_seconds``
     - Sum of per-edge timings.
   * - ``num_threads``
     - Actual C++ thread count reported by the kernel.

Measure and Distance Helpers
----------------------------

.. autofunction:: orc_bound.node_order_and_index

.. autofunction:: orc_bound.weighted_transition_matrix

.. autofunction:: orc_bound.build_m_hop_weighted_lazy_measures

.. autofunction:: orc_bound.build_one_hop_weighted_lazy_measures

.. autofunction:: orc_bound.build_lazy_measures

.. autofunction:: orc_bound.all_pairs_unweighted_shortest_path_matrix

Low-Level Compute Helpers
-------------------------

.. autofunction:: orc_bound.compute_residual_shell_from_measures

.. autofunction:: orc_bound.compute_weighted_one_hop_fast_algo1

.. autofunction:: orc_bound.compute_fast_algo1_from_measures

.. autofunction:: orc_bound.measures_to_csr

.. autofunction:: orc_bound.edges_to_index_array

.. autofunction:: orc_bound.symmetric_edge_matrix

Exact Baseline
--------------

.. autofunction:: orc_bound.exact_w1_ot
