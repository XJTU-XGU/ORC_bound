Algorithm Details
=================

This package implements residual-shell lower bounds for Ollivier-Ricci
curvature. The current public workflow is:

.. code-block:: python

    from orc_bound import residual_shell_based_orc_bound

    result = residual_shell_based_orc_bound(graph)

Mathematical Background
-----------------------

For an edge ``(u, v)``, Ollivier-Ricci curvature is

.. math::

    \kappa(u, v) = 1 - \frac{W_1(\mu_u, \mu_v)}{d(u, v)}.

Here ``mu_u`` and ``mu_v`` are probability measures centered at the endpoints,
``d`` is shortest-path distance, and ``W_1`` is Wasserstein-1 distance. The
package computes an upper bound ``Wbar_1`` for ``W_1``. Therefore

.. math::

    \kappa_{lb}(u, v) = 1 - \frac{\overline{W}_1(\mu_u, \mu_v)}{d(u, v)}

is a lower bound on the true curvature.

k-Hop Weighted Lazy Measures
----------------------------

``build_m_hop_weighted_lazy_measures`` builds

.. math::

    \mu_x^{(k)} = \alpha \delta_x + (1 - \alpha) P^k[x, :].

``P`` is a row-stochastic transition matrix. For each node, outgoing transition
probability is proportional to the selected edge attribute, controlled by
``weight_attr``. Missing attributes are treated as weight ``1.0``. Edges with
zero weight do not receive transition probability mass, and negative weights
are rejected.

Transport distances are separate from these transition weights. The helper
``all_pairs_unweighted_shortest_path_matrix`` computes unweighted hop
distances, and ``residual_shell_based_orc_bound`` uses those hop distances as
the ground metric.

Distance Preprocessing
----------------------

The high-level helper resolves ``distance_cutoff`` before building the dense
distance matrix:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Effect
   * - ``"auto"``
     - Uses ``2 * k_hop + 1``.
   * - ``None``
     - Computes full all-pairs hop distances.
   * - integer
     - Computes hop distances up to that cutoff.

Unreached pairs remain ``numpy.inf`` in the distance matrix.

Residual-Shell W1 Upper Bound
-----------------------------

``compute_residual_shell_from_measures`` converts the Python measure mapping to
CSR arrays and calls the C++ kernel. For every requested edge ``(u, v)``, the
kernel:

1. copies the sparse measures of ``u`` and ``v``;
2. groups support pairs by integer distance shells ``0`` through ``l_shell``;
3. greedily matches available mass shell by shell, starting at distance ``0``;
4. computes the unmatched residual mass;
5. charges the residual mass by ``rbar``;
6. returns ``Wbar_1`` and ``1 - Wbar_1 / d(u, v)``.

The residual distance mode is controlled by ``rbar_mode``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Meaning
   * - ``"local-max"``
     - Uses the maximum finite distance between currently unmatched supports.
   * - ``"global-diam"``
     - Uses the maximum finite value in the distance matrix.

``"local-max"`` is the normal setting. ``"global-diam"`` is more conservative
and can be useful when a single global residual distance is preferred.

End-to-End Function
-------------------

``residual_shell_based_orc_bound`` performs all steps:

.. code-block:: text

    input: NetworkX graph, k_hop, alpha_lazy, l_shell, weight_attr

    nodes, idx = stable node order and contiguous integer index
    edges = graph edges or user-provided edge subset
    cutoff = 2 * k_hop + 1 when distance_cutoff == "auto"
    distances = unweighted shortest-path matrix with cutoff
    measures = alpha * delta_x + (1 - alpha) * P^k[x, :]
    result = C++ residual-shell kernel over requested edges

    return result dictionary with arrays, matrix, node mapping, and timings

The returned ``curvature_matrix`` is a symmetric SciPy CSR matrix. The array
fields ``curvatures`` and ``w1_upper_bounds`` preserve the requested edge order.

Specialized Fast Algorithm 1 Helper
-----------------------------------

``compute_weighted_one_hop_fast_algo1`` and ``compute_fast_algo1_from_measures``
are specialized helpers for the corrected one-hop fast Algorithm 1 kernel. They
return the same style of result dictionary, but the high-level residual-shell
workflow should normally use ``residual_shell_based_orc_bound``.
