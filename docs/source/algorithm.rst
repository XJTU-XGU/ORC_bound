Algorithm Details
================

This page describes the mathematical foundations and algorithmic steps
implemented in ``orc_bound``.

----

Mathematical Background
-----------------------

**Ollivier–Ricci Curvature** (ORC) of an edge :math:`(u, v)` is defined as:

.. math::

    \kappa(u, v) = 1 - \frac{W_1(\mu_u, \mu_v)}{d(u, v)}

where:

- :math:`\mu_u` is a probability measure at node :math:`u`
- :math:`d(u, v)` is the shortest-path distance
- :math:`W_1` is the 1-Wasserstein (earth mover) distance between the two measures

Computing :math:`W_1` exactly is intractable for large graphs.
This package computes a **tight upper bound** on :math:`W_1`, which yields
a **lower bound** on the curvature.

----

Step 1 — k-Hop Lazy Random Walk Measure
---------------------------------------

For each node :math:`x`, the **k-hop lazy random walk measure** is:

.. math::

    \mu_x^{(k)} = \alpha \cdot \delta_x + (1 - \alpha) \cdot P^k(x, \cdot)

where:

- :math:`\alpha \in [0, 1]` is the **lazy parameter**
- :math:`\delta_x` is the Dirac delta at :math:`x`
- :math:`P` is the row-stochastic transition matrix:
  :math:`P[x, y] = 1 / \deg(x)` if :math:`(x, y)` is an edge
- :math:`P^k` is the k-th matrix power, giving k-step transition probabilities

The support of :math:`\mu_x^{(k)}` is exactly the **k-hop neighborhood** of :math:`x`.

**Implementation:** :func:`orc_bound.build_lazy_measures_k`

----

Step 2 — Truncated All-Pairs Shortest Paths
--------------------------------------------

The maximum distance between any two points in
:math:`\mathrm{supp}(\mu_u^{(k)})` and :math:`\mathrm{supp}(\mu_v^{(k)})`
is bounded by:

.. math::

    2k + 1 \quad \text{(when } u \text{ and } v \text{ are adjacent)}

Hence we precompute the distance matrix only up to cutoff :math:`2k + 1`:

.. math::

    D[i, j] = \begin{cases}
        \mathrm{dist}(x_i, x_j) & \text{if } \mathrm{dist}(x_i, x_j) \leq 2k + 1 \\
        \infty                   & \text{otherwise}
    \end{cases}

This avoids the :math:`O(|V|^3)` cost of full APSP.

**Implementation:** :func:`orc_bound.all_pairs_shortest_path_matrix_cutoff`

----

Step 3 — Residual-Shell Upper Bound on :math:`W_1`
---------------------------------------------------

For an edge :math:`(u, v)`, let :math:`\mu = \mu_u^{(k)}` and :math:`\nu = \mu_v^{(k)}`.
We compute an upper bound :math:`\bar{W}_1(\mu, \nu)` as follows.

**Bucket construction.** Group all pairs :math:`(x, y)` with
:math:`x \in \mathrm{supp}(\mu)`, :math:`y \in \mathrm{supp}(\nu)` into
:math:`l + 1` buckets by distance:

.. math::

    \mathrm{Bucket}_r = \{(x, y) : D[x, y] = r\}, \quad r = 0, 1, \dots, l

**Greedy matching.** Initialize :math:`a[x] = \mu[x]`, :math:`b[y] = \nu[y]`.
For each bucket :math:`r` from 0 to :math:`l`:

.. math::

    \delta \leftarrow \min(a[x], b[y]) \quad \forall (x, y) \in \mathrm{Bucket}_r

.. math::

    a[x] \leftarrow a[x] - \delta, \quad
    b[y] \leftarrow b[y] - \delta, \quad
    m_r \leftarrow m_r + \delta

**Residual shell.** After all buckets are processed:

.. math::

    R_l = \sum_x a[x] \quad \text{(remaining unmatched mass)}

If :math:`R_l > 0`, compute the residual distance:

.. math::

    \bar{r} = \max_{\substack{x : a[x] > 0 \\ y : b[y] > 0}} D[x, y]
    \quad \text{("local-max" mode)}

**Upper bound:**

.. math::

    \bar{W}_1(\mu, \nu) = \sum_{r=0}^{l} r \cdot m_r + \bar{r} \cdot R_l

**Implementation:** :func:`orc_bound.residual_shell_upper_bound`

----

Step 4 — Curvature Lower Bound
-------------------------------

The ORC lower bound for edge :math:`(u, v)` is:

.. math::

    \kappa_{lb}(u, v) = 1 - \frac{\bar{W}_1(\mu_u^{(k)}, \mu_v^{(k)})}{d(u, v)}

Note that :math:`d(u, v) = 1` for all edges in an unweighted graph.

**Implementation:** :func:`orc_bound.residual_shell_ricci_approximation`

----

Full Algorithm Pseudocode
--------------------------

.. code-block:: text

    Input:  Graph G = (V, E), integer k ≥ 1, lazy parameter α ∈ [0, 1],
            shell radius l ≥ 0, tolerance tol > 0, n_jobs workers
    Output: Sparse matrix C of ORC lower bounds

    # ── 1. Build k-hop measures ─────────────────────────────────────
    P    ← row-stochastic transition matrix of G
    P^k  ← P to the power k                               # matrix power
    for each node x ∈ V do
        for each node y ∈ V do
            μ_x[y] ← α · [x = y] + (1 - α) · P^k[x, y]

    # ── 2. Truncated APSP ────────────────────────────────────────────
    cutoff ← 2k + 1
    D ← ∞ matrix of size |V| × |V|
    for each node x ∈ V do
        for each node y at distance ≤ cutoff from x do
            D[x, y] ← dist(x, y)

    # ── 3. Parallel edge curvature ───────────────────────────────────
    parallel for each (u, v) ∈ E with n_jobs workers do
        a ← μ_u (copy, prune by tol)
        b ← μ_v (copy, prune by tol)
        for r ← 0 to l do
            Bucket[r] ← []
        for x ∈ keys(a) do
            for y ∈ keys(b) do
                d ← D[x, y]
                if d ≤ l and d < ∞ then
                    Bucket[int(d)].append((x, y))

        for r ← 0 to l do
            for (x, y) ∈ Bucket[r] do
                δ ← min(a[x], b[y])
                if δ > tol then
                    a[x] ← a[x] - δ
                    b[y] ← b[y] - δ
                    m_r ← m_r + δ

        R_l ← Σ_x a[x]
        if R_l > tol then
            RU ← {x | a[x] > tol}
            RV ← {y | b[y] > tol}
            r̄ ← max{ D[x, y] | x ∈ RU, y ∈ RV }
        else
            r̄ ← 0

        W̄ ← Σ_{r=0}^{l} r · m_r + r̄ · R_l
        C[u, v] ← 1 - W̄ / D[u, v]

    return C

----

Complexity Analysis
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Step
     - Time
     - Space
   * - Build :math:`P^k`
     - :math:`O(|V|^3 \log k)` via matrix power
     - :math:`O(|V|^2)`
   * - Truncated APSP
     - :math:`O(|V| \cdot (2k+1) \cdot \bar{d})`
     - :math:`O(|V|^2)`
   * - Per-edge W1 bound
     - :math:`O(|\mathrm{supp}(\mu)| \cdot |\mathrm{supp}(\nu)|)` — parallelized over edges
     - :math:`O(|V|^2)` total
   * - **Total (parallel)**
     - :math:`O(|V| \cdot k \cdot \bar{d} + |E| \cdot d^2 \cdot k_{\max}) / n_{\text{jobs}}`
     - :math:`O(|V|^2)`

where :math:`\bar{d}` is the average node degree.

----

Parameter Guide
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Effect
   * - ``k``
     - 1
     - Number of random walk steps. Larger k spreads mass further (more global measure). Affects both the measure support size and the distance cutoff.
   * - ``alpha_lazy``
     - 0.0
     - Lazy mixing parameter in [0, 1]. Higher values keep more mass at the source node. ``alpha=1`` gives the trivial curvature 0.
   * - ``l_shell``
     - 3
     - Maximum shell distance for bucket matching. ``l_eff = min(l_shell, 2*k+1)`` is the effective shell.
   * - ``rbar_mode``
     - ``"local-max"``
     - How to estimate the residual distance. ``"local-max"`` uses the max distance between residual supports; ``"global"`` uses the max finite distance in the entire matrix.
   * - ``n_jobs``
     - None
     - Number of parallel workers. ``None/0`` = all cores; ``-1`` = all but one; positive int = exact number.
