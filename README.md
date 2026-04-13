# ORC-Bound

**ORC (Ollivier–Ricci Curvature) Lower Bounds** via residual-shell Wasserstein-1 measures with k-hop lazy random walks.

## Overview

This package implements lower bound algorithms for Ollivier–Ricci Curvature (ORC) on graph edges. For each edge `(u, v)` in a graph, it computes:

```
κ_lb(u, v) = 1 - W̄₁(μ_u, μ_v) / dist(u, v)
```

where `μ_u` is the k-hop lazy random walk measure at `u` and `W̄₁` is the residual-shell upper bound on the 1-Wasserstein distance.

## Features

- **Residual-shell Ricci curvature** with k-hop random walk support
- **Multi-threaded edge processing** — leverage all CPU cores for large graphs
- **Sparse matrix output** — memory efficient for large graphs
- **Truncated APSP** — avoids full all-pairs shortest path computation

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

For SerpAPI support:

```bash
pip install -e ".[serpapi]"
```

## Quick Start

```python
import networkx as nx
from orc_bound import residual_shell_ricci_approximation

# Build a graph
G = nx.watts_strogatz_graph(200, 6, 0.2, seed=0)

# Compute curvature with k=1-hop measures and 4 threads
C = residual_shell_ricci_approximation(
    G,
    G.number_of_nodes(),
    k=1,
    n_jobs=4,
)

# Average curvature over all edges
avg_curv = C.sum() / C.nnz
print(f"Average curvature: {avg_curv:.4f}")
```

## API Reference

### `residual_shell_ricci_approximation`

```python
def residual_shell_ricci_approximation(
    graph: nx.Graph,
    num_nodes: int,
    k: int = 1,
    alpha_lazy: float = 0.0,
    l_shell: int = 3,
    rbar_mode: str = "local-max",
    tol: float = 1e-12,
    symmetric: bool = False,
    n_jobs: int | None = None,
) -> csr_matrix:
```

| Parameter     | Type           | Default  | Description                                           |
|---------------|----------------|---------- |-------------------------------------------------------|
| `graph`       | `nx.Graph`     | —         | Input graph                                           |
| `num_nodes`   | `int`          | —         | Number of nodes (must match graph)                    |
| `k`           | `int`          | `1`       | Number of random walk steps (k-hop neighborhood)      |
| `alpha_lazy`  | `float`        | `0.0`     | Lazy mixing parameter [0, 1]                          |
| `l_shell`     | `int`          | `3`       | Maximum shell distance for bucket matching            |
| `rbar_mode`   | `str`          | `"local-max"` | Residual distance estimation method              |
| `tol`         | `float`        | `1e-12`   | Numerical tolerance for mass pruning                  |
| `symmetric`   | `bool`         | `False`   | Whether to populate both (u,v) and (v,u) entries       |
| `n_jobs`      | `int \| None`  | `None`    | Number of parallel workers (None=all cores)           |

### Core Functions

- `build_lazy_measures_k(G, alpha_lazy, k)` — Build k-hop lazy random walk measures
- `all_pairs_shortest_path_matrix_cutoff(G, cutoff)` — Truncated APSP matrix
- `residual_shell_upper_bound(mu_x, mu_y, D, idx, l, ...)` — W1 upper bound

## Multi-Threading

The most expensive step — computing curvature for each edge — is fully parallelizable because each edge's computation is independent.

```python
# Use all CPU cores (default)
C = residual_shell_ricci_approximation(G, n, k=10)

# Use exactly 8 workers
C = residual_shell_ricci_approximation(G, n, k=10, n_jobs=8)

# Use all but one core
C = residual_shell_ricci_approximation(G, n, k=10, n_jobs=-1)
```

### Threading vs Multiprocessing

The algorithm is **memory-bandwidth bound** (dominated by numpy array operations on the precomputed distance matrix), not CPU-bound. Threading avoids the GIL for numpy operations and sidesteps the serialization overhead of multiprocessing, making `ThreadPoolExecutor` the natural choice.

## Web Search via SerpAPI

The package includes a utility for fetching related literature:

```python
from orc_bound.utils.search import search_orc_literature

# Search for ORC-related papers
results = search_orc_literature(
    query="Ollivier Ricci curvature graph networks",
    num_results=10,
    serpapi_key="your-serpapi-key",
)
```

## Algorithm Details

### Mathematical Background

**Ollivier–Ricci Curvature** (ORC) of an edge `(u, v)` is defined as:

$$\kappa(u, v) = 1 - \frac{W_1(\mu_u, \mu_v)}{d(u, v)}$$

where `μ_u` is a probability measure at `u`, `d(u, v)` is the shortest-path distance, and `W_1` is the 1-Wasserstein (earth mover) distance between the two measures.

---

### Step 1 — k-Hop Lazy Random Walk Measure

For each node `x`, define the **k-hop lazy random walk measure** as:

$$\mu_x^{(k)} = \alpha \cdot \delta_x + (1 - \alpha) \cdot P^k(x, \cdot)$$

where:

- `α ∈ [0, 1]` is the **lazy parameter**
- `δ_x` is the Dirac delta at `x`
- `P` is the row-stochastic transition matrix of the graph: `P[x, y] = 1 / deg(x)` if `(x, y)` is an edge
- `P^k` is the `k`-th matrix power, giving the k-step transition probabilities

The support of `μ_x^{(k)}` is exactly the **k-hop neighborhood** of `x`.

---

### Step 2 — Truncated All-Pairs Shortest Paths

The maximum distance between any two points in `supp(μ_u^{(k)})` and `supp(μ_v^{(k)})` is bounded by:

$$2k + 1 \quad \text{(when } u \text{ and } v \text{ are adjacent)}$$

Hence we precompute the distance matrix only up to cutoff `2k + 1`:

$$D[i, j] = \begin{cases} \text{dist}(x_i, x_j) & \text{if } \text{dist}(x_i, x_j) \leq 2k + 1 \\ \infty & \text{otherwise} \end{cases}$$

This avoids the `O(n³)` cost of full APSP.

---

### Step 3 — Residual-Shell Upper Bound on `W₁(μ_u, μ_v)`

For an edge `(u, v)`, let `μ = μ_u^{(k)}` and `ν = μ_v^{(k)}`. We compute an upper bound `W̄₁(μ, ν)` as follows.

**Bucket construction.** Group all pairs `(x, y)` with `x ∈ supp(μ)`, `y ∈ supp(ν)` into `l + 1` buckets by distance:

$$\text{Bucket } r = \{(x, y) : D[x, y] = r\}, \quad r = 0, 1, \dots, l$$

**Greedy matching.** Initialize `a[x] = μ[x]`, `b[y] = ν[y]`. For each bucket `r` from 0 to `l`:

$$\delta_r = \min(a[x], b[y]) \quad \text{for each } (x, y) \in \text{Bucket } r$$
$$a[x] \leftarrow a[x] - \delta_r, \quad b[y] \leftarrow b[y] - \delta_r$$
$$m_r = \sum_{(x,y)\in\text{Bucket } r} \delta_r$$

**Residual shell.** After all buckets are processed, let:

$$R_l = \sum_x a[x] \quad \text{(remaining unmatched mass)}$$

If `R_l > 0`, compute the residual distance:

$$\bar{r} = \max_{\substack{x : a[x] > 0 \\ y : b[y] > 0}} D[x, y] \quad \text{("local-max" mode)}$$

**Upper bound:**

$$\bar{W}_1(\mu, \nu) = \sum_{r=0}^{l} r \cdot m_r + \bar{r} \cdot R_l$$

---

### Step 4 — Curvature Lower Bound

The ORC lower bound for edge `(u, v)` is:

$$\kappa_{lb}(u, v) = 1 - \frac{\bar{W}_1(\mu_u^{(k)}, \mu_v^{(k)})}{d(u, v)}$$

Note that `d(u, v) = 1` for all edges in an unweighted graph.

---

### Full Algorithm Pseudocode

```
Input:  Graph G = (V, E), integer k ≥ 1, lazy parameter α ∈ [0, 1],
        shell radius l ≥ 0, tolerance tol > 0, n_jobs workers
Output: Sparse matrix C of ORC lower bounds

# ── 1. Build k-hop measures ──────────────────────────────────────────────────
P ← row-stochastic transition matrix of G
Pk ← P^k                                          # matrix power
for each node x ∈ V do
    for each node y ∈ V do
        μ_x[y] ← α · [x = y] + (1 - α) · Pk[x, y]

# ── 2. Truncated APSP ────────────────────────────────────────────────────────
cutoff ← 2k + 1
D ← ∞ matrix of size |V| × |V|
for each node x ∈ V do
    for each node y at distance ≤ cutoff from x do
        D[x, y] ← dist(x, y)

# ── 3. Parallel edge curvature ─────────────────────────────────────────────
E_edges ← list of edges of G
parallel for each (u, v) ∈ E_edges with n_jobs workers do
    # Residual-shell upper bound
    a ← μ_u (copy, prune by tol)
    b ← μ_v (copy, prune by tol)
    for r ← 0 to l do
        Bucket[r] ← []                              # reset buckets
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
```

### Complexity Analysis

| Step | Time | Space |
|------|------|-------|
| Build `P^k` | `O(|V|³ log k)` via matrix power | `O(|V|²)` |
| Truncated APSP | `O(|V| · cutoff · avg_degree)` | `O(|V|²)` |
| Per-edge W1 bound | `O(|supp(μ)| · |supp(ν)|)` — parallelized over edges | `O(|V|²)` total |
| Total (parallel) | `O(|V| · cutoff · deg + |E| · d² · k_max) / n_jobs` | `O(|V|²)` |

## License

MIT License
