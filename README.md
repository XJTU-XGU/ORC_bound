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

The **Ollivier–Ricci curvature** (ORC) of an edge $(u,v)$ is defined by
$$
\kappa(u,v)=1-\frac{W_1(\mu_u,\mu_v)}{d(u,v)},
$$
where $\mu_u$ is a probability measure centered at $u$, $d(u,v)$ is the shortest-path distance, and $W_1$ is the $1$-Wasserstein (earth mover's) distance between the two measures.

---

### Step 1. $k$-Hop Lazy Random Walk Measure

For each node $x$, define the **$k$-hop lazy random walk measure** by
$$
\mu_x^{(k)}=\alpha\,\delta_x+(1-\alpha)\,P^k(x,\cdot),
$$
where:

- $\alpha\in[0,1]$ is the lazy parameter;
- $\delta_x$ is the Dirac measure at $x$;
- $P$ is the row-stochastic transition matrix of the graph, given by
  $$
  P(x,y)=
  \begin{cases}
  \dfrac{1}{\deg(x)}, & \text{if } (x,y)\in E,\\[1mm]
  0, & \text{otherwise};
  \end{cases}
  $$
- $P^k$ denotes the $k$-step transition probabilities.

The support of $\mu_x^{(k)}$ is contained in the $k$-hop neighborhood of $x$.

---

### Step 2. Truncated All-Pairs Shortest Paths

When $(u,v)$ is an edge, the maximum distance between any two points in $\operatorname{supp}(\mu_u^{(k)})$ and $\operatorname{supp}(\mu_v^{(k)})$ is bounded by
$$
2k+1.
$$
Hence, we precompute the distance matrix only up to the cutoff $2k+1$:
$$
D[i,j]=
\begin{cases}
\operatorname{dist}(x_i,x_j), & \text{if } \operatorname{dist}(x_i,x_j)\le 2k+1,\\[1mm]
\infty, & \text{otherwise}.
\end{cases}
$$
This avoids the cost of computing the full all-pairs shortest-path matrix.

---

### Step 3. Residual-Shell Upper Bound on $W_1(\mu_u^{(k)},\mu_v^{(k)})$

For an edge $(u,v)$, let
$$
\mu=\mu_u^{(k)}, \qquad \nu=\mu_v^{(k)}.
$$
We compute an upper bound $\overline{W}_1(\mu,\nu)$ as follows.

#### Bucket construction

Group all pairs $(x,y)$ with $x\in\operatorname{supp}(\mu)$ and $y\in\operatorname{supp}(\nu)$ into $l+1$ buckets according to their distance:
$$
\mathcal{B}_r=\{(x,y): D[x,y]=r\}, \qquad r=0,1,\dots,l.
$$

#### Greedy matching

Initialize the residual masses by
$$
a(x)=\mu(x), \qquad b(y)=\nu(y).
$$
For each bucket $r=0,1,\dots,l$, and for each $(x,y)\in\mathcal{B}_r$, define
$$
\delta(x,y)=\min\{a(x),\,b(y)\}.
$$
Then update
$$
a(x)\leftarrow a(x)-\delta(x,y), \qquad
b(y)\leftarrow b(y)-\delta(x,y),
$$
and let
$$
m_r=\sum_{(x,y)\in\mathcal{B}_r}\delta(x,y).
$$

#### Residual shell

After all buckets have been processed, define the remaining unmatched mass by
$$
R_l=\sum_x a(x).
$$
If $R_l>0$, define the residual distance in the local-max mode by
$$
\overline{r}
=
\max_{\substack{x:\,a(x)>0\\ y:\,b(y)>0}} D[x,y].
$$

#### Upper bound

The resulting upper bound is
$$
\overline{W}_1(\mu,\nu)
=
\sum_{r=0}^l r\,m_r+\overline{r}\,R_l.
$$

---

### Step 4. Curvature Lower Bound

The resulting lower bound for the Ollivier--Ricci curvature of the edge $(u,v)$ is
$$
\kappa_{\mathrm{lb}}(u,v)
=
1-\frac{\overline{W}_1\bigl(\mu_u^{(k)},\mu_v^{(k)}\bigr)}{d(u,v)}.
$$
For an unweighted graph, $d(u,v)=1$ for every edge $(u,v)$.

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
|-----------------|-------------------------------------|------------|
| Build `P^k` | `O(|V|³ log k)` via matrix power | `O(|V|²)` |
| Truncated APSP | `O(|V| · cutoff · avg_degree)` | `O(|V|²)` |
| Per-edge W1 bound | `O(|supp(μ)| · |supp(ν)|)` — parallelized over edges | `O(|V|²)` total |
| Total (parallel) | `O(|V| · cutoff · deg + |E| · d² · k_max) / n_jobs` | `O(|V|²)` |

## License

MIT License
