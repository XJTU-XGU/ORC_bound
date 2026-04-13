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

## License

MIT License
