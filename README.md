# ORC-Bound

**ORC (Ollivier–Ricci Curvature) Lower Bounds** via residual-shell Wasserstein-1 measures with k-hop lazy random walks.

Project page: https://orc-bound.readthedocs.io

## Overview

This package implements lower bound algorithms for Ollivier–Ricci Curvature (ORC) on graph edges. For each edge `(u, v)` in a graph, it computes:

```
κ_lb(u, v) = 1 - W̄₁(μ_u, μ_v) / dist(u, v)
```

where `μ_u` is the k-hop lazy random walk measure at `u` and `W̄₁` is the residual-shell upper bound on the 1-Wasserstein distance.

See the [Algorithm Details](https://orc-bound.readthedocs.io/en/latest/algorithm.html) for the full mathematical description.

## Features

- **Residual-shell Ricci curvature** with k-hop random walk support
- **Multi-threaded edge processing** — leverage all CPU cores for large graphs
- **Sparse matrix output** — memory efficient for large graphs
- **Truncated APSP** — avoids full all-pairs shortest path computation

## Installation

```bash
pip install orc-bound
```

From source:

```bash
git clone https://github.com/XJTU-XGU/ORC_bound.git
cd ORC_bound
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

G = nx.watts_strogatz_graph(200, 6, 0.2, seed=0)
C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=2, n_jobs=4)
print(f"Average curvature: {C.sum() / C.nnz:.4f}")
```

## Usage Examples

### Basic Usage — All Default Parameters

```python
import networkx as nx
from orc_bound import residual_shell_ricci_approximation

G = nx.karate_club_graph()
C = residual_shell_ricci_approximation(G, G.number_of_nodes())

# Average curvature over all edges
print(f"Average: {C.sum() / C.nnz:.4f}")

# All edge curvatures
for u, v in G.edges():
    print(f"Edge ({u}, {v}): {C[u, v]:.4f}")
```

### Choosing k — Number of Random Walk Steps

The parameter `k` controls how "global" the random walk measure is.
Larger `k` spreads mass further from the source node.

```python
# k=1: local measure (neighbors only)
C1 = residual_shell_ricci_approximation(G, n, k=1)

# k=5: semi-global measure
C5 = residual_shell_ricci_approximation(G, n, k=5)

# k=20: near-uniform measure
C20 = residual_shell_ricci_approximation(G, n, k=20)
```

### Lazy Mixing Parameter alpha

The `alpha_lazy` parameter controls the weight of the Dirac delta at the source node.
Higher alpha keeps more mass locally.

```python
# alpha=0: pure random walk (mass spreads fully)
C0 = residual_shell_ricci_approximation(G, n, k=3, alpha_lazy=0.0)

# alpha=0.5: balanced
C5 = residual_shell_ricci_approximation(G, n, k=3, alpha_lazy=0.5)

# alpha=1: trivial delta measure (kappa = 0 for all edges)
C1 = residual_shell_ricci_approximation(G, n, k=3, alpha_lazy=1.0)
```

### Multi-Threaded Parallel Computation

The most expensive step — computing curvature for each edge — is fully parallelizable.
Use `n_jobs` to control the number of worker threads.

```python
# Use all available CPU cores (default)
C = residual_shell_ricci_approximation(G, n, k=10)

# Use exactly 8 threads
C = residual_shell_ricci_approximation(G, n, k=10, n_jobs=8)

# Use all but one core
C = residual_shell_ricci_approximation(G, n, k=10, n_jobs=-1)

# Sequential (single-threaded)
C = residual_shell_ricci_approximation(G, n, k=10, n_jobs=1)
```

### Custom Shell Radius l

Control how many distance buckets are used in the residual-shell matching.

```python
# Only bucket 0 (distance 0): fastest, weakest bound
C = residual_shell_ricci_approximation(G, n, k=3, l_shell=0)

# Buckets 0-3: good balance of speed and accuracy
C = residual_shell_ricci_approximation(G, n, k=3, l_shell=3)

# Buckets 0-10: more accurate but slower
C = residual_shell_ricci_approximation(G, n, k=3, l_shell=10)
```

### Build Custom Measures and Distance Matrices

For fine-grained control, use the core functions directly.

```python
import networkx as nx
from orc_bound import (
    build_lazy_measures_k,
    all_pairs_shortest_path_matrix_cutoff,
    residual_shell_upper_bound,
)

G = nx.path_graph(20)
n = G.number_of_nodes()

# Build k-hop measures
mu = build_lazy_measures_k(G, alpha_lazy=0.0, k=3)

# Precompute distance matrix (cutoff = 2*k + 1)
cutoff = 2 * 3 + 1
nodes, idx, D = all_pairs_shortest_path_matrix_cutoff(G, cutoff=cutoff)

# Compute W1 upper bound for a specific edge
ub, m_r, Rl, rbar = residual_shell_upper_bound(
    mu[0], mu[1], D, idx,
    l=3,
    rbar_mode="local-max",
    tol=1e-12,
)
print(f"W1 upper bound: {ub:.4f}")
print(f"Mass per bucket: {m_r}")
print(f"Residual mass: {Rl:.4f}, residual distance: {rbar:.1f}")
```

### Interpreting the Result Matrix

The return value is a ``scipy.sparse.csr_matrix``. Non-zero entries correspond to graph edges.

```python
import networkx as nx
from orc_bound import residual_shell_ricci_approximation
from scipy.sparse import csr_matrix

G = nx.watts_strogatz_graph(80, 6, 0.2, seed=0)
C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=2)

# Basic statistics
print(f"Shape: {C.shape}")           # (80, 80)
print(f"Non-zeros: {C.nnz}")        # equals number of edges
print(f"Average: {C.sum()/C.nnz:.4f}")
print(f"Min: {C.data.min():.4f}")
print(f"Max: {C.data.max():.4f}")

# Get all non-zero entries
print("\nAll edge curvatures:")
for i in range(C.nnz):
    u, v = C.row[i], C.col[i]
    kappa = C.data[i]
    print(f"  ({u}, {v}): {kappa:.4f}")

# Extract as dense array
dense = C.toarray()
```

### Different Graph Types

```python
import networkx as nx
from orc_bound import residual_shell_ricci_approximation

# Small-world graph (Watts-Strogatz)
G_sw = nx.watts_strogatz_graph(200, 8, 0.1, seed=0)
C_sw = residual_shell_ricci_approximation(G_sw, G_sw.number_of_nodes(), k=3)
print(f"Watts-Strogatz avg: {C_sw.sum()/C_sw.nnz:.4f}")

# Scale-free graph (Barabási-Albert)
G_sf = nx.barabasi_albert_graph(200, 3, seed=0)
C_sf = residual_shell_ricci_approximation(G_sf, G_sf.number_of_nodes(), k=3)
print(f"Barabási-Albert avg: {C_sf.sum()/C_sf.nnz:.4f}")

# Regular lattice
G_lat = nx.grid_2d_graph(10, 10)
G_lat = nx.convert_node_labels_to_integers(G_lat)
C_lat = residual_shell_ricci_approximation(G_lat, G_lat.number_of_nodes(), k=2)
print(f"Lattice avg: {C_lat.sum()/C_lat.nnz:.4f}")

# Erdős–Rényi random graph
G_er = nx.erdos_renyi_graph(200, 0.05, seed=0)
C_er = residual_shell_ricci_approximation(G_er, G_er.number_of_nodes(), k=3)
print(f"Erdős–Rényi avg: {C_er.sum()/C_er.nnz:.4f}")
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

See the [Algorithm Details page](https://orc-bound.readthedocs.io/en/latest/algorithm.html) for the full mathematical description including:

- k-Hop lazy random walk measure definition
- Residual-shell W1 upper bound derivation
- Full pseudocode
- Complexity analysis
- Parameter guide

## License

MIT License
