# A fast and Accurate ORC bound

**ORC (Ollivier–Ricci Curvature) Lower Bounds** via residual-shell Wasserstein-1 measures with k-hop lazy random walks.

Project page: https://orc-bound.readthedocs.io
Paper: https://arxiv.org/abs/2604.12211

## Overview

This package implements lower bound algorithms for Ollivier–Ricci Curvature (ORC) on graph edges. For each edge `(u, v)` in a graph, it computes:

```
κ_lb(u, v) = 1 - W̄₁(μ_u, μ_v) / dist(u, v)
```

where `μ_u` is the k-hop lazy random walk measure at `u` and `W̄₁` is the residual-shell upper bound on the 1-Wasserstein distance.

See the [Algorithm Details](https://orc-bound.readthedocs.io/en/latest/algorithm.html) for the full mathematical description.

## Installation

Install from source:

```bash
python -m pip install .
```

If you want an editable install for development:

```bash
python -m pip install -e .
```

Requirements:

- Python >= 3.7
- NetworkX
- NumPy
- SciPy
- a C++17 compiler
- `pybind11`, installed automatically during build

On Windows, install Visual Studio Build Tools with C++ support before installing.

## Core Function

```python
from orc_bound import residual_shell_based_orc_bound

result = residual_shell_based_orc_bound(
    graph,
    k_hop=1,
    alpha_lazy=0.4,
    l_shell=2,
    rbar_mode="local-max",
    tol=1e-12,
    measure_tol=1e-15,
    weight_attr="weight",
    num_threads=0,
    progress_interval=0,
    distance_cutoff="auto",
    nodes=None,
    edges=None,
)
```

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `graph` | required | A NetworkX graph, usually `nx.Graph()`. Node labels are not required to be consecutive integers; they can be integers, strings, or any hashable Python objects. The function internally converts nodes to contiguous integer indices. Edge attributes are optional. |
| `k_hop` | `1` | Hop count of the local measure. The measure is `alpha_lazy * delta_x + (1-alpha_lazy) * P^k[x, :]`, where `P` is the weighted transition matrix and `k = k_hop`. |
| `alpha_lazy` | `0.4` | Lazy mass kept on the center node. Must be in `[0, 1]`. |
| `l_shell` | `2` | Number of distance shells handled explicitly in the residual-shell bound. Larger values can give a tighter bound but cost more time. |
| `rbar_mode` | `"local-max"` | How to bound remaining residual mass. Use `"local-max"` normally. `"global-diam"` uses the graph diameter from the distance matrix. |
| `tol` | `1e-12` | Numerical tolerance used by the C++ kernel. |
| `measure_tol` | `1e-15` | Tiny probability masses less than or equal to this value are dropped when building sparse measures. |
| `weight_attr` | `"weight"` | Name of the edge attribute used as transition weight. The name is not fixed. If your graph stores weights under another key, pass that key here, for example `weight_attr="similarity"` or `weight_attr="capacity"`. Missing attributes are treated as weight `1.0`, so an unweighted NetworkX graph works directly. Weights only affect the local random-walk measures; shortest-path transport distances are still unweighted hop distances. Negative weights are not allowed. Zero-weight edges remain in the graph but receive no transition probability mass. |
| `num_threads` | `0` | Number of OpenMP threads. `0` uses the OpenMP default. |
| `progress_interval` | `0` | If positive, print progress every N computed edges. Useful for large graphs. |
| `distance_cutoff` | `"auto"` | Shortest-path cutoff. `"auto"` uses `2*k_hop + 1`; `None` computes full all-pairs distances; an integer uses that cutoff. |
| `nodes` | `None` | Optional node order. Use this when you need fixed node indexing. |
| `edges` | `None` | Optional edge list to compute. If omitted, all graph edges are used. |

### Graph Weights

The edge weight key is configurable. The default is the common NetworkX
convention `weight`:

```python
G.add_edge("u", "v", weight=2.0)
result = residual_shell_based_orc_bound(G)
```

If your graph uses another edge attribute name, pass it through `weight_attr`:

```python
G.add_edge("u", "v", similarity=2.0)
result = residual_shell_based_orc_bound(G, weight_attr="similarity")
```

If the selected attribute is missing on an edge, that edge is treated as weight
`1.0`. Therefore a normal unweighted NetworkX graph can be used directly.

Weights do not change the ground distance used by transport. The ground
distance is always the unweighted graph hop distance. Weights only change the
transition probabilities used to build the local measures.

### Return Values

The function returns a dictionary. The most useful fields are:

| Field | Description |
| --- | --- |
| `curvatures` | NumPy array of curvature values, one per computed edge. |
| `w1_upper_bounds` | NumPy array of residual-shell W1 upper bounds. |
| `curvature_matrix` | Symmetric SciPy sparse matrix storing curvature values on computed edges. |
| `edge_indices` | Integer edge index array of shape `(num_edges, 2)`. |
| `nodes` | Node order used by the computation. |
| `node_index` | Mapping from original node labels to integer indices. |
| `distances` | Dense hop-distance matrix used by residual-shell. |
| `distance_cutoff` | Actual shortest-path cutoff used. |
| `elapsed_seconds` | C++ kernel time. |
| `num_threads` | Actual thread count used by the C++ kernel. |

### Extracting Edge Curvature

`result["curvatures"]` is ordered in the same order as the computed edge list.
If you do not pass `edges`, the order is `list(graph.edges())`.

```python
edges = list(G.edges())
result = residual_shell_based_orc_bound(G)

edge_to_curvature = {
    edge: curvature
    for edge, curvature in zip(edges, result["curvatures"])
}

print(edge_to_curvature[(0, 1)])
```

For an undirected graph, NetworkX may store the same edge as `(u, v)` while you
query `(v, u)`. A simple robust lookup is:

```python
def get_edge_curvature(edge_to_curvature, u, v):
    if (u, v) in edge_to_curvature:
        return edge_to_curvature[(u, v)]
    return edge_to_curvature[(v, u)]

print(get_edge_curvature(edge_to_curvature, 1, 0))
```

If you pass a custom edge list, use that same list when mapping results:

```python
target_edges = [(0, 1), (2, 3)]
result = residual_shell_based_orc_bound(G, edges=target_edges)

edge_to_curvature = dict(zip(target_edges, result["curvatures"]))
print(edge_to_curvature[(0, 1)])
```

## Minimal Example

```python
import networkx as nx
from orc_bound import residual_shell_based_orc_bound

G = nx.Graph()
G.add_edge(0, 1, weight=2.0)
G.add_edge(1, 2, weight=1.0)
G.add_edge(2, 3, weight=3.0)
G.add_edge(0, 3, weight=1.0)

edges = list(G.edges())
result = residual_shell_based_orc_bound(
    G,
    k_hop=1,
    alpha_lazy=0.4,
    l_shell=2,
    num_threads=4,
    progress_interval=1000,
)

print(result["curvatures"])
print(result["w1_upper_bounds"])

edge_to_curvature = dict(zip(edges, result["curvatures"]))
print("curvature(0, 1) =", edge_to_curvature[(0, 1)])
```

## Examples on Real-World Networks

Download Amazon or Facebook [data](https://github.com/CampanulaBells/HUGE-GAD/tree/main/datasets).
The code below assumes the data file is available at `datasets/Facebook.mat`.
For unweighted graph, run
```python
import networkx as nx
from scipy.io import loadmat
from orc_bound import residual_shell_based_orc_bound

data = loadmat("datasets/Facebook.mat")
A = data["Network"].maximum(data["Network"].T)
A.setdiag(0)
A.eliminate_zeros()

G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
edges = list(G.edges())

result = residual_shell_based_orc_bound(
    G,
    k_hop=1,
    l_shell=2,
    num_threads=8,
    progress_interval=5000,
)

edge_to_curvature = dict(zip(edges, result["curvatures"]))
print("number of edges:", len(edges))
print("curvature of first edge:", edge_to_curvature[edges[0]])
```

For weighted graph, first construct weight, and then apply the algorithm.
```python
import networkx as nx
from scipy.io import loadmat
from orc_bound import residual_shell_based_orc_bound

data = loadmat("datasets/Facebook.mat")
A = data["Network"].maximum(data["Network"].T)
X = data["Attributes"].tocsr()
A.setdiag(0)
A.eliminate_zeros()

G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
for u, v in G.edges():
    xu, xv = X.getrow(u), X.getrow(v)
    denom = (xu.multiply(xu).sum() * xv.multiply(xv).sum()) ** 0.5
    sim = float(xu.multiply(xv).sum() / denom) if denom > 0 else 0.0

    # Example weight only: this is not fixed by the algorithm.
    # Here we use 1 + cosine similarity between node attributes so that
    # attribute-similar endpoints get larger random-walk transition weight.
    # Users can replace this with any non-negative edge weight definition.
    G[u][v]["feature_weight"] = 1.0 + sim

edges = list(G.edges())
result = residual_shell_based_orc_bound(
    G,
    k_hop=1,
    l_shell=2,
    weight_attr="feature_weight",
    num_threads=8,
    progress_interval=5000,
)

edge_to_curvature = dict(zip(edges, result["curvatures"]))
print("number of edges:", len(edges))
print("curvature of first edge:", edge_to_curvature[edges[0]])
```

On the Amazon dataset with 170,000+ edges, our bound runs in less than 1 minute on 8 threads. 


More runnable examples are in the `examples/` directory:

- `examples/basic_usage.py`
- `examples/facebook_dataset_test.py`
