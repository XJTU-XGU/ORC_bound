from __future__ import annotations

import time
from typing import Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from . import _residual_shell_cpp


Node = Hashable
Measure = Mapping[Node, float]


def node_order_and_index(
    graph: nx.Graph,
    nodes: Optional[Iterable[Node]] = None,
) -> Tuple[list[Node], Dict[Node, int]]:
    """Return a stable node list and node-to-row-index map."""
    ordered_nodes = list(graph.nodes() if nodes is None else nodes)
    return ordered_nodes, {node: i for i, node in enumerate(ordered_nodes)}


def all_pairs_unweighted_shortest_path_matrix(
    graph: nx.Graph,
    nodes: Optional[Sequence[Node]] = None,
    idx: Optional[Mapping[Node, int]] = None,
    cutoff: Optional[int] = None,
) -> Tuple[list[Node], Dict[Node, int], np.ndarray]:
    """
    Build a dense hop-distance matrix. Edge weights are deliberately ignored.

    If cutoff is provided, only shortest-path distances <= cutoff are computed;
    larger distances remain np.inf. This is useful for k-hop local measures,
    where residual-shell only needs distances up to 2*k+1 on graph edges.
    """
    if nodes is None or idx is None:
        nodes, idx = node_order_and_index(graph, nodes)
    else:
        nodes = list(nodes)
        idx = dict(idx)
    if cutoff is not None and cutoff < 0:
        raise ValueError("cutoff must be non-negative or None")

    n = len(nodes)
    distances = np.full((n, n), np.inf, dtype=np.float64)
    for source in nodes:
        dmap = nx.single_source_shortest_path_length(graph, source, cutoff=cutoff)
        source_index = idx[source]
        for target, distance in dmap.items():
            if target in idx:
                distances[source_index, idx[target]] = float(distance)
    return nodes, idx, distances


def _resolve_distance_cutoff(distance_cutoff: Union[str, int, None], k_hop: int) -> Optional[int]:
    if distance_cutoff == "auto":
        return 2 * int(k_hop) + 1
    if distance_cutoff is None:
        return None
    cutoff = int(distance_cutoff)
    if cutoff < 0:
        raise ValueError("distance_cutoff must be 'auto', None, or a non-negative integer")
    return cutoff


def build_lazy_measures(
    graph: nx.Graph,
    alpha_lazy: float = 0.4,
) -> Dict[Node, Dict[Node, float]]:
    """Build one-hop unweighted lazy random-walk measures."""
    if not 0.0 <= alpha_lazy <= 1.0:
        raise ValueError("alpha_lazy must be in [0, 1]")

    measures = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            measures[node] = {node: 1.0}
            continue
        neighbor_mass = (1.0 - alpha_lazy) / len(neighbors)
        measure = {node: float(alpha_lazy)}
        for neighbor in neighbors:
            measure[neighbor] = measure.get(neighbor, 0.0) + neighbor_mass
        measures[node] = measure
    return measures


def _edge_weight(graph: nx.Graph, u: Node, v: Node, weight_attr: str) -> float:
    data = graph.get_edge_data(u, v, default={})
    if graph.is_multigraph():
        weight = sum(float(edge_data.get(weight_attr, 1.0)) for edge_data in data.values())
    else:
        weight = float(data.get(weight_attr, 1.0))
    if weight < 0.0:
        raise ValueError(f"Negative edge weight for ({u}, {v}): {weight}")
    return weight


def weighted_transition_matrix(
    graph: nx.Graph,
    nodes: Optional[Sequence[Node]] = None,
    idx: Optional[Mapping[Node, int]] = None,
    weight_attr: str = "weight",
) -> np.ndarray:
    """
    Build a row-stochastic transition matrix.

    Edge weights affect transition probabilities only. The transport ground
    distance should be supplied separately, typically as unweighted hop distance.
    """
    if nodes is None or idx is None:
        nodes, idx = node_order_and_index(graph, nodes)
    else:
        nodes = list(nodes)
        idx = dict(idx)

    n = len(nodes)
    transition = np.zeros((n, n), dtype=np.float64)

    for u in nodes:
        u_index = idx[u]
        weighted_neighbors = []
        total_weight = 0.0
        for v in graph.neighbors(u):
            weight = _edge_weight(graph, u, v, weight_attr)
            if weight > 0.0:
                weighted_neighbors.append((v, weight))
                total_weight += weight

        if total_weight <= 0.0:
            transition[u_index, u_index] = 1.0
        else:
            inv_total = 1.0 / total_weight
            for v, weight in weighted_neighbors:
                transition[u_index, idx[v]] += weight * inv_total

    return transition


def build_m_hop_weighted_lazy_measures(
    graph: nx.Graph,
    k_hop: int,
    alpha_lazy: float = 0.4,
    nodes: Optional[Sequence[Node]] = None,
    idx: Optional[Mapping[Node, int]] = None,
    weight_attr: str = "weight",
    tol: float = 1e-15,
) -> Dict[Node, Dict[Node, float]]:
    """Build mu_x = alpha*delta_x + (1-alpha)*P^k[x, :]."""
    if k_hop < 0:
        raise ValueError("k_hop must be non-negative")
    if not 0.0 <= alpha_lazy <= 1.0:
        raise ValueError("alpha_lazy must be in [0, 1]")
    if nodes is None or idx is None:
        nodes, idx = node_order_and_index(graph, nodes)
    else:
        nodes = list(nodes)
        idx = dict(idx)

    transition = weighted_transition_matrix(
        graph,
        nodes=nodes,
        idx=idx,
        weight_attr=weight_attr,
    )
    transition_m = np.linalg.matrix_power(transition, int(k_hop))

    measures = {}
    for node in nodes:
        row_index = idx[node]
        row = (1.0 - alpha_lazy) * transition_m[row_index].copy()
        row[row_index] += alpha_lazy
        support = np.where(row > tol)[0]
        measures[node] = {nodes[j]: float(row[j]) for j in support}
    return measures


def build_one_hop_weighted_lazy_measures(
    graph: nx.Graph,
    alpha_lazy: float = 0.4,
    nodes: Optional[Sequence[Node]] = None,
    idx: Optional[Mapping[Node, int]] = None,
    weight_attr: str = "weight",
    tol: float = 1e-15,
) -> Dict[Node, Dict[Node, float]]:
    """Convenience wrapper for weighted k_hop=1 lazy measures."""
    return build_m_hop_weighted_lazy_measures(
        graph,
        k_hop=1,
        alpha_lazy=alpha_lazy,
        nodes=nodes,
        idx=idx,
        weight_attr=weight_attr,
        tol=tol,
    )


def measures_to_csr(
    measures: Mapping[Node, Measure],
    nodes: Optional[Sequence[Node]] = None,
    idx: Optional[Mapping[Node, int]] = None,
    tol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert dict-of-dicts measures to CSR arrays accepted by the C++ extension."""
    if nodes is None:
        nodes = list(measures.keys())
    else:
        nodes = list(nodes)
    if idx is None:
        idx = {node: i for i, node in enumerate(nodes)}
    else:
        idx = dict(idx)

    indptr = [0]
    indices = []
    values = []

    for node in nodes:
        measure = measures[node]
        total_mass = 0.0
        row_items = []
        for support_node, mass_value in measure.items():
            mass = float(mass_value)
            if mass > tol:
                if support_node not in idx:
                    raise ValueError(f"Measure for {node!r} contains unknown node {support_node!r}")
                row_items.append((idx[support_node], mass))
                total_mass += mass
        indices.extend(index for index, _ in row_items)
        values.extend(mass for _, mass in row_items)
        indptr.append(len(indices))

        if not np.isclose(total_mass, 1.0, atol=1e-8):
            raise ValueError(f"Measure for node {node!r} sums to {total_mass}, not 1.0")

    return (
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def edges_to_index_array(edges: Iterable[Tuple[Node, Node]], idx: Mapping[Node, int]) -> np.ndarray:
    """Convert graph node-id edges to contiguous row-index edges."""
    return np.asarray([(idx[u], idx[v]) for u, v in edges], dtype=np.int64)


def symmetric_edge_matrix(edge_indices: np.ndarray, edge_values: np.ndarray, num_nodes: int) -> csr_matrix:
    """Build an n x n symmetric CSR matrix from one value per undirected edge."""
    edge_indices = np.asarray(edge_indices, dtype=np.int64)
    edge_values = np.asarray(edge_values, dtype=np.float64)
    rows = np.concatenate((edge_indices[:, 0], edge_indices[:, 1]))
    cols = np.concatenate((edge_indices[:, 1], edge_indices[:, 0]))
    values = np.concatenate((edge_values, edge_values))
    matrix = csr_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float64)
    matrix.eliminate_zeros()
    return matrix


def compute_residual_shell_from_measures(
    edges: Iterable[Tuple[Node, Node]],
    measures: Mapping[Node, Measure],
    distances: np.ndarray,
    nodes: Sequence[Node],
    idx: Mapping[Node, int],
    l_shell: int = 2,
    rbar_mode: str = "local-max",
    tol: float = 1e-12,
    num_threads: int = 0,
    progress_interval: int = 0,
) -> dict:
    """Compute residual-shell W1 upper bounds and curvature values."""
    edges = list(edges)
    edge_indices = edges_to_index_array(edges, idx)
    indptr, measure_indices, measure_values = measures_to_csr(
        measures,
        nodes=nodes,
        idx=idx,
        tol=0.0,
    )

    result = _residual_shell_cpp.compute_residual_shell(
        edge_indices,
        indptr,
        measure_indices,
        measure_values,
        np.asarray(distances, dtype=np.float64, order="C"),
        int(l_shell),
        str(rbar_mode),
        float(tol),
        int(num_threads),
        int(progress_interval),
    )

    curvatures = np.asarray(result["curvatures"], dtype=np.float64)
    result["edge_indices"] = edge_indices
    result["curvature_matrix"] = symmetric_edge_matrix(edge_indices, curvatures, len(nodes))
    return result


def compute_fast_algo1_from_measures(
    edges: Iterable[Tuple[Node, Node]],
    measures: Mapping[Node, Measure],
    nodes: Sequence[Node],
    idx: Mapping[Node, int],
    tol: float = 1e-12,
    num_threads: int = 0,
) -> dict:
    """Compute corrected Fast Algorithm 1 W1 upper bounds and curvature values."""
    edges = list(edges)
    edge_indices = edges_to_index_array(edges, idx)
    indptr, measure_indices, measure_values = measures_to_csr(
        measures,
        nodes=nodes,
        idx=idx,
        tol=0.0,
    )

    result = _residual_shell_cpp.compute_fast_algo1(
        edge_indices,
        indptr,
        measure_indices,
        measure_values,
        float(tol),
        int(num_threads),
    )

    curvatures = np.asarray(result["curvatures"], dtype=np.float64)
    result["edge_indices"] = edge_indices
    result["curvature_matrix"] = symmetric_edge_matrix(edge_indices, curvatures, len(nodes))
    return result


def residual_shell_based_orc_bound(
    graph: nx.Graph,
    k_hop: int = 1,
    alpha_lazy: float = 0.4,
    l_shell: int = 2,
    rbar_mode: str = "local-max",
    tol: float = 1e-12,
    measure_tol: float = 1e-15,
    weight_attr: str = "weight",
    num_threads: int = 0,
    progress_interval: int = 0,
    distance_cutoff: Union[str, int, None] = "auto",
    nodes: Optional[Sequence[Node]] = None,
    edges: Optional[Iterable[Tuple[Node, Node]]] = None,
) -> dict:
    """End-to-end weighted-transition, unweighted-distance residual-shell helper."""
    nodes, idx = node_order_and_index(graph, nodes)
    edges = list(graph.edges() if edges is None else edges)
    resolved_cutoff = _resolve_distance_cutoff(distance_cutoff, k_hop)
    _, _, distances = all_pairs_unweighted_shortest_path_matrix(
        graph,
        nodes=nodes,
        idx=idx,
        cutoff=resolved_cutoff,
    )
    measures = build_m_hop_weighted_lazy_measures(
        graph,
        k_hop=k_hop,
        alpha_lazy=alpha_lazy,
        nodes=nodes,
        idx=idx,
        weight_attr=weight_attr,
        tol=measure_tol,
    )

    started_at = time.perf_counter()
    result = compute_residual_shell_from_measures(
        edges=edges,
        measures=measures,
        distances=distances,
        nodes=nodes,
        idx=idx,
        l_shell=l_shell,
        rbar_mode=rbar_mode,
        tol=tol,
        num_threads=num_threads,
        progress_interval=progress_interval,
    )
    result["python_total_seconds"] = time.perf_counter() - started_at
    result["nodes"] = nodes
    result["node_index"] = idx
    result["distances"] = distances
    result["distance_cutoff"] = resolved_cutoff
    result["measures"] = measures
    return result


def compute_weighted_one_hop_fast_algo1(
    graph: nx.Graph,
    alpha_lazy: float = 0.4,
    tol: float = 1e-12,
    measure_tol: float = 1e-15,
    weight_attr: str = "weight",
    num_threads: int = 0,
    nodes: Optional[Sequence[Node]] = None,
    edges: Optional[Iterable[Tuple[Node, Node]]] = None,
) -> dict:
    """End-to-end weighted one-hop Fast Algorithm 1 helper."""
    nodes, idx = node_order_and_index(graph, nodes)
    edges = list(graph.edges() if edges is None else edges)
    measures = build_one_hop_weighted_lazy_measures(
        graph,
        alpha_lazy=alpha_lazy,
        nodes=nodes,
        idx=idx,
        weight_attr=weight_attr,
        tol=measure_tol,
    )

    started_at = time.perf_counter()
    result = compute_fast_algo1_from_measures(
        edges=edges,
        measures=measures,
        nodes=nodes,
        idx=idx,
        tol=tol,
        num_threads=num_threads,
    )
    result["python_total_seconds"] = time.perf_counter() - started_at
    result["nodes"] = nodes
    result["node_index"] = idx
    result["measures"] = measures
    return result


def exact_w1_ot(
    mu_x: Measure,
    mu_y: Measure,
    distances: np.ndarray,
    idx: Mapping[Node, int],
    tol: float = 1e-12,
    num_threads: int = 1,
    num_iter_max: int = 100000,
) -> float:
    """
    Compute exact W1 with POT when users need an exact baseline.

    POT is an optional dependency and is intentionally not required at install time.
    """
    try:
        import ot
    except ImportError as error:
        raise ImportError("Install POT to use exact_w1_ot: python -m pip install POT") from error

    support_x = [node for node, mass in mu_x.items() if mass > tol]
    support_y = [node for node, mass in mu_y.items() if mass > tol]
    if not support_x or not support_y:
        raise ValueError("Both measures must contain positive mass")

    mass_x = np.asarray([mu_x[node] for node in support_x], dtype=np.float64)
    mass_y = np.asarray([mu_y[node] for node in support_y], dtype=np.float64)
    if np.any(~np.isfinite(mass_x)) or np.any(~np.isfinite(mass_y)):
        raise ValueError("Measure masses must be finite")
    if np.any(mass_x < 0.0) or np.any(mass_y < 0.0):
        raise ValueError("Measure masses must be non-negative")

    total_x = float(np.sum(mass_x))
    total_y = float(np.sum(mass_y))
    if total_x <= tol or total_y <= tol:
        raise ValueError("Both measures must have positive total mass")
    if not np.isclose(total_x, total_y, atol=tol, rtol=tol):
        raise ValueError(f"Measure masses differ: source={total_x}, target={total_y}")

    common_mass = 0.5 * (total_x + total_y)
    mass_x = np.ascontiguousarray(mass_x * (common_mass / total_x))
    mass_y = np.ascontiguousarray(mass_y * (common_mass / total_y))

    rows = np.asarray([idx[node] for node in support_x], dtype=np.int64)
    cols = np.asarray([idx[node] for node in support_y], dtype=np.int64)
    cost = np.ascontiguousarray(np.asarray(distances, dtype=np.float64)[np.ix_(rows, cols)])
    if np.any(~np.isfinite(cost)):
        raise ValueError("Transport cost matrix contains non-finite distances")

    kwargs = {
        "numItermax": int(num_iter_max),
        "log": False,
        "numThreads": int(num_threads),
        "check_marginals": True,
    }
    try:
        return float(ot.emd2(mass_x, mass_y, cost, **kwargs))
    except TypeError as error:
        if "numThreads" not in str(error):
            raise
        kwargs.pop("numThreads")
        return float(ot.emd2(mass_x, mass_y, cost, **kwargs))
