from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.io import loadmat

from orc_bound import residual_shell_based_orc_bound


def default_facebook_path() -> Path:
    # examples/ -> package root/ -> repo root/
    return Path(__file__).resolve().parents[2] / "datasets" / "Facebook.mat"


def parse_distance_cutoff(value: str):
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"none", "full"}:
        return None
    return int(normalized)


def load_facebook_graph(mat_path: Path) -> nx.Graph:
    data = loadmat(mat_path)
    if "Network" not in data:
        raise KeyError(f"{mat_path} does not contain a 'Network' matrix")

    adjacency = data["Network"]
    if not sparse.issparse(adjacency):
        adjacency = sparse.csr_matrix(adjacency)
    else:
        adjacency = adjacency.tocsr()

    # The Facebook matrix is treated as an unweighted undirected graph.
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    try:
        graph = nx.from_scipy_sparse_array(adjacency, create_using=nx.Graph)
    except AttributeError:
        graph = nx.from_scipy_sparse_matrix(adjacency, create_using=nx.Graph)

    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def save_edge_curvatures(
    output_path: Path,
    edges: list[tuple[int, int]],
    curvatures: np.ndarray,
    w1_upper_bounds: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "target", "curvature", "w1_upper_bound"])
        for (source, target), curvature, w1 in zip(edges, curvatures, w1_upper_bounds):
            writer.writerow([source, target, float(curvature), float(w1)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run orc_bound residual-shell curvature on Facebook.mat."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_facebook_path(),
        help="Path to Facebook.mat. Defaults to ../datasets/Facebook.mat from this repo.",
    )
    parser.add_argument("--k-hop", type=int, default=1)
    parser.add_argument("--alpha-lazy", type=float, default=0.4)
    parser.add_argument("--l-shell", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0, help="0 uses OpenMP default.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print C++ progress every N edges. Use 0 to disable.",
    )
    parser.add_argument(
        "--distance-cutoff",
        type=parse_distance_cutoff,
        default="auto",
        help="'auto' uses 2*k+1, 'full' computes all distances, or pass an integer cutoff.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of edges to compute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for edge curvatures.",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    graph = load_facebook_graph(args.dataset)
    load_seconds = time.perf_counter() - start

    edges: Optional[list[tuple[int, int]]] = None
    if args.max_edges is not None:
        if args.max_edges <= 0:
            raise ValueError("--max-edges must be positive")
        edges = list(graph.edges())[: args.max_edges]

    print(f"dataset: {args.dataset}")
    print(f"nodes: {graph.number_of_nodes()}")
    print(f"edges: {graph.number_of_edges()}")
    print(f"load_seconds: {load_seconds:.3f}")
    if edges is not None:
        print(f"edge_limit: {len(edges)}")

    result = residual_shell_based_orc_bound(
        graph,
        k_hop=args.k_hop,
        alpha_lazy=args.alpha_lazy,
        l_shell=args.l_shell,
        num_threads=args.threads,
        progress_interval=args.progress_interval,
        distance_cutoff=args.distance_cutoff,
        edges=edges,
    )

    curvatures = result["curvatures"]
    w1_upper_bounds = result["w1_upper_bounds"]
    stats = summarize(curvatures)

    print(f"computed_edges: {len(curvatures)}")
    print(f"distance_cutoff: {result['distance_cutoff']}")
    print(f"threads: {result['num_threads']}")
    print(f"compute_seconds: {result['elapsed_seconds']:.3f}")
    print(
        "curvature_stats: "
        f"min={stats['min']:.8g}, p25={stats['p25']:.8g}, "
        f"mean={stats['mean']:.8g}, median={stats['median']:.8g}, "
        f"p75={stats['p75']:.8g}, max={stats['max']:.8g}"
    )

    if args.output is not None:
        output_edges = edges if edges is not None else list(graph.edges())
        save_edge_curvatures(args.output, output_edges, curvatures, w1_upper_bounds)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()


