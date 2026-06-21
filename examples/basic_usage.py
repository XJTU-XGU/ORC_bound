import networkx as nx

from orc_bound import (
    build_lazy_measures,
    all_pairs_unweighted_shortest_path_matrix,
    compute_fast_algo1_from_measures,
    compute_residual_shell_from_measures,
)


def main():
    graph = nx.cycle_graph(6)
    nodes, idx, distances = all_pairs_unweighted_shortest_path_matrix(graph)
    measures = build_lazy_measures(graph, alpha_lazy=0.4)
    edges = list(graph.edges())

    residual = compute_residual_shell_from_measures(
        edges=edges,
        measures=measures,
        distances=distances,
        nodes=nodes,
        idx=idx,
        l_shell=2,
        num_threads=2,
        progress_interval=2,
    )
    fast = compute_fast_algo1_from_measures(
        edges=edges,
        measures=measures,
        nodes=nodes,
        idx=idx,
        num_threads=2,
    )

    print("residual curvatures:", residual["curvatures"])
    print("fast curvatures:", fast["curvatures"])
    print("threads:", residual["num_threads"])


if __name__ == "__main__":
    main()



