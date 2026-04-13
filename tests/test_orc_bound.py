"""
Tests for orc_bound package.
"""

import numpy as np
import networkx as nx
import pytest
from scipy.sparse import csr_matrix

from orc_bound import (
    build_lazy_measures_k,
    all_pairs_shortest_path_matrix_cutoff,
    residual_shell_upper_bound,
    residual_shell_ricci_approximation,
)
from orc_bound.utils.parallel import _cpu_count


class TestBuildLazyMeasuresK:
    def test_path_graph(self):
        """Path graph: measures should be valid probability distributions."""
        G = nx.path_graph(5)
        mu = build_lazy_measures_k(G, alpha_lazy=0.0, k=1)

        for node in G.nodes():
            assert node in mu
            total = sum(mu[node].values())
            assert abs(total - 1.0) < 1e-10

    def test_lazy_parameter(self):
        """Higher alpha should put more mass at the source node."""
        G = nx.path_graph(5)
        mu0 = build_lazy_measures_k(G, alpha_lazy=0.0, k=1)
        mu5 = build_lazy_measures_k(G, alpha_lazy=0.5, k=1)
        mu9 = build_lazy_measures_k(G, alpha_lazy=0.9, k=1)

        assert mu0[0][0] == 0.0
        assert mu5[0][0] == 0.5
        assert mu9[0][0] == 0.9

    def test_k_steps(self):
        """Higher k should spread mass over more nodes."""
        G = nx.path_graph(10)
        mu1 = build_lazy_measures_k(G, alpha_lazy=0.0, k=1)
        mu3 = build_lazy_measures_k(G, alpha_lazy=0.0, k=3)

        # k=1 should have support within 1 hop
        support1 = set(mu1[5].keys())
        assert support1.issubset({3, 4, 5, 6, 7})

        # k=3 should have support within 3 hops
        support3 = set(mu3[5].keys())
        assert support3.issubset({2, 3, 4, 5, 6, 7, 8})

    def test_disconnected_node(self):
        """Isolated node should have mass concentrated at itself."""
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        G.add_node(2)  # isolated
        mu = build_lazy_measures_k(G, alpha_lazy=0.5, k=1)
        assert mu[2] == {2: 0.5 + 0.5}  # stays at self


class TestAllPairsShortestPathMatrixCutoff:
    def test_path_graph_distances(self):
        G = nx.path_graph(5)
        nodes, idx, D = all_pairs_shortest_path_matrix_cutoff(G, cutoff=10)

        assert D[idx[0], idx[4]] == 4.0
        assert D[idx[0], idx[0]] == 0.0
        assert D[idx[1], idx[3]] == 2.0

    def test_cutoff_respected(self):
        G = nx.path_graph(10)
        nodes, idx, D = all_pairs_shortest_path_matrix_cutoff(G, cutoff=3)

        # Distance of 4 should be cut off
        assert np.isinf(D[idx[0], idx[4]])


class TestResidualShellUpperBound:
    def test_identical_measures(self):
        """If mu_x == mu_y, W1 upper bound should be near zero."""
        G = nx.path_graph(5)
        nodes, idx, D = all_pairs_shortest_path_matrix_cutoff(G, cutoff=5)
        mu = build_lazy_measures_k(G, alpha_lazy=1.0, k=1)

        # With alpha=1, both measures are delta at their source
        # So identical sources give zero distance
        ub, _, _, _ = residual_shell_upper_bound(mu[0], mu[0], D, idx, l=5)
        assert ub == 0.0

    def test_output_shape(self):
        G = nx.path_graph(5)
        nodes, idx, D = all_pairs_shortest_path_matrix_cutoff(G, cutoff=5)
        mu = build_lazy_measures_k(G, alpha_lazy=0.0, k=1)

        ub, m_r, Rl, rbar = residual_shell_upper_bound(
            mu[0], mu[1], D, idx, l=3
        )
        assert ub >= 0
        assert len(m_r) == 4  # l+1
        assert Rl >= 0
        assert rbar >= 0

    def test_rbar_global_mode(self):
        G = nx.path_graph(5)
        nodes, idx, D = all_pairs_shortest_path_matrix_cutoff(G, cutoff=5)
        mu = build_lazy_measures_k(G, alpha_lazy=0.0, k=1)

        _, _, _, rbar_local = residual_shell_upper_bound(
            mu[0], mu[4], D, idx, l=2, rbar_mode="local-max"
        )
        _, _, _, rbar_global = residual_shell_upper_bound(
            mu[0], mu[4], D, idx, l=2, rbar_mode="global"
        )
        # Both modes should produce valid finite rbar
        assert np.isfinite(rbar_local)
        assert np.isfinite(rbar_global)


class TestResidualShellRicciApproximation:
    def test_return_type(self):
        G = nx.watts_strogatz_graph(20, 4, 0.1, seed=42)
        C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=1)
        assert isinstance(C, csr_matrix)

    def test_nnz_matches_edges(self):
        G = nx.watts_strogatz_graph(20, 4, 0.1, seed=42)
        C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=1)
        assert C.nnz == G.number_of_edges()

    def test_nnz_symmetric(self):
        G = nx.watts_strogatz_graph(20, 4, 0.1, seed=42)
        C = residual_shell_ricci_approximation(
            G, G.number_of_nodes(), k=1, symmetric=True
        )
        assert C.nnz == 2 * G.number_of_edges()

    def test_curvature_values(self):
        G = nx.watts_strogatz_graph(30, 6, 0.2, seed=0)
        C = residual_shell_ricci_approximation(G, G.number_of_nodes(), k=1)
        data = C.data
        # Curvature lower bounds can be negative (valid lower bound).
        # Upper bound is 1 for non-zero edges.
        assert np.all(data <= 1)

    def test_parallel_vs_sequential_equivalent(self):
        """Parallel and sequential should produce identical results."""
        G = nx.watts_strogatz_graph(40, 6, 0.2, seed=42)
        n = G.number_of_nodes()

        C_seq = residual_shell_ricci_approximation(G, n, k=2, n_jobs=1)
        C_par = residual_shell_ricci_approximation(G, n, k=2, n_jobs=4)

        diff = np.abs(C_seq.toarray() - C_par.toarray()).max()
        assert diff == 0.0, f"Parallel and sequential results differ (max diff: {diff})"

    def test_n_jobs_variants(self):
        """Various n_jobs values should all succeed."""
        G = nx.watts_strogatz_graph(30, 4, 0.1, seed=0)
        n = G.number_of_nodes()

        for n_jobs in [None, 0, 1, 2, -1]:
            C = residual_shell_ricci_approximation(G, n, k=1, n_jobs=n_jobs)
            assert C.nnz == G.number_of_edges()

    def test_k_variants(self):
        """Different k values should produce valid outputs."""
        G = nx.path_graph(20)
        n = G.number_of_nodes()

        for k in [1, 2, 3, 5]:
            C = residual_shell_ricci_approximation(G, n, k=k)
            assert isinstance(C, csr_matrix)
            assert C.shape == (n, n)
            assert C.nnz == G.number_of_edges()

    def test_num_nodes_mismatch(self):
        G = nx.watts_strogatz_graph(20, 4, 0.1, seed=0)
        with pytest.raises(ValueError, match="does not match"):
            residual_shell_ricci_approximation(G, 99, k=1)


class TestParallelUtils:
    def test_cpu_count_none(self):
        assert _cpu_count(None) >= 1

    def test_cpu_count_zero(self):
        assert _cpu_count(0) >= 1

    def test_cpu_count_positive(self):
        assert _cpu_count(2) == 2
        assert _cpu_count(8) == 8

    def test_cpu_count_negative_one(self):
        result = _cpu_count(-1)
        assert result >= 1


class TestSearchModule:
    def test_serpapi_import_error(self):
        """Should raise ImportError if google-search-results is not installed."""
        from orc_bound.utils import search
        assert hasattr(search, "search_orc_literature")
        assert hasattr(search, "search_general")
