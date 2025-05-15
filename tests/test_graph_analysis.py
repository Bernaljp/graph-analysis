import unittest
import itertools
import numpy as np
import time
import torch
from graph_analysis import WeightedDigraph, get_canonical_form, run_simulation, precompute_classes, compute_node_invariants, visualize_graphs, graph_features

class TestGraphAnalysis(unittest.TestCase):
    def test_weighted_digraph_validation(self):
        """Test WeightedDigraph validation for valid and invalid matrices."""
        valid_matrix = [[1, 1, 0], [0, 0, -1], [0, 0, 0]]
        invalid_matrix = [[2, 0], [0, 0]]  # Invalid weight
        self.assertIsInstance(WeightedDigraph(valid_matrix), WeightedDigraph)
        with self.assertRaises(ValueError):
            WeightedDigraph(invalid_matrix)

    def test_canonical_form(self):
        """Test that isomorphic graphs have the same canonical form."""
        matrix1 = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
        matrix2 = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
        G1 = WeightedDigraph(matrix1)
        G2 = WeightedDigraph(matrix2)
        self.assertEqual(get_canonical_form(G1), get_canonical_form(G2))

    def test_simulation(self):
        """Test that run_simulation produces output with expected shape."""
        adj = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        y0 = torch.tensor([0.5, 0.5, 0.5])
        ts = torch.linspace(0, 50, 200)
        ts_out, arr = run_simulation(adj, y0, ts)
        self.assertEqual(arr.shape, (200, 3))  # 200 time steps, 3 nodes
        self.assertTrue(np.allclose(ts_out, ts.numpy()))
    
    def test_compute_node_invariants(self):
        matrix = [[1, 1, 0], [0, 0, -1], [0, 0, 0]]
        G = WeightedDigraph(matrix)
        invariants = compute_node_invariants(G)
        expected = [(1, 1, 0, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 1)]
        self.assertEqual(invariants, expected)
    
    def test_precompute_classes(self):
        graphs = precompute_classes(n=2)
        self.assertGreater(len(graphs), 0)
        for matrix in graphs:
            self.assertIsInstance(WeightedDigraph(matrix), WeightedDigraph)
    
    def test_visualize_graphs(self):
        graphs = precompute_classes(n=2)
        feats = np.random.rand(len(graphs), 2)  # Mock PCA data
        fw, ui, run_btn, rand_btn, init_boxes = visualize_graphs(graphs, feats, max_points=10)
        self.assertEqual(len(fw._sampled_graphs), min(10, len(graphs)))
    
    def test_weighted_digraph_validation(self):
        valid_matrix = [[1, 1, 0], [0, 0, -1], [0, 0, 0]]
        invalid_matrix = [[2, 0], [0, 0]]  # Invalid weight
        self.assertIsInstance(WeightedDigraph(valid_matrix, verify=True), WeightedDigraph)
        self.assertIsInstance(WeightedDigraph(valid_matrix, verify=False), WeightedDigraph)
        with self.assertRaises(ValueError):
            WeightedDigraph(invalid_matrix, verify=True)
        # No error with verify=False, even for invalid matrix
        self.assertIsInstance(WeightedDigraph(invalid_matrix, verify=False), WeightedDigraph)
    
    def test_graph_features(self):
        """Test graph_features with and without custom features."""
        adj_matrices = np.array([[[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 1], [0, 0, 0]]])
        n = 3
        num_inits = 2

        # Test default features (mean and std)
        feats = graph_features(adj_matrices, num_inits=num_inits, batch_size=2)
        self.assertEqual(feats.shape, (2, 2 * n))  # 2 graphs, 2*n features

        # Test with custom feature (e.g., max value)
        def max_feature(ys: torch.Tensor) -> np.ndarray:
            return ys.max(dim=2)[0].mean(dim=0).cpu().numpy()  # Max over time, mean over inits
        feats_custom = graph_features(adj_matrices, num_inits=num_inits, batch_size=2, custom_features=[max_feature])
        self.assertEqual(feats_custom.shape, (2, 2 * n + n))  # 2*n default + n custom
    
    def test_precompute_classes_connectivity(self):
        """Test that precompute_classes excludes disconnected graphs."""
        graphs = precompute_classes(2)
        for M in graphs:
            n = M.shape[0]
            adj = {i: set() for i in range(n)}
            for i in range(n):
                for j in range(n):
                    if i != j and (M[i, j] != 0 or M[j, i] != 0):
                        adj[i].add(j)
                        adj[j].add(i)
            visited = set()
            stack = [0]
            while stack:
                u = stack.pop()
                if u not in visited:
                    visited.add(u)
                    for v in adj[u]:
                        if v not in visited:
                            stack.append(v)
            self.assertEqual(len(visited), n, f"Graph is disconnected: {M}")

    def test_precompute_classes_excludes_disconnected_self_loop(self):
        """Test that precompute_classes excludes graphs with disconnected nodes having only self-loops."""
        graphs = precompute_classes(2)
        disconnected_matrix = np.array([[1, 0], [0, 0]])  # Node 0 has self-loop, disconnected
        found = any(np.array_equal(M, disconnected_matrix) for M in graphs)
        self.assertFalse(found, f"Disconnected matrix with self-loop found in graphs: {disconnected_matrix}")

    def test_connectivity_check_performance(self):
        """Benchmark BFS vs zero row check for connectivity."""
        def bfs_check(matrix: np.ndarray) -> bool:
            n = matrix.shape[0]
            adj = {i: set() for i in range(n)}
            for i in range(n):
                for j in range(n):
                    if i != j and (matrix[i, j] != 0 or matrix[j, i] != 0):
                        adj[i].add(j)
                        adj[j].add(i)
            visited = set()
            stack = [0]
            while stack:
                u = stack.pop()
                if u not in visited:
                    visited.add(u)
                    for v in adj[u]:
                        if v not in visited:
                            stack.append(v)
            return len(visited) == n

        def zero_row_check(matrix: np.ndarray) -> bool:
            n = matrix.shape[0]
            M = matrix + matrix.T - 2 * np.eye(n, dtype=int)
            row_sums = np.sum(np.abs(M), axis=1)
            return not np.any(row_sums == 0)

        def hybrid_check(matrix: np.ndarray) -> bool:
            n = matrix.shape[0]
            M = matrix + matrix.T - 2 * np.eye(n, dtype=int)
            row_sums = np.sum(np.abs(M), axis=1)
            if np.any(row_sums == 0):
                return False
            adj = {i: set() for i in range(n)}
            for i in range(n):
                for j in range(n):
                    if i != j and (matrix[i, j] != 0 or matrix[j, i] != 0):
                        adj[i].add(j)
                        adj[j].add(i)
            visited = set()
            stack = [0]
            while stack:
                u = stack.pop()
                if u not in visited:
                    visited.add(u)
                    for v in adj[u]:
                        if v not in visited:
                            stack.append(v)
            return len(visited) == n

        for n in [2, 3, 4]:
            edge_positions = [(i, j) for i in range(n) for j in range(n) if i != j]
            num_edges = len(edge_positions)
            loops = np.array(list(itertools.product([0, 1], repeat=n)))
            off_vals = np.array(list(itertools.product([-1, 0, 1], repeat=num_edges)))
            num_loops, num_off = len(loops), len(off_vals)
            total_matrices = num_loops * num_off
            matrices = np.zeros((total_matrices, n, n), dtype=int)
            loop_idx = np.repeat(np.arange(num_loops), num_off)
            off_idx = np.tile(np.arange(num_off), num_loops)
            for i in range(n):
                matrices[:, i, i] = loops[loop_idx, i]
            for k, (i, j) in enumerate(edge_positions):
                matrices[:, i, j] = off_vals[off_idx, k]

            # Benchmark BFS
            start = time.perf_counter()
            bfs_results = [bfs_check(M) for M in matrices]
            bfs_time = time.perf_counter() - start

            # Benchmark Zero Row
            start = time.perf_counter()
            zero_results = [zero_row_check(M) for M in matrices]
            zero_time = time.perf_counter() - start

            # Benchmark Hybrid
            start = time.perf_counter()
            hybrid_results = [hybrid_check(M) for M in matrices]
            hybrid_time = time.perf_counter() - start

            # Check correctness
            discrepancies = sum(a != b for a, b in zip(bfs_results, zero_results))
            self.assertTrue(discrepancies > 0, f"Zero row check should miss some disconnected graphs for n={n}")
            self.assertEqual(bfs_results, hybrid_results, f"Hybrid check should match BFS for n={n}")

            print(f"n={n}: BFS time={bfs_time:.4f}s, Zero row time={zero_time:.4f}s, Hybrid time={hybrid_time:.4f}s")

if __name__ == '__main__':
    unittest.main()