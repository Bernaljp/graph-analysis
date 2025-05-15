import unittest
import numpy as np
import torch
import time
import itertools
from graph_analysis import WeightedDigraph, get_canonical_form, run_simulation, graph_features, precompute_classes, visualize_graphs

class TestGraphAnalysis(unittest.TestCase):
    def test_weighted_digraph_validation(self):
        """Test WeightedDigraph validation for valid and invalid matrices."""
        valid_matrix = [[1, 1], [0, 0]]  # Valid, connected
        invalid_matrix = [[2, 0], [0, 0]]  # Invalid weight
        disconnected_matrix = [[1, 0], [0, 0]]  # Node 1 disconnected
        self.assertIsInstance(WeightedDigraph(valid_matrix, verify=True), WeightedDigraph)
        self.assertIsInstance(WeightedDigraph(valid_matrix, verify=False), WeightedDigraph)
        with self.assertRaises(ValueError):
            WeightedDigraph(invalid_matrix, verify=True)
        self.assertIsInstance(WeightedDigraph(invalid_matrix, verify=False), WeightedDigraph)
        with self.assertRaises(ValueError):
            WeightedDigraph(disconnected_matrix, verify=True)
        self.assertIsInstance(WeightedDigraph(disconnected_matrix, verify=False), WeightedDigraph)

    def test_canonical_form(self):
        """Test that isomorphic graphs have the same canonical form."""
        matrix1 = [[0, 1], [-1, 0]]  # Edge 0->1 (+1), 1->0 (-1)
        matrix2 = [[0, -1], [1, 0]]  # Relabeled (isomorphic)
        G1 = WeightedDigraph(matrix1, verify=True)
        G2 = WeightedDigraph(matrix2, verify=True)
        self.assertEqual(get_canonical_form(G1), get_canonical_form(G2))

    def test_simulation(self):
        """Test that run_simulation produces output with expected shape."""
        adj = np.array([[0, 1], [-1, 0]])
        y0 = torch.tensor([0.5, 0.5])
        ts = torch.linspace(0, 50, 200)
        ts_out, arr = run_simulation(adj, y0, ts)
        self.assertEqual(arr.shape, (200, 2))  # 200 time steps, 2 nodes
        self.assertTrue(np.allclose(ts_out, ts.numpy()))

    def test_simulation_additive(self):
        """Test that run_simulation works with additive model."""
        adj = np.array([[0, 1], [-1, 0]])
        y0 = torch.tensor([0.5, 0.5])
        ts = torch.linspace(0, 50, 200)
        ts_out, arr = run_simulation(adj, y0, ts, model_type='additive')
        self.assertEqual(arr.shape, (200, 2))  # 200 time steps, 2 nodes
        self.assertTrue(np.allclose(ts_out, ts.numpy()))
        self.assertFalse(np.any(np.isnan(arr)), "Additive model produced NaN values")

    def test_graph_features(self):
        """Test graph_features with and without custom features."""
        adj_matrices = np.array([[[0, 1], [-1, 0]], [[0, -1], [1, 0]]])
        n = 2
        num_inits = 2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feats = graph_features(adj_matrices, num_inits=num_inits, batch_size=2, device=device)
        self.assertEqual(feats.shape, (2, 2 * n))
        def max_feature(ys: torch.Tensor) -> np.ndarray:
            return ys.max(dim=2)[0].mean(dim=0).cpu().numpy()
        feats_custom = graph_features(adj_matrices, num_inits=num_inits, batch_size=2, custom_features=[max_feature], device=device)
        self.assertEqual(feats_custom.shape, (2, 2 * n + n))

    def test_graph_features_additive(self):
        """Test graph_features with additive model."""
        adj_matrices = np.array([[[0, 1], [-1, 0]], [[0, -1], [1, 0]]])
        n = 2
        num_inits = 2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feats = graph_features(adj_matrices, num_inits=num_inits, batch_size=2, device=device, model_type='additive')
        self.assertEqual(feats.shape, (2, 2 * n))
        self.assertFalse(np.any(np.isnan(feats)), "Additive model produced NaN values")

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
        disconnected_matrix = np.array([[1, 0], [0, 0]])  # Node 1 disconnected
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
            M = np.abs(matrix) + np.abs(matrix.T) - 2 * np.eye(n, dtype=int)
            row_sums = np.sum(np.abs(M), axis=1)
            return not np.any(row_sums == 0)

        def hybrid_check(matrix: np.ndarray) -> bool:
            n = matrix.shape[0]
            M = np.abs(matrix) + np.abs(matrix.T) - 2 * np.eye(n, dtype=int)
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

        for n in [2, 3]:
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

            start = time.perf_counter()
            bfs_results = [bfs_check(M) for M in matrices]
            bfs_time = time.perf_counter() - start

            start = time.perf_counter()
            zero_results = [zero_row_check(M) for M in matrices]
            zero_time = time.perf_counter() - start

            start = time.perf_counter()
            hybrid_results = [hybrid_check(M) for M in matrices]
            hybrid_time = time.perf_counter() - start

            for idx, (bfs_res, hybrid_res) in enumerate(zip(bfs_results, hybrid_results)):
                if bfs_res != hybrid_res:
                    print(f"Discrepancy at matrix {idx}:\n{matrices[idx]}\nBFS: {bfs_res}, Hybrid: {hybrid_res}")

            discrepancies = sum(a != b for a, b in zip(bfs_results, zero_results))
            self.assertTrue(discrepancies > 0, f"Zero row check should miss some disconnected graphs for n={n}")
            self.assertEqual(bfs_results, hybrid_results, f"Hybrid check should match BFS for n={n}")

            print(f"n={n}: BFS time={bfs_time:.4f}s, Zero row time={zero_time:.4f}s, Hybrid time={hybrid_time:.4f}s")

if __name__ == '__main__':
    unittest.main()