import unittest
import numpy as np
import torch
from graph_analysis import WeightedDigraph, get_canonical_form, run_simulation, precompute_classes, compute_node_invariants, visualize_graphs

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

if __name__ == '__main__':
    unittest.main()