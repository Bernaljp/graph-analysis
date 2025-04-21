import itertools
import copy
from collections import defaultdict
import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import Callable, List, Tuple, Dict
from sklearn.decomposition import PCA
import umap
import torchode as to
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Configuration class for storing simulation and visualization parameters."""
    K_LOOP = 0.2
    K_EDGE = 0.3
    N_HILL = 3
    T_SPAN = (0.0, 50.0)
    NUM_STEPS = 200
    MAX_POINTS = 1500
    MARKER_SIZE = 10
    NODE_SIZE = 20
    PLOT_HEIGHT = 500
    PLOT_WIDTH = 1200
    EDGE_WIDTH = 2
    COLORS = {'positive': 'RoyalBlue', 'negative': 'Crimson', 'default': 'steelblue', 'selected': 'crimson'}

def get_device(preferred_device: str = 'cuda') -> str:
    """Returns the appropriate device based on availability.

    Args:
        preferred_device: The preferred device, defaults to 'cuda'.

    Returns:
        The available device ('cuda' if available, else 'cpu').
    """
    return preferred_device if torch.cuda.is_available() else 'cpu'

class WeightedDigraph:
    """A class representing a weighted directed graph with specific constraints.

    Attributes:
        n: Number of nodes in the graph.
        matrix: Adjacency matrix as a numpy array.

    Args:
        adjacency_matrix: A square matrix where entry [i][j] is 0 (no edge), +1, or -1
            for i != j, and 0 or +1 for i == j (self-loops).
        verify: If True, validates the adjacency matrix; if False, skips validation
            (use for pre-verified matrices, e.g., from precompute_classes).
    """
    def __init__(self, adjacency_matrix: list[list[int]] | np.ndarray, verify: bool = True) -> None:
        """Initializes a weighted digraph with the given adjacency matrix.

        Args:
            adjacency_matrix: A square matrix where entry [i][j] is 0 (no edge), +1, or -1
                for i != j, and 0 or +1 for i == j (self-loops).
            verify: If True, validates the adjacency matrix; if False, skips validation
                (use for pre-verified matrices, e.g., from precompute_classes).

        Raises:
            ValueError: If verify is True and the adjacency matrix is invalid.
        """
        if verify and not self._is_valid_matrix(adjacency_matrix):
            raise ValueError("Invalid adjacency matrix")
        self.n = len(adjacency_matrix)
        self.matrix = np.array(adjacency_matrix, dtype=int)

    def _is_valid_matrix(self, matrix: list[list[int]] | np.ndarray) -> bool:
        """Validates the adjacency matrix according to graph constraints.

        Args:
            matrix: The adjacency matrix to validate.

        Returns:
            True if the matrix is valid, False otherwise.
        """
        matrix = np.array(matrix)
        n = matrix.shape[0]
        if n == 0 or matrix.shape[1] != n:
            return False

        # 2) Check edge weights and self-loop constraint
        for i in range(n):
            for j in range(n):
                if i == j:
                    if matrix[i, j] not in (0, 1):
                        return False
                else:
                    if matrix[i, j] not in (0, 1, -1):
                        return False

        # 3) Check degree constraint: each node needs at least one incoming or outgoing edge
        for i in range(n):
            has_edge = np.any(matrix[i, :] != 0) or np.any(matrix[:, i] != 0)
            if not has_edge:
                return False

        # 4) Check connectivity of the underlying undirected graph
        #    (ignore loops, treat any non-zero directed arc as an undirected edge)
        adj = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j and (matrix[i, j] != 0 or matrix[j, i] != 0):
                    adj[i].add(j)
                    adj[j].add(i)

        # BFS from node 0
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

    def get_matrix(self) -> np.ndarray:
        """Returns the adjacency matrix of the digraph.

        Returns:
            A numpy array representing the adjacency matrix.
        """
        return self.matrix

    def __str__(self) -> str:
        """Returns a string representation of the digraph's adjacency matrix.

        Returns:
            A string with each row of the matrix on a new line.
        """
        return "\n".join(str(row) for row in self.matrix)

def compute_node_invariants(digraph):
    """
    Compute invariants for each node: (loop, out_+1, out_-1, in_+1, in_-1).
    Returns a list of tuples, one per node.
    """
    matrix = digraph.get_matrix()
    n = digraph.n
    invariants = []
    for i in range(n):
        loop = matrix[i][i]
        out_plus1 = sum(1 for j in range(n) if matrix[i][j] == 1 and i != j)
        out_minus1 = sum(1 for j in range(n) if matrix[i][j] == -1)
        in_plus1 = sum(1 for j in range(n) if matrix[j][i] == 1 and j != i)
        in_minus1 = sum(1 for j in range(n) if matrix[j][i] == -1)
        invariants.append((loop, out_plus1, out_minus1, in_plus1, in_minus1))
    return invariants

def get_candidate_permutations(invariants, n):
    """
    Generate candidate permutations based on node invariants.
    Groups nodes by identical invariants and permutes within groups.
    Returns a list of permutations to check.
    """
    # Group nodes by invariants
    invariant_to_nodes = defaultdict(list)
    for idx, inv in enumerate(invariants):
        invariant_to_nodes[inv].append(idx)
    
    # Sort invariants to prioritize "heavier" nodes (lexicographically larger)
    sorted_invariants = sorted(invariant_to_nodes.keys(), reverse=True)
    
    # Generate permutations of nodes within each invariant group
    group_perms = []
    for inv in sorted_invariants:
        nodes = invariant_to_nodes[inv]
        group_perms.append(list(itertools.permutations(nodes)))
    
    # Combine permutations across groups
    candidate_perms = []
    for group_perm_combo in itertools.product(*group_perms):
        # Flatten the permutation
        perm = []
        for group_perm in group_perm_combo:
            perm.extend(group_perm)
        # Ensure full permutation
        if len(perm) == n:
            candidate_perms.append(perm)
    
    # Fallback to all permutations if none generated (e.g., n=1 or identical invariants)
    if not candidate_perms:
        candidate_perms = list(itertools.permutations(range(n)))
    
    return candidate_perms

def get_canonical_form(digraph):
    """
    Compute a canonical form using node invariants to reduce permutations.
    Returns a tuple of the matrix entries under the lexicographically minimal permutation.
    """
    matrix = digraph.get_matrix()
    n = digraph.n
    # Compute node invariants
    invariants = compute_node_invariants(digraph)
    
    # Get candidate permutations
    permutations = get_candidate_permutations(invariants, n)
    
    canonical_form = None
    for perm in permutations:
        # Create permuted matrix
        permuted_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                permuted_matrix[i][j] = matrix[perm[i]][perm[j]]
            # Early pruning: Check if current row makes matrix non-minimal
            if canonical_form is not None:
                current_row = tuple(permuted_matrix[i])
                min_row = canonical_form[i*n:(i+1)*n]
                if current_row > min_row:
                    break
        else:  # Only if loop completes without breaking
            # Flatten matrix to a tuple
            flat_matrix = tuple(permuted_matrix[i][j] for i in range(n) for j in range(n))
            # Update canonical form if first or lexicographically smaller
            if canonical_form is None or flat_matrix < canonical_form:
                canonical_form = flat_matrix
    
    return canonical_form

def identify_equivalence_class(digraph):
    """
    Identify the equivalence class of the digraph.
    Returns a canonical form (tuple) representing the isomorphism class.
    Note: Does not assign a number without a precomputed mapping.
    """
    return get_canonical_form(digraph)

def precompute_classes(n: int) -> list[np.ndarray]:
    """Precomputes unique graph representatives for graphs with n nodes.

    Args:
        n: Number of nodes in the graphs.

    Returns:
        A list of numpy arrays, each an adjacency matrix of a unique graph.
        Matrices are guaranteed to satisfy WeightedDigraph constraints and are not
        verified during construction.
    """
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
    reps = {}
    for M in tqdm(matrices, total=total_matrices, desc=f"Processing matrices (n={n})"):
        try:
            G = WeightedDigraph(M, verify=False)
            canon = get_canonical_form(G)
            if canon not in reps:
                reps[canon] = M
        except ValueError:
            continue
    return list(reps.values())

class BatchedMotif(torch.nn.Module):
    """A PyTorch module for simulating dynamics on multiple graphs.

    Attributes:
        adj: Tensor of adjacency matrices.
        n: Number of nodes per graph.
        num_graphs: Number of graphs in the batch.
        loop_mask: Mask for self-loops.
        k_loop: Hill coefficient for self-loops.
        k_edge: Hill coefficient for edges.
        n_hill: Hill exponent.
        non_zero_mask: Mask for non-zero edges.

    Args:
        adj_matrices: Array of adjacency matrices.
        device: Device to run computations on ('cuda' or 'cpu').
    """
    def __init__(self, adj_matrices: np.ndarray, device: str = 'cuda') -> None:
        """Initializes the BatchedMotif module.

        Args:
            adj_matrices: Array of adjacency matrices with shape (num_graphs, n, n).
            device: Device to run computations on ('cuda' or 'cpu').
        """
        super().__init__()
        self.device = get_device(device)
        self.adj = torch.tensor(adj_matrices, dtype=torch.float32).to(device=self.device)
        self.n = adj_matrices.shape[-1]
        self.num_graphs = adj_matrices.shape[0]
        self.loop_mask = torch.eye(self.n, dtype=torch.bool).expand_as(self.adj).to(device=self.device)
        self.k_loop = Config.K_LOOP
        self.k_edge = Config.K_EDGE
        self.n_hill = Config.N_HILL
        self.non_zero_mask = (self.adj != 0).unsqueeze(0).to(device=self.device)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the derivative for the ODE system.

        Args:
            t: Time point (scalar).
            y: State tensor with shape (batch_size, num_graphs * n).

        Returns:
            Derivative tensor with shape (batch_size, num_graphs * n).
        """
        batch_size = y.shape[0]
        y = y.view(-1, self.num_graphs, self.n)
        u = y.unsqueeze(2).expand(-1, -1, self.n, -1)
        k = torch.where(self.loop_mask, self.k_loop, self.k_edge)
        hill_act = u**self.n_hill / (k**self.n_hill + u**self.n_hill)
        hill_inh = 1 / (1 + (u/k)**self.n_hill)
        act_mask = (self.adj == 1).unsqueeze(0)
        inh_mask = (self.adj == -1).unsqueeze(0)
        factors = torch.zeros_like(u)
        factors = torch.where(act_mask, hill_act, factors)
        factors = torch.where(inh_mask, hill_inh, factors)
        prod_factors = torch.where(self.non_zero_mask, factors, 1.0)
        prod = torch.prod(prod_factors, dim=3)
        no_incoming = (~self.non_zero_mask.any(dim=3)).squeeze(0)
        prod = torch.where(no_incoming.unsqueeze(0), 0.0, prod)
        dx = prod - y
        return dx.view(batch_size, self.n)

def solve_ivp_torchode(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t_span: tuple[float, float],
    y0: torch.Tensor,
    t_eval: torch.Tensor | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-3
) -> torch.Tensor:
    """Solves an initial value problem (IVP) for ODEs using torchode.

    Args:
        f: ODE function with signature f(t, y) -> dy/dt.
        t_span: Tuple (t_start, t_end) for integration.
        y0: Initial condition tensor (1D, 2D, or 3D).
        t_eval: Time points for evaluation (optional).
        atol: Absolute tolerance for solver.
        rtol: Relative tolerance for solver.

    Returns:
        Solution tensor with shape depending on y0.
    """
    is_3d = y0.dim() == 3
    is_1d = y0.dim() == 1
    if is_3d:
        num_inits, num_graphs, n = y0.shape
        y0 = y0.view(num_inits * num_graphs, n)
    elif is_1d:
        y0 = y0.unsqueeze(0)
    else:
        y0 = y0.unsqueeze(1)
    if t_eval is None:
        t_eval = torch.linspace(t_span[0], t_span[1], Config.NUM_STEPS, device=y0.device)
    if t_eval.dim() == 1:
        t_eval = t_eval.unsqueeze(0).repeat(y0.shape[0], 1)
    term = to.ODETerm(f)
    step = to.Dopri5(term=term)
    ctrl = to.IntegralController(atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(step, ctrl)
    batch_size = y0.shape[0]
    t_start = torch.tensor([t_span[0]], dtype=torch.float32, device=y0.device).expand(batch_size)
    t_end = torch.tensor([t_span[1]], dtype=torch.float32, device=y0.device).expand(batch_size)
    ivp = to.InitialValueProblem(y0=y0, t_start=t_start, t_end=t_end, t_eval=t_eval)
    sol = solver.solve(ivp)
    if is_3d:
        sol.ys = sol.ys.view(num_inits, num_graphs, sol.ys.shape[1], n)
    elif is_1d:
        sol.ys = sol.ys.squeeze(0)
    else:
        sol.ys = sol.ys.unsqueeze(1)
    return sol.ys

def graph_features(
    adj_matrices: np.ndarray,
    num_inits: int = 10,
    batch_size: int = 100,
    device: str = 'cuda',
    include_mean_std: bool = True,
    custom_features: list[Callable[[torch.Tensor], np.ndarray]] | None = None
) -> np.ndarray:
    """Computes features for a batch of graphs by simulating dynamics.

    Args:
        adj_matrices: Array of adjacency matrices with shape (num_graphs, n, n).
        num_inits: Number of initial conditions per graph.
        batch_size: Number of graphs to process per batch.
        device: Device to run computations on ('cuda' or 'cpu').
        include_mean_std: Whether to include mean and std in the output.
        custom_features: List of callable functions, each taking simulation output
            (shape: (num_inits, num_graphs, num_steps, n)) and returning features
            (shape: (num_graphs, k)). Features are appended to the default mean and std.

    Returns:
        Array of features with shape (num_graphs, 2*n + sum(k_i)), where 2*n is from
        mean and std of final states, and k_i is the feature size from each custom function.

    Raises:
        ValueError: If a custom feature function returns an incorrectly shaped array.
    """
    device = get_device(device)
    n = adj_matrices.shape[-1]
    num_graphs = len(adj_matrices)
    custom_features = custom_features or []
    features = []
    num_batches = (num_graphs + batch_size - 1) // batch_size
    include_mean_std = custom_features is None or include_mean_std
    with tqdm(total=num_batches, desc="Simulating dynamics for all graphs") as pbar:
        for i in range(0, num_graphs, batch_size):
            batch_adj = adj_matrices[i:i + batch_size]
            batch_size_actual = len(batch_adj)
            model = BatchedMotif(batch_adj, device=device)
            y0 = torch.rand(num_inits, batch_size_actual, n).to(device=device)
            ys = solve_ivp_torchode(model, Config.T_SPAN, y0)
            final = ys[:, :, -1, :]
            batch_features = []

            if include_mean_std:
                mean_final = final.mean(dim=0)
                std_final = final.std(dim=0)
                batch_features = [mean_final, std_final]

            for cf in custom_features:
                try:
                    cf_result = cf(ys)
                    if not isinstance(cf_result, np.ndarray) or cf_result.shape[0] != batch_size_actual:
                        raise ValueError(f"Custom feature function {cf.__name__} returned invalid shape: {cf_result.shape}")
                    batch_features.append(cf_result)
                except Exception as e:
                    logging.error(f"Error computing custom feature {cf.__name__}: {e}")
                    raise
            batch_features = torch.cat([torch.tensor(f, device='cpu') if isinstance(f, np.ndarray) else f for f in batch_features], dim=1).cpu().numpy()
            features.append(batch_features)
            pbar.update(1)
    return np.concatenate(features, axis=0)

def analyze_all_graphs(
    n: int,
    num_inits: int = 10,
    custom_features: list[Callable[[torch.Tensor], np.ndarray]] | None = None
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Analyzes all unique graphs with n nodes and computes embeddings.

    Args:
        n: Number of nodes in the graphs.
        num_inits: Number of initial conditions for simulations.
        custom_features: List of callable functions, each taking simulation output
            (shape: (num_inits, num_graphs, num_steps, n)) and returning features
            (shape: (num_graphs, k)). Features are appended to the default mean and std.

    Returns:
        A tuple containing:
        - List of unique graph adjacency matrices.
        - Feature array with shape (num_graphs, 2*n + sum(k_i)), where 2*n is from
          mean and std, and k_i is from custom features.
        - PCA embedding (2D).
        - UMAP embedding (2D).
    """
    graphs = precompute_classes(n)
    print(f"Number of unique graphs for n={n}: {len(graphs)}")
    adj_matrices = np.array(graphs)
    feats = graph_features(adj_matrices, num_inits, custom_features=custom_features)
    pca = PCA(n_components=2)
    p2 = pca.fit_transform(feats)
    um = umap.UMAP(n_components=2)
    u2 = um.fit_transform(feats)
    return graphs, feats, p2, u2

def run_simulation(
    adj: np.ndarray | None,
    y0: torch.Tensor,
    ts: torch.Tensor,
    device: str = 'cuda'
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Runs a simulation for a single graph.

    Args:
        adj: Adjacency matrix of the graph, or None.
        y0: Initial condition tensor with shape (n,).
        ts: Time points for evaluation.
        device: Device to run computations on ('cuda' or 'cpu').

    Returns:
        A tuple of (time points, solution array) if adj is not None,
        otherwise (None, None).
    """
    device = get_device(device)
    if adj is None:
        return None, None
    adj = np.expand_dims(adj, 0)
    y0 = torch.tensor(y0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    t_eval = ts.to(device=device) if ts.dim() == 1 else torch.linspace(Config.T_SPAN[0], Config.T_SPAN[1], Config.NUM_STEPS, device=device)
    ys = solve_ivp_torchode(BatchedMotif(adj, device=device), Config.T_SPAN, y0, t_eval=t_eval)
    return t_eval.cpu().numpy(), ys.squeeze(0).squeeze(0).cpu().numpy()

def draw_graph_edges(G: nx.DiGraph, n: int, pos: dict[int, tuple[float, float]]) -> list[dict]:
    """Generates Plotly shapes for drawing graph edges.

    Args:
        G: NetworkX directed graph with weighted edges.
        n: Number of nodes in the graph.
        pos: Dictionary mapping node indices to (x, y) positions.

    Returns:
        A list of Plotly shape dictionaries for edges and arrowheads.
    """
    shapes = []
    for u, v, data in G.edges(data=True):
        w = data['weight']
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if u == v:
            r = 0.25
            sx, sy = x0, y0
            c1 = (x0 - r, y0 + r)
            c2 = (x0 - r, y0 - r)
            ex, ey = sx, sy
            path = f"M {sx},{sy} C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {ex},{ey}"
            shapes.append(dict(
                type="path", path=path, xref="x2", yref="y2",
                line_color="black", line_width=Config.EDGE_WIDTH
            ))
            tx, ty = ex - c2[0], ey - c2[1]
            L = math.hypot(tx, ty) or 1.0
            ux, uy = tx/L, ty/L
            px, py = -uy, ux
            size = 0.06
            tip = (ex, ey)
            base1 = (ex - ux*size + px*size, ey - uy*size + py*size)
            base2 = (ex - ux*size - px*size, ey - uy*size - py*size)
            fill = Config.COLORS['positive'] if w > 0 else Config.COLORS['negative']
            shapes.append(dict(
                type="path",
                path=f"M {tip[0]},{tip[1]} L {base1[0]},{base1[1]} L {base2[0]},{base2[1]} Z",
                xref="x2", yref="y2", fillcolor=fill, line_color=fill, layer="above"
            ))
        else:
            dx, dy = x1 - x0, y1 - y0
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            perp = (-dy, dx)
            norm = math.hypot(*perp) or 1.0
            ux_perp, uy_perp = perp[0] / norm, perp[1] / norm
            offset = 0.2 if G.has_edge(v, u) else 0.0
            cx, cy = mx + ux_perp * offset, my + uy_perp * offset
            path = f"M {x0},{y0} Q {cx},{cy} {x1},{y1}"
            shapes.append(dict(
                type="path", path=path, xref="x2", yref="y2",
                line_color="black", line_width=Config.EDGE_WIDTH
            ))
            L = math.hypot(dx, dy) or 1.0
            ux, uy = dx / L, dy / L
            px, py = -uy, ux
            size = 0.05
            tip = (x1, y1)
            base1 = (x1 - ux * size + px * size, y1 - uy * size + py * size)
            base2 = (x1 - ux * size - px * size, y1 - uy * size - py * size)
            fill = Config.COLORS['positive'] if w > 0 else Config.COLORS['negative']
            shapes.append(dict(
                type="path",
                path=f"M {tip[0]},{tip[1]} L {base1[0]},{base1[1]} L {base2[0]},{base2[1]} Z",
                xref="x2", yref="y2", fillcolor=fill, line_color=fill, layer="above"
            ))
    return shapes

def visualize_graphs(
    graphs: list[np.ndarray],
    pca2: np.ndarray,
    max_points: int = 1500,
    device: str = 'cuda'
) -> tuple[go.FigureWidget, widgets.VBox, widgets.Button, widgets.Button, list[widgets.FloatText]]:
    """Creates an interactive visualization of graphs and their embeddings.

    Args:
        graphs: List of graph adjacency matrices.
        pca2: 2D PCA embedding of graph features.
        max_points: Maximum number of points to display in the PCA plot.
        device: Device to run computations on ('cuda' or 'cpu').

    Returns:
        A tuple containing:
        - Plotly FigureWidget for the visualization.
        - VBox widget for the UI controls.
        - Button widget for running simulations.
        - Button widget for randomizing initial conditions.
        - List of FloatText widgets for initial conditions.
    """
    try:
        device = get_device(device)
        n = len(graphs[0])
        color_cycle = px.colors.qualitative.Plotly[-n-1:]
        current_adj = [None]
        graphs = np.array(graphs, dtype=object)
        pca2 = np.array(pca2)
        num_graphs = len(graphs)

        def get_subsampled_points(x_range=None, y_range=None, max_points=max_points):
            if x_range is None or y_range is None:
                indices = np.random.choice(num_graphs, min(max_points, num_graphs), replace=False)
            else:
                x_min, x_max = x_range
                y_min, y_max = y_range
                mask = (pca2[:, 0] >= x_min) & (pca2[:, 0] <= x_max) & (pca2[:, 1] >= y_min) & (pca2[:, 1] <= y_max)
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > max_points:
                    indices = np.random.choice(valid_indices, max_points, replace=False)
                else:
                    indices = valid_indices
                if len(indices) == 0:
                    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                    distances = np.sqrt((pca2[:, 0] - center[0])**2 + (pca2[:, 1] - center[1])**2)
                    indices = np.argsort(distances)[:max_points]
            return indices

        initial_indices = get_subsampled_points()
        sampled_graphs = graphs[initial_indices].tolist()
        sampled_pca2 = pca2[initial_indices]
        initial_colors = [Config.COLORS['default']] * len(sampled_graphs)

        fig = make_subplots(rows=1, cols=3, subplot_titles=("PCA embedding", "Selected graph", "Simulation"))
        pca_scatter = go.Scatter(
            x=sampled_pca2[:, 0], y=sampled_pca2[:, 1],
            mode='markers',
            marker=dict(size=Config.MARKER_SIZE, color=initial_colors),
            hovertext=[f"graph #{i}" for i in range(len(sampled_graphs))]
        )
        fig.add_trace(pca_scatter, row=1, col=1)
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            marker=dict(size=Config.NODE_SIZE, color=color_cycle),
            text=[],
            textposition="bottom center",
            hoverinfo='text'
        )
        fig.add_trace(node_trace, row=1, col=2)
        for i in range(n):
            sim_trace = go.Scatter(
                x=[], y=[],
                mode='lines',
                name=f"x{i}",
                line=dict(color=color_cycle[i])
            )
            fig.add_trace(sim_trace, row=1, col=3)
        fw = go.FigureWidget(fig)
        fw.update_layout(height=Config.PLOT_HEIGHT, width=Config.PLOT_WIDTH)
        fw._sampled_graphs = sampled_graphs
        fw._sampled_indices = initial_indices

        def show_graph(trace, points, state):
            if not points.point_inds:
                return
            idx = points.point_inds[0]
            new_colors = [Config.COLORS['default']] * len(fw._sampled_graphs)
            new_colors[idx] = Config.COLORS['selected']
            with fw.batch_update():
                fw.data[0].marker.color = new_colors
            adj = fw._sampled_graphs[idx]
            current_adj[0] = adj
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(n):
                    if adj[i][j] != 0:
                        G.add_edge(i, j, weight=adj[i][j])
            pos = nx.circular_layout(G)
            xs = [pos[i][0] for i in range(n)]
            ys = [pos[i][1] for i in range(n)]
            labels = [str(i) for i in range(n)]
            shapes = draw_graph_edges(G, n, pos)
            with fw.batch_update():
                fw.data[1].x = xs
                fw.data[1].y = ys
                fw.data[1].text = labels
                fw.layout.shapes = shapes
                fw.layout.annotations[1].text = f"Graph #{idx}"
                for ti in range(2, 2 + n):
                    fw.data[ti].x = []
                    fw.data[ti].y = []
                fw.layout.annotations[2].text = "Simulation"

        fw.data[0].on_click(show_graph)

        def on_relayout(layout, x_range, y_range):
            try:
                if x_range and y_range:
                    new_indices = get_subsampled_points(x_range, y_range, max_points)
                    new_sampled_graphs = graphs[new_indices].tolist()
                    new_sampled_pca2 = pca2[new_indices]
                    new_colors = [Config.COLORS['default']] * len(new_indices)
                    with fw.batch_update():
                        fw.data[0].x = new_sampled_pca2[:, 0]
                        fw.data[0].y = new_sampled_pca2[:, 1]
                        fw.data[0].hovertext = [f"graph #{i}" for i in range(len(new_indices))]
                        fw.data[0].marker.color = new_colors
                        fw._sampled_graphs = new_sampled_graphs
                        fw._sampled_indices = new_indices
                        logging.info(f"Updated plot with {len(new_indices)} points in ROI")
            except Exception as e:
                logging.error(f"Error updating ROI: {e}")

        fw.layout.on_change(on_relayout, 'xaxis.range', 'yaxis.range')
        run_btn = widgets.Button(description="Run Simulation", button_style="success")
        fw.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
        fw.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
        init_boxes = [
            widgets.FloatText(
                value=0.5,
                description=f"x{i}:",
                step=0.01,
                style={"description_width": "30px"},
                layout=widgets.Layout(width="100px")
            ) for i in range(n)
        ]
        boxes_row = widgets.HBox(
            init_boxes,
            layout=widgets.Layout(flex_flow="row wrap", align_items="center")
        )
        rand_btn = widgets.Button(description="Randomize y₀", button_style="info")
        def on_randomize(_):
            for box in init_boxes:
                box.value = float(torch.rand(1).item())
        rand_btn.on_click(on_randomize)
        def on_run_clicked(b):
            if current_adj[0] is None:
                return
            y0_vals = [box.value for box in init_boxes]
            y0 = torch.tensor(y0_vals, dtype=torch.float32)
            ts = torch.linspace(Config.T_SPAN[0], Config.T_SPAN[1], Config.NUM_STEPS, device=device)
            ts, arr = run_simulation(current_adj[0], y0, ts, device=device)
            with fw.batch_update():
                for i in range(n):
                    fw.data[2 + i].x = ts
                    fw.data[2 + i].y = arr[:, i]
                fw.layout.annotations[2].text = "Simulation"
        run_btn.on_click(on_run_clicked)
        controls = widgets.HBox([rand_btn, run_btn], layout=widgets.Layout(margin="10px"))
        ui = widgets.VBox([boxes_row, controls])
        display(fw)
        display(ui)
        return fw, ui, run_btn, rand_btn, init_boxes
    except Exception as e:
        logging.error(f"Error initializing visualization: {e}")
        raise