import numpy as np
import networkx as nx
from itertools import product
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from utils import plot_data

# --- Configuration ---
RESULTS_DIR = Path("./results_vertes")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
generator = np.random.Generator(np.random.MT19937(SEED))


def draw_graph(graph, file_path: Path | None = None):
    # Setup figure
    _, ax = plt.subplots(figsize=(12, 9))
    pos = nx.get_node_attributes(graph, 'pos')

    # 1. Extract Weights
    weights = [graph[u][v]['weight'] for u, v in graph.edges()]

    if not weights:
        # Handle empty graph case
        print(f"Graph for {file_path} has no edges.")
        plt.close()
        return

    # 2. Manual Color Mapping (Logarithmic)
    min_w, max_w = min(weights), max(weights)
    norm = mcolors.LogNorm(vmin=max(min_w, 1), vmax=max(max_w, 2))  # prevent vmax=1 issue
    cmap = plt.colormaps["YlOrRd"]

    # Generate RGBA colors explicitly
    edge_colors = [cmap(norm(w)) for w in weights]

    # 3. Draw Nodes
    nx.draw_networkx_nodes(graph, pos, node_size=20, node_color='black', ax=ax)

    # 4. Draw Edges
    widths = [np.log1p(w) * 0.8 for w in weights]

    nx.draw_networkx_edges(
        graph, pos,
        width=widths,
        edge_color=edge_colors,
        arrowstyle='->',
        arrowsize=10,
        ax=ax
    )

    # 5. Create Manual Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.colorbar(sm, ax=ax, label='Connection Strength (Log Scale)')

    ax.set_axis_off()
    if file_path:
        plt.title(f"Alpha={file_path.stem.split('_')[-1]} | Edges={len(graph.edges)}")
        plt.savefig(file_path)
    else:
        plt.show()

    plt.close()


def get_connection_prob(node_i: Neuron, node_j: Neuron, alpha: float) -> float:
    diff = np.array([node_i[0] - node_j[0], node_i[1] - node_j[1]])
    dist = np.linalg.norm(diff)
    return np.exp(-alpha * dist)


def init_all(N, alpha: float):
    graph = nx.DiGraph()

    # 1. Create nodes
    for i in range(N):
        coords = (generator.uniform(0, 1), generator.uniform(0, 1))
        graph.add_node(i, pos=coords)

    # 2. Initialize probabilities
    probs = np.zeros((N, N))

    for id1, id2 in product(graph.nodes, graph.nodes):
        if id1 == id2:
            continue

        pos1 = graph.nodes[id1]["pos"]
        pos2 = graph.nodes[id2]["pos"]

        probs[id1, id2] = get_connection_prob(pos1, pos2, alpha)

    # 3. Normalize
    total_sum = probs.sum()
    if total_sum > 0:
        probs = probs / total_sum

    return graph, probs


def run_simulation(alpha: float, N=100, rho: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Vértes improved spatial network algorithm (Vértes et al. 2012 PNAS).

    Algorithm:
    1. Place N nodes randomly in [0,1]^2
    2. Compute P(i,j) ∝ exp(-α * d_ij), normalized so Σ P = 1
    3. Iteratively sample edges according to P until target density ρ is reached
    4. Multiple selections of same edge → weight (connection strength)

    Returns:
        tuple: (weight_matrix NxN directed, positions Nx2 array)
    """
    # Vectorized position generation
    # 1. Generate positions - all N nodes placed at start
    positions = np.column_stack([
        generator.uniform(0, 1, N),
        generator.uniform(0, 1, N)
    ])

    # Vectorized distance computation
    # 2. Compute pairwise Euclidean distances
    diff_x = positions[:, 0:1] - positions[:, 0:1].T
    diff_y = positions[:, 1:2] - positions[:, 1:2].T
    distances = np.sqrt(diff_x**2 + diff_y**2)

    # 3. Compute probability matrix: P(i,j) ∝ exp(-α * d_ij)
    probs = np.exp(-alpha * distances)
    np.fill_diagonal(probs, 0)

    # Normalize so Σ P = 1
    total_sum = probs.sum()
    if total_sum == 0:
        return np.zeros((N, N), dtype=int), positions
    probs = probs / total_sum

    # 4. Target: number of UNIQUE edges = density × max possible directed edges
    # For directed graph without self-loops: max = N(N-1)
    target_unique_edges = math.ceil(rho * N * (N - 1))

    # Flatten probability matrix for sampling
    flat_probs = probs.ravel()

    # Weight matrix tracks how many times each edge was selected
    weight_matrix = np.zeros((N, N), dtype=int)
    num_unique_edges = 0

    # Batch sampling - sample multiple edges per iteration
    batch_size = 100

    # Iteratively sample edges until we reach target unique edges
    while num_unique_edges < target_unique_edges:
        # Sample batch of edge indices according to probability distribution
        sampled_indices = generator.choice(
            len(flat_probs),
            size=batch_size,
            replace=True,
            p=flat_probs
        )

        # Process each sampled edge
        for idx in sampled_indices:
            # Convert flat index to (i, j) coordinates
            n1, n2 = divmod(idx, N)

            # Skip self-loops
            if n1 == n2:
                continue

            # Check if this is a new unique edge
            is_new_edge = (weight_matrix[n1, n2] == 0)

            # Increment edge weight (selection count)
            # Direct array increment instead of graph[n1][n2]['weight'] = ...
            weight_matrix[n1, n2] += 1

            # Track unique edges
            if is_new_edge:
                num_unique_edges += 1
                # Stop when we reach target
                if num_unique_edges >= target_unique_edges:
                    break


    return weight_matrix, positions


# Helper to convert back to graph for plotting
def to_graph(adj_matrix: np.ndarray, positions: np.ndarray) -> nx.DiGraph:
    """Convert adjacency matrix and positions back to NetworkX DiGraph."""
    G = nx.DiGraph()
    N = adj_matrix.shape[0]

    for i in range(N):
        G.add_node(i, pos=(positions[i, 0], positions[i, 1]))

    for i, j in product(range(N), range(N)):
        if adj_matrix[i, j] > 0:
            G.add_edge(i, j, weight=int(adj_matrix[i, j]))

    return G


if __name__ == "__main__":
    # Reset generator at start
    generator = np.random.Generator(np.random.MT19937(SEED))

    # 1. Define range
    alphas = list(np.linspace(0.1, 100, 10)) # Adjusted range for 100 nodes
    graphs = []

    print(f"Starting Simulation Sweep over {len(alphas)} alphas...")

    # 2. Run Sweep
    for alpha in alphas:
        print(f"Alpha: {alpha}")
        # Run simulation
        adj_matrix, positions = run_simulation(alpha=alpha, N=100)
        g = to_graph(adj_matrix, positions)  # Convert for plotting

        # Save visualization
        filename = RESULTS_DIR / f"vertes_alpha_{alpha:.2f}.png"
        draw_graph(g, file_path=filename)

        graphs.append(g)

    # 3. Plot Summary Data
    plot_data(
        alphas,
        graphs,
        RESULTS_DIR / "vertes_clusterings.png",
        RESULTS_DIR / "vertes_lengths.png",
        title_prefix="Vértes"
    )

    print("Done! Results saved to", RESULTS_DIR)