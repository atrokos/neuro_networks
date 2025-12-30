from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from utils import plot_data

RESULTS_DIR = Path("./results_kaiser")
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def get_connection_prob(node_i, node_j, alpha, beta) -> float:
    # Fixed distance calculation to be standard Euclidean distance
    # (x1-x2, y1-y2)
    diff = np.array([node_i[0] - node_j[0], node_i[1] - node_j[1]])
    dist = np.linalg.norm(diff)
    return beta * np.exp(-alpha * dist)

def run(alpha, beta, N) -> tuple[np.ndarray, dict]:
    """
    Run Kaiser & Hilgetag spatial growth algorithm.

    Returns:
        tuple: (adjacency_matrix NxN, positions dict {node_id: (x,y)})
    """
    # Initialize Graph directly
    G = nx.Graph()
    positions = {}  # Store positions separately
    node_id = 0

    # Add the seed node
    positions[node_id] = (0.5, 0.5)
    G.add_node(node_id)
    node_id += 1

    while G.number_of_nodes() < N:
        new_pos = (random.random(), random.random())

        edges_to_add = []

        for existing_id in list(G.nodes()):
            existing_pos = positions[existing_id]
            connection_prob = get_connection_prob(existing_pos, new_pos, alpha, beta)

            if random.random() <= connection_prob:
                edges_to_add.append((existing_id, node_id))

        if edges_to_add:
            positions[node_id] = new_pos
            G.add_node(node_id)
            G.add_edges_from(edges_to_add)
            node_id += 1

    # Return adjacency matrix instead of graph
    adj_matrix = nx.to_numpy_array(G, nodelist=range(N))
    return adj_matrix, positions


# Helper to convert back to graph for plotting
def to_graph(adj_matrix: np.ndarray, positions: dict) -> nx.Graph:
    """Convert adjacency matrix and positions back to NetworkX graph."""
    G = nx.from_numpy_array(adj_matrix)
    nx.set_node_attributes(G, positions, 'pos')
    return G


def draw_graph(G, positions: dict | None = None, file: Path | None = None):
    # Accept positions as parameter or get from graph attributes
    if positions is None:
        positions = nx.get_node_attributes(G, 'pos')
    if not positions:
        # Fallback for old-style graphs with coordinate nodes
        positions = {node: node if isinstance(node, tuple) else (0.5, 0.5) for node in G.nodes()}

    plt.figure(figsize=(10, 8))

    nx.draw(G, positions,
            node_color='lightblue',
            with_labels=False, # False looks cleaner for 100 nodes
            node_size=50,      # Smaller size for better visibility
            edge_color='gray',
            alpha=0.6,
            width=1)

    plt.title(f"Network with Fixed Coordinates (N={G.number_of_nodes()})")
    plt.axis('on')
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if file is None:
        plt.show()
    else:
        plt.savefig(file)

    plt.close()


if __name__ == "__main__":
    # Reset seed at start of main
    random.seed(SEED)
    np.random.seed(SEED)

    # Run simulation
    alphas = list(np.linspace(0.1, 100, 30))
    graphs = []

    for alpha in alphas:
        adj_matrix, positions = run(alpha=alpha, beta=1.0, N=100)
        g = to_graph(adj_matrix, positions)  # Convert for plotting
        draw_graph(g, positions, RESULTS_DIR / f"kaiser_alpha_{alpha:.2f}.png")
        graphs.append(g)


    plot_data(alphas, graphs, RESULTS_DIR / "kaiser_clusterings.png", RESULTS_DIR / "kaiser_lengths.png", title_prefix="Kaiser-Hilgetag")