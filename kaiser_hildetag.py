from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from utils import plot_data

RESULTS_DIR = Path("./results_kaiser")
RESULTS_DIR.mkdir(exist_ok=True)

def get_connection_prob(node_i, node_j, alpha, beta) -> float:
    # Fixed distance calculation to be standard Euclidean distance
    # (x1-x2, y1-y2)
    diff = np.array([node_i[0] - node_j[0], node_i[1] - node_j[1]])
    dist = np.linalg.norm(diff)
    return beta * np.exp(-alpha * dist)

def run(alpha, beta, N):
    # Initialize Graph directly
    G = nx.Graph()
    
    # Add the seed node
    G.add_node((0.5, 0.5))

    while G.number_of_nodes() < N:
        new_neuron = (random.random(), random.random())
        if new_neuron in G:
            continue
        
        edges_to_add = []
        
        for existing_neuron in list(G.nodes()):
            connection_prob = get_connection_prob(existing_neuron, new_neuron, alpha, beta)
            
            if random.random() <= connection_prob:
                edges_to_add.append((existing_neuron, new_neuron))

        if edges_to_add:
            G.add_edges_from(edges_to_add)

    return G

def draw_graph(G, file: Path | None=None):
    # Since nodes are coordinates, map node -> node
    pos = {node: node for node in G.nodes()}

    plt.figure(figsize=(10, 8))
    
    nx.draw(G, pos,
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
    # Run simulation
    alphas = list(np.linspace(0.1, 100, 30))
    graphs = []

    for alpha in alphas:
        graph = run(alpha=alpha, beta=1.0, N=100)
        draw_graph(graph, RESULTS_DIR / f"kaiser_alpha_{alpha:.2f}.png")
        graphs.append(graph)


    plot_data(alphas, graphs, RESULTS_DIR / "kaiser_clusterings.png", RESULTS_DIR / "kaiser_lengths.png")
        