import numpy as np
import networkx as nx
from itertools import product
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import Counter
from pathlib import Path
from utils import plot_data

# --- Configuration ---
RESULTS_DIR = Path("./results_vertes")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# generator = np.random.Generator(np.random.MT19937(42)) # Fixed seed for reproducibility
generator = np.random.Generator(np.random.MT19937()) # Random seed

type Neuron = tuple[float, float]

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
    norm = mcolors.LogNorm(vmin=max(min_w, 1), vmax=max_w)
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

def run_simulation(alpha: float, N=100):
    graph, probs = init_all(N, alpha)
    target_edges = math.ceil(0.1 * N * N)
    
    # Prepare probabilities
    flat_probs = probs.ravel().copy()
    sum_p = flat_probs.sum()
    if sum_p == 0:
        return graph, np.zeros_like(flat_probs)
        
    flat_probs /= sum_p
    
    edge_selection_counts = Counter()
    while graph.number_of_edges() < target_edges:
        sampled_indices = generator.choice(
            len(flat_probs), 
            size=100, 
            replace=True, 
            p=flat_probs
        )
        
        for idx in sampled_indices:
            n1, n2 = divmod(idx, N)
            if n1 == n2: continue
            
            edge_selection_counts[idx] += 1
            
            # Graph Update
            is_new = not graph.has_edge(n1, n2)
            
            if is_new:
                if graph.number_of_edges() < target_edges:
                    graph.add_edge(n1, n2, weight=edge_selection_counts[idx])
                else:
                    break
            else:
                graph[n1][n2]['weight'] = edge_selection_counts[idx]

    return graph

if __name__ == "__main__":
    # 1. Define range
    alphas = list(np.linspace(0.1, 100, 10)) # Adjusted range for 100 nodes
    graphs = []
    
    print(f"Starting Simulation Sweep over {len(alphas)} alphas...")

    # 2. Run Sweep
    for alpha in alphas:
        print(f"Alpha: {alpha}")
        # Run simulation
        g= run_simulation(alpha=alpha, N=100)
        
        # Save visualization
        filename = RESULTS_DIR / f"vertes_alpha_{alpha:.2f}.png"
        draw_graph(g, file_path=filename)
        
        graphs.append(g)

    # 3. Plot Summary Data
    plot_data(
        alphas, 
        graphs, 
        RESULTS_DIR / "vertes_clusterings.png", 
        RESULTS_DIR / "vertes_lengths.png"
    )
    
    print("Done! Results saved to", RESULTS_DIR)

