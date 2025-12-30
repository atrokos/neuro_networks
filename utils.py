from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt


def get_average_path_length(g: nx.Graph) -> float:
    # Handle empty graphs
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return float('inf')

    # Handle disconnected graphs - use largest connected component
    if isinstance(g, nx.DiGraph):
        if not nx.is_strongly_connected(g):
            largest_scc = max(nx.strongly_connected_components(g), key=len)
            g = g.subgraph(largest_scc).copy()
            if g.number_of_nodes() < 2:
                return float('inf')
    elif not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
        if g.number_of_nodes() < 2:
            return float('inf')

    # Use built-in for efficiency on connected subgraph
    try:
        return nx.average_shortest_path_length(g)
    except nx.NetworkXError:
        pass

    # Fallback to manual computation (original code)
    total_length = 0
    total_paths = 0

    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(g))
    for start, val in shortest_path_lengths.items():
        for end, length in val.items():
            if start == end:
                continue
            total_length += length
            total_paths += 1

    return float('inf') if total_paths == 0 else total_length / total_paths


def plot_data(alphas: list[float], networks: list[nx.Graph], file_clust: Path, file_lengths: Path, title_prefix: str = ""):
    print("Computing metrics for summary plots...")
    clusterings = [nx.average_clustering(g) for g in networks]
    avg_lengths = [get_average_path_length(g) for g in networks]

    # --- Plot 1: Clustering Coefficient ---
    plt.figure(figsize=(10, 6))
    plt.scatter(alphas, clusterings, color='teal', alpha=0.7, edgecolors='black', s=50)
    plt.plot(alphas, clusterings, color='teal', alpha=0.3)

    plt.title(f"Clustering Coefficient vs Alpha ({title_prefix})" if title_prefix else "Clustering Coefficient vs Alpha")
    plt.xlabel(r"$\alpha$ (Spatial Penalty)")
    plt.ylabel("Avg Clustering Coefficient")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(file_clust)
    plt.close()

    # --- Plot 2: Average Path Length ---
    plt.figure(figsize=(10, 6))
    plt.scatter(alphas, avg_lengths, color='crimson', alpha=0.7, edgecolors='black', s=50)
    plt.plot(alphas, avg_lengths, color='crimson', alpha=0.3)

    plt.title(f"Average Path Length vs Alpha ({title_prefix})" if title_prefix else "Average Path Length vs Alpha")
    plt.xlabel(r"$\alpha$ (Spatial Penalty)")
    plt.ylabel("Avg Path Length (Largest Connected Component)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(file_lengths)
    plt.close()