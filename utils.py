from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt


def get_average_path_length(g: nx.Graph) -> float:
    total_length = 0
    total_paths = 0

    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(g))
    for start, val in shortest_path_lengths.items():
        for end, length in val.items():
            if start == end:
                continue
            total_length += length
            total_paths += 1

    return total_length / total_paths


def plot_data(alphas: list[float], networks: list[nx.Graph], file_clust: Path, file_lengths: Path):
    print("Computing metrics for summary plots...")
    clusterings = [nx.average_clustering(g) for g in networks]
    avg_lengths = [get_average_path_length(g) for g in networks]

    # --- Plot 1: Clustering Coefficient ---
    plt.figure(figsize=(10, 6))
    plt.scatter(alphas, clusterings, color='teal', alpha=0.7, edgecolors='black', s=50)
    plt.plot(alphas, clusterings, color='teal', alpha=0.3) # Connect lines for readability
    
    plt.title("Clustering Coefficient vs Alpha (Vertes)")
    plt.xlabel(r"$\alpha$ (Spatial Penalty)")
    plt.ylabel("Avg Clustering Coefficient")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(file_clust)
    plt.close()

    # --- Plot 2: Average Path Length ---
    plt.figure(figsize=(10, 6))
    plt.scatter(alphas, avg_lengths, color='crimson', alpha=0.7, edgecolors='black', s=50)
    plt.plot(alphas, avg_lengths, color='crimson', alpha=0.3)
    
    plt.title("Average Path Length vs Alpha (Vertes)")
    plt.xlabel(r"$\alpha$ (Spatial Penalty)")
    plt.ylabel("Avg Path Length (Largest SCC)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(file_lengths)
    plt.close()