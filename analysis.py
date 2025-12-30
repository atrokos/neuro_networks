import random
from pathlib import Path
import kaiser_hildetag as kh
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import vertes as vt
from utils import get_average_path_length

# --- Configuration ---
RESULTS_DIR = Path("./results_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# =============================================================================
# Utility plotting functions
# =============================================================================

def plot_adjacency_matrix(adj_matrix: np.ndarray, title: str = "Adjacency Matrix",
                          file_path: Path | None = None, show_weights: bool = False):
    """Plot adjacency matrix as a heatmap."""
    plt.figure(figsize=(8, 8))

    if show_weights:
        plt.imshow(adj_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Weight')
    else:
        plt.imshow(adj_matrix > 0, cmap='binary', interpolation='nearest')

    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()


def plot_degree_distribution(G: nx.Graph | nx.DiGraph, title: str = "Degree Distribution",
                             file_path: Path | None = None):
    """Plot degree distribution histogram."""
    plt.figure(figsize=(10, 6))

    if isinstance(G, nx.DiGraph):
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        plt.hist([in_degrees, out_degrees], bins=20,
                 label=['In-degree', 'Out-degree'], alpha=0.7, edgecolor='black')
        plt.legend()
    else:
        degrees = [d for n, d in G.degree()]
        plt.hist(degrees, bins=20, color='steelblue', alpha=0.7, edgecolor='black')

    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()


def plot_weight_distribution(G: nx.Graph | nx.DiGraph, title: str = "Weight Distribution",
                             file_path: Path | None = None):
    """Plot weight distribution histogram with log scale."""
    weights = [G[u][v].get('weight', 1) for u, v in G.edges()]

    if not weights:
        print("No edges for weight distribution")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Weight (log scale)')
    plt.ylabel('Count (log scale)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()


def compute_small_world_metrics(G: nx.Graph | nx.DiGraph) -> dict:
    """Compute small-world metrics by comparing to equivalent random graph."""
    N = G.number_of_nodes()
    E = G.number_of_edges()

    if N < 2 or E < 1:
        return {'C': 0, 'L': float('inf'), 'C_rand': 0, 'L_rand': float('inf'),
                'C_norm': 0, 'L_norm': float('inf'), 'sigma': 0}

    C = nx.average_clustering(G)
    L = get_average_path_length(G)

    # Average over multiple random graphs
    C_rands, L_rands = [], []
    for _ in range(10):
        if isinstance(G, nx.DiGraph):
            g_rand = nx.gnm_random_graph(N, E, directed=True)
        else:
            g_rand = nx.gnm_random_graph(N, E, directed=False)
        C_rands.append(nx.average_clustering(g_rand))
        L_rands.append(get_average_path_length(g_rand))

    C_rand = np.mean(C_rands)
    L_rand = np.mean([l for l in L_rands if l != float('inf')]) or float('inf')

    C_norm = C / C_rand if C_rand > 0 else 0
    L_norm = L / L_rand if L_rand > 0 and L_rand != float('inf') else float('inf')
    sigma = C_norm / L_norm if L_norm > 0 and L_norm != float('inf') else 0

    return {'C': C, 'L': L, 'C_rand': C_rand, 'L_rand': L_rand,
            'C_norm': C_norm, 'L_norm': L_norm, 'sigma': sigma}


# =============================================================================
# Task 1: Kaiser-Hilgetag
# =============================================================================

def task_1_2():
    """Task 1.2: Plot adjacency matrix and network for α=1, 5, 10 with β=0.5, N=100"""
    print("=" * 60)
    print("Task 1.2: Kaiser-Hilgetag plots for α=1, 5, 10 (β=0.5, N=100)")
    print("=" * 60)

    for alpha in [1, 5, 10]:
        print(f"  Processing α={alpha}...")
        random.seed(SEED)
        np.random.seed(SEED)

        adj_matrix, positions = kh.run(alpha=alpha, beta=0.5, N=100)
        G = kh.to_graph(adj_matrix, positions)

        # Adjacency matrix
        plot_adjacency_matrix(adj_matrix,
                              title=f"Kaiser-Hilgetag Adjacency (α={alpha}, β=0.5)",
                              file_path=RESULTS_DIR / f"task1_2_kh_adj_alpha_{alpha}.png")

        # Network
        kh.draw_graph(G, positions, RESULTS_DIR / f"task1_2_kh_network_alpha_{alpha}.png")

        print(f"    Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("  Task 1.2 complete!\n")


def task_1_3(num_realizations: int = 5):
    """Task 1.3: Parameter scan α∈[0.1, 100] with β=1, N=100 (with averaging)"""
    print("=" * 60)
    print(f"Task 1.3: Kaiser-Hilgetag parameter scan ({num_realizations} realizations)")
    print("=" * 60)

    alphas = list(np.linspace(0.1, 100, 30))
    all_C = {a: [] for a in alphas}
    all_L = {a: [] for a in alphas}

    for r in range(num_realizations):
        print(f"  Realization {r+1}/{num_realizations}...")
        random.seed(SEED + r)
        np.random.seed(SEED + r)

        for alpha in alphas:
            adj, pos = kh.run(alpha=alpha, beta=1.0, N=100)
            G = kh.to_graph(adj, pos)
            all_C[alpha].append(nx.average_clustering(G))
            all_L[alpha].append(get_average_path_length(G))

    mean_C = [np.mean(all_C[a]) for a in alphas]
    std_C = [np.std(all_C[a]) for a in alphas]
    mean_L = [np.mean([l for l in all_L[a] if l != float('inf')]) for a in alphas]
    std_L = [np.std([l for l in all_L[a] if l != float('inf')]) for a in alphas]

    # Plot with error bars
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel(r"$\alpha$ (Spatial Penalty)")
    ax1.set_ylabel("Clustering Coefficient", color='teal')
    ax1.errorbar(alphas, mean_C, yerr=std_C, fmt='o-', color='teal',
                 alpha=0.7, capsize=3, markersize=4, label='Clustering')
    ax1.tick_params(axis='y', labelcolor='teal')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg Path Length", color='crimson')
    ax2.errorbar(alphas, mean_L, yerr=std_L, fmt='s-', color='crimson',
                 alpha=0.7, capsize=3, markersize=4, label='Path Length')
    ax2.tick_params(axis='y', labelcolor='crimson')

    plt.title("Kaiser-Hilgetag: Clustering & Path Length vs Alpha (cf. Fig 3a)")
    fig.tight_layout()
    plt.savefig(RESULTS_DIR / "task1_3_kh_parameter_scan.png")
    plt.close()

    print("  Task 1.3 complete!\n")
    return alphas, mean_C, mean_L


def task_1_4(num_realizations: int = 5):
    """Task 1.4: Identify small-world parameter regimes"""
    print("=" * 60)
    print("Task 1.4: Kaiser-Hilgetag small-world analysis")
    print("=" * 60)

    alphas = list(np.linspace(0.1, 100, 20))
    results = {a: {'C_norm': [], 'L_norm': [], 'sigma': []} for a in alphas}

    for r in range(num_realizations):
        print(f"  Realization {r+1}/{num_realizations}...")
        random.seed(SEED + r)
        np.random.seed(SEED + r)

        for alpha in alphas:
            adj, pos = kh.run(alpha=alpha, beta=1.0, N=100)
            G = kh.to_graph(adj, pos)
            metrics = compute_small_world_metrics(G)
            results[alpha]['C_norm'].append(metrics['C_norm'])
            results[alpha]['L_norm'].append(metrics['L_norm'])
            results[alpha]['sigma'].append(metrics['sigma'])

    mean_C_norm = [np.mean(results[a]['C_norm']) for a in alphas]
    mean_L_norm = [np.mean([l for l in results[a]['L_norm'] if l != float('inf')]) for a in alphas]
    mean_sigma = [np.mean(results[a]['sigma']) for a in alphas]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(alphas, mean_C_norm, 'o-', color='teal', label=r'$C/C_{rand}$')
    ax1.plot(alphas, mean_L_norm, 's-', color='crimson', label=r'$L/L_{rand}$')
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel("Normalized metric")
    ax1.set_title("Normalized C and L")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(alphas, mean_sigma, 'o-', color='purple')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$\sigma = (C/C_{rand}) / (L/L_{rand})$")
    ax2.set_title("Small-world coefficient σ")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "task1_4_kh_small_world.png")
    plt.close()

    # Print small-world regimes
    print("\n  Small-world regimes (σ > 1):")
    for i, a in enumerate(alphas):
        if mean_sigma[i] > 1:
            print(f"    α={a:.1f}: σ={mean_sigma[i]:.2f}")

    print("  Task 1.4 complete!\n")


# =============================================================================
# Task 2: Vértes
# =============================================================================

def task_2_2():
    """Task 2.2: Plot adjacency matrix and network for α=5, 10, 20 with N=100"""
    print("=" * 60)
    print("Task 2.2: Vértes plots for α=5, 10, 20 (N=100)")
    print("=" * 60)

    for alpha in [5, 10, 20]:
        print(f"  Processing α={alpha}...")
        vt.generator = np.random.Generator(np.random.MT19937(SEED))

        adj_matrix, positions = vt.run_simulation(alpha=alpha, N=100)
        G = vt.to_graph(adj_matrix, positions)

        # Binary adjacency
        plot_adjacency_matrix(adj_matrix,
                              title=f"Vértes Adjacency Binary (α={alpha})",
                              file_path=RESULTS_DIR / f"task2_2_vt_adj_binary_alpha_{alpha}.png",
                              show_weights=False)

        # Weighted adjacency
        plot_adjacency_matrix(adj_matrix,
                              title=f"Vértes Adjacency Weighted (α={alpha})",
                              file_path=RESULTS_DIR / f"task2_2_vt_adj_weighted_alpha_{alpha}.png",
                              show_weights=True)

        # Network
        vt.draw_graph(G, file_path=RESULTS_DIR / f"task2_2_vt_network_alpha_{alpha}.png")

        print(f"    Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("  Task 2.2 complete!\n")


def task_2_3(num_realizations: int = 3):
    """Task 2.3: Parameter scan α∈[0.1, 100] for Vértes"""
    print("=" * 60)
    print(f"Task 2.3: Vértes parameter scan ({num_realizations} realizations)")
    print("=" * 60)

    alphas = list(np.linspace(0.1, 100, 20))
    all_C = {a: [] for a in alphas}
    all_L = {a: [] for a in alphas}

    for r in range(num_realizations):
        print(f"  Realization {r+1}/{num_realizations}...")

        for alpha in alphas:
            vt.generator = np.random.Generator(np.random.MT19937(SEED + r))
            adj, pos = vt.run_simulation(alpha=alpha, N=100)
            G = vt.to_graph(adj, pos)
            all_C[alpha].append(nx.average_clustering(G))
            all_L[alpha].append(get_average_path_length(G))

    mean_C = [np.mean(all_C[a]) for a in alphas]
    mean_L = [np.mean([l for l in all_L[a] if l != float('inf')]) for a in alphas]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(alphas, mean_C, 'o-', color='teal')
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel("Clustering Coefficient")
    ax1.set_title("Vértes: Clustering vs Alpha")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(alphas, mean_L, 'o-', color='crimson')
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("Avg Path Length")
    ax2.set_title("Vértes: Path Length vs Alpha")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "task2_3_vt_parameter_scan.png")
    plt.close()

    print("  Task 2.3 complete!\n")


def task_2_4():
    """Task 2.4: Degree and weight distributions for α=15, N=100"""
    print("=" * 60)
    print("Task 2.4: Vértes distributions (α=15, N=100)")
    print("=" * 60)

    vt.generator = np.random.Generator(np.random.MT19937(SEED))
    adj, pos = vt.run_simulation(alpha=15, N=100)
    G = vt.to_graph(adj, pos)

    # Degree distribution
    plot_degree_distribution(G, title="Vértes Degree Distribution (α=15)",
                             file_path=RESULTS_DIR / "task2_4_vt_degree_dist.png")

    # Weight distribution
    plot_weight_distribution(G, title="Vértes Weight Distribution (α=15)",
                             file_path=RESULTS_DIR / "task2_4_vt_weight_dist.png")

    # Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    in_deg = [d for n, d in G.in_degree()]
    out_deg = [d for n, d in G.out_degree()]
    ax1.hist([in_deg, out_deg], bins=20, label=['In', 'Out'], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Count')
    ax1.set_title('Degree Distribution (α=15)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    weights = [G[u][v]['weight'] for u, v in G.edges()]
    ax2.hist(weights, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Weight (log)')
    ax2.set_ylabel('Count (log)')
    ax2.set_title('Weight Distribution (α=15)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "task2_4_vt_combined.png")
    plt.close()

    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"  Weight range: {min(weights)} - {max(weights)}")
    print("  Task 2.4 complete!\n")


# =============================================================================
# Task 3: Mouse comparison
# =============================================================================

def task_3(mouse_file: str = "mouse_V1_adjacency_matrix.npy"):
    """Task 3: Compare with mouse visual cortex data"""
    print("=" * 60)
    print("Task 3: Mouse visual cortex comparison")
    print("=" * 60)

    # Try to load mouse data
    try:
        mouse_adj = np.load(mouse_file)
        print(f"  Loaded mouse data: shape={mouse_adj.shape}")
    except FileNotFoundError:
        print(f"  WARNING: {mouse_file} not found. Using synthetic data.")
        # Create synthetic small-world network
        np.random.seed(SEED)
        G_synth = nx.watts_strogatz_graph(100, k=10, p=0.1, seed=SEED)
        for u, v in G_synth.edges():
            G_synth[u][v]['weight'] = np.random.randint(1, 10)
        mouse_adj = nx.to_numpy_array(G_synth)

    N = mouse_adj.shape[0]
    is_directed = not np.allclose(mouse_adj, mouse_adj.T)

    # Task 3.1: Visualize mouse network
    print("  3.1: Visualizing mouse network...")
    plot_adjacency_matrix(mouse_adj, "Mouse V1 Adjacency (Binary)",
                          RESULTS_DIR / "task3_1_mouse_adj_binary.png", show_weights=False)
    plot_adjacency_matrix(mouse_adj, "Mouse V1 Adjacency (Weighted)",
                          RESULTS_DIR / "task3_1_mouse_adj_weighted.png", show_weights=True)

    if is_directed:
        G_mouse = nx.from_numpy_array(mouse_adj, create_using=nx.DiGraph)
    else:
        G_mouse = nx.from_numpy_array(mouse_adj)

    # Spring layout visualization
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G_mouse, seed=SEED)
    degrees = dict(G_mouse.degree())
    node_colors = [degrees[n] for n in G_mouse.nodes()]
    nx.draw(G_mouse, pos, node_color=node_colors, node_size=30,
            edge_color='gray', alpha=0.5, width=0.3, cmap=plt.cm.viridis)
    plt.title(f"Mouse V1 Network (N={N})")
    plt.savefig(RESULTS_DIR / "task3_1_mouse_network.png", dpi=150)
    plt.close()

    # Task 3.2: Compute metrics and compare
    print("  3.2: Computing metrics...")
    mouse_density = np.count_nonzero(mouse_adj) / (N * (N-1)) if is_directed else np.count_nonzero(mouse_adj) / (N * (N-1))
    mouse_C = nx.average_clustering(G_mouse)
    mouse_L = get_average_path_length(G_mouse)

    print(f"    Mouse: N={N}, density={mouse_density:.4f}, C={mouse_C:.4f}, L={mouse_L:.2f}")

    # Generate comparison networks
    random.seed(SEED)
    np.random.seed(SEED)
    kh_adj, kh_pos = kh.run(alpha=5, beta=1.0, N=N)
    G_kh = kh.to_graph(kh_adj, kh_pos)
    kh_C = nx.average_clustering(G_kh)
    kh_L = get_average_path_length(G_kh)

    vt.generator = np.random.Generator(np.random.MT19937(SEED))
    vt_adj, vt_pos = vt.run_simulation(alpha=10, N=N, rho=mouse_density)
    G_vt = vt.to_graph(vt_adj, vt_pos)
    vt_C = nx.average_clustering(G_vt)
    vt_L = get_average_path_length(G_vt)

    print(f"    Kaiser-H: C={kh_C:.4f}, L={kh_L:.2f}")
    print(f"    Vértes: C={vt_C:.4f}, L={vt_L:.2f}")

    # Comparison bar plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = ['Mouse', 'Kaiser-H', 'Vértes']

    axes[0].bar(names, [mouse_C, kh_C, vt_C], color=['blue', 'green', 'orange'])
    axes[0].set_ylabel('Clustering')
    axes[0].set_title('Clustering Comparison')

    axes[1].bar(names, [mouse_L, kh_L, vt_L], color=['blue', 'green', 'orange'])
    axes[1].set_ylabel('Path Length')
    axes[1].set_title('Path Length Comparison')

    mouse_deg = [d for n, d in G_mouse.degree()]
    kh_deg = [d for n, d in G_kh.degree()]
    vt_deg = [d for n, d in G_vt.degree()]
    axes[2].hist([mouse_deg, kh_deg, vt_deg], bins=15, label=names, alpha=0.6)
    axes[2].set_xlabel('Degree')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Degree Distribution')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "task3_2_comparison.png")
    plt.close()

    print("  Task 3 complete!\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING ALL ANALYSIS TASKS")
    print("=" * 60 + "\n")

    # Task 1: Kaiser-Hilgetag
    task_1_2()
    task_1_3(num_realizations=3)  # Use more for paper-quality
    task_1_4(num_realizations=3)

    # Task 2: Vértes
    task_2_2()
    task_2_3(num_realizations=2)
    task_2_4()

    # Task 3: Mouse comparison
    task_3()

    print("=" * 60)
    print(f"All tasks complete! Results saved to: {RESULTS_DIR}")
    print("=" * 60)