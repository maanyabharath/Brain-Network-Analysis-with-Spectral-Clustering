import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh
from scipy.cluster.vq import kmeans2
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple


def load_data(
    n_regions: int = 90,
    n_modules: int = 3,
    group_type: str = "healthy",
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Simulate a brain connectivity matrix.

    Parameters
    ----------
    n_regions : int
        Number of brain regions (nodes).
    n_modules : int
        Number of latent functional/structural modules.
    group_type : {"healthy", "alz"}
        Type of simulated group.
    random_state : Optional[int]
        RNG seed for reproducibility.

    Returns
    -------
    conn : (n_regions, n_regions) ndarray
        Symmetric connectivity matrix.
    """
    rng = np.random.default_rng(random_state)

    # Assign each region to a module
    modules = np.repeat(np.arange(n_modules), n_regions // n_modules)
    # If not divisible, assign remaining nodes to last module
    if len(modules) < n_regions:
        modules = np.concatenate(
            [modules, np.full(n_regions - len(modules), n_modules - 1)]
        )

    # Base connectivity parameters
    if group_type == "healthy":
        within_mean, within_std = 0.9, 0.05
        between_mean, between_std = 0.2, 0.05
    elif group_type == "alz":
        # Reduced within-module strength, slightly noisier and more random
        within_mean, within_std = 0.6, 0.1
        between_mean, between_std = 0.25, 0.07
    else:
        raise ValueError("group_type must be 'healthy' or 'alz'")

    conn = np.zeros((n_regions, n_regions), dtype=float)

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            if modules[i] == modules[j]:
                w = rng.normal(within_mean, within_std)
            else:
                w = rng.normal(between_mean, between_std)
            w = max(w, 0.0)  # no negative weights
            conn[i, j] = conn[j, i] = w

    # Zero self-connections
    np.fill_diagonal(conn, 0.0)

    # Slight random noise to avoid perfect block structure
    noise_scale = 0.02 if group_type == "healthy" else 0.04
    noise = rng.normal(0, noise_scale, size=conn.shape)
    noise = (noise + noise.T) / 2.0
    conn = np.clip(conn + noise, 0.0, None)

    return conn


def construct_graph(conn_matrix: np.ndarray, threshold: Optional[float] = None) -> nx.Graph:
    """
    Construct a NetworkX graph from a connectivity matrix.

    If threshold is provided, edges with weight < threshold are removed.
    """
    n_regions = conn_matrix.shape[0]
    if conn_matrix.shape[0] != conn_matrix.shape[1]:
        raise ValueError("Connectivity matrix must be square.")

    mat = conn_matrix.copy()

    if threshold is not None:
        mat[mat < threshold] = 0.0

    # Create weighted, undirected graph
    G = nx.from_numpy_array(mat)
    # Name nodes as R0, R1, ...
    mapping = {i: f"R{i}" for i in range(n_regions)}
    G = nx.relabel_nodes(G, mapping)

    return G


def spectral_clustering(
    G: nx.Graph,
    n_clusters: int = 3,
    use_normalized_laplacian: bool = True,
    random_state: Optional[int] = 42,
) -> Dict[str, int]:
    """
    Perform spectral clustering on a graph.

    Returns a mapping: node -> cluster_label (0..n_clusters-1).
    """
    # Get adjacency matrix with consistent node ordering
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")

    # Compute Laplacian
    L = csgraph.laplacian(A, normed=use_normalized_laplacian)

    # Eigen decomposition (smallest eigenvalues)
    # eigh returns eigenvalues in ascending order
    evals, evecs = eigh(L)

    # Skip first eigenvector (associated with eigenvalue ~0)
    # Take next n_clusters eigenvectors
    X = evecs[:, 1 : n_clusters + 1]

    # Normalize rows to unit length for k-means stability
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    X_norm = X / row_norms

    # k-means clustering on spectral embedding
    rng = np.random.default_rng(random_state)
    # kmeans2 sometimes benefits from multiple initializations,
    # but here we rely on internal randomness for simplicity.
    centroids, labels = kmeans2(
        X_norm.astype(float), n_clusters, minit="points", iter=100, seed=rng.integers(1e9)
    )

    # Map node -> cluster label
    cluster_assignments = {node: int(label) for node, label in zip(nodes, labels)}
    return cluster_assignments


def identify_hubs(G: nx.Graph, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Identify hub regions using weighted degree (strength).
    """
    # Weighted degree (sum of edge weights)
    strength = dict(G.degree(weight="weight"))
    # Sort in descending order
    hubs = sorted(strength.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return hubs


def visualize_network(
    G: nx.Graph,
    clusters: Dict[str, int],
    title: str = "Brain Network",
    node_size: int = 200,
) -> None:
    """
    Plot the graph with nodes colored by cluster.
    """
    plt.figure(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=42, weight="weight", k=None)

    # Group nodes by cluster
    cluster_labels = sorted(set(clusters.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_labels)))

    for c, color in zip(cluster_labels, colors):
        nodes_in_c = [n for n, lab in clusters.items() if lab == c]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_in_c,
            node_color=[color],
            label=f"Cluster {c}",
            node_size=node_size,
            alpha=0.9,
        )

    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.tight_layout()


def visualize_connectivity_matrix(
    conn_matrix: np.ndarray,
    clusters: Dict[str, int],
    title: str = "Connectivity Matrix",
) -> None:
    """
    Plot connectivity matrix as heatmap, ordered by cluster labels.
    """
    # Get node order from cluster labels
    node_names = sorted(clusters.keys(), key=lambda n: int(n[1:]))  # sort by index in R#
    labels = np.array([clusters[n] for n in node_names])

    # Reorder matrix by cluster, then by node index
    order = np.argsort(labels)
    ordered_mat = conn_matrix[order][:, order]
    ordered_labels = labels[order]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        ordered_mat,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Connection strength"},
    )
    plt.title(title)
    plt.xlabel("Regions (ordered by cluster)")
    plt.ylabel("Regions (ordered by cluster)")
    plt.tight_layout()


def compare_cluster_assignments(
    clusters_healthy: Dict[str, int],
    clusters_alz: Dict[str, int],
) -> List[str]:
    """
    Compare cluster assignments between healthy and Alzheimer's graphs.

    Returns list of regions whose cluster changed.
    """
    changed_nodes = []
    for node in clusters_healthy.keys():
        if clusters_healthy[node] != clusters_alz.get(node, None):
            changed_nodes.append(node)
    return changed_nodes


def main():
    # Parameters
    n_regions = 90
    n_modules = 3
    n_clusters = n_modules
    threshold = 0.1  # prune very weak connections

    # 1. Load / simulate data
    conn_healthy = load_data(n_regions=n_regions, n_modules=n_modules, group_type="healthy")
    conn_alz = load_data(n_regions=n_regions, n_modules=n_modules, group_type="alz")

    # 2. Graph construction
    G_healthy = construct_graph(conn_healthy, threshold=threshold)
    G_alz = construct_graph(conn_alz, threshold=threshold)

    # 3. Spectral clustering
    clusters_healthy = spectral_clustering(G_healthy, n_clusters=n_clusters)
    clusters_alz = spectral_clustering(G_alz, n_clusters=n_clusters)

    # 4. Analysis: hubs
    hubs_healthy = identify_hubs(G_healthy, top_k=5)
    hubs_alz = identify_hubs(G_alz, top_k=5)

    # 4. Analysis: cluster comparison
    changed_nodes = compare_cluster_assignments(clusters_healthy, clusters_alz)

    # 6. Output: print clusters and hubs
    print("=== Healthy group: cluster assignments ===")
    for node, lab in sorted(clusters_healthy.items(), key=lambda x: int(x[0][1:])):
        print(f"{node}: Cluster {lab}")

    print("\n=== Alzheimer's group: cluster assignments ===")
    for node, lab in sorted(clusters_alz.items(), key=lambda x: int(x[0][1:])):
        print(f"{node}: Cluster {lab}")

    print("\n=== Healthy group: top hub regions (by strength) ===")
    for node, s in hubs_healthy:
        print(f"{node}: strength={s:.3f}")

    print("\n=== Alzheimer's group: top hub regions (by strength) ===")
    for node, s in hubs_alz:
        print(f"{node}: strength={s:.3f}")

    print("\n=== Nodes whose cluster assignment changed (healthy -> Alzheimer's) ===")
    print(", ".join(changed_nodes) if changed_nodes else "None")

    # 5. Visualization
    plt.figure(figsize=(12, 10))

    # Connectivity matrices
    plt.subplot(2, 2, 1)
    visualize_connectivity_matrix(
        conn_healthy,
        clusters_healthy,
        title="Healthy Connectivity (ordered by clusters)",
    )

    plt.subplot(2, 2, 2)
    visualize_connectivity_matrix(
        conn_alz,
        clusters_alz,
        title="Alzheimer's Connectivity (ordered by clusters)",
    )

    # Graphs
    plt.subplot(2, 2, 3)
    visualize_network(G_healthy, clusters_healthy, title="Healthy Brain Network")

    plt.subplot(2, 2, 4)
    visualize_network(G_alz, clusters_alz, title="Alzheimer's Brain Network")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()