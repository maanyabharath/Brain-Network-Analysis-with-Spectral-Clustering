# Brain Network Analysis Using Spectral Clustering

## Overview

The primary goal of this project is to analyze brain networks to understand the structural and functional connections between different regions of the brain. Graph theory forms the foundation of this analysis, where each brain region is represented as a **node**, and the connections between regions are modeled as **edges**.

This repository implements a complete, reproducible pipeline using **synthetic brain connectivity data**, **graph construction**, and **spectral clustering** to uncover latent groupings of brain regions and highlight disrupted connectivity patterns between a simulated healthy group and an Alzheimer's group.

Key aspects of this project include:

- Observing the global and local organization of the brain.
- Identifying key regions or "hubs" and analyzing how information flows through the brain.
- Using spectral clustering to uncover latent regions and groupings within the brain network.
- Comparing cluster assignments between healthy vs. Alzheimer-like connectivity.

---

## Repository Structure

- `brain_network_spectral_clustering.py`: Main script implementing data simulation, graph construction, spectral clustering, analysis, and visualization.
- `requirements.txt`: Python dependencies required to run the project.
- `README.md`: Project description and usage instructions.

---

## Installation

1. **Clone the repository** (or download the folder):

   ```bash
   git clone <your-repo-url>
   cd Brain-Network-Analysis-with-Spectral-Clustering
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How It Works

### 1. Synthetic Connectivity Data

The script simulates **square connectivity matrices** (regions × regions) for:

- a **healthy** group, and
- an **Alzheimer-like (alz)** group.

Each matrix represents weighted connections between brain regions, with stronger within-module connectivity and weaker between-module connectivity. In the Alzheimer-like group, within-module connectivity is reduced and noise is increased to mimic disrupted network organization.

### 2. Graph Construction

- Each brain region becomes a **node**.
- Each connection (matrix entry) becomes a weighted **edge**.
- `NetworkX` is used to build the graph from the connectivity matrix.
- Very weak connections can be pruned with a threshold to simplify the network.

### 3. Spectral Clustering

For each group (healthy and alz), the pipeline:

1. Computes the (normalized) **graph Laplacian** from the adjacency matrix.
2. Performs **eigenvalue decomposition** using `SciPy` (`eigh`).
3. Uses the smallest non-trivial eigenvectors as a low-dimensional embedding.
4. Applies **k-means** (via `scipy.cluster.vq.kmeans2`) to cluster nodes into latent groups.

This reveals communities (modules) of brain regions with similar connectivity patterns.

### 4. Analysis

- **Hubs** are identified by weighted degree (node strength) within each graph.
- **Cluster assignments** are compared between healthy and Alzheimer-like networks to find regions whose community membership changes, highlighting disrupted connectivity.

### 5. Visualization

The script uses `Matplotlib` and `Seaborn` to produce:

- Heatmaps of the **connectivity matrices**, ordered by cluster label.
- Graph visualizations of the **brain networks**, with nodes colored by their spectral cluster.

---

## Running the Pipeline

From the project root:

```bash
python brain_network_spectral_clustering.py
```

The script will:

- Simulate connectivity matrices for healthy and Alzheimer-like groups.
- Build corresponding graphs with `NetworkX`.
- Run spectral clustering to obtain node clusters.
- Print:
  - cluster assignments for each node (region),
  - top hub regions in each group,
  - nodes whose cluster membership changes between healthy and Alzheimer-like networks.
- Display plots of:
  - connectivity heatmaps (healthy vs. alz), and
  - brain network graphs with nodes colored by cluster.

---

## Technologies Used

- **Python**: Data processing, graph construction, and spectral clustering.
- **NumPy** and **SciPy**: Matrix operations, Laplacian computation, eigenvalue decomposition, and k-means.
- **NetworkX**: Graph representation and analysis.
- **Matplotlib** and **Seaborn**: Visualization of connectivity patterns and clusters.

---

## Extending the Project

You can adapt this pipeline to real brain connectivity data by:

- Replacing the synthetic data generation in `load_data()` with loading real connectivity matrices (e.g., from fMRI or DTI).
- Adjusting the number of regions, modules, and clusters.
- Adding more metrics (e.g., modularity, path length, betweenness centrality) for a richer network analysis.

# Brain Network Analysis Using Spectral Clustering

## Overview

The primary goal of this project is to analyze brain networks to understand the structural and functional connections between different regions of the brain. Graph theory forms the foundation of this analysis, where each brain region is represented as a **node**, and the connections between regions are modeled as **edges**.

Key aspects of this study include:

- Observing the global and local organization of the brain.
- Identifying key regions or "hubs" and analyzing how information flows through the brain.
- Using spectral clustering to uncover latent regions and groupings within the brain network.

In the context of Alzheimer's disease, this approach helps identify disrupted connectivity patterns between brain regions. These insights can contribute to early diagnosis and a deeper understanding of disease progression.

---

## Features

1. **Brain Network Representation**
   - Nodes: Represent distinct brain regions.
   - Edges: Represent connections (functional or structural) between nodes.

2. **Spectral Clustering**
   - Utilized to group brain regions based on connectivity patterns.
   - Helps reveal latent regions and potential hubs in the brain.

3. **Analysis for Alzheimer's Disease**
   - Identification of disrupted connectivity patterns in Alzheimer\u2019s-affected brains.
   - Supports early detection and monitoring of disease progression.

---

## Methodology

1. **Graph Construction**
   - Brain imaging data is used to create a graph representation.
   - Preprocessing includes cleaning and normalizing connectivity matrices.

2. **Spectral Clustering**
   - Compute the Laplacian matrix of the graph.
   - Perform eigenvalue decomposition to identify meaningful clusters.
   - Group nodes into clusters to detect hubs and latent regions.

3. **Analysis**
   - Evaluate the global and local organization of the brain network.
   - Compare connectivity patterns between healthy individuals and Alzheimer's patients.

---

## Results

- **Hubs Identification:** Key regions of the brain responsible for major information flow were identified.
- **Latent Groupings:** Spectral clustering revealed distinct groupings of brain regions.
- **Alzheimer\u2019s Insights:** Significant disruptions in connectivity patterns were observed in Alzheimer's patients, aiding early diagnosis and understanding of the disease.

---

## Technologies Used

- **Python**: For data processing, graph construction, and spectral clustering.
- **Libraries**:
  - `NumPy` and `SciPy`: Matrix operations and eigenvalue computations.
  - `NetworkX`: Graph representation and analysis.
  - `Matplotlib` and `Seaborn`: Visualization of connectivity patterns and clusters.
