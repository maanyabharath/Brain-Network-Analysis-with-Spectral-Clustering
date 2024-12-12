# Brain-Network-Analysis-with-Spectral-Clustering


Overview

The primary goal of this project is to analyze brain networks to understand the structural and functional connections between different regions of the brain. Graph theory forms the foundation of this analysis, where each brain region is represented as a node, and the connections between regions are modeled as edges.

Key aspects of this study include:

Observing the global and local organization of the brain.

Identifying key regions or "hubs" and analyzing how information flows through the brain.

Using spectral clustering to uncover latent regions and groupings within the brain network.

In the context of Alzheimer's disease, this approach helps identify disrupted connectivity patterns between brain regions. These insights can contribute to early diagnosis and a deeper understanding of disease progression.

Features

Brain Network Representation

Nodes: Represent distinct brain regions.

Edges: Represent connections (functional or structural) between nodes.

Spectral Clustering

Utilized to group brain regions based on connectivity patterns.

Helps reveal latent regions and potential hubs in the brain.

Analysis for Alzheimer's Disease

Identification of disrupted connectivity patterns in Alzheimer’s-affected brains.

Supports early detection and monitoring of disease progression.

Methodology

Graph Construction

Brain imaging data is used to create a graph representation.

Preprocessing includes cleaning and normalizing connectivity matrices.

Spectral Clustering

Compute the Laplacian matrix of the graph.

Perform eigenvalue decomposition to identify meaningful clusters.

Group nodes into clusters to detect hubs and latent regions.

Analysis

Evaluate the global and local organization of the brain network.

Compare connectivity patterns between healthy individuals and Alzheimer's patients.

Results

Hubs Identification: Key regions of the brain responsible for major information flow were identified.

Latent Groupings: Spectral clustering revealed distinct groupings of brain regions.

Alzheimer’s Insights: Significant disruptions in connectivity patterns were observed in Alzheimer's patients, aiding early diagnosis and understanding of the disease.

Technologies Used

Python: For data processing, graph construction, and spectral clustering.

Libraries:

NumPy and SciPy: Matrix operations and eigenvalue computations.

NetworkX: Graph representation and analysis.

Matplotlib and Seaborn: Visualization of connectivity patterns and clusters.
