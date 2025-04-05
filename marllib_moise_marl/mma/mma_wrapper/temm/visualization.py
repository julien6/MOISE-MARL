# mma_wrapper/temm/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


def plot_dendrogram(data: List[np.ndarray], title: str, save_path: str):
    """Generic function to plot and save a dendrogram from a list of vectors."""
    dist_matrix = squareform(pdist(data, metric='euclidean'))
    linkage_matrix = linkage(dist_matrix, method="average")
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix)
    plt.title(title)
    plt.xlabel("Trajectory")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pca(data: List[np.ndarray], title: str, save_path: str):
    """Generic function to compute and plot PCA projection from a list of vectors."""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.array(data))
    plt.figure(figsize=(8, 6))
    for i, point in enumerate(reduced):
        plt.scatter(point[0], point[1], label=str(i))
        plt.text(point[0], point[1], str(i), fontsize=8)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# Wrappers for specific use cases

def generate_actions_dendrogram(data: List[np.ndarray]):
    plot_dendrogram(data, "Dendrogram - Action Trajectories",
                    "figures/actions_dendrogram.png")


def generate_observations_dendrogram(data: List[np.ndarray]):
    plot_dendrogram(data, "Dendrogram - Observation Trajectories",
                    "figures/observations_dendrogram.png")


def generate_dendrogram(data: List[np.ndarray]):
    plot_dendrogram(data, "Dendrogram - Full Trajectories",
                    "figures/full_dendrogram.png")


def visualize_action_pca(data: List[np.ndarray]):
    plot_pca(data, "PCA - Actions (1 point = 1 action)",
             "figures/action_pca.png")


def visualize_observation_pca(data: List[np.ndarray]):
    plot_pca(data, "PCA - Observations (1 point = 1 observation)",
             "figures/observation_pca.png")


def visualize_transition_pca(data: List[np.ndarray]):
    plot_pca(data, "PCA - Transitions (1 point = 1 transition)",
             "figures/transition_pca.png")


def visualize_action_trajectory(data: List[np.ndarray]):
    plot_pca(data, "PCA - Action Trajectories (1 point = 1 trajectory)",
             "figures/action_trajectory_pca.png")


def visualize_observation_trajectory(data: List[np.ndarray]):
    plot_pca(data, "PCA - Observation Trajectories (1 point = 1 trajectory)",
             "figures/observation_trajectory_pca.png")


def visualize_trajectory(data: List[np.ndarray]):
    plot_pca(data, "PCA - Full Trajectories (1 point = 1 trajectory)",
             "figures/full_trajectory_pca.png")
