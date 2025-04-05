# cluster_trajectories_from_action.py

import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from mma_wrapper.temm.distances import (
    distance_lcs,
    distance_fuzzy_lcs,
    distance_smith_waterman,
    one_hot_encode_trajectories
)


def compute_distance_matrix(sequences: List, method: str, num_actions: int = None) -> np.ndarray:
    n = len(sequences)
    dist_matrix = np.zeros((n, n))

    if method in ["euclidean", "cosine"]:
        encoded = one_hot_encode_trajectories(sequences, num_actions)
        vectors = [np.concatenate(seq) for seq in encoded]
        if method == "euclidean":
            dist_matrix = euclidean_distances(vectors)
        elif method == "cosine":
            dist_matrix = cosine_distances(vectors)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if method == "lcs":
                    dist = distance_lcs(sequences[i], sequences[j])
                elif method == "fuzzy_lcs":
                    dist = distance_fuzzy_lcs(sequences[i], sequences[j])
                elif method == "smith_waterman":
                    dist = distance_smith_waterman(sequences[i], sequences[j])
                else:
                    raise ValueError(f"Unsupported distance method: {method}")
                dist_matrix[i, j] = dist_matrix[j, i] = dist

    return dist_matrix


def cluster_trajectories_from_action(action_trajectories: List[List[int]], distance_method: str, num_actions: int = None) -> Dict[int, List[int]]:
    """Cluster trajectories based on action sequences."""
    dist_matrix = compute_distance_matrix(
        action_trajectories, distance_method, num_actions)
    linkage_matrix = linkage(dist_matrix, method="average")
    labels = fcluster(linkage_matrix, t=1.0, criterion='distance')
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    return clusters


def cluster_trajectories_from_observation(observation_trajectories: List[List[np.ndarray]], distance_method: str) -> Dict[int, List[int]]:
    """Cluster trajectories based on observation sequences."""
    n = len(observation_trajectories)
    vectors = [np.concatenate(seq) for seq in observation_trajectories]
    if distance_method == "euclidean":
        dist_matrix = euclidean_distances(vectors)
    elif distance_method == "cosine":
        dist_matrix = cosine_distances(vectors)
    else:
        raise ValueError(
            f"Unsupported distance method for observations: {distance_method}")

    linkage_matrix = linkage(dist_matrix, method="average")
    labels = fcluster(linkage_matrix, t=1.0, criterion='distance')
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    return clusters


def cluster_full_trajectories(full_trajectories: List[List[Tuple[np.ndarray, int]]], distance_method: str) -> Dict[int, List[int]]:
    """Cluster trajectories based on full (obs, action) sequences."""
    flattened = [np.concatenate([np.concatenate((obs, np.eye(max([a for _, a in traj])+1)[act])) for obs, act in traj])
                 for traj in full_trajectories]
    if distance_method == "euclidean":
        dist_matrix = euclidean_distances(flattened)
    elif distance_method == "cosine":
        dist_matrix = cosine_distances(flattened)
    else:
        raise ValueError(
            f"Unsupported distance method for full trajectories: {distance_method}")

    linkage_matrix = linkage(dist_matrix, method="average")
    labels = fcluster(linkage_matrix, t=1.0, criterion='distance')
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    return clusters
