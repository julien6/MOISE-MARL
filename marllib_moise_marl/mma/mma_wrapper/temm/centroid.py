# mma_wrapper/temm/centroid.py

import numpy as np
from typing import Dict, List, Any, Union
from collections import Counter
from scipy.spatial.distance import euclidean


def compute_centroids_per_cluster(clusters: Dict[int, List[Any]], mode: str = "mean") -> Dict[int, Any]:
    """
    Compute a centroid for each cluster using the specified mode.
    
    Parameters:
        clusters (Dict[int, List[Any]]): Dictionary mapping cluster_id to a list of elements (sequences or vectors).
        mode (str): "mean" for vector averaging, "fuzzy_lcs" for symbolic sequence centroid.
        
    Returns:
        Dict[int, Any]: Dictionary mapping cluster_id to its centroid.
    """
    print("---===>>> ", clusters)

    num_actions = max(np.concatenate(
        [[a for _, a in traj] for _, trajectories in clusters.items() for traj in trajectories])) + 1
    
    print("---===>>> ", num_actions)
    print("---===>>> ", encoded_action_trajectories)

    encoded_action_trajectories = np.array([np.concatenate([np.concatenate((obs, np.eye(num_actions)[act])) for obs, act in traj]) for _, trajectories in clusters.items() for traj in trajectories])


    centroids = {}
    for cluster_id, elements in clusters.items():
        if mode == "mean":
            centroids[cluster_id] = compute_mean_centroid(encoded_action_trajectories)
        elif mode == "fuzzy_lcs":
            centroids[cluster_id] = compute_fuzzy_lcs_centroid(elements)
        else:
            raise ValueError(f"Unknown centroid mode: {mode}")
    return centroids


def compute_mean_centroid(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Compute the mean centroid from a list of vectors.
    """
    return np.mean(np.stack(vectors), axis=0)


def compute_fuzzy_lcs_centroid(sequences: List[List[int]]) -> List[int]:
    """
    Compute a fuzzy-LCS-based centroid from a list of action sequences (List[int]).
    Uses greedy voting per position (can be improved if needed).
    """
    if not sequences:
        return []

    min_len = min(len(seq) for seq in sequences)
    max_len = max(len(seq) for seq in sequences)

    # Simple approach: voting by position
    consensus = []
    for i in range(min_len):
        counter = Counter(seq[i] for seq in sequences if len(seq) > i)
        most_common = counter.most_common(1)[0][0]
        consensus.append(most_common)

    return consensus
