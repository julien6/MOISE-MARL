# mma_wrapper/temm/selection.py

import numpy as np
from typing import Dict, List, Any, Callable


def select_near_centroid_trajectories(
    clusters: Dict[int, List[Any]],
    centroids: Dict[int, Any],
    inclusion_radius: float,
    distance_fn: Callable[[Any, Any], float] = None
) -> Dict[int, List[Any]]:
    """
    Select trajectories close to the centroid of each cluster.

    Args:
        clusters (Dict[int, List[Any]]): Mapping from cluster id to list of trajectories.
        centroids (Dict[int, Any]): Mapping from cluster id to centroid.
        inclusion_radius (float): Threshold in [0,1] to include elements based on normalized distance.
        distance_fn (Callable): Function to compute distance between trajectory and centroid.
            If None, will use Euclidean distance.

    Returns:
        Dict[int, List[Any]]: Subset of trajectories close to centroid for each cluster.
    """
    selected = {}

    for cluster_id, trajectories in clusters.items():
        centroid = centroids[cluster_id]

        if not trajectories:
            selected[cluster_id] = []
            continue

        if distance_fn is None:
            # Default: Euclidean distance on vectors
            def distance_fn(x, y):
                return np.linalg.norm(x - y)

        distances = [distance_fn(traj, centroid) for traj in trajectories]
        # Avoid division by zero
        max_dist = max(distances) if distances else 1e-8

        selected_trajectories = [
            traj for traj, dist in zip(trajectories, distances)
            if dist <= inclusion_radius * max_dist
        ]
        selected[cluster_id] = selected_trajectories

    return selected
