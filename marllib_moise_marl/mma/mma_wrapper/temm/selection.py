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

    num_actions = max(np.concatenate(
        [[a for _, a in traj] for _, trajectories in clusters.items() for traj in trajectories])) + 1

    for cluster_id, trajectories in clusters.items():
        trajectories = np.array(trajectories)

        encoded_action_trajectories = np.array([np.concatenate([np.concatenate(
            (obs, np.eye(num_actions)[act])) for obs, act in traj]) for traj in trajectories])
        print("---->non enc>>>> ", trajectories)
        print("---->enc>>>> ", encoded_action_trajectories)
        centroid = centroids[cluster_id]
        centroid = np.array(centroid)
        print("---->cent>>> ", centroid)


        if distance_fn is None:
            # Default: Euclidean distance on vectors
            def distance_fn(x, y):
                return np.linalg.norm(x - y)
        distances = np.array([distance_fn(traj, centroid)
                             for traj in encoded_action_trajectories])
        # Avoid division by zero
        max_dist = max(distances) if not np.all(distances == 0.) else 1e-8
        selected_trajectories = [
            traj for traj, dist in zip(trajectories, distances)
            if np.all(dist <= inclusion_radius * max_dist)
        ]
        selected[cluster_id] = selected_trajectories

    return selected
