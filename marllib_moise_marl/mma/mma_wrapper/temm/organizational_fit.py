# mma_wrapper/temm/organizational_fit.py

import numpy as np
from typing import Dict, List, Union


def compute_average_distance(trajectories: List, centroid: Union[np.ndarray, List]) -> float:
    """
    Compute average Euclidean distance between each trajectory and the cluster centroid.
    """
    if not trajectories:
        return 0.0
    total_distance = 0.0
    for traj in trajectories:
        traj_array = np.concatenate([np.array(x).flatten() for x in traj])
        centroid_array = np.concatenate(
            [np.array(x).flatten() for x in centroid])
        total_distance += np.linalg.norm(traj_array - centroid_array)
    return total_distance / len(trajectories)


def compute_sof(
    clusters: Dict[int, List],
    centroids: Dict[int, List]
) -> float:
    """
    Compute Structural Organizational Fit (SOF) from full trajectories.

    Lower average distance to centroid = higher structural fit.
    Normalized between 0 (worst) and 1 (perfect fit).
    """
    total_dist = 0.0
    total_count = 0

    for cluster_id, trajectories in clusters.items():
        centroid = centroids[cluster_id]
        cluster_dist = compute_average_distance(trajectories, centroid)
        total_dist += cluster_dist * len(trajectories)
        total_count += len(trajectories)

    avg_distance = total_dist / total_count if total_count else 0.0

    # Normalize assuming typical max distance ~10.0 (hyperparam or calibration possible)
    max_possible_dist = 10.0
    sof_score = max(0.0, 1.0 - avg_distance / max_possible_dist)

    return round(sof_score, 3)


def compute_fof(
    observation_clusters: Dict[int, List[List[np.ndarray]]],
    observation_centroids: Dict[int, List[np.ndarray]]
) -> float:
    """
    Compute Functional Organizational Fit (FOF) from observation trajectories.

    Based on how close observation trajectories are to the centroid of the cluster.
    """
    total_dist = 0.0
    total_count = 0

    for cluster_id, trajectories in observation_clusters.items():
        centroid_traj = observation_centroids[cluster_id]
        if not centroid_traj:
            continue

        for traj in trajectories:
            traj_array = np.concatenate(traj)
            centroid_array = np.concatenate(centroid_traj)
            total_dist += np.linalg.norm(traj_array - centroid_array)
            total_count += 1

    avg_distance = total_dist / total_count if total_count else 0.0
    max_possible_dist = 10.0
    fof_score = max(0.0, 1.0 - avg_distance / max_possible_dist)

    return round(fof_score, 3)


def compute_overall_fit(
    sof: float,
    fof: float,
    aggregation_method: str = "mean"
) -> float:
    """
    Compute Overall Organizational Fit (OF) from SOF and FOF.
    """
    if aggregation_method == "mean":
        return round((sof + fof) / 2, 3)
    elif aggregation_method == "min":
        return round(min(sof, fof), 3)
    elif aggregation_method == "harmonic":
        if sof + fof == 0:
            return 0.0
        return round((2 * sof * fof) / (sof + fof), 3)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
