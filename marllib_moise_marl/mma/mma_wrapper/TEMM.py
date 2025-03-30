import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


class TEMM:

    def __init__(self, analysis_results_path: str = "./", similarity_threshold: float = 1):
        self.similarity_threshold = similarity_threshold
        self.analysis_results_path = analysis_results_path

    def extract_action_sequences(self, trajectories: Dict[str, List[List[Any]]]) -> Dict[str, List[Any]]:
        return {agent: [act for obs, act in trajectory] for agent, trajectory in trajectories.items()}

    def extract_observation_sequences(self, trajectories: Dict[str, List[List[Any]]]) -> Dict[str, List[Any]]:
        return {agent: [obs for obs, act in trajectory] for agent, trajectory in trajectories.items()}

    def joint_action_similarity(self, a1: Tuple[int, int], a2: Tuple[int, int]) -> float:
        if len(a1) != len(a2):
            return 0.
        direct_similarity = [value for value in a1 if value in a2]
        direct_similarity = len(direct_similarity) / len(a1)

        mirror_similarity = [value for value in a1 if value in a2[::-1]]
        mirror_similarity = len(mirror_similarity) / len(a1)

        return max(mirror_similarity, direct_similarity)

    def joint_action_distance(self, a1: Tuple[int, int], a2: Tuple[int, int]) -> float:
        return 1.0 - self.joint_action_similarity(a1, a2)

    def lcs_generalized(self, seq1: List[Tuple[int, int]], seq2: List[Tuple[int, int]]) -> int:
        n, m = len(seq1), len(seq2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n):
            for j in range(m):
                sim = self.joint_action_similarity(seq1[i], seq2[j])
                if sim >= self.similarity_threshold:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

        return dp[n][m]

    def cluster_joint_trajectories(self, trajectories: List[List[int]]):
        n = len(trajectories)

        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                lcs_len = self.lcs_generalized(
                    trajectories[i], trajectories[j])
                norm_lcs = lcs_len / \
                    max(len(trajectories[i]), len(trajectories[j]))
                dist = 1 - norm_lcs
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        # Condensed distance matrix for linkage
        triu_indices = np.triu_indices(n, 1)
        condensed_dist = dist_matrix[triu_indices]

        linkage_matrix = linkage(condensed_dist, method='average')

        # Plot dendrogram
        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix, labels=[f"{i}" for i in range(n)])
        plt.title("Joint Actions Trajectories Clustering")
        plt.xlabel("Trajectory")
        plt.ylabel("Distance")
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.analysis_results_path, "figures",
                    "joint_actions_trajectories_clustering.png"), dpi=300)

        return linkage_matrix

    def lcs_strict(self, seq1: List[int], seq2: List[int]) -> int:
        n, m = len(seq1), len(seq2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n):
            for j in range(m):
                if seq1[i] == seq2[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

        return dp[n][m]

    def cluster_individual_trajectories(self, trajectories: List[List[int]]):
        n = len(trajectories)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                lcs_len = self.lcs_strict(trajectories[i], trajectories[j])
                norm_lcs = lcs_len / \
                    max(len(trajectories[i]), len(trajectories[j]))
                dist = 1.0 - norm_lcs
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        condensed_dist = squareform(dist_matrix)
        linkage_matrix = linkage(condensed_dist, method='average')

        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix, labels=[f"{i}" for i in range(n)])
        plt.title("Individual Actions Trajectories Clustering")
        plt.xlabel("Trajectory")
        plt.ylabel("Distance")
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.analysis_results_path, "figures",
                    "individual_actions_trajectories_clustering.png"), dpi=300)

        return linkage_matrix

    def generate_figures(self):

        figures_path = os.path.join(self.analysis_results_path, "figures")
        if not os.path.exists(figures_path):
            os.mkdir(figures_path)

        action_trajectories = []
        observation_trajectories = []
        for file_name in os.listdir(os.path.join(self.analysis_results_path, "trajectories")):
            if "trajectories_" in file_name and file_name.endswith(".json"):
                trajectories = json.load(
                    open(os.path.join(self.analysis_results_path, "trajectories", file_name), "r"))
                action_trajectories += list(
                    self.extract_action_sequences(trajectories).values())
                observation_trajectories += list(
                    self.extract_observation_sequences(trajectories).values())

        self.cluster_individual_trajectories(action_trajectories)
        self.cluster_individual_observation_trajectories(observation_trajectories)

        joint_action_trajectories = []
        for file_name in os.listdir(os.path.join(self.analysis_results_path, "trajectories")):
            if "trajectories_" in file_name and file_name.endswith(".json"):
                trajectories = json.load(
                    open(os.path.join(self.analysis_results_path, "trajectories", file_name), "r"))
                action_trajectories = self.extract_action_sequences(
                    trajectories)
                joint_action_trajectory = []
                for i in range(len(action_trajectories[list(action_trajectories.keys())[0]])):
                    joint_action = []
                    for agent in list(action_trajectories.keys()):
                        joint_action += [action_trajectories[agent][i]]
                    joint_action_trajectory += [joint_action]
                joint_action_trajectories += [joint_action_trajectory]

        self.cluster_joint_trajectories(joint_action_trajectories)

    def cluster_individual_observation_trajectories(self, trajectories: List[List[np.ndarray]]):
        n = len(trajectories)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dists = [np.linalg.norm(np.array(o1) - np.array(o2)) for o1, o2 in zip(trajectories[i], trajectories[j])]
                avg_dist = np.mean(dists) if dists else 0
                dist_matrix[i, j] = dist_matrix[j, i] = avg_dist

        condensed_dist = squareform(dist_matrix)
        linkage_matrix = linkage(condensed_dist, method='average')

        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix, labels=[f"{i}" for i in range(n)])
        plt.title("Individual Observations Trajectories Clustering")
        plt.xlabel("Trajectory")
        plt.ylabel("Average Observation Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_results_path, "figures", "individual_observation_trajectories_clustering.png"), dpi=300)

        return linkage_matrix
