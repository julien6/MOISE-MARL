import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import numpy as np

from typing import Any, Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap
from scipy.cluster.hierarchy import fcluster
from itertools import combinations

class TEMM:

    def __init__(self, analysis_results_path: str = "./", similarity_threshold: float = 1, role_homogeneity_threshold=0.1):
        self.similarity_threshold = similarity_threshold
        self.analysis_results_path = analysis_results_path
        self.role_homogeneity_threshold = role_homogeneity_threshold

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

    def lcs_strict(self, seq1: List[int], seq2: List[int]) -> Tuple[List[int], int]:
        n, m = len(seq1), len(seq2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        direction = [[None] * (m + 1) for _ in range(n + 1)]

        for i in range(n):
            for j in range(m):
                if seq1[i] == seq2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                    direction[i + 1][j + 1] = 'diag'
                elif dp[i][j + 1] >= dp[i + 1][j]:
                    dp[i + 1][j + 1] = dp[i][j + 1]
                    direction[i + 1][j + 1] = 'up'
                else:
                    dp[i + 1][j + 1] = dp[i + 1][j]
                    direction[i + 1][j + 1] = 'left'

        # Backtrack to find the actual LCS
        lcs = []
        i, j = n, m
        while i > 0 and j > 0:
            if direction[i][j] == 'diag':
                lcs.append(seq1[i - 1])
                i -= 1
                j -= 1
            elif direction[i][j] == 'up':
                i -= 1
            else:
                j -= 1

        lcs.reverse()
        return lcs, len(lcs)

    def visualize_observation_trajectories_pca(self, observation_trajectories: List[List[np.ndarray]]):

        all_points = np.concatenate(observation_trajectories)
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(all_points)

        # more diverse colors
        cmap = get_cmap('tab20', len(observation_trajectories))
        index = 0
        plt.figure(figsize=(10, 8))
        for i, traj in enumerate(observation_trajectories):
            length = len(traj)
            traj_2d = points_2d[index:index + length]
            index += length
            color = cmap(i)
            plt.plot(traj_2d[:, 0] + random.random() / 5, traj_2d[:, 1] + random.random() / 5, marker='o', linestyle='-', linewidth=1,
                     markersize=4, alpha=0.8, label=f"Agent {i}", color=color)
            # Annotate points with step index
            for step, (x, y) in enumerate(traj_2d):
                plt.text(x, y, str(step), fontsize=6, color=color, alpha=0.8)

        plt.title("PCA Projection for Observations Trajectories")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_results_path,
                    "figures", "observation_trajectories_pca.png"), dpi=300)
        plt.close()

    def pca_on_trajectory_representations(self, observation_trajectories: List[List[np.ndarray]]):

        traj_lengths = [len(traj) for traj in observation_trajectories]
        if len(set(traj_lengths)) != 1:
            return

        # Concatenation of observations into a single vector per trajectory
        representations = np.array(
            [np.concatenate(traj).flatten() for traj in observation_trajectories])

        # PCA
        pca = PCA(n_components=2)
        traj_points_2d = pca.fit_transform(representations)

        # Display
        plt.figure(figsize=(8, 6))
        for i, point in enumerate(traj_points_2d):
            plt.scatter(point[0], point[1], label=f'Trajectory {i}', alpha=0.8)
            plt.text(point[0], point[1], str(i), fontsize=8)

        plt.title("PCA Projection of Full Trajectories")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_results_path,
                    "figures", "trajectory_level_pca.png"), dpi=300)
        plt.close()

    def cluster_individual_trajectories(self, trajectories: List[List[int]]):
        n = len(trajectories)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                lcs, lcs_len = self.lcs_strict(
                    trajectories[i], trajectories[j])
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

    def is_fuzzy_subsequence(self, sub, seq):
        it = iter(seq)
        return all(item in it for item in sub)

    def generate_candidates(self, seq, min_len):
        n = len(seq)
        for l in range(min_len, n + 1):
            for indices in combinations(range(n), l):
                yield tuple(seq[i] for i in indices)

    def find_best_subsequence(self, sequences, min_len=3):
        candidates = set(self.generate_candidates(sequences[0], min_len))
        best = ()
        best_count = -1
        best_len = -1
        best_diversity = -1

        for cand in candidates:
            count = sum(self.is_fuzzy_subsequence(cand, seq) for seq in sequences)
            diversity = len(set(cand))
            if (
                count > best_count
                or (count == best_count and len(cand) > best_len)
                or (count == best_count and len(cand) == best_len and diversity > best_diversity)
            ):
                best = cand
                best_count = count
                best_len = len(cand)
                best_diversity = diversity

        return list(best), best_count

    def compute_fuzzy_lcs_clustering(self, sequences):

        n = len(sequences)

        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                subseq, _ = self.find_best_subsequence([sequences[i], sequences[j]])
                sim_matrix[i, j] = len(subseq) / max(len(sequences[i]), len(sequences[j]))

        dist_matrix = 1 - sim_matrix
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='average')

        cluster_seqs = {i: sequences[i] for i in range(n)}
        cluster_labels = {i: str(sequences[i]) for i in range(n)}  # Pour afficher

        for cluster_id, (i, j, _, _) in enumerate(linkage_matrix, start=n):
            left_seq = cluster_seqs[int(i)]
            right_seq = cluster_seqs[int(j)]
            merged_seq, _ = self.find_best_subsequence([left_seq, right_seq])
            cluster_seqs[cluster_id] = merged_seq
            cluster_labels[cluster_id] = f"{merged_seq}"

        # === Fonction de labeling personnalisé ===

        def fancy_label_func(id):
            return f"{id}={str(cluster_labels[id])}"

        # === Affichage du dendrogramme avec labels custom ===

        plt.figure(figsize=(14, 7))
        dendrogram(
            linkage_matrix,
            labels=[fancy_label_func(i) for i in range(n)],
            leaf_font_size=10,
        )

        # Afficher séquences inférées sur les branches internes
        def annotate_dendrogram(linkage_matrix, cluster_labels):
            n = len(sequences)
            for k, (i, j, _, _) in enumerate(linkage_matrix):
                node_id = n + k
                x = 20 + (i * 5)
                y = linkage_matrix[k, 2]
                label = cluster_labels[node_id]
                plt.text(x, y + 0.01, f"{node_id}={label}", fontsize=9, ha='center', va='bottom', rotation=0)

        annotate_dendrogram(linkage_matrix, cluster_labels)

        plt.title("Dendrogramme hiérarchique avec séquences inférées")
        plt.xlabel("Séquences (représentation brute)")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def generate_figures(self):

        figures_path = os.path.join(self.analysis_results_path, "figures")
        if not os.path.exists(figures_path):
            os.mkdir(figures_path)

        full_trajectories = []
        action_trajectories = []
        observation_trajectories = []
        for file_name in os.listdir(os.path.join(self.analysis_results_path, "trajectories")):
            if "trajectories_" in file_name and file_name.endswith(".json"):
                trajectories = json.load(
                    open(os.path.join(self.analysis_results_path, "trajectories", file_name), "r"))
                full_trajectories += [[(obs, act) for obs, act in trajectory]
                                      for agent, trajectory in trajectories.items()]
                action_trajectories += list(
                    self.extract_action_sequences(trajectories).values())
                observation_trajectories += list(
                    self.extract_observation_sequences(trajectories).values())

        self.compute_fuzzy_lcs_clustering(action_trajectories)

        # self.infer_roles_from_clusters(full_trajectories)

        # self.cluster_individual_trajectories(action_trajectories)
        # self.cluster_individual_observation_trajectories(
        #     observation_trajectories)
        # self.visualize_observation_trajectories_pca(observation_trajectories)
        # self.pca_on_trajectory_representations(observation_trajectories)

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

        # self.cluster_joint_trajectories(joint_action_trajectories)

    def cluster_individual_observation_trajectories(self, trajectories: List[List[np.ndarray]]):
        n = len(trajectories)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dists = [np.linalg.norm(np.array(o1) - np.array(o2))
                         for o1, o2 in zip(trajectories[i], trajectories[j])]
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
        plt.savefig(os.path.join(self.analysis_results_path, "figures",
                    "individual_observation_trajectories_clustering.png"), dpi=300)

        return linkage_matrix

    def infer_roles_from_clusters(self, trajectories: List[List[Tuple[np.ndarray, int]]]) -> Dict[int, Dict[str, List[Tuple[np.ndarray, int]]]]:
        """
        Infers abstract roles from clustered trajectories using the CCRA approach.
        Each trajectory is a list of (observation, action) tuples.
        """

        # Step 1: Compute pairwise distances using LCS-based similarity
        n = len(trajectories)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                lcs, lcs_len = self.lcs_strict(
                    [a for o, a in trajectories[i]],
                    [a for o, a in trajectories[j]]
                )
                norm_lcs = lcs_len / \
                    max(len(trajectories[i]), len(trajectories[j]))
                dist = 1.0 - norm_lcs
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        condensed_dist = squareform(dist_matrix)
        linkage_matrix = linkage(condensed_dist, method='average')

        # Step 2: Threshold-based flat clustering
        cluster_labels = fcluster(
            linkage_matrix, t=self.role_homogeneity_threshold, criterion='distance')

        # Step 3: Group trajectories by cluster
        clusters: Dict[int, List[List[Tuple[np.ndarray, int]]]] = {}
        for idx, label in enumerate(cluster_labels):
            clusters.setdefault(label, []).append(trajectories[idx])

        # Step 4: Compute centroids and extract abstract rules
        inferred_roles = {}
        for cluster_id, cluster_trajectories in clusters.items():
            # Choose the trajectory with minimum average distance to others (medoid)
            min_sum_dist = float('inf')
            centroid = cluster_trajectories[0]
            for t1 in cluster_trajectories:
                total_dist = 0.0
                for t2 in cluster_trajectories:
                    lcs, lcs_len = self.lcs_strict(
                        [a for o, a in t1],
                        [a for o, a in t2]
                    )
                    norm_lcs = lcs_len / max(len(t1), len(t2))
                    total_dist += 1.0 - norm_lcs
                if total_dist < min_sum_dist:
                    min_sum_dist = total_dist
                    centroid = t1

            # Step 5: Find close trajectories to centroid
            close_trajectories = []
            for traj in cluster_trajectories:
                lcs, lcs_len = self.lcs_strict(
                    [a for o, a in traj],
                    [a for o, a in centroid]
                )
                norm_lcs = lcs_len / max(len(traj), len(centroid))
                if 1.0 - norm_lcs <= self.role_homogeneity_threshold:
                    close_trajectories.append(traj)

            # Step 6: Extract observation → action rules from close trajectories
            rules = []
            for traj in close_trajectories:
                for obs, act in traj:
                    rules.append((obs, act))  # Can be generalized later

            inferred_roles[cluster_id] = {
                "centroid": centroid,
                "rules": rules,
                "support": len(close_trajectories),
            }

        json.dump(self.convert_roles_to_json(inferred_roles), open(os.path.join(self.analysis_results_path, "inferred_organizational_specifications",
                                                                                "inferred_roles.json"), "w+"))

        return inferred_roles

    def convert_roles_to_json(self, inferred_roles: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Converts inferred_roles to a JSON-serializable dictionary.
        """
        roles_json = {}
        for cluster_id, data in inferred_roles.items():
            centroid_serialized = [(obs, int(act))
                                   for obs, act in data["centroid"]]
            rules_serialized = [(obs, int(act))
                                for obs, act in data["rules"]]
            roles_json[str(cluster_id)] = {
                "centroid": centroid_serialized,
                "rules": rules_serialized,
                "support": data["support"]
            }
        return roles_json


if __name__ == '__main__':

    t = TEMM(
        "/home/soulej/Documents/MOISE-MARL/marllib_moise_marl/test_scenarios/analysis_results")
    t.generate_figures()

    # print(t.lcs_strict([0, 0, 0, 0, 1, 2, 3, 0], [1, 2, 3, 0, 0, 0, 0, 0]))

    # sequences = [
    #     [0, 0, 1, 0, 2, 3, 4, 0],
    #     [0, 0, 1, 2, 0, 3, 4, 0],
    #     [0, 1, 2, 3, 0, 4, 0, 0],
    #     [0, 1, 0, 2, 0, 3, 4, 0],
    #     [0, 0, 1, 2, 0, 3, 0, 4],
    #     [7, 8, 9, 1, 0, 0, 1, 2],
    #     [7, 8, 9, 1, 0, 1, 2, 0],
    #     [9, 8, 9, 1, 0, 2, 2, 0],
    #     [0, 2, 0, 1, 0, 0, 0, 0],
    # ]

    # print(find_best_subsequence([[0, 1, 0, 2, 0, 3, 4, 0], [0, 0, 1, 0, 2, 3, 4, 0], [0, 0, 1, 2, 0, 3, 4, 0]]))

    # print(find_best_subsequence(sequences))
