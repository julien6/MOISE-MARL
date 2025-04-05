import unittest
import os
import numpy as np
import shutil

from mma_wrapper.temm.visualization import (
    generate_dendrogram,
    generate_actions_dendrogram,
    generate_observations_dendrogram,
    visualize_action_pca,
    visualize_observation_pca,
    visualize_transition_pca,
    visualize_action_trajectory,
    visualize_observation_trajectory,
    visualize_trajectory,
)


class TestVisualization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = "test_figures"
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir)

    def test_generate_dendrogram(self):
        clusters = {0: [[0, 1, 2]], 1: [[2, 1, 0]]}
        generate_dendrogram(clusters, save_path=os.path.join(
            self.output_dir, "full_dendrogram.png"))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_dir, "full_dendrogram.png")))

    def test_generate_actions_dendrogram(self):
        clusters = {0: [[0, 1, 2]], 1: [[2, 1, 0]]}
        generate_actions_dendrogram(clusters, save_path=os.path.join(
            self.output_dir, "actions_dendrogram.png"))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_dir, "actions_dendrogram.png")))

    def test_generate_observations_dendrogram(self):
        clusters = {0: [[np.array([1, 0]), np.array([0, 1])]], 1: [
            [np.array([0, 1]), np.array([1, 0])]]}
        generate_observations_dendrogram(
            clusters, save_path=os.path.join(self.output_dir, "obs_dendrogram.png"))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_dir, "obs_dendrogram.png")))

    def test_visualize_action_pca(self):
        trajectories = [
            [0, 1, 2],
            [2, 1, 0],
        ]
        visualize_action_pca(trajectories, save_path=os.path.join(
            self.output_dir, "action_pca.png"))
        self.assertTrue(os.path.exists(
            os.path.join(self.output_dir, "action_pca.png")))

    def test_visualize_observation_pca(self):
        trajectories = [
            [np.array([1, 0, 1]), np.array([0, 1, 0])],
            [np.array([0, 1, 1]), np.array([1, 0, 0])]
        ]
        visualize_observation_pca(
            trajectories, save_path=os.path.join(self.output_dir, "obs_pca.png"))
        self.assertTrue(os.path.exists(
            os.path.join(self.output_dir, "obs_pca.png")))

    def test_visualize_transition_pca(self):
        full_trajectories = [
            [(np.array([1, 0]), 0), (np.array([0, 1]), 1)],
            [(np.array([0, 1]), 2), (np.array([1, 0]), 1)]
        ]
        visualize_transition_pca(full_trajectories, save_path=os.path.join(
            self.output_dir, "transition_pca.png"))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_dir, "transition_pca.png")))

    def test_visualize_action_trajectory(self):
        trajectories = [
            [0, 1, 2],
            [2, 1, 0],
        ]
        visualize_action_trajectory(trajectories, save_path=os.path.join(
            self.output_dir, "action_trajectory.png"))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_dir, "action_trajectory.png")))

    def test_visualize_observation_trajectory(self):
        trajectories = [
            [np.array([1, 0]), np.array([0, 1])],
            [np.array([0, 1]), np.array([1, 0])]
        ]
        visualize_observation_trajectory(
            trajectories, save_path=os.path.join(self.output_dir, "obs_trajectory.png"))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_dir, "obs_trajectory.png")))

    def test_visualize_trajectory(self):
        full_trajectories = [
            [(np.array([1, 0]), 0), (np.array([0, 1]), 1)],
            [(np.array([0, 1]), 2), (np.array([1, 0]), 1)]
        ]
        visualize_trajectory(full_trajectories, save_path=os.path.join(
            self.output_dir, "full_trajectory.png"))
        self.assertTrue(os.path.exists(os.path.join(
            self.output_dir, "full_trajectory.png")))


if __name__ == '__main__':
    unittest.main()
