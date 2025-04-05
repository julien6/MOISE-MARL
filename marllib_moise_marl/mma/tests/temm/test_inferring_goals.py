# test_inferring_goals.py

import unittest
import numpy as np
from mma_wrapper.temm.inferring_goals import extract_goals_from_trajectories, summarize_goals


class TestInferringGoals(unittest.TestCase):

    def setUp(self):
        # Mock data: clusters of observation trajectories
        # Each trajectory is a list of observation vectors
        self.mock_clusters = {
            0: [
                [np.array([1.0, 0.0]), np.array([0.5, 0.5])],
                [np.array([1.0, 0.1]), np.array([0.6, 0.4])],
            ],
            1: [
                [np.array([0.0, 1.0]), np.array([0.1, 0.9])],
                [np.array([0.05, 0.95]), np.array([0.0, 1.0])],
            ]
        }

    def test_extract_goals_from_trajectories(self):
        goals = extract_goals_from_trajectories(self.mock_clusters)
        self.assertIsInstance(goals, dict)
        self.assertIn(0, goals)
        self.assertIsInstance(goals[0], list)
        self.assertGreater(len(goals[0]), 0)
        for obs in goals[0]:
            self.assertIn("observation", obs)
            self.assertIn("weight", obs)

    def test_summarize_goals(self):
        goals = extract_goals_from_trajectories(self.mock_clusters)
        summarized = summarize_goals(goals, min_goal_weight=0.05)
        self.assertIsInstance(summarized, dict)
        for cluster_id, goal_list in summarized.items():
            for g in goal_list:
                self.assertGreaterEqual(g["weight"], 0.05)
                self.assertIn("observation", g)


if __name__ == "__main__":
    unittest.main()
