import os
import sys
import unittest

# Ensure the project root is on sys.path so the `rl_knapsack` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_knapsack.config import ITEMS, KNAPSACK_CAPACITY
from rl_knapsack.knapsack_env import KnapsackEnv


class TestKnapsackEnvLifecycle(unittest.TestCase):
    """Tests for the State-pattern lifecycle guards on KnapsackEnv."""

    def setUp(self):
        self.env = KnapsackEnv(KNAPSACK_CAPACITY, ITEMS)

    def test_step_before_reset_raises(self):
        """A fresh env must refuse to step until reset() is called."""
        with self.assertRaises(RuntimeError):
            self.env.step((0, 0), 0)

    def test_episode_runs_to_completion(self):
        """Skipping all items completes the episode exactly after len(items) steps."""
        self.env.reset()
        state = (0, 0)
        steps = 0
        while not self.env.is_done():
            state, _ = self.env.step(state, 0)
            steps += 1
        self.assertEqual(steps, len(ITEMS))
        self.assertEqual(state, (0, len(ITEMS)))
        self.assertTrue(self.env.is_done())

    def test_step_after_finish_raises(self):
        """Stepping past the last item must raise RuntimeError."""
        self.env.reset()
        state = (0, 0)
        while not self.env.is_done():
            state, _ = self.env.step(state, 1)
        with self.assertRaises(RuntimeError):
            self.env.step(state, 1)

    def test_reset_restarts_mid_episode(self):
        """reset() mid-episode returns to the initial state."""
        self.env.reset()
        state, _ = self.env.step((0, 0), 1)  # take first item
        self.assertFalse(self.env.is_done())
        state = self.env.reset()
        self.assertFalse(self.env.is_done())
        self.assertEqual(state, (0, 0))

    def test_reset_restarts_after_finish(self):
        """reset() after the episode finished starts a new episode."""
        self.env.reset()
        state = (0, 0)
        while not self.env.is_done():
            state, _ = self.env.step(state, 0)
        self.assertTrue(self.env.is_done())
        self.env.reset()
        self.assertFalse(self.env.is_done())

    def test_reward_behavior_unaffected(self):
        """The reward strategy still behaves as before (fits/penalty/skip)."""
        self.env.reset()
        self.assertEqual(self.env.step((0, 0), 1), ((2, 1), 3))       # take item 0 -> fits
        self.assertEqual(self.env.step((0, 1), 0), ((0, 2), 0.0))     # skip item 1
        self.assertEqual(self.env.step((29, 3), 1), ((34, 4), -10.0)) # item 3 (w=5) overfills


if __name__ == '__main__':
    unittest.main()