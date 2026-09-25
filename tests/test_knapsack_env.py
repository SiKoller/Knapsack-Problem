import os
import sys
import unittest

# Ensure the project root is on sys.path so the `rl_knapsack` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_knapsack.items_loader import load_items
from rl_knapsack.config import Item
from rl_knapsack.knapsack_env import KnapsackEnv


class TestKnapsackEnvLifecycle(unittest.TestCase):
    """Tests for the State-pattern lifecycle guards on KnapsackEnv."""

    def setUp(self):
        capacity, items = load_items()
        self.env = KnapsackEnv(capacity, items)

    def test_step_before_reset_raises(self):
        """A fresh env must refuse to step until reset() is called."""
        with self.assertRaises(RuntimeError):
            self.env.step((0.0, 0), 0)

    def test_episode_runs_to_completion(self):
        """Skipping all items completes the episode exactly after len(items) steps."""
        self.env.reset()
        state = (0.0, 0)
        steps = 0
        while not self.env.is_done():
            state, _ = self.env.step(state, 0)
            steps += 1
        self.assertEqual(steps, len(self.env.items))
        self.assertEqual(state, (0.0, len(self.env.items)))
        self.assertTrue(self.env.is_done())

    def test_step_after_finish_raises(self):
        """Stepping past the last item must raise RuntimeError."""
        self.env.reset()
        state = (0.0, 0)
        while not self.env.is_done():
            state, _ = self.env.step(state, 1)
        with self.assertRaises(RuntimeError):
            self.env.step(state, 1)

    def test_reset_restarts_mid_episode(self):
        """reset() mid-episode returns to the initial state."""
        self.env.reset()
        state, _ = self.env.step((0.0, 0), 1)  # take first item
        self.assertFalse(self.env.is_done())
        state = self.env.reset()
        self.assertFalse(self.env.is_done())
        self.assertEqual(state, (0.0, 0))

    def test_reset_restarts_after_finish(self):
        """reset() after the episode finished starts a new episode."""
        self.env.reset()
        state = (0.0, 0)
        while not self.env.is_done():
            state, _ = self.env.step(state, 0)
        self.assertTrue(self.env.is_done())
        self.env.reset()
        self.assertFalse(self.env.is_done())

    def test_reward_behavior_unaffected(self):
        """The reward strategy still behaves as before (fits/penalty/skip)."""
        self.env.reset()
        self.assertEqual(self.env.step((0.0, 0), 1), ((2.0, 1), 3.0))
        self.assertEqual(self.env.step((0.0, 1), 0), ((0.0, 2), 0.0))
        self.assertEqual(self.env.step((29.0, 3), 1), ((34.0, 4), -10.0))

    def test_solution_types_for_fractional_and_empty_selections(self):
        env = KnapsackEnv(3.75, [Item(1.25, 3), Item(2.5, 4)])
        self.assertIs(type(env.reset()[0]), float)
        for take, expected in ((True, ([0, 1], 3.75, 7)), (False, ([], 0.0, 0))):
            with self.subTest(take=take):
                solution = env.get_solution(lambda state, action: float(action == int(take)))
                self.assertEqual(solution, expected)
                self.assertIs(type(solution[1]), float)
                self.assertIs(type(solution[2]), int)


if __name__ == '__main__':
    unittest.main()
