import os
import sys
import unittest

# Ensure the project root is on sys.path so the `rl_knapsack` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_knapsack.result_tracker import ResultTracker

class TestResultTracker(unittest.TestCase):
    def test_initial_state(self):
        t = ResultTracker()
        self.assertEqual(t.best_items, [])
        self.assertIs(type(t.best_weight), float)
        self.assertEqual(t.best_weight, 0.0)
        self.assertIsNone(t.best_value)

    def test_update_improves(self):
        t = ResultTracker()
        t.update(10, [1,2], 5.5)
        self.assertEqual(t.best_value, 10)
        self.assertEqual(t.best_items, [1,2])
        self.assertEqual(t.best_weight, 5.5)
        self.assertIs(type(t.best_value), int)
        # updating with a worse value should not change
        t.update(9, [3], 1.0)
        self.assertEqual(t.best_value, 10)
        self.assertEqual(t.best_items, [1,2])

    def test_reset(self):
        t = ResultTracker()
        t.update(5, [0], 2.0)
        t.reset()
        self.assertEqual(t.best_items, [])
        self.assertEqual(t.best_weight, 0.0)
        self.assertIsNone(t.best_value)

if __name__ == '__main__':
    unittest.main()
