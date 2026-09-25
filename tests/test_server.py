import unittest

from server import solve


class TestWebSolver(unittest.TestCase):
    def test_training_uses_selected_problem(self):
        result = solve({"capacity": 5, "episodes": 20, "epsilon": 0.2,
                        "items": [[2, 3], [3, 4]]})
        self.assertEqual(len(result["history"]), 20)
        self.assertEqual(result["optimum"], {"value": 7, "weight": 5, "indices": [0, 1]})
        self.assertLessEqual(result["learned"]["weight"], 5)

    def test_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "Choose 1 to 16 items"):
            solve({"capacity": 5, "episodes": 20, "epsilon": 0.2, "items": []})


if __name__ == "__main__":
    unittest.main()
