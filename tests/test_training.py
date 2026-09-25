import csv
import io
import tempfile
import unittest
from pathlib import Path

from rl_knapsack.config import Item
from rl_knapsack.knapsack_env import KnapsackEnv
from rl_knapsack.logger import CsvLogger, TerminalLogger
from rl_knapsack.q_learning_agent import QLearningAgent
from rl_knapsack.result_tracker import ResultTracker


class TestTraining(unittest.TestCase):
    def test_training_updates_and_logs_the_supplied_tracker(self):
        stream = io.StringIO()
        logger = TerminalLogger(stream=stream, level="debug")
        self.addCleanup(logger.close)
        tracker = ResultTracker()
        agent = QLearningAgent(KnapsackEnv(2.5, [Item(2.5, 3)]), tracker, logger=logger)
        agent.explore = lambda state, actions, get_q: 1
        agent.train(1)
        self.assertEqual((tracker.best_items, tracker.best_weight, tracker.best_value), ([0], 2.5, 3))
        self.assertIs(type(tracker.best_weight), float)
        self.assertIs(type(tracker.best_value), int)
        self.assertIn("Best: 3\n", stream.getvalue())

    def test_csv_rows_keep_empty_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            logger = CsvLogger(path)
            try:
                logger.log_message("start")
                logger.log_episode(1, 3.0, 2.5, 2.5, 3)
                logger.log_solution([(0, 2.5, 3)], 2.5, 3, 2.5, 1)
            finally:
                logger.close()
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["event"] for row in rows], ["message", "episode", "solution"])
            self.assertEqual(rows[0]["message"], "start")
            self.assertEqual(rows[0]["episode"], "")
            self.assertEqual(rows[1]["best_value"], "3")
            self.assertEqual(rows[2]["total_value"], "3")
            self.assertEqual(rows[2]["total_weight"], "2.5")
            self.assertEqual(rows[2]["capacity"], "2.5")
            self.assertEqual(rows[2]["message"], "")


if __name__ == "__main__":
    unittest.main()
