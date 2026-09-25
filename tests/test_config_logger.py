import io
import logging
import unittest

from pydantic import ValidationError

from rl_knapsack.config import Config, Item, ProblemConfig
from rl_knapsack.logger import RESULT, TerminalLogger


class TestConfigAndLogger(unittest.TestCase):
    def test_config_validates_items_and_settings(self):
        config = ProblemConfig.model_validate({"capacity": 5, "items": [[2, 3]]})
        self.assertEqual(config.items, (Item(2.0, 3),))
        self.assertIs(type(config.capacity), float)
        self.assertIs(type(config.items[0].weight), float)
        self.assertIs(type(config.items[0].value), int)
        with self.assertRaises(ValidationError):
            ProblemConfig(capacity=0.0, items=[Item(2.0, 3)])
        with self.assertRaises(ValidationError):
            Config(epsilon=2)
        with self.assertRaises(ValidationError):
            ProblemConfig(capacity=5, items=[["2", 3]])

    def test_item_values_require_integers(self):
        for value in (3.0, 3.5, "3", True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ProblemConfig(capacity=5.0, items=[[2.5, value]])

    def test_weights_reject_strings_and_booleans(self):
        for weight in ("2.5", True):
            with self.subTest(weight=weight), self.assertRaises(ValidationError):
                ProblemConfig(capacity=5.0, items=[[weight, 3]])

    def test_terminal_levels_filter_episode_lines(self):
        stream = io.StringIO()
        logger = TerminalLogger(stream=stream)
        logger.log_episode(1, 3.0, 2.5, 5.0, 3)
        logger.log_message("start")
        logger.log_message("best", RESULT)
        logger.log_solution([(0, 2.5, 3)], 2.5, 3, 5.0, 1)
        output = stream.getvalue()
        self.assertNotIn("Episode", output)
        self.assertIn("[INFO] start", output)
        self.assertIn("[RESULT] best", output)
        self.assertIn("[RESULT] OPTIMAL SOLUTION", output)
        self.assertIn("Total value:  3\n", output)
        self.assertIn("Total weight: 2.5 / 5.0", output)

        stream = io.StringIO()
        logger = TerminalLogger(stream=stream, level="debug")
        logger.log_episode(1, 3.0, 2.5, 5.0, 3)
        self.assertIn("[DEBUG] Episode", stream.getvalue())

        stream = io.StringIO()
        logger = TerminalLogger(stream=stream, level="result")
        logger.log_message("start")
        logger.log_message("best", RESULT)
        self.assertEqual(stream.getvalue(), "[RESULT] best\n")

    def test_standard_levels_filter_the_whole_solution(self):
        stream = io.StringIO()
        logger = TerminalLogger(stream=stream, level="warning")
        logger.log_solution([(0, 2.5, 3)], 2.5, 3, 5.0, 1)
        logger.log_message("check", logging.WARNING)
        self.assertEqual(stream.getvalue(), "[WARNING] check\n")
        logger.close()
        self.assertFalse(stream.closed)


if __name__ == "__main__":
    unittest.main()
