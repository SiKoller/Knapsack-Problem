"""RL Knapsack solver package.

Modules:
- config:            Pydantic settings and the Item type
- logger:            Logger strategy interface + Terminal/Csv loggers
- result_tracker:    best-solution tracking
- reward_strategy:   RewardStrategy interface + KnapsackReward
- knapsack_env:      the 0/1 knapsack environment
- q_learning_agent:  the Q-learning agent (decorators + closures + class)
- items_loader:      load_items: reads the knapsack problem from a JSON file
- presentation:      banner, result display, and CLI argument parsing
"""

from .config import CONFIG, Config, Item
from .items_loader import load_items
from .knapsack_env import KnapsackEnv
from .logger import RESULT, CsvLogger, Logger, TerminalLogger, make_logger
from .presentation import display_result, print_banner, terminal_arg_parser
from .q_learning_agent import QLearningAgent
from .result_tracker import ResultTracker
from .reward_strategy import KnapsackReward, RewardStrategy

__all__ = [
    "CONFIG",
    "Config",
    "CsvLogger",
    "Item",
    "KnapsackEnv",
    "KnapsackReward",
    "Logger",
    "RESULT",
    "QLearningAgent",
    "ResultTracker",
    "RewardStrategy",
    "TerminalLogger",
    "display_result",
    "load_items",
    "make_logger",
    "print_banner",
    "terminal_arg_parser",
]
