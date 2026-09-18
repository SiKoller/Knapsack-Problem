import random

from rl_knapsack.config import (
    DEFAULT_ALPHA,
    DEFAULT_EPISODES,
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    RANDOM_SEED,
)
from rl_knapsack.items_loader import load_items
from rl_knapsack.knapsack_env import KnapsackEnv
from rl_knapsack.logger import make_logger
from rl_knapsack.presentation import display_result, print_banner, terminal_arg_parser
from rl_knapsack.q_learning_agent import QLearningAgent
from rl_knapsack.result_tracker import ResultTracker


# --- Main ---
# This ties everything together: environment, agent, and training loop.

def run_training(env, agent, tracker, logger):
    """Run training and display the results."""
    try:
        agent.train(episodes=DEFAULT_EPISODES)

        # === RESULTS ===
        logger.log_message(f"\n  Q-table entries: {agent.q_size()}")
        display_result(env, agent)
        logger.log_message(
            f"\nBest tracked value: {tracker.best_value:.1f} "
            f"(weight {tracker.best_weight})"
        )
    finally:
        logger.close()


def main():
    random.seed(RANDOM_SEED)

    # Parse the arguments from the terminal
    args = terminal_arg_parser()

    # === LOAD PROBLEM (capacity + items) FROM JSON ===
    capacity, items = load_items(args.items)

    # === CREATE ENVIRONMENT, TRACKER, AND AGENT ===
    env = KnapsackEnv(capacity, items)
    tracker = ResultTracker()
    logger = make_logger(args.log, csv_path=args.csv_path)
    agent = QLearningAgent(
        env,
        logger=logger,
        epsilon=DEFAULT_EPSILON,
        alpha=DEFAULT_ALPHA,
        gamma=DEFAULT_GAMMA,
        tracker=tracker,
    )

    print_banner(env, show_qtable_hint=args.log == "terminal")
    run_training(env, agent, tracker, logger)


if __name__ == "__main__":
    main()