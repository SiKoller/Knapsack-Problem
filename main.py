import random

from rl_knapsack.config import CONFIG
from rl_knapsack.items_loader import load_items
from rl_knapsack.knapsack_env import KnapsackEnv
from rl_knapsack.logger import RESULT, make_logger
from rl_knapsack.presentation import display_result, print_banner, terminal_arg_parser
from rl_knapsack.q_learning_agent import QLearningAgent
from rl_knapsack.result_tracker import ResultTracker


# --- Main ---
# This ties everything together: environment, agent, and training loop.

def run_training(env, agent, tracker, logger):
    """Run training and display the results."""
    try:
        agent.train(episodes=CONFIG.episodes)

        # === RESULTS ===
        logger.log_message(f"Q-table entries: {agent.q_size()}", RESULT)
        display_result(env, agent)
        logger.log_message(
            f"\nBest tracked value: {tracker.best_value} "
            f"(weight {tracker.best_weight})",
            RESULT,
        )
    finally:
        logger.close()


def main():
    random.seed(CONFIG.random_seed)

    # Parse the arguments from the terminal
    args = terminal_arg_parser()

    # === LOAD PROBLEM (capacity + items) FROM JSON ===
    capacity, items = load_items(args.items)

    # === CREATE ENVIRONMENT, TRACKER, AND AGENT ===
    env = KnapsackEnv(capacity, items)
    tracker = ResultTracker()
    logger = make_logger(args.log, csv_path=args.csv_path, level=args.log_level)
    agent = QLearningAgent(
        env,
        logger=logger,
        epsilon=CONFIG.epsilon,
        alpha=CONFIG.alpha,
        gamma=CONFIG.gamma,
        tracker=tracker,
    )

    print_banner(env, show_qtable_hint=args.log == "terminal")
    run_training(env, agent, tracker, logger)


if __name__ == "__main__":
    main()
