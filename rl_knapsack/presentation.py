import argparse

from .config import BANNER_WIDTH
from .items_loader import DEFAULT_ITEMS_PATH


# --- Presentation ---
# Terminal-facing concerns: showing the banner, rendering the final
# solution, and parsing CLI arguments (logger selection).

def display_result(env, agent):
    """Show the optimal solution via the agent's Logger strategy."""
    indices, total_w, total_v = env.get_solution(agent.get_q)
    selected = [(i, *env.items[i]) for i in indices]  # (index, weight, value)
    agent.logger.log_solution(
        items=selected,
        total_w=total_w,
        total_v=total_v,
        capacity=env.capacity,
        available=len(env.items),
    )


def terminal_arg_parser():
    # === LOGGER SELECTION (Strategy pattern) ===
    # --log terminal -> TerminalLogger (formatted text on stdout)
    # --log csv      -> CsvLogger      (rows in a CSV file)
    parser = argparse.ArgumentParser(description="RL knapsack solver")
    parser.add_argument(
        "--log",
        choices=["terminal", "csv"],
        default="terminal",
        help="Logger strategy to use: 'terminal' (default) or 'csv'",
    )
    parser.add_argument(
        "--csv-path",
        default="training_log.csv",
        help="Output file for the CSV logger (default: training_log.csv)",
    )
    parser.add_argument(
        "--items",
        default=DEFAULT_ITEMS_PATH,
        help=(
            "Path to a JSON file defining the knapsack problem "
            "({'capacity': int, 'items': [[weight, value], ...]}) "
            f"(default: {DEFAULT_ITEMS_PATH})"
        ),
    )

    return parser.parse_args()


def print_banner(env, show_qtable_hint=True):
    """Print the program banner to the terminal.

    The "Q-table entries after training:" line is a teaser for the results
    that follow on the terminal. When logging to CSV the results go into
    the file instead, so the hint is suppressed (pass show_qtable_hint=False).
    """
    print("=" * BANNER_WIDTH)
    print("  RL KNAPSACK SOLVER")
    print("=" * BANNER_WIDTH)
    print(f"  Capacity: {env.capacity}")
    print(f"  Items:    {env.items}")
    if show_qtable_hint:
        print(f"  Q-table entries after training:")