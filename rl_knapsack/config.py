# --- Configuration Constants ---
# Central place for all shared constants so that environment, agent, and
# presentation code can import them without circular dependencies.

# Reproducibility
RANDOM_SEED = 42

# Penalty for taking an item that overfills the backpack
PENALTY_FOR_OVERFILL = -10.0

# Knapsack problem definition
KNAPSACK_CAPACITY = 30
ITEMS = [
    # (weight, value) for each item
    (2, 3), (3, 4), (4, 8), (5, 8), (7, 12),
    (8, 10), (9, 14), (10, 11), (12, 18), (15, 20),
]

# Q-learning hyperparameter defaults
DEFAULT_EPSILON = 0.2   # exploration rate
DEFAULT_ALPHA = 0.1     # learning rate
DEFAULT_GAMMA = 0.95    # discount factor
DEFAULT_EPISODES = 500   # number of training episodes

# Presentation
BANNER_WIDTH = 50