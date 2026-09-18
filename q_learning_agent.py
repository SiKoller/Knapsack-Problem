import random
import time
from functools import wraps

from config import DEFAULT_ALPHA, DEFAULT_EPSILON, DEFAULT_GAMMA
from logger import TerminalLogger


# --- Decorator Pattern ---

def log_episode(func):
    """Decorator that logs each training episode via the agent's Logger."""
    @wraps(func)
    def wrapper(self, episode):
        result = func(self, episode)
        value, weight = result
        # Best value known so far: prefer the ResultTracker if present,
        # otherwise fall back to the agent's own best_value attribute.
        tracker = getattr(self, "tracker", None)
        if tracker is not None and tracker.best_value != float("-inf"):
            best = tracker.best_value
        else:
            best = getattr(self, "best_value", float("-inf"))
        # The agent's logger is an injected Strategy service (terminal or
        # CSV): the decorator only depends on the Logger interface.
        self.logger.log_episode(
            episode, value, weight,
            capacity=self.env.capacity,
            best=best,
        )
        return result
    return wrapper


def time_execution(func):
    """Decorator that measures execution time (logged via self.logger)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # Method decorator: the first argument is the instance (the agent).
        args[0].logger.log_message(f"\nTraining finished in {elapsed:.3f}s")
        return result
    return wrapper


# --- Closure ---
# Closures encapsulate state inside a function's scope, keeping it private
# and returning only the functions that manipulate it.

def make_q_table():
    """Closure that encapsulates a Q-table with get/set operations.

    The Q-table is the agent's "brain" — it stores a value for every
    (state, action) pair, representing how good that action is in that state.
    Q-values start at 0.0 and are updated through training via the
    Bellman equation (see QLearningAgent.train_episode).

    The table dict is private to this closure — no outside code can
    access it directly, only through get_q / set_q / size.
    """
    table = {}  # PRIVATE: {(state, action) -> Q-value}

    def get_q(state, action):
        """Return Q-value for a (state, action) pair. Defaults to 0.0."""
        return table.get((state, action), 0.0)

    def set_q(state, action, value):
        """Update Q-value for a (state, action) pair."""
        table[(state, action)] = value

    def size():
        """Return number of entries in the Q-table."""
        return len(table)

    return get_q, set_q, size


def make_epsilon_greedy(epsilon):
    """Closure that encapsulates an epsilon-greedy action selector.

    Epsilon-greedy balances EXPLOITATION (pick the best known action)
    vs EXPLORATION (try a random action to discover new strategies).

    With probability epsilon, a random action is chosen (exploration).
    Otherwise, the action with the highest Q-value is chosen (exploitation).
    """
    def choose(state, actions, get_q):
        if random.random() < epsilon:
            return random.choice(actions)          # EXPLORE: random action
        q_values = [(get_q(state, a), a) for a in actions]
        return max(q_values, key=lambda x: (x[0], random.random()))[1]  # EXPLOIT: best Q
    return choose


# --- Agent ---
# The agent is the "learner" — it interacts with the environment,
# takes actions, receives rewards, and updates its Q-table.

class QLearningAgent:
    """Q-Learning agent that learns optimal knapsack packing.

    === Q-LEARNING OVERVIEW ===
    Q-learning is a model-free RL algorithm. The agent learns a
    Q-function Q(s, a) = "how good is action a in state s?"

    The agent updates Q-values using the BELLMAN EQUATION:
        Q(s,a) = Q(s,a) + α * [reward + γ * max Q(s',a') - Q(s,a)]
                 ^^^^^^^   ^     ^^^^^   ^     ^^^^^^^^^^^   ^^^^^^^
                 old Q    lr    reward  disc   best future    old Q
                                        factor   Q-value

    - α (alpha, learning rate): how much new info overrides old (0.1)
    - γ (gamma, discount factor): how much future rewards matter (0.95)
    - The term [reward + γ * max Q(s',a') - Q(s,a)] is the "TD error"
      (temporal difference error) — the gap between predicted and actual value.
    """
    def __init__(self, env, logger=None, epsilon=DEFAULT_EPSILON, alpha=DEFAULT_ALPHA, gamma=DEFAULT_GAMMA, tracker=None):
        self.env = env
        # Logger strategy (injected service): TerminalLogger by default,
        # or a CsvLogger for file output — swap by passing a different one.
        self.logger = logger if logger is not None else TerminalLogger()
        # Closure: Q-table encapsulates the learning memory
        self.get_q, self.set_q, self.q_size = make_q_table()
        # Closure: epsilon-greedy policy encapsulates exploration strategy
        self.explore = make_epsilon_greedy(epsilon)
        self.alpha = alpha    # Learning rate (α): how fast to learn
        self.gamma = gamma    # Discount factor (γ): value of future rewards
        # Result tracking is delegated to an optional ResultTracker
        self.tracker = tracker
        if tracker is None:
            # Fallback bookkeeping when no ResultTracker is supplied
            self.best_value = float("-inf")
            self.best_items = []

    @log_episode
    def train_episode(self, episode):
        """Run one complete episode (one pass through all items).

        Each episode:
        1. Reset to empty backpack
        2. For each item: observe state -> choose action -> get reward -> update Q
        3. Track best solution found so far
        """
        state = self.env.reset()
        total_reward = 0.0

        while not self.env.is_done(state):
            # EXPLORE: agent selects action using epsilon-greedy policy
            action = self.explore(state, [0, 1], self.get_q)

            # ENVIRONMENT: execute action, observe next_state and reward
            next_state, reward = self.env.step(state, action)

            # === Q-LEARNING UPDATE (Bellman Equation) ===
            # This is the core of the learning algorithm:
            best_next = max(self.get_q(next_state, a) for a in [0, 1])
            old_q = self.get_q(state, action)
            new_q = old_q + self.alpha * (reward + self.gamma * best_next - old_q)
            self.set_q(state, action, new_q)
            # After many episodes, Q-values converge to the true value
            # of each (state, action) pair.

            total_reward += reward
            state = next_state

        # After episode: check solution quality and notify tracker if any
        items, weight, value = self.env.get_solution(self.get_q)
        if self.tracker is not None:
            self.tracker.update(value, items, weight)
        elif value > self.best_value:
            self.best_value = value
            self.best_items = items

        return total_reward, weight

    @time_execution
    def train(self, episodes):
        """Run multiple training episodes.

        Each episode the agent gets slightly smarter as Q-values
        converge toward optimal values through repeated updates.
        """
        self.logger.log_message("\n--- Training Start ---\n")
        for ep in range(1, episodes + 1):
            self.train_episode(ep)
        self.logger.log_message("\n--- Training End ---")