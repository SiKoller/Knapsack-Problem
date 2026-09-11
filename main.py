import random
import time
from functools import wraps


# --- Decorator Pattern ---

def log_episode(func):
    """Decorator that logs each training episode."""
    @wraps(func)
    def wrapper(self, episode):
        result = func(self, episode)
        value, weight = result
        best = self.best_value
        print(
            f"  Episode {episode:5d} | "
            f"Reward: {value:7.1f} | "
            f"Weight: {weight}/{self.env.capacity} | "
            f"Best: {best:.1f}"
        )
        return result
    return wrapper


def time_execution(func):
    """Decorator that measures execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"\nTraining finished in {elapsed:.3f}s")
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


# --- Environment ---
# The environment defines the "world" the agent interacts with.
# It defines the STATE SPACE, ACTION SPACE, and REWARD FUNCTION —
# the three core components of any RL problem.

class KnapsackEnv:
    """
    0/1 Knapsack environment.

    === STATE SPACE ===
    State: tuple (current_weight, next_item_index)
    - current_weight: how much weight is currently in the backpack
    - next_item_index: which item we are deciding on next
    The state space has capacity * len(items) possible states.

    === ACTION SPACE ===
    Actions: 0 = skip item, 1 = take item
    At each step the agent sees one item and decides to include it or not.

    === REWARD FUNCTION (lines 87-89) ===
    Rewards guide the agent toward optimal behavior:
    - +item_value  if the item fits and is taken   (line 88)
    - -10.0        if taking the item overfills     (line 87) -> PENALTY
    -  0.0         if the item is skipped            (line 89)

    The penalty (-10.0) teaches the agent to avoid overfilling.
    The item's value teaches the agent to prefer valuable items.
    The cumulative reward across an episode = total value of selected items.
    """
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = items  # list of (weight, value)

    def reset(self):
        """Reset environment to initial state: empty backpack, first item."""
        return (0, 0)

    def step(self, state, action):
        """Execute one step: return (next_state, reward).

        REWARD FUNCTION — this is where the agent learns what is "good":
        """
        weight, idx = state
        if action == 1:
            item_w, item_v = self.items[idx]
            new_weight = weight + item_w
            if new_weight > self.capacity:
                # REWARD: Negative penalty for overfilling the backpack
                return (new_weight, idx + 1), -10.0
            # REWARD: Positive value for successfully taking an item
            return (new_weight, idx + 1), item_v
        # REWARD: Zero for skipping an item (no gain, no loss)
        return (weight, idx + 1), 0.0

    def is_done(self, state):
        """Episode ends when all items have been considered."""
        return state[1] >= len(self.items)

    def get_solution(self, get_q):
        """Extract the best items according to current Q-table.

        Uses the learned Q-values to make greedy decisions (no exploration).
        This shows what the agent has "learned" after training.
        """
        state = self.reset()
        taken = []
        total_w, total_v = 0, 0
        while not self.is_done(state):
            # Greedy: always pick the action with highest Q-value
            action = max(
                [0, 1],
                key=lambda a: get_q(state, a),
            )
            if action == 1:
                w, v = self.items[state[1]]
                if total_w + w <= self.capacity:
                    taken.append(state[1])
                    total_w += w
                    total_v += v
            state, _ = self.step(state, action)
        return taken, total_w, total_v


# --- Agent ---
# The agent is the "learner" — it interacts with the environment,
# takes actions, receives rewards, and updates its Q-table.

class QLearningAgent:
    """Q-Learning agent that learns optimal knapsack packing.

    === Q-LEARNING OVERVIEW ===
    Q-learning is a model-free RL algorithm. The agent learns a
    Q-function Q(s, a) = "how good is action a in state s?"

    The agent updates Q-values using the BELLMAN EQUATION (line 137):
        Q(s,a) = Q(s,a) + α * [reward + γ * max Q(s',a') - Q(s,a)]
                 ^^^^^^^   ^     ^^^^^   ^     ^^^^^^^^^^^   ^^^^^^^
                 old Q    lr    reward  disc   best future    old Q
                                        factor   Q-value

    - α (alpha, learning rate): how much new info overrides old (0.1)
    - γ (gamma, discount factor): how much future rewards matter (0.95)
    - The term [reward + γ * max Q(s',a') - Q(s,a)] is the "TD error"
      (temporal difference error) — the gap between predicted and actual value.
    """
    def __init__(self, env, epsilon=0.2, alpha=0.1, gamma=0.95):
        self.env = env
        # Closure: Q-table encapsulates the learning memory
        self.get_q, self.set_q, self.q_size = make_q_table()
        # Closure: epsilon-greedy policy encapsulates exploration strategy
        self.explore = make_epsilon_greedy(epsilon)
        self.alpha = alpha    # Learning rate (α): how fast to learn
        self.gamma = gamma    # Discount factor (γ): value of future rewards
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

        # After episode: check if this was the best solution so far
        items, weight, value = self.env.get_solution(self.get_q)
        if value > self.best_value:
            self.best_value = value
            self.best_items = items

        return total_reward, weight

    @time_execution
    def train(self, episodes):
        """Run multiple training episodes.

        Each episode the agent gets slightly smarter as Q-values
        converge toward optimal values through repeated updates.
        """
        print("\n--- Training Start ---\n")
        for ep in range(1, episodes + 1):
            self.train_episode(ep)
        print("\n--- Training End ---")


# --- Presentation ---

def display_result(env, agent):
    items, total_w, total_v = env.get_solution(agent.get_q)
    print("\n" + "=" * 50)
    print("  OPTIMAL SOLUTION")
    print("=" * 50)
    print(f"  Backpack capacity: {env.capacity}")
    print(f"  Items available:   {len(env.items)}")
    print()
    print("  Selected items:")
    for i in items:
        w, v = env.items[i]
        print(f"    Item {i}: weight={w}, value={v}")
    print()
    print(f"  Total weight: {total_w} / {env.capacity}")
    print(f"  Total value:  {total_v:.1f}")
    print("=" * 50)


# --- Main ---
# This ties everything together: environment, agent, and training loop.

def main():
    random.seed(42)  # Fixed seed for reproducible results

    # === PROBLEM SETUP ===
    capacity = 30  # Maximum weight the backpack can hold
    items = [
        # (weight, value) for each item
        (2, 3), (3, 4), (4, 8), (5, 8), (7, 12),
        (8, 10), (9, 14), (10, 11), (12, 18), (15, 20),
    ]

    # === CREATE ENVIRONMENT AND AGENT ===
    env = KnapsackEnv(capacity, items)
    agent = QLearningAgent(env, epsilon=0.2, alpha=0.1, gamma=0.95)

    print("=" * 50)
    print("  RL KNAPSACK SOLVER")
    print("=" * 50)
    print(f"  Capacity: {capacity}")
    print(f"  Items:    {items}")
    print(f"  Q-table entries after training:")

    # === TRAINING ===
    # 500 episodes = 500 complete passes through all items.
    # Early episodes explore randomly; later episodes exploit learned Q-values.
    agent.train(episodes=10)

    # === RESULTS ===
    print(f"\n  Q-table entries: {agent.q_size()}")
    display_result(env, agent)


if __name__ == "__main__":
    main()
