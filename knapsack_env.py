from config import PENALTY_FOR_OVERFILL


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

    === REWARD FUNCTION ===
    Rewards guide the agent toward optimal behavior:
    - +item_value  if the item fits and is taken
    - PENALTY      if taking the item overfills (see config.PENALTY_FOR_OVERFILL)
    -  0.0         if the item is skipped

    The penalty teaches the agent to avoid overfilling.
    The item's value teaches the agent to prefer valuable items.
    The cumulative reward across an episode = total value of selected items.
    """
    def __init__(self, capacity, items, penalty=PENALTY_FOR_OVERFILL):
        self.capacity = capacity
        self.items = items  # list of (weight, value)
        self.penalty = penalty

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
                return (new_weight, idx + 1), self.penalty
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