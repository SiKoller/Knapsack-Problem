from reward_strategy import KnapsackReward, RewardStrategy


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
    The reward is delegated to an injected RewardStrategy (KnapsackReward
    by default). Each strategy encodes its own reward logic:
    - +item_value  if the item fits and is taken
    - penalty      if taking the item overfills
    -  0.0         if the item is skipped
    """
    def __init__(self, capacity, items, reward_strategy=None):
        self.capacity = capacity
        self.items = items  # list of (weight, value)
        # Reward function is an injected Strategy: swap in any
        # RewardStrategy subclass without changing the transition logic.
        self.reward_strategy = (
            reward_strategy if reward_strategy is not None else KnapsackReward()
        )

    def reset(self):
        """Reset environment to initial state: empty backpack, first item."""
        return (0, 0)

    def step(self, state, action):
        """Execute one step: return (next_state, reward).

        The transition logic (state update) lives here; the reward is
        delegated to the injected RewardStrategy.
        """
        weight, idx = state
        if action == 1:
            item_w = self.items[idx][0]
            next_state = (weight + item_w, idx + 1)
        else:
            next_state = (weight, idx + 1)
        reward = self.reward_strategy.reward(state, action, next_state, self)
        return next_state, reward

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