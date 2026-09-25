from abc import ABC, abstractmethod

from .config import CONFIG


# --- Reward Strategy ---
# The reward function encodes the problem's objective. By defining it as a
# Strategy, the environment can swap reward functions without touching its
# transition logic or the agent. This mirrors the Logger strategy: the env
# only depends on the RewardStrategy interface.

class RewardStrategy(ABC):
    """Interface for computing the reward of a single transition."""

    @abstractmethod
    def reward(self, state, action, next_state, env) -> float:
        """Return the reward for taking `action` from `state`.

        Args:
            state:      state before the action (e.g. (weight, item_index))
            action:     action taken (0 = skip, 1 = take)
            next_state: state after the action
            env:        the environment, for inspecting capacity/items
        """
        raise NotImplementedError


class KnapsackReward(RewardStrategy):
    """Reward function for the 0/1 knapsack problem.

    Rewards guide the agent toward optimal behavior:
    - +item_value  if the item fits and is taken
    - penalty      if taking the item overfills the backpack
    -  0.0         if the item is skipped

    The penalty teaches the agent to avoid overfilling.
    The item's value teaches the agent to prefer valuable items.
    The cumulative reward across an episode = total value of selected items.
    """

    def __init__(self, penalty=CONFIG.penalty_for_overfill):
        self.penalty = penalty

    def reward(self, state, action, next_state, env) -> float:
        weight, idx = state
        if action == 1:
            item = env.items[idx]
            if weight + item.weight > env.capacity:
                # Negative penalty for overfilling the backpack
                return self.penalty
            # Positive value for successfully taking an item
            return float(item.value)
        # Zero for skipping an item (no gain, no loss)
        return 0.0
