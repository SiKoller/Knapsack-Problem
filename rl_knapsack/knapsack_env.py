from abc import ABC, abstractmethod

from .reward_strategy import KnapsackReward, RewardStrategy


# --- Env Lifecycle (State pattern) ---
# The environment is a state machine with three phases. Each phase is a
# State object that handles step() calls: NotStarted and EpisodeFinished
# refuse to step, InEpisode executes the transition. This mirrors the
# Gymnasium contract: step() before reset() or after the episode is done
# is an error.

class EnvState(ABC):
    """Phase of a KnapsackEnv episode lifecycle."""

    @abstractmethod
    def step(self, env, state, action):
        """Handle a step() call in this phase; raise if not permitted."""
        raise NotImplementedError


class NotStarted(EnvState):
    """Fresh environment: before the first reset(). No stepping allowed."""

    def step(self, env, state, action):
        raise RuntimeError("step() called before reset()")


class InEpisode(EnvState):
    """Inside an episode: stepping is allowed."""

    def step(self, env, state, action):
        return env._advance(state, action)


class EpisodeFinished(EnvState):
    """All items have been considered. No more stepping allowed."""

    def step(self, env, state, action):
        raise RuntimeError("step() called after the episode finished")


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

    === LIFECYCLE (State pattern) ===
    The env is a state machine: NotStarted -> InEpisode -> EpisodeFinished.
    step() is only valid inside an episode; reset() can be called from any
    phase to (re)start an episode.

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
        # Lifecycle phase (State pattern): stepping is not allowed until
        # reset() has been called.
        self._phase = NotStarted()

    def reset(self):
        """Reset environment to initial state: empty backpack, first item.

        Valid from any phase: starts (or restarts) an episode.
        """
        self._phase = InEpisode()
        return (0, 0)

    def step(self, state, action):
        """Execute one step: return (next_state, reward).

        Delegated to the current lifecycle phase: illegal steps (before
        reset() or after the episode finished) raise a RuntimeError.
        """
        return self._phase.step(self, state, action)

    def is_done(self):
        """Episode ends when all items have been considered."""
        return isinstance(self._phase, EpisodeFinished)

    def _advance(self, state, action):
        """Transition math + reward; called by the InEpisode phase.

        The reward is delegated to the injected RewardStrategy.
        """
        weight, idx = state
        if action == 1:
            item_w = self.items[idx][0]
            next_state = (weight + item_w, idx + 1)
        else:
            next_state = (weight, idx + 1)
        reward = self.reward_strategy.reward(state, action, next_state, self)
        if next_state[1] >= len(self.items):
            self._phase = EpisodeFinished()
        return next_state, reward

    def get_solution(self, get_q):
        """Extract the best items according to current Q-table.

        Uses the learned Q-values to make greedy decisions (no exploration).
        This shows what the agent has "learned" after training.
        """
        state = self.reset()
        taken = []
        total_w, total_v = 0, 0
        while not self.is_done():
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