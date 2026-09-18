"""ResultTracker: remembers the best solution found during training.

Extracted from QLearningAgent so that learning (Q-table updates) and
bookkeeping (remembering the best result) are separate responsibilities.

    best_value  highest total value seen so far (starts at -inf)
    best_items  item indices of that best solution
    best_weight total weight of that best solution
"""


class ResultTracker:
    """Remembers the best (highest-value) solution seen so far.

    A candidate only becomes the best if its value strictly exceeds the
    current best; ties keep the existing (earlier) best.
    """

    def __init__(self):
        self.reset()

    def update(self, value, items, weight):
        """Record a candidate solution if it beats the current best."""
        if value > self.best_value:
            self.best_value = value
            self.best_items = list(items)
            self.best_weight = weight

    def reset(self):
        """Forget all tracked results (back to the initial state)."""
        self.best_value = float("-inf")
        self.best_items = []
        self.best_weight = 0