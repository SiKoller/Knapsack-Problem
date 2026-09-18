"""
Logger implementations demonstrating the Strategy pattern, used as an
injected service.

STRATEGY: define a family of interchangeable algorithms, encapsulate each
one, and make them swappable behind a common interface. Here the
"algorithms" are the ways a training run can be logged — formatted text on
the terminal vs. rows written to a CSV file.

SERVICE: the logger is a dependency of QLearningAgent, handed to it via
constructor injection. The agent only depends on the Logger interface, so
the concrete strategy can be swapped without touching the agent or main().

    Logger            (strategy interface)
    ├── TerminalLogger  (strategy: formatted text on a stream)
    └── CsvLogger       (strategy: serialized rows in a CSV file)
    make_logger()     (simple factory that picks the strategy via --log)
"""
import csv
import sys
from abc import ABC, abstractmethod


class Logger(ABC):
    """Strategy interface: every logger strategy implements these methods."""

    @abstractmethod
    def log_episode(self, episode, reward, weight, capacity, best):
        """Log the outcome of one training episode."""

    @abstractmethod
    def log_message(self, message):
        """Log a free-form message (training start/end, timing, ...)."""

    @abstractmethod
    def log_solution(self, items, total_w, total_v, capacity, available):
        """Log the final solution.

        items:    list of (index, weight, value) tuples for selected items
        total_w / total_v: weight and value of the selected items
        capacity / available: backpack capacity and number of items available
        """

    @abstractmethod
    def close(self):
        """Release any resources held by the logger (e.g. close a file)."""


class TerminalLogger(Logger):
    """Strategy: converts structured log events into formatted text lines.

    All presentation details (column width, indentation) live here, so the
    client code only deals with structured data.
    """

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout

    def log_episode(self, episode, reward, weight, capacity, best):
        # "N/A" until the first improvement has been tracked (-inf sentinel)
        best_display = "N/A" if best == float("-inf") else f"{best:.1f}"
        print(
            f"  Episode {episode:5d} | "
            f"Reward: {reward:7.1f} | "
            f"Weight: {weight}/{capacity} | "
            f"Best: {best_display}",
            file=self.stream,
        )

    def log_message(self, message):
        print(message, file=self.stream)

    def log_solution(self, items, total_w, total_v, capacity, available):
        print("\n" + "=" * 50, file=self.stream)
        print("  OPTIMAL SOLUTION", file=self.stream)
        print("=" * 50, file=self.stream)
        print(f"  Backpack capacity: {capacity}", file=self.stream)
        print(f"  Items available:   {available}", file=self.stream)
        print(file=self.stream)
        print("  Selected items:", file=self.stream)
        for index, w, v in items:
            print(f"    Item {index}: weight={w}, value={v}", file=self.stream)
        print(file=self.stream)
        print(f"  Total weight: {total_w} / {capacity}", file=self.stream)
        print(f"  Total value:  {total_v:.1f}", file=self.stream)
        print("=" * 50, file=self.stream)

    def close(self):
        """No resources to release for a plain text stream."""
        pass


class CsvLogger(Logger):
    """Strategy: serializes structured log events as rows in a CSV file.

    Uses the standard library ``csv.DictWriter`` directly — writes a header
    row on construction and one row per logged event.
    """

    COLUMNS = [
        "event", "episode", "reward", "weight", "best_value",
        "total_weight", "total_value", "capacity", "message",
    ]

    def __init__(self, path="training_log.csv"):
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()  # header row

    def _row(self, **fields):
        """Write one event row; fields not provided become empty cells."""
        row = {col: "" for col in self.COLUMNS}
        row.update(fields)
        self._writer.writerow(row)

    def log_episode(self, episode, reward, weight, capacity, best):
        # Leave best_value empty until the tracker records a first best
        # (the -inf sentinel would otherwise leak into the CSV as junk).
        best_cell = "" if best == float("-inf") else best
        self._row(
            event="episode",
            episode=episode,
            reward=reward,
            weight=weight,
            best_value=best_cell,
        )

    def log_message(self, message):
        # Normalize free-form text so CSV cells stay clean (single line)
        self._row(event="message", message=message.strip())

    def log_solution(self, items, total_w, total_v, capacity, available):
        self._row(
            event="solution",
            total_weight=total_w,
            total_value=total_v,
            capacity=capacity,
        )

    def close(self):
        self._file.close()


# --- Simple factory ---

def make_logger(kind, csv_path="training_log.csv", stream=None):
    """Return the logger strategy matching ``kind`` ("terminal" or "csv")."""
    if kind == "terminal":
        return TerminalLogger(stream=stream)
    if kind == "csv":
        return CsvLogger(path=csv_path)
    raise ValueError(f"Unknown logger kind: {kind!r} (use 'terminal' or 'csv')")