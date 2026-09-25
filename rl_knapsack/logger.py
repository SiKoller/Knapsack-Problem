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
import logging
import sys
from abc import ABC, abstractmethod
from .config import CONFIG


RESULT = 25
logging.addLevelName(RESULT, "RESULT")


class Logger(ABC):
    """Strategy interface: every logger strategy implements these methods."""

    @abstractmethod
    def log_episode(self, episode: int, reward: float, weight: float, capacity: float, best: int | None):
        """Log the outcome of one training episode."""

    @abstractmethod
    def log_message(self, message, level=logging.INFO):
        """Log a message at the given level."""

    @abstractmethod
    def log_solution(self, items: list[tuple[int, float, int]], total_w: float, total_v: int, capacity: float, available: int):
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

    def __init__(self, stream=None, level=logging.INFO):
        self._logger = logging.Logger(__name__)
        self._logger.setLevel(level.upper() if isinstance(level, str) else level)
        self._handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
        self._handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self._logger.addHandler(self._handler)

    def log_episode(self, episode: int, reward: float, weight: float, capacity: float, best: int | None):
        best_display = "N/A" if best is None else str(best)
        self._logger.debug(
            f"Episode {episode:5d} | "
            f"Reward: {reward:7.1f} | "
            f"Weight: {weight}/{capacity} | "
            f"Best: {best_display}",
        )

    def log_message(self, message, level=logging.INFO):
        self._logger.log(level, message.strip("\n"))

    def log_solution(self, items: list[tuple[int, float, int]], total_w: float, total_v: int, capacity: float, available: int):
        lines = [
            "OPTIMAL SOLUTION",
            "=" * CONFIG.banner_width,
            f"  Backpack capacity: {capacity}",
            f"  Items available:   {available}",
            "",
            "  Selected items:",
            *(f"    Item {index}: weight={w}, value={v}" for index, w, v in items),
            "",
            f"  Total weight: {total_w} / {capacity}",
            f"  Total value:  {total_v}",
            "=" * CONFIG.banner_width,
        ]
        self.log_message("\n".join(lines), RESULT)

    def close(self):
        """Release the handler. Keep the output stream open."""
        self._logger.removeHandler(self._handler)
        self._handler.close()


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
        self._writer.writerow(fields)

    def log_episode(self, episode: int, reward: float, weight: float, capacity: float, best: int | None):
        # Leave the cell empty until a result is recorded.
        best_cell = "" if best is None else best
        self._row(
            event="episode",
            episode=episode,
            reward=reward,
            weight=weight,
            best_value=best_cell,
        )

    def log_message(self, message, level=logging.INFO):
        # Normalize free-form text so CSV cells stay clean (single line)
        self._row(event="message", message=message.strip())

    def log_solution(self, items: list[tuple[int, float, int]], total_w: float, total_v: int, capacity: float, available: int):
        self._row(
            event="solution",
            total_weight=total_w,
            total_value=total_v,
            capacity=capacity,
        )

    def close(self):
        self._file.close()


# --- Simple factory ---

def make_logger(kind, csv_path="training_log.csv", stream=None, level=logging.INFO):
    """Return the logger strategy matching ``kind`` ("terminal" or "csv")."""
    if kind == "terminal":
        return TerminalLogger(stream=stream, level=level)
    if kind == "csv":
        return CsvLogger(path=csv_path)
    raise ValueError(f"Unknown logger kind: {kind!r} (use 'terminal' or 'csv')")
