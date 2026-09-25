from pathlib import Path

from pydantic import ValidationError

from .config import Item, ProblemConfig


# --- Items Loading ---
# The knapsack problem (capacity + items) lives in a JSON file and is
# loaded at runtime. ProblemConfig parses and checks the JSON data.

# Default problem file (JSON). Resolved relative to this module so the
# default works no matter which directory the program is run from.
DEFAULT_ITEMS_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "items.json"
)


def load_items(path=DEFAULT_ITEMS_PATH) -> tuple[float, list[Item]]:
    """Load a knapsack problem (capacity + items) from a JSON file.

    Expected format:
        {
            "capacity": 30,
            "items": [[weight, value], ...]
        }

    Returns a tuple (capacity, items) where items is a list of Item values.
    Raises ValueError with a helpful message on missing/malformed input.
    """
    try:
        problem = ProblemConfig.model_validate_json(Path(path).read_bytes())
    except FileNotFoundError:
        raise ValueError(f"Items file not found: {path}") from None
    except ValidationError as exc:
        raise ValueError(f"Items file {path}: {exc}") from None

    return problem.capacity, list(problem.items)
