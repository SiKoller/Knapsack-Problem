import json
from pathlib import Path


# --- Items Loading ---
# The knapsack problem (capacity + items) lives in a JSON file and is
# loaded at runtime. Keeping the loader in its own module keeps config.py
# a pure constants module.

# Default problem file (JSON). Resolved relative to this module so the
# default works no matter which directory the program is run from.
DEFAULT_ITEMS_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "items.json"
)


def load_items(path=DEFAULT_ITEMS_PATH):
    """Load a knapsack problem (capacity + items) from a JSON file.

    Expected format:
        {
            "capacity": 30,
            "items": [[weight, value], ...]
        }

    Returns a tuple (capacity, items) where items is a list of
    (weight, value) tuples — exactly what KnapsackEnv expects.
    Raises ValueError with a helpful message on missing/malformed input.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Items file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Items file is not valid JSON: {path} ({exc})"
        ) from None

    if not isinstance(data, dict) or "capacity" not in data or "items" not in data:
        raise ValueError(
            f"Items file {path} must contain 'capacity' and 'items' keys"
        )

    capacity = data["capacity"]
    if not isinstance(capacity, (int, float)) or capacity <= 0:
        raise ValueError(
            f"Items file {path}: 'capacity' must be a positive number, "
            f"got {capacity!r}"
        )

    raw_items = data["items"]
    if not isinstance(raw_items, list):
        raise ValueError(f"Items file {path}: 'items' must be a list")

    items = []
    for i, pair in enumerate(raw_items):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(n, (int, float)) for n in pair)
        ):
            raise ValueError(
                f"Items file {path}: item {i} must be a [weight, value] "
                f"pair, got {pair!r}"
            )
        items.append((pair[0], pair[1]))

    return capacity, items