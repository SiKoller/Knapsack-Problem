import json
import os
import sys
import tempfile
import unittest

# Ensure the project root is on sys.path so the `rl_knapsack` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_knapsack.items_loader import load_items
from rl_knapsack.config import Item


class TestLoadItems(unittest.TestCase):
    """Tests for loading the knapsack problem from a JSON file."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = os.path.join(self.tmpdir.name, "items.json")

    def write(self, payload):
        with open(self.path, "w") as f:
            json.dump(payload, f)

    def test_loads_capacity_and_items(self):
        """A valid file returns (capacity, items as (weight, value) tuples)."""
        self.write({"capacity": 30, "items": [[2, 3], [3, 4]]})
        capacity, items = load_items(self.path)
        self.assertEqual(capacity, 30)
        self.assertEqual(items, [(2, 3), (3, 4)])
        self.assertIsInstance(items[0], Item)
        self.assertIs(type(capacity), float)
        self.assertIs(type(items[0].weight), float)
        self.assertIs(type(items[0].value), int)

    def test_fractional_weights_and_integer_values(self):
        self.write({"capacity": 3.75, "items": [[1.25, 3], [2.5, 4]]})
        capacity, items = load_items(self.path)
        self.assertEqual(capacity, 3.75)
        self.assertEqual(items, [Item(1.25, 3), Item(2.5, 4)])

    def test_float_value_is_rejected(self):
        self.write({"capacity": 3.75, "items": [[1.25, 3.0]]})
        with self.assertRaises(ValueError):
            load_items(self.path)

    def test_default_file_loads(self):
        """The default problem file supplies typed items."""
        from rl_knapsack.items_loader import DEFAULT_ITEMS_PATH

        self.assertTrue(os.path.isfile(DEFAULT_ITEMS_PATH))
        capacity, items = load_items(DEFAULT_ITEMS_PATH)
        self.assertGreater(capacity, 0)
        self.assertTrue(items)
        self.assertTrue(all(isinstance(item, Item) for item in items))

    def test_missing_file_raises(self):
        """A non-existent path must raise ValueError with a clear message."""
        with self.assertRaises(ValueError):
            load_items(os.path.join(self.tmpdir.name, "nope.json"))

    def test_malformed_json_raises(self):
        """Invalid JSON content must raise ValueError."""
        with open(self.path, "w") as f:
            f.write("not json")
        with self.assertRaises(ValueError):
            load_items(self.path)

    def test_missing_capacity_key_raises(self):
        """A file without 'capacity' must raise ValueError."""
        self.write({"items": [[2, 3]]})
        with self.assertRaises(ValueError):
            load_items(self.path)

    def test_bad_capacity_raises(self):
        """A non-positive capacity must raise ValueError."""
        self.write({"capacity": -5, "items": [[2, 3]]})
        with self.assertRaises(ValueError):
            load_items(self.path)

    def test_missing_items_key_raises(self):
        self.write({"capacity": 30})
        with self.assertRaises(ValueError):
            load_items(self.path)

    def test_items_must_be_an_array(self):
        self.write({"capacity": 30, "items": "2,3"})
        with self.assertRaises(ValueError):
            load_items(self.path)

    def test_bad_item_shape_raises(self):
        """An item that is not a [weight, value] pair must raise ValueError."""
        self.write({"capacity": 30, "items": [[2, 3], [5]]})
        with self.assertRaises(ValueError):
            load_items(self.path)


if __name__ == '__main__':
    unittest.main()
