# Knapsack Lab

An interactive frontend for the Q-learning knapsack solver. The dashboard trains the existing Python agent, plots its progress, and compares its final choice with an exact benchmark.

## Run locally

Requires Python 3.14 or newer.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python server.py
```

Open <http://127.0.0.1:8000>. Change the backpack capacity, training episodes, exploration rate, or included items, then click **Train the agent**.

Run the tests with `.venv/bin/python -m unittest discover -s tests`.
