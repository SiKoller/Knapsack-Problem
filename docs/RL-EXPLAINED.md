# Reinforcement Learning Knapsack Solver

This document explains the reinforcement learning (RL) algorithm used to solve the 0/1 knapsack problem in `main.py`.

---

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [RL Concepts Overview](#rl-concepts-overview)
3. [Environment](#environment)
4. [Agent](#agent)
5. [Q-Learning Algorithm](#q-learning-algorithm)
6. [Design Patterns](#design-patterns)
7. [How It All Fits Together](#how-it-all-fits-together)

---

## Problem Definition

The **0/1 Knapsack Problem**: Given a backpack with a weight capacity and a set of items (each with a weight and a value), select items to maximize total value without exceeding the capacity.

- Each item can be taken **once** (0 or 1 — hence "0/1")
- Items cannot be broken into pieces
- Goal: maximize total value within the weight limit

**Example:**

Weights and capacity use floats. Item values and solution totals use integers.
Rewards and Q-values use floats.

```
Capacity: 30.0
Items: [(2.0,3), (3.0,4), (4.0,8), (5.0,8), (7.0,12), (8.0,10), (9.0,14), (10.0,11), (12.0,18), (15.0,20)]
        weight, value

Optimal: Items 0,1,2,3,4,6 → weight=30.0, value=49
```

---

## RL Concepts Overview

Reinforcement learning learns optimal behavior through trial and error. The key components:

| Concept | Description | In This Code |
|---|---|---|
| **Agent** | The learner/decision maker | `QLearningAgent` |
| **Environment** | The world the agent interacts with | `KnapsackEnv` |
| **State** | Current situation of the agent | `(current_weight, next_item_index)` |
| **Action** | What the agent can do | `0` (skip) or `1` (take item) |
| **Reward** | Feedback from the environment | Item value or penalty |
| **Policy** | Strategy for choosing actions | Epsilon-greedy |
| **Q-Value** | Estimated quality of (state, action) | Stored in Q-table |

---

## Environment

**File:** `main.py` — `KnapsackEnv` class

The environment defines the rules of the problem.

### State Space

```
State = (current_weight, next_item_index)
```

- `current_weight`: How much weight is currently in the backpack (0 to capacity)
- `next_item_index`: Which item we are deciding on next (0 to len(items)-1)

**Example:** `(12.0, 5)` means the backpack weighs 12.0 and the next item is item 5.

### Action Space

```
Action = 0 or 1
```

- `0` = skip the current item
- `1` = take the current item (if it fits)

### Reward Function

The reward function tells the agent what is "good" and "bad":

| Action | Condition | Reward | Purpose |
|---|---|---|---|
| Take item | Item fits | `+item_value` | Encourage valuable items |
| Take item | Overfills | `-10.0` | Penalize overfilling |
| Skip item | Any | `0.0` | Neutral (no gain, no loss) |

**Why -10.0?** The penalty must be large enough that the agent learns to avoid overfilling, but not so large that it overshadows item values. A value of -10 works because the maximum item value is 20, making overfilling clearly worse than any single item.

### Key Methods

| Method | Purpose |
|---|---|
| `reset()` | Start a new episode with empty backpack |
| `step(state, action)` | Execute action, return (next_state, reward) |
| `is_done(state)` | Check if all items have been considered |
| `get_solution(get_q)` | Extract best items using learned Q-values |

---

## Agent

**File:** `main.py` — `QLearningAgent` class

The agent interacts with the environment and learns from experience.

### Components

| Component | Source | Purpose |
|---|---|---|
| Q-table | `make_q_table()` closure | Stores learned values for (state, action) pairs |
| Policy | `make_epsilon_greedy()` closure | Decides which action to take |
| Learning rate (α) | `self.alpha = 0.1` | How fast to update Q-values |
| Discount factor (γ) | `self.gamma = 0.95` | Importance of future rewards |

### Training Loop

Each episode follows this cycle:

```
for each item:
    1. Observe current state (weight, item_index)
    2. Choose action using epsilon-greedy policy
    3. Execute action in environment → get reward
    4. Update Q-value using Bellman equation
    5. Move to next state
```

---

## Q-Learning Algorithm

**File:** `main.py` — `QLearningAgent.train_episode()` method

Q-learning learns a function `Q(state, action)` that estimates "how good" an action is in a given state.

### The Bellman Equation

This is the core learning rule (line 159 in `main.py`):

```python
new_q = old_q + alpha * (reward + gamma * best_next_q - old_q)
```

Or in mathematical notation:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',a') − Q(s,a)]
```

Where:
- `Q(s,a)` = current estimated value of taking action `a` in state `s`
- `α` (alpha) = learning rate (0.1) — how much new info overrides old
- `r` = reward received from the environment
- `γ` (gamma) = discount factor (0.95) — value of future rewards
- `max Q(s',a')` = best estimated value from the next state

The term `[r + γ · max Q(s',a') − Q(s,a)]` is called the **TD error** (temporal difference error) — the gap between what we predicted and what we actually got.

### How Learning Happens

1. **Early episodes:** Q-values are all 0.0, agent explores randomly.
2. **Middle episodes:** Agent discovers good actions, Q-values start to differentiate.
3. **Late episodes:** Q-values converge, agent exploits best known actions.

### Convergence

After ~15 episodes, the agent finds the optimal solution (value 49). The Q-table grows to ~267 entries after 500 episodes.

---

## Design Patterns

### Decorator Pattern

**Files:** `log_episode()`, `time_execution()`

Decorators wrap functions to add behavior without modifying the original:

```python
@log_episode
def train_episode(self, episode):
    # ... training logic ...
```

`log_episode` wraps `train_episode` to print episode stats after each run.

```python
@time_execution
def train(self, episodes):
    # ... training loop ...
```

`time_execution` wraps `train` to measure and print total execution time.

### Closure Pattern

**Files:** `make_q_table()`, `make_epsilon_greedy()`

Closures encapsulate private state inside a function:

```python
def make_q_table():
    table = {}  # PRIVATE — cannot be accessed from outside

    def get_q(state, action):
        return table.get((state, action), 0.0)

    def set_q(state, action, value):
        table[(state, action)] = value

    return get_q, set_q, size  # Only these are exposed
```

The `table` dict is private to `make_q_table`. Outside code can only interact with it through `get_q` and `set_q`.

---

## How It All Fits Together

```
main()
  │
  ├── Create KnapsackEnv(capacity, items)
  │     └── Defines state space, action space, reward function
  │
  ├── Create QLearningAgent(env, epsilon, alpha, gamma)
  │     ├── make_q_table() → get_q, set_q (closure)
  │     └── make_epsilon_greedy(epsilon) → explore (closure)
  │
  └── agent.train(500)
        │
        └── for each episode:
              │
              ├── env.reset() → initial state (0.0, 0)
              │
              └── for each item:
                    │
                    ├── explore(state, actions) → action
                    │     └── epsilon-greedy: random or best Q
                    │
                    ├── env.step(state, action) → next_state, reward
                    │     └── REWARD FUNCTION: +value, -penalty, or 0
                    │
                    └── Bellman update:
                          new_q = old_q + α * (r + γ * max_q - old_q)
```

---

## Quick Reference

| Term | Meaning |
|---|---|
| **Episode** | One complete pass through all items |
| **Q-Table** | Dictionary mapping (state, action) → estimated value |
| **Epsilon (ε)** | Probability of exploring random action (0.2) |
| **Alpha (α)** | Learning rate — how fast to update (0.1) |
| **Gamma (γ)** | Discount factor — value of future rewards (0.95) |
| **TD Error** | Difference between predicted and actual value |
| **Exploitation** | Picking the best known action |
| **Exploration** | Trying a random action to discover new strategies |
