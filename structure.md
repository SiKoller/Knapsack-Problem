# Structure of `main.py`

Mermaid class diagram showing the structure of the RL Knapsack solver in `Lesson1/main.py` (plus its supporting modules `logger.py` and `result_tracker.py`).

```mermaid
classDiagram
    direction TB

    class KnapsackEnv {
        +int capacity
        +list items
        +float penalty
        +reset() tuple
        +step(state, action) (next_state, reward)
        +is_done(state) bool
        +get_solution(get_q) (items, total_w, total_v)
    }

    class QLearningAgent {
        +KnapsackEnv env
        +Logger logger  (injected service)
        +get_q, set_q, q_size  (closure)
        +explore               (closure)
        +float epsilon
        +float alpha
        +float gamma
        +ResultTracker tracker  (optional, default None)
        +train_episode(episode) tuple
        +train(episodes) None
    }

    class ResultTracker {
        +float best_value
        +list best_items
        +int best_weight
        +update(value, items, weight) None
        +reset() None
    }

    class Logger {
        <<strategy interface>>
        +log_episode(episode, reward, weight, capacity, best)
        +log_message(message)
        +log_solution(items, total_w, total_v, capacity, available)
        +close()
    }

    class TerminalLogger {
        <<concrete strategy>>
        +stream
        +close()  (no-op)
    }

    class CsvLogger {
        <<concrete strategy>>
        -csv.DictWriter _writer
        +close()  (closes file)
    }

    class make_logger {
        <<simple factory>>
        +make_logger(kind, csv_path, stream) Logger
    }

    class log_episode {
        <<decorator>>
        +wrapper(self, episode)
    }

    class time_execution {
        <<decorator>>
        +wrapper(*args, **kwargs)
    }

    class make_q_table {
        <<closure factory>>
        +table: dict  (private)
        +returns get_q, set_q, size
    }

    class make_epsilon_greedy {
        <<closure factory>>
        +epsilon: float
        +returns choose(state, actions, get_q)
    }

    class display_result {
        +display_result(env, agent) None
    }

    class main {
        +main() None
    }

    QLearningAgent --> KnapsackEnv : uses / owns env
    KnapsackEnv <.. QLearningAgent : observes (state, reward)
    QLearningAgent --> Logger : uses (constructor injection)
    QLearningAgent --> ResultTracker : optional best-tracking
    Logger <|-- TerminalLogger
    Logger <|-- CsvLogger
    QLearningAgent ..> make_q_table : builds Q-table closure
    QLearningAgent ..> make_epsilon_greedy : builds exploration policy
    QLearningAgent ..> log_episode : @decorates train_episode
    QLearningAgent ..> time_execution : @decorates train
    main ..> make_logger : builds strategy
    make_logger ..> TerminalLogger : creates
    make_logger ..> CsvLogger : creates
    main ..> ResultTracker : instantiates tracker
    display_result --> KnapsackEnv : reads capacity/items/solution
    display_result --> QLearningAgent : uses its logger + Q-values
    main ..> KnapsackEnv : instantiates env
    main ..> QLearningAgent : instantiates agent
    main ..> display_result : calls
```

## Key relationships

| Component | Role |
| --- | --- |
| `KnapsackEnv` | Defines the state space, action space, and reward function (0/1 knapsack). |
| `QLearningAgent` | The learner — runs the Q-learning loop with the Bellman equation. Owns a `KnapsackEnv` and receives a `Logger` via constructor injection. |
| `ResultTracker` | Remembers the best (highest-value) solution found during training (`update` / `reset`). |
| `Logger` (interface) | **Strategy contract** — the only interface the agent/main code depends on for output. |
| `TerminalLogger` | **Concrete strategy** — renders structured log events as formatted text on a stream (default stdout). |
| `CsvLogger` | **Concrete strategy** — serializes structured log events into CSV rows via the stdlib `csv.DictWriter` (header + event rows). |
| `make_logger(kind, ...)` | Simple factory returning the right strategy ("terminal" or "csv") from the `--log` CLI flag. |
| `make_q_table()` | Closure returning `get_q`, `set_q`, `size` — encapsulates the Q-table (private dict). |
| `make_epsilon_greedy(epsilon)` | Closure returning `choose` — balances exploration vs exploitation. |
| `log_episode` / `time_execution` | Decorators applied to `QLearningAgent` methods; both delegate output to the agent's `Logger`. |
| `display_result(env, agent)` | Delegates the solution report to the agent's `Logger` strategy. |
| `main()` | Ties everything together: parses `--log`/`--csv-path`, builds `env`, `tracker`, `logger`, `agent`, trains, shows results. |