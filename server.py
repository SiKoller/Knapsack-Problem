"""Small web interface for the existing knapsack learner. Run: python3 server.py"""

import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from rl_knapsack.config import CONFIG, Item
from rl_knapsack.items_loader import load_items
from rl_knapsack.knapsack_env import KnapsackEnv
from rl_knapsack.q_learning_agent import QLearningAgent
from rl_knapsack.result_tracker import ResultTracker


ROOT = Path(__file__).parent


class RunLogger:
    def __init__(self):
        self.history = []

    def log_episode(self, episode, reward, weight, capacity, best):
        self.history.append({"episode": episode, "reward": reward, "weight": weight, "best": best})

    def log_message(self, message):
        pass

    def log_solution(self, *args, **kwargs):
        pass

    def close(self):
        pass


def solve(payload):
    capacity = payload.get("capacity")
    episodes = payload.get("episodes")
    epsilon = payload.get("epsilon")
    items = payload.get("items")
    if type(capacity) not in (int, float) or not 1 <= capacity <= 1000:
        raise ValueError("Capacity must be a number from 1 to 1000.")
    if type(episodes) is not int or not 1 <= episodes <= 5000:
        raise ValueError("Episodes must be an integer from 1 to 5000.")
    if type(epsilon) not in (int, float) or not 0 <= epsilon <= 1:
        raise ValueError("Exploration must be between 0 and 1.")
    if not isinstance(items, list) or not 1 <= len(items) <= 16 or any(
        not isinstance(item, list) or len(item) != 2
        or type(item[0]) not in (int, float) or not 0 < item[0] <= 1000
        or type(item[1]) is not int or not 1 <= item[1] <= 1000
        for item in items
    ):
        raise ValueError("Choose 1 to 16 items with positive integer weights and values.")

    random.seed(CONFIG.random_seed)
    env = KnapsackEnv(capacity, [Item(float(weight), value) for weight, value in items])
    tracker = ResultTracker()
    logger = RunLogger()
    agent = QLearningAgent(env, logger=logger, epsilon=epsilon,
                           alpha=CONFIG.alpha, gamma=CONFIG.gamma, tracker=tracker)
    agent.train(episodes)
    learned_indices, learned_weight, learned_value = env.get_solution(agent.get_q)

    # Exact benchmark is small enough to enumerate for the UI's item limit.
    optimum = {"value": 0, "weight": 0, "indices": []}
    for mask in range(1 << len(items)):
        indices = [i for i in range(len(items)) if mask & (1 << i)]
        weight = sum(items[i][0] for i in indices)
        value = sum(items[i][1] for i in indices)
        if weight <= capacity and value > optimum["value"]:
            optimum = {"value": value, "weight": weight, "indices": indices}

    return {
        "history": logger.history,
        "learned": {"indices": learned_indices, "weight": learned_weight, "value": learned_value},
        "best": {"indices": tracker.best_items, "weight": tracker.best_weight, "value": tracker.best_value},
        "optimum": optimum,
        "qStates": agent.q_size(),
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/problem":
            capacity, items = load_items()
            return self.send_json(200, {"capacity": capacity, "items": items})
        files = {"/": ("web/index.html", "text/html"),
                 "/style.css": ("web/style.css", "text/css"),
                 "/app.js": ("web/app.js", "text/javascript")}
        if self.path not in files:
            return self.send_json(404, {"error": "Not found"})
        path, mime = files[self.path]
        body = (ROOT / path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/train":
            return self.send_json(404, {"error": "Not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("Request is too large or empty.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            self.send_json(200, solve(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("Knapsack Lab running at http://127.0.0.1:8000")
    server.serve_forever()
