from __future__ import annotations
from typing import Dict, Type
from functools import lru_cache
import os
import json


class ReplayAgent:
    """Agent that replays actions from an agent_log.jsonl file.

    The path to the log file is read from the REPLAY_ACTIONS_PATH
    environment variable.
    """

    def __init__(self, id: str, name: str):
        super().__init__(id, name)
        log_path = os.environ.get("REPLAY_ACTIONS_PATH")
        if not log_path:
            raise ValueError("REPLAY_ACTIONS_PATH environment variable must be set for ReplayAgent")
        with open(log_path, "r") as f:
            self._actions = [json.loads(line)["action"] for line in f if line.strip()]
        self._step = 0

    def _act(self, obs: Dict) -> str:
        if self._step >= len(self._actions):
            raise IndexError(
                f"ReplayAgent ran out of actions at step {self._step} "
                f"(only {len(self._actions)} actions available)"
            )
        action = self._actions[self._step]
        self._step += 1
        return action, 0, 0, ""


@lru_cache(maxsize=None)
def create_replay_agent(Agent: Type):
    class_name = (
        f"ReplayAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (ReplayAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )
