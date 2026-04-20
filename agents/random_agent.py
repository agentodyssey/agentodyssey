from __future__ import annotations
from typing import Dict, Type
from functools import lru_cache
import random

class RandomAgent:
    def __init__(self, id: str, name: str):
        super().__init__(id, name)

    def _act(self, obs: Dict) -> str:
        assert "valid_actions" in obs, "RandomAgent requires 'valid_actions' in the observation."
        valid_actions = obs["valid_actions"]
        random_action_type = random.choice(list(valid_actions.keys()))
        return random.choice(valid_actions[random_action_type]), 0, 0, ""
    
@lru_cache(maxsize=None)
def create_random_agent(Agent: Type):
    class_name = (
        f"RandomAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (RandomAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )