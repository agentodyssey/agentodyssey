from __future__ import annotations
from typing import Dict, Type
from functools import lru_cache

class WaitAgent:
    def __init__(self, id: str, name: str):
        super().__init__(id, name)

    def _act(self, obs: Dict) -> str:
        return "wait", 0, 0, ""
    
@lru_cache(maxsize=None)
def create_wait_agent(Agent: Type):
    class_name = (
        f"WaitAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (WaitAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )