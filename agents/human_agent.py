from __future__ import annotations
from typing import Dict, Type
from functools import lru_cache
from termcolor import colored

class HumanAgent:
    def __init__(self, id: str, name: str):
        super().__init__(id, name)

    def _act(self, obs: Dict):
        print(obs["text"])
        text = input(colored("> My action: ", "yellow"))
        return text, 0, 0, ""
    
@lru_cache(maxsize=None)
def create_human_agent(Agent: Type):
    class_name = (
        f"HumanAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (HumanAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )