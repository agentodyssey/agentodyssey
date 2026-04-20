from __future__ import annotations
from typing import Optional, Dict, Any, Type
from functools import lru_cache

from agents.llm_agent_config import LLMAgentConfig

class NoMemoryAgent:
    def __init__(self, id: str, name: str, cfg: Optional[LLMAgentConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = LLMAgentConfig(available_actions=self.available_actions)
        self.llm = self.cfg.get_llm()
        
    def _act(self, obs: Dict) -> str:
        lm_output = self.llm.generate(user_prompt=f"My Current Observation: {obs['text']}\n" + f"{self.cfg.action_prompt}", system_prompt=self.cfg.system_prompt)
        # print("llm output:", lm_output)
        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]
        readable = self.cfg.format_response(parsed)
        # print("parsed action:", parsed_action)
        return parsed_action, lm_output["num_input_tokens"], lm_output["num_output_tokens"], lm_output["response"]
    
@lru_cache(maxsize=None)
def create_no_memory_agent(Agent: Type):
    class_name = (
        f"NoMemoryAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (NoMemoryAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )