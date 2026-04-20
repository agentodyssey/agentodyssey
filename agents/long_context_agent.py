from __future__ import annotations
import os
from typing import Optional, Dict, Any, Type
from functools import lru_cache
from utils import atomic_write
from agents.llm_agent_config import LLMAgentConfig

class Memory:
    def __init__(self):
        self.context: str = ""

    def add(self, info: str):
        self.context += info + "\n"

    def get_all_memory(self) -> str:
        return self.context
    
    def reset(self):
        self.context = ""

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            self.context = f.read()

    def save(self, path: str) -> None:
        atomic_write(path, self.context)

class LongContextAgent:
    def __init__(self, id: str, name: str, cfg: Optional[LLMAgentConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = LLMAgentConfig(available_actions=self.available_actions)
        self.memory = Memory()
        self.memory_paths = ["content.txt"]
        self.llm = self.cfg.get_llm()

    def construct_user_prompt_long_context(self, memory: str, observation: str) -> str:
        memory_prompt = "My Memories: " + memory if memory else ""
        assert self.cfg.action_prompt is not None
        user_prompt = memory_prompt + f"My Current Observation: {observation}\n" + f"{self.cfg.action_prompt}"
        return user_prompt

    def memorize(self, info: str):
        self.memory.add(info)

    def _act(self, obs: Dict[str, Any]):
        obs_text = obs["text"]

        retrieved_text = self.memory.get_all_memory()

        reflection = None
        if self.cfg.enable_reflection and retrieved_text:
            reflection = self.cfg.reflect(self.llm, obs_text, retrieved_text)
            if reflection:
                self.memorize(reflection)

        prompt_memory = self.memory.get_all_memory()
        user_prompt = self.construct_user_prompt_long_context(prompt_memory, obs_text)
        lm_output = self.llm.generate(user_prompt=user_prompt, system_prompt=self.cfg.system_prompt)
        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]

        readable = self.cfg.format_response(parsed)
        to_store = f"{obs_text}\n{readable}"
        print(to_store)

        if self.cfg.enable_summarization:
            self.memorize(self.cfg.summarize(self.llm, to_store))
        else:
            self.memorize(to_store)

        return parsed_action, lm_output["num_input_tokens"], lm_output["num_output_tokens"], lm_output["response"]

    def save_memory(self, full_memory_dir: str) -> None:
        self.memory.save(os.path.join(full_memory_dir, self.memory_paths[0]))

    def load_memory(self, full_memory_dir: str) -> None:
        self.memory.load(os.path.join(full_memory_dir, self.memory_paths[0]))

@lru_cache(maxsize=None)
def create_long_context_agent(Agent: Type):
    class_name = (
        f"LongContextAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (LongContextAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )