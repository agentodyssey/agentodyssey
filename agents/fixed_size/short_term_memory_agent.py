from __future__ import annotations
from typing import Any, Dict, Optional, List, Type
from functools import lru_cache
import os
import json

from utils import atomic_write
from agents.fixed_size.fixed_size_memory_agent_config import FixedSizeMemoryAgentConfig


class ShortTermMemoryAgent:
    def __init__(self, id: str, name: str, cfg: Optional[FixedSizeMemoryAgentConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = FixedSizeMemoryAgentConfig(available_actions=self.available_actions)

        self.short_term_memory: List[str] = []
        self.short_term_memory_size: int = getattr(self.cfg, "short_term_memory_size", 5)
        print("Short-term memory size:", self.short_term_memory_size)

        self.memory_paths = ["memory.json"]
        self.llm = self.cfg.get_llm()
        self.last_user_prompt: Optional[str] = None

    def memorize(self, info: str):
        self.short_term_memory.append(info)
        # Cap using short_term_memory_size
        if len(self.short_term_memory) > self.short_term_memory_size:
            self.short_term_memory = self.short_term_memory[-self.short_term_memory_size:]

    def _act(self, obs: Dict[str, Any]):
        obs_text = obs["text"]

        retrieved = self.short_term_memory[-self.short_term_memory_size:] if self.short_term_memory else []
        memory_text = "\n".join(retrieved) if retrieved else ""

        user_prompt = self.cfg.construct_user_prompt_long_context(
            memory=memory_text,
            observation=obs_text,
        )
        self.last_user_prompt = user_prompt

        lm_output = self.llm.generate(user_prompt=user_prompt, system_prompt=self.cfg.system_prompt)

        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]
        readable = self.cfg.format_response(parsed)

        step_memory = f"{obs_text}\n\n{readable}"
        self.memorize(step_memory)

        return (
            parsed_action,
            lm_output["num_input_tokens"],
            lm_output["num_output_tokens"],
            lm_output["response"],
        )

    def save_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "agent_id": self.id,
            "agent_name": self.name,
            "cfg": {
                "llm_name": getattr(self.cfg, "llm_name", None),
                "short_term_memory_size": self.short_term_memory_size,
            },
            "short_term_memory": list(self.short_term_memory),
        }
        atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def load_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        if not os.path.exists(path):
            print(f"[ShortTermMemoryAgent] No memory file found at {path}", flush=True)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfgd = data.get("cfg", {})
        if cfgd.get("short_term_memory_size") is not None:
            self.short_term_memory_size = cfgd["short_term_memory_size"]

        self.short_term_memory = list(data.get("short_term_memory", []))
        
        # Cap the loaded short-term memory (matching LoRASFTAgent pattern)
        if len(self.short_term_memory) > self.short_term_memory_size:
            self.short_term_memory = self.short_term_memory[-self.short_term_memory_size:]
        
        print(
            "[Memory] Loaded short-term memory: "
            f"size={len(self.short_term_memory)}",
            flush=True,
        )


@lru_cache(maxsize=None)
def create_short_term_memory_agent(Agent: Type):
    class_name = f"ShortTermMemoryAgent__{Agent.__module__}.{Agent.__name__}"

    return type(
        class_name,
        (ShortTermMemoryAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )