from __future__ import annotations
from typing import Optional, Dict, Type, Any
from functools import lru_cache
import os
import json
import re

from utils import atomic_write
from agents.fixed_size.fixed_size_memory_agent_config import FixedSizeMemoryAgentConfig


class Memory:
    def __init__(self, max_tokens: int = 512, tokenizer=None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.is_text: str = ""

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer(text)["input_ids"])
            except Exception:
                pass
        return max(1, len(text) // 4) # rough fallback that assumes ~1 token ≈ 4 characters

    def _trim(self, is_text: str) -> str:
        while self._count_tokens(is_text) > self.max_tokens:
            is_text = is_text[-800:] if len(is_text) > 800 else is_text[-400:]
            if self._count_tokens(is_text) <= self.max_tokens:
                break
            is_text = is_text[-300:]
        return is_text

    def fragments(self) -> str:
        return f"<IS>{self.is_text}</IS>" if self.is_text else ""

    def update(self, new_is: str):
        self.is_text = self._trim(new_is.strip())

    def to_dict(self) -> dict:
        return {"is_text": self.is_text}

    def load_from_dict(self, d: dict) -> None:
        self.is_text = d.get("is_text", "")

    def save(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def load(self, path: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class Mem1Agent:
    def __init__(self, id: str, name: str, cfg: Optional[FixedSizeMemoryAgentConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = FixedSizeMemoryAgentConfig(available_actions=self.available_actions)
        self.llm = self.cfg.get_llm()
        tokenizer = getattr(self.llm, "tokenizer", None)
        self.memory = Memory(max_tokens=self.cfg.max_memory_tokens, tokenizer=tokenizer)
        self.memory_paths = ["memory.json"]

    def _act(self, obs: Dict) -> str:
        memory = self.memory.fragments()
        user_prompt = self.cfg.construct_user_prompt_constant_context(memory, obs["text"])
        lm_output = self.llm.generate(user_prompt=user_prompt, system_prompt=self.cfg.system_prompt)
        m = re.search(r"<IS>(.*?)</IS>", lm_output["response"], re.S | re.I)
        new_is = m.group(1).strip() if m else (self.memory.is_text + f"\n{obs['text'][:300]}").strip()
        self.memory.update(new_is=new_is)

        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]
        readable = self.cfg.format_response(parsed)

        return (
            parsed_action,
            lm_output["num_input_tokens"],
            lm_output["num_output_tokens"],
            lm_output["response"],
        )

    def save_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        payload = {
            "agent_id": self.id,
            "agent_name": self.name,
            "cfg": {
                "llm_name": getattr(self.cfg, "llm_name", None),
                "max_memory_tokens": getattr(self.cfg, "max_memory_tokens", None),
            },
            "memory": self.memory.to_dict(),
        }
        self.memory.save(path, payload)

    def load_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        data = self.memory.load(path)
        if not data:
            print(f"[Mem1Agent] No memory file found at {path}", flush=True)
            return

        cfgd = data.get("cfg", {})
        for k in ["llm_name", "max_memory_tokens"]:
            if k in cfgd and cfgd[k] is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, cfgd[k])

        self.memory.load_from_dict(data.get("memory", {}))


@lru_cache(maxsize=None)
def create_mem1_agent(Agent: Type):
    class_name = (
        f"Mem1Agent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (Mem1Agent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )