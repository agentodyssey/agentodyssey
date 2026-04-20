from __future__ import annotations
from typing import Any, Dict, Optional, List, Type
from functools import lru_cache
import os
import json
import torch

from utils import atomic_write
from agents.rag.rag_agent_config import RAGAgentConfig

class Memory:
    def __init__(self, embedder):
        self.embedder = embedder
        self.mem: List[str] = []
        self.emb: Optional[torch.Tensor] = None

    def add(self, info: str):
        new_emb = self.embedder.encode([info])
        self.mem.append(info)
        self.emb = new_emb if self.emb is None else torch.cat([self.emb, new_emb], dim=0)

    @torch.inference_mode()
    def retrieve(self, observation: str, memory_retrieve_limit: int) -> List[str]:
        if self.emb is None or self.emb.shape[0] == 0:
            return []
        query = self.embedder.encode([observation])
        sims = (query @ self.emb.T).squeeze(0)
        topk = torch.topk(sims, k=min(memory_retrieve_limit, sims.numel()))
        return [self.mem[i] for i in topk.indices.tolist()]

    def to_dict(self) -> dict:
        return {"mem": list(self.mem)}

    def load_from_dict(self, d: dict) -> None:
        self.mem = list(d.get("mem", []))
        self.emb = None

    def rebuild_embeddings(self) -> None:
        if not self.mem:
            self.emb = None
            return
        print(f"[Memory] Rebuilding embeddings, size={len(self.mem)}", flush=True)
        self.emb = self.embedder.encode(self.mem)

    def save(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def load(self, path: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class VanillaRAGAgent:
    def __init__(self, id: str, name: str, cfg: Optional[RAGAgentConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = RAGAgentConfig(available_actions=self.available_actions)

        self.embedder = self.cfg.get_embedder()
        self.memory = Memory(self.embedder)

        self.short_term_memory: List[str] = []
        self.short_term_memory_size = getattr(self.cfg, "short_term_memory_size", 5)

        self.enable_short_term_memory = getattr(self.cfg, "enable_short_term_memory", False)
        print(
            f"[VanillaRAGAgent] Short-term memory enabled = {self.enable_short_term_memory}",
            flush=True,
        )

        self.memory_paths = ["memory.json"]
        self.llm = self.cfg.get_llm()
        self.last_user_prompt: Optional[str] = None

    def memorize(self, info: str):
        self.memory.add(info)

        if not getattr(self.cfg, "enable_short_term_memory", False):
            return

        self.short_term_memory.append(info)
        if len(self.short_term_memory) > self.short_term_memory_size:
            self.short_term_memory = self.short_term_memory[-self.short_term_memory_size:]

    def _format_step_memory(
        self,
        *,
        state_action: str,
        current_reflection: Optional[str],
    ) -> str:
        parts: List[str] = []
        parts.append(state_action)
        if current_reflection:
            parts.append(current_reflection)
        return "\n\n".join(parts)

    def _act(self, obs: Dict[str, Any]):
        obs_text = obs["text"]

        retrieved_long_term = self.memory.retrieve(
            obs_text, memory_retrieve_limit=self.cfg.memory_retrieve_limit
        )

        if getattr(self.cfg, "enable_short_term_memory", False):
            retrieved_short_term = (
                self.short_term_memory[-self.short_term_memory_size:] if self.short_term_memory else []
            )
        else:
            retrieved_short_term = []

        # Deduplicate long-term against short-term, then concat: long-term first, short-term second
        short_term_set = set(retrieved_short_term)
        retrieved_long_term_deduped = [item for item in retrieved_long_term if item not in short_term_set]

        retrieved = retrieved_long_term_deduped + retrieved_short_term

        current_reflection: Optional[str] = None
        if self.cfg.enable_reflection and retrieved:
            retrieved_text = "\n".join(retrieved)
            current_reflection = self.cfg.reflect(self.llm, obs_text, retrieved_text)

        user_prompt = self.cfg.construct_user_prompt_with_current_reflection(
            retrieved_memories=retrieved,
            observation=obs_text,
            current_reflection=current_reflection,
        )
        self.last_user_prompt = user_prompt

        lm_output = self.llm.generate(user_prompt=user_prompt, system_prompt=self.cfg.system_prompt)

        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]
        readable = self.cfg.format_response(parsed)

        to_summarize = f"{obs_text}\n{readable}"
        step_text = to_summarize

        if self.cfg.enable_summarization:
            summary = self.cfg.summarize(self.llm, to_summarize)
            step_text = (summary if summary else to_summarize)

        step_memory = self._format_step_memory(
            state_action=step_text,
            current_reflection=current_reflection,
        )
        self.memorize(step_memory)

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
                "embed_name": getattr(self.cfg, "embed_name", None),
                "memory_retrieve_limit": getattr(self.cfg, "memory_retrieve_limit", None),
                "enable_reflection": getattr(self.cfg, "enable_reflection", None),
                "enable_summarization": getattr(self.cfg, "enable_summarization", None),
                "enable_short_term_memory": getattr(self.cfg, "enable_short_term_memory", None),
                "short_term_memory_size": getattr(self.cfg, "short_term_memory_size", None),
            },
            "memory": self.memory.to_dict(),
            "short_term_memory": (
                list(self.short_term_memory)
                if getattr(self.cfg, "enable_short_term_memory", False)
                else []
            ),
        }
        self.memory.save(path, payload)

    def load_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        data = self.memory.load(path)
        if not data:
            print(f"[VanillaRAGAgent] No memory file found at {path}", flush=True)
            return

        cfgd = data.get("cfg", {})
        for k in [
            "llm_name",
            "embed_name",
            "memory_retrieve_limit",
            "enable_reflection",
            "enable_summarization",
            "enable_short_term_memory",
            "short_term_memory_size",
        ]:
            if k in cfgd and cfgd[k] is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, cfgd[k])

        self.memory.load_from_dict(data.get("memory", {}))
        self.memory.rebuild_embeddings()

        self.short_term_memory_size = getattr(self.cfg, "short_term_memory_size", 5)

        if getattr(self.cfg, "enable_short_term_memory", False):
            self.short_term_memory = list(data.get("short_term_memory", []))
            if len(self.short_term_memory) > self.short_term_memory_size:
                self.short_term_memory = self.short_term_memory[-self.short_term_memory_size:]
            print(
                "[Memory] Loaded short-term memory: "
                f"size={len(self.short_term_memory)}",
                flush=True,
            )
        else:
            self.short_term_memory = []

@lru_cache(maxsize=None)
def create_vanilla_rag_agent(Agent: Type):
    class_name = (
        f"VanillaRAGAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (VanillaRAGAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )