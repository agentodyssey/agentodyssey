from __future__ import annotations
from typing import List, Optional, Dict, Type
from functools import lru_cache
import os
import json
import shutil

from mem0 import Memory as Mem0Memory

from providers.vllm import vllmLanguageModel
from providers.openai_api import OpenAILanguageModel
from agents.rag.rag_agent_config import RAGAgentConfig
from utils import atomic_write


class Memory:
    def __init__(self, config: dict):
        self.config = config
        self.mem0 = Mem0Memory.from_config(config)

    def add(self, info: str):
        self.mem0.add(info, agent_id="self")

    def retrieve(self, observation: str, memory_retrieve_limit: int) -> List[str]:
        relevant_memories = self.mem0.search(query=observation, agent_id="self", limit=memory_retrieve_limit)['results']
        return [m['memory'] for m in relevant_memories]


class Mem0RAGAgent:
    def __init__(self, id: str, name: str, cfg: Optional[RAGAgentConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = RAGAgentConfig(available_actions=self.available_actions)

        self.memory_paths = ["memory.jsonl"]
        self.vector_store_subdir = "mem0_store"

        # Create vector store directory and Memory
        os.makedirs(self.cfg.full_mem_path, exist_ok=True)
        self.memory = Memory(self.cfg.mem0_config)

        self._pending_memory_log: List[str] = []
        self.provider = self.cfg.get_vllm_provider()

        if self.provider == "openai":
            self.llm = OpenAILanguageModel(llm_name=self.cfg.llm_name)
        else:
            self.llm = vllmLanguageModel(
                llm_name=self.cfg.llm_name,
                endpoint=self.cfg.vllm_endpoint,
                port=self.cfg.vllm_port,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                presence_penalty=self.cfg.presence_penalty,
                max_new_tokens=self.cfg.max_new_tokens,
            )

    def _replay_jsonl(self, jsonl_path: str) -> None:
        """Fallback: replay JSONL through LLM (slow, only use if no vectors exist)."""
        if not os.path.exists(jsonl_path):
            return
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text", None)
                if isinstance(text, str) and text:
                    print("adding to text", flush=True)
                    self.memory.add(text)

    def memorize(self, info: str):
        self.memory.add(info)
        self._pending_memory_log.append(info)

    def _act(self, obs: Dict) -> str:
        obs_text = obs["text"]

        retrieved = self.memory.retrieve(
            obs_text,
            memory_retrieve_limit=self.cfg.memory_retrieve_limit,
        )

        lm_output = self.llm.generate(
            user_prompt=self.cfg.construct_user_prompt(retrieved, obs_text),
            system_prompt=self.cfg.system_prompt,
        )

        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]
        readable = self.cfg.format_response(parsed)

        self.memorize(f"{obs_text}\n{readable}")

        return (
            parsed_action,
            lm_output["num_input_tokens"],
            lm_output["num_output_tokens"],
            lm_output["response"],
        )

    def save_memory(self, full_memory_dir: str) -> None:
        os.makedirs(full_memory_dir, exist_ok=True)

        # Save JSONL as audit log (optional backup)
        jsonl_path = os.path.join(full_memory_dir, self.memory_paths[0])
        if self._pending_memory_log:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                for text in self._pending_memory_log:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            self._pending_memory_log.clear()

        # Note: FAISS vectors are auto-saved by mem0 to the configured path
        # No need to manually copy them

        meta_path = os.path.join(full_memory_dir, f"{self.id}.json")
        meta = {
            "agent_id": self.id,
            "agent_name": self.name,
            "cfg": {
                "llm_name": getattr(self.cfg, "llm_name", None),
                "embed_name": getattr(self.cfg, "embed_name", None),
                "memory_retrieve_limit": getattr(self.cfg, "memory_retrieve_limit", None),
                "vllm_endpoint": getattr(self.cfg, "vllm_endpoint", None),
                "vllm_port": getattr(self.cfg, "vllm_port", None),
                "temperature": getattr(self.cfg, "temperature", None),
                "top_p": getattr(self.cfg, "top_p", None),
                "presence_penalty": getattr(self.cfg, "presence_penalty", None),
                "max_new_tokens": getattr(self.cfg, "max_new_tokens", None),
            },
            "memory_jsonl": self.memory_paths[0],
            "vector_store_subdir": self.vector_store_subdir,
        }
        atomic_write(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

    def _vector_store_exists(self, full_memory_dir: str) -> bool:
        """Check if FAISS vector store files exist."""
        vector_store_path = os.path.join(full_memory_dir, self.vector_store_subdir)
        # FAISS saves as {collection_name}.faiss and {collection_name}.pkl
        # Your collection_name is "test" based on the config
        faiss_file = os.path.join(vector_store_path, "test.faiss")
        pkl_file = os.path.join(vector_store_path, "test.pkl")
        return os.path.exists(faiss_file) and os.path.exists(pkl_file)

    def load_memory(self, full_memory_dir: str) -> None:
        """
        Load memory from persistent FAISS vector store.
        
        If vectors exist, just reinitialize mem0 pointing to them (instant).
        Otherwise, fall back to slow JSONL replay.
        """
        if self._vector_store_exists(full_memory_dir):
            # Fast path: vectors already exist, just reinitialize Memory
            # Update the config to point to the correct path
            vector_store_path = os.path.join(full_memory_dir, self.vector_store_subdir)
            
            # Build config with the correct vector store path
            config = self.cfg.mem0_config.copy()
            config["vector_store"] = config["vector_store"].copy()
            config["vector_store"]["config"] = config["vector_store"]["config"].copy()
            config["vector_store"]["config"]["path"] = vector_store_path
            
            # Reinitialize memory - FAISS will load existing index automatically
            self.memory = Memory(config)
            print(f"Loaded existing FAISS vectors from {vector_store_path}", flush=True)
        else:
            # Slow fallback: replay JSONL through LLM
            print(f"No vectors found, falling back to JSONL replay (slow)...", flush=True)
            jsonl_path = os.path.join(full_memory_dir, self.memory_paths[0])
            self._replay_jsonl(jsonl_path)
        
        self._pending_memory_log.clear()


@lru_cache(maxsize=None)
def create_mem0_rag_agent(Agent: Type):
    class_name = (
        f"Mem0RAGAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (Mem0RAGAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )