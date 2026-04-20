from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Type
from functools import lru_cache
import os
import json

from utils import atomic_write
from agents.rag.rag_agent_config import RAGAgentConfig
from agents.rag.raptor import BaseSummarizationModel
from agents.rag.raptor import BaseQAModel
from agents.rag.raptor import BaseEmbeddingModel
from agents.rag.raptor import RetrievalAugmentation, RetrievalAugmentationConfig

class Memory:
    def __init__(self, embedder, summarizer):
        self.embedder = embedder
        self.summarizer = summarizer
        self.mem: str = ""
        self.RA = self._create_ra()
        self.RA.add_documents(self.mem)

    def _create_ra(self):
        embedder_model = self.embedder
        summarizer_model = self.summarizer

        class CustomSummarizationModel(BaseSummarizationModel):
            def __init__(self):
                self.llm = summarizer_model

            def summarize(self, context: str):
                user_prompt = f"Write a summary of the following, including as many key details as possible: {context}:"
                lm_output = self.llm.generate(user_prompt=user_prompt, system_prompt="You are a helpful assistant.", think=False)
                return lm_output['response']

        class DummyQAModel(BaseQAModel):
            def __init__(self):
                pass

            def answer_question(self, context: str, question: str):
                return "dummy"

        class CustomEmbeddingModel(BaseEmbeddingModel):
            def __init__(self):
                self.embedder = embedder_model

            def create_embedding(self, text: str):
                return self.embedder.encode(text)

        custom_config = RetrievalAugmentationConfig(
            summarization_model=CustomSummarizationModel(),
            embedding_model=CustomEmbeddingModel(),
            qa_model=DummyQAModel()
        )
        return RetrievalAugmentation(config=custom_config)

    def add(self, info: str):
        self.mem += info + "\n"
        self.RA.add_documents(self.mem)

    def retrieve(self, observation: str, memory_retrieve_limit: int) -> str:
        return self.RA.retrieve(observation, top_k=memory_retrieve_limit)[0]

    def to_dict(self) -> dict:
        return {"mem": self.mem}

    def load_from_dict(self, d: dict) -> None:
        self.mem = d.get("mem", "")

    def rebuild(self) -> None:
        self.RA = self._create_ra()
        self.RA.add_documents("")
        if not self.mem:
            return
        print(f"[Memory] Rebuilding RAPTOR tree, mem_len={len(self.mem)}", flush=True)
        self.RA.add_documents(self.mem)

    def save(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def load(self, path: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class RaptorRAGAgent:
    def __init__(self, id: str, name: str, cfg: Optional[RAGAgentConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = RAGAgentConfig(available_actions=self.available_actions)
        self.embedder = self.cfg.get_embedder()
        self.summarizer = self.cfg.get_llm()
        self.memory = Memory(embedder=self.embedder, summarizer=self.summarizer)
        self.llm = self.cfg.get_llm()
        self.memory_paths = ["memory.json"]

    def memorize(self, info: str):
        self.memory.add(info)

    def _act(self, obs: Dict) -> str:
        obs_text = obs["text"]
        retrieved = self.memory.retrieve(obs_text, memory_retrieve_limit=self.cfg.memory_retrieve_limit)

        lm_output = self.llm.generate(
            user_prompt=self.cfg.construct_raptor_user_prompt(retrieved, obs_text),
            system_prompt=self.cfg.system_prompt,
        )

        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]

        readable = self.cfg.format_response(parsed)
        to_store = f"{obs_text}\n{readable}"
        self.memorize(to_store)

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
            },
            "memory": self.memory.to_dict(),
        }
        self.memory.save(path, payload)

    def load_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        data = self.memory.load(path)
        if not data:
            print(f"[RaptorRAGAgent] No memory file found at {path}", flush=True)
            return

        cfgd = data.get("cfg", {})
        for k in ["llm_name", "embed_name", "memory_retrieve_limit"]:
            if k in cfgd and cfgd[k] is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, cfgd[k])

        self.memory.load_from_dict(data.get("memory", {}))
        self.memory.rebuild()


@lru_cache(maxsize=None)
def create_raptor_rag_agent(Agent: Type):
    class_name = (
        f"RaptorRAGAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (RaptorRAGAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )