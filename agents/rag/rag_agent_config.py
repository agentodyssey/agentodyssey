from dataclasses import dataclass
from typing import List, Optional, Literal
from agents.llm_agent_config import LLMAgentConfig
import os

@dataclass
class RAGAgentConfig(LLMAgentConfig):
    vllm_endpoint: str = "http://localhost"
    vllm_port: str = "8088"
    memory_retrieve_limit: int = 5

    def get_vllm_provider(self) -> Literal["openai", "huggingface", "vllm"]:
        """Override to default to vllm instead of huggingface."""
        name = self.llm_name.lower()
        if name.startswith("gpt"):
            return "openai"
        return "vllm"


    @property
    def mem0_config(self):
        provider = self.get_vllm_provider()
        
        if provider == "openai":
            llm_config = {
                "provider": "openai",
                "config": {
                    "model": self.llm_name,
                },
            }
        else:  # vllm
            llm_config = {
                "provider": "vllm",
                "config": {
                    "model": self.llm_name,
                    "vllm_base_url": f"{self.vllm_endpoint}:{self.vllm_port}/v1",
                },
            }
        
        vector_store_path = os.path.join(self.full_mem_path, "mem0_store")
        
        return {
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": "test",
                    "path": vector_store_path,
                    "embedding_model_dims": 1024,
                    "distance_strategy": "cosine"
                }
            }, 
            "llm": llm_config,
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": self.embed_name,
                },
            },
        }

    def construct_user_prompt(self, retrieved: List[str], observation: str) -> str:
        memory_prompt = f"My Memories:\n" + "\n".join(retrieved) + "\n" if retrieved else ""
        user_prompt = memory_prompt + f"\n\nMy Current Observation: {observation}\n" + f"{self.action_prompt}"
        return user_prompt
    
    def construct_raptor_user_prompt(self, retrieved: str, observation: str) -> str:
        memory_prompt = "My Memories: " + retrieved if retrieved else ""
        user_prompt = memory_prompt + f"\n\nMy Current Observation: {observation}\n" + f"{self.action_prompt}"
        return user_prompt

    def construct_user_prompt_with_current_reflection(
        self,
        retrieved_memories: List[str],
        observation: str,
        current_reflection: Optional[str] = None,
    ) -> str:
        parts = []

        if retrieved_memories:

            parts.append("My Memories: ".join(retrieved_memories))

        parts.append(f"My Current Observation: {observation}\n")

        if current_reflection:
            parts.append(f"{current_reflection}\n")

        parts.append(self.action_prompt)
        return "\n\n".join(parts) + "\n"

