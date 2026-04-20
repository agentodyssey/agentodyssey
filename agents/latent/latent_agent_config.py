from dataclasses import dataclass
from agents.llm_agent_config import LLMAgentConfig

@dataclass
class LatentAgentConfig(LLMAgentConfig):
    
    def construct_user_prompt(self, observation: str) -> str:
        user_prompt = f"My Current Observation: {observation}\n" + f"{self.action_prompt}"
        return user_prompt