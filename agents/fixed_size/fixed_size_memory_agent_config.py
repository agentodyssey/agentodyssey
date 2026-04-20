from dataclasses import dataclass
from agents.llm_agent_config import LLMAgentConfig

@dataclass
class FixedSizeMemoryAgentConfig(LLMAgentConfig):
    max_memory_tokens: int = 512 # for Constant Context Agent

    def construct_user_prompt_constant_context(self, memory: str, observation: str) -> str:
        instruction_prompt = (
            "I am an agent with constant memory.\n"
            "Given the consolidated state and the new observation:\n"
            "1) Update <IS> with a concise cumulative summary of essentials only.\n"
            f"2) {self.action_prompt}\n"
            "Output exactly in this order:\n"
            "<IS>...</IS>\n"
            "Action: <action>\n"
        )
        memory_prompt = "My Memories: " + memory if memory else ""
        return memory_prompt + f"\nMy Current Observation: {observation}\n" + instruction_prompt