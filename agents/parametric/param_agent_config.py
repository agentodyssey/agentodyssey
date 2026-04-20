from dataclasses import dataclass
from agents.llm_agent_config import LLMAgentConfig

@dataclass
class ParamAgentConfig(LLMAgentConfig):

    max_seq_len: int = 4096
    full_param_lr: float = 1e-6
    lr: float = 5e-6
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 1
    fp16: bool = True
    
    summarizer_llm_name: str = None  # default: use llm_name
    summarizer_temperature: float = 0.6
    summarizer_top_p: float = 0.95
    summarizer_max_new_tokens: int = 1024
    use_sfted_summarizer: bool = True

    seed: int = 666

    def __post_init__(self) -> None:
        # Run parent initialization (seed + system_prompt formatting)
        super().__post_init__()

    @property
    def lora_config(self) -> dict:
        return {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                # "gate_proj",
                # "up_proj",
                # "down_proj",
            ],
        }
    
    def construct_user_prompt(self, observation: str) -> str:
        user_prompt = f"My Current Observation: {observation}\n" + f"{self.action_prompt}"
        return user_prompt