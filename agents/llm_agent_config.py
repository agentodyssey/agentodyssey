import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
import torch
import json
from games.base.agent import Action

@dataclass
class LLMAgentConfig:
    llm_name: str = "Qwen/Qwen3-4B"
    llm_provider: str = None
    embed_name: str = "Qwen/Qwen3-Embedding-0.6B"
    max_new_tokens: int = 4096 # Do not set to None, as None lead to different behaviors in different model providers
    temperature: float = 0.7
    presence_penalty: float = 1.5
    top_p: float = 0.8
    action_prompt: str = "What should I do next? Return only the JSON object."
    available_actions: list[Action] = field(default_factory=list)

    torch_seed: int = 42

    enable_reflection: bool = False
    reflection_verbose: bool = True

    enable_summarization: bool = False
    summarizer_verbose: bool = True

    enable_short_term_memory: bool = False
    short_term_memory_size: int = 5

    full_mem_path: str = None 

    system_prompt: str = """
You are the player in a text adventure games. The world is described in text form.
At each turn, you may choose ONE action from the action space below.

Action space:
{}

Output format (STRICT):
Return a single JSON object with exactly these keys:
{{
  "reasoning": "A few sentences explaining why you choose the action.",
  "action": "<action>"
}}

Rules:
- The JSON must be the ONLY content in your reply (no extra text before/after).
- The action must exactly match one option from the action space.
"""

    def __post_init__(self) -> None:
        if self.torch_seed is None:
            self.torch_seed = int.from_bytes(os.urandom(4), "little")
            print("Setting a random torch seed: ", self.torch_seed)
        self.seed_torch(self.torch_seed)
        
        if self.available_actions:
            actions_formatted = [f"- {action.verb} " + " ".join(f"<{p}>" for p in action.params) for action in self.available_actions]
            actions_formatted_str = "\n".join(actions_formatted)
            self.system_prompt = self.system_prompt.format(actions_formatted_str)
            # print("\nSystem prompt:\n" + self.system_prompt)
    
    @staticmethod
    def seed_torch(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def json_parser(self, text: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(cleaned)
            return parsed
        except Exception as e:
            # print(f"Error parsing JSON: {e}")
            pass
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            parsed = None
        else:
            json_str = match.group(0)
            try:
                parsed = json.loads(json_str)
            except Exception:
                parsed = None
        return parsed



    def response_parser(self, text: str) -> dict:
        parsed = self.json_parser(text)
        if parsed is None:
            return {"reasoning": "Failed to parse JSON.", "action": "wait"}

        if not {"reasoning", "action"}.issubset(parsed.keys()):
            return {"reasoning": "Missing required keys.", "action": "wait"}

        if type(parsed["action"]) is not str:
            return {"reasoning": "Action is not a string.", "action": "wait"}

        reasoning = parsed.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return {
            "action": parsed["action"].strip(),
            "reasoning": reasoning.strip(),
        }


    def format_response(self, parsed: dict) -> str:
        reasoning = parsed.get("reasoning", "").strip()
        action = parsed.get("action", "wait").strip()

        if reasoning:
            return f"Reasoning: {reasoning}\nAction: {action}"
        return f"Action: {action}"


    
    def get_llm(self, llm_name: str = None, llm_provider: str = None):
        llm_name = llm_name if llm_name is not None else self.llm_name
        llm_provider = llm_provider if llm_provider is not None else self.llm_provider

        if llm_provider is None:
            raise ValueError("LLM provider must be specified either in the config or as an argument.")
        if llm_name is None:
            raise ValueError("LLM name must be specified either in the config or as an argument.")

        if llm_provider == "openai":
            from providers.openai_api import OpenAILanguageModel
            return OpenAILanguageModel(
                llm_name=llm_name,
                max_new_tokens=self.max_new_tokens,
            )
        elif llm_provider == "azure":
            from providers.azure_api import AzureLanguageModel
            return AzureLanguageModel(
                llm_name=llm_name,
                max_new_tokens=self.max_new_tokens,
            )
        elif llm_provider == "azure_openai":
            from providers.azure_openai_api import AzureOpenAILanguageModel
            return AzureOpenAILanguageModel(
                llm_name=llm_name,
                max_new_tokens=self.max_new_tokens,
            )
        elif llm_provider == "vllm":
            from providers.vllm import vllmLanguageModel
            return vllmLanguageModel(
                llm_name=llm_name,
                endpoint=os.environ.get("VLLM_ENDPOINT", "http://localhost"),
                port=os.environ.get("VLLM_PORT", "8088"),
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                max_new_tokens=self.max_new_tokens,
            )
        elif llm_provider == "claude":
            from providers.claude_api import ClaudeLanguageModel
            return ClaudeLanguageModel(
                llm_name=llm_name,
                max_new_tokens=self.max_new_tokens,
            )
        elif llm_provider == "gemini":
            from providers.gemini_api import GeminiLanguageModel
            return GeminiLanguageModel(
                llm_name=llm_name,
                max_new_tokens=self.max_new_tokens,
            )
        elif llm_provider == "huggingface":
            from providers.huggingface import hfLanguageModel
            return hfLanguageModel(
                llm_name=llm_name,
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                max_new_tokens=self.max_new_tokens,
                device=self.get_device(),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def get_embedder(self, embed_name: str = None):
        from providers.huggingface import hfEmbeddingModel
        embed_name = embed_name if embed_name is not None else self.embed_name
        return hfEmbeddingModel(
            embedder_name=self.embed_name,
            device=self.get_device(),
        )
    
    def reflect(self, llm, obs_text: str, retrieved: str) -> Optional[str]:
        retrieved_text = (retrieved or "").strip()
        if not retrieved_text:
            return None

        reflection_prompt = (
            "Current observation:\n"
            f"{obs_text}\n\n"
            "Past memory:\n"
            f"{retrieved_text}\n\n"
            "Do NOT restate, paraphrase, or summarize the observation or past memory. "
            "Analyze your current situation and past experience. "
            "Identify any suboptimal strategies and failure patterns. "
            "Explain why those issues occurred. "
            "Derive concrete lessons or decision rules that should guide your future behavior. "
            "Focus on improving both long-term and short-term strategy."
        )

        lm_output = llm.generate(
            user_prompt=reflection_prompt,
            system_prompt="""
Reflect on the current observation and past memory. 

Output format (STRICT):
Return a single JSON object with exactly this key:
{
  "reflection": "<reflection>",
}

Rules:
- The JSON must be the ONLY content in your reply (no extra text before/after).
""",
        )
        parsed = self.json_parser(lm_output["response"])
        if parsed is None or "reflection" not in parsed:
            return ""
        if self.reflection_verbose:
            print("Reflection:", parsed["reflection"], flush=True)
        return parsed["reflection"]

    def summarize(self, llm, text: str) -> str:
        lm_output = llm.generate(
            user_prompt=text,
            system_prompt="""
Summarize the text to distill the key information.

Output format (STRICT):
Return a single JSON object with exactly this key:
{
  "summary": "<summary>",
}
Rules:
- The JSON must be the ONLY content in your reply (no extra text before/after).
""",
        )
        parsed = self.json_parser(lm_output["response"])
        if parsed is None or "summary" not in parsed:
            return ""
        if self.summarizer_verbose:
            print("Summarization:", parsed["summary"], flush=True)
        return parsed["summary"]