import anthropic
from typing import List, Dict, Optional, Any
import os


def get_claude_api_credentials():
    try:
        api_key = os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("Missing CLAUDE_API_KEY in environment variables.")
        return api_key
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


class ClaudeClient:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def run_prompt(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        chat_history = (history or []) + (messages or [])

        kwargs = {
            "model": model,
            "messages": chat_history,
            "max_tokens": max_new_tokens or 1024,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        result = {
            "response": response.content[0].text,
            "num_input_tokens": response.usage.input_tokens,
            "num_output_tokens": response.usage.output_tokens,
        }

        chat_history.append({"role": "assistant", "content": response.content[0].text})
        result["history"] = chat_history
        return result


class ClaudeLanguageModel:
    def __init__(self, llm_name: str, max_new_tokens: int = None):
        api_key = get_claude_api_credentials()
        self.client = ClaudeClient(api_key)
        self.llm_name = llm_name
        self.type = "claude"
        self.max_new_tokens = max_new_tokens

    def generate(self, user_prompt: str, system_prompt: str = None, history=None, think: bool = False):
        prompt = [{"role": "user", "content": user_prompt}]

        return self.client.run_prompt(
            model=self.llm_name,
            messages=prompt,
            max_new_tokens=self.max_new_tokens,
            history=history,
            system_prompt=system_prompt,
        )


if __name__ == "__main__":
    model = ClaudeLanguageModel(
        llm_name="claude-sonnet-4-6",
        max_new_tokens=256,
    )

    result = model.generate(
        user_prompt="What is 1 + 1?",
        system_prompt="You are a helpful assistant.",
    )

    print(result)
