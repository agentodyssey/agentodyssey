from openai import OpenAI
from typing import List, Dict, Optional, Any
import os


def get_azure_api_credentials():
    try:
        endpoint = os.environ.get("AZURE_ENDPOINT")
        if not endpoint:
            raise ValueError("Missing AZURE_ENDPOINT in environment variables.")
        api_key = os.environ.get("AZURE_API_KEY")
        if not api_key:
            raise ValueError("Missing AZURE_API_KEY in environment variables.")
        return [endpoint, api_key]
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

class AzureClient:
    def __init__(self, endpoint: str, api_key: str):
        self.client = OpenAI(
            base_url=f"{endpoint}",
            api_key=api_key
        )

    def run_prompt(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        chat_history = (history or []) + (messages or [])

        kwargs = {
            "model": model,
            "messages": chat_history,
        }

        if max_new_tokens is not None:
            kwargs["max_completion_tokens"] = max_new_tokens

        response = self.client.chat.completions.create(**kwargs)

        result = {
            "response": response.choices[0].message.content,
            "num_input_tokens": getattr(response.usage, "prompt_tokens", None),
            "num_output_tokens": getattr(response.usage, "completion_tokens", None)
        }

        chat_history.append({"role": "assistant", "content": response.choices[0].message.content})
        result["history"] = chat_history
        return result

class AzureLanguageModel:
    def __init__(self, llm_name: str, max_new_tokens: int = None):
        api_credentials = get_azure_api_credentials()
        self.client = AzureClient(*api_credentials)
        self.llm_name = llm_name
        self.type = "azure"
        self.max_new_tokens = max_new_tokens

    def generate(self, user_prompt: str, system_prompt: str = None, history=None, think: bool = False):
        prompt = []
        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": user_prompt})

        return self.client.run_prompt(
            model=self.llm_name,
            messages=prompt,
            max_new_tokens=self.max_new_tokens,
            history=history,
        )


if __name__ == "__main__":
    model = AzureLanguageModel(
        llm_name="grok-4-1-fast-reasoning",
        max_new_tokens=256,
    )

    result = model.generate(
        user_prompt="What is 1 + 1?",
        system_prompt="You are a helpful assistant.",
    )

    print(result)