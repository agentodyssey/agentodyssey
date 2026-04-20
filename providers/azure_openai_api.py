from openai import AzureOpenAI
from typing import List, Dict, Optional, Any
import os


def get_azure_openai_api_credentials():
    try:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT in environment variables.")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing AZURE_OPENAI_API_KEY in environment variables.")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        if not api_version:
            raise ValueError("Missing AZURE_OPENAI_API_VERSION in environment variables.")
        return [endpoint, api_key, api_version]
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

class AzureOpenAIClient:
    def __init__(self, endpoint: str, api_key: str, api_version: str):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
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

class AzureOpenAILanguageModel:
    def __init__(self, llm_name: str, max_new_tokens: int = None):
        api_credentials = get_azure_openai_api_credentials()
        self.client = AzureOpenAIClient(*api_credentials)
        self.llm_name = llm_name
        self.type = "azure_openai"
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
    model = AzureOpenAILanguageModel(
        llm_name="gpt-5",
        max_new_tokens=16384,
    )

    result = model.generate(
        user_prompt="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        system_prompt="You are a helpful assistant.",
    )

    print(result)
