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
        responses_api: bool = False,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        chat_history = (history or []) + (messages or [])
        response_: Dict[str, Any] = {}

        if responses_api:
            reasoning_config = {"summary": reasoning_summary} if reasoning_summary else {}
            if reasoning_effort:
                reasoning_config["effort"] = reasoning_effort
            prompt_kwargs = dict(
                model=model,
                input=chat_history,
                reasoning=reasoning_config,
            )
            if max_new_tokens:
                prompt_kwargs["max_output_tokens"] = max_new_tokens
            response = self.client.responses.create(**prompt_kwargs)
            response_["num_input_tokens"] = response.usage.input_tokens
            response_["num_output_tokens"] = response.usage.output_tokens
            reasoning_summary_items = [item for item in response.output if item.type == "reasoning"]
            if reasoning_summary_items and reasoning_summary_items[0].summary:
                response_["reasoning_summary"] = [summary.text for summary in reasoning_summary_items[0].summary]
            else:
                response_["reasoning_summary"] = None
            message_items = [item for item in response.output if item.type == "message"]
            assert message_items and message_items[0].content
            response_["response"] = message_items[0].content[0].text
        else:
            chat_kwargs = dict(
                model=model,
                messages=chat_history,
            )
            if max_new_tokens is not None:
                chat_kwargs["max_completion_tokens"] = max_new_tokens
            response = self.client.chat.completions.create(**chat_kwargs)
            response_["num_input_tokens"] = getattr(response.usage, "prompt_tokens", None)
            response_["num_output_tokens"] = getattr(response.usage, "completion_tokens", None)
            response_["response"] = response.choices[0].message.content
            response_["reasoning_summary"] = None

        chat_history.append({"role": "assistant", "content": response_["response"]})
        response_["history"] = chat_history
        return response_

class AzureOpenAILanguageModel:
    def __init__(
        self,
        llm_name: str,
        max_new_tokens: int = None,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        responses_api: Optional[bool] = None,
    ):
        api_credentials = get_azure_openai_api_credentials()
        self.client = AzureOpenAIClient(*api_credentials)
        self.llm_name = llm_name
        self.type = "azure_openai"
        self.max_new_tokens = max_new_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        # Azure deployment names are user-defined, so allow explicit override.
        # Default to the responses API for gpt-5 family deployments.
        self.responses_api = (
            responses_api if responses_api is not None else llm_name.startswith("gpt-5")
        )

    def generate(self, user_prompt: str, system_prompt: str = None, history=None, think: bool = False):
        prompt = []
        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": user_prompt})

        return self.client.run_prompt(
            model=self.llm_name,
            messages=prompt,
            max_new_tokens=self.max_new_tokens,
            responses_api=self.responses_api,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            history=history,
        )


if __name__ == "__main__":
    model = AzureOpenAILanguageModel(
        llm_name="gpt-5",
        max_new_tokens=16384,
        reasoning_effort="medium",
        reasoning_summary="auto",
    )

    result = model.generate(
        user_prompt="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        system_prompt="You are a helpful assistant.",
    )

    print(result)
