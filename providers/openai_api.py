from openai import OpenAI
from typing import List, Dict, Optional
import os


def get_openai_api_credentials():
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables.")
        return [api_key]
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def run_prompt(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        responses_api: bool = False,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, any]:
        chat_history = history + (messages or []) if history else (messages or [])
        # print(f"Chat History: {chat_history}")
        response_ = {}
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
                chat_kwargs["max_new_tokens"] = max_new_tokens
            response = self.client.chat.completions.create(**chat_kwargs)
            response_["num_input_tokens"] = response.usage.prompt_tokens
            response_["num_output_tokens"] = response.usage.completion_tokens
            response_["response"] = response.choices[0].message.content
            response_["reasoning_summary"] = None
        chat_history.append({"role": "assistant", "content": response_["response"]})
        response_["history"] = chat_history
        return response_

class OpenAILanguageModel:
    def __init__(self, 
                 llm_name: str, 
                 max_new_tokens: int = None):
        api_credentials = get_openai_api_credentials()
        self.client = OpenAIClient(*api_credentials)
        self.llm_name = llm_name
        self.type = "openai"
        self.responses_api = llm_name.startswith("gpt-5")
        self.max_new_tokens = max_new_tokens

    def generate(self, user_prompt: str, system_prompt: str = None, history=None, think: bool = True):
        # void the think parameter for OpenAI models as it is model-specific
        prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}] if system_prompt else [{"role": "user", "content": user_prompt}]
        response = self.client.run_prompt(
            model=self.llm_name,
            messages=prompt,
            max_new_tokens=self.max_new_tokens,
            responses_api=self.responses_api,
            history=history,
        )
        return response
    
if __name__ == "__main__":
    model = OpenAILanguageModel(
        llm_name="gpt-5-mini",
        max_new_tokens=256,
    )
    result = model.generate(
        user_prompt="What is 1 + 1?",
        system_prompt="You are a helpful assistant.",
    )
    print(f"Response: {result['response']}")
    print(f"Input tokens: {result['num_input_tokens']}")
    print(f"Output tokens: {result['num_output_tokens']}")