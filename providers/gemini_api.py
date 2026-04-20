from google import genai
from typing import List, Dict, Optional, Any
import os


def get_gemini_api_credentials():
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment variables.")
        return api_key
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def run_prompt(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        chat_history = (history or []) + (messages or [])

        gemini_contents = []
        for msg in chat_history:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append(
                genai.types.Content(
                    role=role,
                    parts=[genai.types.Part(text=msg["content"])],
                )
            )

        config = genai.types.GenerateContentConfig(
            max_output_tokens=max_new_tokens or 1024,
            system_instruction=system_prompt,
        )

        response = self.client.models.generate_content(
            model=model,
            contents=gemini_contents,
            config=config,
        )

        result = {
            "response": response.text,
            "num_input_tokens": response.usage_metadata.prompt_token_count,
            "num_output_tokens": response.usage_metadata.candidates_token_count,
        }

        chat_history.append({"role": "assistant", "content": response.text})
        result["history"] = chat_history
        return result


class GeminiLanguageModel:
    def __init__(self, llm_name: str, max_new_tokens: int = None):
        api_key = get_gemini_api_credentials()
        self.client = GeminiClient(api_key)
        self.llm_name = llm_name
        self.type = "gemini"
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
    model = GeminiLanguageModel(
        llm_name="gemini-3.1-pro-preview",
        max_new_tokens=256,
    )

    result = model.generate(
        user_prompt="What is 1 + 1?",
        system_prompt="You are a helpful assistant.",
    )

    print(result)
