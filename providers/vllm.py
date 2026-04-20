import os
import sys
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
import logging
import torch
import requests
import subprocess
import time
import warnings
from openai import OpenAI
from termcolor import colored
from tools.logger import get_logger


def _test_vllm_response(endpoint: str, port: str, llm_name: str, timeout: int, message: str = "ping", max_new_tokens: int = 1):
    payload = {
        "model": llm_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    url = f"{endpoint}:{port}/v1/chat/completions"
    r = requests.post(url, json=payload, timeout=timeout)
    return r

def _start_vllm(llm_name: str, endpoint: str, port: str, timeout: int = 10, logger: logging.Logger = None):
    subprocess.Popen(
        [
            "vllm", "serve", llm_name,
            "--port", str(port),
        ],
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.STDOUT,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    while True:
        try:
            r = requests.get(f"{endpoint}:{port}/v1/models", timeout=timeout)
            if r.status_code == 200:
                logger.info("✅ vLLM is now running.")
                break
        except Exception:
            pass
        time.sleep(5)
    
    while True:
        try:
            r = _test_vllm_response(endpoint, port, llm_name, timeout=timeout)
            if r.status_code == 200:
                logger.info(f"✅ vLLM model {llm_name} is fully loaded and successfully tested.")
                break
        except Exception:
            pass
        logger.warning("⏳ Waiting for model weights to load...")
        time.sleep(5)

class vllmLanguageModel:
    def __init__(
        self,
        llm_name: str,
        endpoint: str,
        port: str,
        temperature: float,
        top_p: float,
        presence_penalty: float,
        max_new_tokens: int = None,
    ):
        self.llm_name = llm_name
        self.logger = get_logger("vllmLanguageModelLogger")

        try:
            r1 = requests.get(f"{endpoint}:{port}/v1/models", timeout=3)
            r2 = _test_vllm_response(endpoint, port, llm_name, timeout=3)
            if r1.status_code == 200 and r2.status_code == 200:
                self.logger.info(f"✅ vLLM already running and model {llm_name} is fully loaded.")
            else:
                self.logger.warning("❌ vLLM is not running.")
                self.logger.info("🚀 Restarting vLLM automatically...")
                _start_vllm(llm_name, endpoint, port, logger=self.logger)
        except requests.exceptions.ConnectionError:
            self.logger.warning("❌ vLLM is not running.")
            self.logger.info("🚀 Restarting vLLM automatically...")
            _start_vllm(llm_name, endpoint, port, logger=self.logger)
        except Exception as e:
            self.logger.error(f"❌ Error probing vLLM server: {e}")
            raise

        self.client = OpenAI(
            base_url=f"{endpoint}:{port}/v1",
            api_key="-",
        )
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.max_new_tokens = max_new_tokens

    def generate(self, user_prompt, system_prompt: str = None) -> str:
        try:
            if system_prompt == None:
                messages = [
                    {'role': 'user', 'content': user_prompt}
                ]
            else:
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]

            res = self.client.chat.completions.create(
                messages=messages,
                model=self.llm_name,
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                max_new_tokens=self.max_new_tokens
            )
            message = res.choices[0].message
            num_input_tokens = getattr(res.usage, "prompt_tokens", 0)
            num_output_tokens = getattr(res.usage, "completion_tokens", 0)
            return {
                "response": message.content,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens
            }
        except Exception as e:
            self.logger.error(f"❌ Error generating response for model={self.llm_name}: {e}")
            return {
                "response": None,
                "num_input_tokens": 0,
                "num_output_tokens": 0
            }
        
if __name__ == "__main__":
    model = vllmLanguageModel(
        llm_name="Qwen/Qwen3-8B",
        endpoint="http://localhost",
        port="8088",
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        presence_penalty=1.5,
    )
    
    prompt = "How are you?"
    response = model.generate(user_prompt=prompt)
    print("Response:", response)

    print(colored("\nThe vllm server will keep running in the background. You can stop it by first running `ps aux | grep vllm` and `nvidia-smi` to check the vllm processes and then killing the process by `kill -9 <PID>` for both CPU process and GPU process.", "yellow"))