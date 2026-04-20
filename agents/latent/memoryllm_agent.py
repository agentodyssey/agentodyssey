from __future__ import annotations
from typing import Optional, Dict, Type, List
from functools import lru_cache
import os
import json

import torch
from transformers import AutoTokenizer
from agents.latent.modeling_memoryllm import MemoryLLM
from agents.latent.latent_agent_config import LatentAgentConfig

from utils import atomic_write


class MemoryLLMAgent:
    def __init__(self, id: str, name: str, cfg: Optional[LatentAgentConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = LatentAgentConfig(available_actions=self.available_actions)
        
        # Changed to MemoryLLM with chat model
        self.model = MemoryLLM.from_pretrained(
            "YuWangX/memoryllm-8b-chat",
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            device_map="cuda",
        )
        self.tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b-chat")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # MemoryLLM-8b-chat uses Llama3 chat template, so we can use the built-in one
        # or keep custom template if needed
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|'+message['role']+'|>' }}\n{{ message['content'] }}\n"
            "{% endfor %}"
            "<|assistant|>\n"
        )

        self.model = self.model.to(torch.bfloat16)
        self.model = self.model.cuda()

        # Track memorized text for checkpointing
        self.memory_log: List[str] = []

        self.memory_paths = ["memory.json"]

    def _inject_to_model(self, info: str):
        """Inject into model memory without logging."""
        self.model.inject_memory(
            self.tokenizer(info, return_tensors="pt", add_special_tokens=False).input_ids.to(
                self.device
            ),
            update_memory=True,
        )

    def memorize(self, info: str):
        self.memory_log.append(info)
        self._inject_to_model(info)

    def _act(self, obs: Dict) -> str:
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": self.cfg.construct_user_prompt(obs["text"])},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)[:, 1:].to(self.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|user|>"),
            self.tokenizer.convert_tokens_to_ids("|>"),
        ]
        terminators = [t for t in terminators if t is not None]

        outputs = self.model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=300,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True, 
            temperature=0.7, 
            top_p=0.8, 
            repetition_penalty=1.5,
        )
        output_tokens = outputs[0][inputs.shape[1]:]
        lm_output = {} 
        lm_output["response"] = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        lm_output["num_input_tokens"] = inputs.shape[1]
        lm_output["num_output_tokens"] = len(output_tokens)
        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]
        readable = self.cfg.format_response(parsed)
        self.memorize(f"{obs['text']}\n{readable}\n")
        return (
            parsed_action,
            lm_output["num_input_tokens"],
            lm_output["num_output_tokens"],
            lm_output["response"],
        )

    # REVISED: LongContextAgent-style signature
    def save_memory(self, full_memory_dir: str) -> None:
        os.makedirs(full_memory_dir, exist_ok=True)
        path = os.path.join(full_memory_dir, self.memory_paths[0])

        data = {
            "agent_id": self.id,
            "agent_name": self.name,
            "cfg": {
                "max_new_tokens": getattr(self.cfg, "max_new_tokens", None),
            },
            "memory": {"memory_log": list(self.memory_log)},
        }

        atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))

    # REVISED: LongContextAgent-style signature
    def load_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfgd = data.get("cfg", {}) or {}
        for k in ["max_new_tokens"]:
            if k in cfgd and cfgd[k] is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, cfgd[k])

        memd = data.get("memory", {}) or {}
        saved_log = list(memd.get("memory_log", []) or [])
        self.memory_log = saved_log

        # Replay memory into MemoryLLM
        for info in saved_log:
            self._inject_to_model(info)

@lru_cache(maxsize=None)
def create_memoryllm_agent(Agent: Type):
    class_name = (
        f"MemoryLLMAgent__"  # Updated prefix
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (MemoryLLMAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )