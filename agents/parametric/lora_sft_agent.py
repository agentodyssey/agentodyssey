from __future__ import annotations
from typing import List, Optional, Dict, Any, Type
from functools import lru_cache
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from agents.parametric.param_agent_config import ParamAgentConfig

from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from utils import atomic_write


class _CLMDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, texts: List[str], max_len: int):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(
            t,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _collate_pad(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        if k == "labels":
            pad_val = -100
        elif k == "input_ids":
            pad_val = pad_id
        else:
            pad_val = 0
        out[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=pad_val)
    return out


class TrainableLM:
    def __init__(self, base, lora_config: Optional[Dict[str, Any]] = None):
        self.base = base
        self.model = base.model
        self.tokenizer = base.tokenizer
        self.device = base.device
        self.lora_config = lora_config or {}
        self._peft: Optional[PeftModel] = None

    def _ensure_peft(self):
        if self._peft is not None:
            return self._peft
        cfg = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["alpha"],
            lora_dropout=self.lora_config["dropout"],
            target_modules=self.lora_config["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._peft = get_peft_model(self.model, cfg)
        self.base.model = self._peft
        return self._peft

    def train_on_texts(
        self,
        texts: List[str],
        max_seq_len: int = 1024,
        lr: float = 1e-5,
        epochs: int = 1,
        batch_size: int = 2,
        grad_accum_steps: int = 2,
        fp16: bool = True,
    ) -> int:
        if not texts:
            return 0
        peft_m = self._ensure_peft()
        peft_m.train()
        ds = _CLMDataset(self.tokenizer, texts, max_seq_len)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: _collate_pad(
                b, self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            ),
        )
        opt = AdamW(peft_m.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=(fp16 and torch.cuda.is_available()))
        total_steps = 0
        for _ in range(epochs):
            epoch_loss = []
            for batch in dl:
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                with torch.amp.autocast("cuda", enabled=(fp16 and torch.cuda.is_available())):
                    out = peft_m(**batch)
                    loss = out.loss / grad_accum_steps
                    epoch_loss.append(loss.item())
                scaler.scale(loss).backward()
                if (total_steps + 1) % grad_accum_steps == 0:
                    scaler.unscale_(opt)  # unscale before clipping
                    torch.nn.utils.clip_grad_norm_(peft_m.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                total_steps += 1
            if epoch_loss:
                print(f"Epoch Avg Loss: {sum(epoch_loss) / len(epoch_loss):.4f}", flush=True)
        peft_m.eval()
        return total_steps

    def save_adapter(self, path: str):
        if self._peft is None:
            return
        os.makedirs(path, exist_ok=True)
        self._peft.save_pretrained(path)

    def load_adapter(self, path: str):
        self._peft = PeftModel.from_pretrained(self.model, path, is_trainable=True)
        self.base.model = self._peft
        self._peft.eval()

    @torch.inference_mode()
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        return self.base.generate(user_prompt=user_prompt, system_prompt=system_prompt)


class LoRASFTAgent:
    def __init__(
        self,
        id: str,
        name: str,
        cfg: Optional[ParamAgentConfig] = None,
        train_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = ParamAgentConfig(available_actions=self.available_actions)

        self.llm = self.cfg.get_llm()
        self.tlm = TrainableLM(self.llm, self.cfg.lora_config)

        self.enable_short_term_memory = getattr(self.cfg, "enable_short_term_memory", False)
        print(
            f"[LoRASFTAgent] Short-term memory enabled = {self.enable_short_term_memory}",
            flush=True,
        )
        self.enable_reflection = getattr(self.cfg, "enable_reflection", False)
        print(
            f"[LoRASFTAgent] Reflection enabled = {self.enable_reflection}",
            flush=True,
        )

        # Persistent short-term memory: used for reflection + saved to disk
        self.short_term_memory: List[str] = []

        # Hyperparams cached locally (refresh after load_memory)
        self.max_seq_len = self.cfg.max_seq_len
        self.lr = self.cfg.lr
        self.epochs = self.cfg.epochs
        self.batch_size = self.cfg.batch_size
        self.grad_accum = self.cfg.grad_accum
        self.fp16 = self.cfg.fp16

        # Fixed short-term memory size (1 if disabled, else cfg.short_term_memory_size)
        self.short_term_memory_size = self._get_short_term_memory_size()
        self.memories_since_train: int = 0

        self.steps_trained_total: int = 0

        self.memory_paths = ["memory.json"]
        self.adapter_subdir = "lora"

    def _get_short_term_memory_size(self) -> int:
        if self.enable_short_term_memory:
            size = getattr(self.cfg, "short_term_memory_size", 1)
        else:
            size = 1
        return max(1, int(size))

    def memorize(self, info: str) -> int:
        # Store in fixed-size short-term memory (size=1 if disabled).
        self.short_term_memory.append(info)
        if len(self.short_term_memory) > self.short_term_memory_size:
            self.short_term_memory = self.short_term_memory[-self.short_term_memory_size :]
        print("size of the short term memory:", len(self.short_term_memory))

        trained_steps = 0
        self.memories_since_train += 1
        if (
            len(self.short_term_memory) >= self.short_term_memory_size
            and self.memories_since_train >= self.short_term_memory_size
        ):
            print("start training ...")
            trained_steps = self.tlm.train_on_texts(
                texts=list(self.short_term_memory),
                max_seq_len=self.max_seq_len,
                lr=self.lr,
                epochs=self.epochs,
                batch_size=self.batch_size,
                grad_accum_steps=self.grad_accum,
                fp16=self.fp16,
            )
            self.steps_trained_total += trained_steps
            self.memories_since_train = 0

        return trained_steps

    def _format_step_memory(
        self,
        *,
        state_action: str,
        current_reflection: Optional[str],
    ) -> str:
        parts: List[str] = []
        parts.append(state_action)
        if current_reflection:
            parts.append(current_reflection)
        return "\n\n".join(parts)

    def _act(self, obs: Dict) -> str:
        obs_text = obs["text"]

        if self.enable_short_term_memory:
            retrieved = "\n".join(self.short_term_memory) if self.short_term_memory else ""
        else:
            retrieved = ""

        # print("retrieved memory:", retrieved, flush=True)

        current_reflection: Optional[str] = None
        if self.enable_reflection:
            current_reflection = self.cfg.reflect(self.llm, obs_text, retrieved)

        user_prompt = self.cfg.construct_user_prompt(obs_text)

        if self.enable_short_term_memory and retrieved:
            user_prompt = f"\n{retrieved}\n\n" + user_prompt

        if current_reflection:
            user_prompt = f"\n{current_reflection}\n\n" + user_prompt

        lm_output = self.tlm.generate(
            user_prompt=user_prompt,
            system_prompt=self.cfg.system_prompt,
        )

        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]
        readable = self.cfg.format_response(parsed)

        step_text = f"{obs_text}\n{readable}"

        if getattr(self.cfg, "enable_summarization", False):
            step_text = self.cfg.summarize(self.llm, step_text)
            step_text = (step_text if step_text else readable)

        step_memory = self._format_step_memory(
            state_action=step_text,
            current_reflection=current_reflection
        )

        self.memorize(step_memory + "\n")

        return (
            parsed_action,
            lm_output["num_input_tokens"],
            lm_output["num_output_tokens"],
            lm_output["response"],
        )


    def save_adapter(self, path: str):
        self.tlm.save_adapter(path)

    def load_adapter(self, path: str):
        self.tlm.load_adapter(path)

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        return self.tlm.generate(user_prompt=user_prompt, system_prompt=system_prompt)

    def save_memory(self, full_memory_dir: str) -> None:
        os.makedirs(full_memory_dir, exist_ok=True)

        # 1) Save adapter weights under full_memory_dir/adapter_subdir
        adapter_dir = os.path.join(full_memory_dir, self.adapter_subdir)
        os.makedirs(adapter_dir, exist_ok=True)
        self.save_adapter(adapter_dir)

        # 2) Save JSON checkpoint to full_memory_dir/memory_paths[0]
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        data = {
            "agent_id": self.id,
            "agent_name": self.name,
            "cfg": {
                "llm_name": getattr(self.cfg, "llm_name", None),
                "enable_reflection": getattr(self.cfg, "enable_reflection", None),
                "enable_summarization": getattr(self.cfg, "enable_summarization", None),
                "enable_short_term_memory": getattr(self.cfg, "enable_short_term_memory", None),
                "max_seq_len": getattr(self.cfg, "max_seq_len", None),
                "lr": getattr(self.cfg, "lr", None),
                "epochs": getattr(self.cfg, "epochs", None),
                "batch_size": getattr(self.cfg, "batch_size", None),
                "grad_accum": getattr(self.cfg, "grad_accum", None),
                "fp16": getattr(self.cfg, "fp16", None),
                "lora_config": getattr(self.cfg, "lora_config", None),
            },
            "memory": {
                "short_term_memory": list(self.short_term_memory),
                "memories_since_train": int(self.memories_since_train),
                "steps_trained_total": int(self.steps_trained_total),
            },
            "adapter_subdir": self.adapter_subdir,
        }

        atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))

    def load_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfgd = data.get("cfg", {}) or {}
        for k in [
            "llm_name",
            "enable_reflection",
            "enable_summarization",
            "enable_short_term_memory",
            "max_seq_len",
            "lr",
            "epochs",
            "batch_size",
            "grad_accum",
            "fp16",
        ]:
            if k in cfgd and cfgd[k] is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, cfgd[k])

        memd = data.get("memory", {}) or {}
        self.short_term_memory = list(memd.get("short_term_memory", []) or [])
        self.short_term_memory_size = self._get_short_term_memory_size()
        if len(self.short_term_memory) > self.short_term_memory_size:
            self.short_term_memory = self.short_term_memory[-self.short_term_memory_size :]
        print(
            "[Memory] Getting previous short-term memory: ",
            f"size={len(self.short_term_memory)}",
            flush=True,
        )
        self.steps_trained_total = int(memd.get("steps_trained_total", 0))
        self.memories_since_train = int(memd.get("memories_since_train", 0))

        self.max_seq_len = self.cfg.max_seq_len
        self.lr = self.cfg.lr
        self.epochs = self.cfg.epochs
        self.batch_size = self.cfg.batch_size
        self.grad_accum = self.cfg.grad_accum
        self.fp16 = self.cfg.fp16
        self.short_term_memory_size = self._get_short_term_memory_size()

        adapter_subdir = data.get("adapter_subdir", self.adapter_subdir)
        adapter_dir = os.path.join(full_memory_dir, adapter_subdir)
        if os.path.exists(adapter_dir):
            print("Loading adapter from ", adapter_dir, flush=True)
            self.load_adapter(adapter_dir)



@lru_cache(maxsize=None)
def create_lora_sft_agent(Agent: Type):
    class_name = (
        f"LoRASFTAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (LoRASFTAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )
