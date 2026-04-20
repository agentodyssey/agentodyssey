from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, RepetitionPenaltyLogitsProcessor
from typing import List
import torch
import warnings
from termcolor import colored
from tools.logger import get_logger


class hfEmbeddingModel:
    def __init__(self, embedder_name: str, device: str):
        self.logger = get_logger("hfEmbeddingModelLogger")
        try:
            self.model = SentenceTransformer(embedder_name, device=device)
            self.logger.info(f"✅ Loaded HuggingFace embedding model {embedder_name} on {device}")
        except Exception as e:
            self.logger.error(f"❌ Error loading model {embedder_name}: {e}")
            raise

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        try:
            emb = self.model.encode(
                texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
            )
            return emb.cpu()
        except Exception as e:
            self.logger.error(f"❌ Error encoding texts: {e}")
            raise

class hfLanguageModel:
    def __init__(
        self,
        llm_name: str,
        temperature: float,
        top_p: float,
        presence_penalty: float,
        max_new_tokens: int = None,
        device: str = "cuda",
    ):
        try:
            self.device = device
            self.logger = get_logger("hfLanguageModelLogger")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            self.logger.info(f"✅ Loaded HuggingFace model {llm_name} on {self.device}.")
            self.temperature = temperature
            self.top_p = top_p
            self.repetition_penalty = (1.0 + 0.2 * presence_penalty) if presence_penalty else None
            self.max_new_tokens = max_new_tokens
        except Exception as e:
            self.logger.error(f"❌ Error loading model {llm_name}: {e}")
            raise

    @torch.inference_mode()
    def generate(self, user_prompt: str, system_prompt: str = None, think: bool = True) -> str:
        try:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            else:
                messages = [{"role": "user", "content": user_prompt}]
            
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=think,
            )
            
            inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)
            num_input_tokens = inputs["input_ids"].shape[1]

            logits_processor = LogitsProcessorList()
            if self.repetition_penalty and self.repetition_penalty != 1.0:
                logits_processor.append(RepetitionPenaltyLogitsProcessor(self.repetition_penalty))
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                logits_processor=logits_processor,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

            output_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            num_output_tokens = len(output_tokens)

            return {
                "response": response,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens
            }

        except Exception as e:
            self.logger.error(f"❌ Error generating response: {e}")
            return {
                "response": None,
                "num_input_tokens": 0,
                "num_output_tokens": 0
            }