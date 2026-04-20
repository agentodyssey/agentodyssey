import os
import re
import sys
import json
import math
import torch
from collections import Counter
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from utils import dynamic_load_game_class
from agents.llm_agent_config import LLMAgentConfig
from agents.parametric.param_agent_config import ParamAgentConfig
from agents.rag.rag_agent_config import RAGAgentConfig
from agents.fixed_size.fixed_size_memory_agent_config import FixedSizeMemoryAgentConfig
from agents.latent.latent_agent_config import LatentAgentConfig
from agents.rag.voyager_agent import VoyagerAgentConfig
from tools.generate_episodic_memory_qa import generate_episodic_memory_qa


def normalize_actions(actions, action_types):
    normalized = []
    for action in actions:
        for p in action_types:
            if action.startswith(p):
                normalized.append(p)
                break
    return normalized

def entropy_diversity_score(window, num_actions_total):
    counts = Counter(window)
    total = len(window)
    H = -sum(
        (c / total) * math.log(c / total)
        for c in counts.values()
    )
    H_max = math.log(num_actions_total)
    diversity = H / H_max
    diversity = max(0.0, min(1.0, diversity))
    return diversity

def entropy_diversity_sliding_window_scores(actions, k, num_actions_total):
    scores = []

    for i in range(len(actions) - k + 1):
        window = actions[i:i + k]
        diversity = entropy_diversity_score(window, num_actions_total)
        scores.append(diversity)

    return scores

def cumulative_sum(values):
    total = 0.0
    out = []
    for v in values:
        total += v
        out.append(total)
    return out

def plot_series(x, y, ylabel, run_dir, filename):
    plt.figure(figsize=(5, 3.5))
    plt.plot(x, y, linewidth=1.0, marker="o", markersize=2, alpha=0.9)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, filename), dpi=300)
    plt.close()

def get_agent_log_stats_and_gen_plots(run_dir, agent_id):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    config_path = os.path.join(run_dir, "config.json")
    agent_log_path = os.path.join(run_dir, agent_id, "agent_log.jsonl")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    records = []
    with open(agent_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                records.append(obj)

    records.sort(key=lambda r: r["step"])

    steps = [r["step"] for r in records]
    actions = [r["action"] for r in records if "Tutorial, room" not in r["observation"]["text"]]

    num_input_tokens = [int(r["num_input_tokens"]) for r in records]
    num_output_tokens = [int(r["num_output_tokens"]) for r in records]
    total_tokens = [a + b for a, b in zip(num_input_tokens, num_output_tokens)]
    decision_time = [float(r["decision_time"]) for r in records]

    plot_series(steps, num_input_tokens, "Input tokens", run_dir, "input_tokens.pdf")
    plot_series(steps, num_output_tokens, "Output tokens", run_dir, "output_tokens.pdf")
    plot_series(steps, total_tokens, "Total tokens", run_dir, "total_tokens.pdf")
    plot_series(steps, decision_time, "Decision times", run_dir, "decision_times.pdf")

    reward_categories = ["quest", "side_quest", "exploration", "unique_kill", "craft"]
    reward_series = {c: [] for c in reward_categories}
    for r in records:
        reward = r.get("reward") or {}
        for c in reward_categories:
            reward_series[c].append(float(reward.get(c, 0.0) or 0.0))
    for c in reward_categories:
        reward_series[c] = cumulative_sum(reward_series[c])

    for c in reward_categories:
        plot_series(
            steps,
            reward_series[c],
            f"Cumulative reward ({c})",
            run_dir,
            f"cumulative_reward_{c}.pdf",
        )

    average_input_tokens = sum(num_input_tokens) / len(num_input_tokens)
    average_output_tokens = sum(num_output_tokens) / len(num_output_tokens)
    average_total_tokens = sum(total_tokens) / len(total_tokens)
    average_decision_time = sum(decision_time) / len(decision_time)

    action_types = [a["verb"] for a in config["agents"][0]["available_actions"]]
    actions_normalized = normalize_actions(actions, action_types)

    all_actions_entropy_diversity_score = entropy_diversity_score(actions_normalized, num_actions_total=len(action_types))

    actions_sliding_window_entropy_diversity_scores = entropy_diversity_sliding_window_scores(actions_normalized, k=100, num_actions_total=len(action_types))

    plot_series(
        list(range(len(actions_sliding_window_entropy_diversity_scores))),
        actions_sliding_window_entropy_diversity_scores,
        "Diversity score",
        run_dir,
        "entropy_diversity_sliding_window.pdf",
    )

    return {
        "average_input_tokens": average_input_tokens,
        "average_output_tokens": average_output_tokens,
        "average_total_tokens": average_total_tokens,
        "average_decision_time": average_decision_time,
        "explored_actions": str(len(set(actions_normalized))) + " / " + str(len(action_types)),
        "invalid_action_percentage": sum(1 for r in records if bool(r["invalid_action"])) / len(records) * 100.0,
        "all_actions_entropy_diversity_score": all_actions_entropy_diversity_score,
    }

def build_agent(game_name, agent_name, llm_name, agent_id, llm_provider=None):
    if agent_name in {"VanillaRAGAgent", "Mem0RAGAgent", "RaptorRAGAgent"}:
        cfg = RAGAgentConfig(llm_name=llm_name, llm_provider=llm_provider)
        if agent_name == "Mem0RAGAgent":
            cfg.full_mem_path = f"output/game_{game_name}/{llm_name}/{agent_name}/no_extras/{agent_id}/memory/mem0_store"
            # print(cfg.full_mem_path)
    elif agent_name in {"VanillaParamAgent", "LoRASFTAgent","RLAgent"}:
        cfg = ParamAgentConfig(llm_name=llm_name, llm_provider=llm_provider)
    elif agent_name in {"LongContextAgent", "Mem1Agent", "ShortTermMemoryAgent"}:
        cfg = FixedSizeMemoryAgentConfig(llm_name=llm_name, llm_provider=llm_provider)
    elif agent_name in {"MPlusAgent", "MemoryLLMAgent"}:
        cfg = LatentAgentConfig(llm_name=llm_name, llm_provider=llm_provider)
    elif agent_name in {"NoMemoryAgent"}:
        cfg = LLMAgentConfig(llm_name=llm_name, llm_provider=llm_provider)
    elif agent_name in {"VoyagerAgent"}:
        cfg = VoyagerAgentConfig(llm_name=llm_name, llm_provider=llm_provider)
    
    agent_info = {"id": agent_id, "name": agent_name, "cfg": cfg}

    AgentCls = dynamic_load_game_class(game_name, "agent", "Agent")

    if agent_name == "HumanAgent":
        from agents.human_agent import create_human_agent
        HumanAgent = create_human_agent(AgentCls)
        return HumanAgent(**agent_info)
    if agent_name == "VanillaRAGAgent":
        from agents.rag.vanilla_rag_agent import create_vanilla_rag_agent
        VanillaRAGAgent = create_vanilla_rag_agent(AgentCls)
        return VanillaRAGAgent(**agent_info)
    if agent_name == "Mem0RAGAgent":
        from agents.rag.mem0_rag_agent import create_mem0_rag_agent
        Mem0RAGAgent = create_mem0_rag_agent(AgentCls)
        return Mem0RAGAgent(**agent_info)
    if agent_name == "RaptorRAGAgent":
        from agents.rag.raptor_rag_agent import create_raptor_rag_agent
        RaptorRAGAgent = create_raptor_rag_agent(AgentCls)
        return RaptorRAGAgent(**agent_info)
    if agent_name == "LoRASFTAgent" or agent_name == "RLAgent":
        from agents.parametric.lora_sft_agent import create_lora_sft_agent
        LoRASFTAgent = create_lora_sft_agent(AgentCls)
        return LoRASFTAgent(**agent_info)
    if agent_name == "MPlusAgent":
        from agents.latent.mplus_agent import create_mplus_agent
        MPlusAgent = create_mplus_agent(AgentCls)
        return MPlusAgent(**agent_info)
    if agent_name == "MemoryLLMAgent":
        from agents.latent.memoryllm_agent import create_memoryllm_agent
        MemoryLLMAgent = create_memoryllm_agent(AgentCls)
        return MemoryLLMAgent(**agent_info)
    if agent_name == "LongContextAgent":
        from agents.long_context_agent import create_long_context_agent
        LongContextAgent = create_long_context_agent(AgentCls)
        return LongContextAgent(**agent_info)
    if agent_name == "Mem1Agent":
        from agents.fixed_size.mem1_agent import create_mem1_agent
        Mem1Agent = create_mem1_agent(AgentCls)
        return Mem1Agent(**agent_info)
    if agent_name == "VoyagerAgent":
        from agents.rag.voyager_agent import create_voyager_agent
        VoyagerAgent = create_voyager_agent(AgentCls)
        return VoyagerAgent(**agent_info)
    if agent_name == "ShortTermMemoryAgent":
        from agents.fixed_size.short_term_memory_agent import create_short_term_memory_agent
        ShortTermMemoryAgent = create_short_term_memory_agent(AgentCls)
        return ShortTermMemoryAgent(**agent_info)
    if agent_name == "NoMemoryAgent":
        from agents.no_memory_agent import create_no_memory_agent
        NoMemoryAgent = create_no_memory_agent(AgentCls)
        return NoMemoryAgent(**agent_info)

    raise ValueError(f"Unknown agent: {agent_name}")

def load_agent_memory(agent, full_memory_dir):
    if not hasattr(agent, "load_memory"):
        print(f"Agent {agent.id} does not support load_memory.")
        return False

    memory_paths = getattr(agent, "memory_paths", [])
    full_memory_paths = [os.path.join(full_memory_dir, path) for path in memory_paths]
    if memory_paths and all(os.path.exists(mem_path) for mem_path in full_memory_paths):
        agent.load_memory(full_memory_dir=full_memory_dir)
        # print(f"Loaded memory for agent {agent.id} from {full_memory_paths}.")
        return True

    # print(f"No previous memory found for agent {agent.id} in {full_memory_dir}.")
    return False

def eval_qa(qa, agent, agent_name, enable_memory):
    total_correct = 0
    total_count = 0
    per_type_stats = {}

    for q_type, questions in tqdm(qa.items(), desc="types"):
        correct = 0
        count = 0

        for item in tqdm(questions, desc=f"{q_type} questions", leave=False):
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer_idx"]

            choices_text = "\n".join(
                f"{i}. {choice.strip()}" for i, choice in enumerate(choices)
            )

            prompt = (
                "Answer the following multiple-choice question.\n"
                "Respond ONLY with the index (0-based) of the correct choice.\n\n"
                f"Question:\n{question}\n\n"
                f"Choices:\n{choices_text}\n\n"
                "Answer:"
            )

            # print("prompt:\n", prompt)

            response = prompt_agent(agent, agent_name, prompt, enable_memory)

            if agent.cfg.llm_name.startswith("Qwen"):
                response = re.sub(r"<think>[\s\S]*?</think>", "", response)

            response = response.strip()

            # print(f"Q: {question}")
            print(f"Raw model response: {response}")
            # print(f"Correct answer index: {answer_idx}")
            # print(f"Correct answer text: {choices[answer_idx]}")

            if response.strip().isdigit():
                pred_idx = int(response)
                if pred_idx == answer_idx:
                    correct += 1
                    # print("=> Correct!")
                else:
                    pass
                    # print("=> Incorrect.")
            else:
                pass
                # print("=> Invalid response format.")

            count += 1
        per_type_stats[q_type] = {
            "correct": correct,
            "count": count,
            "accuracy": (correct / count) if count else 0.0,
        }
        total_correct += correct
        total_count += count

    overall_accuracy = (total_correct / total_count) if total_count else 0.0
    # print("\nPer-type accuracy:")
    for q_type, stats in per_type_stats.items():
        pass
        # print(f"- {q_type}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['count']})")
    # print(f"\nOverall accuracy: {overall_accuracy:.3f} ({total_correct}/{total_count})")
    return overall_accuracy, per_type_stats


def prompt_agent(agent, agent_name, prompt, enable_memory=True):
    if agent_name == "LongContextAgent":
        if enable_memory:
            prompt_memory = agent.memory.get_all_memory()
            if agent.cfg.llm_name == "gpt-5" or agent.cfg.llm_name == "gpt-5-mini":
                max_tokens = 400000
            elif agent.cfg.llm_name == "Qwen/Qwen3-4B":
                max_tokens = 131072
            token_character_ratio = 3
            # Truncate prompt_memory if too long
            estimated_tokens = len(prompt_memory) // token_character_ratio
            # print("Max # tokens:", max_tokens)
            # print("Estimated # tokens in memory:", estimated_tokens)
            if estimated_tokens > max_tokens:
                reserved_length = 40000
                prompt_memory = prompt_memory[-((max_tokens * token_character_ratio) - reserved_length):]
                # print("Truncated memory to fit max tokens.")
                # print("New estimated # tokens in memory:", len(prompt_memory) // token_character_ratio)
            lm_output = agent.llm.generate(user_prompt=prompt_memory + '\n' + prompt, system_prompt=None)
        else:
            lm_output = agent.llm.generate(user_prompt=prompt, system_prompt=None)
        response = lm_output["response"]
        return response

    if agent_name == "Mem1Agent":
        if enable_memory:
            # Mem1Agent stores memory as is_text, accessed via fragments()
            # fragments() returns "<IS>{is_text}</IS>" format
            prompt_memory = agent.memory.fragments()
            # print(prompt_memory, flush=True)
            lm_output = agent.llm.generate(user_prompt=prompt_memory + '\n' + prompt, system_prompt=None)
        else:
            lm_output = agent.llm.generate(user_prompt=prompt, system_prompt=None)
        response = lm_output["response"]
        return response

    if agent_name == "ShortTermMemoryAgent":
        if enable_memory:
            # ShortTermMemoryAgent uses sliding window of last N memories
            retrieved = agent.short_term_memory[-agent.short_term_memory_size:] if agent.short_term_memory else []
            # print(retrieved)
            prompt_memory = "\n".join(retrieved)
            lm_output = agent.llm.generate(user_prompt=prompt_memory + '\n' + prompt, system_prompt=None)
        else:
            lm_output = agent.llm.generate(user_prompt=prompt, system_prompt=None)
        response = lm_output["response"]
        return response

    if agent_name == "VanillaRAGAgent":
        if enable_memory:
            # VanillaRAGAgent uses embedding-based retrieval with the prompt as query
            retrieved_long_term = agent.memory.retrieve(prompt, agent.cfg.memory_retrieve_limit)
            
            # Also include short-term memory if enabled
            if getattr(agent.cfg, "enable_short_term_memory", False):
                retrieved_short_term = agent.short_term_memory[-agent.short_term_memory_size:] if agent.short_term_memory else []
                # Deduplicate long-term against short-term
                short_term_set = set(retrieved_short_term)
                retrieved_long_term = [item for item in retrieved_long_term if item not in short_term_set]
                retrieved = retrieved_long_term + retrieved_short_term
            else:
                retrieved = retrieved_long_term
            
            prompt_memory = "\n".join(retrieved)
            # print(prompt_memory)
            lm_output = agent.llm.generate(user_prompt=prompt_memory + '\n' + prompt, system_prompt=None)
        else:
            lm_output = agent.llm.generate(user_prompt=prompt, system_prompt=None)
        response = lm_output["response"]
        return response

    if agent_name == "Mem0RAGAgent":
            if enable_memory:
                # Mem0RAGAgent uses mem0 library for semantic retrieval, returns List[str]
                retrieved = agent.memory.retrieve(prompt, agent.cfg.memory_retrieve_limit)
                prompt_memory = "\n".join(retrieved)
                # print(prompt_memory)
                lm_output = agent.llm.generate(user_prompt=prompt_memory + '\n' + prompt, system_prompt=None)
            else:
                lm_output = agent.llm.generate(user_prompt=prompt, system_prompt=None)
            response = lm_output["response"]
            return response

    if agent_name == "RaptorRAGAgent":
        if enable_memory:
            # RaptorRAGAgent uses RAPTOR tree retrieval, returns str directly
            retrieved = agent.memory.retrieve(prompt, agent.cfg.memory_retrieve_limit)
            prompt_memory = retrieved  # already a string
            # print(prompt_memory)
            lm_output = agent.llm.generate(user_prompt=prompt_memory + '\n' + prompt, system_prompt=None)
        else:
            lm_output = agent.llm.generate(user_prompt=prompt, system_prompt=None)
        response = lm_output["response"]
        return response

    if agent_name == "VoyagerAgent":
        if enable_memory:
            # VoyagerAgent has multiple memory components:
            # 1. skill_lib - learned skills with embeddings
            # 2. progress - completed/failed tasks
            # 3. trace - past observation/action pairs
            
            # Retrieve relevant skills using prompt as query
            retrieved_skills = agent.skill_lib.retrieve(prompt, agent.cfg.retrieval_top_k)
            
            # Build memory context
            memory_parts = []
            
            # Add skill information
            if retrieved_skills:
                skills_text = "Relevant learned skills:\n" + "\n".join(
                    [f"- {s.name}: {s.desc}\n  Body: {s.body}" for s in retrieved_skills]
                )
                memory_parts.append(skills_text)
            
            # Add progress information
            if agent.progress.completed_tasks:
                memory_parts.append(f"Completed tasks: {agent.progress.completed_tasks[-10:]}")
            if agent.progress.failed_tasks:
                memory_parts.append(f"Failed tasks: {agent.progress.failed_tasks[-10:]}")
            
            # Add recent trace (last few steps)
            if agent.trace:
                recent_trace = agent.trace[-5:]
                trace_text = "Recent history:\n" + "\n".join(
                    [f"- Obs: {t['obs'][:100]}... Action: {t['action']}" for t in recent_trace]
                )
                memory_parts.append(trace_text)
            
            prompt_memory = "\n\n".join(memory_parts)
            # print(prompt_memory)
            lm_output = agent.llm.generate(user_prompt=prompt_memory + '\n\n' + prompt, system_prompt=None)
        else:
            lm_output = agent.llm.generate(user_prompt=prompt, system_prompt=None)
        response = lm_output["response"]
        return response

    if agent_name == "LoRASFTAgent" or agent_name == "RLAgent":
        if enable_memory:
            # LoRASFTAgent has two forms of memory:
            # 1. Parametric memory: LoRA adapter weights (implicitly used via tlm.generate)
            # 2. Short-term memory: recent experience strings (if enabled)
            
            # The LoRA adapter IS the primary memory - it's been fine-tuned on experiences
            # We use tlm.generate() which automatically uses the fine-tuned adapter
            lm_output = agent.tlm.generate(user_prompt=prompt, system_prompt=None)
        else:
            # Without memory = use base model without LoRA weights
            # But since LoRA is merged, we still use tlm but just skip short-term memory
            lm_output = agent.tlm.generate(user_prompt=prompt, system_prompt=None)
        response = lm_output["response"]
        return response

    if agent_name == "MemoryLLMAgent":
        if enable_memory:
            # MemoryLLMAgent uses latent memory injection - memory is encoded in model's hidden state
            # The memory is already "inside" the model via inject_memory(), so we just generate
            messages = [
                {"role": "user", "content": prompt},
            ]
            inputs = agent.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )[:, 1:].to(agent.device)
            
            terminators = [
                agent.tokenizer.eos_token_id,
                agent.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                agent.tokenizer.convert_tokens_to_ids("<|user|>"),
                agent.tokenizer.convert_tokens_to_ids("|>"),
                agent.tokenizer.convert_tokens_to_ids("|user|>"),
                agent.tokenizer.convert_tokens_to_ids("|user|")
            ]
            terminators = [t for t in terminators if t is not None]
            
            outputs = agent.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=300,
                eos_token_id=terminators,
                pad_token_id=agent.tokenizer.pad_token_id,
                do_sample=False,

            )
            output_tokens = outputs[0][inputs.shape[1]:]
            response = agent.tokenizer.decode(output_tokens, skip_special_tokens=True)
        else:
            # Without memory context - but latent memory is still in model state
            # To truly disable, we'd need to reset the model, which isn't practical
            messages = [{"role": "user", "content": prompt}]
            inputs = agent.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )[:, 1:].to(agent.device)
            
            terminators = [
                agent.tokenizer.eos_token_id,
                agent.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                agent.tokenizer.convert_tokens_to_ids("<|user|>"),
                agent.tokenizer.convert_tokens_to_ids("|>"),
                agent.tokenizer.convert_tokens_to_ids("|user|>"),
                agent.tokenizer.convert_tokens_to_ids("|user|")
            ]
            terminators = [t for t in terminators if t is not None]
            
            outputs = agent.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=100,
                eos_token_id=terminators,
                pad_token_id=agent.tokenizer.pad_token_id,
                do_sample=False
            )
            output_tokens = outputs[0][inputs.shape[1]:]
            response = agent.tokenizer.decode(output_tokens, skip_special_tokens=True)
        raw = response.strip()

        # Cut at the first occurrence of any chat marker
        STOP_MARKERS_RE = r"(\|user\|>|\|</assistant\|>|\|</user\|>)"
        answer_part = re.split(STOP_MARKERS_RE, raw, maxsplit=1)[0].strip()
        answer_part = answer_part or raw  # fallback if empty

        m = re.search(r"The answer is\s+(.+?)(?:[.\n\r|]|$)", answer_part, flags=re.IGNORECASE)
        if m:
            parsed = m.group(1).strip().strip(' "\'').rstrip(" .,:;!?")
            response = parsed or answer_part
        else:
            response = answer_part

        return response

    if agent_name == "MPlusAgent":
        if enable_memory:
            # MPlusAgent uses latent memory injection similar to MemoryLLMAgent
            # Memory is encoded in model's latent state via inject_memory()
            messages = [
                {"role": "user", "content": prompt},
            ]
            inputs = agent.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )[:, 1:].to(agent.device)
            
            terminators = [
                agent.tokenizer.eos_token_id,
                agent.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                agent.tokenizer.convert_tokens_to_ids("<|user|>"),
            ]
            terminators = [t for t in terminators if t is not None]
            
            outputs = agent.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=100,
                eos_token_id=terminators,
                pad_token_id=agent.tokenizer.pad_token_id,
                do_sample=False
            )
            output_tokens = outputs[0][inputs.shape[1]:]
            response = agent.tokenizer.decode(output_tokens, skip_special_tokens=True)
        else:
            messages = [{"role": "user", "content": prompt}]
            inputs = agent.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )[:, 1:].to(agent.device)
            
            terminators = [
                agent.tokenizer.eos_token_id,
                agent.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                agent.tokenizer.convert_tokens_to_ids("<|user|>"),
            ]
            terminators = [t for t in terminators if t is not None]
            
            outputs = agent.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=100,
                eos_token_id=terminators,
                pad_token_id=agent.tokenizer.pad_token_id,
                do_sample=False
            )
            output_tokens = outputs[0][inputs.shape[1]:]
            response = agent.tokenizer.decode(output_tokens, skip_special_tokens=True)
        raw = response.strip()

        # Cut at the first occurrence of any chat marker
        STOP_MARKERS_RE = r"(\|user\|>|\|</assistant\|>|\|</user\|>)"
        answer_part = re.split(STOP_MARKERS_RE, raw, maxsplit=1)[0].strip()
        answer_part = answer_part or raw  # fallback if empty

        m = re.search(r"The answer is\s+(.+?)(?:[.\n\r|]|$)", answer_part, flags=re.IGNORECASE)
        if m:
            parsed = m.group(1).strip().strip(' "\'').rstrip(" .,:;!?")
            response = parsed or answer_part
        else:
            response = answer_part

        return response

    if agent_name == "NoMemoryAgent":
        lm_output = agent.llm.generate(user_prompt=prompt, system_prompt=None)
        response = lm_output["response"]
        return response


    raise ValueError(f"Unknown agent: {agent_name}")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_full", type=str, required=True)
    parser.add_argument("--llm_provider", type=str, choices=["openai", "huggingface", "vllm", "azure", "azure_openai", "claude", "gemini"], help="Which LLM provider to use", required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--enable_memory", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--game_name", type=str, default="remnant")
    parser.add_argument("--agent_id", type=str, default="agent_adam_davis")
    parser.add_argument("--extra_dir", type=str, default="0")
    parser.add_argument("--memory_dir", type=str, default="memory")

    args = parser.parse_args(argv)

    if args.game_name in (None, "", "base"):
        world_definition_path = f"assets/world_definitions/base/default.json"
    else:
        world_definition_path = f"assets/world_definitions/generated/{args.game_name}/default.json"

    run_dir = os.path.join("output", "game_" + args.game_name)
    world_knowledge_qa_path = os.path.join("output", "game_" + args.game_name, "world_knowledge_qa_sampled.json")
    if args.llm_name is not None:
        run_dir = os.path.join(run_dir, args.llm_name.replace("/", "_"))
    if '_' in args.agent_full:
        agent_name, option = args.agent_full.split("_", 1)
    else:
        agent_name, option = args.agent_full, "noextras"
    run_dir = os.path.join(run_dir, agent_name)
    if option == "withshorttermmemory":
        run_dir = os.path.join(run_dir, "with_short_term_memory")
    elif option == "withreflection":
        run_dir = os.path.join(run_dir, "with_reflection")
    elif option == "withsummarization":
        run_dir = os.path.join(run_dir, "with_summarization")
    else:
        run_dir = os.path.join(run_dir, "no_extras")
    if args.extra_dir is not None:
        run_dir = os.path.join(run_dir, str(args.extra_dir))
    config_path = os.path.join(run_dir, "config.json")

    results_path = os.path.join(run_dir, f"results_{'before' if not args.enable_memory else 'after'}.json")
    if os.path.exists(results_path) and not args.overwrite:
        results = json.load(open(results_path, "r"))
        print(f"Results already exist at {results_path}. This run will not re-evaluate world knowledge and episodic QA. Use --overwrite to re-evaluate.")
    else:
        results = {}
    
    with open(config_path, "r") as f:
        config = json.load(f)
        # print(f"LLM: {args.llm_name}, Agent: {args.agent_full}")
        scores = config["scores"][args.agent_id]
        results["steps"] = config["step"]
        results["progress"] = [scores['exploration'], scores['craft'], scores['unique_kill'], scores['trade'], scores['death'], scores['quest'], scores['side_quest']]

    # gather agent log stats
    # print(f"Gathering agent log stats for agent {args.agent_full} with LLM {args.llm_name}")
    agent_log_stats = get_agent_log_stats_and_gen_plots(run_dir, args.agent_id)
    results["agent_log_stats"] = agent_log_stats

    # explored object percentage
    world_definition = json.load(open(world_definition_path, "r"))
    results["agent_log_stats"]["explored_objects"] = str(len(config["curr_agents_state"]["objects_acquired"][args.agent_id])) + " / " + str(len(world_definition["entities"]["objects"]))
    
    # build agent and load memory
    if os.path.exists(results_path) and not args.overwrite:
        pass
    else:
        agent = build_agent(args.game_name, agent_name, args.llm_name, args.agent_id, llm_provider=args.llm_provider)
        if args.enable_memory:
            load_agent_memory(agent, os.path.join(run_dir, args.agent_id, args.memory_dir))
        else:
            # For latent/parametric agents, the fresh agent IS the "no memory" baseline
            # No need to call load_memory - the model is already in its pristine state
            pass

    # prepare questions - start with world knowledge
    if os.path.exists(results_path) and not args.overwrite:
        pass
    else:
        print(f"Evaluating agent {args.agent_full} with LLM {args.llm_name} on world knowledge with memory={args.enable_memory}")
        world_knowledge_qa = json.load(open(world_knowledge_qa_path, "r"))
        overall_accuracy, per_type_stats = eval_qa(world_knowledge_qa, agent, agent_name, args.enable_memory)
        results["world_knowledge_qa_accuracy"] = overall_accuracy
        results["world_knowledge_qa_per_type"] = per_type_stats

    # prepare questions - episodic experience questions to be generated during run time
    if os.path.exists(results_path) and not args.overwrite:
        pass
    else:
        print(f"Evaluating agent {args.agent_full} with LLM {args.llm_name} on episodic experience with memory={args.enable_memory}")
        agent_log_path = os.path.join(run_dir, args.agent_id, "agent_log.jsonl")
        episodic_qa = generate_episodic_memory_qa(world_definition_path, config_path, agent_log_path, num_samples_per_category=20, seed=0)
        overall_accuracy, per_type_stats = eval_qa(episodic_qa, agent, agent_name, args.enable_memory)
        results["episodic_qa_accuracy"] = overall_accuracy
        results["episodic_qa_per_type"] = per_type_stats
    
    # dump results
    results_path = os.path.join(run_dir, f"results_{'before' if not args.enable_memory else 'after'}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()