from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Type
from functools import lru_cache

import torch
import os
from providers.huggingface import hfEmbeddingModel
from agents.rag.rag_agent_config import RAGAgentConfig
from utils import atomic_write

@dataclass
class Skill:
    name: str
    desc: str
    body: str

class SkillLibrary:
    def __init__(self, embedder: hfEmbeddingModel):
        self.embedder = embedder
        self.skills: List[Skill] = []
        self.emb: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.skills)

    def add(self, skill: Skill):
        # index the skill by its name and description (as done in Voyager)
        text_for_index = f"{skill.name}: {skill.desc}"
        new_emb = self.embedder.encode([text_for_index])
        self.skills.append(skill)
        self.emb = new_emb if self.emb is None else torch.cat([self.emb, new_emb], dim=0)

    @torch.inference_mode()
    def retrieve(self, query: str, top_k: int) -> List[Skill]:
        if self.emb is None or self.emb.shape[0] == 0:
            return []
        q = self.embedder.encode([query])
        sims = (q @ self.emb.T).squeeze(0)
        k = min(top_k, sims.numel())
        top = torch.topk(sims, k=k)
        return [self.skills[i] for i in top.indices.tolist()]

    def to_dict(self) -> dict:
        return {"skills": [{"name": s.name, "desc": s.desc, "body": s.body} for s in self.skills]}

    def load_from_dict(self, d: dict) -> None:
        self.skills = []
        self.emb = None
        for item in d.get("skills", []) or []:
            name = item.get("name", "")
            desc = item.get("desc", "")
            body = item.get("body", "")
            if name and desc and body:
                self.skills.append(Skill(name=name, desc=desc, body=body))

    def rebuild_embeddings(self) -> None:
        if not self.skills:
            self.emb = None
            return
        texts = [f"{s.name}: {s.desc}" for s in self.skills]
        self.emb = self.embedder.encode(texts)


class ProgressMemory:
    def __init__(self):
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []

    def mark_success(self, task: str):
        self.completed_tasks.append(task)

    def mark_failure(self, task: str):
        self.failed_tasks.append(task)

    def to_dict(self) -> dict:
        return {
            "completed_tasks": list(self.completed_tasks),
            "failed_tasks": list(self.failed_tasks),
        }

    def load_from_dict(self, d: dict) -> None:
        self.completed_tasks = list(d.get("completed_tasks", []) or [])
        self.failed_tasks = list(d.get("failed_tasks", []) or [])

# Will separate this class later
@dataclass
class VoyagerAgentConfig(RAGAgentConfig):
    retrieval_top_k: int = 5
    distill_window: int = 8

    # Debug flags
    debug: bool = True
    debug_show_system_prompts: bool = False
    debug_show_user_prompts: bool = True
    debug_show_raw_outputs: bool = True
    debug_max_chars: int = None

    # Debug output file
    debug_log_file: Optional[str] = None
    debug_also_print: bool = False

    # maximum env steps allowed per subgoal before abandoning it
    max_subgoal_steps: int = 12

    # Role-specific system prompts
    curriculum_system_prompt: str = (
        "You are the automatic curriculum agent for an open-ended text-based adventure game. Based on the agent's current state and progress, suggest the next task."
        "Return ONLY JSON: {\"task\":\"...\",\"context\":\"...\"}."
    )
    critic_system_prompt: str = (
        "You judge whether the current task is completed from the latest observation.\n"
        "Return ONLY JSON: {\"success\": true|false, \"critique\": \"...\"}."
    )
    skill_system_prompt: str = (
        "You are a skill distillation assistant. Your role is to analyze a successful action trace and extract a reusable high-level skill from it.\n"
        "Return ONLY JSON: {\"skill_name\":\"...\",\"skill_desc\":\"...\",\"skill_body\":\"...\"}."
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # Override the default action agent system prompt with additional guidance
        if self.available_actions: 
            actions_formatted = [f"- {action.verb} " + " ".join(f"<{p}>" for p in action.params) for action in self.available_actions]
            actions_formatted_str = "\n".join(actions_formatted)

            self.curriculum_system_prompt += (
                "\n\nThe Actor Agent can only use the following actions:\n" f"{actions_formatted_str}\n"
            )

            self.critic_system_prompt += (
                "\n\nThe Actor Agent can only use the following actions:\n" f"{actions_formatted_str}\n"
            )
            self.skill_system_prompt += (
                "\n\nThe Actor Agent can only use the following actions:\n"
                f"{actions_formatted_str}\n"
            )

    def _cap(self, s: str) -> str:
        if s is None:
            return ""
        if self.debug_max_chars is None:
            return s
        return s[: self.debug_max_chars] + ("…[truncated]" if len(s) > self.debug_max_chars else "")

    def curriculum_user_prompt(
        self,
        observation: str,
        completed_tasks: List[str],
        failed_tasks: List[str],
        last_task: Optional[str],
        last_critique: str,
    ) -> str:
        return "\n\n".join([
            "Agent progress:",
            f"Completed (recent): {completed_tasks[-10:]}",
            f"Failed (recent): {failed_tasks[-10:]}",
            f"Last task: {last_task}",
            f"Last critique: {last_critique}",
            "Agent Current Observation:\n" + observation,
        ])

    def critic_user_prompt(
        self,
        task: str,
        context: str,
        observation: str,
        retrieved_skills: List[Skill],
    ) -> str:
        parts = [
            f"Current Task: {task}",
            f"Task Context: {context}",
        ]
        if retrieved_skills:
            parts.append("Relevant skills:\n" + "\n".join([f"- {s.name}: {s.desc}" for s in retrieved_skills]))
        parts.append("Agent Current Observation:\n" + observation)
        return "\n\n".join(parts)

    def action_user_prompt(
        self,
        task: str,
        context: str,
        observation: str,
        retrieved_skills: List[Skill],
        critique: str,
        memory: Optional[str] = None,
    ) -> str:
        parts = []
        parts.append(f"Current Task: {task}")
        if context: parts.append(f"Task Context: {context}")
        if retrieved_skills:
            parts.append("Relevant skills:\n" + "\n".join([f"- {s.name}: {s.desc}" for s in retrieved_skills]))
        if critique: parts.append("Critic feedback:\n" + critique)
        if memory: parts.append("Recent memory:\n" + memory)
        parts.append("My Current Observation:\n" + observation)
        parts.append(self.action_prompt)
        return "\n\n".join(parts)

    def skill_user_prompt(
        self,
        task: str,
        context: str,
        trace: List[Dict[str, str]],
    ) -> str:
        parts = [
            f"Completed Task: {task}",
            f"Task Context: {context}",
            "Successful trace (most recent steps):",
            json.dumps(trace, ensure_ascii=False, indent=2),
            "Instruction: Distill ONE reusable skill with (skill_name, skill_desc, skill_body).",
        ]
        return "\n\n".join(parts)

class VoyagerAgent:
    def __init__(self, id: str, name: str, cfg: Optional[VoyagerAgentConfig] = None):
        super().__init__(id, name)
        if cfg is None:
            self.cfg = VoyagerAgentConfig(available_actions=self.available_actions)
        else:
            if not getattr(cfg, "available_actions", None):
                merged_cfg = dict(cfg.__dict__)
                merged_cfg["available_actions"] = self.available_actions
                self.cfg = VoyagerAgentConfig(**merged_cfg)
            else:
                self.cfg = cfg

        self._log_fh = None
        if self.cfg.debug and self.cfg.debug_log_file is not None:
            self._log_fh = open(self.cfg.debug_log_file, "a", encoding="utf-8")
            self._log_fh.write("\n\n========== NEW RUN ==========\n")
            self._log_fh.flush()

        self.llm = self.cfg.get_llm()
        self.embedder = self.cfg.get_embedder()

        self.skill_lib = SkillLibrary(self.embedder)
        self.progress = ProgressMemory()

        self.current_task: Optional[str] = None
        self.current_context: str = ""
        self.last_task_for_curriculum: Optional[str] = None
        self.last_critique: str = ""
        self.trace: List[Dict[str, str]] = []

        self.subgoal_steps: int = 0

        self.memory_paths = ["memory.json"]

    def __del__(self):
        if getattr(self, "_log_fh", None) is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass

    def _log(self, text: str):
        if not self.cfg.debug:
            return
        if self.cfg.debug_also_print:
            print(text)
        if self._log_fh is not None:
            self._log_fh.write(text + "\n")
            self._log_fh.flush()

    def _dbg(self, title: str, content: str):
        if not self.cfg.debug:
            return
        self._log(f"\n\033[36m[{title}]\033[0m")
        self._log(self.cfg._cap(content))

    def _dbg_json(self, title: str, obj: Any):
        if not self.cfg.debug:
            return
        try:
            s = json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            s = str(obj)
        self._dbg(title, s)

    def _parse_curriculum(self, text: str) -> Tuple[str, str]:
        parsed = self.cfg.json_parser(text)
        task = "Explore and gather information"
        context = "Take a step to gather more information."
        if isinstance(parsed, dict):
            task = str(parsed.get("task", "")).strip() or task
            context = str(parsed.get("context", "")).strip() or context
        return task, context

    def _parse_critic(self, text: str) -> Tuple[bool, str]:
        parsed = self.cfg.json_parser(text)
        if not isinstance(parsed, dict):
            return False, ""
        return bool(parsed.get("success", False)), str(parsed.get("critique", "")).strip()

    def _parse_skill(self, text: str) -> Optional[Skill]:
        parsed = self.cfg.json_parser(text)
        if not isinstance(parsed, dict):
            return None
        name = str(parsed.get("skill_name", "")).strip()
        desc = str(parsed.get("skill_desc", "")).strip()
        body = str(parsed.get("skill_body", "")).strip()
        if not name or not desc or not body:
            return None
        return Skill(name=name, desc=desc, body=body)

    def _retrieve_skills(self, observation: str) -> List[Skill]:
        q = (
            f"TASK={self.current_task}\nContext={self.current_context}\nObservation={observation}"
        )
        skills = self.skill_lib.retrieve(query=q, top_k=self.cfg.retrieval_top_k)
        if self.cfg.debug:
            self._dbg("SkillRetrieval.query", q)
            self._dbg("SkillRetrieval.results", "\n".join([f"- {s.name}: {s.desc}" for s in skills]) or "(none)")
        return skills

    def _call_curriculum(self, observation: str) -> Dict[str, Any]:
        user_prompt = self.cfg.curriculum_user_prompt(
            observation=observation,
            completed_tasks=self.progress.completed_tasks,
            failed_tasks=self.progress.failed_tasks,
            last_task=self.last_task_for_curriculum,
            last_critique=self.last_critique,
        )
        if self.cfg.debug_show_system_prompts:
            self._dbg("Curriculum.system_prompt", self.cfg.curriculum_system_prompt)
        if self.cfg.debug_show_user_prompts:
            self._dbg("Curriculum.user_prompt", user_prompt)

        out = self.llm.generate(user_prompt=user_prompt, system_prompt=self.cfg.curriculum_system_prompt)
        if self.cfg.debug_show_raw_outputs:
            self._dbg("Curriculum.raw_output", out["response"])
            self._dbg_json("Curriculum.tokens", {"in": out.get("num_input_tokens"), "out": out.get("num_output_tokens")})

        task, context = self._parse_curriculum(out["response"])
        self.current_task, self.current_context =  task, context
        self.last_task_for_curriculum = task
        self.last_critique = ""

        # reset subgoal step counter when a new task is proposed
        self.subgoal_steps = 0

        if self.cfg.debug:
            self._dbg_json("Curriculum.parsed", {"task": task, "context": context})
        return out

    def _call_critic(self, observation: str, retrieved_skills: List[Skill]) -> Tuple[bool, str, Dict[str, Any]]:
        user_prompt = self.cfg.critic_user_prompt(
            task=self.current_task or "",
            context=self.current_context,
            observation=observation,
            retrieved_skills=retrieved_skills,
        )
        if self.cfg.debug_show_system_prompts:
            self._dbg("Critic.system_prompt", self.cfg.critic_system_prompt)
        if self.cfg.debug_show_user_prompts:
            self._dbg("Critic.user_prompt", user_prompt)

        out = self.llm.generate(user_prompt=user_prompt, system_prompt=self.cfg.critic_system_prompt)
        if self.cfg.debug_show_raw_outputs:
            self._dbg("Critic.raw_output", out["response"])
            self._dbg_json("Critic.tokens", {"in": out.get("num_input_tokens"), "out": out.get("num_output_tokens")})

        success, critique = self._parse_critic(out["response"])
        if self.cfg.debug:
            self._dbg_json("Critic.parsed", {"success": success, "critique": critique})
        return success, critique, out

    def _call_skill_distiller(self):
        if not self.current_task or not self.trace:
            return
        recent_trace = self.trace[-self.cfg.distill_window:]
        user_prompt = self.cfg.skill_user_prompt(
            task=self.current_task,
            context=self.current_context,
            trace=recent_trace,
        )
        if self.cfg.debug_show_system_prompts:
            self._dbg("SkillDistiller.system_prompt", self.cfg.skill_system_prompt)
        if self.cfg.debug_show_user_prompts:
            self._dbg("SkillDistiller.user_prompt", user_prompt)

        out = self.llm.generate(user_prompt=user_prompt, system_prompt=self.cfg.skill_system_prompt)
        if self.cfg.debug_show_raw_outputs:
            self._dbg("SkillDistiller.raw_output", out["response"])
            self._dbg_json("SkillDistiller.tokens", {"in": out.get("num_input_tokens"), "out": out.get("num_output_tokens")})

        skill = self._parse_skill(out["response"])
        if self.cfg.debug:
            self._dbg_json("SkillDistiller.parsed", None if skill is None else skill.__dict__)
        if skill is not None:
            self.skill_lib.add(skill)
            if self.cfg.debug:
                self._dbg("SkillLibrary.added", skill.name)

    def _call_action(self, observation: str, retrieved_skills: List[Skill]) -> Tuple[str, Dict[str, Any]]:
        user_prompt = self.cfg.action_user_prompt(
            task=self.current_task or "",
            context=self.current_context,
            observation=observation,
            retrieved_skills=retrieved_skills,
            critique=self.last_critique,
            memory=None,
        )
        if self.cfg.debug_show_system_prompts:
            self._dbg("Action.system_prompt", self.cfg.system_prompt)
        if self.cfg.debug_show_user_prompts:
            self._dbg("Action.user_prompt", user_prompt)

        out = self.llm.generate(user_prompt=user_prompt, system_prompt=self.cfg.system_prompt)
        if self.cfg.debug_show_raw_outputs:
            self._dbg("Action.raw_output", out["response"])
            self._dbg_json("Action.tokens", {"in": out.get("num_input_tokens"), "out": out.get("num_output_tokens")})

        parsed = self.cfg.response_parser(out["response"])
        action = parsed["action"]
        readable = self.cfg.format_response(parsed)

        if self.cfg.debug:
            self._dbg_json("Action.parsed", {"action": action, "readable": readable})

        out["readable"] = readable
        return action, out

    def _act(self, obs: Dict) -> Tuple[str, int, int, str]:
        observation = obs["text"]
        if self.cfg.debug:
            self._dbg("Env.observation", observation)

        # 0) Ensure we have a task
        new_task_this_step = False
        if self.current_task is None:
            self._call_curriculum(observation)
            new_task_this_step = True

        # 1) Critic checks completion of the current task based on the latest observation.
        if (not new_task_this_step) and (self.current_task is not None) and (self.subgoal_steps > 0):
            success, critique, _ = self._call_critic(observation, retrieved_skills=[])
            if success:
                self.progress.mark_success(self.current_task or "UNKNOWN_TASK")
                if self.cfg.debug:
                    self._dbg("Task.success", f"Completed: {self.current_task}")
                self._call_skill_distiller()

                # Immediately get a new task and reset counters
                self._call_curriculum(observation)
            else:
                if critique:
                    self.last_critique = critique

        # 2) Abandon task if exceeded retry budget
        if self.current_task is not None and self.subgoal_steps >= self.cfg.max_subgoal_steps:
            abandoned = self.current_task
            self.progress.mark_failure(abandoned or "UNKNOWN_TASK")
            if self.cfg.debug:
                self._dbg("Task.abandon", f"Abandoning task after {self.subgoal_steps} attempts: {abandoned}")

            # force new task
            self.current_task = None
            self.current_context = ""
            self.last_task_for_curriculum = abandoned
            self.last_critique = ""
            self._call_curriculum(observation)  # resets subgoal_steps to 0

        # 3) retrieve skills for the task
        retrieved_skills = self._retrieve_skills(observation)

        # 4) Act
        action, out = self._call_action(observation, retrieved_skills)

        # 5) Log trace
        self.trace.append(
            {
                "obs": observation,
                "lm_response": out.get("readable", out.get("response", "")),
                "action": action,
            }
        )

        # 6) Increment attempts after taking an action decision
        self.subgoal_steps += 1

        if self.cfg.debug:
            self._dbg_json(
                "Agent.state",
                {
                    "task": self.current_task,
                    "context": self.current_context,
                    "completed_tasks": len(self.progress.completed_tasks),
                    "failed_tasks": len(self.progress.failed_tasks),
                    "num_skills_learned": len(self.skill_lib),
                    "subgoal_steps": self.subgoal_steps,
                },
            )

        return (
            action,
            out.get("num_input_tokens", 0),
            out.get("num_output_tokens", 0),
            out.get("response", ""),
        )

    def save_memory(self, full_memory_dir: str) -> None:
        os.makedirs(full_memory_dir, exist_ok=True)
        path = os.path.join(full_memory_dir, self.memory_paths[0])

        data = {
            "agent_id": self.id,
            "agent_name": self.name,
            "cfg": {
                "llm_name": getattr(self.cfg, "llm_name", None),
                "embed_name": getattr(self.cfg, "embed_name", None),
                "retrieval_top_k": getattr(self.cfg, "retrieval_top_k", None),
                "distill_window": getattr(self.cfg, "distill_window", None),
                "max_subgoal_steps": getattr(self.cfg, "max_subgoal_steps", None),
            },
            "state": {
                "current_task": self.current_task,
                "current_context": self.current_context,
                "last_task_for_curriculum": self.last_task_for_curriculum,
                "last_critique": self.last_critique,
                "subgoal_steps": self.subgoal_steps,
            },
            "progress": self.progress.to_dict(),
            "skill_library": self.skill_lib.to_dict(),
            "trace": list(self.trace),
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
            "embed_name",
            "retrieval_top_k",
            "distill_window",
            "use_action_memory",
            "action_memory_window",
            "max_subgoal_steps",
        ]:
            if k in cfgd and cfgd[k] is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, cfgd[k])

        # restore progress
        self.progress.load_from_dict(data.get("progress", {}) or {})

        # restore skills and rebuild embeddings with current embedder
        self.skill_lib.embedder = self.embedder
        self.skill_lib.load_from_dict(data.get("skill_library", {}) or {})
        self.skill_lib.rebuild_embeddings()

        # restore agent state
        st = data.get("state", {}) or {}
        self.current_task = st.get("current_task", None)
        self.current_context = st.get("current_context", "") or ""
        self.last_task_for_curriculum = st.get("last_task_for_curriculum", None)
        self.last_critique = st.get("last_critique", "") or ""
        self.subgoal_steps = int(st.get("subgoal_steps", 0) or 0)

        self.trace = list(data.get("trace", []) or [])

@lru_cache(maxsize=None)
def create_voyager_agent(Agent: Type):
    class_name = (
        f"VoyagerAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (VoyagerAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )