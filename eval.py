import time
import argparse
import os
import json
import shutil
import random
import copy
from tools.logger import get_logger

from utils import dynamic_load_game_class, get_hardware_info, convert_json_to_jsonl


def _ensure_valid_actions(obs):
    if not isinstance(obs, dict):
        return
    for agent_id, agent_obs in obs.items():
        if not isinstance(agent_obs, dict):
            continue
        valid_actions = agent_obs.get("valid_actions")
        if valid_actions is None:
            agent_obs["valid_actions"] = {}
        elif not isinstance(valid_actions, dict):
            normalized = {"default": list(valid_actions) if isinstance(valid_actions, (list, tuple)) else []}
            agent_obs["valid_actions"] = {k: v for k, v in normalized.items() if v}
        else:
            cleaned = {}
            for key, value in list(valid_actions.items()):
                if value is None:
                    continue
                if not isinstance(value, list):
                    value = list(value) if isinstance(value, (tuple, set)) else [value]
                value = [item for item in value if item is not None and str(item).strip() != ""]
                if value:
                    cleaned[key] = value
            agent_obs["valid_actions"] = cleaned

def build_agent_config(agent_type, common_cfg_kwargs):
    # Lazy-import agent config classes to avoid importing heavy/optional provider SDKs
    if agent_type in {"VanillaRAGAgent", "Mem0RAGAgent", "RaptorRAGAgent", "VoyagerAgent"}:
        from agents.rag.rag_agent_config import RAGAgentConfig
        return RAGAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"VanillaParamAgent", "LoRASFTAgent"}:
        from agents.parametric.param_agent_config import ParamAgentConfig
        return ParamAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"LongContextAgent", "Mem1Agent", "ShortTermMemoryAgent"}:
        from agents.fixed_size.fixed_size_memory_agent_config import FixedSizeMemoryAgentConfig
        return FixedSizeMemoryAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"MPlusAgent", "MemoryLLMAgent"}:
        from agents.latent.latent_agent_config import LatentAgentConfig
        return LatentAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"NoMemoryAgent", "FakeLLMAgent"}:
        from agents.llm_agent_config import LLMAgentConfig
        return LLMAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"ReplayAgent"}:
        return None
    return None

def instantiate_agent(agent_type, agent_info, AgentCls):
    """Create and return an agent instance of the given *agent_type*."""
    factory_map = {
        "HumanAgent":              ("agents.human_agent",                          "create_human_agent"),
        "VanillaRAGAgent":         ("agents.rag.vanilla_rag_agent",                "create_vanilla_rag_agent"),
        "Mem0RAGAgent":            ("agents.rag.mem0_rag_agent",                   "create_mem0_rag_agent"),
        "RaptorRAGAgent":          ("agents.rag.raptor_rag_agent",                 "create_raptor_rag_agent"),
        "VoyagerAgent":            ("agents.rag.voyager_agent",                    "create_voyager_agent"),
        "LoRASFTAgent":            ("agents.parametric.lora_sft_agent",            "create_lora_sft_agent"),
        "MPlusAgent":              ("agents.latent.mplus_agent",                   "create_mplus_agent"),
        "MemoryLLMAgent":          ("agents.latent.memoryllm_agent",               "create_memoryllm_agent"),
        "LongContextAgent":        ("agents.long_context_agent",                   "create_long_context_agent"),
        "ShortTermMemoryAgent":    ("agents.fixed_size.short_term_memory_agent",   "create_short_term_memory_agent"),
        "Mem1Agent":               ("agents.fixed_size.mem1_agent",               "create_mem1_agent"),
        "NoMemoryAgent":           ("agents.no_memory_agent",                      "create_no_memory_agent"),
        "RandomAgent":             ("agents.random_agent",                         "create_random_agent"),
        "WaitAgent":               ("agents.wait_agent",                           "create_wait_agent"),
        "FakeLLMAgent":            ("agents.fake_llm_agent",                       "create_fake_llm_agent"),
        "ReplayAgent":             ("agents.replay_agent",                         "create_replay_agent"),
    }
    if agent_type not in factory_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    module_path, factory_name = factory_map[agent_type]
    import importlib
    mod = importlib.import_module(module_path)
    factory_fn = getattr(mod, factory_name)
    DerivedCls = factory_fn(AgentCls)
    return DerivedCls(**agent_info)

def load_agents_config(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["agents"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # agents configuration file - for multi-agent settings, not required for single-agent runs
    parser.add_argument("--agents_config", type=str, default=None,
                        help="Path to a JSON file for agents configurations. "
                             "When provided, single-agent CLI flags (--agent, --agent_id, "
                             "--agent_name, --llm_name, --enable_*) are ignored.")

    # single-agent flags (used when --agents_config is not given)
    parser.add_argument("--agent", type=str, default="HumanAgent", help="The agent to evaluate")
    parser.add_argument("--agent_id", type=str, default="agent_adam_davis", help="Unique id for the agent")
    parser.add_argument("--agent_name", type=str, default="adam_davis", help="Human readable agent name")
    parser.add_argument("--llm_provider", type=str, default=None, choices=["openai", "huggingface", "vllm", "azure", "azure_openai", "claude", "gemini"], help="Which LLM provider to use")
    parser.add_argument("--llm_name", type=str, default=None, help="Name of the LLM to use")
    parser.add_argument("--enable_short_term_memory", action="store_true")
    parser.add_argument("--short_term_memory_size", type=int, default=5)
    parser.add_argument("--enable_reflection", action="store_true")
    parser.add_argument("--enable_summarization", action="store_true")

    # env flags
    parser.add_argument("--game_name", type=str, default="base", help="Which game to run: base or a folder under games/generated/")
    parser.add_argument("--world_definition_path", type=str, default=None, help="Path to the world definition JSON file, auto routing if not specified")
    parser.add_argument("--env_config_path", type=str, default=None, help="Path to the initial env config JSON/JSONL file, auto routing if not specified")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to the general output directory")
    parser.add_argument("--run_dir", type=str, default=None, help="Path storing the current run's data; if None, a new directory will be created under output_dir")
    parser.add_argument("--extra_dir", type=str, default=None, help="Adding an extra directory level under run_dir for multiple runs over same configuration")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite the run directory if it exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum number of steps to run in the environment")
    parser.add_argument("--enforce_same_hardware", action="store_true", help="Whether to enforce the same hardware for resumed runs")
    parser.add_argument("--enable_obs_valid_actions", action="store_true", help="Whether to include valid actions in the observation")  # required for RandomAgent
    parser.add_argument("--cumulative_config_save", action="store_true", help="Save cumulative env config each step")
    parser.add_argument("--debug", action="store_true", help="(Deprecated) Alias for enabling both --cumulative_agent_log and --cumulative_config_save")
    parser.add_argument("--resume_from_step", type=int, default=None, help="If specified, resume from the given step number")
    parser.add_argument("--save_dep_graph_steps", type=int, default=None, help="Number of steps before a new dependency graph will be saved; if None, dependency tracking will be disabled")

    parser.add_argument("--memory_dir", type=str, default="memory", help="Directory to save agent memory checkpoints under run_dir")
    parser.add_argument("--agent_memory_save_frequency", type=int, default=5,
                        help="Save agent memory every N environment steps. If None, disabled.")
    args = parser.parse_args()

    if args.debug:
        args.cumulative_config_save = True

    if args.game_name in (None, "", "base"):
        world_definition_path = f"assets/world_definitions/base/default.json"
        env_config_path = f"assets/env_configs/base/initial.json"
    else:
        world_definition_path = f"assets/world_definitions/generated/{args.game_name}/default.json"
        env_config_path = f"assets/env_configs/generated/{args.game_name}/initial.json"

    if args.world_definition_path is not None:
        world_definition_path = args.world_definition_path
    if args.env_config_path is not None:
        env_config_path = args.env_config_path

    logger = get_logger("EvalLogger")

    if args.agents_config is not None:
        agent_specs = load_agents_config(args.agents_config)
    else:
        agent_specs = [{
            "agent_type": args.agent,
            "agent_id": args.agent_id,
            "agent_name": args.agent_name,
            "llm_name": args.llm_name,
            "llm_provider": args.llm_provider,
            "enable_short_term_memory": args.enable_short_term_memory,
            "short_term_memory_size": args.short_term_memory_size,
            "enable_reflection": args.enable_reflection,
            "enable_summarization": args.enable_summarization,
        }]

    run_dir = args.run_dir if args.run_dir is not None else os.path.join(args.output_dir, "game_" + args.game_name)
    if args.agents_config is None:
        if args.llm_name is not None:
            run_dir = os.path.join(run_dir, args.llm_name.replace("/", "_"))
        run_dir = os.path.join(run_dir, args.agent)
        if args.enable_short_term_memory:
            run_dir = os.path.join(run_dir, "with_short_term_memory")
        elif args.enable_reflection:
            run_dir = os.path.join(run_dir, "with_reflection")
        elif args.enable_summarization:
            run_dir = os.path.join(run_dir, "with_summarization")
        else:
            run_dir = os.path.join(run_dir, "no_extras")
    if args.extra_dir is not None:
        run_dir = os.path.join(run_dir, args.extra_dir)

    if args.resume_from_step and (not os.path.exists(run_dir) or not args.cumulative_config_save):
        raise ValueError("To resume from a specific step, the run_dir must exist and --cumulative_config_save must be enabled.")
    if args.overwrite and os.path.exists(run_dir):
        logger.warning(f"Overwriting the run directory: {run_dir}")
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    AgentCls = dynamic_load_game_class(args.game_name, "agent", "Agent")
    agents = []
    agent_dirs = {}
    agent_log_paths = {}

    for spec in agent_specs:
        a_type = spec["agent_type"]
        a_id   = spec["agent_id"]
        a_name = spec["agent_name"]
        a_llm  = spec.get("llm_name")
        a_prov = spec.get("llm_provider")
        a_stm  = spec.get("enable_short_term_memory", False)
        a_stms = spec.get("short_term_memory_size", 5)
        a_ref  = spec.get("enable_reflection", False)
        a_sum  = spec.get("enable_summarization", False)

        agent_dir = os.path.join(run_dir, a_id)
        os.makedirs(os.path.join(agent_dir, args.memory_dir), exist_ok=True)
        agent_dirs[a_id] = agent_dir
        agent_log_paths[a_id] = os.path.join(agent_dir, "agent_log.jsonl")

        common_cfg_kwargs = dict(
            llm_name=a_llm,
            llm_provider=a_prov,
            enable_reflection=a_ref,
            enable_summarization=a_sum,
            enable_short_term_memory=a_stm,
            short_term_memory_size=a_stms,
            full_mem_path=os.path.join(agent_dir, args.memory_dir),
        )

        cfg = build_agent_config(a_type, common_cfg_kwargs)
        agent_info = {"id": a_id, "name": a_name}
        if cfg is not None:
            agent_info["cfg"] = cfg
            logger.info(f"Agent {a_id} ({a_type}) config: {cfg}")

        agent = instantiate_agent(a_type, agent_info, AgentCls)
        agents.append(agent)

    logger.info(f"Created {len(agents)} agent(s): {[a.id for a in agents]}")
    for a_id, a_dir in agent_dirs.items():
        logger.info(f"{a_id} memory dir: {os.path.join(a_dir, args.memory_dir)}")
    logger.info(f"Agent memory save frequency: {args.agent_memory_save_frequency}")

    config_path = os.path.join(run_dir, "config.json")
    if args.cumulative_config_save:
        config_path = os.path.join(run_dir, "config.jsonl")

    from_config = False
    if not os.path.exists(config_path):
        seed_config_path = env_config_path
        logger.info(f"Initiating new game from the world config: {world_definition_path}")
        logger.info(f"Initiating new game from the environment config: {seed_config_path}")
        if args.cumulative_config_save:
            convert_json_to_jsonl(seed_config_path, config_path)
        else:
            shutil.copy(seed_config_path, config_path)
    else:
        logger.info(f"Continuing the game from config: {config_path}")
        from_config = True

    args.seed = args.seed if args.seed is not None else random.randint(0, 10000)

    EnvCls = dynamic_load_game_class(args.game_name, "env", "AgentOdysseyEnv")
    env = EnvCls(
        seed=args.seed,
        agents=agents,
        world_definition_path=world_definition_path,
        run_dir=run_dir,
        config_path=config_path,
        enable_obs_valid_actions=args.enable_obs_valid_actions,
        from_step=args.resume_from_step,
        save_dep_graph_steps=args.save_dep_graph_steps,
    )
    
    this_hardware = get_hardware_info()
    if not args.overwrite and args.enforce_same_hardware and "hardware" in env.config:
        saved_hardware = env.config["hardware"]
        for key in this_hardware:
            if this_hardware[key] != saved_hardware.get(key, None):
                raise ValueError(f"The current hardware {key}: {this_hardware[key]} does not match the saved hardware {key}: {saved_hardware[key]}. Cannot resume the run.")
        logger.info("The current hardware matches the saved hardware. Resuming the run...")
    else:
        env.config["hardware"] = this_hardware

    if not args.overwrite:
        for arg_name, arg_val in [("game_name", args.game_name), ("world_definition_path", world_definition_path), ("seed", args.seed)]:
            if arg_name in env.config and env.config[arg_name] != arg_val:
                raise ValueError(f"The current {arg_name}: {arg_val} does not match the saved {arg_name}: {env.config[arg_name]}. Cannot resume the run.")
    env.config["game_name"] = args.game_name
    env.config["world_definition_path"] = world_definition_path
    env.config["seed"] = args.seed
    env.update_config(update_env_config=False, hardware=env.config["hardware"], update_file=False, cumulative=args.cumulative_config_save)

    obs = env.reset(from_config=from_config)
    env.update_config(update_env_config=True, update_file=False, cumulative=args.cumulative_config_save)
    if args.enable_obs_valid_actions:
        _ensure_valid_actions(obs)

    # load per-agent memory from each agent's own directory
    for agent in env.agents:
        if hasattr(agent, "load_memory"):
            a_dir = agent_dirs[agent.id]
            full_memory_paths = [os.path.join(a_dir, args.memory_dir, mp) for mp in agent.memory_paths]
            if all(os.path.exists(p) for p in full_memory_paths):
                agent.load_memory(full_memory_dir=os.path.join(a_dir, args.memory_dir))
                logger.info(f"Loaded memory for agent {agent.id} from {full_memory_paths}.")
            else:
                logger.info(f"No previous memory found for agent {agent.id}. Starting fresh.")

    while True:
        action_strs = {}
        agents_log = {}
        # collect per-agent outputs from act()
        for agent in env.agents:
            start_time = time.perf_counter()
            action_strs[agent.id], num_input_tokens, num_output_tokens, response = agent.act(obs[agent.id])
            end_time = time.perf_counter()
            decision_time = end_time - start_time
            agents_log[agent.id] = {
                "observation": obs[agent.id],
                "action": action_strs[agent.id],
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "decision_time": decision_time,
                "response": response,
            }

        if args.agent_memory_save_frequency is not None and args.agent_memory_save_frequency > 0:
            if env.steps % args.agent_memory_save_frequency == 0:
                for agent in env.agents:
                    a_dir = agent_dirs[agent.id]
                    if hasattr(agent, "save_memory"):
                        agent.save_memory(
                            full_memory_dir=os.path.join(a_dir, args.memory_dir),
                        )
                        logger.info(f"Saving memory for {agent.id} at step {env.steps}")
                    else:
                        logger.warning(f"Agent {agent.id} does not have save_memory method. Skipping memory save.")

        try:
            obs, reward, done, info = env.step(action_strs)
            if args.enable_obs_valid_actions:
                _ensure_valid_actions(obs)
        except Exception as e:
            logger.error(f"Error during env.step: {e}; Agent actions: {action_strs}")
            raise

        # update scores in env
        env.update_scores(reward)

        invalids = info.get("step_invalid_action", {}) if isinstance(info, dict) else {}
        for agent in env.agents:
            log_path = agent_log_paths[agent.id]
            combined = {
                "step": env.steps - 1,
                "action": action_strs.get(agent.id),
                "decision_time": agents_log[agent.id]["decision_time"],
                "num_input_tokens": agents_log[agent.id]["num_input_tokens"],
                "num_output_tokens": agents_log[agent.id]["num_output_tokens"],
                "invalid_action": bool(invalids.get(agent.id, False)),
                "reward": copy.deepcopy(reward[agent.id].__dict__),
                "observation": agents_log[agent.id].get("observation"),
                "response": agents_log[agent.id].get("response"),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(combined) + "\n")

        env.update_config(cumulative=args.cumulative_config_save)

        if done or env.steps >= args.max_steps:
            if args.agent_memory_save_frequency is not None and args.agent_memory_save_frequency > 0:
                for agent in env.agents:
                    a_dir = agent_dirs[agent.id]
                    if hasattr(agent, "save_memory"):
                        agent.save_memory(full_memory_dir=os.path.join(a_dir, args.memory_dir))
                        print(f"Saving the last memory checkpoint for {agent.id} at step: {env.steps}")
            logger.info(f"Episode finished after {env.steps} steps")
            break
