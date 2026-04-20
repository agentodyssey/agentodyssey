"""Generate an cumulative config.jsonl by replaying actions from an agent log.

Given an agent_log.jsonl (recorded actions from a previous run) and the
same game/env setup, this tool re-runs the game with a ReplayAgent that
feeds the recorded actions, producing the cumulative config.jsonl that
would have been created had ``--cumulative_config_save`` been enabled.

Usage example:
    python tools/generate_cumulative_config.py \
        --game_name remnant \
        --agent_log_path output/game_remnant/gpt-5/LongContextAgent/no_extras/0/agent_adam_davis/agent_log.jsonl \
        --save_path output/config.jsonl

The temporary run directory is created under ``tmp/`` and cleaned up
after the config.jsonl is copied to ``--save_path``.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(
        description="Replay an agent log to generate cumulative config.jsonl"
    )
    parser.add_argument(
        "--game_name", type=str, required=True,
        help="Game name, e.g. 'base' or a folder under games/generated/"
    )
    parser.add_argument(
        "--world_definition_path", type=str, default=None,
        help="Path to the world definition JSON file (auto-routed if not specified)"
    )
    parser.add_argument(
        "--env_config_path", type=str, default=None,
        help="Path to the initial env config JSON/JSONL file (auto-routed if not specified)"
    )
    parser.add_argument(
        "--agent_log_path", type=str, required=True,
        help="Path to the agent_log.jsonl file to replay"
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Destination path for the generated config.jsonl"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (should match the original run)"
    )
    args = parser.parse_args()

    # ── Validate inputs ──────────────────────────────────────────────
    agent_log_path = os.path.abspath(args.agent_log_path)
    if not os.path.isfile(agent_log_path):
        sys.exit(f"Error: agent log not found: {agent_log_path}")

    # Infer agent_id from parent directory name (e.g. .../agent_adam_davis/agent_log.jsonl)
    agent_id = os.path.basename(os.path.dirname(agent_log_path))

    # Read agent log to determine the number of steps and agent_name
    with open(agent_log_path, "r") as f:
        log_lines = [line.strip() for line in f if line.strip()]
    if not log_lines:
        sys.exit("Error: agent log is empty")

    num_steps = len(log_lines)

    # Derive agent_name from agent_id (strip "agent_" prefix, replace _ with space)
    agent_name = agent_id
    if agent_name.startswith("agent_"):
        agent_name = agent_name[len("agent_"):]
    agent_name = agent_name.replace("_", " ")

    # ── Prepare temporary run directory ──────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = os.path.join(project_root, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    run_dir = os.path.join(tmp_dir, f"replay_{uuid.uuid4().hex[:12]}")

    # ── Build eval.py command ────────────────────────────────────────
    eval_script = os.path.join(project_root, "eval.py")
    cmd = [
        sys.executable, eval_script,
        "--agent", "ReplayAgent",
        "--agent_id", agent_id,
        "--agent_name", agent_name,
        "--game_name", args.game_name,
        "--run_dir", run_dir,
        "--cumulative_config_save",
        "--max_steps", str(num_steps),
        "--seed", str(args.seed),
        "--overwrite",
    ]
    if args.world_definition_path is not None:
        cmd.extend(["--world_definition_path", args.world_definition_path])
    if args.env_config_path is not None:
        cmd.extend(["--env_config_path", args.env_config_path])

    # Set the replay actions path for ReplayAgent via environment variable
    env = os.environ.copy()
    env["REPLAY_ACTIONS_PATH"] = agent_log_path

    print(f"Replaying {num_steps} actions from: {agent_log_path}")
    print(f"Temporary run dir: {run_dir}")
    print(f"Running: {' '.join(cmd)}")

    # ── Execute eval.py ──────────────────────────────────────────────
    try:
        result = subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        # Clean up on failure too
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        sys.exit(f"eval.py failed with return code {e.returncode}")

    # ── Copy config.jsonl to save_path ───────────────────────────────
    # eval.py appends agent-type subdirs to --run_dir (e.g. ReplayAgent/no_extras/),
    # so we search the tree for the generated config.jsonl.
    config_jsonl = None
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        if "config.jsonl" in filenames:
            config_jsonl = os.path.join(dirpath, "config.jsonl")
            break
    if config_jsonl is None:
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        sys.exit(f"Error: config.jsonl was not generated under {run_dir}")

    save_path = os.path.abspath(args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shutil.copy2(config_jsonl, save_path)
    print(f"Saved cumulative config to: {save_path}")

    # ── Clean up temporary run directory ─────────────────────────────
    shutil.rmtree(run_dir)
    print(f"Cleaned up temporary run dir: {run_dir}")


if __name__ == "__main__":
    main()
