"""Official evaluator for the NeurIPS 2026 TTCL Workshop Challenge.

Aggregates AgentOdyssey run results into the official challenge scores over
the three ranking games: 'remnant', 'mark', and 'metropolis'.

Workshop website and full challenge rules: https://ttcl-agents.github.io/

Ranking rules
-------------
- The ranking score is the unweighted arithmetic mean of the raw main quest
  rewards across the three games (no per-game normalization).
- Ties are broken by the unweighted arithmetic mean of the raw total
  supplementary rewards across the three games.
- The supplementary reward is the sum of four components:
    * side quest   — number of completed side quests
    * exploration  — number of explored areas
    * craft        — number of UNIQUE object types crafted
    * defeat       — number of UNIQUE NPCs defeated
- Each game evaluation is standalone: agent memory and learned state must be
  reset before evaluating each game. The maximum number of environment steps
  is max_steps = 500.
- If a game is evaluated with multiple runs, the reported score for that game
  is the mean over the runs (selecting the best run is not allowed).

Usage
-----
Each run directory is the --run_dir of an eval.py / `agentodyssey run`
invocation (the directory containing config.json or config.jsonl):

    python challenge_eval.py \
        --remnant    output/game_remnant/<...>/<run> \
        --mark       output/game_mark/<...>/<run> \
        --metropolis output/game_metropolis/<...>/<run>

Pass several run directories per game to average over multiple runs, and
--json to emit a machine-readable report.

This script only needs the Python standard library, so it can be run on any
copy of the run directories without installing AgentOdyssey.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

RANKING_GAMES = ["remnant", "mark", "metropolis"]
MAX_STEPS = 500

# scores-field -> official supplementary component name
SUPPLEMENTARY_COMPONENTS = {
    "side_quest": "side quest",
    "exploration": "exploration",
    "craft": "craft",
    "defeat": "defeat",
}


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_run_config(run_dir: str) -> dict:
    """Load the final environment config of a run (config.json or config.jsonl)."""
    json_path = os.path.join(run_dir, "config.json")
    jsonl_path = os.path.join(run_dir, "config.jsonl")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    if os.path.exists(jsonl_path):
        last_line = None
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line is None:
            _fail(f"{jsonl_path} is empty.")
        return json.loads(last_line)
    _fail(f"No config.json or config.jsonl found in run directory: {run_dir}")


def select_agent_id(config: dict, run_dir: str, agent_id: str | None) -> str:
    scores = config.get("scores") or {}
    if not scores:
        _fail(
            f"Run {run_dir} has no recorded scores "
            "(the run may not have progressed past step 0)."
        )
    if agent_id is not None:
        if agent_id not in scores:
            _fail(
                f"Agent id '{agent_id}' not found in {run_dir} "
                f"(available: {', '.join(sorted(scores))})."
            )
        return agent_id
    if len(scores) > 1:
        _fail(
            f"Run {run_dir} contains multiple agents "
            f"({', '.join(sorted(scores))}); disambiguate with --agent_id."
        )
    return next(iter(scores))


def extract_run_metrics(game: str, run_dir: str, agent_id: str | None) -> dict:
    """Extract the official challenge metrics for a single run."""
    config = load_run_config(run_dir)

    config_game = config.get("game_name")
    if config_game is not None and config_game != game:
        _fail(
            f"Run {run_dir} was recorded for game '{config_game}', "
            f"but was passed as a run of '{game}'."
        )

    steps = int(config.get("step", 0))
    warnings = []
    if steps > MAX_STEPS:
        warnings.append(
            f"run has {steps} environment steps, which exceeds the "
            f"challenge limit of max_steps = {MAX_STEPS}"
        )

    aid = select_agent_id(config, run_dir, agent_id)
    scores = config["scores"][aid]

    main_quest = float(scores.get("quest", 0))
    supplementary = {
        "side quest": float(scores.get("side_quest", 0)),
        "exploration": float(scores.get("exploration", 0)),
        "craft": float(scores.get("craft", 0)),
        "defeat": float(scores.get("unique_kill", 0)),
    }

    # consistency checks against the recorded agent state (unique counts)
    state = config.get("curr_agents_state") or {}
    state_checks = {
        "side quest": len((state.get("side_quest_completed_tasks") or {}).get(aid, []) or []),
        "craft": len((state.get("objects_crafted") or {}).get(aid, {}) or {}),
        "defeat": len((state.get("unique_npcs_killed") or {}).get(aid, []) or []),
    }
    for name, state_value in state_checks.items():
        if supplementary[name] != state_value:
            warnings.append(
                f"accumulated '{name}' reward ({supplementary[name]:g}) does not match "
                f"the recorded agent state ({state_value}); using the accumulated reward"
            )

    return {
        "run_dir": run_dir,
        "agent_id": aid,
        "steps": steps,
        "main_quest": main_quest,
        "supplementary": supplementary,
        "supplementary_total": sum(supplementary.values()),
        "warnings": warnings,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_game(game: str, run_dirs: list[str], agent_id: str | None) -> dict:
    runs = [extract_run_metrics(game, rd, agent_id) for rd in run_dirs]
    return {
        "game": game,
        "num_runs": len(runs),
        "runs": runs,
        "main_quest": mean([r["main_quest"] for r in runs]),
        "supplementary": {
            name: mean([r["supplementary"][name] for r in runs])
            for name in SUPPLEMENTARY_COMPONENTS.values()
        },
        "supplementary_total": mean([r["supplementary_total"] for r in runs]),
    }


def print_report(games: dict, aggregate: dict) -> None:
    header = (
        f"{'Game':<12} {'Runs':>4} {'Main Quest':>11} {'Side Quest':>11} "
        f"{'Exploration':>12} {'Craft':>6} {'Defeat':>7} {'Suppl. Total':>13}"
    )
    rule = "-" * len(header)

    print()
    print("NeurIPS 2026 TTCL Workshop Challenge — Official Evaluation")
    print(rule)
    print(header)
    print(rule)
    for game in RANKING_GAMES:
        g = games.get(game)
        if g is None:
            print(f"{game:<12} {'-':>4} {'-':>11} {'-':>11} {'-':>12} {'-':>6} {'-':>7} {'-':>13}")
            continue
        s = g["supplementary"]
        print(
            f"{game:<12} {g['num_runs']:>4} {g['main_quest']:>11.3f} {s['side quest']:>11.3f} "
            f"{s['exploration']:>12.3f} {s['craft']:>6.3f} {s['defeat']:>7.3f} "
            f"{g['supplementary_total']:>13.3f}"
        )
    print(rule)

    if aggregate is not None:
        print(f"Ranking score      (mean main quest reward over 3 games): {aggregate['main_quest_mean']:.3f}")
        print(f"Tiebreaker (mean total supplementary reward over 3 games): {aggregate['supplementary_total_mean']:.3f}")
    else:
        missing = [g for g in RANKING_GAMES if g not in games]
        print(
            "Partial evaluation only — no ranking score. "
            f"Missing game(s): {', '.join(missing)}."
        )
    print(rule)

    for game in RANKING_GAMES:
        g = games.get(game)
        if g is None:
            continue
        for run in g["runs"]:
            for w in run["warnings"]:
                print(f"WARNING [{game} @ {run['run_dir']}]: {w}")

    print(
        "\nReminder: agent memory and learned state must be reset before each game,\n"
        "runs must use max_steps = 500 with a Qwen3-4B or Qwen3.5-4B backbone, and\n"
        "multiple runs of a game must all be reported (means, not best-of-N).\n"
        "Report per-game main quest and supplementary rewards in your paper in\n"
        "addition to the three-game means.\n"
        "Full challenge rules: https://ttcl-agents.github.io/"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official three-game score aggregation for the TTCL Challenge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[0],
    )
    for game in RANKING_GAMES:
        parser.add_argument(
            f"--{game}",
            nargs="+",
            metavar="RUN_DIR",
            default=None,
            help=f"One or more run directories for the '{game}' game "
                 "(multiple runs are averaged).",
        )
    parser.add_argument(
        "--agent_id",
        default=None,
        help="Agent id to score when a run contains multiple agents.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of a table.",
    )
    args = parser.parse_args()

    provided = {game: getattr(args, game) for game in RANKING_GAMES if getattr(args, game)}
    if not provided:
        parser.error(
            "no run directories given; pass at least one of "
            "--remnant / --mark / --metropolis."
        )

    games = {
        game: aggregate_game(game, run_dirs, args.agent_id)
        for game, run_dirs in provided.items()
    }

    aggregate = None
    if all(game in games for game in RANKING_GAMES):
        aggregate = {
            "main_quest_mean": mean([games[g]["main_quest"] for g in RANKING_GAMES]),
            "supplementary_total_mean": mean(
                [games[g]["supplementary_total"] for g in RANKING_GAMES]
            ),
        }

    if args.json:
        print(json.dumps({"games": games, "aggregate": aggregate}, indent=2))
    else:
        print_report(games, aggregate)


if __name__ == "__main__":
    main()
