"""Simplified top-level API for AgentOdyssey.

Python usage:
    from agentodyssey import AgentOdyssey

    game = AgentOdyssey.generate("a pirate-themed island adventure")
    game = AgentOdyssey.generate()
    game.run()

    AgentOdyssey.run(agent="LongContextAgent", llm_name="gpt-5", game_name="haunted")

CLI usage (after `pip install -e .`):
    agentodyssey generate "a pirate-themed island adventure"
    agentodyssey generate
    agentodyssey generate --game-name haunted --num-places 4 --num-objects 20
    agentodyssey run --agent LongContextAgent --llm-name gpt-5 --game-name haunted
    agentodyssey run --agent HumanAgent
"""
from __future__ import annotations

import json
import os                                                                                                                                                                                                                                                                                                                                                                     
import sys
import subprocess
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)

PROVIDER_MAP = {
    "openai": ("providers.openai_api", "OpenAILanguageModel"),
    "azure": ("providers.azure_api", "AzureLanguageModel"),
    "azure_openai": ("providers.azure_openai_api", "AzureOpenAILanguageModel"),
    "claude": ("providers.claude_api", "ClaudeLanguageModel"),
    "gemini": ("providers.gemini_api", "GeminiLanguageModel"),
    "vllm": ("providers.vllm", "vllmLanguageModel"),
    "huggingface": ("providers.huggingface", "hfLanguageModel"),
}

def _get_llm(provider: str, model: str):
    if provider not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Choose from: {', '.join(sorted(PROVIDER_MAP))}"
        )
    module_path, class_name = PROVIDER_MAP[provider]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(llm_name=model)

@dataclass
class GeneratedGame:
    game_name: str
    game_dir: str
    assets_dir: str
    world_definition_path: str
    env_config_path: str
    description: str

    def __repr__(self) -> str:
        return (
            f"GeneratedGame(name={self.game_name!r}, "
            f"world={self.world_definition_path!r})"
        )

    def run(self, agent: str = "HumanAgent", **kwargs):
        return AgentOdyssey.run(agent=agent, game_name=self.game_name, **kwargs)

class AgentOdyssey:
    @staticmethod
    def _save_generation_metadata(
        *,
        game_name: str,
        assets_dir: str,
        world_definition_path: str,
        env_config_path: str,
        description: str,
        game_name_was_auto_generated: bool,
        generation_parameters: dict,
    ) -> str:
        metadata = {
            "game_name": game_name,
            "description": description,
            "game_name_was_auto_generated": game_name_was_auto_generated,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "paths": {
                "world_definitions_dir": assets_dir,
                "world_definition_path": world_definition_path,
                "env_config_path": env_config_path,
            },
            "generation_parameters": generation_parameters,
        }

        metadata_path = os.path.join(current_directory, assets_dir, "metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata_path

    @staticmethod
    def generate(
        description: str | None = None,
        game_name: str | None = None,
        *,
        num_places: int = 2,
        num_objects: int = 10,
        num_npcs: int = 5,
        max_level: int = 5,
        num_quest_chapters: int = 1,
        quest_description: str | None = None,
        branching_factor: int = 1,
        num_action_rules: int = 0,
        num_step_rules: int = 0,
        new_action_rules: list[str] | None = None,
        new_step_rules: list[str] | None = None,
        llm_name: str = "gpt-5",
        llm_provider: str = "openai",
        backend: str = "llm",
        generate_graph: bool = False,
        overwrite: bool = False,
        skip_test: bool = False,
        verbose: bool = True,
    ) -> GeneratedGame:
        """Generate a complete game world in one call.

        Parameters
        ----------
        description : str, optional
            Natural-language description of the game theme.
            If ``None``, a default fantasy world is generated.
        game_name : str, optional
            Identifier for the generated game (used as folder name under
            ``games/generated/``, ``assets/env_configs/generated/``, and ``assets/world_definitions/generated/``).
            Auto-generated from the description if not provided.
        num_places : int
            Number of places (each with 2-4 areas) to generate.
        num_objects : int
            Number of objects (weapons, armour, materials, …) to generate.
        num_npcs : int
            Number of NPCs (merchants, enemies, quest givers, …) to generate.
        max_level : int
            Maximum difficulty level for generated entities.
        num_quest_chapters : int
            Number of main-quest chapters to generate.
        quest_description : str, optional
            Theme / description specifically for the quest.
            Falls back to ``description`` if not provided.
        branching_factor : int
            Controls goal structure within each quest chapter.
            ``1`` (default) produces sequential (linear) quest stages.
            ``2`` or higher produces a **goal tree** per chapter where
            each internal node can have up to *branching_factor* children.
            Leaf goals are broadcast first; parent goals unlock bottom-up
            as their children are completed.
        num_action_rules : int
            Number of action rules to auto-generate (the LLM suggests
            novel rules, then aider implements them).  ``0`` to skip.
        num_step_rules : int
            Number of step rules to auto-generate.  ``0`` to skip.
        new_action_rules : list[str], optional
            Explicit descriptions of new player-actions to add (e.g.
            ``["allow the player to fish in water areas"]``).
        new_step_rules : list[str], optional
            Explicit descriptions of new automatic step-rules to add (e.g.
            ``["weather system that affects combat"]``).
        llm_name : str
            LLM model to use for generation (default ``"gpt-5"``).
        llm_provider : str
            LLM provider to use: ``"openai"`` (default), ``"azure"``,
            ``"azure_openai"``, ``"claude"``, ``"gemini"``, ``"vllm"``,
            or ``"huggingface"``.
        backend : str
            Backend for entity generation: ``"llm"`` (direct) or ``"aider"``
            (code-aware with iterative testing).
        generate_graph : bool
            Also generate a predefined area-connection graph in the world
            definition (topology-aware, supports quest progression).
        overwrite : bool
            Overwrite an existing game with the same name.
        skip_test : bool
            Skip the automated smoke-test after generation.
        verbose : bool
            Print progress information.

        Returns
        -------
        GeneratedGame
            Dataclass with paths to the generated game, ready to be passed
            to :pymeth:`AgentOdyssey.run`.
        """
        from tools.generators.rule_generator import ensure_generated_game
        from tools.generators.entity_generator import (
            ensure_generated_assets,
            generate_entities,
            merge_entities,
            validate_entities,
            extract_game_rules,
            analyze_difficulty_progression,
            print_generation_summary,
            test_game as test_game_entities,
            generate_with_aider as generate_entities_aider,
        )

        game_name_was_auto_generated = game_name is None

        if description is None:
            if verbose:
                print("No description provided — asking the LLM for a creative world idea…")
            llm = _get_llm(llm_provider, llm_name)

            if game_name is None:
                import json as _json
                resp = llm.generate(
                    user_prompt=(
                        "Propose a single, vivid, and creative game-world concept for a "
                        "text-based RPG (think: 'Resident Evil-like survival horror on a "
                        "cursed ocean liner' or 'solarpunk city where plants are currency').\n\n"
                        "Also propose a short, catchy game name (suitable as a "
                        "folder name — a single lowercase word, no spaces, no underscores, "
                        "no special "
                        "characters).\n\n"
                        'Reply with ONLY a JSON object like:\n'
                        '{"description": "...", "game_name": "..."}\n'
                        "Nothing else."
                    ),
                    system_prompt="You are a world-building expert. Be imaginative and concise.",
                )
                raw = resp["response"].strip()
                try:
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                    data = _json.loads(raw)
                    description = str(data.get("description", raw)).strip().strip('"').strip("'")
                    game_name = str(data.get("game_name", "")).strip()
                except (_json.JSONDecodeError, Exception):
                    description = raw.strip().strip('"').strip("'")

                game_name = (game_name or "").strip().strip('"').strip("'").lower()
                if not game_name:
                    game_name = f"game{uuid.uuid4().hex[:8]}"
            else:
                resp = llm.generate(
                    user_prompt=(
                        "Propose a single, vivid, and creative game-world concept for a "
                        "text-based RPG (think: 'Resident Evil-like survival horror on a "
                        "cursed ocean liner' or 'solarpunk city where plants are currency'). "
                        "Reply with ONLY the one-sentence description, nothing else."
                    ),
                    system_prompt="You are a world-building expert. Be imaginative and concise.",
                )
                description = resp["response"].strip().strip('"').strip("'")

            if verbose:
                print(f"  → Description: {description}")
                print(f"  → Game name  : {game_name}")

        elif game_name is None:
            if verbose:
                print("No game name provided — asking the LLM to suggest one…")
            llm = _get_llm(llm_provider, llm_name)
            resp = llm.generate(
                user_prompt=(
                    f"Given this game-world description:\n\n"
                    f'"{description}"\n\n'
                    "Propose a short, catchy game name. "
                    "Reply with ONLY a single lowercase word (no spaces, no underscores, "
                    "letters/numbers only). Nothing else."
                ),
                system_prompt="You are a world-building expert. Be concise.",
            )
            game_name = resp["response"].strip().strip('"').strip("'").lower()
            if not game_name:
                game_name = f"game{uuid.uuid4().hex[:8]}"
            if verbose:
                print(f"  → {game_name}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"  AgentOdyssey.generate()")
            print(f"  Game name  : {game_name}")
            print(f"  Description: {description[:80]}{'…' if len(description) > 80 else ''}")
            print(f"{'='*60}\n")

        if verbose:
            print("[1/4] Initiating generated game…")
        game_dir, assets_dir = ensure_generated_game(game_name, overwrite=overwrite)
        ensure_generated_assets(game_name, overwrite=overwrite)

        world_definition_path = f"assets/world_definitions/generated/{game_name}/default.json"
        env_config_path = f"assets/env_configs/generated/{game_name}/initial.json"

        # ── generation report ──
        import time as _time
        _gen_report: dict = {
            "start_time": _time.time(),
            "stages": {},
        }

        if verbose:
            print(f"[2/4] Generating entities ({num_places} places, {num_objects} objects, {num_npcs} NPCs)...")

        if backend == "aider":
            success, msg = generate_entities_aider(
                game_name=game_name,
                world_definition_path=world_definition_path,
                num_places=num_places,
                num_objects=num_objects,
                num_npcs=num_npcs,
                max_level=max_level,
                model=llm_name,
                description=description,
                provider=llm_provider,
                generate_graph=generate_graph,
            )
            if not success:
                raise RuntimeError(f"Entity generation failed: {msg}")
        else:
            world_def_abs = os.path.join(current_directory, world_definition_path)
            with open(world_def_abs, "r", encoding="utf-8") as f:
                world_data = json.load(f)

            new_entities = generate_entities(
                llm_name=llm_name,
                num_places=num_places,
                num_objects=num_objects,
                num_npcs=num_npcs,
                max_level=max_level,
                world_data=world_data,
                game_name=game_name,
                description=description,
                llm=_get_llm(llm_provider, llm_name),
                generate_graph=generate_graph,
            )
            if verbose:
                print_generation_summary(new_entities, world_data)

            merged = merge_entities(world_data, new_entities)
            output_path = os.path.join(current_directory, world_definition_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)

        _gen_report["stages"]["entities"] = {"status": "ok"}

        metadata_path = AgentOdyssey._save_generation_metadata(
            game_name=game_name,
            assets_dir=assets_dir,
            world_definition_path=world_definition_path,
            env_config_path=env_config_path,
            description=description,
            game_name_was_auto_generated=game_name_was_auto_generated,
            generation_parameters={
                "num_places": num_places,
                "num_objects": num_objects,
                "num_npcs": num_npcs,
                "max_level": max_level,
                "num_quest_chapters": num_quest_chapters,
                "quest_description": quest_description,
                "effective_quest_description": quest_description or description,
                "branching_factor": branching_factor,
                "num_action_rules": num_action_rules,
                "num_step_rules": num_step_rules,
                "new_action_rules": list(new_action_rules or []),
                "new_step_rules": list(new_step_rules or []),
                "llm_name": llm_name,
                "llm_provider": llm_provider,
                "backend": backend,
                "overwrite": overwrite,
                "skip_test": skip_test,
                "verbose": verbose,
            },
        )
        if verbose:
            print(f"[meta] Saved generation metadata: {metadata_path}")

        action_rule_descs: list[str] = list(new_action_rules or [])
        step_rule_descs: list[str] = list(new_step_rules or [])

        if num_action_rules > 0 or num_step_rules > 0:
            from tools.generators.rule_generator import suggest_novel_rule
            all_suggested: list[str] = []  # track everything suggested so far
            if verbose and num_action_rules > 0:
                print(f"[3/4] Auto-suggesting {num_action_rules} action rule(s)...")
            for _ in range(num_action_rules):
                suggested = suggest_novel_rule("action", llm_name, game_name, already_suggested=all_suggested, game_description=description, llm=_get_llm(llm_provider, llm_name))
                if verbose:
                    print(f"  💡 {suggested}")
                action_rule_descs.append(suggested)
                all_suggested.append(suggested)
            if verbose and num_step_rules > 0:
                print(f"[3/4] Auto-suggesting {num_step_rules} step rule(s)...")
            for _ in range(num_step_rules):
                suggested = suggest_novel_rule("step", llm_name, game_name, already_suggested=all_suggested, game_description=description, llm=_get_llm(llm_provider, llm_name))
                if verbose:
                    print(f"  💡 {suggested}")
                step_rule_descs.append(suggested)
                all_suggested.append(suggested)

        total_rules = len(action_rule_descs) + len(step_rule_descs)
        if total_rules > 0:
            if verbose:
                print(f"[3/4] Implementing {total_rules} rule(s)...")
            from tools.generators.rule_generator import generate_rule, suggest_novel_rule as _suggest

            max_retries_per_rule = 2   # re-suggest up to 2 times on failure
            succeeded: list[tuple[str, str]] = []   # (type, desc)
            failed: list[tuple[str, str]] = []      # (type, desc)

            def _try_rule(rule_type: str, desc: str, idx: int, total: int) -> bool:
                # Attempt to generate a rule and retry with a fresh suggestion on failure.
                attempt_desc = desc
                for attempt in range(1 + max_retries_per_rule):
                    label = f"  [{idx}/{total}] {rule_type.title()} rule"
                    if attempt > 0:
                        label += f" (retry {attempt})"
                    if verbose:
                        print(f"{label}: {attempt_desc}")
                    ok, msg = generate_rule(
                        backend="aider",
                        rule_type=rule_type,
                        description=attempt_desc,
                        game_name=game_name,
                        world_definition_path=world_definition_path,
                        llm_name=llm_name,
                        game_description=description,
                        llm_provider=llm_provider,
                    )
                    if ok:
                        if verbose:
                            print(f"  ✓ {msg}")
                        succeeded.append((rule_type, attempt_desc))
                        return True
                    if verbose:
                        print(f"  ✗ {msg}")
                    # Re-suggest a different rule for the next attempt
                    if attempt < max_retries_per_rule:
                        if verbose:
                            print("  ↻ Suggesting a replacement rule...")
                        attempt_desc = _suggest(
                            rule_type, llm_name, game_name,
                            already_suggested=all_suggested + [attempt_desc],
                            game_description=description,
                            llm=_get_llm(llm_provider, llm_name),
                        )
                        if verbose:
                            print(f"  💡 {attempt_desc}")
                failed.append((rule_type, desc))
                return False

            rule_idx = 0
            for rule_desc in action_rule_descs:
                rule_idx += 1
                _try_rule("action", rule_desc, rule_idx, total_rules)
            for rule_desc in step_rule_descs:
                rule_idx += 1
                _try_rule("step", rule_desc, rule_idx, total_rules)

            # ── summary ──
            _gen_report["stages"]["rules"] = {
                "requested_action": num_action_rules + len(new_action_rules or []),
                "requested_step": num_step_rules + len(new_step_rules or []),
                "succeeded": succeeded,
                "failed": failed,
            }
            if verbose:
                print(f"\n{'─'*60}")
                print(f"  Rule generation summary: {len(succeeded)}/{total_rules} succeeded")
                if failed:
                    print(f"  Failed ({len(failed)}):")
                    for rtype, fd in failed:
                        print(f"    ✗ [{rtype}] {fd}")
                print(f"{'─'*60}")
        else:
            _gen_report["stages"]["rules"] = {"requested_action": 0, "requested_step": 0, "succeeded": [], "failed": []}
            if verbose:
                print("[3/4] No rules requested — skipping.")

        if num_quest_chapters > 0:
            if verbose:
                print(f"[4/4] Generating {num_quest_chapters} quest chapter(s)...")
            from tools.generators.quest_generator import generate_quest
            q_desc = quest_description or description
            quest_ok, quest_msg = generate_quest(
                description=q_desc,
                game_name=game_name,
                world_definition_path=world_definition_path,
                num_chapters=num_quest_chapters,
                llm_name=llm_name,
                llm_provider=llm_provider,
                branching_factor=branching_factor,
            )
            _gen_report["stages"]["quest"] = {"status": "ok" if quest_ok else "failed", "message": quest_msg, "chapters_requested": num_quest_chapters}
            if verbose:
                status = "✓" if quest_ok else "✗"
                print(f"  {status} Quest generation: {quest_msg}")
        else:
            _gen_report["stages"]["quest"] = {"status": "skipped"}
            if verbose:
                print("[4/4] No quest chapters requested — skipping.")

        smoke_ok = True
        if not skip_test:
            if verbose:
                print("\nRunning smoke test…")
            smoke_ok, smoke_output = test_game_entities(game_name, world_definition_path, timeout=120)
            _gen_report["stages"]["smoke_test"] = {"status": "ok" if smoke_ok else "failed", "output": smoke_output[-500:] if not smoke_ok else ""}
            if smoke_ok:
                if verbose:
                    print("✓ Smoke test passed!")
            else:
                if verbose:
                    print(f"✗ Smoke test failed:\n{smoke_output[-500:]}")
        else:
            _gen_report["stages"]["smoke_test"] = {"status": "skipped"}

        game = GeneratedGame(
            game_name=game_name,
            game_dir=game_dir,
            assets_dir=assets_dir,
            world_definition_path=world_definition_path,
            env_config_path=env_config_path,
            description=description,
        )

        if verbose:
            elapsed = _time.time() - _gen_report["start_time"]
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)

            print(f"\n{'='*60}")
            print(f"  GENERATION REPORT — {game_name}")
            print(f"{'='*60}")
            print(f"  Description : {description[:72]}{'…' if len(description) > 72 else ''}")
            print(f"  Model       : {llm_name} ({llm_provider or 'default'})")
            print(f"  Duration    : {elapsed_m}m {elapsed_s}s")
            print()

            # Stage 1 — Entities
            ent = _gen_report["stages"].get("entities", {})
            ent_status = ent.get("status", "unknown")
            print(f"  [1/4] Entities: {'✓' if ent_status == 'ok' else '✗'} ({num_places} places, {num_objects} objects, {num_npcs} NPCs requested)")

            # Stage 2 — Rules
            rules = _gen_report["stages"].get("rules", {})
            req_action = rules.get("requested_action", 0)
            req_step = rules.get("requested_step", 0)
            ok_rules = rules.get("succeeded", [])
            fail_rules = rules.get("failed", [])
            ok_action = sum(1 for t, _ in ok_rules if t == "action")
            ok_step = sum(1 for t, _ in ok_rules if t == "step")
            fail_action = sum(1 for t, _ in fail_rules if t == "action")
            fail_step = sum(1 for t, _ in fail_rules if t == "step")

            if req_action + req_step > 0:
                print(f"  [2/4] Action rules: {ok_action}/{req_action} succeeded" + (f" ({fail_action} failed)" if fail_action else ""))
                print(f"         Step rules : {ok_step}/{req_step} succeeded" + (f" ({fail_step} failed)" if fail_step else ""))
                if ok_rules:
                    print(f"         Succeeded:")
                    for rtype, rd in ok_rules:
                        print(f"           ✓ [{rtype}] {rd[:70]}{'…' if len(rd) > 70 else ''}")
                if fail_rules:
                    print(f"         Failed:")
                    for rtype, rd in fail_rules:
                        print(f"           ✗ [{rtype}] {rd[:70]}{'…' if len(rd) > 70 else ''}")
            else:
                print(f"  [2/4] Rules: skipped (none requested)")

            # Stage 3 — Quest
            quest = _gen_report["stages"].get("quest", {})
            quest_status = quest.get("status", "skipped")
            if quest_status == "skipped":
                print(f"  [3/4] Quest: skipped (none requested)")
            else:
                ch_req = quest.get("chapters_requested", 0)
                q_msg = quest.get("message", "")
                print(f"  [3/4] Quest: {'✓' if quest_status == 'ok' else '✗'} ({ch_req} chapters requested) — {q_msg[:60]}")

            # Stage 4 — Smoke test
            smoke = _gen_report["stages"].get("smoke_test", {})
            smoke_status = smoke.get("status", "skipped")
            if smoke_status == "skipped":
                print(f"  [4/4] Smoke test: skipped")
            else:
                print(f"  [4/4] Smoke test: {'✓ passed' if smoke_status == 'ok' else '✗ FAILED'}")

            print()

            # Print full game stats
            try:
                from tools.game_stats import main as _game_stats_main
                import sys as _sys
                _saved_argv = _sys.argv
                _sys.argv = ["game_stats", game_name]
                _game_stats_main()
                _sys.argv = _saved_argv
            except Exception as e:
                print(f"  (Could not print game stats: {e})")

            print()
            print(f"  Game code : {game_dir}")
            print(f"  Assets    : {assets_dir}")
            print(f"  World def : {world_definition_path}")
            print()
            print(f"  Run it with:")
            print(f"    AgentOdyssey.run(agent='HumanAgent', game_name='{game_name}')")
            print(f"  or:")
            print(f"    python eval.py --agent HumanAgent --game_name {game_name}")
            print(f"  or (using the cli tool):")
            print(f"    agentodyssey run --agent HumanAgent --game_name {game_name}")
            print(f"{'='*60}\n")

        return game

    @staticmethod
    def run(
        agent: str = "HumanAgent",
        game_name: str = "v1",
        *,
        agents_config: str | None = None,
        agent_id: str = "agent_adam_davis",
        agent_name: str = "adam_davis",
        llm_name: str | None = None,
        llm_provider: str | None = None,
        max_steps: int = 300,
        seed: int = 42,
        output_dir: str = "output",
        run_dir: str | None = None,
        extra_dir: str | None = None,
        world_definition_path: str | None = None,
        env_config_path: str | None = None,
        overwrite: bool = False,
        cumulative_config_save: bool = True,
        enable_short_term_memory: bool = False,
        enable_reflection: bool = False,
        enable_summarization: bool = False,
        enable_obs_valid_actions: bool = False,
        enforce_same_hardware: bool = False,
        render: bool = False,
        debug: bool = False,
        resume_from_step: int | None = None,
        save_dep_graph_steps: int | None = None,
        memory_dir: str = "memory",
        agent_memory_save_frequency: int = 5,
    ) -> None:
        """Launch an evaluation run.

        This is a thin wrapper around ``eval.py`` that builds the CLI
        command and executes it as a subprocess.

        Parameters
        ----------
        agent : str
            Agent class name (e.g. ``"HumanAgent"``, ``"VanillaRAGAgent"``).
        game_name : str
            Game variant to run (``"base"``, ``"v1"``, or a custom name).
        agents_config : str, optional
            Path to a JSON file for multi-agent configurations.  When
            provided, single-agent flags are ignored.
        agent_id : str
            Unique identifier for the agent.
        agent_name : str
            Human-readable agent name.
        llm_name : str, optional
            LLM model identifier (e.g. ``"gpt-5"``, ``"Qwen/Qwen3-4B"``).
        llm_provider : str, optional
            LLM provider identifier (e.g. ``"openai"``, ``"claude"``,
            ``"gemini"``, ``"vllm"``, ``"huggingface"``).
        max_steps : int
            Maximum number of environment steps.
        seed : int
            Random seed for reproducibility.
        output_dir : str
            Path to the general output directory.
        run_dir : str, optional
            Path storing the current run's data.
        extra_dir : str, optional
            Extra directory level under *run_dir* for multiple runs.
        world_definition_path : str, optional
            Path to the world definition JSON file.
        env_config_path : str, optional
            Path to the initial env config JSON/JSONL file.
        overwrite : bool
            Overwrite the run directory if it exists.
        cumulative_config_save : bool
            Save cumulative env config each step.
        enable_short_term_memory : bool
            Enable short-term memory for the agent.
        enable_reflection : bool
            Enable reflection for the agent.
        enable_summarization : bool
            Enable summarization for the agent.
        enable_obs_valid_actions : bool
            Include valid actions in the observation (required for
            ``RandomAgent``).
        enforce_same_hardware : bool
            Enforce the same hardware for resumed runs.
        render : bool
            Render a 2D top-down view using pygame.
        debug : bool
            Deprecated alias for ``cumulative_config_save``.
        resume_from_step : int, optional
            Resume from the given step number.
        save_dep_graph_steps : int, optional
            Steps between dependency-graph saves (``None`` to disable).
        memory_dir : str
            Directory for agent memory checkpoints under *run_dir*.
        agent_memory_save_frequency : int
            Save agent memory every N environment steps.
        """
        cmd = [
            sys.executable, os.path.join(current_directory, "eval.py"),
            "--agent", agent,
            "--game_name", game_name,
            "--agent_id", agent_id,
            "--agent_name", agent_name,
            "--max_steps", str(max_steps),
            "--seed", str(seed),
            "--output_dir", output_dir,
            "--memory_dir", memory_dir,
            "--agent_memory_save_frequency", str(agent_memory_save_frequency),
        ]
        if agents_config is not None:
            cmd.extend(["--agents_config", agents_config])
        if llm_name is not None:
            cmd.extend(["--llm_name", llm_name])
        if llm_provider is not None:
            cmd.extend(["--llm_provider", llm_provider])
        if run_dir is not None:
            cmd.extend(["--run_dir", run_dir])
        if extra_dir is not None:
            cmd.extend(["--extra_dir", extra_dir])
        if world_definition_path is not None:
            cmd.extend(["--world_definition_path", world_definition_path])
        if env_config_path is not None:
            cmd.extend(["--env_config_path", env_config_path])
        if overwrite:
            cmd.append("--overwrite")
        if cumulative_config_save:
            cmd.append("--cumulative_config_save")
        if enable_short_term_memory:
            cmd.append("--enable_short_term_memory")
        if enable_reflection:
            cmd.append("--enable_reflection")
        if enable_summarization:
            cmd.append("--enable_summarization")
        if enable_obs_valid_actions:
            cmd.append("--enable_obs_valid_actions")
        if enforce_same_hardware:
            cmd.append("--enforce_same_hardware")
        if render:
            cmd.append("--render")
        if debug:
            cmd.append("--debug")
        if resume_from_step is not None:
            cmd.extend(["--resume_from_step", str(resume_from_step)])
        if save_dep_graph_steps is not None:
            cmd.extend(["--save_dep_graph_steps", str(save_dep_graph_steps)])

        print(f"Running: {' '.join(cmd)}\n")
        subprocess.run(cmd, cwd=current_directory)

def _build_cli():
    import argparse

    parser = argparse.ArgumentParser(
        prog="agentodyssey",
        description="AgentOdyssey — generate and run text-world games for LLM agent evaluation.",
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate a new game world.")
    gen.add_argument("description", nargs="?", default=None,
                     help="Natural-language description of the game theme.")
    gen.add_argument("--game-name", default=None,
                     help="Folder name under games/generated/.")
    gen.add_argument("--num-places", type=int, default=2)
    gen.add_argument("--num-objects", type=int, default=10)
    gen.add_argument("--num-npcs", type=int, default=5)
    gen.add_argument("--max-level", type=int, default=5)
    gen.add_argument("--num-quest-chapters", type=int, default=1)
    gen.add_argument("--quest-description", default=None)
    gen.add_argument("--num-action-rules", type=int, default=0,
                     help="Number of new action rules to auto-generate.")
    gen.add_argument("--num-step-rules", type=int, default=0,
                     help="Number of new step rules to auto-generate.")
    gen.add_argument("--llm-name", dest="llm_name", default="gpt-5",
                     help="LLM model to use for generation.")
    gen.add_argument("--llm-provider", dest="llm_provider", default="openai",
                     choices=["openai", "azure", "azure_openai", "claude",
                              "gemini", "vllm", "huggingface"],
                     help="LLM provider to use for generation.")
    gen.add_argument("--backend", default="llm", choices=["llm", "aider"])
    gen.add_argument("--generate-graph", action="store_true",
                     help="Also generate a predefined area-connection graph.")
    gen.add_argument("--branching-factor", type=int, default=1,
                     help="Goal tree branching factor (1=linear, 2+=tree). Each node can have up to this many children.")
    gen.add_argument("--overwrite", action="store_true")
    gen.add_argument("--skip-test", action="store_true")
    gen.add_argument("--quiet", action="store_true")

    run = sub.add_parser("run", help="Run an evaluation.")
    run.add_argument("--agent", default="HumanAgent")
    run.add_argument("--game-name", default="base")
    run.add_argument("--agents-config", default=None,
                     help="Path to a JSON file for multi-agent configurations.")
    run.add_argument("--agent-id", default="agent_adam_davis")
    run.add_argument("--agent-name", default="adam_davis")
    run.add_argument("--llm-name", default=None)
    run.add_argument("--llm-provider", default=None,
                     choices=["openai", "huggingface", "vllm", "azure", "azure_openai", "claude", "gemini"])
    run.add_argument("--max-steps", type=int, default=300)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output-dir", default="output")
    run.add_argument("--run-dir", default=None)
    run.add_argument("--extra-dir", default=None)
    run.add_argument("--world-definition-path", default=None)
    run.add_argument("--env-config-path", default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--cumulative-config-save", action="store_true")
    run.add_argument("--render", action="store_true")
    run.add_argument("--enable-short-term-memory", action="store_true")
    run.add_argument("--enable-reflection", action="store_true")
    run.add_argument("--enable-summarization", action="store_true")
    run.add_argument("--enable-obs-valid-actions", action="store_true",
                     help="Include valid actions in the observation (required for RandomAgent).")
    run.add_argument("--enforce-same-hardware", action="store_true")
    run.add_argument("--debug", action="store_true",
                     help="(Deprecated) Alias for --cumulative-config-save.")
    run.add_argument("--resume-from-step", type=int, default=None)
    run.add_argument("--save-dep-graph-steps", type=int, default=None)
    run.add_argument("--memory-dir", default="memory")
    run.add_argument("--agent-memory-save-frequency", type=int, default=5)

    return parser

def main():
    parser = _build_cli()
    args = parser.parse_args()

    if args.command == "generate":
        AgentOdyssey.generate(
            description=args.description,
            game_name=args.game_name,
            num_places=args.num_places,
            num_objects=args.num_objects,
            num_npcs=args.num_npcs,
            max_level=args.max_level,
            num_quest_chapters=args.num_quest_chapters,
            quest_description=args.quest_description,
            branching_factor=args.branching_factor,
            num_action_rules=args.num_action_rules,
            num_step_rules=args.num_step_rules,
            llm_name=args.llm_name,
            llm_provider=args.llm_provider,
            backend=args.backend,
            generate_graph=args.generate_graph,
            overwrite=args.overwrite,
            skip_test=args.skip_test,
            verbose=not args.quiet,
        )

    elif args.command == "run":
        AgentOdyssey.run(
            agent=args.agent,
            game_name=args.game_name,
            agents_config=args.agents_config,
            agent_id=args.agent_id,
            agent_name=args.agent_name,
            llm_name=args.llm_name,
            llm_provider=args.llm_provider,
            max_steps=args.max_steps,
            seed=args.seed,
            output_dir=args.output_dir,
            run_dir=args.run_dir,
            extra_dir=args.extra_dir,
            world_definition_path=args.world_definition_path,
            env_config_path=args.env_config_path,
            overwrite=args.overwrite,
            cumulative_config_save=args.cumulative_config_save,
            render=args.render,
            enable_short_term_memory=args.enable_short_term_memory,
            enable_reflection=args.enable_reflection,
            enable_summarization=args.enable_summarization,
            enable_obs_valid_actions=args.enable_obs_valid_actions,
            enforce_same_hardware=args.enforce_same_hardware,
            debug=args.debug,
            resume_from_step=args.resume_from_step,
            save_dep_graph_steps=args.save_dep_graph_steps,
            memory_dir=args.memory_dir,
            agent_memory_save_frequency=args.agent_memory_save_frequency,
        )

    else:
        parser.print_help()

if __name__ == "__main__":
    main()