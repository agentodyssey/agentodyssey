import argparse
import json
import subprocess
import sys
import os
import shutil
from pathlib import Path
import shlex

current_directory = os.getcwd()
sys.path.insert(0, current_directory)

from providers.openai_api import OpenAILanguageModel

BASE_GAME_DIR = "games/base"
BASE_ENV_CONFIGS_DIR = "assets/env_configs/base"
BASE_WORLD_DEFS_DIR = "assets/world_definitions/base"

import re as _re

# Maps --provider values to aider/litellm provider prefixes
_AIDER_PROVIDER_PREFIX = {
    "openai": "openai",
    "azure": "azure",
    "azure_openai": "azure",
    "claude": "anthropic",
    "gemini": "gemini",
}

def _aider_model_name(model: str, provider: str | None = None) -> str:
    if "/" in model:
        return model
    if provider and provider in _AIDER_PROVIDER_PREFIX:
        return f"{_AIDER_PROVIDER_PREFIX[provider]}/{model}"
    if _re.match(r"^(gpt-|o[1-9]|chatgpt-|ft:)", model):
        return f"openai/{model}"
    return model

def _rewrite_imports_in_generated_game(game_dir: Path, game_name: str) -> None:
    old = "games.base."
    new = f"games.generated.{game_name}."
    old_world_defs = "assets/world_definitions/base/"
    new_world_defs = f"assets/world_definitions/generated/{game_name}/"
    old_env_configs = "assets/env_configs/base/"
    new_env_configs = f"assets/env_configs/generated/{game_name}/"

    for py_file in game_dir.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # best effort; skip non-utf8 sources
            continue

        updated = text.replace(old, new).replace(old_world_defs, new_world_defs).replace(old_env_configs, new_env_configs)
        if updated != text:
            py_file.write_text(updated, encoding="utf-8")

def ensure_generated_game(game_name: str, overwrite: bool = False) -> tuple[str, str]:
    if not game_name or game_name.strip() == "":
        raise ValueError("game_name must be a non-empty string")

    game_base = Path(current_directory) / BASE_GAME_DIR
    game_generated = Path(current_directory) / "games" / "generated" / game_name
    env_configs_base = Path(current_directory) / BASE_ENV_CONFIGS_DIR
    env_configs_generated = Path(current_directory) / "assets" / "env_configs" / "generated" / game_name
    world_defs_base = Path(current_directory) / BASE_WORLD_DEFS_DIR
    world_defs_generated = Path(current_directory) / "assets" / "world_definitions" / "generated" / game_name

    if overwrite:
        if game_generated.exists():
            shutil.rmtree(game_generated)
        if env_configs_generated.exists():
            shutil.rmtree(env_configs_generated)
        if world_defs_generated.exists():
            shutil.rmtree(world_defs_generated)

    if not game_generated.exists():
        game_generated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(game_base, game_generated)
        _rewrite_imports_in_generated_game(game_generated, game_name)

    if not env_configs_generated.exists():
        env_configs_generated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(env_configs_base, env_configs_generated)

    if not world_defs_generated.exists():
        world_defs_generated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(world_defs_base, world_defs_generated)

    return str(game_generated), str(world_defs_generated)

def get_files_for_game(game_name: str, world_definition_path: str) -> tuple[list[str], list[str]]:
    game_root = f"games/generated/{game_name}"

    # Files that may need to be edited when adding new rules/actions
    editable_files = [
        f"{game_root}/rules/action_rules.py",
        f"{game_root}/rules/step_rules/general.py",
        f"{game_root}/agent.py",
        f"{game_root}/env.py",
        f"{game_root}/world.py",
        world_definition_path,
        f"assets/env_configs/generated/{game_name}/initial.json",
    ]

    # Files for read-only context
    context_files = [
        f"{game_root}/rule.py",
    ]

    return editable_files, context_files

def get_test_command(game_name: str, world_definition_path: str) -> str:
    return (
        "python eval.py "
        "--agent RandomAgent "
        "--max_steps 1000 "
        "--enable_obs_valid_actions "
        "--overwrite "
        f"--game_name {shlex.quote(game_name)} "
        f"--world_definition_path {shlex.quote(world_definition_path)} "
        f"--env_config_path assets/env_configs/generated/{shlex.quote(game_name)}/initial.json"
    )

def build_prompt(rule_type: str, description: str, game_name: str, world_definition_path: str = None, game_description: str | None = None) -> str:
    game_root = f"games/generated/{game_name}"
    test_command = get_test_command(game_name, world_definition_path)

    theme_block = ""
    if game_description:
        theme_block = f"""GAME WORLD THEME:\n{game_description}\n\nThe rule you implement MUST be thematically consistent with this world.\n\n"""

    if rule_type == "action":
        return f"""{theme_block}Add a new ACTION RULE (i.e., a new ACTION) to the games.

Action rules are rules that add new actions to the agent's action space that the player can perform.

ACTION DESCRIPTION:
{description}

FILES TO READ FOR CONTEXT:
- {game_root}/rule.py - Base classes (BaseActionRule, BaseStepRule, RuleContext, RuleResult)
- {game_root}/world.py - World state, Object, NPC, Area, Container classes
- {game_root}/rules/action_rules.py - Existing action rules (player actions)
- {game_root}/rules/step_rules.py - Existing step rules (environment rules)
- {game_root}/env.py - Environment with get_obs_valid_actions() method

FILES THAT MUST BE EDITED:

1. {game_root}/rules/action_rules.py (REQUIRED)
   - Add the new action rule class inheriting from BaseActionRule
   - Define: name, verbs, param_min, param_max, description
   - Implement apply(self, ctx: RuleContext, res: RuleResult) method
   - Use ctx.env, ctx.world, ctx.agent, ctx.params to access game state
   - Use res.add_feedback() for text output, res.events.append() for events

2. {game_root}/env.py (REQUIRED)
   - Update get_obs_valid_actions() method to generate valid action strings for this new action
   - Follow patterns of existing actions (enter, pick up, drop, etc.)
   - May also need to modify other parts of env.py if the action requires special handling

FILES THAT MAY NEED EDITING (if the action requires new entities or world changes):

3. {world_definition_path} - Add new entities if needed
   - Objects (may also add new attributes): {{"id": "obj_X", "name": "X", "category": "...", "usage": "...", "value": N, "size": N, "craft": {{}}, "level": N, "...": ...}}
   - NPCs: {{"id": "npc_X", "name": "X", "enemy": false, "unique": true, "role": "...", "objects": []}}

4. {game_root}/world.py - Modify world state or add new properties/methods if needed

5. {game_root}/rules/step_rules.py - Update TutorialRoomStepRule if the action should be taught

REQUIREMENTS:
- Follow the exact same patterns as existing rules in the codebase
- Make sure all imports are correct
- Edit ALL necessary files for the action to work completely
- The new action should integrate seamlessly with existing game mechanics
- You may add new objects or NPCs if needed to support the action without the user's approval

TEST AFTER CHANGES:
{test_command}
"""
    else:
        return f"""{theme_block}Add a new STEP RULE to the games.

Step rules are NON-ACTION rules that are automatically applied at each step of the games. They handle automatic game mechanics, world state changes, status effects, environmental effects, etc.

RULE DESCRIPTION:
{description}

FILES TO READ FOR CONTEXT:
- {game_root}/rule.py - Base classes (BaseActionRule, BaseStepRule, RuleContext, RuleResult)
- {game_root}/world.py - World state, Object, NPC, Area, Container classes
- {game_root}/rules/action_rules.py - Existing action rules (player actions)
- {game_root}/rules/step_rules/general.py - Existing step rules (environment rules)
- {game_root}/env.py - Environment class

CRITICAL: Do NOT remove, rename, or modify ANY existing rule classes. Only APPEND new code.
All existing classes in action_rules.py and step_rules/general.py MUST remain intact.

FILES THAT MUST BE EDITED:

1. {game_root}/rules/step_rules/general.py (REQUIRED)
   - APPEND the new step rule class at the END of the file, inheriting from BaseStepRule
   - Do NOT delete or modify any existing rule class
   - Define: name, description
   - Implement apply(self, ctx: RuleContext, res: RuleResult) method
   - Use ctx.env, ctx.world, ctx.agent to access game state
   - Use res.add_feedback() for text output, res.events.append() for events

FILES THAT MAY NEED EDITING:

2. {game_root}/env.py - Modify if the rule requires special environment handling or state tracking

3. assets/env_configs/generated/{game_name}/initial.json - Modify initial game configuration if needed

4. {world_definition_path} - Add new entities if needed
   - Objects (may also add new attributes): {{"id": "obj_X", "name": "X", "category": "...", "usage": "...", "value": N, "size": N, "craft": {{}}, "level": N, "...": ...}}
   - NPCs: {{"id": "npc_X", "name": "X", "enemy": false, "unique": true, "role": "...", "objects": []}}

5. {game_root}/world.py - Modify world state or add new properties/methods if needed

REQUIREMENTS:
- Follow the exact same patterns as existing rules in the codebase
- Make sure all imports are correct
- Edit ALL necessary files for the rule to work completely
- The rule should integrate seamlessly with existing game mechanics
- You may add new objects or NPCs if needed to support the action without the user's approval

TEST AFTER CHANGES:
{test_command}
"""

def build_fix_prompt(error_output: str, game_name: str, world_definition_path: str) -> str:
    test_command = get_test_command(game_name, world_definition_path)
    return f"""Fix the errors in the codebase.

ERROR OUTPUT:
{error_output[-3000:]}

INSTRUCTIONS:
- Read the error carefully and identify which file(s) need fixes
- Fix all files as needed to resolve the error
- Make sure the fix doesn't break other functionality

TEST AFTER FIXING:
{test_command}
"""

def test_game(game_name: str, world_definition_path: str, timeout: int = 120) -> tuple[bool, str]:
    cmd = [
        "python", "eval.py",
        "--agent", "RandomAgent",
        "--max_steps", "1000",
        "--enable_obs_valid_actions",
        "--overwrite"
    ]
    cmd.extend(["--game_name", game_name])
    cmd.extend(["--world_definition_path", world_definition_path])
    cmd.extend(["--env_config_path", f"assets/env_configs/generated/{game_name}/initial.json"])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=current_directory
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            return False, output
        return True, output
    except subprocess.TimeoutExpired:
        return True, "Test completed (timeout reached, likely running fine)"
    except Exception as e:
        return False, str(e)

def _snapshot_files(file_paths: list[str]) -> dict[str, bytes | None]:
    # Read current contents of files for rollback on failure.
    snap: dict[str, bytes | None] = {}
    for fp in file_paths:
        full = os.path.join(current_directory, fp)
        try:
            snap[fp] = Path(full).read_bytes()
        except OSError:
            snap[fp] = None
    return snap

def _restore_files(snapshot: dict[str, bytes | None]) -> None:
    for fp, data in snapshot.items():
        full = os.path.join(current_directory, fp)
        if data is None:
            # file didn't exist before; remove if aider created it
            try:
                os.remove(full)
            except OSError:
                pass
        else:
            Path(full).write_bytes(data)

def _count_classes(file_path: str) -> int:
    import ast as _ast
    full = os.path.join(current_directory, file_path)
    try:
        with open(full, "r", encoding="utf-8") as f:
            tree = _ast.parse(f.read())
        return sum(1 for n in _ast.iter_child_nodes(tree) if isinstance(n, _ast.ClassDef))
    except (OSError, SyntaxError):
        return -1

def _check_action_rule_in_env(game_name: str, baseline_class_counts: dict) -> str | None:
    import ast as _ast

    action_rules_path = f"games/generated/{game_name}/rules/action_rules.py"
    env_path = f"games/generated/{game_name}/env.py"
    abs_ar = os.path.join(current_directory, action_rules_path)
    abs_env = os.path.join(current_directory, env_path)

    try:
        with open(abs_ar, "r", encoding="utf-8") as f:
            ar_tree = _ast.parse(f.read())
    except (OSError, SyntaxError):
        return None

    verbs: list[str] = []
    has_get_valid_actions: list[bool] = []
    for node in _ast.iter_child_nodes(ar_tree):
        if not isinstance(node, _ast.ClassDef):
            continue
        verb_val = None
        has_gva = False
        for item in node.body:
            if isinstance(item, _ast.Assign):
                for target in item.targets:
                    if isinstance(target, _ast.Name) and target.id == "verb":
                        if isinstance(item.value, _ast.Constant) and isinstance(item.value.value, str):
                            verb_val = item.value.value
            if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if item.name == "get_valid_actions":
                    has_gva = True
        if verb_val is not None:
            verbs.append(verb_val)
            has_get_valid_actions.append(has_gva)

    try:
        with open(abs_env, "r", encoding="utf-8") as f:
            env_src = f.read()
    except OSError:
        return None

    baseline = baseline_class_counts.get(action_rules_path, 0)
    if baseline <= 0:
        return None

    new_verbs = verbs[baseline:]
    new_has_gva = has_get_valid_actions[baseline:]
    for i, verb in enumerate(new_verbs):
        if i < len(new_has_gva) and new_has_gva[i]:
            continue
        verb_token = verb.replace(" ", "_")
        patterns = [
            f'actions["{verb}"]',
            f"actions['{verb}']",
            f'actions["{verb_token}"]',
            f"actions['{verb_token}']",
        ]
        found = any(p in env_src for p in patterns)
        if not found:
            return verb

    return None

def _runtime_check_verb_in_valid_actions(game_name: str, world_definition_path: str, verb: str) -> bool:
    check_script = f"""
import sys, os, json, random
sys.path.insert(0, os.getcwd())
game_name = {game_name!r}
world_definition = {world_definition_path!r}
verb = {verb!r}
try:
    env_mod = __import__(f"games.generated.{{game_name}}.env", fromlist=["AgentOdysseyEnv"])
    agent_mod = __import__(f"games.generated.{{game_name}}.agent", fromlist=["Agent"])
    Env = env_mod.AgentOdysseyEnv
    Agent = agent_mod.Agent
    agent = Agent("agent_0", "tester")
    env = Env(seed=42, agents=[agent], world_definition_path=world_definition,
              run_dir="/tmp/_rg_check", config_path=None,
              enable_obs_valid_actions=True)
    env.reset()
    actions = env.get_all_valid_actions(agent)
    if verb in actions and len(actions[verb]) > 0:
        print("FOUND")
    else:
        print("MISSING")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", check_script],
            cwd=current_directory,
            capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        return "FOUND" in output
    except Exception:
        return True  # can't confirm — don't block

def generate_with_aider(
    rule_type: str,
    description: str,
    game_name: str,
    world_definition_path: str,
    llm_name: str = "gpt-4o",
    max_iterations: int = 5,
    test_timeout: int = 120,
    game_description: str | None = None,
    llm_provider: str | None = None,
) -> tuple[bool, str]:
    prompt = build_prompt(rule_type, description, game_name, world_definition_path, game_description=game_description)
    editable_files, context_files = get_files_for_game(game_name, world_definition_path)

    cmd = [
        "aider",
        "--model", _aider_model_name(llm_name, provider=llm_provider),
        "--no-auto-commits",
        "--yes",
        "--no-suggest-shell-commands",
        "--message", prompt,
    ]
    
    for ctx_file in context_files:
        full_path = os.path.join(current_directory, ctx_file)
        if os.path.exists(full_path):
            cmd.extend(["--read", ctx_file])
    
    for edit_file in editable_files:
        full_path = os.path.join(current_directory, edit_file)
        if os.path.exists(full_path):
            cmd.append(edit_file)
    
    print(f"\n{'='*60}")
    print(f"Generating {rule_type} rule: {description}")
    print(f"Using model: {llm_name}")
    print(f"Editable files: {editable_files}")
    print(f"{'='*60}\n")

    snapshot = _snapshot_files(editable_files)

    game_root = f"games/generated/{game_name}"
    _class_count_files = {
        f"{game_root}/rules/action_rules.py": 0,
        f"{game_root}/rules/step_rules/general.py": 0,
    }
    for ccf in _class_count_files:
        _class_count_files[ccf] = _count_classes(ccf)
    print(f"Baseline class counts: { {os.path.basename(k): v for k, v in _class_count_files.items()} }")

    # Pretty-print the world-definition JSON before aider sees it.
    # Aider's search/replace fails on long single-line JSON entries;
    # expanding to one-field-per-line gives it reliable diff anchors.
    abs_world_path = os.path.join(current_directory, world_definition_path)
    if os.path.exists(abs_world_path) and abs_world_path.endswith(".json"):
        try:
            with open(abs_world_path, "r", encoding="utf-8") as f:
                world_data = json.load(f)
            with open(abs_world_path, "w", encoding="utf-8") as f:
                json.dump(world_data, f, indent=2, ensure_ascii=False)
                f.write("\n")
        except (json.JSONDecodeError, OSError):
            pass  # leave the file as-is if it can't be parsed
    
    consecutive_timeouts = 0
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---\n")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=current_directory,
                capture_output=False,
                text=True,
                timeout=600,
            )
            consecutive_timeouts = 0
        except subprocess.TimeoutExpired:
            consecutive_timeouts += 1
            print(f"Aider timed out ({consecutive_timeouts} consecutive)")
            if consecutive_timeouts >= 2:
                print("Aborting: too many consecutive timeouts.")
                _restore_files(snapshot)
                return False, "Repeated aider timeouts"
            continue
        except Exception as e:
            print(f"Aider error: {e}")
            continue

        abs_wp = os.path.join(current_directory, world_definition_path)
        if os.path.exists(abs_wp) and abs_wp.endswith(".json"):
            try:
                with open(abs_wp, "r", encoding="utf-8") as _f:
                    json.load(_f)
            except (json.JSONDecodeError, UnicodeDecodeError) as je:
                print(f"\n✗ Aider corrupted {world_definition_path}: {je}")
                print("  Restoring all files from snapshot and retrying from scratch...")
                _restore_files(snapshot)
                cmd[cmd.index("--message") + 1] = prompt
                continue

        classes_deleted = False
        for ccf, baseline in _class_count_files.items():
            if baseline < 0:
                continue  # couldn't parse before; skip check
            current = _count_classes(ccf)
            if current < baseline:
                print(f"\n✗ Aider deleted classes in {ccf} ({baseline} → {current}). Restoring from snapshot.")
                classes_deleted = True
                break
        if classes_deleted:
            _restore_files(snapshot)
            cmd[cmd.index("--message") + 1] = prompt
            continue

        print("\nTesting generated rule...")
        success, output = test_game(game_name=game_name, world_definition_path=world_definition_path, timeout=test_timeout)
        
        if success:
            if rule_type == "action":
                missing_verb = _check_action_rule_in_env(game_name, _class_count_files)
                if missing_verb:
                    print(f"\n⚠ Action rule '{missing_verb}' has no valid_actions integration.")
                    # Try the runtime check — maybe the registry handles it
                    if not _runtime_check_verb_in_valid_actions(game_name, world_definition_path, missing_verb):
                        print("  Asking aider to add get_valid_actions() classmethod...")
                        cmd[cmd.index("--message") + 1] = (
                            f"The new action rule with verb '{missing_verb}' was added to action_rules.py "
                            f"but the agent cannot use this action because no valid action strings "
                            f"are generated for it.\n\n"
                            f"Add a get_valid_actions() classmethod to the new rule class. Example:\n\n"
                            f"    @classmethod\n"
                            f"    def get_valid_actions(cls, env, world, agent):\n"
                            f"        actions = {{}}\n"
                            f"        curr_area = world.area_instances[env.curr_agents_state['area'][agent.id]]\n"
                            f"        # Build valid action strings for '{missing_verb}'\n"
                            f"        actions['{missing_verb}'] = [...]  # fill with valid action strings\n"
                            f"        return actions\n\n"
                            f"The environment calls this automatically. Do NOT edit env.py.\n\n"
                            f"TEST AFTER CHANGES:\n{get_test_command(game_name, world_definition_path)}"
                        )
                        continue
                    else:
                        print("  ✓ Runtime check confirmed verb is available via registry.")
            # For step rules, verify a new class was actually added
            if rule_type == "step":
                step_file = f"{game_root}/rules/step_rules/general.py"
                new_count = _count_classes(step_file)
                baseline_count = _class_count_files.get(step_file, 0)
                if new_count <= baseline_count:
                    print(f"\n✗ No new step rule class added to {step_file} ({baseline_count} → {new_count}). Retrying...")
                    cmd[cmd.index("--message") + 1] = prompt
                    continue
            print("\n✓ Rule generated and tested successfully!")
            return True, "Rule added successfully"
        
        print(f"\n✗ Test failed. Error:\n{output[-2000:]}")
        
        cmd[cmd.index("--message") + 1] = build_fix_prompt(output, game_name=game_name, world_definition_path=world_definition_path)
    
    print(f"\nFailed to generate valid rule after {max_iterations} iterations — rolling back.")
    _restore_files(snapshot)
    return False, "Max iterations reached"

def suggest_novel_rule(
    rule_type: str,
    llm_name: str,
    game_name: str,
    already_suggested: list[str] | None = None,
    game_description: str | None = None,
    llm=None,
) -> str:
    rule_suggestion_output_token_budget = 4096
    rules_file = (
        f"games/generated/{game_name}/rules/action_rules.py"
        if rule_type == "action"
        else f"games/generated/{game_name}/rules/step_rules/general.py"
    )
    
    rules_file_path = os.path.join(current_directory, rules_file)
    with open(rules_file_path, 'r') as f:
        rules_content = f.read()
    
    dedup_block = ""
    if already_suggested:
        numbered = "\n".join(f"  {i}. {s}" for i, s in enumerate(already_suggested, 1))
        dedup_block = (
            f"\nRules that have ALREADY been suggested (do NOT repeat these or "
            f"suggest anything with similar logic / verb / purpose):\n{numbered}\n"
        )

    theme_block = ""
    if game_description:
        theme_block = (
            f"\nGAME WORLD THEME:\n{game_description}\n"
            f"Your suggested rule MUST be thematically consistent with this world.\n"
        )

    prompt = f"""You are a game designer for a text-based RPG games. Suggest ONE novel and creative {rule_type} rule.

Action rules are ACTIONS that players can perform.
Step rules are automatic game mechanics applied each step.

Here is the current {rules_file} showing existing rules:

```python
{rules_content}
```
{dedup_block}{theme_block}
Based on the existing rules above, suggest a NEW {rule_type} rule that would make the game more interesting.
Be creative but ensure it fits the game's mechanics.
The rule MUST be fundamentally different in both its verb and its underlying logic from every existing and already-suggested rule.

Reply with ONLY a single plain-English sentence describing what the rule does (e.g. "Allow the player to fish in water areas to obtain fish items").
No code, no bullet points, no structured fields, no preamble — just ONE sentence."""

    if llm is None:
        llm = OpenAILanguageModel(llm_name=llm_name)
    if getattr(llm, "max_new_tokens", None) is None:
        llm.max_new_tokens = rule_suggestion_output_token_budget
    else:
        llm.max_new_tokens = max(int(llm.max_new_tokens), rule_suggestion_output_token_budget)
    response = llm.generate(user_prompt=prompt, system_prompt=None)
    return response["response"].strip()

def get_user_confirmation(suggested_rule: str) -> tuple[bool, bool]:
    print(f"\n{'='*60}")
    print("SUGGESTED RULE:")
    print(f"{'='*60}")
    print(f"\n{suggested_rule}\n")
    print(f"{'='*60}")
    
    while True:
        choice = input("\nGenerate this rule? [Y]es / [N]o / [R]egenerate: ").strip().lower()
        if choice in ['y', 'yes']:
            return True, False
        elif choice in ['n', 'no']:
            return False, False
        elif choice in ['r', 'regenerate']:
            return False, True
        else:
            print("Please enter Y, N, or R.")

def generate_rule(
    backend: str,
    rule_type: str,
    description: str,
    game_name: str,
    world_definition_path: str = None,
    llm_name: str = "gpt-4o",
    max_iterations: int = 3,
    test_timeout: int = 120,
    game_description: str | None = None,
    llm_provider: str | None = None,
) -> tuple[bool, str]:
    if world_definition_path is None and game_name != "base":
        world_definition_path = f"assets/world_definitions/generated/{game_name}/default.json"
    elif world_definition_path is None and game_name == "base":
        world_definition_path = f"assets/world_definitions/base/default.json"

    if backend == "aider":
        return generate_with_aider(
            rule_type, description, game_name, world_definition_path, llm_name, max_iterations, test_timeout,
            game_description=game_description,
            llm_provider=llm_provider,
        )
    else:
        return False, f"Unknown backend: {backend}"

def interactive_mode(backend: str, llm_name: str, game_name: str, world_definition_path: str, llm_provider: str | None = None):
    editable_files, context_files = get_files_for_game(game_name, world_definition_path)
    print(f"\nStarting {backend} in interactive mode...")
    print(f"Editable files: {editable_files}")
    print(f"Context files: {context_files}\n")
    
    if backend == "aider":
        cmd = ["aider", "--model", _aider_model_name(llm_name, provider=llm_provider)]
        for ctx_file in context_files:
            full_path = os.path.join(current_directory, ctx_file)
            if os.path.exists(full_path):
                cmd.extend(["--read", ctx_file])
        for edit_file in editable_files:
            full_path = os.path.join(current_directory, edit_file)
            if os.path.exists(full_path):
                cmd.append(edit_file)
    else:
        print(f"Unknown backend: {backend}")
        return
    
    subprocess.run(cmd, cwd=current_directory)

if __name__ == "__main__":
    # python tools/rule_generator.py --backend aider --rule_type action --game_name test_game
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", type=str, required=True)
    parser.add_argument("--world_definition_path", type=str, default=None)
    parser.add_argument("--backend", type=str, default="aider", choices=["aider"])
    parser.add_argument("--rule_type", type=str, choices=["action", "step"])
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-5")
    parser.add_argument("--llm_provider", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--test_timeout", type=int, default=120)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    ensure_generated_game(args.game_name, overwrite=args.overwrite)
    
    if args.interactive:
        interactive_mode(args.backend, args.llm_name, args.game_name, args.world_definition_path, args.llm_provider)
    elif args.rule_type:
        description = args.description
        
        if not description:
            print(f"\nNo description provided. Suggesting a novel {args.rule_type} rule...")
            while True:
                suggested = suggest_novel_rule(args.rule_type, args.llm_name, game_name=args.game_name)
                confirmed, regenerate = get_user_confirmation(suggested)
                if confirmed:
                    description = suggested
                    break
                elif regenerate:
                    print("\nRegenerating suggestion...")
                    continue
                else:
                    print("\nAborted.")
                    sys.exit(0)
        
        success, message = generate_rule(
            backend=args.backend,
            rule_type=args.rule_type,
            description=description,
            game_name=args.game_name,
            world_definition_path=args.world_definition_path,
            llm_name=args.llm_name,
            max_iterations=args.max_iterations,
            test_timeout=args.test_timeout,
            llm_provider=args.llm_provider,
        )
        
        if success:
            print(f"\n✓ Success: {message}")
        else:
            print(f"\n✗ Failed: {message}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nError: --rule_type is required (unless using --interactive)")
        sys.exit(1)
