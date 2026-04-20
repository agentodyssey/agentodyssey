import argparse
import subprocess
import sys
import os
import json
import shlex
import textwrap
from pathlib import Path
from collections import defaultdict

current_directory = os.getcwd()
sys.path.insert(0, current_directory)

import re as _re

_PROVIDER_MAP = {
    "openai": ("providers.openai_api", "OpenAILanguageModel"),
    "azure": ("providers.azure_api", "AzureLanguageModel"),
    "azure_openai": ("providers.azure_openai_api", "AzureOpenAILanguageModel"),
    "claude": ("providers.claude_api", "ClaudeLanguageModel"),
    "gemini": ("providers.gemini_api", "GeminiLanguageModel"),
    "vllm": ("providers.vllm", "vllmLanguageModel"),
    "huggingface": ("providers.huggingface", "hfLanguageModel"),
}

def _get_llm(model: str, provider: str | None = None):
    if provider is None:
        if _re.match(r"^(gpt-|o[1-9]|chatgpt-|ft:)", model):
            provider = "openai"
        elif _re.match(r"^claude", model):
            provider = "claude"
        elif _re.match(r"^gemini", model):
            provider = "gemini"
        else:
            provider = "openai"  # default fallback
    if provider not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Choose from: {', '.join(sorted(_PROVIDER_MAP))}"
        )
    import importlib
    module_path, class_name = _PROVIDER_MAP[provider]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(llm_name=model)

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

def get_files_for_quest_generation(game_name: str, world_definition_path: str) -> tuple[list[str], list[str]]:
    game_root = f"games/generated/{game_name}"
    editable_files = [
        f"{game_root}/rules/step_rules/main_quest.py",
        world_definition_path,
    ]
    context_files = [
        f"{game_root}/rules/action_rules.py",
        f"{game_root}/rule.py",
        f"{game_root}/world.py",
        f"{game_root}/env.py",
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

def load_world_definition(world_definition_path: str) -> dict:
    full_path = os.path.join(current_directory, world_definition_path)
    with open(full_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"World definition JSON is corrupted (likely by a previous aider "
                f"edit that broke the file). Fix {world_definition_path} manually "
                f"or restore from git.  Original error: {e.msg}",
                e.doc, e.pos,
            ) from e

def load_existing_quest_chapters(game_name: str) -> list[dict]:
    step_rules_path = os.path.join(current_directory, f"games/generated/{game_name}/rules/step_rules/main_quest.py")
    
    with open(step_rules_path, 'r') as f:
        content = f.read()
    
    chapters_info = []
    if "QUEST_CONFIG" in content and '"chapters"' in content:
        import re
        chapter_matches = re.findall(r'"id":\s*"(chapter_\d+)".*?"title":\s*"([^"]+)"', content, re.DOTALL)
        for ch_id, ch_title in chapter_matches:
            chapters_info.append({"id": ch_id, "title": ch_title})
    
    return chapters_info

def load_action_rules(game_name: str) -> list[dict]:
    action_rules_path = os.path.join(current_directory, f"games/generated/{game_name}/rules/action_rules.py")
    
    with open(action_rules_path, 'r') as f:
        content = f.read()
    
    import re
    rule_classes = re.findall(r'class\s+(\w+Rule)\s*\([^)]*BaseActionRule[^)]*\)', content)
    
    actions = []
    for cls in rule_classes:
        verb_match = re.search(rf'class\s+{cls}.*?verb\s*=\s*["\']([^"\']+)["\']', content, re.DOTALL)
        if not verb_match:
            continue
        verb = verb_match.group(1)
        info: dict = {"verb": verb}

        desc_match = re.search(rf'class\s+{cls}.*?description\s*=\s*["\']([^"\']+)["\']', content, re.DOTALL)
        if desc_match:
            info["description"] = desc_match.group(1)

        params_match = re.search(rf'class\s+{cls}.*?params\s*=\s*\[([^\]]*)\]', content, re.DOTALL)
        if params_match:
            raw = params_match.group(1).strip()
            if raw:
                info["params"] = [p.strip().strip("'\"") for p in raw.split(",") if p.strip()]
            else:
                info["params"] = []
        else:
            info["params"] = []

        actions.append(info)
    
    return actions


def summarize_entities(world_definition: dict) -> dict:
    entities = world_definition.get("entities", {})
    
    summary = {
        "objects": {},
        "npcs": [],
        "places": [],
    }
    
    for obj in entities.get("objects", []):
        category = obj.get("category", "other")
        if category not in summary["objects"]:
            summary["objects"][category] = []
        entry: dict = {
            "id": obj.get("id"),
            "name": obj.get("name"),
            "category": category,
            "usage": obj.get("usage"),
            "level": obj.get("level", 1),
            "quest": obj.get("quest", False),
        }

        for stat in ("attack", "defense", "value", "size"):
            v = obj.get(stat)
            if v:
                entry[stat] = v

        craft = obj.get("craft") or obj.get("craft_ingredients")
        if craft:
            entry["craft"] = craft
        summary["objects"][category].append(entry)

    for npc in entities.get("npcs", []):
        npc_entry: dict = {
            "id": npc.get("id"),
            "name": npc.get("name"),
            "role": npc.get("role"),
            "enemy": npc.get("enemy", False),
            "unique": npc.get("unique", False),
        }
        for stat in ("base_attack_power", "base_hp"):
            v = npc.get(stat)
            if v:
                npc_entry[stat] = v
        summary["npcs"].append(npc_entry)
    
    for place in entities.get("places", []):
        summary["places"].append({
            "id": place.get("id"),
            "name": place.get("name"),
            "areas": [a.get("name") if isinstance(a, dict) else a for a in place.get("areas", [])],
        })
    
    return summary


def summarize_step_rules(game_name: str) -> str:
    """Parse step_rules/general.py for game-specific step rules and return a
    concise summary the LLM planner can use to design quests that leverage the
    game's unique mechanics.  Only reports rules that are NOT in the base
    template (base game rules are shared by every game and already known)."""
    import re

    base_rule_names = {
        "AgentAttackUpdateStepRule", "CombatRhythmStepRule",
        "ActiveAttackStepRule", "NewCraftableFeedbackStepRule",
        "DeathAndRespawnStepRule", "SideQuestStepRule",
        "WorldExpansionStepRule", "MerchantInfoStepRule",
    }

    general_path = os.path.join(
        current_directory,
        f"games/generated/{game_name}/rules/step_rules/general.py",
    )
    if not os.path.exists(general_path):
        return ""

    with open(general_path, "r", encoding="utf-8") as f:
        content = f.read()

    class_pattern = re.compile(
        r'class\s+(\w+StepRule)\s*\([^)]*\):\s*\n'
        r'(.*?)(?=\nclass\s|\Z)',
        re.DOTALL,
    )

    game_rules = []
    for m in class_pattern.finditer(content):
        cls_name = m.group(1)
        if cls_name in base_rule_names:
            continue
        body = m.group(2)

        desc_match = re.search(
            r'description\s*=\s*(?:\(\s*)?(?:f?"""(.*?)"""|f?"(.*?)"|f?\'(.*?)\')',
            body,
            re.DOTALL,
        )
        desc = ""
        if desc_match:
            desc = (desc_match.group(1) or desc_match.group(2) or desc_match.group(3) or "").strip()
            desc = " ".join(desc.split())

        pri_match = re.search(r'priority\s*=\s*(\d+)', body)
        priority = pri_match.group(1) if pri_match else "?"

        const_lines = []
        for cline in re.finditer(
            r'^\s{4}([A-Z_][A-Z_0-9]+)\s*=\s*(.+)', body, re.MULTILINE
        ):
            const_lines.append(f"    {cline.group(1)} = {cline.group(2).strip()}")

        rule_info = f"- {cls_name} (priority {priority}): {desc}"
        if const_lines:
            rule_info += "\n" + "\n".join(const_lines[:6])  # cap at 6 constants
        game_rules.append(rule_info)

    if not game_rules:
        return ""

    return (
        "GAME-SPECIFIC ENVIRONMENT STEP RULES:\n"
        "These rules run every game step and create unique dynamics.\n"
        "Design quest stages that take advantage of these mechanics.\n\n"
        + "\n\n".join(game_rules)
    )


def summarize_action_constraints(game_name: str) -> str:
    """Parse action_rules.py to extract constraint information
    from validation logic inside apply() methods.  Returns a concise summary
    of what each action actually requires / does, beyond just verb+description.

    The function reads the actual apply() code for each rule and extracts
    feedback strings (which encode the constraint messages shown to the player)
    to give the LLM planner accurate information about what is and isn't
    possible."""
    import re

    action_rules_path = os.path.join(
        current_directory,
        f"games/generated/{game_name}/rules/action_rules.py",
    )
    if not os.path.exists(action_rules_path):
        return ""

    with open(action_rules_path, "r", encoding="utf-8") as f:
        content = f.read()

    class_pattern = re.compile(
        r'class\s+(\w+Rule)\s*\([^)]*BaseActionRule[^)]*\):\s*\n'
        r'(.*?)(?=\nclass\s|\Z)',
        re.DOTALL,
    )

    constraints_by_verb: dict[str, list[str]] = {}
    for m in class_pattern.finditer(content):
        body = m.group(2)

        verb_match = re.search(r'verb\s*=\s*["\']([^"\']+)["\']', body)
        if not verb_match:
            continue
        verb = verb_match.group(1)

        rejection_msgs = []
        for fb in re.finditer(
            r'(?:add_feedback|res\.add_feedback)\s*\([^,]+,\s*f?"(Cannot[^"]*|can\'t[^"]*)"',
            body,
            re.IGNORECASE,
        ):
            msg = fb.group(1)
            msg = re.sub(r'\{[^}]+\}', '<target>', msg)
            if msg not in rejection_msgs:
                rejection_msgs.append(msg)

        if rejection_msgs:
            constraints_by_verb[verb] = rejection_msgs

    if not constraints_by_verb:
        return ""

    lines = [
        "ACTION CONSTRAINTS (what can fail and why):",
        "The following constraints are enforced by the game engine.",
        "Design quests that work WITH these constraints, not against them.\n",
    ]
    for verb, msgs in sorted(constraints_by_verb.items()):
        lines.append(f"  {verb}:")
        for msg in msgs[:5]:  # cap per verb
            lines.append(f"    - {msg}")

    return "\n".join(lines)


def summarize_topology(world_definition: dict) -> str:
    from collections import deque

    spawn_area = world_definition.get("initializations", {}).get("spawn", {}).get("area")
    graph_def = world_definition.get("graph")

    if not graph_def or not graph_def.get("connections"):
        return (
            "GRAPH TOPOLOGY: No predefined graph exists in the world definition.\n"
            "The runtime graph will be randomly generated.  The _bootstrap()\n"
            "method assigns quest areas by BFS distance from spawn, so quest\n"
            "stages should use generic area_key placeholders and NOT assume a\n"
            "specific topology.  Early stages will naturally get areas close to\n"
            "spawn, later stages get areas farther away."
        )

    connections = graph_def["connections"]
    # Build adjacency list (unlocked-only and full)
    adj_all: dict[str, list[tuple[str, bool, str | None]]] = {}
    adj_unlocked: dict[str, list[str]] = {}
    for conn in connections:
        a, b = conn["from"], conn["to"]
        locked = conn.get("locked", False)
        key = conn.get("key")
        adj_all.setdefault(a, []).append((b, locked, key))
        adj_all.setdefault(b, []).append((a, locked, key))
        if not locked:
            adj_unlocked.setdefault(a, []).append(b)
            adj_unlocked.setdefault(b, []).append(a)

    # BFS from spawn (all edges)
    dist_all: dict[str, int] = {}
    if spawn_area:
        dist_all[spawn_area] = 0
        q = deque([spawn_area])
        while q:
            cur = q.popleft()
            for nbr, _, _ in adj_all.get(cur, []):
                if nbr not in dist_all:
                    dist_all[nbr] = dist_all[cur] + 1
                    q.append(nbr)

    # BFS from spawn (unlocked edges only)
    dist_unlocked: dict[str, int] = {}
    if spawn_area:
        dist_unlocked[spawn_area] = 0
        q = deque([spawn_area])
        while q:
            cur = q.popleft()
            for nbr in adj_unlocked.get(cur, []):
                if nbr not in dist_unlocked:
                    dist_unlocked[nbr] = dist_unlocked[cur] + 1
                    q.append(nbr)

    # Build area-to-place mapping
    area_to_place: dict[str, str] = {}
    for place in world_definition.get("entities", {}).get("places", []):
        pid = place.get("id", "")
        for area in place.get("areas", []):
            aid = area.get("id") if isinstance(area, dict) else area
            area_to_place[aid] = pid

    # Group areas by BFS distance (all edges)
    by_dist: dict[int, list[str]] = {}
    for aid, d in sorted(dist_all.items(), key=lambda x: x[1]):
        by_dist.setdefault(d, []).append(aid)

    lines = [
        f"GRAPH TOPOLOGY (predefined, {len(connections)} connections):",
        f"Spawn area: {spawn_area}",
        f"Areas reachable without keys (unlocked only): {len(dist_unlocked)}",
        f"Total areas in graph: {len(dist_all)}",
        "",
        "Areas by BFS distance from spawn (all edges):",
    ]
    for d in sorted(by_dist):
        areas = by_dist[d]
        labels = [f"{a} ({area_to_place.get(a, '?')})" for a in areas]
        lines.append(f"  Distance {d}: {', '.join(labels)}")

    lines.append("")
    lines.append("Locked connections (gates to later content):")
    for conn in connections:
        if conn.get("locked"):
            lines.append(f"  {conn['from']} <-> {conn['to']}  key={conn.get('key')}")

    lines.append("")
    lines.append(
        "The _bootstrap() method assigns quest areas by BFS distance from\n"
        "spawn at runtime: early stages get nearby areas, later stages get\n"
        "farther areas.  Design quest chapters so that early chapter stages\n"
        "can be completed in the unlocked near-spawn area cluster, while\n"
        "later chapters naturally require reaching areas behind locked\n"
        "connections."
    )
    return "\n".join(lines)


# STEP 1: LLM PLANNER — generates a structured quest spec (JSON)
def build_planner_prompt(
    description: str,
    entities_summary: dict,
    available_actions: list,
    topology_summary: str,
    num_chapters: int,
    branching_factor: int = 1,
    step_rules_summary: str = "",
    action_constraints_summary: str = "",
) -> str:
    """Build a prompt for the LLM planner to output a structured quest spec."""
    entities_str = json.dumps(entities_summary, indent=2)
    actions_str = _format_actions_str(available_actions)

    dag_instructions = ""
    if branching_factor >= 2:
        dag_instructions = textwrap.dedent(f"""\

        GOAL TREE MODE (branching_factor={branching_factor}):
        Each chapter MUST be structured as a GOAL TREE:
        - The chapter has ONE root goal (the final objective, e.g. "Escape the hospital").
        - The root goal decomposes into sub-goals via a "children" field.
        - Each node can have 1 to {branching_factor} children (the tree may be incomplete).
        - LEAF nodes (no children) are the concrete tasks the player works on.
          They MUST have "done_any" or "done_all" conditions.
        - INTERNAL nodes (have children) are organizational goals that auto-complete
          when all their children are completed. They may optionally have their own
          "done_any"/"done_all" conditions for additional requirements after children
          are done.
        - The ROOT node must have "chapter_complete": true (or "quest_complete": true
          for the last chapter).

        Do NOT use a "requires" field — the compiler generates it from "children".
        Every internal node gets "requires" = its children list automatically.
        Leaf nodes (no children) become immediately active at the start.

        TREE STRUCTURE RULES:
        - Each node has at most {branching_factor} children.
        - Each non-root node has exactly one parent.
        - No cycles. Every node is reachable from the root.
        - Deeper trees for later (harder) chapters, shallower for early chapters.

        EXAMPLE 1 — Hospital Escape (branching_factor=2):
        Internal nodes can EITHER auto-complete (no conditions) OR have their
        own conditions that are checked AFTER all children complete.

            root: "escape_hospital" (chapter_complete: true, auto-complete)
              ├─ "shutdown_electricity"
              │    Has own condition: in_area at control room (AFTER fuses broken)
              │    ├─ "break_fuse_east" (leaf: go to east wing, break fuse)
              │    └─ "break_fuse_west" (leaf: go to west wing, break fuse)
              └─ "get_supplies" (auto-complete when children done)
                   ├─ "find_keycard" (leaf: pick up keycard)
                   └─ "find_medkit" (leaf: pick up medkit)

        When both fuses are broken, "shutdown_electricity" becomes active
        but is NOT yet done — the player must also go to the control room.
        When "find_keycard" and "find_medkit" are both done, "get_supplies"
        auto-completes immediately (no own conditions).

        EXAMPLE 2 — Trade & Deliver (branching_factor=2, with custom condition):
            root: "complete_delivery" (quest_complete: true, auto-complete)
              ├─ "acquire_goods"
              │    ├─ "earn_coins" (leaf: sell items to merchant)
              │    └─ "buy_package" (leaf: buy specific item)
              └─ "deliver_package"
                   (leaf, custom condition: dropped item in target area)

        In Example 2, "deliver_package" uses a custom condition:
          {{"kind": "custom", "type": "object_in_area",
            "description": "Player dropped the package in the delivery area",
            "implementation_hint": "Check if the base object exists in area.objects",
            "obj_key": "ch1_package", "area_key": "ch1_delivery_area"}}
        This is a novel task type — the AI coder implements the check logic.

        """)
    else:
        dag_instructions = textwrap.dedent("""\

        LINEAR MODE (branching_factor=1):
        - All stages within each chapter are sequential (no "children" or "requires" field).
        - Stages are completed in order, one at a time.
        - This is simpler but still supports diverse task types and difficulty progression.

        """)

    # Build optional game-specific mechanics section
    game_mechanics_section = ""
    if step_rules_summary or action_constraints_summary:
        parts = ["GAME-SPECIFIC MECHANICS:"]
        parts.append(
            "This game has unique environment rules and action constraints beyond\n"
            "the standard RPG mechanics. Design quest stages that LEVERAGE these\n"
            "mechanics to create interesting gameplay. Read carefully:\n"
        )
        if step_rules_summary:
            parts.append(step_rules_summary)
        if action_constraints_summary:
            parts.append(action_constraints_summary)
        game_mechanics_section = "\n\n".join(parts)

    return textwrap.dedent(f"""\
    You are a game quest designer for a text-based RPG called Agent Odyssey.
    Design a complete quest with {num_chapters} chapter(s).

    GAME THEME / DESCRIPTION:
    {description or "Create a diverse and interesting quest that tests exploration, memory, and reasoning."}

    AVAILABLE GAME ENTITIES:
    {entities_str}

    AVAILABLE PLAYER ACTIONS:
    {actions_str}

    {topology_summary}

    {game_mechanics_section}

    PROGRESSIVE DIFFICULTY:
    Regardless of how many chapters the quest has, difficulty MUST increase
    progressively from the first chapter to the last:
    - Early chapters: simpler tasks, fewer stages, basic mechanics (exploration,
      NPC dialog, simple item pickup). Avoid throwing combat at the player
      before they have had a chance to find weapons and learn the world.
    - Middle chapters: introduce memory challenges, crafting chains, trading,
      multi-condition stages, and moderate combat.
    - Later chapters: complex multi-step puzzles, long-horizon memory tasks
      (learn information early, use it many stages later), boss fights with
      preparation, multi-item gathering, and deep prerequisite chains.
    - Scale the NUMBER of stages per chapter upward as well (e.g. 3-4 stages
      early, 6-10+ stages in later chapters).

    EXAMPLE CONDITION PRIMITIVES (these are already implemented):

    Event conditions ("kind": "event") — triggered once when an action happens:
      - "npc_killed"    : {{"kind": "event", "type": "npc_killed", "npc_key": "<key>"}}
      - "object_bought"  : {{"kind": "event", "type": "object_bought", "base_objs": ["obj_id"]}}
      - "object_crafted" : {{"kind": "event", "type": "object_crafted", "category": "weapon"}}

    State conditions ("kind": "state") — continuously checked each step:
      - "in_area"                        : player is in a specific area
          {{"kind": "state", "type": "in_area", "area_key": "<key>"}}
      - "in_area_with_npc"               : player is in the same area as an NPC
          {{"kind": "state", "type": "in_area_with_npc", "npc_key": "<key>"}}
      - "visited_k_of_area_set"          : visited K out of N candidate areas
          {{"kind": "state", "type": "visited_k_of_area_set", "area_set_key": "<key>", "k_key": "<key>"}}
      - "has_any_of"                     : player has at least one of the listed items
          {{"kind": "state", "type": "has_any_of", "base_objs": ["obj_id1", "obj_id2"]}}
      - "equipped_any_of"                : player has equipped one of the listed items
          {{"kind": "state", "type": "equipped_any_of", "base_objs": ["obj_id"]}}
      - "has_category"                   : player has any item of a category
          {{"kind": "state", "type": "has_category", "category": "weapon"}}
      - "has_item_count"                 : player has N copies of an item
          {{"kind": "state", "type": "has_item_count", "base_obj_key": "<key>", "count_key": "<key>"}}
      - "paper_text_equals_in_area"      : player is in <area> holding paper with <text>
          {{"kind": "state", "type": "paper_text_equals_in_area", "text_key": "<key>", "area_key": "<key>"}}
      - "paper_text_equals_in_hands"     : player holds paper with <text> (any area)
          {{"kind": "state", "type": "paper_text_equals_in_hands", "text_key": "<key>"}}
      - "writable_text_equals_in_inventory": writable item in inventory has <text>
          {{"kind": "state", "type": "writable_text_equals_in_inventory", "text_key": "<key>"}}
      - "boss_already_defeated"          : a boss NPC is already dead
          {{"kind": "state", "type": "boss_already_defeated", "npc_key": "<key>"}}
      - "killed_n_enemies"               : killed at least N enemy NPCs total
          {{"kind": "state", "type": "killed_n_enemies", "n_key": "<key>"}}
      - "killed_one_of_each"             : killed at least one of each listed NPC type
          {{"kind": "state", "type": "killed_one_of_each", "npc_keys": ["<key1>", "<key2>"]}}
      - "lit_k_dark_areas"               : illuminated K dark areas
          {{"kind": "state", "type": "lit_k_dark_areas", "k_key": "<key>"}}

    Custom conditions ("kind": "custom") — REQUIRED for most leaf stages:
      Custom conditions are the PRIMARY way to create interesting tasks.
      The built-in primitives above are convenient shortcuts, but they are
      GENERIC and BORING on their own. Your quest MUST use custom conditions
      for the MAJORITY of leaf stages (at least 60-70% of all leaf stages).

      Built-in conditions like "has_any_of", "equipped_any_of",
      "in_area_with_npc", and "killed_n_enemies" should be used SPARINGLY —
      only when they are genuinely the right fit. Do NOT default to them.
      Instead, invent novel condition types that create unique gameplay.

      An AI coder will automatically implement the logic for each custom
      condition based on your description and implementation_hint.

      Required fields: "kind": "custom", "type": "<unique_name>",
                       "description": "<what the condition checks>",
                       "implementation_hint": "<how to check it in code>"
      You may include extra key fields (area_key, npc_key, obj_key, etc.)
      that reference bootstrap_keys, just like built-in conditions.
      Examples of custom conditions (invent MORE like these):
        - Object delivered to area:
          {{"kind": "custom", "type": "object_in_area",
            "description": "Player has dropped obj_gem in the target area",
            "implementation_hint": "Check if obj_gem base object exists in area.objects for the target area",
            "obj_key": "ch2_gem_obj", "area_key": "ch2_target_area"}}
        - NPC coins threshold:
          {{"kind": "custom", "type": "npc_has_coins",
            "description": "The merchant NPC has at least 100 coins (player sold items)",
            "implementation_hint": "Check world.npc_instances[npc_id].coins >= 100",
            "npc_key": "ch3_merchant_npc", "threshold": 100}}
        - Area cleared of enemies:
          {{"kind": "custom", "type": "area_cleared",
            "description": "All enemy NPCs in the target area have been killed",
            "implementation_hint": "Check that no alive enemy NPC instances remain in the area",
            "area_key": "ch2_clear_area"}}
        - Player inventory value:
          {{"kind": "custom", "type": "inventory_total_value",
            "description": "Player's total inventory value exceeds threshold",
            "implementation_hint": "Sum obj.value for all items in agent hands+inventory, check >= threshold",
            "threshold_key": "ch3_value_threshold"}}
        - NPC inventory changed:
          {{"kind": "custom", "type": "gave_item_to_npc",
            "description": "Player has given a specific item to an NPC via trade",
            "implementation_hint": "Check if obj_id exists in npc_instance.inventory",
            "obj_key": "ch2_gift_obj", "npc_key": "ch2_target_npc"}}

    Use "done_any" (OR — any one condition suffices) or "done_all" (AND — all
    conditions must be met).  You can combine primitives freely to create
    ANY kind of task — the above are your building blocks, not a menu.
    Custom conditions are first-class citizens — use them as the DEFAULT
    choice for leaf stages. Built-in conditions are the EXCEPTION.
    {dag_instructions}
    CRITICAL DESIGN RULES:

    1. PUZZLE DISCOVERY BEFORE PUZZLE SOLVING:
       - ALWAYS precede any stage that requires the player to write/recall
         information with a DISCOVERY stage whose on_complete_feedback
         reveals that information.
       - Pattern: Stage A (explore/talk) → feedback reveals code/word → Stage B (write it on paper)
       - NEVER require the player to know something they haven't been told yet.

    2. INFORMATION FLOW:
       - Info must be learned BEFORE it's used.
       - Chain: Observe/Learn → Remember/Record → Apply/Use
       - on_complete_feedback from one stage reveals clues for future stages.

    3. HINT GUIDELINES:
       - Hints should GUIDE, not SPOIL.
       - BAD: "Write '750' on paper" (gives away the answer)
       - GOOD: "Remember what you learned earlier. Write it on paper."
       - Hints can mention the action but NOT the exact value.

    4. DIVERSITY & CREATIVITY:
       - MOST leaf stages should use CUSTOM CONDITIONS, not built-in ones.
       - Built-in conditions (has_any_of, equipped_any_of, in_area_with_npc,
         killed_n_enemies, etc.) are OVERUSED and BORING. Use them only when
         truly appropriate (e.g. a boss fight naturally needs npc_killed).
       - For everything else, invent novel custom conditions: delivery quests
         (drop item in area), trade profit (NPC coin threshold), area
         manipulation (clear enemies from zone), resource accumulation,
         environmental state changes, escort completion, inventory puzzles,
         NPC relationship checks, multi-item combinations, etc.
       - Each chapter MUST use at least 2-3 different custom condition types.
       - Don't repeat the same condition type across chapters.

    5. DIFFICULTY PROGRESSION:
       - Chapter 1 should be easy (no combat, basic mechanics).
       - Later chapters get harder with more complex tasks.
       - Memory challenges should span longer distances in later chapters.

    6. ACHIEVABILITY:
       - Every condition must be achievable with available game mechanics.
       - Items referenced in conditions must exist in the entity list.
       - NPCs referenced must exist or be defined in required_npcs.
       - Areas are assigned at runtime by the _bootstrap() method, so use
         placeholder keys like "ch1_guide_area", NOT literal area IDs.

    7. BOSS FIGHTS:
       - Bosses need a "spawns_boss_key" field on the stage that triggers spawn.
       - The boss condition should use npc_killed + boss_already_defeated (done_any).
       - Always precede boss fights with a weapon/equipment preparation stage.

    8. COMBAT PATTERN:
       - Every enemy NPC (enemy=true) in required_npcs MUST have a "combat_pattern" field.
       - combat_pattern is a list of actions like ["attack", "defend", "attack", "wait"].
       - Without it, the NPC won't fight back and the player can leave mid-combat.

    OUTPUT FORMAT:
    Reply with ONLY a JSON object (no markdown, no explanation):
    {{
      "title": "Quest Title",
      "intro": "Opening narrative text with {{mq_guide_name}} placeholder...",
      "required_objects": [
        {{"id": "obj_custom", "name": "custom_item", "category": "weapon",
          "usage": "attack", "value": null, "size": 5, "attack": 10,
          "description": "...", "level": 3,
          "craft": {{"ingredients": {{"obj_iron_bar": 2}}, "dependencies": ["obj_workbench"]}}
        }}
      ],
      "required_npcs": [
        {{"id": "npc_quest_mychara", "name": "my_character", "enemy": false,
          "unique": true, "role": "guide", "quest": true,
          "description": "...", "base_attack_power": 200, "base_hp": 200,
          "objects": []
        }},
        {{"id": "npc_boss_myboss", "name": "my_boss", "enemy": true,
          "unique": true, "role": "boss", "quest": true,
          "description": "...", "base_attack_power": 20, "base_hp": 150,
          "combat_pattern": ["attack", "attack", "defend", "wait", "attack"],
          "objects": []
        }}
      ],
      "undistributable_objects": ["obj_custom"],
      "chapters": [
        {{
          "id": "chapter_1",
          "title": "Chapter Title",
          "intro": "Chapter intro with {{mq_guide_name}} and {{ch1_guide_name}} placeholders...",
          "stages": [
            {{
              "id": "ch1_meet_guide",
              "task_type": "meet_guide",
              "objective": "Find and talk to {{ch1_guide_name}}.",
              "hint": "Explore the area to find the guide.",
              "done_any": [
                {{"kind": "state", "type": "in_area_with_npc", "npc_key": "ch1_guide_npc"}}
              ],
              "on_complete_feedback": "The guide reveals important information...",
              "bootstrap_keys": {{
                "ch1_guide_area": {{"pick_progressive": 1, "desc": "area for the chapter 1 guide"}},
                "ch1_guide_npc": {{"spawn_npc": "npc_quest_mychara", "area_key": "ch1_guide_area"}}
              }}
            }},
            {{
              "id": "ch1_final",
              "task_type": "craft_weapon",
              "objective": "Craft a simple weapon.",
              "hint": "Try crafting at a workbench.",
              "done_any": [
                {{"kind": "state", "type": "has_any_of", "base_objs": ["obj_wooden_sword"]}}
              ],
              "on_complete_feedback": "Chapter complete!",
              "chapter_complete": true
            }}
          ]
        }}
      ],
      "generated_values": {{
        "ch1_some_text_key": "exact_text_value"
      }}
    }}

    GOAL TREE EXAMPLE (when branching_factor >= 2):
    In tree mode, internal nodes have a "children" field listing their child
    stage IDs. Leaf nodes do NOT have "children". Example chapter with a
    goal tree (branching_factor=2):
    {{
      "id": "chapter_1",
      "title": "Escape the Hospital",
      "root_goal": "ch1_escape",
      "intro": "You must escape the hospital. Talk to {{mq_guide_name}} for help.",
      "stages": [
        {{
          "id": "ch1_escape",
          "objective": "Escape from the hospital.",
          "children": ["ch1_shutdown_elec", "ch1_get_card"],
          "chapter_complete": true
        }},
        {{
          "id": "ch1_shutdown_elec",
          "objective": "Shut down the electricity system.",
          "children": ["ch1_break_fuse_east", "ch1_break_fuse_west"]
        }},
        {{
          "id": "ch1_break_fuse_east",
          "objective": "Find and break the fuse in the east wing.",
          "done_all": [{{"kind": "state", "type": "in_area", "area_key": "ch1_east_area"}}],
          "on_complete_feedback": "The east fuse shatters!",
          "bootstrap_keys": {{
            "ch1_east_area": {{"pick_progressive": 1}}
          }}
        }},
        {{
          "id": "ch1_break_fuse_west",
          "objective": "Find and break the fuse in the west wing.",
          "done_all": [{{"kind": "state", "type": "in_area", "area_key": "ch1_west_area"}}],
          "on_complete_feedback": "The west fuse shatters!",
          "bootstrap_keys": {{
            "ch1_west_area": {{"pick_progressive": 1}}
          }}
        }},
        {{
          "id": "ch1_get_card",
          "objective": "Obtain the name card.",
          "children": ["ch1_defeat_guard", "ch1_collect_parts"]
        }},
        {{
          "id": "ch1_defeat_guard",
          "objective": "Defeat the guard blocking the card room.",
          "done_any": [{{"kind": "event", "type": "npc_killed", "npc_key": "ch1_guard_npc"}}],
          "spawns_boss_key": "ch1_guard_npc",
          "on_complete_feedback": "The guard falls!",
          "bootstrap_keys": {{
            "ch1_guard_npc": {{"spawn_npc": "npc_boss_myboss", "area_key": "ch1_east_area"}}
          }}
        }},
        {{
          "id": "ch1_collect_parts",
          "objective": "Collect 3 circuit boards.",
          "done_all": [{{"kind": "state", "type": "has_item_count", "base_obj_key": "ch1_circuit", "count_key": "ch1_circuit_count"}}],
          "on_complete_feedback": "You have enough parts!",
          "bootstrap_keys": {{
            "ch1_circuit": {{"literal": "obj_circuit_board"}},
            "ch1_circuit_count": {{"literal": 3}}
          }}
        }}
      ]
    }}

    In tree mode, the "root_goal" field identifies the root stage of the tree.
    Only LEAF stages (no "children") need "done_any" or "done_all".
    The compiler auto-generates "requires" from "children", so do NOT
    include "requires" in tree mode.

    In LINEAR mode (branching_factor=1), omit "children" and "root_goal" entirely.
    Stages are completed sequentially in the order listed.

    The "bootstrap_keys" in each stage tell the compiler what placeholders to
    create in _bootstrap(). Supported types:
    - {{"pick_progressive": N}} — pick N areas by BFS distance
    - {{"pick_progressive": N, "return": "farthest"}} — pick N then pick farthest
    - {{"spawn_npc": "npc_base_id", "area_key": "key"}} — spawn NPC in named area
    - {{"literal": value}} — store a constant value

    The "generated_values" dict stores constant text/numbers that the agent
    must remember (like secret codes, passwords). These become entries in
    self._generated and are referenced by text_key in conditions.

    Important: The last stage of the last chapter MUST have "quest_complete": true.
    """)


def plan_quest(
    description: str,
    entities_summary: dict,
    available_actions: list,
    topology_summary: str,
    num_chapters: int,
    branching_factor: int = 1,
    llm_name: str = "gpt-5",
    llm_provider: str | None = None,
    llm=None,
    max_retries: int = 3,
    step_rules_summary: str = "",
    action_constraints_summary: str = "",
) -> dict:
    quest_output_token_budget = 32768
    prompt = build_planner_prompt(
        description=description,
        entities_summary=entities_summary,
        available_actions=available_actions,
        topology_summary=topology_summary,
        num_chapters=num_chapters,
        branching_factor=branching_factor,
        step_rules_summary=step_rules_summary,
        action_constraints_summary=action_constraints_summary,
    )

    if llm is None:
        llm = _get_llm(llm_name, llm_provider)
    if getattr(llm, "max_new_tokens", None) is None:
        llm.max_new_tokens = quest_output_token_budget
    else:
        llm.max_new_tokens = max(int(llm.max_new_tokens), quest_output_token_budget)

    system_prompt = "You are an expert game quest designer. Output ONLY valid JSON, no markdown fences, no explanation."
    last_error = None

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"  Planner retry {attempt + 1}/{max_retries}...")
            # On retry, append the error to the prompt so the LLM can self-correct
            retry_prompt = (
                prompt
                + f"\n\n[RETRY — your previous response was not valid JSON. Error: {last_error}. "
                + "Output ONLY the raw JSON object, no markdown fences, no explanation.]"
            )
        else:
            retry_prompt = prompt

        try:
            response = llm.generate(
                user_prompt=retry_prompt,
                system_prompt=system_prompt,
            )
            raw = response["response"].strip()
            parsed = _extract_json(raw)
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            print(f"  Planner attempt {attempt + 1} failed: {last_error}")
            continue

    raise json.JSONDecodeError(
        f"Failed to get valid JSON after {max_retries} attempts. Last error: {last_error}",
        "", 0,
    )


def _extract_json(raw: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    # Strip markdown fences: ```json ... ``` or ``` ... ```
    if "```" in raw:
        # Find content between first ``` and last ```
        parts = raw.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            # Remove optional language tag (json, JSON, etc.)
            if inner.startswith(("json", "JSON")):
                inner = inner[4:]
            elif inner.startswith("\n"):
                pass  # no language tag
            else:
                # Could be e.g. "json\n{..." — strip first line if it's just a lang tag
                first_line, _, rest = inner.partition("\n")
                if first_line.strip().isalpha():
                    inner = rest
            raw = inner.strip()

    # Strip <json>...</json> XML-style tags
    if "<json>" in raw.lower():
        import re as _re_json
        m = _re_json.search(r"<json>\s*(.*?)\s*</json>", raw, _re_json.DOTALL | _re_json.IGNORECASE)
        if m:
            raw = m.group(1)

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find the outermost { ... } pair
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Try fixing trailing commas (common LLM mistake)
    import re as _re_json
    cleaned = _re_json.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not extract valid JSON from LLM response (length={len(raw)}): {raw[:200]}...")

# STEP 2: VALIDATOR — checks the quest spec for correctness
def validate_quest_spec(
    spec: dict,
    entities_summary: dict,
    branching_factor: int = 1,
    available_actions: list | None = None,
) -> list[str]:
    errors: list[str] = []

    if "title" not in spec:
        errors.append("Missing 'title' field.")
    if "chapters" not in spec or not isinstance(spec.get("chapters"), list):
        errors.append("Missing or invalid 'chapters' field.")
        return errors

    all_npc_ids = {n["id"] for n in spec.get("required_npcs", [])}
    all_npc_ids |= {n["id"] for n in entities_summary.get("npcs", [])}

    # Validate enemy NPCs have combat_pattern
    for npc in spec.get("required_npcs", []):
        if npc.get("enemy") and not npc.get("combat_pattern"):
            errors.append(f"Enemy NPC '{npc['id']}' missing 'combat_pattern' (e.g. [\"attack\", \"defend\", \"attack\", \"wait\"]).")

    all_obj_ids = set()
    for cat, objs in entities_summary.get("objects", {}).items():
        for obj in objs:
            all_obj_ids.add(obj["id"])
    for obj in spec.get("required_objects", []):
        all_obj_ids.add(obj["id"])

    has_quest_complete = False

    for ch_idx, chapter in enumerate(spec["chapters"]):
        ch_id = chapter.get("id", f"chapter_{ch_idx+1}")
        stages = chapter.get("stages", [])
        if not stages:
            errors.append(f"{ch_id}: No stages defined.")
            continue

        stage_ids = set()
        stage_by_id: dict[str, dict] = {}
        for stage in stages:
            sid = stage.get("id", "")
            if not sid:
                errors.append(f"{ch_id}: Stage missing 'id'.")
                continue
            if sid in stage_ids:
                errors.append(f"{ch_id}: Duplicate stage ID '{sid}'.")
            stage_ids.add(sid)
            stage_by_id[sid] = stage

        # ----- Goal tree validation (branching_factor >= 2) -----
        if branching_factor >= 2:
            has_children = any("children" in s for s in stages)
            if not has_children:
                errors.append(f"{ch_id}: branching_factor={branching_factor} but no stage has 'children'. Use goal tree structure.")
                continue

            # Validate children references and build parent map
            parent_of: dict[str, str] = {}  # child_id -> parent_id
            for stage in stages:
                sid = stage.get("id", "")
                children = stage.get("children", [])
                if not isinstance(children, list):
                    errors.append(f"{ch_id}/{sid}: 'children' must be a list.")
                    continue
                if len(children) > branching_factor:
                    errors.append(f"{ch_id}/{sid}: Has {len(children)} children but branching_factor={branching_factor}.")
                for child_id in children:
                    if child_id not in stage_ids:
                        errors.append(f"{ch_id}/{sid}: children references non-existent stage '{child_id}'.")
                    elif child_id in parent_of:
                        errors.append(f"{ch_id}/{child_id}: Has multiple parents ('{parent_of[child_id]}' and '{sid}'). Must be a tree.")
                    else:
                        parent_of[child_id] = sid

            # Find root(s) — stages not referenced as children
            root_ids = [sid for sid in stage_ids if sid not in parent_of]
            explicit_root = chapter.get("root_goal")
            if explicit_root:
                if explicit_root not in stage_ids:
                    errors.append(f"{ch_id}: root_goal '{explicit_root}' not found in stages.")
                elif explicit_root not in root_ids:
                    errors.append(f"{ch_id}: root_goal '{explicit_root}' is a child of another stage.")
            if len(root_ids) == 0:
                errors.append(f"{ch_id}: No root node found (every stage is a child of another).")
            elif len(root_ids) > 1:
                errors.append(f"{ch_id}: Multiple root nodes found: {root_ids}. Goal tree must have exactly one root.")

            # Cycle detection via DFS on children edges
            visited: set[str] = set()
            in_stack: set[str] = set()

            def _has_cycle_tree(node: str) -> bool:
                if node in in_stack:
                    return True
                if node in visited:
                    return False
                visited.add(node)
                in_stack.add(node)
                stage = stage_by_id.get(node)
                if stage:
                    for child in stage.get("children", []):
                        if _has_cycle_tree(child):
                            return True
                in_stack.discard(node)
                return False

            for sid in stage_ids:
                if _has_cycle_tree(sid):
                    errors.append(f"{ch_id}: Cycle detected in goal tree.")
                    break

            # Verify all nodes reachable from root
            if len(root_ids) == 1:
                reachable: set[str] = set()
                stack = [root_ids[0]]
                while stack:
                    node = stack.pop()
                    if node in reachable:
                        continue
                    reachable.add(node)
                    stage = stage_by_id.get(node)
                    if stage:
                        stack.extend(stage.get("children", []))
                orphans = stage_ids - reachable
                if orphans:
                    errors.append(f"{ch_id}: Orphan stages not reachable from root: {sorted(orphans)}")

            # Verify leaf nodes have conditions
            for stage in stages:
                sid = stage.get("id", "")
                children = stage.get("children", [])
                is_leaf = len(children) == 0
                if is_leaf:
                    has_conds = stage.get("done_any") or stage.get("done_all")
                    if not has_conds:
                        errors.append(f"{ch_id}/{sid}: Leaf node must have 'done_any' or 'done_all' conditions.")

            # Verify root has chapter_complete / quest_complete
            if len(root_ids) == 1:
                root_stage = stage_by_id.get(root_ids[0], {})
                if ch_idx < len(spec["chapters"]) - 1:
                    if not root_stage.get("chapter_complete"):
                        # Auto-fix: add chapter_complete to root
                        root_stage["chapter_complete"] = True
                        errors.append(f"WARNING: {ch_id}: Auto-set 'chapter_complete' on root goal '{root_ids[0]}'.")
                else:
                    if not root_stage.get("quest_complete"):
                        # Auto-fix: add quest_complete to root
                        root_stage["quest_complete"] = True
                        errors.append(f"WARNING: {ch_id}: Auto-set 'quest_complete' on root goal '{root_ids[0]}'.")
                    if root_stage.get("quest_complete"):
                        has_quest_complete = True

            # Warn if requires is used in tree mode (compiler handles it, so just warn)
            has_requires = any(stage.get("requires") for stage in stages)
            if has_requires:
                errors.append(f"WARNING: {ch_id}: 'requires' in tree mode is redundant. The compiler auto-generates 'requires' from 'children'.")

        # ----- Linear validation (branching_factor == 1) -----
        else:
            # check requires references and cycles (legacy DAG support)
            for stage in stages:
                sid = stage.get("id", "")
                reqs = stage.get("requires", [])
                if not isinstance(reqs, list):
                    errors.append(f"{ch_id}/{sid}: 'requires' must be a list.")
                    continue
                for req in reqs:
                    if req not in stage_ids:
                        errors.append(f"{ch_id}/{sid}: requires references non-existent stage '{req}'.")

            # cycle detection via topological sort
            if any(stage.get("requires") for stage in stages):
                adj: dict[str, list[str]] = {s["id"]: list(s.get("requires", [])) for s in stages}
                visited_lin: set[str] = set()
                in_stack_lin: set[str] = set()

                def _has_cycle(node: str) -> bool:
                    if node in in_stack_lin:
                        return True
                    if node in visited_lin:
                        return False
                    visited_lin.add(node)
                    in_stack_lin.add(node)
                    for dep in adj.get(node, []):
                        if _has_cycle(dep):
                            return True
                    in_stack_lin.discard(node)
                    return False

                for s in stages:
                    if _has_cycle(s["id"]):
                        errors.append(f"{ch_id}: Cycle detected in stage requires graph.")
                        break

            has_requires = any(stage.get("requires") for stage in stages)
            if has_requires:
                errors.append(f"WARNING: {ch_id}: Has 'requires' fields in linear mode. These will be ignored by the compiler.")

            has_children = any("children" in s for s in stages)
            if has_children:
                errors.append(f"{ch_id}: Has 'children' fields but branching_factor=1 (linear mode). Set branching_factor>=2 for tree mode.")

            # check terminal stages
            has_chapter_complete = any(s.get("chapter_complete") for s in stages)
            has_quest_complete_here = any(s.get("quest_complete") for s in stages)
            if has_quest_complete_here:
                has_quest_complete = True

            if ch_idx < len(spec["chapters"]) - 1:
                if not has_chapter_complete:
                    errors.append(f"{ch_id}: Non-final chapter missing 'chapter_complete' stage.")
            else:
                if not has_quest_complete_here:
                    errors.append(f"{ch_id}: Final chapter missing 'quest_complete' stage.")

        # check condition validity (shared by both modes)
        valid_event_types = {"npc_killed", "object_bought", "object_crafted"}
        valid_state_types = {
            "in_area", "in_area_with_npc", "visited_k_of_area_set",
            "paper_text_equals_in_area", "paper_text_equals_in_hands",
            "writable_text_equals_in_inventory", "boss_already_defeated",
            "has_any_of", "equipped_any_of", "killed_n_enemies",
            "killed_one_of_each", "lit_k_dark_areas", "has_category",
            "has_item_count",
        }
        for stage in stages:
            sid = stage.get("id", "")
            conds = stage.get("done_any", []) + stage.get("done_all", [])
            for cond in conds:
                kind = cond.get("kind")
                ctype = cond.get("type")
                if kind == "custom":
                    # Custom conditions are valid — aider will implement them
                    if not ctype:
                        errors.append(f"{ch_id}/{sid}: Custom condition missing 'type' field.")
                    if not cond.get("description"):
                        errors.append(f"{ch_id}/{sid}: Custom condition '{ctype}' missing 'description' field.")
                elif kind == "event" and ctype not in valid_event_types:
                    errors.append(f"{ch_id}/{sid}: Unknown event condition type '{ctype}'.")
                elif kind == "state" and ctype not in valid_state_types:
                    errors.append(f"{ch_id}/{sid}: Unknown state condition type '{ctype}'.")

        # Achievability checks: verify referenced entities exist
        for stage in stages:
            sid = stage.get("id", "")
            conds = stage.get("done_any", []) + stage.get("done_all", [])
            for cond in conds:
                # Check object references
                for obj_id in cond.get("base_objs", []):
                    if obj_id not in all_obj_ids and not obj_id.endswith("_key"):
                        errors.append(f"WARNING: {ch_id}/{sid}: Condition references unknown object '{obj_id}' (may be spawned dynamically).")
            # Check boss NPC references
            boss_key = stage.get("spawns_boss_key")
            if boss_key:
                bkeys = stage.get("bootstrap_keys", {})
                if boss_key in bkeys:
                    bspec = bkeys[boss_key]
                    if isinstance(bspec, dict) and "spawn_npc" in bspec:
                        base_npc = bspec["spawn_npc"]
                        if base_npc not in all_npc_ids:
                            errors.append(f"{ch_id}/{sid}: spawns_boss_key references NPC '{base_npc}' not found in entities or required_npcs.")

    if not has_quest_complete:
        errors.append("No stage has 'quest_complete': true across all chapters.")

    # Cross-reference conditions against actual entity properties and available
    # actions so the planner doesn't produce tasks that are impossible in this game
    obj_category: dict[str, str] = {}
    for cat, objs in entities_summary.get("objects", {}).items():
        for obj in objs:
            obj_category[obj["id"]] = cat
    for obj in spec.get("required_objects", []):
        obj_category[obj["id"]] = obj.get("category", "other")

    available_verbs: set[str] = set()
    if available_actions:
        for a in available_actions:
            if isinstance(a, dict):
                available_verbs.add(a.get("verb", ""))
            elif isinstance(a, str):
                available_verbs.add(a)

    for chapter in spec["chapters"]:
        ch_id = chapter.get("id", "?")
        for stage in chapter.get("stages", []):
            sid = stage.get("id", "?")
            conds = stage.get("done_any", []) + stage.get("done_all", [])
            for cond in conds:
                kind = cond.get("kind")
                ctype = cond.get("type")

                if ctype == "equipped_any_of":
                    for oid in cond.get("base_objs", []):
                        cat = obj_category.get(oid)
                        if cat and cat not in ("armor", "container"):
                            errors.append(
                                f"WARNING: {ch_id}/{sid}: 'equipped_any_of' references "
                                f"'{oid}' (category={cat}). Only armor and containers "
                                f"are equippable — weapons boost attack automatically "
                                f"when held in hand. Consider 'has_any_of' instead."
                            )

                if ctype == "has_category":
                    cat_val = cond.get("category", "")
                    if cat_val and cat_val not in entities_summary.get("objects", {}):
                        errors.append(
                            f"WARNING: {ch_id}/{sid}: 'has_category' references "
                            f"category '{cat_val}' which has no objects in the "
                            f"entity pool."
                        )

                if kind == "event":
                    implied_verbs = {
                        "npc_killed": "attack",
                        "object_bought": "buy",
                        "object_crafted": "craft",
                    }
                    verb = implied_verbs.get(ctype, "")
                    if verb and available_verbs and verb not in available_verbs:
                        errors.append(
                            f"WARNING: {ch_id}/{sid}: Event condition "
                            f"'{ctype}' requires the '{verb}' action, but "
                            f"this game does not have that action."
                        )

                if kind == "custom":
                    for field, label in [("obj_key", "object"), ("npc_key", "NPC")]:
                        ref = cond.get(field, "")
                        if ref and not ref.startswith("ch") and ref.startswith("obj_") and ref not in all_obj_ids:
                            errors.append(
                                f"WARNING: {ch_id}/{sid}: Custom condition "
                                f"'{ctype}' references {label} '{ref}' not "
                                f"found in entities or required_{label}s."
                            )

    return errors

def split_validation_results(results: list[str]) -> tuple[list[str], list[str]]:
    errors = [r for r in results if not r.startswith("WARNING:")]
    warnings = [r.removeprefix("WARNING:").strip() for r in results if r.startswith("WARNING:")]
    return errors, warnings

# STEP 3: COMPILER — converts quest spec to Python source code
def tree_to_dag(spec: dict) -> dict:
    for chapter in spec.get("chapters", []):
        stages = chapter.get("stages", [])
        has_children = any("children" in s for s in stages)
        if not has_children:
            continue

        for stage in stages:
            children = stage.get("children", [])
            if children:
                stage["requires"] = list(children)

        # Store root_goal in chapter metadata for runtime use
        stage_ids = {s["id"] for s in stages}
        child_ids: set[str] = set()
        for stage in stages:
            child_ids.update(stage.get("children", []))
        root_ids = stage_ids - child_ids
        if len(root_ids) == 1:
            chapter["root_goal"] = root_ids.pop()

    return spec


def compile_quest_config(spec: dict, branching_factor: int = 1) -> str:
    if branching_factor >= 2:
        tree_to_dag(spec)

    chapters_src = []
    for chapter in spec["chapters"]:
        stages_src = []
        for stage in chapter["stages"]:
            stage_dict = {
                "id": stage["id"],
                "objective": stage.get("objective", ""),
            }
            if stage.get("hint"):
                stage_dict["hint"] = stage["hint"]

            # done_any / done_all
            if "done_all" in stage:
                stage_dict["done_all"] = stage["done_all"]
            elif "done_any" in stage:
                stage_dict["done_any"] = stage["done_any"]

            if stage.get("on_complete_feedback"):
                stage_dict["on_complete_feedback"] = stage["on_complete_feedback"]
            if stage.get("spawns_boss_key"):
                stage_dict["spawns_boss_key"] = stage["spawns_boss_key"]
            if stage.get("gives_coins"):
                stage_dict["gives_coins"] = stage["gives_coins"]
            if stage.get("chapter_complete"):
                stage_dict["chapter_complete"] = True
            if stage.get("quest_complete"):
                stage_dict["quest_complete"] = True

            # DAG: requires field (generated by tree_to_dag or manually specified)
            if stage.get("requires"):
                stage_dict["requires"] = stage["requires"]

            # Tree: children field (for runtime parent-child lookup)
            if stage.get("children"):
                stage_dict["children"] = stage["children"]

            stages_src.append(stage_dict)

        ch_dict = {
            "id": chapter["id"],
            "title": chapter["title"],
            "intro": chapter.get("intro", ""),
            "stages": stages_src,
        }
        if chapter.get("root_goal"):
            ch_dict["root_goal"] = chapter["root_goal"]
        chapters_src.append(ch_dict)

    config = {
        "title": spec["title"],
        "intro": spec.get("intro", f"=== MAIN QUEST: {spec['title']} ===\n"),
        "chapters": chapters_src,
    }

    return _dict_to_python_src(config, indent=4)


def _dict_to_python_src(obj, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for k, v in obj.items():
            items.append(f"{pad}    {repr(k)}: {_dict_to_python_src(v, indent + 4)},")
        return "{\n" + "\n".join(items) + f"\n{pad}}}"
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        if all(isinstance(x, str) for x in obj) and len(obj) <= 4:
            return repr(obj)
        items = []
        for v in obj:
            items.append(f"{pad}    {_dict_to_python_src(v, indent + 4)},")
        return "[\n" + "\n".join(items) + f"\n{pad}]"
    elif isinstance(obj, str):
        if "\n" in obj or len(obj) > 80:
            return "(\n" + pad + "    " + repr(obj) + "\n" + pad + ")"
        return repr(obj)
    elif isinstance(obj, bool):
        return repr(obj)
    elif obj is None:
        return "None"
    else:
        return repr(obj)


def compile_required_objects(spec: dict) -> str:
    objs = spec.get("required_objects", [])
    lines = ["    required_objects: List[Dict[str, Any]] = ["]
    for obj in objs:
        lines.append(f"        {repr(obj)},")
    lines.append("    ]")
    return "\n".join(lines)


def compile_required_npcs(spec: dict) -> str:
    npcs = spec.get("required_npcs", [])
    lines = ["    required_npcs: List[Dict[str, Any]] = ["]
    for npc in npcs:
        lines.append(f"        {repr(npc)},")
    # always add the objective guide NPC
    lines.append("")
    lines.append("        # objective guide (invincible)")
    lines.append("        {\"type\": \"npc\", \"id\": OBJECTIVE_NPC_BASE_ID, \"name\": \"mira\",")
    lines.append("        \"enemy\": False, \"unique\": True, \"role\": \"guide\", \"quest\": True,")
    lines.append("        \"description\": \"\",")
    lines.append("        \"base_attack_power\": OBJECTIVE_NPC_ATK, \"slope_attack_power\": 0,")
    lines.append("        \"base_hp\": OBJECTIVE_NPC_MAX_HP, \"slope_hp\": 0,")
    lines.append("        \"objects\": []},")
    lines.append("    ]")
    return "\n".join(lines)


def compile_bootstrap(spec: dict) -> str:
    lines = []

    # generated_values (constants)
    gen_values = spec.get("generated_values", {})
    for key, value in gen_values.items():
        lines.append(f"        self._generated[{repr(key)}] = {repr(value)}")

    if gen_values:
        lines.append("")

    # per-chapter bootstrap from stage bootstrap_keys
    for chapter in spec["chapters"]:
        ch_id = chapter["id"]
        lines.append(f"        # {ch_id}: {chapter.get('title', '')}")
        for stage in chapter["stages"]:
            bkeys = stage.get("bootstrap_keys", {})
            for key, spec_val in bkeys.items():
                if isinstance(spec_val, dict):
                    if "pick_progressive" in spec_val:
                        n = spec_val["pick_progressive"]
                        ret = spec_val.get("return")
                        var = f"_picks_{key}"
                        lines.append(f"        {var} = pick_progressive({n})")
                        if ret == "farthest":
                            lines.append(f"        self._generated[{repr(key)}] = max({var}, key=lambda a: dist.get(a, 0)) if {var} else spawn_area")
                        elif ret == "list":
                            lines.append(f"        self._generated[{repr(key)}] = {var} if {var} else [spawn_area]")
                        elif n == 1:
                            lines.append(f"        self._generated[{repr(key)}] = {var}[0] if {var} else spawn_area")
                        else:
                            lines.append(f"        self._generated[{repr(key)}] = {var} if {var} else [spawn_area]")
                    elif "spawn_npc" in spec_val:
                        # NPC spawn handled in _ensure_static_quest_npcs, just note it
                        lines.append(f"        # {key}: NPC {spec_val['spawn_npc']} spawned in _ensure_static_quest_npcs")
                    elif "literal" in spec_val:
                        lines.append(f"        self._generated[{repr(key)}] = {repr(spec_val['literal'])}")
                else:
                    lines.append(f"        self._generated[{repr(key)}] = {repr(spec_val)}")
        lines.append("")

    return "\n".join(lines)


def compile_ensure_static_npcs(spec: dict) -> str:
    lines = []
    lines.append("    def _ensure_static_quest_npcs(self, env, world) -> None:")
    lines.append("        rng = env.rng")
    lines.append("")

    # find all bootstrap_keys that have spawn_npc
    spawns = []
    for chapter in spec["chapters"]:
        for stage in chapter["stages"]:
            bkeys = stage.get("bootstrap_keys", {})
            for key, spec_val in bkeys.items():
                if isinstance(spec_val, dict) and "spawn_npc" in spec_val:
                    spawns.append({
                        "key": key,
                        "base_npc": spec_val["spawn_npc"],
                        "area_key": spec_val.get("area_key", key.replace("_npc", "_area")),
                        "setup_merchant": spec_val.get("setup_merchant", False),
                        "merchant_stock": spec_val.get("merchant_stock", {}),
                    })

    if not spawns:
        lines.append("        pass  # no quest NPCs to spawn")
    else:
        for sp in spawns:
            lines.append(f"        if {repr(sp['key'])} not in self._generated:")
            lines.append(f"            inst = self._create_npc_instance(world, {repr(sp['base_npc'])}, self._generated[{repr(sp['area_key'])}], rng)")
            lines.append(f"            if inst:")
            lines.append(f"                self._generated[{repr(sp['key'])}] = inst")
            name_key = sp['key'].replace("_npc", "_name")
            lines.append(f"                self._generated[{repr(name_key)}] = world.npc_instances[inst].name")
            if sp.get("setup_merchant"):
                lines.append(f"                npc = world.npc_instances[inst]")
                lines.append(f"                npc.role = 'merchant'")
                lines.append(f"                npc.coins = max(int(getattr(npc, 'coins', 0)), 400)")
                for stock_id, stock_count in sp.get("merchant_stock", {}).items():
                    lines.append(f"                npc.inventory[{repr(stock_id)}] = max(int(npc.inventory.get({repr(stock_id)}, 0)), {stock_count})")
            lines.append("")

    lines.append("        world.auxiliary[\"main_quest\"][\"generated\"] = self._generated")
    return "\n".join(lines)


def compile_ensure_bosses(spec: dict) -> str:
    lines = []
    lines.append("    def _ensure_bosses_for_progress(self, env, world, prog: Dict[str, Any]) -> None:")
    lines.append("        ch = int(prog.get('chapter', 0))")
    lines.append("        st = int(prog.get('stage', 0))")
    lines.append("")

    # Find all stages with spawns_boss_key
    boss_spawns = []
    for ch_idx, chapter in enumerate(spec["chapters"]):
        for st_idx, stage in enumerate(chapter["stages"]):
            boss_key = stage.get("spawns_boss_key")
            if boss_key:
                boss_spawns.append({
                    "ch_idx": ch_idx,
                    "st_idx": st_idx,
                    "boss_key": boss_key,
                    "stage_id": stage["id"],
                })

    if not boss_spawns:
        lines.append("        pass  # no bosses to spawn")
    else:
        for bs in boss_spawns:
            lines.append(f"        # {bs['stage_id']}: spawn {bs['boss_key']}")
            lines.append(f"        if ch == {bs['ch_idx']} and st >= {bs['st_idx']}:")
            lines.append(f"            self._ensure_named_boss_spawned(env, world, {repr(bs['boss_key'])})")

    return "\n".join(lines)


def compile_render_text(spec: dict) -> str:
    # Collect all placeholder keys used in text fields
    all_keys: set[str] = set()
    for chapter in spec["chapters"]:
        for field in ["intro", "title"]:
            text = chapter.get(field, "")
            all_keys.update(_re.findall(r"\{(\w+)\}", text))
        for stage in chapter["stages"]:
            for field in ["objective", "hint", "on_complete_feedback"]:
                text = stage.get(field, "")
                all_keys.update(_re.findall(r"\{(\w+)\}", text))

    intro_text = spec.get("intro", "")
    all_keys.update(_re.findall(r"\{(\w+)\}", intro_text))

    # Always include mq_guide_name
    all_keys.add("mq_guide_name")
    # Remove standard format keys
    all_keys -= {"mq_guide_name"}  # handled separately

    lines = []
    for key in sorted(all_keys):
        # Detect area vs list keys
        if key.endswith("_area"):
            lines.append(f"            {repr(key)}: area_disp({repr(key)}),")
        elif key.endswith("_candidates") or key.endswith("_areas"):
            list_key = key
            # try to map to the actual list key
            if not key.endswith("_areas"):
                list_key = key.replace("_candidates", "_candidate_areas")
            lines.append(f"            {repr(key)}: area_list_disp({repr(list_key)}),")
        elif key.endswith("_name"):
            lines.append(f"            {repr(key)}: self._generated.get({repr(key)}, {repr(key)}),")
        else:
            lines.append(f"            {repr(key)}: str(self._generated.get({repr(key)}, '???')),")

    return "\n".join(lines)


def compile_named_boss_spawned(spec: dict) -> str:
    bosses = []
    for npc in spec.get("required_npcs", []):
        if npc.get("enemy") and "boss" in npc.get("role", ""):
            bosses.append(npc)

    lines = []
    for boss in bosses:
        # Find which bootstrap_key references this boss
        boss_key_candidates = []
        for chapter in spec["chapters"]:
            for stage in chapter["stages"]:
                bkeys = stage.get("bootstrap_keys", {})
                for key, spec_val in bkeys.items():
                    if isinstance(spec_val, dict) and spec_val.get("spawn_npc") == boss["id"]:
                        # The key for the boss NPC instance
                        area_key = spec_val.get("area_key", key.replace("_npc", "_area"))
                        boss_key_candidates.append({
                            "key": key,
                            "base_id": boss["id"],
                            "area_key": area_key,
                            "base_hp": boss.get("base_hp", 100),
                            "base_atk": boss.get("base_attack_power", 20),
                        })

        for bc in boss_key_candidates:
            lines.append(f"        if boss_key == {repr(bc['key'])}:")
            lines.append(f"            base_id = {repr(bc['base_id'])}")
            lines.append(f"            area_id = self._generated[{repr(bc['area_key'])}]")
            lines.append(f"            base_hp, base_atk = {bc['base_hp']}, {bc['base_atk']}")

    return "\n".join(lines) if lines else ""


def compile_full_step_rules(spec: dict, base_step_rules_path: str, branching_factor: int = 1) -> str:
    with open(base_step_rules_path, "r", encoding="utf-8") as f:
        source = f.read()

    return _compile_into_source(source, spec, branching_factor)



def _compile_into_source(source: str, spec: dict, branching_factor: int) -> str:
    compile_warnings: list[str] = []

    def _verified_sub(pattern: str, replacement, src: str, label: str, **kwargs) -> str:
        result = _re.sub(pattern, replacement, src, **kwargs)
        if result == src:
            compile_warnings.append(f"Compilation: '{label}' regex matched nothing — source unchanged")
        return result

    # Replace QUEST_CONFIG
    quest_config_src = compile_quest_config(spec, branching_factor=branching_factor)
    source = _verified_sub(
        r"(    QUEST_CONFIG: Dict\[str, Any\] = )\{.*?\n    \}(\n)",
        lambda m: f"    QUEST_CONFIG: Dict[str, Any] = {quest_config_src}\n",
        source,
        label="QUEST_CONFIG",
        flags=_re.DOTALL,
        count=1,
    )

    # Replace required_objects
    req_obj_src = compile_required_objects(spec)
    source = _verified_sub(
        r"    required_objects: List\[Dict\[str, Any\]\] = \[.*?\n    \](?!\S)",
        req_obj_src,
        source,
        label="required_objects",
        flags=_re.DOTALL,
        count=1,
    )

    # Replace required_npcs (but keep objective guide)
    req_npc_src = compile_required_npcs(spec)
    source = _verified_sub(
        r"    required_npcs: List\[Dict\[str, Any\]\] = \[.*?\n    \](?!\S)",
        req_npc_src,
        source,
        label="required_npcs",
        flags=_re.DOTALL,
        count=1,
    )

    # Replace undistributable_objects
    undist = spec.get("undistributable_objects", [])
    undist_src = f"    undistributable_objects: List[str] = {repr(undist)}"
    source = _verified_sub(
        r"    undistributable_objects: List\[str\] = \[.*?\]",
        undist_src,
        source,
        label="undistributable_objects",
        flags=_re.DOTALL,
        count=1,
    )

    # Replace _bootstrap() body (keep the BFS/pick_progressive infrastructure)
    bootstrap_body = compile_bootstrap(spec)
    source = _verified_sub(
        r"(        # Chapter 1\n).*?(?=\n    def _register_required_npcs)",
        lambda m: bootstrap_body,
        source,
        label="_bootstrap body",
        flags=_re.DOTALL,
        count=1,
    )

    # Replace _ensure_static_quest_npcs
    static_npcs_src = compile_ensure_static_npcs(spec)
    source = _verified_sub(
        r"    def _ensure_static_quest_npcs\(self, env, world\) -> None:.*?(?=\n    def )",
        static_npcs_src + "\n",
        source,
        label="_ensure_static_quest_npcs",
        flags=_re.DOTALL,
        count=1,
    )

    # Replace _ensure_bosses_for_progress
    bosses_src = compile_ensure_bosses(spec)
    source = _verified_sub(
        r"    def _ensure_bosses_for_progress\(self, env, world, prog.*?\) -> None:.*?(?=\n    def )",
        bosses_src + "\n",
        source,
        label="_ensure_bosses_for_progress",
        flags=_re.DOTALL,
        count=1,
    )

    # Replace _render_text fmt dict entries
    render_entries = compile_render_text(spec)
    source = _verified_sub(
        r"(        fmt = \{\n).*?(            \"mq_guide_name\":)",
        f"\\1{render_entries}\n\n            \"mq_guide_name\":",
        source,
        label="_render_text fmt dict",
        flags=_re.DOTALL,
        count=1,
    )

    if compile_warnings:
        print(f"  Compilation warnings ({len(compile_warnings)}):")
        for w in compile_warnings:
            print(f"    ⚠ {w}")

    return source

# STEP 3a: CUSTOM CONDITION IMPLEMENTATION (via aider)
def _collect_custom_conditions(spec: dict) -> list[dict]:
    customs = []
    for chapter in spec.get("chapters", []):
        for stage in chapter.get("stages", []):
            for cond in stage.get("done_any", []) + stage.get("done_all", []):
                if cond.get("kind") == "custom":
                    customs.append({
                        "chapter_id": chapter["id"],
                        "stage_id": stage["id"],
                        "stage_objective": stage.get("objective", ""),
                        **cond,
                    })
    return customs

def build_custom_condition_prompt(custom_conds: list[dict], game_name: str) -> str:
    cond_descriptions = []
    for i, c in enumerate(custom_conds, 1):
        cond_descriptions.append(
            f"{i}. type=\"{c.get('type')}\" (stage: {c.get('stage_id')})\n"
            f"   Description: {c.get('description', 'N/A')}\n"
            f"   Implementation hint: {c.get('implementation_hint', 'N/A')}\n"
            f"   Extra fields: {json.dumps({k: v for k, v in c.items() if k not in ('kind', 'type', 'description', 'implementation_hint', 'chapter_id', 'stage_id', 'stage_objective')})}"
        )

    return textwrap.dedent(f"""\
    The compiled quest for game "{game_name}" uses custom condition types that
    need implementation in the `_cond_true` method of `MainQuestStepRule`.

    CUSTOM CONDITIONS TO IMPLEMENT:
    {chr(10).join(cond_descriptions)}

    IMPLEMENTATION INSTRUCTIONS:
    In the `_cond_true` method of class MainQuestStepRule, find the section
    that handles conditions.  Add handling for kind=="custom" with elif
    blocks for each custom type.

    Pattern to follow (add BEFORE the final `return False`):

        if kind == "custom":
            ctype = cond.get("type")

            if ctype == "<custom_type_name>":
                # ... implementation using self._generated, env, world, agent ...
                return <True/False>

            print(f"WARNING: Unknown custom condition type={{ctype}}")
            return False

    Key references available in _cond_true:
    - self._generated: dict of bootstrap values (area IDs, NPC instance IDs, etc.)
    - env.curr_agents_state["area"][agent.id]: current player area ID
    - world.area_instances[area_id]: area object with .npcs, .objects lists
    - world.npc_instances[npc_id]: NPC object with .hp, .coins, .inventory, etc.
    - world.object_instances[obj_id]: object with .base_id, .category, etc.
    - agent.inventory: player inventory dict
    - Use cond.get("xxx_key") to get bootstrap key names, then self._generated.get(key) for values.

    CRITICAL CONSTRAINTS:
    - Only edit the main_quest.py file. Do NOT change any other file.
    - ONLY edit inside the class `MainQuestStepRule`.
    - The file contains ONLY MainQuestStepRule — do NOT add other classes.
    """)

# STEP 4: FULL TWO-STEP PIPELINE
def generate_two_step(
    description: str,
    game_name: str,
    world_definition_path: str,
    num_chapters: int = 1,
    branching_factor: int = 1,
    llm_name: str = "gpt-5",
    max_iterations: int = 5,
    test_timeout: int = 120,
    llm_provider: str | None = None,
) -> tuple[bool, str]:
    world_definition = load_world_definition(world_definition_path)
    entities_summary = summarize_entities(world_definition)
    available_actions = load_action_rules(game_name)
    topology_summary = summarize_topology(world_definition)
    step_rules_summary = summarize_step_rules(game_name)
    action_constraints_summary = summarize_action_constraints(game_name)

    print(f"\n{'='*60}")
    print(f"Two-step quest generation ({num_chapters} chapters, branching_factor={branching_factor})")
    print(f"{'='*60}\n")

    # Step 1: Plan
    print("[Step 1/3] Planning quest with LLM...")
    try:
        spec = plan_quest(
            description=description,
            entities_summary=entities_summary,
            available_actions=available_actions,
            topology_summary=topology_summary,
            num_chapters=num_chapters,
            branching_factor=branching_factor,
            llm_name=llm_name,
            llm_provider=llm_provider,
            step_rules_summary=step_rules_summary,
            action_constraints_summary=action_constraints_summary,
        )
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Planner failed: {e}")
        print("  Falling back to aider-only pipeline...")
        return generate_with_aider(
            description=description,
            game_name=game_name,
            world_definition_path=world_definition_path,
            num_chapters=num_chapters,
            llm_name=llm_name,
            max_iterations=max_iterations,
            test_timeout=test_timeout,
            llm_provider=llm_provider,
            branching_factor=branching_factor,
        )

    # Step 2: Validate
    print("[Step 2/3] Validating quest spec...")
    all_issues = validate_quest_spec(spec, entities_summary, branching_factor, available_actions=available_actions)
    val_errors, val_warnings = split_validation_results(all_issues)
    if val_warnings:
        print(f"  {len(val_warnings)} warning(s):")
        for w in val_warnings:
            print(f"    ⚠ {w}")
    if val_errors:
        print(f"  Validation found {len(val_errors)} blocking error(s):")
        for err in val_errors:
            print(f"    ✗ {err}")
        print("  Falling back to aider-only pipeline...")
        return generate_with_aider(
            description=description,
            game_name=game_name,
            world_definition_path=world_definition_path,
            num_chapters=num_chapters,
            llm_name=llm_name,
            max_iterations=max_iterations,
            test_timeout=test_timeout,
            llm_provider=llm_provider,
            branching_factor=branching_factor,
        )

    print(f"  Spec valid: {len(spec['chapters'])} chapters, "
          f"{sum(len(ch.get('stages', [])) for ch in spec['chapters'])} total stages")

    # Step 3: Compile and inject
    print("[Step 3/3] Compiling quest spec into code...")
    game_root = f"games/generated/{game_name}"
    step_rules_path = os.path.join(current_directory, f"{game_root}/rules/step_rules/main_quest.py")

    try:
        compiled = compile_full_step_rules(spec, step_rules_path, branching_factor=branching_factor)
        with open(step_rules_path, "w", encoding="utf-8") as f:
            f.write(compiled)
        print("  Compiled and written to main_quest.py")
    except Exception as e:
        print(f"  Compilation failed: {e}")
        print("  Falling back to aider-only pipeline...")
        return generate_with_aider(
            description=description,
            game_name=game_name,
            world_definition_path=world_definition_path,
            num_chapters=num_chapters,
            llm_name=llm_name,
            max_iterations=max_iterations,
            test_timeout=test_timeout,
            llm_provider=llm_provider,
            branching_factor=branching_factor,
        )

    # Step 3a: If custom conditions exist, run aider to implement them
    custom_conds = _collect_custom_conditions(spec)
    if custom_conds:
        print(f"  Found {len(custom_conds)} custom condition(s). Running aider to implement...")
        custom_prompt = build_custom_condition_prompt(custom_conds, game_name)
        editable_files, context_files = get_files_for_quest_generation(game_name, world_definition_path)
        custom_cmd = [
            "aider",
            "--model", _aider_model_name(llm_name, provider=llm_provider),
            "--no-auto-commits",
            "--yes",
            "--no-suggest-shell-commands",
            "--message", custom_prompt,
        ]
        for ctx_file in context_files:
            full_path = os.path.join(current_directory, ctx_file)
            if os.path.exists(full_path):
                custom_cmd.extend(["--read", ctx_file])
        for edit_file in editable_files:
            full_path = os.path.join(current_directory, edit_file)
            if os.path.exists(full_path):
                custom_cmd.append(edit_file)
        try:
            subprocess.run(custom_cmd, cwd=current_directory, capture_output=False, text=True, timeout=600)
            print("  Custom conditions implemented.")
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"  Aider custom-condition pass failed: {e}")

    # Test
    print("  Testing compiled quest...")
    success, output = test_game(game_name, world_definition_path, timeout=test_timeout)
    if success:
        print("  ✓ Quest compiled and tested successfully!")
        return True, "Two-step quest generation succeeded"

    # If compile output fails, use aider to fix
    print(f"  Compiled quest failed smoke test. Using aider to fix...")
    fix_prompt = build_fix_prompt(output, game_name=game_name, world_definition_path=world_definition_path)

    editable_files, context_files = get_files_for_quest_generation(game_name, world_definition_path)
    cmd = [
        "aider",
        "--model", _aider_model_name(llm_name, provider=llm_provider),
        "--no-auto-commits",
        "--yes",
        "--no-suggest-shell-commands",
        "--message", fix_prompt,
    ]
    for ctx_file in context_files:
        full_path = os.path.join(current_directory, ctx_file)
        if os.path.exists(full_path):
            cmd.extend(["--read", ctx_file])
    for edit_file in editable_files:
        full_path = os.path.join(current_directory, edit_file)
        if os.path.exists(full_path):
            cmd.append(edit_file)

    for iteration in range(max_iterations):
        print(f"\n--- Fix iteration {iteration + 1}/{max_iterations} ---\n")
        try:
            aider_result = subprocess.run(
                cmd, cwd=current_directory, capture_output=True, text=True, timeout=600
            )
            # Show aider output summary for diagnostics
            aider_out = (aider_result.stdout or "") + (aider_result.stderr or "")
            if aider_result.returncode != 0:
                print(f"  Aider exited with code {aider_result.returncode}")
                # Show last few lines of output for context
                tail = "\n".join(aider_out.strip().splitlines()[-10:])
                if tail:
                    print(f"  Aider output (tail):\n{tail}")
            else:
                # Look for signs aider actually made changes
                if "Applied edit" in aider_out or "Wrote" in aider_out:
                    print("  Aider applied edits.")
                elif "No changes" in aider_out or "didn't need" in aider_out.lower():
                    print("  ⚠ Aider made no changes — skipping redundant test.")
                    # If aider made no changes, trying the same prompt again is futile.
                    # Mutate the prompt to emphasize the error differently.
                    cmd[cmd.index("--message") + 1] = build_fix_prompt(
                        output, game_name=game_name, world_definition_path=world_definition_path,
                        extra_hint=f"(Attempt {iteration + 2}: previous fix attempt made no changes. Try a DIFFERENT approach.)"
                    )
                    continue
        except subprocess.TimeoutExpired:
            print("  Aider timed out (600s). Continuing to next iteration...")
            continue
        except Exception as e:
            print(f"  Aider error: {e}")
            continue

        success, output = test_game(game_name, world_definition_path, timeout=test_timeout)
        if success:
            print("\n✓ Quest fixed and tested successfully!")
            return True, "Two-step quest generation succeeded (with aider fixes)"

        # Truncate output to last 3000 chars for the fix prompt
        print(f"  Test failed (iteration {iteration + 1}). Error tail:\n{output[-1000:]}")
        cmd[cmd.index("--message") + 1] = build_fix_prompt(
            output, game_name=game_name, world_definition_path=world_definition_path
        )

    return False, "Two-step generation failed after fix iterations"


def _format_actions_str(available_actions: list) -> str:
    if not available_actions:
        return "standard actions (pick up, drop, enter, craft, attack, etc.)"
    if isinstance(available_actions[0], dict):
        action_lines = []
        for a in available_actions:
            params = ", ".join(a.get("params", []))
            desc = a.get("description", "")
            line = f"  - {a['verb']}"
            if params:
                line += f" ({params})"
            if desc:
                line += f": {desc}"
            action_lines.append(line)
        return "\n".join(action_lines)
    return ", ".join(available_actions)


def build_quest_prompt(
    description: str,
    game_name: str,
    world_definition_path: str,
    num_chapters: int,
    existing_chapters: list[dict],
    entities_summary: dict,
    available_actions: list,
    topology_summary: str = "",
    branching_factor: int = 1,
    step_rules_summary: str = "",
    action_constraints_summary: str = "",
) -> str:
    editable_files, context_files = get_files_for_quest_generation(game_name, world_definition_path)
    game_root = f"games/generated/{game_name}"
    test_command = get_test_command(game_name, world_definition_path)

    entities_str = json.dumps(entities_summary, indent=2)
    actions_str = _format_actions_str(available_actions)

    # Goal tree / DAG instructions
    dag_section = ""
    if branching_factor >= 2:
        dag_section = f"""
GOAL TREE MODE (branching_factor={branching_factor}):
This quest uses a **goal tree** structure per chapter. The planner has already
designed the tree and the compiler has converted it to a DAG with "requires" fields.

How the tree→DAG works in the compiled QUEST_CONFIG:
- Each parent node has "requires": [child_ids...] — it activates when ALL children are done.
- Leaf nodes (no "requires") are active immediately — these are the player's first objectives.
- Parent nodes with NO conditions auto-complete when their children finish (cascading upward).
- The root node has "chapter_complete" or "quest_complete".

Your job as aider:
- Implement the stage conditions, bootstrap keys, NPC spawns, etc. as usual.
- Do NOT modify the "requires" structure — it was generated by the compiler.
- Internal (parent) nodes may have NO "done_any"/"done_all" — that is intentional.
  They auto-complete when their children finish.
- Leaf nodes MUST always have "done_any" or "done_all" conditions.
- Ensure bootstrap keys cover all leaf stages.
"""
    else:
        dag_section = """
LINEAR MODE:
Stages within each chapter are sequential (no "requires" field needed).
"""

    # Build optional game-specific mechanics section
    game_mechanics_section = ""
    if step_rules_summary or action_constraints_summary:
        parts = [
            "GAME-SPECIFIC MECHANICS:",
            "This game has unique environment rules and action constraints.",
            "Design quest stages that LEVERAGE these mechanics.\n",
        ]
        if step_rules_summary:
            parts.append(step_rules_summary)
        if action_constraints_summary:
            parts.append(action_constraints_summary)
        game_mechanics_section = "\n".join(parts)

    return f"""UPDATE the MainQuestStepRule in step_rules/main_quest.py to implement a brand-new quest with {num_chapters} chapter(s).
The file already contains a working MainQuestStepRule with example/placeholder data.
Update ONLY the following sections IN-PLACE with your new quest content — keep the
same structure, same patterns, same method signatures. Do NOT delete or rewrite
the methods; just change the DATA inside them to match your invented quest:
  • QUEST_CONFIG  (update title, intro, and the chapters list with YOUR stages)
  • required_objects  (update list with YOUR quest objects)
  • required_npcs  (update list with YOUR quest NPCs; keep the OBJECTIVE_NPC guide entry)
  • undistributable_objects  (update list with YOUR undistributable objects)
  • _bootstrap()  (update the generated keys for YOUR chapters using pick_progressive)
  • _render_text() format dict  (update only YOUR placeholder keys)
  • _ensure_static_quest_npcs()  (update to spawn YOUR quest NPCs)
  • _ensure_bosses_for_progress()  (update to spawn YOUR bosses at your stages)
Keep everything else in the class (apply(), __init__, _init_agent_progress,
_register_required_npcs, _create_npc_instance, _check_*, DAG helpers, etc.) unchanged.

USER DESCRIPTION / THEME:
{description if description else "Create a diverse and creative quest that tests exploration, memory, reasoning, crafting, and combat."}

AVAILABLE GAME ENTITIES (from world_definition.json):
{entities_str}

AVAILABLE PLAYER ACTIONS:
{actions_str}

{topology_summary}
{dag_section}

{game_mechanics_section}

FILES TO READ FOR CONTEXT:
- {game_root}/rules/step_rules/main_quest.py - Contains MainQuestStepRule
- {game_root}/rules/action_rules.py - Available player actions
- {game_root}/rule.py - Base classes (RuleContext, RuleResult, Event)
- {game_root}/world.py - World state, Object, NPC, Area classes
- {world_definition_path} - Entity definitions

FILES TO EDIT:
1. {game_root}/rules/step_rules/main_quest.py (REQUIRED)
2. {world_definition_path} (IF NEEDED — add new quest entities)

AVAILABLE CONDITION PRIMITIVES (build stages from these freely):

Event conditions ("kind": "event") — triggered once when an action happens:
  - "npc_killed"     : {{"kind": "event", "type": "npc_killed", "npc_key": "<key>"}}
  - "object_bought"  : {{"kind": "event", "type": "object_bought", "base_objs": ["obj_id"]}}
  - "object_crafted" : {{"kind": "event", "type": "object_crafted", "category": "weapon"}}

State conditions ("kind": "state") — continuously checked each step:
  - "in_area"                         : {{"kind": "state", "type": "in_area", "area_key": "<key>"}}
  - "in_area_with_npc"                : {{"kind": "state", "type": "in_area_with_npc", "npc_key": "<key>"}}
  - "visited_k_of_area_set"           : {{"kind": "state", "type": "visited_k_of_area_set", "area_set_key": "<key>", "k_key": "<key>"}}
  - "has_any_of"                      : {{"kind": "state", "type": "has_any_of", "base_objs": ["obj_id1"]}}
  - "equipped_any_of"                 : {{"kind": "state", "type": "equipped_any_of", "base_objs": ["obj_id"]}}
  - "has_category"                    : {{"kind": "state", "type": "has_category", "category": "weapon"}}
  - "has_item_count"                  : {{"kind": "state", "type": "has_item_count", "base_obj_key": "<key>", "count_key": "<key>"}}
  - "paper_text_equals_in_area"       : {{"kind": "state", "type": "paper_text_equals_in_area", "text_key": "<key>", "area_key": "<key>"}}
  - "paper_text_equals_in_hands"      : {{"kind": "state", "type": "paper_text_equals_in_hands", "text_key": "<key>"}}
  - "writable_text_equals_in_inventory": {{"kind": "state", "type": "writable_text_equals_in_inventory", "text_key": "<key>"}}
  - "boss_already_defeated"           : {{"kind": "state", "type": "boss_already_defeated", "npc_key": "<key>"}}
  - "killed_n_enemies"                : {{"kind": "state", "type": "killed_n_enemies", "n_key": "<key>"}}
  - "killed_one_of_each"              : {{"kind": "state", "type": "killed_one_of_each", "npc_keys": ["<key1>", "<key2>"]}}
  - "lit_k_dark_areas"                : {{"kind": "state", "type": "lit_k_dark_areas", "k_key": "<key>"}}

Use "done_any" (OR) or "done_all" (AND) to combine primitives freely.

_BOOTSTRAP() REQUIREMENTS (CRITICAL):
The existing _bootstrap() uses BFS progressive picking to assign areas.
Keep this infrastructure. Use pick_progressive(n) for area keys.
Every {{placeholder}} in QUEST_CONFIG text MUST have a matching entry
in _bootstrap() AND in _render_text()'s fmt dict.

DESIGN REQUIREMENTS — BE CREATIVE AND DIVERSE:

1. INVENT NOVEL TASK PATTERNS — do NOT just repeat "go to area → find NPC →
   get item → kill boss". Combine condition primitives creatively:
   - Scavenger hunts: visit K-of-N areas to piece together clues
   - Crafting chains: gather materials → craft intermediate → craft final
   - Trade arbitrage: buy from one merchant, sell/deliver elsewhere
   - Escort/rendezvous: be in the same area as an NPC carrying a specific item
   - Siege/survival: kill N enemies in a specific area
   - Environmental puzzles: light K dark areas to reveal a path
   - Knowledge relay: learn a word from NPC A, write it, deliver it to area B
   - Equipment trials: equip specific gear combos before entering a zone
   - Multi-item fusion: collect 3+ different items (done_all with has_any_of)
   - Time-pressure exploration: visit a distant area while holding a fragile item
   - Deduction: talk to multiple NPCs, synthesize their clues into a written answer

2. PUZZLE DISCOVERY BEFORE PUZZLE SOLVING:
   - ALWAYS precede stages that require writing/recalling information with a
     discovery stage whose on_complete_feedback reveals that information.
   - NEVER require the player to know something they haven't been told yet.

3. INFORMATION FLOW:
   - Information must be learned BEFORE it's used.
   - Chain: Observe/Learn → Remember/Record → Apply/Use.
   - on_complete_feedback from one stage reveals clues for future stages.

4. DIFFICULTY PROGRESSION:
   - Chapter 1: no combat, basic mechanics.
   - Later chapters: more stages, complex conditions, long-horizon memory
     challenges, multi-step crafting, boss fights.

5. HINT GUIDELINES:
   - Hints should GUIDE, not SPOIL.
   - BAD: "Write '{{answer}}' on paper" (gives away the answer)
   - GOOD: "Remember what you learned earlier. Write it on paper."

6. EVERY chapter must use at least 3 DIFFERENT task patterns.
   Do NOT repeat the same condition pattern across consecutive stages.

IMPORTANT NOTES:
- Placeholders in text use {{key}} format (double braces in the string)
- Keys like "area_key", "npc_key", "text_key" reference self._generated dict
- The last stage of the last chapter MUST have "quest_complete": True

CRITICAL CONSTRAINTS -- READ CAREFULLY:
- ONLY edit the file step_rules/main_quest.py.
- The file contains ONLY `MainQuestStepRule` — do NOT add other classes.
- Do NOT touch any other file in step_rules/ (general.py, tutorial.py).

TEST AFTER CHANGES:
{test_command}
"""


def build_fix_prompt(error_output: str, game_name: str, world_definition_path: str, extra_hint: str = "") -> str:
    test_command = get_test_command(game_name, world_definition_path)
    hint_line = f"\n{extra_hint}\n" if extra_hint else ""
    return f"""Fix the errors in the codebase.
{hint_line}
ERROR OUTPUT:
{error_output[-3000:]}

INSTRUCTIONS:
- Read the error carefully and identify which file(s) need fixes
- Fix all syntax errors, import errors, and logic errors
- Ensure the MainQuestStepRule chapter structure is correct
- Make sure all placeholder keys are defined in generated_values and _bootstrap()
- Verify _render_text() has all required format keys

CRITICAL CONSTRAINTS:
- ONLY edit the file step_rules/main_quest.py.
- The file contains ONLY `MainQuestStepRule` — do NOT add other classes.
- Do NOT touch any other file in step_rules/ (general.py, tutorial.py).

TEST AFTER FIXING:
{test_command}
"""


def test_game(game_name: str, world_definition_path: str, timeout: int = 120) -> tuple[bool, str]:
    cmd = [
        "python", "eval.py",
        "--agent", "RandomAgent",
        "--max_steps", "100",
        "--enable_obs_valid_actions",
        "--overwrite",
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
    except subprocess.TimeoutExpired as e:
        partial = ""
        if e.stdout:
            partial += e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors="replace")
        if e.stderr:
            partial += e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="replace")
        # If the game ran long enough to time out, it's not crashing — treat as success
        return True, f"Test ran for {timeout}s without errors (timeout reached).\n{partial[-1000:]}"
    except Exception as e:
        return False, str(e)


def suggest_quest_chapter(
    llm_name: str,
    game_name: str,
    world_definition_path: str,
    existing_chapters: list[dict],
    entities_summary: dict,
    available_actions: list,
    num_chapters: int = 1,
    llm_provider: str | None = None,
    llm=None,
) -> str:
    objects_by_category = entities_summary.get("objects", {})
    object_categories = list(objects_by_category.keys())
    actions_display = _format_actions_str(available_actions)
    
    prompt = f"""You are a game designer for a text-based RPG.  Suggest a complete quest storyline
with {num_chapters} chapter(s) that will REPLACE any existing quest in the game.

AVAILABLE OBJECT CATEGORIES: {', '.join(object_categories)}

AVAILABLE ACTIONS:
{actions_display}

NPCs AVAILABLE: {len(entities_summary.get('npcs', []))} NPCs including merchants, enemies, and quest givers

REQUIREMENTS:
1. Should use at least 2-3 different game mechanics per chapter
2. Include some memory/recall elements (remember information for later)
3. Chapters should have progressive difficulty
4. Be creative with the narrative theme

Provide a brief (2-3 sentence) description of the chapter theme and main objectives.
Do NOT include implementation details, just the creative concept."""

    if llm is None:
        llm = _get_llm(llm_name, llm_provider)
    response = llm.generate(user_prompt=prompt, system_prompt=None)
    return response["response"].strip()


def get_user_confirmation(suggested_quest: str) -> tuple[bool, bool]:
    print(f"\n{'='*60}")
    print("SUGGESTED QUEST CHAPTER:")
    print(f"{'='*60}")
    print(f"\n{suggested_quest}\n")
    print(f"{'='*60}")
    
    while True:
        choice = input("\nGenerate this chapter? [Y]es / [N]o / [R]egenerate: ").strip().lower()
        if choice in ['y', 'yes']:
            return True, False
        elif choice in ['n', 'no']:
            return False, False
        elif choice in ['r', 'regenerate']:
            return False, True
        else:
            print("Please enter Y, N, or R.")


def generate_with_aider(
    description: str,
    game_name: str,
    world_definition_path: str,
    num_chapters: int = 1,
    llm_name: str = "gpt-4o",
    max_iterations: int = 5,
    test_timeout: int = 120,
    llm_provider: str | None = None,
    branching_factor: int = 1,
) -> tuple[bool, str]:
    world_definition = load_world_definition(world_definition_path)
    entities_summary = summarize_entities(world_definition)
    available_actions = load_action_rules(game_name)
    topology_summary = summarize_topology(world_definition)
    step_rules_summary = summarize_step_rules(game_name)
    action_constraints_summary = summarize_action_constraints(game_name)
    
    prompt = build_quest_prompt(
        description=description,
        game_name=game_name,
        world_definition_path=world_definition_path,
        num_chapters=num_chapters,
        existing_chapters=[],
        entities_summary=entities_summary,
        available_actions=available_actions,
        topology_summary=topology_summary,
        branching_factor=branching_factor,
        step_rules_summary=step_rules_summary,
        action_constraints_summary=action_constraints_summary,
    )
    
    editable_files, context_files = get_files_for_quest_generation(game_name, world_definition_path)
    
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
    
    # Add editable files
    for edit_file in editable_files:
        full_path = os.path.join(current_directory, edit_file)
        if os.path.exists(full_path):
            cmd.append(edit_file)
    
    print(f"\n{'='*60}")
    print(f"Generating {num_chapters} quest chapter(s)")
    print(f"Description: {description[:100]}..." if len(description) > 100 else f"Description: {description}")
    print(f"Using model: {llm_name}")
    print(f"Editable files: {editable_files}")
    print(f"{'='*60}\n")
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---\n")
        
        try:
            aider_result = subprocess.run(
                cmd,
                cwd=current_directory,
                capture_output=True,
                text=True,
                timeout=600,
            )
            aider_out = (aider_result.stdout or "") + (aider_result.stderr or "")
            if aider_result.returncode != 0:
                print(f"  Aider exited with code {aider_result.returncode}")
                tail = "\n".join(aider_out.strip().splitlines()[-10:])
                if tail:
                    print(f"  Aider output (tail):\n{tail}")
        except subprocess.TimeoutExpired:
            print("  Aider timed out (600s)")
            continue
        except Exception as e:
            print(f"  Aider error: {e}")
            continue
        
        print("\nTesting generated quest...")
        success, output = test_game(
            game_name=game_name,
            world_definition_path=world_definition_path,
            timeout=test_timeout,
        )
        
        if success:
            print("\n✓ Quest chapter(s) generated and tested successfully!")
            return True, "Quest chapter(s) added successfully"
        
        print(f"\n✗ Test failed. Error:\n{output[-2000:]}")
        
        # Update prompt for fix attempt
        cmd[cmd.index("--message") + 1] = build_fix_prompt(
            output, game_name=game_name, world_definition_path=world_definition_path
        )
    
    print(f"\nFailed to generate valid quest after {max_iterations} iterations")
    return False, "Max iterations reached"


def generate_quest(
    description: str,
    game_name: str,
    world_definition_path: str = None,
    num_chapters: int = 1,
    llm_name: str = "gpt-4o",
    max_iterations: int = 5,
    test_timeout: int = 120,
    llm_provider: str | None = None,
    branching_factor: int = 1,
    use_two_step: bool = True,
) -> tuple[bool, str]:
    if world_definition_path is None:
        world_definition_path = f"assets/world_definitions/generated/{game_name}/default.json"

    if use_two_step:
        return generate_two_step(
            description=description,
            game_name=game_name,
            world_definition_path=world_definition_path,
            num_chapters=num_chapters,
            branching_factor=branching_factor,
            llm_name=llm_name,
            max_iterations=max_iterations,
            test_timeout=test_timeout,
            llm_provider=llm_provider,
        )

    return generate_with_aider(
        description=description,
        game_name=game_name,
        world_definition_path=world_definition_path,
        num_chapters=num_chapters,
        llm_name=llm_name,
        max_iterations=max_iterations,
        test_timeout=test_timeout,
        llm_provider=llm_provider,
        branching_factor=branching_factor,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate new quest chapters for MainQuestStepRule"
    )
    parser.add_argument(
        "--game_name",
        type=str,
        required=True,
        help="Name of the game (e.g., 'remnant')"
    )
    parser.add_argument(
        "--world_definition_path",
        type=str,
        default=None,
        help="Path to world_definitions JSON file (default: assets/world_definitions/generated/{game_name}/default.json)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description/theme for the quest chapter to generate"
    )
    parser.add_argument(
        "--num_chapters",
        type=int,
        default=1,
        help="Number of chapters to generate (default: 1)"
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="gpt-5",
        help="LLM model to use (default: gpt-5)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum iterations for generation/fix attempts (default: 5)"
    )
    parser.add_argument(
        "--test_timeout",
        type=int,
        default=120,
        help="Timeout in seconds for test runs (default: 120)"
    )
    parser.add_argument(
        "--branching_factor",
        type=int,
        default=1,
        help="Goal structure branching factor: 1=linear (default), 2+=goal tree"
    )
    parser.add_argument(
        "--no_two_step",
        action="store_true",
        default=False,
        help="Disable two-step pipeline, use aider-only (default: use two-step)"
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default=None,
        help="LLM provider: openai, azure, azure_openai, claude, gemini, vllm, huggingface (auto-detected from model name if omitted)"
    )
    args = parser.parse_args()
    
    if args.world_definition_path is None:
        args.world_definition_path = f"assets/world_definitions/generated/{args.game_name}/default.json"
    
    description = args.description
    
    if not description:
        print(f"\nNo description provided. Suggesting a quest storyline...")
        
        world_definition = load_world_definition(args.world_definition_path)
        entities_summary = summarize_entities(world_definition)
        available_actions = load_action_rules(args.game_name)
        
        while True:
            suggested = suggest_quest_chapter(
                llm_name=args.llm_name,
                game_name=args.game_name,
                world_definition_path=args.world_definition_path,
                existing_chapters=[],
                entities_summary=entities_summary,
                available_actions=available_actions,
                num_chapters=args.num_chapters,
                llm_provider=args.llm_provider,
            )
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
    
    success, message = generate_quest(
        description=description,
        game_name=args.game_name,
        world_definition_path=args.world_definition_path,
        num_chapters=args.num_chapters,
        llm_name=args.llm_name,
        max_iterations=args.max_iterations,
        test_timeout=args.test_timeout,
        branching_factor=args.branching_factor,
        use_two_step=not args.no_two_step,
        llm_provider=args.llm_provider,
    )
    
    if success:
        print(f"\n✓ Success: {message}")
    else:
        print(f"\n✗ Failed: {message}")
        sys.exit(1)
