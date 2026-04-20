import argparse
import json
import os
import sys
import re
import shutil
import subprocess
import inspect
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from providers.openai_api import OpenAILanguageModel

current_directory = os.getcwd()
sys.path.insert(0, current_directory)

BASE_ENV_CONFIGS_DIR = "assets/env_configs/base"
BASE_WORLD_DEFS_DIR = "assets/world_definitions/base"
BASE_GAME_DIR = "games/base"


# Maps --provider values to aider/litellm provider prefixes
_AIDER_PROVIDER_PREFIX = {
    "openai": "openai",
    "azure": "azure",
    "azure_openai": "azure",
    "claude": "anthropic",
    "gemini": "gemini",
}

def _aider_model_name(model: str, provider: str | None = None) -> str:
    if "/" in model:          # already has provider prefix
        return model
    if provider and provider in _AIDER_PROVIDER_PREFIX:
        return f"{_AIDER_PROVIDER_PREFIX[provider]}/{model}"
    if re.match(r"^(gpt-|o[1-9]|chatgpt-|ft:)", model):
        return f"openai/{model}"
    return model

def _load_module_from_path(module_name: str, file_path: Path):
    # dynamically load a python module from a file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Warning: Could not load module {module_name}: {e}")
        return None

def _extract_accessed_attributes(source_code: str) -> Set[str]:
    # extract attribute accesses from source code
    accessed = set()
    
    attr_pattern = r'\.([a-z_][a-z0-9_]*)'
    for match in re.finditer(attr_pattern, source_code, re.IGNORECASE):
        attr = match.group(1)
        # filter out common method names and non-property attributes
        if attr not in ('get', 'keys', 'values', 'items', 'append', 'extend', 'pop', 
                        'remove', 'copy', 'update', 'setdefault', 'add', 'lower', 
                        'upper', 'strip', 'split', 'join', 'format', 'replace',
                        'startswith', 'endswith', 'find', 'index', 'count', 'sort',
                        'apply', 'id', 'name', 'type', 'description', 'objects', 
                        'npcs', 'areas', 'inventory', 'items_in_hands', 'neighbors'):
            accessed.add(attr)
    
    return accessed

def _extract_string_literals(source_code: str) -> Set[str]:
    # extract string literals that might be category/usage/role values
    literals = set()
    pattern = r'["\']([a-z_][a-z0-9_]*)["\']'
    for match in re.finditer(pattern, source_code, re.IGNORECASE):
        literal = match.group(1).lower()
        if len(literal) > 2 and literal not in ('the', 'and', 'for', 'not', 'you', 'can', 'was', 'has'):
            literals.add(literal)
    return literals

def _infer_entity_requirements_from_code(source_code: str) -> Dict[str, Set[str]]:
    # analyze source code to infer entity requirements
    requirements = {
        "object_categories": set(),
        "object_usages": set(),
        "npc_roles": set(),
        "object_properties": set(),
    }
    
    # known valid categories, usages, and roles from the game
    known_categories = {'weapon', 'armor', 'material', 'station', 'container', 'tool', 'currency', 'consumable'}
    known_usages = {'attack', 'defend', 'craft', 'store', 'unlock', 'writable', 'write', 'exchange', 'consumable'}
    known_roles = {'merchant', 'enemy', 'quest', 'scout', 'guard', 'villager'}
    known_properties = {'attack_power', 'defense', 'hp', 'value', 'capacity', 'heal_amount', 
                        'base_hp', 'base_attack_power', 'slope_hp', 'slope_attack_power',
                        'coins', 'level', 'size', 'max_text_length'}
    
    accessed = _extract_accessed_attributes(source_code)
    string_literals = _extract_string_literals(source_code)
    
    for attr in accessed:
        if attr in known_properties:
            requirements["object_properties"].add(attr)
    
    for literal in string_literals:
        if literal in known_categories:
            requirements["object_categories"].add(literal)
        if literal in known_usages:
            requirements["object_usages"].add(literal)
        if literal in known_roles:
            requirements["npc_roles"].add(literal)
    
    code_lower = source_code.lower()
    
    cat_pattern = r'category\s*[=!]=\s*["\'](\w+)["\']'
    for match in re.finditer(cat_pattern, code_lower):
        cat = match.group(1)
        if cat in known_categories:
            requirements["object_categories"].add(cat)
    
    usage_pattern = r'usage\s*[=!]=\s*["\'](\w+)["\']'
    for match in re.finditer(usage_pattern, code_lower):
        usage = match.group(1)
        if usage in known_usages:
            requirements["object_usages"].add(usage)
    
    role_pattern = r'role\s*[=!]=\s*["\'](\w+)["\']'
    for match in re.finditer(role_pattern, code_lower):
        role = match.group(1)
        if role in known_roles:
            requirements["npc_roles"].add(role)
    
    if '.enemy' in code_lower or 'npc.enemy' in code_lower or 'enemy=true' in code_lower.replace(' ', ''):
        requirements["npc_roles"].add("enemy")
    
    prop_patterns = [
        (r'\.attack_power', 'attack_power'),
        (r'\.defense', 'defense'),
        (r'\.hp', 'hp'),
        (r'\.base_hp', 'base_hp'),
        (r'\.base_attack_power', 'base_attack_power'),
        (r'\.value', 'value'),
        (r'\.capacity', 'capacity'),
        (r'\.heal_amount', 'heal_amount'),
        (r'\.coins', 'coins'),
    ]
    for pattern, prop in prop_patterns:
        if re.search(pattern, code_lower):
            requirements["object_properties"].add(prop)
    
    return requirements

def extract_game_rules(game_name: str) -> Dict[str, Any]:
    # extract action and step rules, analyze code for entity requirements
    game_root = f"games/generated/{game_name}" if game_name != "base" else "games/base"
    
    action_rules_path = Path(current_directory) / game_root / "rules" / "action_rules.py"
    step_rules_dir = Path(current_directory) / game_root / "rules" / "step_rules"
    
    rules_info = {
        "action_rules": [],
        "step_rules": [],
        "entity_requirements": {
            "required_object_categories": set(),
            "required_object_usages": set(),
            "required_npc_roles": set(),
            "object_properties": set(),
        },
        "rule_details": []
    }
    
    if action_rules_path.exists():
        content = action_rules_path.read_text(encoding="utf-8")
        
        class_pattern = r'class\s+(\w+Rule)\(BaseActionRule\):(.*?)(?=\nclass\s|\Z)'
        matches = re.findall(class_pattern, content, re.DOTALL)
        
        for class_name, class_body in matches:
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', class_body)
            verb_match = re.search(r'verb\s*=\s*["\']([^"\']+)["\']', class_body)
            desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', class_body)
            params_match = re.search(r'params\s*=\s*\[([^\]]*)\]', class_body)
            
            rule_info = {
                "class": class_name,
                "name": name_match.group(1) if name_match else class_name,
                "verb": verb_match.group(1) if verb_match else "",
                "description": desc_match.group(1) if desc_match else "",
                "params": [],
            }
            
            if params_match:
                params_str = params_match.group(1)
                rule_info["params"] = re.findall(r'["\']([^"\']+)["\']', params_str)
            
            rules_info["action_rules"].append(rule_info)
            
            rule_requirements = _infer_entity_requirements_from_code(class_body)
            
            for key in rule_requirements:
                target_key = "required_" + key if not key.startswith("object_") else key.replace("object_", "required_object_")
                if key == "object_properties":
                    target_key = "object_properties"
                if target_key in rules_info["entity_requirements"]:
                    rules_info["entity_requirements"][target_key].update(rule_requirements[key])
                elif key == "object_properties":
                    rules_info["entity_requirements"]["object_properties"].update(rule_requirements[key])
            
            rules_info["rule_details"].append({
                "class": class_name,
                "verb": rule_info["verb"],
                "inferred_requirements": {k: list(v) for k, v in rule_requirements.items()}
            })
    
    if step_rules_dir.exists():
        content = "\n".join(
            f.read_text(encoding="utf-8")
            for f in sorted(step_rules_dir.glob("*.py"))
            if f.name != "__init__.py"
        )
        
        class_pattern = r'class\s+(\w+StepRule)\(BaseStepRule\):(.*?)(?=\nclass\s|\Z)'
        matches = re.findall(class_pattern, content, re.DOTALL)
        
        for class_name, class_body in matches:
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', class_body)
            desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', class_body)
            
            rule_info = {
                "class": class_name,
                "name": name_match.group(1) if name_match else class_name,
                "description": desc_match.group(1) if desc_match else "",
            }
            rules_info["step_rules"].append(rule_info)
            
            rule_requirements = _infer_entity_requirements_from_code(class_body)
            
            for key in rule_requirements:
                target_key = "required_" + key if not key.startswith("object_") else key.replace("object_", "required_object_")
                if key == "object_properties":
                    target_key = "object_properties"
                if target_key in rules_info["entity_requirements"]:
                    rules_info["entity_requirements"][target_key].update(rule_requirements[key])
                elif key == "object_properties":
                    rules_info["entity_requirements"]["object_properties"].update(rule_requirements[key])
    
    # convert sets to sorted lists for json serialization and consistency
    for key in rules_info["entity_requirements"]:
        rules_info["entity_requirements"][key] = sorted(list(rules_info["entity_requirements"][key]))
    
    return rules_info

def analyze_difficulty_progression(world_data: Dict[str, Any]) -> Dict[str, Any]:
    # analyze existing entities for difficulty progression
    analysis = {
        "existing_levels": {"areas": [], "objects": [], "npcs": []},
        "max_level": 1,
        "min_level": 1,
        "level_distribution": {},
        "stat_ranges": {
            "attack_power": {"min": float('inf'), "max": 0},
            "defense": {"min": float('inf'), "max": 0},
            "hp": {"min": float('inf'), "max": 0},
            "value": {"min": float('inf'), "max": 0},
        }
    }
    
    entities = world_data.get("entities", {})
    
    for place in entities.get("places", []):
        for area in place.get("areas", []):
            level = area.get("level", 1)
            analysis["existing_levels"]["areas"].append(level)
            analysis["level_distribution"][level] = analysis["level_distribution"].get(level, 0) + 1
    
    for obj in entities.get("objects", []):
        level = obj.get("level", 1)
        analysis["existing_levels"]["objects"].append(level)
        
        for stat in ["attack_power", "defense", "value"]:
            if stat in obj and obj[stat] is not None:
                val = obj[stat]
                analysis["stat_ranges"][stat]["min"] = min(analysis["stat_ranges"][stat]["min"], val)
                analysis["stat_ranges"][stat]["max"] = max(analysis["stat_ranges"][stat]["max"], val)
    
    for npc in entities.get("npcs", []):
        if npc.get("enemy", False):
            base_hp = npc.get("base_hp", npc.get("hp", 10))
            base_attack = npc.get("base_attack_power", npc.get("attack_power", 5))
            analysis["stat_ranges"]["hp"]["min"] = min(analysis["stat_ranges"]["hp"]["min"], base_hp)
            analysis["stat_ranges"]["hp"]["max"] = max(analysis["stat_ranges"]["hp"]["max"], base_hp)
            analysis["stat_ranges"]["attack_power"]["min"] = min(analysis["stat_ranges"]["attack_power"]["min"], base_attack)
            analysis["stat_ranges"]["attack_power"]["max"] = max(analysis["stat_ranges"]["attack_power"]["max"], base_attack)
    
    all_levels = (analysis["existing_levels"]["areas"] + 
                  analysis["existing_levels"]["objects"])
    if all_levels:
        analysis["max_level"] = max(all_levels)
        analysis["min_level"] = min(all_levels)
    
    for stat in analysis["stat_ranges"]:
        if analysis["stat_ranges"][stat]["min"] == float('inf'):
            analysis["stat_ranges"][stat]["min"] = 0
    
    return analysis

SYSTEM_PROMPT = """You are a game world entity generator for a text-based RPG. You create entities (places, objects, NPCs) that are:
1. USEFUL - Every entity must serve a purpose in the game mechanics
2. BALANCED - Difficulty increases progressively with levels
3. INTERCONNECTED - Objects should be craftable from materials, NPCs should drop useful items

GAME MECHANICS AWARENESS:
- Players can: pick up, drop, store, craft, equip, attack, buy/sell, enter areas, etc.
- Enemies attack players and drop loot when defeated
- Objects have levels that determine when they become relevant
- Areas have levels that determine enemy strength and available resources

DIFFICULTY PROGRESSION RULES:
- Level 1-2: Basic materials, weak enemies, simple crafts
- Level 3-4: Intermediate materials, moderate enemies, useful equipment  
- Level 5+: Rare materials, dangerous enemies, powerful equipment

ENTITY DESIGN PRINCIPLES:
1. Materials should be used in multiple craft recipes
2. NPCs: Merchants should sell useful items; Enemies should drop crafting materials
3. Areas: Lower level areas should be accessible from spawn; higher levels require progression
4. Every object should either be: craftable, a crafting ingredient, purchasable, or dropped by enemies

NAMING CONVENTIONS:
- Prefer single-word names for objects, NPCs, and areas (e.g. "dagger", "coal", "cavern").
- Two-word names are acceptable when needed (e.g. "iron ore", "dark forest").
- NEVER use names longer than 3 words. Avoid verbose names like "ancient enchanted longsword".
- Place names may be 1-2 words (e.g. "Ironforge", "Salt Flats").

Return ONLY valid JSON, no explanations."""


def build_user_prompt(
    world_data: Dict[str, Any],
    rules_info: Dict[str, Any],
    difficulty_analysis: Dict[str, Any],
    num_places: int,
    num_objects: int,
    num_npcs: int,
    max_level: int,
    description: str | None = None,
    generate_graph: bool = False,
) -> str:
    action_verbs = [r["verb"] for r in rules_info["action_rules"]]
    entity_reqs = rules_info["entity_requirements"]
    
    theme_block = ""
    if description:
        theme_block = f"""WORLD THEME / DESCRIPTION:
{description}

All generated entities (places, objects, NPCs) MUST fit this theme. Names,
descriptions, lore, and aesthetics should be consistent with the world
described above.\n\n"""

    # build graph generation block for the prompt
    graph_instructions = ""
    graph_json_block = ""
    if generate_graph:
        existing_areas = []
        for place in world_data.get("entities", {}).get("places", []):
            for area in place.get("areas", []):
                existing_areas.append(area["id"])
        unlock_objs = [o["id"] for o in world_data.get("entities", {}).get("objects", [])
                       if o.get("usage") == "unlock"]
        unlock_note = (
            f"Available key objects for locking: {', '.join(unlock_objs)}"
            if unlock_objs else
            "No unlock objects exist yet — set all connections to unlocked (locked=false, key=null)."
        )

        spawn_area = world_data.get('initializations', {}).get('spawn', {}).get('area', 'unknown')

        graph_instructions = f"""

GRAPH CONNECTION GENERATION:
Also generate a "graph" section with connections between ALL areas (both
existing and newly generated).  The connections define which rooms are
traversable and whether they are locked.

RULES FOR CONNECTIONS:
1. Every area must be reachable from the spawn area ({spawn_area}).
2. Within a place, areas MUST form a branching graph, NOT a straight line.
   Most areas should connect to 2 other areas; some may connect to 1
   (dead ends) or 3 (hubs).  Add occasional cross-links so there are
   multiple paths between points.
3. Between places, add 1-2 connections per pair of adjacent places so
   all places are reachable — prefer connecting different areas rather
   than always the same "gateway" area.
4. The overall graph should resemble a web, not a corridor.
   Aim for an average node degree of ~2 (total connections ≈ area count).
5. Locked places ("unlocked": false) should have their inter-place connections
   locked, requiring a key object.  Connections within an unlocked place should
   generally be unlocked.
6. {unlock_note}
7. Each connection is bidirectional — only list it once.
8. Existing areas: {', '.join(existing_areas)}

MAIN QUEST & TOPOLOGY AWARENESS:
The game has a main quest whose stages are assigned to areas by BFS
distance from the spawn area at runtime (closest areas first, expanding
outward through later chapters).  The graph you generate MUST support
this progressive exploration pattern:

- At least 4-5 areas must be reachable from spawn WITHOUT any locked
  doors (through unlocked connections only).  These serve the early
  quest stages (guide NPC, shrine candidates, etc.).
- Locked connections should gate access to higher-level areas that are
  used by later quest chapters (lair, bosses, final areas).
- The overall graph should form a roughly concentric expansion from
  spawn: low-level unlocked areas close to spawn, mid-level areas
  behind one lock, high-level areas deeper still.
- Ensure there are at least 8 distinct areas reachable overall so the
  quest has enough unique locations to assign across all its stages.
- If the world has dark areas (light=false), place some of them at
  medium depth (2-3 hops from spawn) — the quest may require combat
  in dark areas.
"""
        graph_json_block = """,
  "graph": {{
    "connections": [
      {{"from": "area_A", "to": "area_B", "locked": false, "key": null}},
      {{"from": "area_C", "to": "area_D", "locked": true, "key": "obj_key"}}
    ]
  }}"""

    prompt = f"""{theme_block}REFERENCE WORLD DEFINITION (for learning entity format and crafting patterns):
{json.dumps(world_data, indent=2)}
{graph_instructions}

IMPORTANT — REPLACE MODE:
You are generating a COMPLETE new set of entities that will REPLACE the
existing ones.  Do NOT copy entities from the reference above — generate
entirely new, theme-appropriate entities.  The reference is provided only
so you can learn the JSON schema, crafting depth patterns, and stat ranges.

HARDCODED OBJECTS THAT MUST BE INCLUDED:
The game engine hardcodes certain object IDs.  Your generated entity list
MUST include objects with these exact IDs (adapt their names/descriptions
to fit the theme, but keep the IDs and functional fields intact):
- obj_coin: category="currency", usage="exchange" — used by the economy system
- obj_key: category="tool", usage="unlock" — used by the unlock action rule
- obj_key_blank: category="material", usage="craft" — tutorial crafts obj_key from this
- obj_pen: category="tool", usage="write" — tutorial uses this for writing
- obj_paper: category="tool", usage="writable" — tutorial uses this for writing
- obj_small_bag: category="container", usage="store" — tutorial starting container
- obj_wood_log: category="material", usage="craft", level=1 — basic raw material
- obj_iron_bar: category="material", usage="craft", level=3 — intermediate material
- obj_workbench: category="station", usage="craft", level=3 — crafting station
- obj_furnace: category="station", usage="craft", level=4 — smelting station
You may change their names, descriptions, and theme-specific flavor, but
the IDs, categories, and usages above must remain exactly as listed.

GAME ACTION RULES (what players can do):
{', '.join(action_verbs)}

REQUIRED ENTITY TYPES FOR GAME MECHANICS:
- Object Categories Needed: {', '.join(entity_reqs['required_object_categories']) or 'None specified'}
- Object Usages Needed: {', '.join(entity_reqs['required_object_usages']) or 'None specified'}
- NPC Roles Needed: {', '.join(entity_reqs['required_npc_roles']) or 'None specified'}
- Object Properties to Include: {', '.join(entity_reqs['object_properties']) or 'None specified'}

GENERATION REQUIREMENTS:
Generate a COMPLETE entity set for levels 1 to {max_level}:
- EXACTLY {num_places} places (each with 2-4 areas, levels distributed across 1-{max_level})
- EXACTLY {num_objects} objects with progressive difficulty (no more, no fewer):
  - ~50% materials (various levels, used in crafting)
  - ~20% weapons (attack_power scales with level)
  - ~20% armor (defense scales with level)
  - ~10% tools/misc (containers, consumables, etc.)
  (includes the hardcoded objects listed above)
- EXACTLY {num_npcs} NPCs (no more, no fewer):
  - At least 1 merchant per 3 areas
  - Enemies with stats scaling by level: base_hp ~{max(20, difficulty_analysis['stat_ranges']['hp']['min'])} + level*{max(15, (difficulty_analysis['stat_ranges']['hp']['max'] - difficulty_analysis['stat_ranges']['hp']['min'])//max_level)}, similar for attack

BALANCE GUIDELINES:
1. Early game (level 1-2): Players should be able to survive and progress with basic gear
2. Mid game (level 3-4): Require crafted weapons/armor to handle enemies
3. Late game (level 5+): Challenging enemies, powerful rewards
4. Ensure crafting chains: raw material → processed material → equipment
5. Each area should have 1-3 crafting materials matching its level
6. Objects used for "craft" shall at least be referenced in one other object's crafting recipe

PLAYABILITY CONSTRAINTS:
NOTE: If the game description / world theme above specifies hardcoded
requirements (e.g. "all areas are dark", "no merchants", "single biome"),
those theme requirements override the default constraints below (such as
the light/dark mix or NPC role requirements).

A. Object Category & Usage Coverage
 1. Every required object category ({', '.join(entity_reqs['required_object_categories']) or 'weapon, armor, material, station, container, tool, currency, consumable'}) must have at least one object.
 2. Every required object usage ({', '.join(entity_reqs['required_object_usages']) or 'attack, defend, craft, store, unlock, writable, write, exchange, consumable'}) must be represented by at least one object.
 3. An object with id="obj_key" (usage="unlock") MUST exist — the unlock action rule hardcodes this ID.
 4. An object with id="obj_key_blank" (category="material") MUST exist — the tutorial requires crafting obj_key from obj_key_blank.

B. Crafting Integrity
 5. Every material (category="material") must appear as an ingredient in at least one crafting recipe.
 6. Every crafting recipe must reference only object IDs that exist in the full entity list.
 7. At least one multi-step crafting chain must exist (raw material → intermediate → final product, minimum 2 steps).
 8. Every station (category="station") must be referenced as a craft dependency by at least one recipe.

B2. Crafting Hierarchy Depth as Difficulty Control
 Crafting hierarchy depth (the number of sequential crafting steps from raw
 materials to the final product) is a key difficulty lever. Higher-level
 equipment should require DEEPER crafting chains. Use the existing world
 definition above as an in-context example of good depth-level correlation:
   - Level 1-2 items: depth 0-1 (raw materials, trivial single-step crafts)
   - Level 3-4 items: depth 2-4 (multi-step intermediate processing)
   - Level 5+  items: depth 5-6 (long chains requiring many prerequisite crafts)
 A deeper chain means the player must explore more areas, defeat more enemies,
 and master more recipes before obtaining that item — this IS the difficulty.

C. NPC Completeness
 9. At least 1 merchant NPC (role="merchant", enemy=false) must exist.
10. At least 2 distinct enemy NPC types (enemy=true) must exist.
11. Every enemy must have base_hp, base_attack_power, slope_hp, slope_attack_power.
12. At least one enemy must drop loot (non-empty objects list).

D. Area & Place Design
13. Every place must contain at least 1 area.
14. Every area must have a level field (integer >= 1).
15. Areas should include a mix of light=true and light=false (aim for roughly 60-70%% lit, 30-40%% dark).
16. Levels should be distributed across the full range 1-{max_level}; no level may be unrepresented across all areas.

E. Equipment Progression
17. Weapon attack_power must increase with level (no higher-level weapon weaker than a lower-level one).
18. Armor defense must increase with level (no higher-level armor weaker than a lower-level one).
19. At least one weapon and one armor must exist at both the lowest and highest level tiers.

F. Economy & Loot
20. obj_coin (currency) must exist in the entity list — the economy system hardcodes this ID.
21. At least one merchant must have a non-empty inventory or objects list.

Return as JSON with structure:
{{
  "places": [
    {{"type": "place", "id": "place_X", "name": "Place Name", "unlocked": true/false, "areas": [
      {{"type": "area", "id": "area_X", "name": "area_name", "level": N}}
    ]}}
  ],
  "objects": [
    {{"type": "object", "id": "obj_X", "name": "object_name", "category": "...", "usage": "...", "value": N, "size": N, "description": "...", "craft": {{"ingredients": {{}}, "dependencies": []}}, "level": N, ...extra_properties...}}
  ],
  "npcs": [
    {{"type": "npc", "id": "npc_X", "name": "npc_name", "enemy": bool, "unique": bool, "description": "...", "role": "...", ...stats_if_enemy..., "combat_pattern": ["attack", "defend", "attack", "wait"] (REQUIRED for enemy NPCs)}}
  ]{graph_json_block}
}}

Ensure all IDs are unique and follow existing naming patterns. Only return valid JSON."""
    
    return prompt

def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return Path(current_directory) / pp

def ensure_generated_assets(game_name: str, overwrite: bool = False) -> str:
    if not game_name or game_name.strip() == "":
        raise ValueError("game_name must be a non-empty string")

    env_configs_base = Path(current_directory) / BASE_ENV_CONFIGS_DIR
    env_configs_generated = Path(current_directory) / "assets" / "env_configs" / "generated" / game_name
    world_defs_base = Path(current_directory) / BASE_WORLD_DEFS_DIR
    world_defs_generated = Path(current_directory) / "assets" / "world_definitions" / "generated" / game_name

    if overwrite:
        if env_configs_generated.exists():
            shutil.rmtree(env_configs_generated)
        if world_defs_generated.exists():
            shutil.rmtree(world_defs_generated)

    if not env_configs_generated.exists():
        env_configs_generated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(env_configs_base, env_configs_generated)

    if not world_defs_generated.exists():
        world_defs_generated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(world_defs_base, world_defs_generated)

    return str(world_defs_generated)

def get_default_world_paths(game_name: str) -> tuple[str, str]:
    world_definition_path = f"assets/world_definitions/generated/{game_name}/default.json"
    output_path = f"assets/world_definitions/generated/{game_name}/populated.json"
    return world_definition_path, output_path

def extract_json_from_response(response: str) -> Dict[str, Any]:
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        response = json_match.group()

    return json.loads(response)

def merge_entities(original: Dict[str, Any], new_entities: Dict[str, Any]) -> Dict[str, Any]:
    """Replace mode: new entities fully replace existing ones."""
    merged = json.loads(json.dumps(original))

    merged.setdefault("entities", {})

    if "places" in new_entities and isinstance(new_entities["places"], list):
        merged["entities"]["places"] = new_entities["places"]

    if "objects" in new_entities and isinstance(new_entities["objects"], list):
        merged["entities"]["objects"] = new_entities["objects"]

    if "npcs" in new_entities and isinstance(new_entities["npcs"], list):
        merged["entities"]["npcs"] = new_entities["npcs"]

    if "graph" in new_entities and isinstance(new_entities["graph"], dict):
        merged["graph"] = new_entities["graph"]

    # --- Fix initializations to match newly generated entities ---
    _fix_initializations(merged)

    return merged


def _fix_initializations(merged: Dict[str, Any]) -> None:
    inits = merged.get("initializations", {})
    if not inits:
        return

    # Collect valid IDs from merged entities
    all_area_ids: list[str] = []
    for place in merged.get("entities", {}).get("places", []):
        for area in place.get("areas", []):
            aid = area.get("id")
            if aid:
                all_area_ids.append(aid)
    all_obj_ids = {obj["id"] for obj in merged.get("entities", {}).get("objects", [])}

    # Fix spawn area: if it doesn't exist in the new world, use the first area
    spawn = inits.get("spawn", {})
    if spawn.get("area") and spawn["area"] not in set(all_area_ids):
        old_area = spawn["area"]
        spawn["area"] = all_area_ids[0] if all_area_ids else old_area

    # Fix spawn objects: remove any that don't exist in the new world
    spawn_objs = spawn.get("objects", {})
    if isinstance(spawn_objs, dict):
        invalid_spawn_objs = [oid for oid in spawn_objs if oid not in all_obj_ids]
        for oid in invalid_spawn_objs:
            del spawn_objs[oid]

    # Fix undistributable_objects: remove any that don't exist in the new world
    undist = inits.get("undistributable_objects", [])
    if isinstance(undist, list):
        inits["undistributable_objects"] = [oid for oid in undist if oid in all_obj_ids]

def validate_entities(new_entities: Dict[str, Any], world_data: Dict[str, Any], rules_info: Dict[str, Any]) -> Tuple[bool, List[str]]:
    # validate generated entities for compatibility and balance
    errors = []
    warnings = []
    
    existing_objects = {obj["id"] for obj in world_data.get("entities", {}).get("objects", [])}
    existing_npcs = {npc["id"] for npc in world_data.get("entities", {}).get("npcs", [])}
    new_object_ids = {obj["id"] for obj in new_entities.get("objects", [])}
    all_objects = existing_objects | new_object_ids
    
    # validate objects
    for obj in new_entities.get("objects", []):
        obj_id = obj.get("id", "unknown")
        
        if not obj.get("name"):
            errors.append(f"Object {obj_id} missing 'name'")
        if not obj.get("category"):
            errors.append(f"Object {obj_id} missing 'category'")
        
        craft = obj.get("craft", {})
        ingredients = craft.get("ingredients", {})
        for ing_id in ingredients.keys():
            if ing_id not in all_objects:
                errors.append(f"Object {obj_id} references non-existent ingredient: {ing_id}")
        
        dependencies = craft.get("dependencies", [])
        for dep_id in dependencies:
            if dep_id not in all_objects:
                errors.append(f"Object {obj_id} references non-existent dependency: {dep_id}")
        
        if obj.get("category") == "weapon" and not obj.get("attack_power"):
            warnings.append(f"Weapon {obj_id} missing 'attack_power'")
        
        if obj.get("category") == "armor" and not obj.get("defense"):
            warnings.append(f"Armor {obj_id} missing 'defense'")
        
        if obj.get("category") == "container" and not obj.get("capacity"):
            warnings.append(f"Container {obj_id} missing 'capacity'")
    
    # validate NPCs
    for npc in new_entities.get("npcs", []):
        npc_id = npc.get("id", "unknown")
        
        if not npc.get("name"):
            errors.append(f"NPC {npc_id} missing 'name'")
        
        if npc.get("enemy", False):
            if not npc.get("base_hp") and not npc.get("hp"):
                warnings.append(f"Enemy NPC {npc_id} missing HP stats")
            if not npc.get("base_attack_power") and not npc.get("attack_power"):
                warnings.append(f"Enemy NPC {npc_id} missing attack stats")
            if not npc.get("combat_pattern"):
                warnings.append(f"Enemy NPC {npc_id} missing combat_pattern (e.g. ['attack', 'defend', 'attack', 'wait'])")
        
        if npc.get("role") == "merchant" and npc.get("enemy", False):
            errors.append(f"NPC {npc_id} cannot be both merchant and enemy")
    
    for place in new_entities.get("places", []):
        place_id = place.get("id", "unknown")
        
        if not place.get("name"):
            errors.append(f"Place {place_id} missing 'name'")
        
        areas = place.get("areas", [])
        if not areas:
            warnings.append(f"Place {place_id} has no areas")
        
        for area in areas:
            if not area.get("id"):
                errors.append(f"Area in {place_id} missing 'id'")
            if not area.get("level"):
                warnings.append(f"Area {area.get('id', 'unknown')} in {place_id} missing 'level'")
    
    for warning in warnings:
        print(f"  Warning: {warning}")
    
    # --- Playability constraint checks (warnings, non-fatal) ---

    # Hardcoded object IDs that must exist
    HARDCODED_OBJ_IDS = {
        "obj_coin", "obj_key", "obj_key_blank", "obj_pen", "obj_paper",
        "obj_small_bag", "obj_wood_log", "obj_iron_bar", "obj_workbench", "obj_furnace",
    }
    for hid in sorted(HARDCODED_OBJ_IDS):
        if hid not in all_objects:
            warnings.append(f"Hardcoded object '{hid}' missing from entity list (game engine requires it)")

    # Category coverage
    REQUIRED_CATEGORIES = {"weapon", "armor", "material", "station", "container", "tool", "currency", "consumable"}
    found_categories = {obj.get("category") for obj in new_entities.get("objects", [])}
    for cat in sorted(REQUIRED_CATEGORIES - found_categories):
        warnings.append(f"No object with category='{cat}' found")

    # Usage coverage
    REQUIRED_USAGES = {"attack", "defend", "craft", "store", "unlock", "writable", "write", "exchange", "consumable"}
    found_usages = {obj.get("usage") for obj in new_entities.get("objects", [])}
    for usage in sorted(REQUIRED_USAGES - found_usages):
        warnings.append(f"No object with usage='{usage}' found")

    # Materials used in at least one recipe
    all_obj_map = {obj["id"]: obj for obj in new_entities.get("objects", [])}
    all_obj_map.update({obj["id"]: obj for obj in world_data.get("entities", {}).get("objects", [])})
    materials = {oid for oid, o in all_obj_map.items() if o.get("category") == "material"}
    used_in_recipes = set()
    for o in all_obj_map.values():
        used_in_recipes.update(o.get("craft", {}).get("ingredients", {}).keys())
    unused_mats = materials - used_in_recipes
    if unused_mats:
        warnings.append(f"Materials not used in any recipe: {', '.join(sorted(unused_mats))}")

    # Station referenced as dependency
    stations = {oid for oid, o in all_obj_map.items() if o.get("category") == "station"}
    deps_referenced = set()
    for o in all_obj_map.values():
        deps_referenced.update(o.get("craft", {}).get("dependencies", []))
    unreferenced_stations = stations - deps_referenced
    if unreferenced_stations:
        warnings.append(f"Stations not referenced as craft dependency: {', '.join(sorted(unreferenced_stations))}")

    # NPC checks
    npcs = new_entities.get("npcs", [])
    merchants = [n for n in npcs if n.get("role") == "merchant" and not n.get("enemy", False)]
    enemies = [n for n in npcs if n.get("enemy", False)]
    if not merchants:
        warnings.append("No merchant NPC found")
    if len(enemies) < 2:
        warnings.append(f"Only {len(enemies)} enemy NPC type(s) found (need at least 2)")
    enemies_with_loot = [e for e in enemies if e.get("objects")]
    if enemies and not enemies_with_loot:
        warnings.append("No enemy drops loot (all have empty objects list)")
    
    return len(errors) == 0, errors

def check_difficulty_balance(new_entities: Dict[str, Any], max_level: int) -> Tuple[bool, List[str]]:
    # check if entities have proper difficulty progression
    issues = []
    
    objects = new_entities.get("objects", [])
    level_counts = {}
    for obj in objects:
        level = obj.get("level", 1)
        level_counts[level] = level_counts.get(level, 0) + 1
    
    for level in range(1, max_level + 1):
        if level_counts.get(level, 0) == 0:
            issues.append(f"No objects at level {level}")
    
    weapons = [o for o in objects if o.get("category") == "weapon"]
    if weapons:
        weapons_sorted = sorted(weapons, key=lambda x: x.get("level", 1))
        last_power = 0
        for w in weapons_sorted:
            power = w.get("attack_power", 0)
            if power and power < last_power:
                issues.append(f"Weapon {w.get('name')} (level {w.get('level')}) has lower power ({power}) than lower-level weapons ({last_power})")
            last_power = max(last_power, power or 0)
    
    armors = [o for o in objects if o.get("category") == "armor"]
    if armors:
        armors_sorted = sorted(armors, key=lambda x: x.get("level", 1))
        last_defense = 0
        for a in armors_sorted:
            defense = a.get("defense", 0)
            if defense and defense < last_defense:
                issues.append(f"Armor {a.get('name')} (level {a.get('level')}) has lower defense ({defense}) than lower-level armor ({last_defense})")
            last_defense = max(last_defense, defense or 0)
    
    enemies = [n for n in new_entities.get("npcs", []) if n.get("enemy", False)]
    if enemies:
        # enemies should have reasonable HP ranges
        for enemy in enemies:
            base_hp = enemy.get("base_hp", enemy.get("hp", 0))
            if base_hp and base_hp > 200:
                issues.append(f"Enemy {enemy.get('name')} has very high base HP ({base_hp}), may be too difficult")
    
    return len(issues) == 0, issues

def check_crafting_material_coverage(new_entities: Dict[str, Any], world_data: Dict[str, Any], max_level: int) -> Tuple[bool, List[str]]:
    # check if crafting materials have proper coverage across levels and areas
    issues = []
    
    all_objects = {}
    for obj in world_data.get("entities", {}).get("objects", []):
        all_objects[obj["id"]] = obj
    for obj in new_entities.get("objects", []):
        all_objects[obj["id"]] = obj
    
    all_area_ids = set()
    for place in world_data.get("entities", {}).get("places", []):
        for area in place.get("areas", []):
            all_area_ids.add(area["id"])
    for place in new_entities.get("places", []):
        for area in place.get("areas", []):
            all_area_ids.add(area["id"])
    
    craft_materials = {obj_id: obj for obj_id, obj in all_objects.items() 
                      if obj.get("usage") == "craft" and obj.get("category") == "material"}
    
    if not craft_materials:
        issues.append("No crafting materials found in generated or existing objects")
        return False, issues
    
    material_levels = {}
    for mat_id, mat in craft_materials.items():
        level = mat.get("level", 1)
        if level not in material_levels:
            material_levels[level] = []
        material_levels[level].append(mat_id)
    
    for level in range(1, max_level + 1):
        if level not in material_levels:
            issues.append(f"no crafting material at level {level}")
    
    materials_by_area = {area_id: [] for area_id in all_area_ids}
    for mat_id, mat in craft_materials.items():
        mat_areas = mat.get("areas", [])
        if isinstance(mat_areas, list):
            for area_id in mat_areas:
                if area_id in materials_by_area:
                    materials_by_area[area_id].append(mat_id)
    
    missing_area_coverage = [area_id for area_id, mats in materials_by_area.items() if not mats]
    if missing_area_coverage:
        issues.append(f"crafting materials do not cover these areas: {', '.join(missing_area_coverage)}")
    
    used_materials = set()
    for obj_id, obj in all_objects.items():
        craft = obj.get("craft", {})
        ingredients = craft.get("ingredients", {})
        for ing_id in ingredients.keys():
            used_materials.add(ing_id)
    
    unused_materials = set(craft_materials.keys()) - used_materials
    if unused_materials:
        unused_list = sorted(list(unused_materials))
        issues.append(f"crafting materials not used in any recipe: {', '.join(unused_list)}")
    
    return len(issues) == 0, issues

def generate_entities(
    llm_name: str,
    num_places: int,
    num_objects: int,
    num_npcs: int,
    max_level: int,
    world_data: Dict[str, Any],
    game_name: str = "base",
    description: str | None = None,
    llm=None,
    generate_graph: bool = False,
) -> Dict[str, Any]:
    # generate entities using llm with rules-awareness
    if llm is None:
        llm = OpenAILanguageModel(llm_name=llm_name)

    entity_output_token_budget = 32768
    if getattr(llm, "max_new_tokens", None) is None:
        llm.max_new_tokens = entity_output_token_budget
    else:
        llm.max_new_tokens = max(int(llm.max_new_tokens), entity_output_token_budget)
    
    rules_info = extract_game_rules(game_name)
    print(f"Extracted {len(rules_info['action_rules'])} action rules and {len(rules_info['step_rules'])} step rules")
    print(f"Entity generation max output tokens: {llm.max_new_tokens}")
    
    difficulty_analysis = analyze_difficulty_progression(world_data)
    
    user_prompt = build_user_prompt(
        world_data=world_data,
        rules_info=rules_info,
        difficulty_analysis=difficulty_analysis,
        num_places=num_places,
        num_objects=num_objects,
        num_npcs=num_npcs,
        max_level=max_level,
        description=description,
        generate_graph=generate_graph,
    )

    response = llm.generate(user_prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
    print(response["response"])
    new_entities = extract_json_from_response(response["response"])
    
    print("\nValidating generated entities...")
    valid, errors = validate_entities(new_entities, world_data, rules_info)
    if not valid:
        print("Validation errors found:")
        for error in errors:
            print(f"  Error: {error}")
        raise ValueError(f"Generated entities failed validation: {errors}")
    
    print("Checking difficulty balance...")
    balanced, issues = check_difficulty_balance(new_entities, max_level)
    if not balanced:
        print("Balance issues found (non-fatal):")
        for issue in issues:
            print(f"  Issue: {issue}")
    
    print("Checking crafting material coverage...")
    material_coverage, material_issues = check_crafting_material_coverage(new_entities, world_data, max_level)
    if not material_coverage:
        print("Material coverage issues found (non-fatal):")
        for issue in material_issues:
            print(f"  Issue: {issue}")
    
    return new_entities

def test_game(game_name: str, world_definition_path: str, timeout: int = 120) -> Tuple[bool, str]:
    # test the game with generated entities
    cmd = [
        "python", "eval.py",
        "--agent", "RandomAgent",
        "--max_steps", "500",
        "--enable_obs_valid_actions",
        "--overwrite",
        "--game_name", game_name,
        "--world_definition_path", world_definition_path,
        "--env_config_path", f"assets/env_configs/generated/{game_name}/initial.json"
    ]
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

def build_aider_prompt(
    game_name: str,
    world_definition_path: str,
    num_places: int,
    num_objects: int, 
    num_npcs: int,
    max_level: int,
    description: str | None = None,
    generate_graph: bool = False,
) -> str:
    # build a prompt for aider to generate entities with code awareness
    game_root = f"games/generated/{game_name}" if game_name != "base" else "games/base"
    
    theme_block = ""
    if description:
        theme_block = f"""\nWORLD THEME / DESCRIPTION:\n{description}\n\nAll generated entities (places, objects, NPCs) MUST fit this theme. Names,\ndescriptions, lore, and aesthetics should be consistent with the world\ndescribed above.\n"""
    
    return f"""Generate new entities for the text RPG game world definition.
{theme_block}
TASK: REPLACE the entire entities section in {world_definition_path}

THE FILE IS FULLY INDENTED JSON (2-space indent, one field per line).
When you write your SEARCH/REPLACE blocks, copy lines from the file
EXACTLY as they appear — including every space, comma, and line break.

IMPORTANT — REPLACE MODE:
REWRITE the "entities" section with entirely new, theme-appropriate
entities.  Remove all existing places, objects, and NPCs and replace
them with the new ones.  The existing entities are only there as a
format reference — do NOT keep them.

HARDCODED OBJECTS THAT MUST BE INCLUDED:
The game engine hardcodes certain object IDs.  Your generated entity list
MUST include objects with these exact IDs (adapt their names/descriptions
to fit the theme, but keep the IDs and functional fields intact):
- obj_coin: category="currency", usage="exchange"
- obj_key: category="tool", usage="unlock"
- obj_key_blank: category="material", usage="craft" (crafted into obj_key)
- obj_pen: category="tool", usage="write"
- obj_paper: category="tool", usage="writable"
- obj_small_bag: category="container", usage="store"
- obj_wood_log: category="material", usage="craft", level=1
- obj_iron_bar: category="material", usage="craft", level=3
- obj_workbench: category="station", usage="craft", level=3
- obj_furnace: category="station", usage="craft", level=4

REQUIREMENTS:
- Generate exactly {num_places} places (with 2-4 areas each)
- Generate exactly {num_objects} objects (materials, weapons, armor, tools)
  (includes the hardcoded objects above)
- Generate exactly {num_npcs} NPCs (mix of merchants and enemies)
- Level range: 1 to {max_level}

RULES AWARENESS:
Read the action rules in {game_root}/rules/action_rules.py to understand:
- What actions players can perform
- What entity types are required (weapons need attack_power, armor needs defense, etc.)
- How crafting works (craft.ingredients and craft.dependencies)

Read the step rules in {game_root}/rules/step_rules/ to understand:
- How combat works (enemy NPCs need base_hp, base_attack_power, slope_hp, slope_attack_power, and combat_pattern)
- combat_pattern is a list of actions like ["attack", "defend", "attack", "wait"] that enemy NPCs cycle through during combat
- How merchants work (NPCs with role="merchant" need inventory)

BALANCE REQUIREMENTS:
1. Level 1-2: Basic materials (easy to find), simple weapons (10-15 attack), basic armor (5-10 defense)
2. Level 3-4: Intermediate materials, better weapons (20-25 attack), better armor (10-15 defense)
3. Level 5+: Rare materials, powerful weapons (30+ attack), strong armor (20+ defense)

4. Enemy NPCs should scale:
   - Level 1: base_hp=20-30, base_attack=5-10
   - Level 3: base_hp=50-70, base_attack=15-20
   - Level 5: base_hp=100+, base_attack=25+

5. Crafting chains should exist:
   - Raw materials → Processed materials → Equipment
   - Higher level items should require ingredients from previous levels
   - The levels of raw materials shall span the full range from 1 to {max_level}
   - The areas of raw materials shall span all the areas in the world definition
   - Objects used for "craft" shall at least be referenced in one other object's crafting recipe

6. Crafting hierarchy depth controls difficulty:
   Crafting hierarchy depth (the number of sequential crafting steps from raw
   materials to the final product) is a key difficulty lever. Higher-level
   equipment should require DEEPER crafting chains. Learn from the existing
   base game world definition provided as context:
     - Level 1-2 items: depth 0-1 (raw materials, trivial single-step crafts)
     - Level 3-4 items: depth 2-4 (multi-step intermediate processing)
     - Level 5+  items: depth 5-6 (long chains requiring many prerequisite crafts)
   A deeper chain means the player must explore more areas, defeat more enemies,
   and master more recipes before obtaining that item — this IS the difficulty.

NAMING CONVENTIONS:
- Prefer single-word names (e.g. "dagger", "coal", "cavern").
- Two-word names are acceptable when needed (e.g. "iron ore").
- NEVER use names longer than 3 words.
- Place names may be 1-2 words.

ENTITY FORMAT (use expanded multi-line JSON matching the file's indentation):
- Object example:
  {{
    "type": "object",
    "id": "obj_example",
    "name": "example",
    "category": "material",
    "usage": "craft",
    "value": 5,
    "size": 1,
    "description": "An example object.",
    "craft": {{
      "ingredients": {{}},
      "dependencies": []
    }},
    "level": 1
  }}
- NPC example:
  {{
    "type": "npc",
    "id": "npc_example",
    "name": "example",
    "enemy": true,
    "unique": false,
    "description": "An example enemy.",
    "role": "enemy",
    "base_hp": 30,
    "slope_hp": 20,
    "base_attack_power": 5,
    "slope_attack_power": 10,
    "objects": ["obj_cloth"]
  }}
- Place example:
  {{
    "type": "place",
    "id": "place_example",
    "name": "Example Place",
    "unlocked": true,
    "areas": [
      {{
        "type": "area",
        "id": "area_example",
        "name": "clearing",
        "level": 1
      }}
    ]
  }}

Edit {world_definition_path} to REPLACE the entire "entities" section with the new entities.
Ensure all IDs are unique.

PLAYABILITY CONSTRAINTS:
NOTE: If the world theme above specifies hardcoded requirements (e.g.
"all areas are dark"), those override the defaults below.

A. Object Category & Usage Coverage
 - Every category (weapon, armor, material, station, container, tool,
   currency, consumable) must have at least one object.
 - Every usage (attack, defend, craft, store, unlock, writable, write,
   exchange, consumable) must be represented.

B. Crafting Integrity
 - Every material must appear as an ingredient in at least one recipe.
 - Every station must be referenced as a craft dependency by at least one recipe.
 - Crafting hierarchy depth controls difficulty: Level 1-2 items depth 0-1,
   Level 3-4 depth 2-4, Level 5+ depth 5-6.

C. NPC Completeness
 - At least 1 merchant (enemy=false) and at least 2 enemy types.
 - Every enemy must have base_hp, base_attack_power, slope_hp, slope_attack_power.
 - At least one enemy must drop loot (non-empty objects list).

D. Area & Place Design
 - Mix of light=true (60-70%%) and light=false (30-40%%) areas.
 - Levels distributed across the full range 1-{max_level}.

E. Equipment Progression
 - Weapon attack_power and armor defense must increase with level.
 - At least one weapon and armor at both lowest and highest level tiers.

F. Economy
 - obj_coin (currency) must exist. At least one merchant must have inventory.

{_build_aider_graph_block(game_name, world_definition_path, generate_graph)}
TEST COMMAND:
python eval.py --agent RandomAgent --max_steps 500 --game_name {game_name} --world_definition_path {world_definition_path}
"""

def _build_aider_graph_block(game_name: str, world_definition_path: str, generate_graph: bool) -> str:
    """Return the graph-generation instructions for the aider prompt."""
    if not generate_graph:
        return ""

    # Load current world definition to find areas and unlock objects
    abs_path = os.path.join(current_directory, world_definition_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            wd = json.load(f)
    except Exception:
        wd = {}

    existing_areas = []
    for place in wd.get("entities", {}).get("places", []):
        for area in place.get("areas", []):
            existing_areas.append(area["id"])
    unlock_objs = [o["id"] for o in wd.get("entities", {}).get("objects", [])
                   if o.get("usage") == "unlock"]
    spawn_area = wd.get("initializations", {}).get("spawn", {}).get("area", "unknown")
    unlock_note = (
        f"Available key objects for locking: {', '.join(unlock_objs)}"
        if unlock_objs else
        "No unlock objects exist yet — set all connections to unlocked (locked=false, key=null)."
    )

    return f"""
GRAPH CONNECTION GENERATION:
Also generate or update the top-level "graph" section with connections
between ALL areas (existing + newly generated).  The connections define
which rooms are traversable and whether they are locked.

RULES FOR CONNECTIONS:
1. Every area must be reachable from spawn ({spawn_area}).
2. Within a place, areas MUST form a branching graph, NOT a straight line.
   Most areas should connect to 2 other areas; some may connect to 1
   (dead ends) or 3 (hubs).  Add occasional cross-links so there are
   multiple paths.
3. Between places, add 1-2 connections per pair of adjacent places —
   prefer connecting different areas rather than always the same gateway.
4. The overall graph should resemble a web, not a corridor.
   Aim for an average node degree of ~2 (total connections ≈ area count).
5. Locked places ("unlocked": false) should have their inter-place connections
   locked, requiring a key object.  Within an unlocked place, connections
   should generally be unlocked.
6. {unlock_note}
7. Each connection is bidirectional — only list it once.
8. Existing areas: {', '.join(existing_areas)}
9. Connection format:
   {{"from": "area_A", "to": "area_B", "locked": false, "key": null}}

MAIN QUEST & TOPOLOGY AWARENESS:
The game has a main quest whose stages are assigned to areas by BFS
distance from the spawn area at runtime (closest areas first, expanding
outward through later chapters).  The graph you generate MUST support
this progressive exploration pattern:

- At least 4-5 areas must be reachable from spawn WITHOUT any locked
  doors (through unlocked connections only).  These serve the early
  quest stages (guide NPC, shrine candidates, etc.).
- Locked connections should gate access to higher-level areas that are
  used by later quest chapters (lair, bosses, final areas).
- The overall graph should form a roughly concentric expansion from
  spawn: low-level unlocked areas close to spawn, mid-level areas
  behind one lock, high-level areas deeper still.
- Ensure there are at least 8 distinct areas reachable overall so the
  quest has enough unique locations to assign across all its stages.
- If the world has dark areas (light=false), place some of them at
  medium depth (2-3 hops from spawn) — the quest may require combat
  in dark areas.

You MUST replace the entire "graph" object in the JSON file with the
updated connections covering all old and new areas.
"""


def generate_with_aider(
    game_name: str,
    world_definition_path: str,
    num_places: int,
    num_objects: int,
    num_npcs: int,
    max_level: int,
    llm_name: str = "gpt-4o",
    max_iterations: int = 3,
    test_timeout: int = 120,
    description: str | None = None,
    llm_provider: str | None = None,
    generate_graph: bool = False,
) -> Tuple[bool, str]:
    # generate entities using aider with iterative testing
    game_root = f"games/generated/{game_name}" if game_name != "base" else "games/base"
    
    prompt = build_aider_prompt(
        game_name, world_definition_path,
        num_places, num_objects, num_npcs, max_level,
        description=description,
        generate_graph=generate_graph,
    )
    
    context_files = [
        f"{game_root}/rules/action_rules.py",
        f"{game_root}/rules/step_rules/general.py",
        f"{game_root}/rules/step_rules/main_quest.py",
        f"{game_root}/rules/step_rules/tutorial.py",
        f"{game_root}/world.py",
    ]
    
    editable_files = [world_definition_path]
    
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
    
    abs_world_path = os.path.join(current_directory, world_definition_path)
    with open(abs_world_path, "r", encoding="utf-8") as f:
        original_world = json.load(f)

    # pretty-print the JSON so every field is on its own line.
    with open(abs_world_path, "w", encoding="utf-8") as f:
        json.dump(original_world, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n{'='*60}")
    print(f"Generating entities with aider (REPLACE mode)")
    print(f"Model: {llm_name}")
    print(f"Target: {num_places} places, {num_objects} objects, {num_npcs} NPCs")
    print(f"{'='*60}\n")
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---\n")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=current_directory,
                capture_output=False,
                text=True,
                timeout=600
            )
        except subprocess.TimeoutExpired:
            print("Aider timed out")
            continue
        except Exception as e:
            print(f"Aider error: {e}")
            continue
        
        # load and validate generated entities
        try:
            with open(abs_world_path, "r", encoding="utf-8") as f:
                generated_world = json.load(f)
            
            all_places  = generated_world.get("entities", {}).get("places", [])
            all_objects  = generated_world.get("entities", {}).get("objects", [])
            all_npcs     = generated_world.get("entities", {}).get("npcs", [])

            # --- trim excess beyond requested counts ---
            # Collect object IDs referenced by NPCs (inventory, loot, etc.)
            referenced_obj_ids = set()
            for npc in all_npcs:
                for field in ("inventory", "objects", "loot", "drops"):
                    for ref in (npc.get(field) or []):
                        ref_id = ref if isinstance(ref, str) else ref.get("id", "")
                        referenced_obj_ids.add(ref_id)
                # craft recipes reference materials
                for obj in all_objects:
                    for mat_id in (obj.get("craft") or {}).get("materials", {}):
                        referenced_obj_ids.add(mat_id)

            if len(all_places) > num_places:
                print(f"  Trimming places: {len(all_places)} → {num_places}")
                all_places = all_places[:num_places]
            if len(all_objects) > num_objects:
                print(f"  Trimming objects: {len(all_objects)} → {num_objects}")
                # Keep referenced objects, trim from the unreferenced ones
                referenced = [o for o in all_objects if o.get("id") in referenced_obj_ids]
                unreferenced = [o for o in all_objects if o.get("id") not in referenced_obj_ids]
                keep_unreferenced = max(0, num_objects - len(referenced))
                all_objects = referenced + unreferenced[:keep_unreferenced]
            if len(all_npcs) > num_npcs:
                print(f"  Trimming NPCs: {len(all_npcs)} → {num_npcs}")
                all_npcs = all_npcs[:num_npcs]

            # write the trimmed version back
            generated_world["entities"]["places"] = all_places
            generated_world["entities"]["objects"] = all_objects
            generated_world["entities"]["npcs"]    = all_npcs
            with open(abs_world_path, "w", encoding="utf-8") as f:
                json.dump(generated_world, f, indent=2, ensure_ascii=False)

            new_entities = {
                "places": all_places,
                "objects": all_objects,
                "npcs": all_npcs
            }

            print("\nValidating generated entities...")
            print(f"  Final counts → places: {len(all_places)}, "
                  f"objects: {len(all_objects)}, "
                  f"npcs: {len(all_npcs)}")

            # In replace mode, validate against an empty base (all entities are new)
            empty_world = {"entities": {"places": [], "objects": [], "npcs": []}}
            rules_info = extract_game_rules(game_name)
            valid, errors = validate_entities(new_entities, empty_world, rules_info)
            if not valid:
                print("Validation errors found:")
                for error in errors:
                    print(f"  Error: {error}")
                # non-fatal: try to continue with testing anyway
            
            print("Checking difficulty balance...")
            balanced, issues = check_difficulty_balance(new_entities, max_level)
            if not balanced:
                print("Balance issues found (non-fatal):")
                for issue in issues:
                    print(f"  Issue: {issue}")
            
            print("Checking crafting material coverage...")
            empty_world_for_mat = {"entities": {"places": [], "objects": [], "npcs": []}}
            material_coverage, material_issues = check_crafting_material_coverage(new_entities, empty_world_for_mat, max_level)
            if not material_coverage:
                print("Material coverage issues found (non-fatal):")
                for issue in material_issues:
                    print(f"  Issue: {issue}")
        except Exception as e:
            print(f"Error during validation: {e}")
            continue
        
        print("\nTesting generated entities...")
        success, output = test_game(game_name, world_definition_path, test_timeout)
        
        if success:
            print("\n✓ Entities generated and tested successfully!")
            return True, "Entities added successfully"
        
        print(f"\n✗ Test failed. Error:\n{output[-2000:]}")
        
        fix_prompt = f"""Fix the errors in {world_definition_path}.

The file is fully indented JSON (2-space indent, one field per line).
Copy lines EXACTLY as they appear in the file when writing SEARCH/REPLACE blocks.

ERROR OUTPUT:
{output[-3000:]}

INSTRUCTIONS:
- Fix any JSON syntax errors
- Ensure all object/NPC IDs are unique
- Ensure craft ingredients reference valid object IDs
- Fix any missing required fields
- Ensure crafting materials cover all levels and areas
- Ensure every crafting material is used in at least one recipe

TEST COMMAND:
python eval.py --agent RandomAgent --max_steps 500 --game_name {game_name} --world_definition_path {world_definition_path}
"""
        cmd[cmd.index("--message") + 1] = fix_prompt
    
    print(f"\nFailed to generate valid entities after {max_iterations} iterations")
    return False, "Max iterations reached"

def _is_under_assets_base(path_str: str) -> bool:
    # check if path is under base assets directories
    p = _resolve_path(path_str).resolve()
    for base_sub in ("world_definitions", "env_configs"):
        base_dir = (Path(current_directory) / "assets" / base_sub / "base").resolve()
        try:
            p.relative_to(base_dir)
            return True
        except Exception:
            pass
    return False

def print_generation_summary(new_entities: Dict[str, Any], world_data: Dict[str, Any]):
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    
    places = new_entities.get("places", [])
    objects = new_entities.get("objects", [])
    npcs = new_entities.get("npcs", [])
    
    print(f"\nPlaces generated: {len(places)}")
    for place in places:
        areas = place.get("areas", [])
        area_levels = [a.get("level", "?") for a in areas]
        print(f"  - {place.get('name')} ({len(areas)} areas, levels: {area_levels})")
    
    print(f"\nObjects generated: {len(objects)}")
    by_category = {}
    for obj in objects:
        cat = obj.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    for cat, count in sorted(by_category.items()):
        print(f"  - {cat}: {count}")
    
    by_level = {}
    for obj in objects:
        level = obj.get("level", 1)
        by_level[level] = by_level.get(level, 0) + 1
    print(f"  Level distribution: {dict(sorted(by_level.items()))}")
    
    print(f"\nNPCs generated: {len(npcs)}")
    merchants = [n for n in npcs if n.get("role") == "merchant"]
    enemies = [n for n in npcs if n.get("enemy", False)]
    print(f"  - Merchants: {len(merchants)}")
    print(f"  - Enemies: {len(enemies)}")
    
    if enemies:
        print("  Enemy stats:")
        for enemy in enemies:
            hp = enemy.get("base_hp", enemy.get("hp", "?"))
            atk = enemy.get("base_attack_power", enemy.get("attack_power", "?"))
            print(f"    - {enemy.get('name')}: HP={hp}, ATK={atk}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new world entities using LLM (rules-aware, balanced difficulty).")
    parser.add_argument("--game_name", type=str, required=True, help="Name of the generated game.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated assets by recopying from base.")
    parser.add_argument("--world_definition_path", type=str, default=None, help="Path to input world definition JSON. Defaults to generated game's default.json.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to output populated JSON. Defaults to generated game's populated.json.")
    parser.add_argument("--llm_name", type=str, default="gpt-5", help="LLM model to use for generation.")
    parser.add_argument("--llm_provider", type=str, default=None,
                        help="LLM provider: openai, azure, azure_openai, claude, gemini, vllm, huggingface.")
    parser.add_argument("--num_places", type=int, default=2, help="Number of places to generate.")
    parser.add_argument("--num_objects", type=int, default=10, help="Number of objects to generate.")
    parser.add_argument("--num_npcs", type=int, default=5, help="Number of NPCs to generate.")
    parser.add_argument("--max_level", type=int, default=5, help="Maximum level for entities.")
    parser.add_argument("--backend", type=str, default="llm", choices=["llm", "aider"], 
                        help="Backend to use: 'llm' for direct LLM generation, 'aider' for code-aware generation with testing.")
    parser.add_argument("--max_iterations", type=int, default=3, help="Max iterations for aider backend.")
    parser.add_argument("--test_timeout", type=int, default=120, help="Timeout for game testing (aider backend).")
    parser.add_argument("--skip_validation", action="store_true", help="Skip entity validation.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information.")
    parser.add_argument("--generate_graph", action="store_true",
                        help="Also generate graph connections between areas, respecting place lock/unlock status.")

    parser.add_argument(
        "--allow_base_mutation",
        action="store_true",
        help="Allow writing output into assets/base (disabled by default).",
    )

    args = parser.parse_args()

    ensure_generated_assets(args.game_name, overwrite=args.overwrite)

    default_world_path, default_out_path = get_default_world_paths(args.game_name)
    world_definition_path = args.world_definition_path or default_world_path
    output_path = args.output_path or default_out_path

    if _is_under_assets_base(output_path) and not args.allow_base_mutation:
        raise ValueError(
            f"Refusing to write to base assets path: {output_path}\n"
            f"Either change --output_path to assets/world_definitions/generated/{args.game_name}/..., "
            f"or pass --allow_base_mutation to explicitly allow this."
        )

    world_definition_abs = _resolve_path(world_definition_path)
    output_abs = _resolve_path(output_path)

    if not world_definition_abs.exists():
        raise FileNotFoundError(f"World definition file not found: {world_definition_abs}")

    with open(world_definition_abs, "r", encoding="utf-8") as f:
        world_data = json.load(f)
    print(f"Loaded world definition from {world_definition_path}")

    rules_info = extract_game_rules(args.game_name)
    print(f"\nGame Rules Analysis:")
    print(f"  Action rules: {len(rules_info['action_rules'])} ({[r['verb'] for r in rules_info['action_rules']]})")
    print(f"  Step rules: {len(rules_info['step_rules'])}")
    print(f"  Required categories: {rules_info['entity_requirements']['required_object_categories']}")
    print(f"  Required usages: {rules_info['entity_requirements']['required_object_usages']}")
    print(f"  Required NPC roles: {rules_info['entity_requirements']['required_npc_roles']}")
    
    difficulty = analyze_difficulty_progression(world_data)
    print(f"\nCurrent Difficulty Analysis:")
    print(f"  Level range: {difficulty['min_level']} - {difficulty['max_level']}")
    print(f"  Level distribution: {difficulty['level_distribution']}")

    print(f"\nGenerating {args.num_places} places, {args.num_objects} objects, {args.num_npcs} NPCs...")
    print(f"Using backend: {args.backend}, model: {args.llm_name}")
    
    if args.backend == "aider":
        success, message = generate_with_aider(
            game_name=args.game_name,
            world_definition_path=world_definition_path,
            num_places=args.num_places,
            num_objects=args.num_objects,
            num_npcs=args.num_npcs,
            max_level=args.max_level,
            llm_name=args.llm_name,
            max_iterations=args.max_iterations,
            test_timeout=args.test_timeout,
            llm_provider=args.llm_provider,
            generate_graph=args.generate_graph,
        )
        if success:
            print(f"\n✓ {message}")
        else:
            print(f"\n✗ {message}")
            sys.exit(1)
    else:
        new_entities = generate_entities(
            llm_name=args.llm_name,
            num_places=args.num_places,
            num_objects=args.num_objects,
            num_npcs=args.num_npcs,
            max_level=args.max_level,
            world_data=world_data,
            game_name=args.game_name,
            generate_graph=args.generate_graph,
        )

        print_generation_summary(new_entities, world_data)

        merged_data = merge_entities(world_data, new_entities)

        output_abs.parent.mkdir(parents=True, exist_ok=True)
        with open(output_abs, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        print(f"Saved populated world definition to {output_path}")
        print(
            f"Added {len(new_entities.get('places', []))} places, "
            f"{len(new_entities.get('objects', []))} objects, "
            f"{len(new_entities.get('npcs', []))} NPCs "
            f"to {output_path}"
        )
        
        print("\nTesting generated world...")
        success, output = test_game(args.game_name, str(output_abs), args.test_timeout)
        if success:
            print("✓ World loads and runs successfully!")
        else:
            print(f"✗ World test failed:\n{output[-1000:]}")
