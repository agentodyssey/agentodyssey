import os
import sys
import json
import random
import argparse
import re
from collections import deque
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from utils import build_choices_with_answer_idx
from agents.llm_agent_config import LLMAgentConfig

def compute_distances_from_spawn(area_instances: dict, spawn_area_id: str) -> dict[str, int]:
    graph: dict[str, set[str]] = {}
    for area_id, area_inst in area_instances.items():
        neighbors = (area_inst or {}).get("neighbors") or {}
        if isinstance(neighbors, dict):
            graph[str(area_id)] = set(str(n) for n in neighbors.keys())
        else:
            graph[str(area_id)] = set()
    
    # BFS from spawn
    distances: dict[str, int] = {}
    INF = 9999
    for area_id in graph:
        distances[area_id] = INF
    
    if spawn_area_id not in graph:
        return distances
    
    distances[spawn_area_id] = 0
    queue = deque([spawn_area_id])
    
    while queue:
        current = queue.popleft()
        current_dist = distances[current]
        for neighbor in graph.get(current, set()):
            if distances.get(neighbor, INF) > current_dist + 1:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    
    return distances

def compute_object_weight(obj: dict, area_distances: dict[str, int], max_distance: int = 10) -> float:
    obj_areas = obj.get("areas") or []
    
    if not obj_areas:
        min_dist = 0
    else:
        min_dist = min(
            (area_distances.get(str(area_id), 9999) for area_id in obj_areas),
            default=9999
        )

    weight = 1.0 / (1.0 + min_dist)
    return weight

def compute_area_weight(area_id: str, area_distances: dict[str, int]) -> float:
    dist = area_distances.get(str(area_id), 9999)
    weight = 1.0 / (1.0 + dist)
    return weight

def weighted_sample(items: list, weights: list[float], k: int) -> list:
    if not items or k <= 0:
        return []
    k = min(k, len(items))
    
    total_weight = sum(weights)
    if total_weight == 0:
        return random.sample(items, k)
    
    remaining_items = list(items)
    remaining_weights = list(weights)
    selected = []
    
    for _ in range(k):
        if not remaining_items:
            break
        total = sum(remaining_weights)
        if total == 0:
            idx = random.randint(0, len(remaining_items) - 1)
        else:
            r = random.uniform(0, total)
            cumulative = 0
            idx = 0
            for i, w in enumerate(remaining_weights):
                cumulative += w
                if r <= cumulative:
                    idx = i
                    break
        
        selected.append(remaining_items[idx])
        remaining_items.pop(idx)
        remaining_weights.pop(idx)
    
    return selected

def _json_array_from_text(text: str):
    if not text:
        return None
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(text[start : end + 1])
    except Exception:
        return None

def _remove_class_block(source: str, class_name: str) -> str:
    m = re.search(rf"^class\s+{re.escape(class_name)}\b.*?:\s*$", source, flags=re.MULTILINE)
    if not m:
        return source
    start = m.start()
    m2 = re.search(r"^class\s+\w+\b.*?:\s*$", source[m.end():], flags=re.MULTILINE)
    end = (m.end() + m2.start()) if m2 else len(source)
    return source[:start] + "\n" + source[end:]

def _validate_rule_mcq_item(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    q = item.get("question")
    choices = item.get("choices")
    answer = item.get("answer")
    if not isinstance(q, str) or not q.strip():
        return None
    if not isinstance(choices, list) or len(choices) < 2:
        return None
    normalized = [str(c).strip() for c in choices]
    if any(not c for c in normalized):
        return None
    if len(set(normalized)) != len(normalized):
        return None
    if not isinstance(answer, str) or not answer.strip():
        return None
    answer_str = answer.strip()
    if answer_str not in normalized:
        return None
    return {"question": q.strip(), "choices": normalized, "answer": answer_str}

def _generate_rule_questions(model, kind: str, code_text: str, num_questions: int = 20) -> list[dict]:
    example = {
        "question": "When you sell an item to a merchant, what fraction of the item's value do you receive?",
        "choices": [
            "You receive the full base value of the item.",
            "You receive 75% of the item's value.",
            "You receive 50% of the item's value.",
            "You receive 25% of the item's value.",
        ],
        "answer": "You receive 50% of the item's value.",
    }
    base_prompt = (
        f"You are writing a world knowledge test about a game world's dynamics based on {kind} rules.\n"
        "The test is for an AGENT that plays this survival/crafting game through text feedback.\n\n"
        "=== CRITICAL: QUESTIONS MUST BE ABOUT OBSERVABLE GAMEPLAY, NOT CODE ===\n"
        "The agent being tested is a powerful LLM (like GPT-5). It will easily guess based on:\n"
        "- Common game patterns (e.g., 'defending reduces damage' - too generic!)\n"
        "- Standard RPG mechanics (e.g., 'attack power comes from weapons')\n"
        "- Code implementation details (e.g., 'STEPS_PER_HOUR' - agent never sees this!)\n\n"
        "YOUR GOAL: Create questions that ONLY someone who played THIS SPECIFIC game can answer.\n\n"
        "=== ABSOLUTELY FORBIDDEN (agent CANNOT observe these) ===\n"
        "NEVER ask about:\n"
        "- Internal constants: 'STEPS_PER_HOUR', 'MAX_CACHE_VALUE', step counts, thresholds\n"
        "- Code variable names: 'ambush_prob_override', 'modifier_key', 'rhythm_index'\n"
        "- Hidden probabilities: spawn rates, ambush chances, shatter percentages\n"
        "- Time-based thresholds: 'after N steps', 'every M hours'\n"
        "- Internal modifiers: damage multipliers, probability adjustments\n"
        "- Implementation details: 'how the system decides', 'what condition triggers'\n\n"
        "=== WHAT THE AGENT ACTUALLY OBSERVES (ask about THESE) ===\n"
        "The agent ONLY sees text messages like:\n"
        "- 'You took 5 damage' → can calculate damage reduction\n"
        "- 'You received 50 coins' → can calculate sell fraction\n"
        "- 'Cannot store bag. Containers cannot be nested.' → learns the rule\n"
        "- 'Crafted iron_sword appeared on the ground.' → learns where items go\n"
        "- 'Key was consumed.' → learns keys are one-use\n"
        "- 'Cannot pick up, hands full (2/2).' → learns hand capacity\n"
        "- 'The goblin_warrior is defending/attacking/waiting.' → learns NPC states\n\n"
        "=== QUESTION CATEGORIES ===\n\n"
        "CATEGORY 1: VALUES FROM FEEDBACK (30%)\n"
        "Ask about values the agent calculates from numerical feedback:\n"
        "- 'What fraction of damage does defending block?' → agent sees damage dealt\n"
        "- 'What fraction of sell value do you receive?' → agent sees coins received\n"
        "- 'How many item stacks can you hold in hands?' → agent hits limit\n"
        "- 'What HP does attacking with low stamina cost?' → agent sees HP change\n\n"
        "CATEGORY 2: ORDER/PRIORITY RULES (20%)\n"
        "Ask which source is checked first (agent learns by observing what disappears):\n"
        "- 'When crafting, are ingredients consumed from hands or inventory first?'\n"
        "- 'When eating, where is food consumed from first?'\n"
        "- 'If holding multiple weapons, how is attack calculated?' (max? sum? average?)\n\n"
        "CATEGORY 3: NEGATIVE QUESTIONS (35%)\n"
        "Ask about features that DON'T exist but sound plausible:\n"
        "- 'What backstab bonus do you get for attacking from behind?' → 'None exists'\n"
        "- 'What critical hit chance do weapons have?' → 'There are no critical hits'\n"
        "- 'Do weapons break after many uses?' → 'No durability system exists'\n"
        "- 'Can you parry attacks with perfect timing?' → 'No parry mechanic'\n"
        "- 'What bonus for attacking at night?' → 'Time doesn't affect combat'\n"
        "- 'Can you enchant weapons?' → 'No enchanting system'\n"
        "- 'Do potions stack their effects?' → 'There are no potions' (if true)\n"
        "Distractors should describe plausible mechanics from Minecraft/Dark Souls/Skyrim.\n\n"
        "CATEGORY 4: COUNTER-INTUITIVE QUIRKS (15%)\n"
        "Unique mechanics that differ from typical games:\n"
        "- 'Where do crafted items appear?' → 'On the ground, not in hand'\n"
        "- 'Can you put a bag inside another bag?' → 'No, containers can't nest'\n"
        "- 'Are lockpicks consumed on failed attempts?' → 'Yes, always consumed'\n\n"
        "=== FORMAT REQUIREMENTS ===\n"
        "- Use natural language only - NO code terms, variable names, or constants\n"
        "- All 4 choices must be equally plausible\n"
        "- Questions must be answerable from gameplay feedback text\n"
        "- Distractors should be mechanics from other popular games\n\n"
        f"Return ONLY a JSON array of exactly {num_questions} objects.\n"
        "Each object must have keys: question (string), choices (array of 4 strings), answer (string, must match EXACTLY one of the choices).\n\n"
        f"Example format:\n{json.dumps(example)}\n\n"
        f"REFERENCE MATERIAL (extract ONLY observable behaviors, ignore internal constants):\n{code_text}\n"
    )

    out: list[dict] = []
    attempts = 0
    while len(out) < num_questions and attempts < 3:
        attempts += 1
        need = num_questions - len(out)
        prompt = base_prompt + (
            f"\nGenerate {need} more questions now. Return a JSON array only."
        )
        try:
            raw = model.generate(user_prompt=prompt, system_prompt="Return JSON only.").get("response", "")
        except Exception as e:
            print(f"LLM generation failed for {kind} rule questions: {e}")
            break
        arr = _json_array_from_text(raw)
        if not isinstance(arr, list):
            continue
        for it in arr:
            valid = _validate_rule_mcq_item(it) if isinstance(it, dict) else None
            if valid is None:
                continue
            correct_answer = valid["answer"]
            distractors = set(valid["choices"]) - {correct_answer}
            shuffled_choices, shuffled_answer_idx = build_choices_with_answer_idx(
                correct_answer,
                distractors,
                max_choices=len(valid["choices"]),
            )
            valid = {
                "question": valid["question"],
                "choices": shuffled_choices,
                "answer_idx": shuffled_answer_idx,
            }
            out.append(valid)
            if len(out) >= num_questions:
                break
    return out[:num_questions]

def _clean_obj_name(name: str) -> str:
    name = name.strip()
    name = name.strip("`\"'")
    name = re.sub(r"\s+", " ", name)
    return name

def _parse_llm_string_list(text: str) -> list[str]:
    if not text:
        return []
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
    except Exception:
        pass
    items: list[str] = []
    for line in re.split(r"[\n\r]+", text):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        parts = [p.strip() for p in line.split(",") if p.strip()]
        items.extend(parts if parts else [line])
    return items

def generate_plausible_ingredient_distractors(
    model,
    target_object_name: str,
    existing_object_names: set[str],
    num_distractors: int,
    excluded_names: set[str],
) -> set[str]:
    if num_distractors <= 0:
        return set()

    # provide examples of existing object names for naming conventions
    example_names = sorted(existing_object_names - excluded_names)[:30]
    request_n = max(num_distractors * 2, num_distractors + 3)
    
    prompt = (
        f"You are generating distractor choices for a multiple-choice question about crafting in a game.\n\n"
        f"The question asks: 'What ingredient is needed to craft the object {target_object_name}?'\n\n"
        f"CRITICAL: The distractors must be MORE PLAUSIBLE by common sense than the actual answer!\n"
        f"In this game, crafting recipes are often COUNTER-INTUITIVE. For example:\n"
        f"- A 'wooden_sword' might actually require 'thread' (not wood)\n"
        f"- A 'stone_plate' might require 'stone_slab' (not raw stone)\n"
        f"- A 'travel_map' might require 'paper' (but maybe not ink)\n\n"
        f"Your job: Generate ingredients that COMMON SENSE would suggest for '{target_object_name}',\n"
        f"so that someone who has NEVER played this game would pick your distractors over the real answer.\n\n"
        f"Generate exactly {request_n} FAKE ingredient names that:\n"
        f"1. Follow the naming convention of these existing game objects: {', '.join(example_names)}. Use underscores instead of spaces.\n"
        f"2. Are what a REASONABLE PERSON would expect as ingredients for '{target_object_name}'\n"
        f"3. Think: 'What would I guess if I had to craft this in real life?'\n"
        f"4. Are NOT any of these excluded names: {', '.join(sorted(excluded_names)[:20])}\n\n"
        f"Examples of good distractors (common-sense guesses):\n"
        f"- For 'iron_sword': 'iron_ingot', 'sword_hilt', 'leather_grip', 'blade_mold'\n"
        f"- For 'bread': 'flour', 'yeast', 'wheat_bundle', 'water'\n"
        f"- For 'wooden_bow': 'bow_string', 'flexible_wood', 'sinew', 'tree_branch'\n"
        f"- For 'leather_armor': 'tanned_leather', 'armor_padding', 'leather_straps', 'metal_buckle'\n\n"
        f"Return ONLY a JSON array of strings (no extra text)."
    )

    try:
        raw = model.generate(user_prompt=prompt, system_prompt="You generate JSON only.").get(
            "response", ""
        )
    except Exception as e:
        print(f"LLM generation failed for ingredient distractors: {e}")
        return set()

    new_names: set[str] = set()
    for item in _parse_llm_string_list(raw):
        cleaned = _clean_obj_name(str(item))
        if not cleaned:
            continue
        if cleaned in excluded_names or cleaned in new_names:
            continue
        if len(cleaned) > 60:
            continue
        new_names.add(cleaned)
        if len(new_names) >= num_distractors:
            break
    
    return new_names

def generate_thematic_area_distractors(
    model,
    query_area_name: str,
    existing_area_names: set[str],
    num_distractors: int,
    excluded_names: set[str],
) -> set[str]:
    if num_distractors <= 0:
        return set()

    example_names = sorted(existing_area_names - excluded_names)[:30]
    request_n = max(num_distractors * 2, num_distractors + 3)
    
    prompt = (
        f"You are generating distractor choices for a multiple-choice question about area connections in a game.\n\n"
        f"The question asks: 'What area is connected to the area {query_area_name}?'\n\n"
        f"CRITICAL: Generate areas that would LOGICALLY connect to '{query_area_name}' in real life!\n"
        f"In this game, area connections are often COUNTER-INTUITIVE. For example:\n"
        f"- 'armory' might connect to 'plain' (not 'barracks' or 'training_ground')\n"
        f"- 'library' might connect to 'marsh' (not 'study' or 'archives')\n"
        f"- 'forge' might connect to 'grove' (not 'mine' or 'smeltery')\n\n"
        f"Your job: Generate area names that COMMON SENSE would expect to connect to '{query_area_name}',\n"
        f"so someone who hasn't explored the game would pick your distractors over the real answer.\n\n"
        f"Generate exactly {request_n} FAKE area names that:\n"
        f"1. Follow the naming convention of these existing game areas: {', '.join(example_names)}. Use underscores instead of spaces.\n"
        f"2. Would LOGICALLY be adjacent to '{query_area_name}' in a real medieval/fantasy world\n"
        f"3. Think: 'If I were designing a realistic castle/world, what would connect to this area?'\n"
        f"4. Are NOT any of these excluded names: {', '.join(sorted(excluded_names)[:20])}\n\n"
        f"Examples of good distractors (common-sense connections):\n"
        f"- For 'forge': 'smeltery', 'coal_storage', 'anvil_room', 'metalworks', 'bellows_chamber'\n"
        f"- For 'library': 'study', 'archives', 'scriptorium', 'reading_room', 'scroll_vault'\n"
        f"- For 'armory': 'barracks', 'training_yard', 'weapons_hall', 'guard_room', 'arsenal'\n"
        f"- For 'harbor': 'docks', 'shipyard', 'warehouse', 'fish_remnantet', 'pier'\n\n"
        f"Return ONLY a JSON array of strings (no extra text)."
    )

    try:
        raw = model.generate(user_prompt=prompt, system_prompt="You generate JSON only.").get(
            "response", ""
        )
    except Exception as e:
        print(f"LLM generation failed for area distractors: {e}")
        return set()

    new_names: set[str] = set()
    for item in _parse_llm_string_list(raw):
        cleaned = _clean_obj_name(str(item))
        if not cleaned:
            continue
        if cleaned in excluded_names or cleaned in new_names:
            continue
        if len(cleaned) > 60:
            continue
        new_names.add(cleaned)
        if len(new_names) >= num_distractors:
            break
    
    return new_names

def generate_plausible_area_distractors_for_object(
    model,
    object_name: str,
    existing_area_names: set[str],
    num_distractors: int,
    excluded_names: set[str],
) -> set[str]:
    if num_distractors <= 0:
        return set()

    example_names = sorted(existing_area_names - excluded_names)[:30]
    request_n = max(num_distractors * 2, num_distractors + 3)
    
    prompt = (
        f"You are generating distractor choices for a multiple-choice question about object distribution in a game.\n\n"
        f"The question asks: 'Where can the {object_name} be found in the world?'\n\n"
        f"CRITICAL: Generate area names where COMMON SENSE would expect to find '{object_name}'!\n"
        f"In this game, object locations are often COUNTER-INTUITIVE. For example:\n"
        f"- 'iron_ore' might be found in 'garden' (not 'mine' or 'cave')\n"
        f"- 'ancient_tome' might spawn in 'stable' (not 'library' or 'study')\n"
        f"- 'fresh_fish' might appear in 'armory' (not 'harbor' or 'river')\n\n"
        f"Your job: Generate area names where COMMON SENSE would expect '{object_name}' to spawn,\n"
        f"so someone who hasn't played the game would pick your distractors over the real answer.\n\n"
        f"Generate exactly {request_n} FAKE area names that:\n"
        f"1. Follow the naming convention of these existing game areas: {', '.join(example_names)}. Use underscores instead of spaces.\n"
        f"2. Would LOGICALLY contain '{object_name}' in a real medieval/fantasy world\n"
        f"3. Think: 'If I were looking for {object_name} in real life, where would I search?'\n"
        f"4. Are NOT any of these excluded names: {', '.join(sorted(excluded_names)[:20])}\n\n"
        f"Examples of good distractors (common-sense locations):\n"
        f"- For 'iron_sword': 'armory', 'blacksmith', 'weapons_hall', 'barracks', 'guard_post'\n"
        f"- For 'bread': 'bakery', 'kitchen', 'tavern', 'dining_hall', 'remnantet'\n"
        f"- For 'ancient_scroll': 'library', 'archive', 'monastery', 'wizard_tower', 'temple'\n"
        f"- For 'healing_potion': 'apothecary', 'healer_hut', 'hospital', 'alchemy_lab', 'herb_shop'\n\n"
        f"Return ONLY a JSON array of strings (no extra text)."
    )

    try:
        raw = model.generate(user_prompt=prompt, system_prompt="You generate JSON only.").get(
            "response", ""
        )
    except Exception as e:
        print(f"LLM generation failed for object distribution area distractors: {e}")
        return set()

    new_names: set[str] = set()
    for item in _parse_llm_string_list(raw):
        cleaned = _clean_obj_name(str(item))
        if not cleaned:
            continue
        if cleaned in excluded_names or cleaned in new_names:
            continue
        if len(cleaned) > 60:
            continue
        new_names.add(cleaned)
        if len(new_names) >= num_distractors:
            break
    
    return new_names

def generate_plausible_npc_inventory_distractors(
    model,
    npc_name: str,
    existing_object_names: set[str],
    num_distractors: int,
    excluded_names: set[str],
) -> set[str]:
    if num_distractors <= 0:
        return set()

    example_names = sorted(existing_object_names - excluded_names)[:30]
    request_n = max(num_distractors * 2, num_distractors + 3)
    
    prompt = (
        f"You are generating distractor choices for a multiple-choice question about NPC inventory in a game.\n\n"
        f"The question asks: 'What object is held by the NPC {npc_name}?'\n\n"
        f"CRITICAL: Generate object names that COMMON SENSE would expect '{npc_name}' to carry!\n"
        f"In this game, NPC inventories are often COUNTER-INTUITIVE. For example:\n"
        f"- 'goblin_warrior' might carry 'flower_bouquet' (not 'rusty_sword' or 'shield')\n"
        f"- 'village_healer' might hold 'battle_axe' (not 'healing_potion' or 'bandages')\n"
        f"- 'blacksmith' might have 'ancient_tome' (not 'hammer' or 'iron_ingot')\n\n"
        f"Your job: Generate object names that COMMON SENSE would expect '{npc_name}' to carry,\n"
        f"so someone who hasn't played the game would pick your distractors over the real answer.\n\n"
        f"Generate exactly {request_n} FAKE object names that:\n"
        f"1. Follow the naming convention of these existing game objects: {', '.join(example_names)}. Use underscores instead of spaces.\n"
        f"2. Would LOGICALLY be carried by '{npc_name}' based on their role/profession\n"
        f"3. Think: 'If I met a {npc_name} in a medieval/fantasy world, what would they be carrying?'\n"
        f"4. Are NOT any of these excluded names: {', '.join(sorted(excluded_names)[:20])}\n\n"
        f"Examples of good distractors (common-sense items for NPCs):\n"
        f"- For 'goblin_warrior': 'crude_sword', 'wooden_club', 'bone_dagger', 'leather_shield', 'tribal_mask'\n"
        f"- For 'village_healer': 'healing_salve', 'herb_pouch', 'medicine_bottle', 'bandage_roll', 'mortar_pestle'\n"
        f"- For 'royal_guard': 'steel_sword', 'tower_shield', 'plate_armor', 'royal_insignia', 'guard_helmet'\n"
        f"- For 'traveling_merchant': 'coin_purse', 'trade_goods', 'merchant_ledger', 'sample_wares', 'travel_pack'\n\n"
        f"Return ONLY a JSON array of strings (no extra text)."
    )

    try:
        raw = model.generate(user_prompt=prompt, system_prompt="You generate JSON only.").get(
            "response", ""
        )
    except Exception as e:
        print(f"LLM generation failed for NPC inventory distractors: {e}")
        return set()

    new_names: set[str] = set()
    for item in _parse_llm_string_list(raw):
        cleaned = _clean_obj_name(str(item))
        if not cleaned:
            continue
        if cleaned in excluded_names or cleaned in new_names:
            continue
        if len(cleaned) > 60:
            continue
        new_names.add(cleaned)
        if len(new_names) >= num_distractors:
            break
    
    return new_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", type=str, default="remnant")
    parser.add_argument("--world_definition_path", type=str, default="assets/world_definitions/generated/remnant/default.json")
    parser.add_argument("--env_config_path", type=str, default="output/game_remnant/gpt-5/LongContextAgent/no_extras/config.json")
    parser.add_argument("--action_rules_path", type=str, default="games/generated/remnant/rules/action_rules.py")
    parser.add_argument("--step_rules_path", type=str, default="games/generated/remnant/rules/step_rules/general.py")
    parser.add_argument("--regenerate", action="store_true", help="Whether to regenerate the world knowledge questions.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of questions to sample for testing.")
    parser.add_argument("--full", action="store_true", help="Generate ALL possible questions (no sampling) except action/step rules which get 60 each.")
    parser.add_argument("--augment_candidates", action="store_true", help="Use an LLM to generate extra object/area name candidates for distractors.")
    parser.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "huggingface", "vllm", "azure", "azure_openai", "claude", "gemini"])
    parser.add_argument("--llm_name", type=str, default="gpt-5-mini")
    args = parser.parse_args()

    llm_cfg = LLMAgentConfig(llm_name=args.llm_name, llm_provider=args.llm_provider)

    output_path = os.path.join("output", "game_" + args.game_name, "world_knowledge_qa_sampled.json")

    if not args.regenerate and os.path.exists(output_path):
        print(f"World knowledge test already exists at {output_path}. Use --regenerate to regenerate.")
    else:
        print("Generating world knowledge test...")
        qa = {
            "craft_ingredient": [],
            "craft_ingredient_quantity": [],
            "npc_holds_object": [],
            "object_distribution": [],
            "spatial_connection": [],
            "action_rules": [],
            "step_rules": [],
        }
        with open(args.world_definition_path, "r") as f:
            world_definitions = json.load(f)
        places = world_definitions["entities"].get("places", [])
        area_id_to_area_name: dict[str, str] = {}
        for place in places:
            place_name = place.get("name")
            if not place_name:
                continue
            for area in place.get("areas", []) or []:
                area_id = area.get("id")
                area_name = area.get("name")
                if area_id and area_name:
                    area_id_to_area_name[str(area_id)] = str(area_name)
        undistributable_objects: set[str] = set(
            world_definitions.get("initializations", {}).get("undistributable_objects", [])
            or []
        )
        # aggregate candidates
        candidates = {
            "obj_name": set(),
            "area_name": set(),
        }
        entities = world_definitions["entities"]
        for obj in entities["objects"]:
            candidates["obj_name"].add(obj["name"])
        for place in places:
            for area in place.get("areas", []) or []:
                if area.get("name"):
                    candidates["area_name"].add(str(area["name"]))
        
        # load env config early to compute area distances for weighted sampling
        with open(args.env_config_path, "r") as f:
            env_config = json.load(f)
        area_instances = (env_config.get("world") or {}).get("area_instances") or {}
        
        # compute distances from spawn area (castle hall) to all areas
        SPAWN_AREA_ID = "area_castle_hall"
        area_distances = compute_distances_from_spawn(area_instances, SPAWN_AREA_ID)
        print(f"Area distances from spawn ({SPAWN_AREA_ID}):")
        for area_id, dist in sorted(area_distances.items(), key=lambda x: x[1]):
            print(f"  {area_id}: {dist}")
        
        # compute weights for all objects based on their nearest area to spawn
        object_weights: dict[str, float] = {}
        for obj in entities["objects"]:
            obj_id = str(obj.get("id", ""))
            if obj_id:
                object_weights[obj_id] = compute_object_weight(obj, area_distances)
        
        # compute weights for all areas
        area_weights: dict[str, float] = {}
        for area_id in area_instances.keys():
            area_weights[area_id] = compute_area_weight(area_id, area_distances)
        
        print(f"Object weights (sample):")
        for obj_id, weight in sorted(object_weights.items(), key=lambda x: -x[1])[:30]:
            print(f"  {obj_id}: {weight:.3f}")

        # type 1: ingredients needed to craft each object (weighted sampling)
        distractor_model = None
        if args.augment_candidates:
            try:
                distractor_model = llm_cfg.get_llm()
            except Exception as e:
                print(f"Warning: Could not initialize LLM for distractor generation: {e}")
                distractor_model = None
        
        craft_ingredient_candidates = []
        for obj in entities["objects"]:
            obj_id = str(obj.get("id", ""))
            obj_name = obj["name"]
            ingredients = obj["craft"]["ingredients"]
            weight = object_weights.get(obj_id, 0.01)
            
            if not ingredients:
                craft_ingredient_candidates.append({
                    "obj": obj,
                    "weight": weight,
                    "answer": "none",
                    "ingredient_names": set(),
                })
            else:
                craft_ingredient_candidates.append({
                    "obj": obj,
                    "weight": weight,
                    "ingredients": ingredients,
                })
        
        num_craft_questions = len(craft_ingredient_candidates) if args.full else min(args.num_samples, len(craft_ingredient_candidates))
        if args.full:
            sampled_craft_items = craft_ingredient_candidates
        else:
            sampled_craft_items = weighted_sample(
                craft_ingredient_candidates,
                [c["weight"] for c in craft_ingredient_candidates],
                num_craft_questions
            )
        
        print(f"Generating {len(sampled_craft_items)} craft_ingredient questions...")
        for idx, item in enumerate(sampled_craft_items):
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  Processing craft_ingredient {idx + 1}/{len(sampled_craft_items)}...")
            obj = item["obj"]
            obj_name = obj["name"]
            
            # Hard-coded: 5 LLM-generated distractors + 4 environment distractors = 9 total
            NUM_LLM_DISTRACTORS = 5
            NUM_ENV_DISTRACTORS = 4
            
            if "answer" in item and item["answer"] == "none":
                answer = "none"
                existing_distractor_pool = candidates["obj_name"]
                excluded_names = {answer}
            else:
                ingredients = item["ingredients"]
                ingredient_id = random.sample(list(ingredients.keys()), 1)[0]
                ingredient_name = next(
                    (o["name"] for o in entities["objects"] if o["id"] == ingredient_id),
                    ingredient_id
                )
                answer = ingredient_name
                ingredient_names = {
                    next((o["name"] for o in entities["objects"] if o["id"] == ing_id), ing_id)
                    for ing_id in ingredients.keys()
                }
                existing_distractor_pool = candidates["obj_name"] - ingredient_names - {answer}
                excluded_names = ingredient_names | {answer}
            
            # Generate LLM distractors for both "none" and regular cases
            llm_distractors = set()
            if distractor_model is not None:
                # Keep retrying until we get exactly NUM_LLM_DISTRACTORS, accumulating results
                max_retries = 5
                all_llm_distractors = set()
                for attempt in range(max_retries):
                    new_distractors = generate_plausible_ingredient_distractors(
                        model=distractor_model,
                        target_object_name=obj_name,
                        existing_object_names=candidates["obj_name"],
                        num_distractors=NUM_LLM_DISTRACTORS * 2,  # Request more to ensure we get enough
                        excluded_names=excluded_names | all_llm_distractors,
                    )
                    all_llm_distractors |= new_distractors
                    if len(all_llm_distractors) >= NUM_LLM_DISTRACTORS:
                        break
                # Take exactly NUM_LLM_DISTRACTORS (or all if fewer)
                llm_distractors = set(list(all_llm_distractors)[:NUM_LLM_DISTRACTORS])
                if len(llm_distractors) < NUM_LLM_DISTRACTORS:
                    print(f"[env hint] LLM failed to generate {NUM_LLM_DISTRACTORS} distractors for '{obj_name}' after {max_retries} attempts (got {len(llm_distractors)})")
            
            # Sample exactly NUM_ENV_DISTRACTORS from environment
            existing_distractor_list = list(existing_distractor_pool - llm_distractors)
            if len(existing_distractor_list) >= NUM_ENV_DISTRACTORS:
                existing_distractors = set(random.sample(existing_distractor_list, NUM_ENV_DISTRACTORS))
            else:
                existing_distractors = set(existing_distractor_list)
            
            distractor = llm_distractors | existing_distractors
            choices, answer_idx = build_choices_with_answer_idx(answer, distractor, max_choices=10)
            qa["craft_ingredient"].append({
                "question": f"What ingredient is needed to craft the object {obj_name}?",
                "choices": choices,
                "answer_idx": answer_idx,
            })
        # type 2: quantity of a specific ingredient needed to craft each object (weighted sampling)
        craft_quantity_candidates = []
        for obj in entities["objects"]:
            obj_id = str(obj.get("id", ""))
            ingredients = obj["craft"]["ingredients"]
            if not ingredients:
                continue
            weight = object_weights.get(obj_id, 0.01)
            craft_quantity_candidates.append({
                "obj": obj,
                "weight": weight,
                "ingredients": ingredients,
            })
        
        num_quantity_questions = len(craft_quantity_candidates) if args.full else min(args.num_samples, len(craft_quantity_candidates))
        if args.full:
            sampled_quantity_items = craft_quantity_candidates
        else:
            sampled_quantity_items = weighted_sample(
                craft_quantity_candidates,
                [c["weight"] for c in craft_quantity_candidates],
                num_quantity_questions
            )
        
        for item in sampled_quantity_items:
            obj = item["obj"]
            obj_name = obj["name"]
            ingredients = item["ingredients"]
            ingredient_id = random.sample(list(ingredients.keys()), 1)[0]
            ingredient_name = next(
                (o["name"] for o in entities["objects"] if o["id"] == ingredient_id),
                ingredient_id
            )
            quantity = ingredients[ingredient_id]
            choices_excluding_quantity = [q for q in [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100] if q != quantity]
            distractor = {str(x) for x in choices_excluding_quantity}
            choices, answer_idx = build_choices_with_answer_idx(str(quantity), distractor, max_choices=10)
            qa["craft_ingredient_quantity"].append({
                "question": f"What quantity of the ingredient {ingredient_name} is needed to craft the object {obj_name}?",
                "choices": choices,
                "answer_idx": answer_idx,
            })
        
        # type 3: objects held by specific NPCs
        # only include NPCs from the specified whitelist (unless --full is set)
        INCLUDED_NPC_IDS = {
            "npc_goblin_warrior",
            "npc_swamp_raider", 
            "npc_cave_stalker",
            "npc_peak_vulture",
            "npc_goblin_chief",
        }
        
        for npc in entities["npcs"]:
            npc_id = npc.get("id", "")
            if not args.full and npc_id not in INCLUDED_NPC_IDS:
                continue
            if "objects" not in npc or not npc["objects"]:
                continue
            npc_name = npc["name"]
            obj = random.sample(npc["objects"], 1)[0]
            obj_name = next(
                (o["name"] for o in entities["objects"] if o["id"] == obj),
                str(obj)
            )
            
            # use half LLM-generated plausible distractors and half existing object distractors
            existing_distractors = candidates["obj_name"] - {obj}
            
            if distractor_model:
                num_existing = min(4, len(existing_distractors))
                num_llm_generated = 5
                
                sampled_existing = set(random.sample(list(existing_distractors), num_existing)) if existing_distractors else set()
                
                excluded_for_llm = sampled_existing | {obj}
                llm_distractors = generate_plausible_npc_inventory_distractors(
                    model=distractor_model,
                    npc_name=npc_name,
                    existing_object_names=candidates["obj_name"],
                    num_distractors=num_llm_generated,
                    excluded_names=excluded_for_llm,
                )
                
                combined_distractors = sampled_existing | llm_distractors
            else:
                combined_distractors = existing_distractors
            
            choices, answer_idx = build_choices_with_answer_idx(obj, combined_distractors, max_choices=10)
            qa["npc_holds_object"].append({
                "question": f"What object is held by the NPC {npc_name}?",
                "choices": choices,
                "answer_idx": answer_idx,
            })
        
        # type 4: object distribution (weighted sampling with 80/20 split)
        # objects WITH areas field = 80% of questions
        distribution_candidates: set[str] = set(candidates.get("area_name", set())) | {"none"}
        
        objects_with_areas = []
        objects_without_areas = []
        for obj in entities["objects"]:
            obj_id = str(obj.get("id", ""))
            weight = object_weights.get(obj_id, 0.01)
            obj_areas = obj.get("areas")
            if obj_areas:
                covered_areas = {
                    area_id_to_area_name.get(str(area_id))
                    for area_id in obj_areas
                    if area_id_to_area_name.get(str(area_id))
                }
                if covered_areas:
                    objects_with_areas.append({
                        "obj": obj,
                        "weight": weight,
                        "covered_areas": covered_areas,
                    })
                else:
                    objects_without_areas.append({
                        "obj": obj,
                        "weight": weight,
                    })
            else:
                objects_without_areas.append({
                    "obj": obj,
                    "weight": weight,
                })
        
        print(f"Object distribution: {len(objects_with_areas)} with specific areas, {len(objects_without_areas)} without areas (none)")
        
        # calculate 80/20 split for sampling
        if args.full:
            sampled_with_areas = objects_with_areas
            sampled_without_areas = objects_without_areas
        else:
            total_questions = args.num_samples
            num_with_areas = min(int(total_questions * 0.8), len(objects_with_areas))
            num_without_areas = min(total_questions - num_with_areas, len(objects_without_areas))
            
            sampled_with_areas = weighted_sample(
                objects_with_areas,
                [c["weight"] for c in objects_with_areas],
                num_with_areas
            )
            sampled_without_areas = weighted_sample(
                objects_without_areas,
                [c["weight"] for c in objects_without_areas],
                num_without_areas
            )
        
        print(f"Sampled {len(sampled_with_areas)} with areas + {len(sampled_without_areas)} without areas")
        
        # generate questions for objects WITH specific areas (answer is an area name)
        for item in sampled_with_areas:
            obj = item["obj"]
            obj_name = obj["name"]
            covered_areas = item["covered_areas"]
            distribution_answer = random.sample(list(covered_areas), 1)[0]

            existing_distractors = distribution_candidates - covered_areas
            
            if distractor_model:
                other_existing = existing_distractors - {"none"}
                num_other_existing = min(3, len(other_existing))
                num_llm_generated = 5
                
                sampled_existing = set(random.sample(list(other_existing), num_other_existing)) if other_existing else set()
                sampled_existing.add("none")  # Always include "none" as distractor
                
                excluded_for_llm = covered_areas | sampled_existing | {distribution_answer}
                llm_distractors = generate_plausible_area_distractors_for_object(
                    model=distractor_model,
                    object_name=obj_name,
                    existing_area_names=set(candidates.get("area_name", set())),
                    num_distractors=num_llm_generated,
                    excluded_names=excluded_for_llm,
                )
                
                combined_distractors = sampled_existing | llm_distractors
            else:
                combined_distractors = existing_distractors
            
            choices, answer_idx = build_choices_with_answer_idx(distribution_answer, combined_distractors, max_choices=10)
            qa["object_distribution"].append(
                {
                    "question": (
                        f"How is the {obj_name} distributed in the world? "
                        "Answer with an area name, or 'none' (meaning it does not spawn anywhere in the world)."
                    ),
                    "choices": choices,
                    "answer_idx": answer_idx,
                }
            )
        
        # generate questions for objects WITHOUT areas (answer is "none")
        for item in sampled_without_areas:
            obj = item["obj"]
            obj_name = obj["name"]
            distribution_answer = "none"
            
            existing_distractors = distribution_candidates - {distribution_answer}
            
            if distractor_model:
                num_existing = min(4, len(existing_distractors))
                num_llm_generated = 5
                
                sampled_existing = set(random.sample(list(existing_distractors), num_existing)) if existing_distractors else set()
                
                excluded_for_llm = sampled_existing | {distribution_answer}
                llm_distractors = generate_plausible_area_distractors_for_object(
                    model=distractor_model,
                    object_name=obj_name,
                    existing_area_names=set(candidates.get("area_name", set())),
                    num_distractors=num_llm_generated,
                    excluded_names=excluded_for_llm,
                )
                
                combined_distractors = sampled_existing | llm_distractors
            else:
                combined_distractors = existing_distractors
            
            choices, answer_idx = build_choices_with_answer_idx(distribution_answer, combined_distractors, max_choices=10)
            qa["object_distribution"].append(
                {
                    "question": (
                        f"How is the {obj_name} distributed in the world? "
                        "Answer with an area name, or 'none' (meaning it does not spawn anywhere in the world)."
                    ),
                    "choices": choices,
                    "answer_idx": answer_idx,
                }
            )
        
        # type 5: spatial connections (weighted sampling based on area distance)
        spatial_candidates = []
        for area_id, area_inst in area_instances.items():
            neighbors = (area_inst or {}).get("neighbors") or {}
            if not isinstance(neighbors, dict) or not neighbors:
                continue
            weight = area_weights.get(str(area_id), 0.01)
            spatial_candidates.append({
                "area_id": area_id,
                "area_inst": area_inst,
                "neighbors": neighbors,
                "weight": weight,
            })
        
        num_spatial_questions = len(spatial_candidates) if args.full else min(10, len(spatial_candidates))
        if args.full:
            sampled_spatial_items = spatial_candidates
        else:
            sampled_spatial_items = weighted_sample(
                spatial_candidates,
                [c["weight"] for c in spatial_candidates],
                num_spatial_questions
            )
        
        print(f"Spatial connection test: sampled {len(sampled_spatial_items)} areas using weighted sampling")
        
        area_distractor_model = None
        if args.augment_candidates:
            try:
                area_distractor_model = llm_cfg.get_llm()
            except Exception as e:
                print(f"Warning: Could not initialize LLM for area distractor generation: {e}")
                area_distractor_model = None
        
        for idx, item in enumerate(sampled_spatial_items):
            if args.augment_candidates and ((idx + 1) % 10 == 0 or idx == 0):
                print(f"  Processing spatial_connection {idx + 1}/{len(sampled_spatial_items)}...")
            area_id = item["area_id"]
            area_inst = item["area_inst"]
            neighbors = item["neighbors"]
            
            query_area_name = area_id_to_area_name.get(
                str(area_id), str((area_inst or {}).get("name") or area_id)
            )
            neighbor_ids = list(neighbors.keys())
            neighbor_names: set[str] = set()
            for n_id in neighbor_ids:
                n_id_str = str(n_id)
                n_inst = area_instances.get(n_id_str) or {}
                n_name = area_id_to_area_name.get(n_id_str) or n_inst.get("name") or n_id_str
                neighbor_names.add(str(n_name))

            neighbor_names.discard(query_area_name)
            if not neighbor_names:
                continue
            answer = random.sample(list(neighbor_names), 1)[0]
            
            existing_distractor_pool = set(candidates.get("area_name", set()))
            existing_distractor_pool.discard(query_area_name)
            existing_distractor_pool -= neighbor_names
            existing_distractor_pool.discard(answer)
            
            total_distractors_needed = 9
            if area_distractor_model is not None:
                half_distractors = total_distractors_needed // 2
                llm_distractors = generate_thematic_area_distractors(
                    model=area_distractor_model,
                    query_area_name=query_area_name,
                    existing_area_names=candidates.get("area_name", set()),
                    num_distractors=half_distractors,
                    excluded_names=neighbor_names | {answer, query_area_name},
                )
            else:
                llm_distractors = set()
            
            existing_distractors_needed = total_distractors_needed - len(llm_distractors)
            existing_distractor_list = list(existing_distractor_pool - llm_distractors)
            if len(existing_distractor_list) >= existing_distractors_needed:
                existing_distractors = set(random.sample(existing_distractor_list, existing_distractors_needed))
            else:
                existing_distractors = set(existing_distractor_list)
            
            distractor_pool = llm_distractors | existing_distractors
            choices, answer_idx = build_choices_with_answer_idx(answer, distractor_pool, max_choices=10)
            qa["spatial_connection"].append(
                {
                    "question": f"What area is connected to the area {query_area_name}?",
                    "choices": choices,
                    "answer_idx": answer_idx,
                }
            )
        # type 6: action rules
        try:
            with open(args.action_rules_path, "r", encoding="utf-8") as f:
                action_rules_code = f.read()
        except Exception:
            action_rules_code = ""
        if action_rules_code.strip():
            num_action_rule_questions = 60 if args.full else 20
            print(f"Generating {num_action_rule_questions} action rules (world dynamics) questions with LLM...")
            rules_model = llm_cfg.get_llm()
            action_rule_qs = _generate_rule_questions(
                rules_model,
                kind="action",
                code_text=action_rules_code,
                num_questions=num_action_rule_questions,
            )
            qa["action_rules"].extend(action_rule_qs)
        # type 7: step rules
        try:
            with open(args.step_rules_path, "r", encoding="utf-8") as f:
                step_rules_code = f.read()
        except Exception:
            step_rules_code = ""
        if step_rules_code.strip():
            for cls in ("TutorialRoomStepRule", "MainQuestStepRule", "SideQuestStepRule"):
                step_rules_code = _remove_class_block(step_rules_code, cls)
            num_step_rule_questions = 60 if args.full else 20
            print(f"Generating {num_step_rule_questions} step rules (world dynamics) questions with LLM...")
            rules_model = llm_cfg.get_llm()
            step_rule_qs = _generate_rule_questions(
                rules_model,
                kind="step",
                code_text=step_rules_code,
                num_questions=num_step_rule_questions,
            )
            qa["step_rules"].extend(step_rule_qs)

        # save world knowledge test
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(qa, f, indent=2)
        print(f"World knowledge test saved to {output_path}")