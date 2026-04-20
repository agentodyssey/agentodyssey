import os
import sys
import json
import random
import re
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from utils import build_choices_with_answer_idx

def _parse_drop_or_discard_action(action: str) -> tuple[str, str] | None:
    if not action:
        return None
    m = re.match(r"^(drop|discard)\s+(.+?)\s*$", str(action).strip(), flags=re.IGNORECASE)
    if not m:
        return None
    verb = m.group(1).lower().strip()
    obj = m.group(2).strip()
    if not obj:
        return None
    return verb, obj

def _parse_current_time(obs_text: str) -> str | None:
    if not obs_text:
        return None
    m = re.search(r"^Current Time:\s*(.+?)\s*$", obs_text, re.MULTILINE)
    return m.group(1).strip() if m else None

def _parse_current_location(obs_text: str) -> str | None:
    if not obs_text:
        return None
    m = re.search(r"^Current Location:\s*(.+?)\s*$", obs_text, re.MULTILINE)
    return m.group(1).strip() if m else None

def _safe_sample(population: list[str], k: int) -> list[str]:
    if k <= 0 or not population:
        return []
    if len(population) <= k:
        return list(population)
    return random.sample(population, k)

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_agent_log(path: str) -> list[dict]:
    entries: list[dict] = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict) or not obj:
                continue
            if "observation" not in obj and "action" not in obj:
                continue
            payload = obj
            agent_id = obj.get("agent_id") or "agent_adam_davis"
            observation = payload.get("observation") or {}
            obs_text = (observation.get("text") or "").strip()
            step = observation.get("step")
            action = str(payload.get("action") or "").strip()
            entries.append(
                {
                    "agent_id": str(agent_id),
                    "step": step,
                    "time": _parse_current_time(obs_text),
                    "location": _parse_current_location(obs_text),
                    "action": action,
                }
            )
    def _step_key(e: dict):
        try:
            return int(e.get("step"))
        except Exception:
            return 10**18
    entries.sort(key=_step_key)
    return entries

def _extract_world_mappings(world_definitions: dict, env_config: dict):
    area_id_to_name: dict[str, str] = {}
    obj_id_to_name: dict[str, str] = {}
    npc_id_to_name: dict[str, str] = {}

    # From base world definition
    entities = (world_definitions.get("entities") or {})
    for obj in (entities.get("objects") or []):
        if isinstance(obj, dict) and obj.get("id") and obj.get("name"):
            obj_id_to_name[str(obj["id"])] = str(obj["name"])
    for npc in (entities.get("npcs") or []):
        if isinstance(npc, dict) and npc.get("id") and npc.get("name"):
            npc_id_to_name[str(npc["id"])] = str(npc["name"])
    for place in (entities.get("places") or []):
        for area in (place.get("areas") or []) if isinstance(place, dict) else []:
            if isinstance(area, dict) and area.get("id") and area.get("name"):
                area_id_to_name[str(area["id"])] = str(area["name"])

    # From env_config world snapshot (includes quest objects etc.)
    world = (env_config.get("world") or {})
    for obj_id, obj in (world.get("objects") or {}).items():
        if isinstance(obj, dict) and obj.get("name"):
            obj_id_to_name[str(obj_id)] = str(obj["name"])
    for npc_id, npc in (world.get("npcs") or {}).items():
        if isinstance(npc, dict) and npc.get("name"):
            npc_id_to_name[str(npc_id)] = str(npc["name"])
    for npc_inst_id, npc_inst in (world.get("npc_instances") or {}).items():
        if isinstance(npc_inst, dict) and npc_inst.get("name"):
            npc_id_to_name[str(npc_inst_id)] = str(npc_inst["name"])
    for area_id, area_inst in (world.get("area_instances") or {}).items():
        if isinstance(area_inst, dict) and area_inst.get("name"):
            area_id_to_name[str(area_id)] = str(area_inst["name"])

    area_names: set[str] = {str(v) for v in area_id_to_name.values() if str(v)}
    obj_names: set[str] = {str(v) for v in obj_id_to_name.values() if str(v)}
    npc_names: set[str] = {str(v) for v in npc_id_to_name.values() if str(v)}
    return area_id_to_name, obj_id_to_name, npc_id_to_name, area_names, obj_names, npc_names

def _npc_name_from_id(npc_id_or_inst: str, npc_id_to_name: dict[str, str]) -> str:
    if not npc_id_or_inst:
        return ""
    s = str(npc_id_or_inst)
    if s in npc_id_to_name:
        return npc_id_to_name[s]
    m = re.match(r"^(.*)_\d+$", s)
    if m:
        base = m.group(1)
        if base in npc_id_to_name:
            return npc_id_to_name[base]
    return s

def generate_episodic_memory_qa(world_definition_path, env_config_path, agent_log_path, num_samples_per_category=20, seed=0):
    random.seed(seed)

    print("Generating episodic memory test...")
    qa: dict = {
        "visited_area": [],
        "crafted_object": [],
        "killed_npc": [],
        "acquired_object": [],
        "action_at_time_location": [],
        "next_action": [],
        "drop_discard_location": [],
    }
    world_definitions = _load_json(world_definition_path)
    env_config = _load_json(env_config_path)
    agent_log = _load_agent_log(agent_log_path)

    (
        area_id_to_name,
        obj_id_to_name,
        npc_id_to_name,
        area_names,
        obj_names,
        npc_names,
    ) = _extract_world_mappings(world_definitions, env_config)

    curr_agents_state = env_config.get("curr_agents_state") or {}
    agent_ids: set[str] = set()
    for key in ("areas_visited", "objects_crafted", "npcs_killed", "objects_acquired"):
        agent_ids |= set((curr_agents_state.get(key) or {}).keys())
    if not agent_ids:
        agent_ids = {"agent_adam_davis"}

    all_area_candidates: set[str] = set(area_names) | {"none"}
    all_obj_candidates: set[str] = set(obj_names) | {"none"}
    all_npc_candidates: set[str] = set(npc_names) | {"none"}

    for agent_id in sorted(agent_ids):
        # 1) visited areas
        visited_area_ids = (curr_agents_state.get("areas_visited") or {}).get(agent_id) or []
        visited_area_names = {
            area_id_to_name.get(str(aid), str(aid)) for aid in visited_area_ids if str(aid)
        }
        visited_area_names.discard("")
        if visited_area_names:
            for answer in _safe_sample(sorted(visited_area_names), min(num_samples_per_category, len(visited_area_names))):
                distractor_pool = set(all_area_candidates) - visited_area_names
                choices, answer_idx = build_choices_with_answer_idx(answer, distractor_pool, max_choices=10)
                qa["visited_area"].append(
                    {"question": "Which area did you visit?", "choices": choices, "answer_idx": answer_idx}
                )
        else:
            choices, answer_idx = build_choices_with_answer_idx("none", all_area_candidates - {"none"}, max_choices=10)
            qa["visited_area"].append(
                {"question": "Which area did you visit?", "choices": choices, "answer_idx": answer_idx}
            )

        # 2) crafted objects
        crafted_obj_ids = (curr_agents_state.get("objects_crafted") or {}).get(agent_id) or {}
        crafted_obj_names = {
            obj_id_to_name.get(str(oid), str(oid)) for oid in crafted_obj_ids.keys() if str(oid)
        }
        crafted_obj_names.discard("")
        if crafted_obj_names:
            for answer in _safe_sample(sorted(crafted_obj_names), min(num_samples_per_category, len(crafted_obj_names))):
                distractor_pool = set(all_obj_candidates) - crafted_obj_names
                choices, answer_idx = build_choices_with_answer_idx(answer, distractor_pool, max_choices=10)
                qa["crafted_object"].append(
                    {"question": "Which object did you craft?", "choices": choices, "answer_idx": answer_idx}
                )
        else:
            choices, answer_idx = build_choices_with_answer_idx("none", all_obj_candidates - {"none"}, max_choices=10)
            qa["crafted_object"].append(
                {"question": "Which object did you craft?", "choices": choices, "answer_idx": answer_idx}
            )

        # 3) killed NPCs
        killed_npc_ids = (curr_agents_state.get("npcs_killed") or {}).get(agent_id) or []
        killed_npc_names = {
            _npc_name_from_id(str(nid), npc_id_to_name) for nid in killed_npc_ids if str(nid)
        }
        killed_npc_names.discard("")
        if killed_npc_names:
            for answer in _safe_sample(sorted(killed_npc_names), min(num_samples_per_category, len(killed_npc_names))):
                distractor_pool = set(all_npc_candidates) - killed_npc_names
                choices, answer_idx = build_choices_with_answer_idx(answer, distractor_pool, max_choices=10)
                qa["killed_npc"].append(
                    {"question": "Which NPC did you kill?", "choices": choices, "answer_idx": answer_idx}
                )
        else:
            choices, answer_idx = build_choices_with_answer_idx("none", all_npc_candidates - {"none"}, max_choices=10)
            qa["killed_npc"].append(
                {"question": "Which NPC did you kill?", "choices": choices, "answer_idx": answer_idx}
            )

        # 4) acquired objects
        acquired_obj_ids = (curr_agents_state.get("objects_acquired") or {}).get(agent_id) or []
        acquired_obj_names = {
            obj_id_to_name.get(str(oid), str(oid)) for oid in acquired_obj_ids if str(oid)
        }
        acquired_obj_names.discard("")
        if acquired_obj_names:
            for answer in _safe_sample(sorted(acquired_obj_names), min(num_samples_per_category, len(acquired_obj_names))):
                distractor_pool = set(all_obj_candidates) - acquired_obj_names
                choices, answer_idx = build_choices_with_answer_idx(answer, distractor_pool, max_choices=10)
                qa["acquired_object"].append(
                    {"question": "Which object did you acquire?", "choices": choices, "answer_idx": answer_idx}
                )
        else:
            choices, answer_idx = build_choices_with_answer_idx("none", all_obj_candidates - {"none"}, max_choices=10)
            qa["acquired_object"].append(
                {"question": "Which object did you acquire?", "choices": choices, "answer_idx": answer_idx}
            )

    action_vocab: set[str] = {e.get("action", "") for e in agent_log if e.get("action")}
    action_vocab.discard("")

    location_vocab: set[str] = {e.get("location", "") for e in agent_log if e.get("location")}
    location_vocab.discard("")

    # 5) What did you do in {location} at {time}?
    eligible_events = [
        e for e in agent_log if e.get("action") and e.get("time") and e.get("location")
    ]
    for e in _safe_sample(eligible_events, min(num_samples_per_category, len(eligible_events))):
        answer = str(e["action"])
        time_str = str(e["time"])
        loc_str = str(e["location"])
        distractor_pool = set(action_vocab) - {answer}
        choices, answer_idx = build_choices_with_answer_idx(answer, distractor_pool, max_choices=10)
        qa["action_at_time_location"].append(
            {
                "question": f"What did you do in {loc_str} at {time_str}?",
                "choices": choices,
                "answer_idx": answer_idx,
            }
        )

    # 6) Ordering: 5 consecutive actions -> next action
    actions_only = [e.get("action") for e in agent_log if e.get("action")]
    if len(actions_only) >= 6:
        max_start = len(actions_only) - 6
        start_indices = list(range(0, max_start + 1))
        sampled_starts = _safe_sample(start_indices, min(num_samples_per_category, len(start_indices)))
        for s in sampled_starts:
            seq = [str(a) for a in actions_only[s : s + 5]]
            answer = str(actions_only[s + 5])
            distractor_pool = set(action_vocab) - {answer}
            choices, answer_idx = build_choices_with_answer_idx(answer, distractor_pool, max_choices=10)
            qa["next_action"].append(
                {
                    "question": (
                        "Given these 5 consecutive actions: "
                        + ", ".join(seq)
                        + ". What action did you do next?"
                    ),
                    "choices": choices,
                    "answer_idx": answer_idx,
                }
            )

    # 7) Where did you drop/discard {object}?
    drop_events = []
    for e in agent_log:
        action = e.get("action")
        loc = e.get("location")
        if not action or not loc:
            continue
        parsed = _parse_drop_or_discard_action(str(action))
        if not parsed:
            continue
        verb, obj = parsed
        drop_events.append(
            {
                "verb": verb,
                "object": obj,
                "location": str(loc),
            }
        )

    obj_to_events: dict[str, list[dict]] = {}
    for ev in drop_events:
        key = ev.get("object", "").strip()
        if not key:
            continue
        obj_to_events.setdefault(key, []).append(ev)

    unique_drop_events: list[dict] = []
    for obj, evs in obj_to_events.items():
        locs = {str(x.get("location", "")).strip() for x in evs}
        locs.discard("")
        if len(locs) != 1:
            continue
        unique_drop_events.append(evs[0])

    for ev in _safe_sample(unique_drop_events, min(num_samples_per_category, len(unique_drop_events))):
        obj = str(ev["object"]).strip()
        answer = str(ev["location"]).strip()
        if not obj or not answer:
            continue
        distractor_pool = set(location_vocab) - {answer}
        choices, answer_idx = build_choices_with_answer_idx(answer, distractor_pool, max_choices=10)
        if ev['verb'] == "discard":
            obj_sep = obj.split()
            obj = " ".join(obj_sep[:-1])
        qa["drop_discard_location"].append(
            {
                "question": f"Where did you {ev['verb']} {obj}?",
                "choices": choices,
                "answer_idx": answer_idx,
            }
        )

    # generate statistics
    for key in qa.keys():
        print(f"Generated {len(qa[key])} questions for '{key}'")
    
    return qa