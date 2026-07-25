"""Calculate statistics for a game: NPCs, objects, areas, actions, step rules, quest chapters."""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_world_definition(game_name: str) -> dict:
    """Load world_definitions/default.json for a game."""
    if game_name == "base":
        path = ROOT / "assets" / "world_definitions" / "base" / "default.json"
    else:
        path = ROOT / "assets" / "world_definitions" / "generated" / game_name / "default.json"
    if not path.exists():
        sys.exit(f"World definition not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_rules_dir(game_name: str) -> Path:
    """Return path to rules/ directory for a game."""
    if game_name == "base":
        return ROOT / "games" / "base" / "rules"
    return ROOT / "games" / "generated" / game_name / "rules"


def count_world_entities(world_def: dict) -> dict:
    """Count NPCs, objects, and areas from the world definition JSON."""
    entities = world_def.get("entities", {})
    npcs = entities.get("npcs", [])
    objects = entities.get("objects", [])
    places = entities.get("places", [])
    areas = [a for p in places for a in p.get("areas", [])]
    return {
        "places": len(places),
        "areas": len(areas),
        "objects": len(objects),
        "npcs": len(npcs),
    }


def count_rule_classes(filepath: Path, base_class: str | None = None) -> list[str]:
    """Count top-level class definitions in a Python file, optionally filtered by base class."""
    if not filepath.exists():
        return []
    try:
        tree = ast.parse(filepath.read_text())
    except SyntaxError:
        return []
    names = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if base_class is None:
            names.append(node.name)
        else:
            for b in node.bases:
                bname = getattr(b, "id", None) or getattr(getattr(b, "attr", None), "__str__", lambda: "")()
                if bname == base_class:
                    names.append(node.name)
                    break
    return names


def count_quest_chapters(step_rules_path: Path) -> dict | None:
    """Extract QUEST_CONFIG from MainQuestStepRule and count chapters/stages."""
    if not step_rules_path.exists():
        return None

    text = step_rules_path.read_text()

    # Find QUEST_CONFIG dict literal using brace-matching
    match = re.search(r"QUEST_CONFIG[^=]*=\s*\{", text)
    if not match:
        return None

    start = match.start() + match.group().index("{")
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    raw = text[start:end]

    # Count chapters by looking for "id": "chapter_..." patterns
    chapter_ids = re.findall(r'"id"\s*:\s*"(chapter_\w+)"', raw)

    # Count stages by looking for "id": "ch..._..." patterns (non-chapter ids inside stages)
    stage_ids = re.findall(r'"id"\s*:\s*"(ch\d+_\w+)"', raw)

    # Extract quest title
    title_match = re.search(r'"title"\s*:\s*"([^"]+)"', raw)
    title = title_match.group(1) if title_match else "(unknown)"

    return {
        "quest_title": title,
        "chapters": len(chapter_ids),
        "total_stages": len(stage_ids),
        "chapter_details": [],
    }


def count_quest_chapters_detailed(step_rules_path: Path) -> dict | None:
    """Extract quest chapter details with per-chapter stage counts."""
    if not step_rules_path.exists():
        return None

    text = step_rules_path.read_text()
    match = re.search(r"QUEST_CONFIG[^=]*=\s*\{", text)
    if not match:
        return None

    start = match.start() + match.group().index("{")
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    raw = text[start:end]

    # Extract quest title
    title_match = re.search(r'"title"\s*:\s*"([^"]+)"', raw)
    title = title_match.group(1) if title_match else "(unknown)"

    # Find chapter blocks by splitting on chapter id markers
    chapter_splits = list(re.finditer(r'"id"\s*:\s*"(chapter_\w+)"', raw))
    chapter_details = []
    for idx, ch_match in enumerate(chapter_splits):
        ch_id = ch_match.group(1)
        # Get title for this chapter
        rest = raw[ch_match.end():]
        ch_title_m = re.search(r'"title"\s*:\s*"([^"]+)"', rest)
        ch_title = ch_title_m.group(1) if ch_title_m else ch_id

        # Determine text range for this chapter
        ch_start = ch_match.start()
        ch_end = chapter_splits[idx + 1].start() if idx + 1 < len(chapter_splits) else len(raw)
        chapter_text = raw[ch_start:ch_end]
        stage_ids = re.findall(r'"id"\s*:\s*"(ch\d+_\w+)"', chapter_text)
        chapter_details.append({
            "id": ch_id,
            "title": ch_title,
            "stages": len(stage_ids),
        })

    total_stages = sum(c["stages"] for c in chapter_details)
    return {
        "quest_title": title,
        "chapters": len(chapter_details),
        "total_stages": total_stages,
        "chapter_details": chapter_details,
    }


def count_quest_required_entities(step_rules_path: Path) -> dict:
    """Count required_npcs and required_objects defined in step rule classes."""
    if not step_rules_path.exists():
        return {"quest_npcs": 0, "quest_objects": 0}

    text = step_rules_path.read_text()
    # Count entries in required_npcs lists (look for dict entries with "type": "npc")
    quest_npcs = len(re.findall(r'"type"\s*:\s*"npc"', text))
    # Count entries in required_objects lists (look for dict entries with "type": "object"
    # that are inside required_objects blocks — use a simpler heuristic: count
    # occurrences on lines inside required_objects blocks)
    quest_objs = 0
    in_req_obj = False
    bracket_depth = 0
    for line in text.splitlines():
        if "required_objects" in line and "List" in line:
            in_req_obj = True
            bracket_depth = 0
        if in_req_obj:
            bracket_depth += line.count("[") - line.count("]")
            if '"type": "object"' in line or "'type': 'object'" in line:
                quest_objs += 1
            if bracket_depth <= 0 and in_req_obj and bracket_depth != 0:
                in_req_obj = False
            if bracket_depth == 0 and "]" in line and in_req_obj:
                in_req_obj = False

    return {"quest_npcs": quest_npcs, "quest_objects": quest_objs}


def main():
    parser = argparse.ArgumentParser(description="Calculate game statistics.")
    parser.add_argument(
        "game",
        nargs="?",
        default="base",
        help="Game name (e.g. 'base', 'forgebane_alliance', 'metropolis'). Default: base",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    game = args.game
    world_def = load_world_definition(game)
    rules_dir = get_rules_dir(game)

    # --- World definition stats ---
    entity_counts = count_world_entities(world_def)

    # --- Action rules ---
    action_rules_path = rules_dir / "action_rules.py"
    action_names = count_rule_classes(action_rules_path, "BaseActionRule")

    # --- Step rules (scan all .py files under step_rules/) ---
    step_rules_dir = rules_dir / "step_rules"
    step_names_general = []
    step_names_other: dict[str, list[str]] = {}  # filename -> class names
    if step_rules_dir.is_dir():
        for py_file in sorted(step_rules_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            classes = count_rule_classes(py_file, "BaseStepRule")
            if py_file.name == "general.py":
                step_names_general = classes
            elif classes:
                step_names_other[py_file.stem] = classes
    all_step_names = step_names_general + [c for v in step_names_other.values() for c in v]

    # --- Base game comparison ---
    base_rules_dir = get_rules_dir("base")
    base_action_names = set(count_rule_classes(base_rules_dir / "action_rules.py", "BaseActionRule"))
    base_step_dir = base_rules_dir / "step_rules"
    base_step_names = set()
    if base_step_dir.is_dir():
        for py_file in sorted(base_step_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            base_step_names.update(count_rule_classes(py_file, "BaseStepRule"))
    new_action_names = [n for n in action_names if n not in base_action_names]
    new_step_names = [n for n in all_step_names if n not in base_step_names]

    # --- Quest chapters ---
    main_quest_path = step_rules_dir / "main_quest.py" if step_rules_dir.is_dir() else rules_dir / "step_rules.py"
    quest_info = count_quest_chapters_detailed(main_quest_path)

    # --- Quest-injected entities ---
    quest_entities = count_quest_required_entities(main_quest_path)

    # --- Custom events ---
    custom_events = world_def.get("custom_events", [])

    # --- Features ---
    features = world_def.get("features", {})

    results = {
        "game": game,
        "world_definition": entity_counts,
        "quest_injected": quest_entities,
        "total_npcs": entity_counts["npcs"] + quest_entities["quest_npcs"],
        "total_objects": entity_counts["objects"] + quest_entities["quest_objects"],
        "action_rules": len(action_names),
        "new_action_rules": len(new_action_names),
        "step_rules": len(all_step_names),
        "new_step_rules": len(new_step_names),
        "quest": quest_info,
        "custom_events": custom_events,
        "features": features,
    }

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Pretty print
    print(f"{'=' * 60}")
    print(f"  Game Statistics: {game}")
    print(f"{'=' * 60}")
    print()

    print("World Definition (default.json)")
    print(f"  Places:  {entity_counts['places']}")
    print(f"  Areas:   {entity_counts['areas']}")
    print(f"  Objects: {entity_counts['objects']}")
    print(f"  NPCs:    {entity_counts['npcs']}")
    print()

    if quest_entities["quest_npcs"] or quest_entities["quest_objects"]:
        print("Quest-Injected Entities (from step rules)")
        print(f"  Quest NPCs:    {quest_entities['quest_npcs']}")
        print(f"  Quest Objects: {quest_entities['quest_objects']}")
        print()

    print("Totals (world def + quest-injected)")
    print(f"  Total NPCs:    {results['total_npcs']}")
    print(f"  Total Objects: {results['total_objects']}")
    print()

    print(f"Action Rules:  {len(action_names)} ({len(new_action_names)} new)")
    for name in action_names:
        marker = " *" if name in new_action_names else ""
        print(f"  - {name}{marker}")
    print()

    print(f"Step Rules:    {len(all_step_names)} ({len(new_step_names)} new)")
    if step_names_general:
        print(f"  general.py ({len(step_names_general)}):")
        for name in step_names_general:
            marker = " *" if name in new_step_names else ""
            print(f"    - {name}{marker}")
    for fname, classes in step_names_other.items():
        print(f"  {fname}.py ({len(classes)}):")
        for name in classes:
            marker = " *" if name in new_step_names else ""
            print(f"    - {name}{marker}")
    print()

    if quest_info:
        print(f"Main Quest: \"{quest_info['quest_title']}\"")
        print(f"  Chapters:     {quest_info['chapters']}")
        print(f"  Total Stages: {quest_info['total_stages']}")
        for ch in quest_info["chapter_details"]:
            print(f"    {ch['id']}: \"{ch['title']}\" ({ch['stages']} stages)")
    else:
        print("Main Quest: (none found)")
    print()

    print(f"Custom Events: {custom_events}")
    if features:
        print(f"Features:      {features}")

    print()


if __name__ == "__main__":
    main()
