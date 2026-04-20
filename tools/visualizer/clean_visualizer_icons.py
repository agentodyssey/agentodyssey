"""Remove visualizer icons that are not referenced by any world definition."""

import argparse
import json
import re
from pathlib import Path
from typing import Set


def normalize_id(entity_id: str) -> str:
    return re.sub(r'_\d+$', '', entity_id)


def collect_referenced_ids(world_defs_dir: Path) -> Set[str]:
    """Collect all area IDs referenced across all world definitions."""
    referenced = set()

    for def_file in world_defs_dir.rglob("*.json"):
        with open(def_file, 'r') as f:
            world_definition = json.load(f)

        entities = world_definition.get("entities", {})
        for place in entities.get("places", []):
            for area in place.get("areas", []):
                normalized = normalize_id(area["id"])
                referenced.add(normalized)

        custom_events = world_definition.get("custom_events", [])
        if "tutorial" in custom_events:
            referenced.add("area_tutorial_room")

    return referenced


def main():
    parser = argparse.ArgumentParser(
        description="Remove visualizer icons not referenced by any world definition"
    )
    parser.add_argument(
        "--world_defs_dir",
        type=str,
        default="assets/world_definitions",
        help="Path to world definitions directory",
    )
    parser.add_argument(
        "--icons_dir",
        type=str,
        default="assets/visualizer_icons",
        help="Path to visualizer icons directory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print files that would be removed without deleting them",
    )
    args = parser.parse_args()

    world_defs_dir = Path(args.world_defs_dir)
    icons_dir = Path(args.icons_dir)

    if not world_defs_dir.exists():
        print(f"Error: world definitions directory not found at {world_defs_dir}")
        return 1

    if not icons_dir.exists():
        print(f"Error: icons directory not found at {icons_dir}")
        return 1

    referenced_ids = collect_referenced_ids(world_defs_dir)
    print(f"Found {len(referenced_ids)} referenced area IDs across world definitions")

    removed = 0
    kept = 0
    for icon_file in sorted(icons_dir.glob("*.png")):
        if icon_file.stem in referenced_ids:
            kept += 1
        else:
            if args.dry_run:
                print(f"  [DRY RUN] Would remove: {icon_file.name}")
            else:
                icon_file.unlink()
                print(f"  Removed: {icon_file.name}")
            removed += 1

    action = "Would remove" if args.dry_run else "Removed"
    print(f"\n{action} {removed} icons, kept {kept} icons")
    return 0


if __name__ == "__main__":
    exit(main())
