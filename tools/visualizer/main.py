import argparse
import json
import os
import shutil
import sys
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import re


def load_world_definition(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def _load_config_jsonl(config_path: str):
    """Load all snapshots from a config.jsonl file.

    Returns:
        (snapshots_by_step, last_snapshot) where snapshots_by_step maps
        step number -> snapshot dict, and last_snapshot is the final entry.

    Raises:
        ValueError  if config_path does not end with '.jsonl'.
        FileNotFoundError  if the file does not exist.
    """
    if not config_path.lower().endswith('.jsonl'):
        raise ValueError(
            f"--config must be an cumulative/cumulative .jsonl file, got: {config_path}\n"
            "Run eval.py with --cumulative_config_save, or convert an existing run with:\n"
            "  python tools/generate_cumulative_config.py --agent_log_path <path> --save_path <out.jsonl>"
        )
    snapshots_by_step: Dict[int, dict] = {}
    last_snapshot = None
    decoder = json.JSONDecoder()
    with open(config_path, 'r') as f:
        content = f.read()
    idx = 0
    while idx < len(content):
        # Skip whitespace between objects
        while idx < len(content) and content[idx] in ' \t\n\r':
            idx += 1
        if idx >= len(content):
            break
        snap, end = decoder.raw_decode(content, idx)
        idx = end
        last_snapshot = snap
        step = snap.get('step')
        if step is not None:
            snapshots_by_step[step] = snap
    return snapshots_by_step, last_snapshot


class TrajectoryDataExtractor:
    def __init__(self, agent_log_path: str):
        self.agent_log_path = Path(agent_log_path)
        self.data = []
        self.world_info = {
            'areas': {},
            'places': {},
            'edges': {},
            'trajectory': [],
        }
        self.area_contents: Dict[str, Any] = {}
        self.area_contents_by_step: Dict[int, Dict[str, Any]] = {}

    def load_agent_log(self) -> list:
        if not self.agent_log_path.exists():
            raise FileNotFoundError(f"Agent log not found: {self.agent_log_path}")

        with open(self.agent_log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} steps from agent log")
        return self.data

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _extract_area_contents(
        self,
        snap: dict,
        npc_id_to_name: Dict[str, str],
        obj_id_to_name: Dict[str, str],
    ) -> Dict[str, Any]:
        """Extract {area_id -> {npcs, objects}} from a single config snapshot."""
        world_config = snap.get('world', {})
        area_instances = world_config.get('area_instances', {})
        area_contents: Dict[str, Any] = {}
        for area_id, area_inst in area_instances.items():
            npcs = []
            for npc_inst_id in area_inst.get('npcs', []):
                npc_name = npc_id_to_name.get(npc_inst_id)
                if not npc_name:
                    base_id = re.sub(r'_\d+$', '', npc_inst_id)
                    npc_name = npc_id_to_name.get(base_id, npc_inst_id)
                npcs.append(npc_name)
            objects = []
            for obj_id, amount in area_inst.get('objects', {}).items():
                obj_name = obj_id_to_name.get(obj_id, obj_id)
                objects.append({'name': obj_name, 'amount': amount})
            area_contents[area_id] = {'npcs': npcs, 'objects': objects}
        return area_contents

    # ------------------------------------------------------------------

    def build_world_graph_from_config(
        self,
        world_definition: dict,
        config_data: dict,
        config_snapshots: Optional[Dict[int, dict]] = None,
    ) -> Dict[str, Any]:
        """Build the world graph from world definition + env config.

        Args:
            world_definition:        World definition JSON.
            config_data:      A single config snapshot used for world structure
                              (area topology, tutorial detection).  Usually the
                              last (or initial) snapshot.
            config_snapshots: Optional dict of step -> snapshot.  When provided,
                              ``area_contents_by_step`` is populated so the
                              frontend can animate world-state changes.
        """
        # Build place -> area mapping from world definition
        place_id_to_name = {}
        for place in world_definition.get('entities', {}).get('places', []):
            place_name = place['name']
            place_id = place['id']
            place_id_to_name[place_id] = place_name
            self.world_info['places'][place_name] = {
                'name': place_name,
                'areas': []
            }
            for area_def in place.get('areas', []):
                area_id = area_def['id']
                self.world_info['areas'][area_id] = {
                    'id': area_id,
                    'name': area_def['name'],
                    'place': place_name,
                    'neighbors': []
                }
                self.world_info['places'][place_name]['areas'].append(area_id)

        # Check if tutorial is enabled via custom_events
        custom_events = config_data.get('custom_events', []) if config_data else []
        if not custom_events:
            custom_events = world_definition.get('custom_events', [])
        if 'tutorial' in custom_events:
            tutorial_area_id = 'area_tutorial_room'
            tutorial_place = 'Tutorial'
            self.world_info['places'][tutorial_place] = {
                'name': tutorial_place,
                'areas': [tutorial_area_id]
            }
            self.world_info['areas'][tutorial_area_id] = {
                'id': tutorial_area_id,
                'name': 'room',
                'place': tutorial_place,
                'neighbors': []
            }

        # Build ID-to-name lookups from world definition
        npc_id_to_name = {
            npc_def['id']: npc_def['name']
            for npc_def in world_definition.get('entities', {}).get('npcs', [])
        }
        obj_id_to_name = {
            obj_def['id']: obj_def['name']
            for obj_def in world_definition.get('entities', {}).get('objects', [])
        }

        # Access config nested under "world" key (for topology)
        world_config = config_data.get('world', {}) if config_data else {}
        area_instances = world_config.get('area_instances', {})

        # Build edges from config area_instances (neighbors with lock info)
        for area_id, area_inst in area_instances.items():
            neighbors = area_inst.get('neighbors', {})
            for neighbor_id, edge_info in neighbors.items():
                if area_id in self.world_info['areas'] and neighbor_id in self.world_info['areas']:
                    if neighbor_id not in self.world_info['areas'][area_id]['neighbors']:
                        self.world_info['areas'][area_id]['neighbors'].append(neighbor_id)
                    if area_id not in self.world_info['areas'][neighbor_id]['neighbors']:
                        self.world_info['areas'][neighbor_id]['neighbors'].append(area_id)
                    edge_key = '|'.join(sorted([area_id, neighbor_id]))
                    is_locked = edge_info.get('locked', False)
                    if edge_key in self.world_info['edges']:
                        is_locked = is_locked or self.world_info['edges'][edge_key]['locked']
                    self.world_info['edges'][edge_key] = {'locked': is_locked}

        # Static area contents from config_data (step-0 / last snapshot)
        self.area_contents = self._extract_area_contents(
            config_data, npc_id_to_name, obj_id_to_name
        )

        # Per-step area contents from all snapshots
        self.area_contents_by_step = {}
        if config_snapshots:
            for step_num, snap in config_snapshots.items():
                self.area_contents_by_step[step_num] = self._extract_area_contents(
                    snap, npc_id_to_name, obj_id_to_name
                )

        # Extract trajectory from agent log
        for step in self.data:
            obs = step.get('observation', {})
            text = obs.get('text', '') if isinstance(obs, dict) else ''
            loc_match = re.search(r'Current Location:\s*([^,\n]+),\s*(\w+)', text)
            if loc_match:
                place_name = loc_match.group(1).strip()
                area_name = loc_match.group(2).strip()
                area_id = self._find_area_id(place_name, area_name)
                if area_id:
                    if not self.world_info['trajectory'] or self.world_info['trajectory'][-1] != area_id:
                        self.world_info['trajectory'].append(area_id)

        return self.world_info

    def _find_area_id(self, place_name: str, area_name: str) -> Optional[str]:
        """Find the canonical area_id given a place name and area name from observation text."""
        for area_id, area_info in self.world_info['areas'].items():
            if (area_info['place'].lower() == place_name.lower() and
                    area_info['name'].lower() == area_name.lower()):
                return area_id
        return None

    def compute_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_steps': len(self.data),
            'areas_visited': len(set(self.world_info['trajectory'])),
            'places_visited': len(self.world_info['places']),
            'total_rewards': {},
            'action_counts': {},
            'invalid_actions': 0,
            'total_decision_time': 0,
        }

        for step in self.data:
            reward = step.get('reward', {})
            for k, v in reward.items():
                stats['total_rewards'][k] = stats['total_rewards'].get(k, 0) + v

            action = step.get('action', '').split()[0] if step.get('action') else 'unknown'
            stats['action_counts'][action] = stats['action_counts'].get(action, 0) + 1

            if step.get('invalid_action'):
                stats['invalid_actions'] += 1

            stats['total_decision_time'] += step.get('decision_time', 0)

        stats['avg_decision_time'] = stats['total_decision_time'] / len(self.data) if self.data else 0

        return stats

    def to_json(self) -> str:
        return json.dumps({
            'trajectory_data': self.data,
            'world_graph': self.world_info,
            'statistics': self.compute_statistics(),
            'area_contents': self.area_contents,
            'area_contents_by_step': self.area_contents_by_step,
        }, indent=2)


def create_bundle(
    agent_log_path: str,
    world_def_path: Optional[str],
    config_path: Optional[str],
    output_dir: str,
    icons_dir: Optional[str] = None,
    header_label: Optional[str] = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    visualizer_src = Path(__file__).parent / 'main.html'
    if visualizer_src.exists():
        shutil.copy(visualizer_src, output_path / 'index.html')
    else:
        print(f"Warning: HTML visualizer not found at {visualizer_src}")

    project_root = Path(__file__).parent.parent.parent
    logo_src = project_root / 'assets' / 'logo.png'
    if logo_src.exists():
        shutil.copy(logo_src, output_path / 'logo.png')

    extractor = TrajectoryDataExtractor(agent_log_path)
    extractor.load_agent_log()

    if world_def_path and os.path.exists(world_def_path) and config_path and os.path.exists(config_path):
        with open(world_def_path, 'r') as f:
            wd = json.load(f)
        config_snapshots, last_snapshot = _load_config_jsonl(config_path)
        extractor.build_world_graph_from_config(wd, last_snapshot, config_snapshots)

    data_dir = output_path / 'data'
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / 'trajectory.json', 'w') as f:
        f.write(extractor.to_json())

    shutil.copy(agent_log_path, data_dir / 'agent_log.jsonl')

    if world_def_path and os.path.exists(world_def_path):
        shutil.copy(world_def_path, data_dir / 'world_definition.json')

    if config_path and os.path.exists(config_path):
        shutil.copy(config_path, data_dir / 'config.jsonl')

    if icons_dir and os.path.exists(icons_dir):
        icons_output = output_path / 'icons'
        shutil.copytree(icons_dir, icons_output, dirs_exist_ok=True)

    if header_label is not None:
        with open(output_path / 'preloaded_data.js', 'w') as f:
            f.write(
                "// Auto-generated preloaded data (bundle)\n"
                "window.PRELOADED_DATA = {\n"
                f"    headerLabel: {json.dumps(header_label)}\n"
                "};\n"
            )

    print(f"Bundle created at: {output_path}")
    return output_path


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        if args and '404' in str(args[0]):
            super().log_message(format, *args)


def start_server(directory: str, port: int = 8000) -> threading.Thread:
    os.chdir(directory)

    handler = QuietHTTPHandler

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        httpd.serve_forever()


def main():
    project_root = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_log",
        type=str,
        default="output/temp/agent_log.jsonl",
    )
    parser.add_argument(
        "--world_definition_path",
        type=str,
        default="assets/world_definitions/generated/remnant/default.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="output/temp/config.jsonl",
        help="Path to the cumulative config.jsonl file (must be .jsonl).",
    )
    parser.add_argument(
        "--icons_dir",
        type=str,
        default=str(project_root / "assets" / "visualizer_icons"),
    )
    parser.add_argument(
        "--bundle",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8003,
    )
    parser.add_argument(
        "--no_browser",
        action="store_true",
    )
    parser.add_argument(
        "--skip_assets_generation",
        action="store_true",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
    )
    parser.add_argument(
        "--header",
        type=str,
        default="Long Context Agent w/ GPT-5",
        help="Label shown in the center of the visualizer top bar.",
    )

    args = parser.parse_args()

    extractor = TrajectoryDataExtractor(args.agent_log)
    extractor.load_agent_log()

    # Load world def
    world_def_data = None
    if args.world_definition_path and os.path.exists(args.world_definition_path):
        with open(args.world_definition_path, 'r') as f:
            world_def_data = json.load(f)

    # Validate and load config — must be .jsonl
    config_data = None
    config_snapshots: Dict[int, dict] = {}
    if args.config:
        if not args.config.lower().endswith('.jsonl'):
            sys.exit(
                f"Error: --config must be an cumulative/cumulative .jsonl file, got: {args.config}\n"
                "Run eval.py with --cumulative_config_save, or generate one with:\n"
                "  python tools/generate_cumulative_config.py --agent_log_path <path> --save_path <out.jsonl>"
            )
        if not os.path.exists(args.config):
            sys.exit(
                f"Error: config file not found: {args.config}\n"
                "Generate it with:\n"
                f"  python tools/generate_cumulative_config.py \\\n"
                f"      --agent_log_path {args.agent_log} \\\n"
                f"      --save_path {args.config}"
            )
        config_snapshots, config_data = _load_config_jsonl(args.config)
        print(f"Loaded {len(config_snapshots)} config snapshots from {args.config}")

    if world_def_data and config_data:
        extractor.build_world_graph_from_config(world_def_data, config_data, config_snapshots)
    else:
        print("Warning: world_definition or config not found, world graph will be empty")

    stats = extractor.compute_statistics()

    print("\n" + "=" * 50)
    print("TRAJECTORY STATISTICS")
    print("=" * 50)
    print(f"Total steps: {stats['total_steps']}")
    print(f"Areas visited: {stats['areas_visited']}")
    print(f"Places visited: {stats['places_visited']}")
    print(f"Invalid actions: {stats['invalid_actions']}")
    print(f"Avg decision time: {stats['avg_decision_time']:.3f}s")
    print("\nRewards:")
    for k, v in stats['total_rewards'].items():
        if v != 0:
            print(f"  {k}: {v}")
    print("\nAction distribution:")
    for k, v in sorted(stats['action_counts'].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("=" * 50 + "\n")

    if args.stats_only:
        return 0

    icons_dir = args.icons_dir or "assets/visualizer_icons"
    if not args.skip_assets_generation:
        from generate_assets import AssetGenerator

        world_definition = load_world_definition(args.world_definition_path)
        generator = AssetGenerator(output_dir=icons_dir)
        custom_events = (config_data or {}).get('custom_events') or world_definition.get('custom_events', [])
        generator.generate_from_world_definition(world_definition, custom_events=custom_events)
    args.icons_dir = icons_dir

    if args.bundle:
        bundle_path = create_bundle(
            args.agent_log,
            args.world_definition_path,
            args.config,
            args.bundle,
            args.icons_dir,
            header_label=args.header,
        )
        serve_dir = str(bundle_path)
    else:
        serve_dir = str(project_root)
        data_js_path = Path(__file__).parent / 'preloaded_data.js'

        icons_data = {}
        icons_base_path = ""
        if args.icons_dir and os.path.exists(args.icons_dir):
            icons_base_path = "/" + os.path.relpath(args.icons_dir, serve_dir)
            for root, dirs, files in os.walk(args.icons_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.webp', '.svg')):
                        rel_path = "/" + os.path.relpath(os.path.join(root, file), serve_dir)
                        name = os.path.splitext(file)[0]
                        parent_folder = os.path.basename(root)
                        if parent_folder in ['areas', 'objects', 'npcs', 'actions']:
                            icons_data[f"{parent_folder}/{name}"] = rel_path
                        icons_data[name] = rel_path
            print(f"Found {len(icons_data)} icons in {args.icons_dir}")

        preloaded_js = f"""// Auto-generated preloaded data
window.PRELOADED_DATA = {{
    trajectoryData: {json.dumps(extractor.data)},
    worldDefinition: {json.dumps(world_def_data)},
    configData: {json.dumps(config_data)},
    worldGraph: {json.dumps(extractor.world_info)},
    statistics: {json.dumps(stats)},
    icons: {json.dumps(icons_data)},
    iconsBasePath: {json.dumps(icons_base_path)},
    areaContents: {json.dumps(extractor.area_contents)},
    areaContentsByStep: {json.dumps(extractor.area_contents_by_step)},
    headerLabel: {json.dumps(args.header)}
}};
console.log('Preloaded data available:', Object.keys(window.PRELOADED_DATA));
console.log('Icons loaded:', Object.keys(window.PRELOADED_DATA.icons).length);
console.log('Config snapshots loaded:', Object.keys(window.PRELOADED_DATA.areaContentsByStep).length);
"""
        with open(data_js_path, 'w') as f:
            f.write(preloaded_js)
        print(f"Created preloaded data at: {data_js_path}")

    url = f"http://localhost:{args.port}"
    if args.bundle:
        url += "/index.html"
    else:
        url += "/tools/visualizer/main.html"

    if not args.no_browser:
        print(f"Opening browser at {url}")
        webbrowser.open(url)

    print(f"\nStarting server in {serve_dir}")
    try:
        os.chdir(serve_dir)
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            print(f"Serving at http://localhost:{args.port}")
            print("Press Ctrl+C to stop the server\n")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")

    return 0


if __name__ == "__main__":
    exit(main())
