import os
import sys
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
import numpy as np
import random
import argparse
import networkx as nx
import json
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from utils import dynamic_load_game_class

def visualize(world, figsize=(10, 8), seed: int = random.randint(0, 10000)) -> None:

    G = nx.Graph()

    for area in world.area_instances.values():
        G.add_node(area.id, label=area.name)

    for area in world.area_instances.values():
        for path in area.neighbors.values():
            if not G.has_edge(area.id, path.to_id):
                G.add_edge(area.id, path.to_id, locked=getattr(path, "locked", False))

    pos = nx.spring_layout(G, seed=seed, k=1.5)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_axis_off()

    area_to_place: dict[str, str] = {}
    for pid, place in world.place_instances.items():
        for aid in place.areas:
            area_to_place[aid] = place.name

    places: dict[str, list] = {}
    for node, data in G.nodes(data=True):
        p = area_to_place.get(node, "Unknown")
        places.setdefault(p, []).append(pos[node])

    for i, (place_name, coords) in enumerate(places.items()):
        coords = np.array(coords)
        if len(coords) > 2:
            hull = ConvexHull(coords)
            poly_coords = coords[hull.vertices]
        else:
            poly_coords = coords + np.random.normal(0, 0.05, coords.shape)
        color = plt.cm.Set3(i / max(1, len(places)))
        polygon = Polygon(
            poly_coords,
            closed=True,
            linewidth=1,
            edgecolor=color,
            facecolor=color,
            alpha=0.12,
            linestyle="--",
        )
        ax.add_patch(polygon)
        cx, cy = coords.mean(axis=0)
        ax.text(cx, cy, place_name, ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    unlocked_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("locked", False)]
    locked_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("locked", False)]
    nx.draw_networkx_edges(G, pos, edgelist=unlocked_edges, width=1.5, edge_color="gray")
    nx.draw_networkx_edges(G, pos, edgelist=locked_edges, width=1.5, edge_color="red", style="dashed")

    place_colors = {p: plt.cm.Set3(i / max(1, len(places))) for i, p in enumerate(places)}
    node_colors = [place_colors.get(area_to_place.get(n, "Unknown"), plt.cm.Set3(0)) for n, _ in G.nodes(data=True)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.9)

    labels = {n: d["label"] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title("World Map", fontsize=14)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", type=str, default="base", help="Which game to run: base or a folder under games/generated/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for world graph connectivity" )
    args = parser.parse_args()

    if args.game_name in (None, "", "base"):
        world_definition_path = "assets/world_definitions/base/default.json"
    else:
        world_definition_path = f"assets/world_definitions/generated/{args.game_name}/default.json"

    with open(world_definition_path, "r", encoding="utf-8") as f:
        world_definition = json.load(f)

    WorldCls = dynamic_load_game_class(args.game_name, "world", "World")

    world = WorldCls.generate(world_definition, seed=args.seed)
    visualize(world, figsize=(10, 8), seed=random.randint(0, 10000))