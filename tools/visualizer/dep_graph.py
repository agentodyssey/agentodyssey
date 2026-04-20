"""Generate Mermaid graphs and x-axis SVG visualizations from a
DependencyTracker ``to_dict()`` export.

Usage as a library
------------------

    from tools.dep_graph_visualizer import dict_to_mermaid, dict_to_axis_svg

    dep_dict = tracker.to_dict()

    # full graph
    mermaid = dict_to_mermaid(dep_dict)
    svg     = dict_to_axis_svg(dep_dict)

    # hide specific dependency types
    mermaid = dict_to_mermaid(dep_dict, exclude={"init -> pick up"})

    # only show specific dependency types
    svg = dict_to_axis_svg(dep_dict, include={"pick up -> craft", "craft -> sell"})

Usage as CLI
------------

    # Mermaid (full)
    python -m tools.dep_graph_visualizer dep.json --format mermaid -o graph.mmd

    # SVG with filtering
    python -m tools.dep_graph_visualizer dep.json --format svg -o graph.svg \\
        --exclude "init -> pick up" --exclude "init -> tutorial_room_step"

    # Only keep certain edge types
    python -m tools.dep_graph_visualizer dep.json --format svg -o graph.svg \\
        --include "pick up -> craft" --include "craft -> sell"

    # List all edge types present in the dict
    python -m tools.dep_graph_visualizer dep.json --list-edges
"""

from __future__ import annotations

import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_dep_dict(
    dep_dict: Dict[str, Any],
    *,
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    edges = dict(dep_dict.get("edges", {}))

    if include is not None:
        edges = {k: v for k, v in edges.items() if k in include}
    if exclude is not None:
        edges = {k: v for k, v in edges.items() if k not in exclude}

    return {
        "edges": edges,
        "step_hints": dep_dict.get("step_hints", {}),
        "steps": dep_dict.get("steps", []),
    }

def _collect_step_edges(
    dep_dict: Dict[str, Any],
) -> Tuple[List[int], Dict[str, List[str]], Dict[str, Set[str]]]:
    edges = dep_dict.get("edges", {})
    step_hints: Dict[str, List[str]] = dep_dict.get("step_hints", {})
    steps: List[int] = sorted(s for s in set(dep_dict.get("steps", [])) if s >= 0)

    step_edge_map: Dict[str, Set[str]] = defaultdict(set)
    for _key, pairs in edges.items():
        for pair in pairs:
            src_step, dst_step = int(pair[0]), int(pair[1])
            if src_step < 0 or dst_step < 0:
                continue
            src_sn = f"S_{src_step}"
            dst_sn = f"S_{dst_step}"
            if src_sn != dst_sn:
                step_edge_map[dst_sn].add(src_sn)

    return steps, step_hints, step_edge_map

def dict_to_mermaid(
    dep_dict: Dict[str, Any],
    *,
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
    max_hint_lines: int = 8,
) -> str:
    filtered = filter_dep_dict(dep_dict, include=include, exclude=exclude)
    steps, step_hints, step_edges = _collect_step_edges(filtered)

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines = ["flowchart TD"]

    for s in steps:
        if s < 0:
            continue
        hints = step_hints.get(str(s), [])
        if hints:
            shown = hints[:max_hint_lines]
            extra = len(hints) - len(shown)
            hint_block = "<br/>".join(esc(h) for h in shown)
            if extra > 0:
                hint_block += f"<br/>... (+{extra} more)"
            label = esc(f"step {s}") + f"<br/>{hint_block}"
        else:
            label = esc(f"step {s}")
        lines.append(f'  S_{s}["{label}"]')

    for dst_sn, src_sns in step_edges.items():
        for src_sn in src_sns:
            lines.append(f"  {src_sn} --> {dst_sn}")

    return "\n".join(lines)

def dict_to_axis_svg(
    dep_dict: Dict[str, Any],
    *,
    include: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
    step_px: int = 70,
    margin_x: int = 30,
    margin_y: int = 30,
    tick_len: int = 10,
    base_arc: int = 80,
    level_gap: int = 80,
    font_size: int = 15,
    show_all_steps: bool = True,
    background: str = "#ffffff",
    axis_color: str = "#111111",
    grid_color: str = "#e9e9e9",
    arc_color: str = "#111111",
    arc_width: float = 2.0,
    arc_opacity: float = 0.65,
    axis_width: float = 2.5,
    tick_width: float = 2.0,
    dot_radius: float = 3.0,
    dot_stroke_width: float = 2.0,
    font_family: str = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
    show_grid: bool = True,
    banded: bool = True,
    arrows: bool = True,
) -> str:
    filtered = filter_dep_dict(dep_dict, include=include, exclude=exclude)
    all_steps, _step_hints, step_edges = _collect_step_edges(filtered)

    present = sorted(s for s in all_steps if s >= 0)
    max_step = max(present) if present else 0

    if show_all_steps:
        axis_steps = list(range(0, max_step + 1))
    else:
        axis_steps = present

    step_to_idx = {s: i for i, s in enumerate(axis_steps)}

    def stepnode_to_step(sn: str) -> int:
        return int(sn.split("_", 1)[1])

    interval_info: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for dst_sn, src_sns in step_edges.items():
        dst_step = stepnode_to_step(dst_sn)
        if dst_step not in step_to_idx:
            continue
        dst_i = step_to_idx[dst_step]
        for src_sn in src_sns:
            src_step = stepnode_to_step(src_sn)
            if src_step not in step_to_idx:
                continue
            src_i = step_to_idx[src_step]
            if src_i == dst_i:
                continue
            a, b = (src_i, dst_i) if src_i < dst_i else (dst_i, src_i)
            key = (a, b)
            info = interval_info.get(key)
            if info is None:
                info = {"lr": False, "rl": False, "examples": []}
                interval_info[key] = info
            if src_i < dst_i:
                info["lr"] = True
            else:
                info["rl"] = True
            info["examples"].append((axis_steps[src_i], axis_steps[dst_i]))

    intervals = sorted(interval_info.keys(), key=lambda ab: (ab[0], ab[1]))

    level_last_end: list[int] = []
    assigned: list[tuple[int, int, int]] = []
    for a, b in intervals:
        placed = False
        for lvl, last_end in enumerate(level_last_end):
            if a >= last_end:
                level_last_end[lvl] = b
                assigned.append((a, b, lvl))
                placed = True
                break
        if not placed:
            level_last_end.append(b)
            assigned.append((a, b, len(level_last_end) - 1))

    def above_level(lvl: int) -> int:
        return lvl // 2

    def is_above(lvl: int) -> bool:
        return (lvl % 2) == 0

    max_above = max((above_level(lvl) for _, _, lvl in assigned if is_above(lvl)), default=-1)
    max_below = max((above_level(lvl) for _, _, lvl in assigned if not is_above(lvl)), default=-1)

    n = len(axis_steps)
    width = margin_x * 2 + (n - 1) * step_px + 60

    top_room = (max_above + 1) * level_gap + base_arc if max_above >= 0 else base_arc
    bottom_room = (max_below + 1) * level_gap + base_arc if max_below >= 0 else base_arc
    axis_y = margin_y + top_room
    height = axis_y + bottom_room + margin_y + 60

    def x_at(i: int) -> float:
        return float(margin_x + i * step_px)

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
        )

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">'
    )

    parts.append("<defs>")
    parts.append(
        "<style><![CDATA["
        f".bg{{fill:{background};}}"
        f".axis{{stroke:{axis_color};stroke-width:{axis_width};stroke-linecap:round;}}"
        f".tick{{stroke:{axis_color};stroke-width:{tick_width};stroke-linecap:round;}}"
        f".grid{{stroke:{grid_color};stroke-width:1;}}"
        f".label{{fill:{axis_color};font-family:{font_family};font-size:{font_size}px;}}"
        f".dot{{fill:{background};stroke:{axis_color};stroke-width:{dot_stroke_width};}}"
        f".arc{{stroke:{arc_color};stroke-width:{arc_width};stroke-linecap:round;fill:none;opacity:{arc_opacity};}}"
        ".arcBelow{stroke-dasharray:6 4;}"
        ".band{fill:rgba(0,0,0,0.03);}"
        "]]></style>"
    )
    if arrows:
        parts.append(
            f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{arc_color}"/></marker>'
        )
        parts.append(
            f'<marker id="axisArrow" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="8" markerHeight="8" orient="auto">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{axis_color}"/></marker>'
        )
    parts.append("</defs>")

    parts.append(f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>')

    if banded and n >= 2:
        for i in range(n - 1):
            if i % 2 == 1:
                x_left = x_at(i)
                parts.append(
                    f'<rect class="band" x="{x_left}" y="0" width="{step_px}" height="{height}"/>'
                )

    x0 = x_at(0)
    x1 = x_at(n - 1) + 40.0
    if arrows:
        parts.append(
            f'<line class="axis" x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" '
            f'marker-end="url(#axisArrow)"/>'
        )
    else:
        parts.append(
            f'<line class="axis" x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}"/>'
        )

    if show_grid:
        for i in range(n):
            xi = x_at(i)
            parts.append(
                f'<line class="grid" x1="{xi}" y1="{margin_y / 2}" '
                f'x2="{xi}" y2="{height - margin_y / 2}"/>'
            )

    for i, s in enumerate(axis_steps):
        xi = x_at(i)
        parts.append(
            f'<line class="tick" x1="{xi}" y1="{axis_y - tick_len / 2}" '
            f'x2="{xi}" y2="{axis_y + tick_len / 2}"/>'
        )
        label = f"{s}"
        parts.append(
            f'<text class="label" x="{xi}" y="{axis_y + tick_len / 2 + font_size + 8}" '
            f'text-anchor="middle">{esc(label)}</text>'
        )
        parts.append(f'<circle class="dot" cx="{xi}" cy="{axis_y}" r="{dot_radius}"/>')

    for a, b, lvl in assigned:
        info = interval_info[(a, b)]
        lr = bool(info["lr"])
        rl = bool(info["rl"])

        left_x, right_x = x_at(a), x_at(b)

        hlevel = above_level(lvl)
        arc_h = base_arc + hlevel * level_gap
        cy = axis_y - arc_h if is_above(lvl) else axis_y + arc_h
        midx = (left_x + right_x) / 2.0

        draw_l2r = lr and not rl
        draw_r2l = rl and not lr

        if draw_l2r:
            sx, ex_ = left_x, right_x
        elif draw_r2l:
            sx, ex_ = right_x, left_x
        else:
            sx, ex_ = left_x, right_x

        cls = "arc arcBelow" if not is_above(lvl) else "arc"
        marker = ' marker-end="url(#arrow)"' if arrows and (draw_l2r or draw_r2l) else ""

        if info["examples"]:
            ex0 = info["examples"][0]
            title_txt = f"{ex0[0]} -> {ex0[1]}"
        else:
            title_txt = "dependency"

        parts.append(
            f'<path class="{cls}" d="M {sx} {axis_y} Q {midx} {cy} {ex_} {axis_y}"{marker}>'
            f"<title>{esc(str(title_txt))}</title></path>"
        )

    parts.append("</svg>")
    return "\n".join(parts)

def load_dep_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_dep_dict(dep_dict: Dict[str, Any], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dep_dict, f, indent=2, ensure_ascii=False)

def list_edge_types(dep_dict: Dict[str, Any]) -> List[str]:
    return sorted(dep_dict.get("edges", {}).keys())

def main():
    parser = argparse.ArgumentParser(
        description="Generate Mermaid / SVG from a DependencyTracker JSON dict."
    )
    parser.add_argument("input", help="Path to the JSON dependency dict file.")
    parser.add_argument(
        "--format", "-f",
        choices=["mermaid", "svg", "both"],
        default="both",
        help="Output format (default: both).",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path (omit extension for 'both').")
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Only keep edges matching this key (repeatable). E.g. --include 'pick up -> craft'.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Drop edges matching this key (repeatable). E.g. --exclude 'init -> pick up'.",
    )
    parser.add_argument(
        "--list-edges",
        action="store_true",
        help="Print all edge types and exit.",
    )
    args = parser.parse_args()

    dep_dict = load_dep_dict(args.input)

    if args.list_edges:
        for et in list_edge_types(dep_dict):
            count = len(dep_dict["edges"][et])
            print(f"  {et}  ({count} pair{'s' if count != 1 else ''})")
        return

    inc = set(args.include) if args.include else None
    exc = set(args.exclude) if args.exclude else None

    if args.format in ("mermaid", "both"):
        mmd = dict_to_mermaid(dep_dict, include=inc, exclude=exc)
        if args.format == "mermaid":
            out = args.output or "dep_graph.mmd"
        else:
            base = args.output or "dep_graph"
            out = f"{base}.mmd"
        with open(out, "w", encoding="utf-8") as f:
            f.write(mmd)
        print(f"✅  Mermaid written to {out}")

    if args.format in ("svg", "both"):
        svg = dict_to_axis_svg(dep_dict, include=inc, exclude=exc)
        if args.format == "svg":
            out = args.output or "dep_graph.svg"
        else:
            base = args.output or "dep_graph"
            out = f"{base}.svg"
        with open(out, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"✅  SVG written to {out}")


if __name__ == "__main__":
    main()
