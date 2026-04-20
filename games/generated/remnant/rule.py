from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import DefaultDict, List, Dict, Any, Optional, TYPE_CHECKING, Deque, Set, Tuple
from abc import ABC, abstractmethod
from tools.logger import get_logger

if TYPE_CHECKING:
    from games.generated.remnant.env import AgentOdysseyEnv
    from games.generated.remnant.agent import Agent
    from games.generated.remnant.world import World


@dataclass
class Event:
    type: str
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RuleContext:
    env: "AgentOdysseyEnv"  # back-reference; contains world, curr_agents_state, rng, logger
    world: "World"
    agent: "Agent"
    action: str  # canonical verb, e.g. "pick up"
    params: List[str]
    step_index: int

@dataclass
class RuleResult:
    feedback: Dict[str, str]  # agent_id -> text
    info_flags: Dict[str, Dict[str, Any]]  # e.g. {"step_invalid_action": {agent_id: bool}}
    events: List[Event] = field(default_factory=list)

    def add_feedback(self, agent_id: str, text: str) -> None:
        self.feedback[agent_id] = self.feedback.get(agent_id, "") + text

    # --- methods for dependency tracking ---
    def tloc(self, kind: str, id: str) -> Dict[str, str]:
        # location encoding used by the dependency tracker
        return {"kind": str(kind), "id": str(id)}

    def track_move(self, agent_id: str, obj_id: str, amount: int, src: Dict[str, str], dst: Dict[str, str]) -> None:
        self.events.append(Event(
            type="track.move",
            agent_id=str(agent_id),
            data={"obj_id": obj_id, "amount": int(amount), "src": src, "dst": dst},
        ))

    def track_spawn(self, agent_id: str, obj_id: str, amount: int, dst: Dict[str, str]) -> None:
        self.events.append(Event(
            type="track.spawn",
            agent_id=str(agent_id),
            data={"obj_id": obj_id, "amount": int(amount), "dst": dst},
        ))

    def track_consume(self, agent_id: str, obj_id: str, amount: int, src: Dict[str, str]) -> None:
        self.events.append(Event(
            type="track.consume",
            agent_id=str(agent_id),
            data={"obj_id": obj_id, "amount": int(amount), "src": src},
        ))
    
    def track_utilize(self, agent_id: str, obj_id: str, amount: int, src: Dict[str, str]) -> None:
        self.events.append(Event(
            type="track.utilize",
            agent_id=str(agent_id),
            data={"obj_id": obj_id, "amount": int(amount), "src": src},
        ))

class BaseActionRule(ABC):
    name: str = "action_base"
    verb: str = ""
    params: List[str] = []
    param_min: int = 0
    param_max: Optional[int] = None
    description: str = "Base agent rule"

    def matches(self, action: str) -> bool:
        return action in self.verbs

    def validate_params(self, ctx: RuleContext, res: RuleResult) -> bool:
        n = len(ctx.params)
        if n < self.param_min or (self.param_max is not None and n > self.param_max):
            correct_format = f"'{ctx.action} " + " ".join(f"<{p}>" for p in self.params) + "'"
            res.add_feedback(
                ctx.agent.id,
                f"Invalid action. Action '{ctx.action}' requires the format {correct_format}, but got {n} parameter(s). \n"
                f"Default to wait.\n"
            )
            res.info_flags.setdefault("step_invalid_action", {})[ctx.agent.id] = True
            return False
        return True

    """Agent-level rule that applies when an agent takes action matching verbs."""
    @abstractmethod
    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        """Mutate world / env.curr_agents_state and write to res."""
        ...
    
class BaseStepRule(ABC):
    name: str = "base_step"
    description: str = "Base environment step rule"
    priority: int = 0  # lower number = higher priority

    """Environment-level rule that applies every step, e.g. NPC active attack"""
    @abstractmethod
    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        ...

class ActionRuleEngine:
    def __init__(self, rules: list[BaseActionRule]):
        self.logger = get_logger("RuleEngine")
        self.by_verb: dict[str, BaseActionRule] = {}
        for rule in rules:
            v = rule.verb
            if v in self.by_verb:
                self.logger.warning(f"Duplicate rule verb '{v}', overriding previous rule.")
            self.by_verb[v] = rule

    def dispatch(self, ctx: RuleContext, res: RuleResult) -> None:
        rule = self.by_verb.get(ctx.action)
        if rule is None:
            res.add_feedback(ctx.agent.id, f"Invalid action: '{ctx.action}'. Default to wait.\n")
            res.info_flags.setdefault("step_invalid_action", {})[ctx.agent.id] = True
            return

        if not rule.validate_params(ctx, res):
            return

        rule.apply(ctx, res)

@dataclass
class RewardBreakdown:
    unique_kill: int = 0
    kill: int = 0
    craft: int = 0
    exploration: int = 0
    death: int = 0
    trade: int = 0
    quest: int = 0
    side_quest: int = 0

    @property
    def total(self) -> int:
        return self.kill + self.craft + self.exploration + self.quest + self.side_quest + self.trade + self.unique_kill - self.death
    
    @property
    def xp_total(self) -> int:
        return self.craft * 5 + self.kill * 5 + self.exploration * 5 + self.side_quest * 5 + self.trade * 10 + self.quest * 20
    
    @property
    def rl_total(self) -> int:
        return self.xp_total + 5 * self.unique_kill - 10 * self.death
    
    @property
    def score_total(self) -> int:
        return self.total - self.kill

class RewardFunction(ABC):
    @abstractmethod
    def compute(
        self,
        env: "AgentOdysseyEnv",
        prev_state: dict,
        res: RuleResult
    ) -> dict[str, RewardBreakdown]:
        ...

Location = Tuple[str, str]  # (kind, id)
@dataclass(frozen=True)
class NodeMeta:
    node_id: str
    step: int
    actor: str
    rule: str
    seq: int

class DependencyTracker:
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

        # each unit becomes a token id (obj_id#k)
        self._next_token = 0
        self._token_last: Dict[str, Optional[str]] = {}   # token_id -> node_id that last moved/spawned it
        self._loc_tokens: Dict[Location, Dict[str, Deque[str]]] = defaultdict(lambda: defaultdict(deque))

        self._nodes: Dict[str, NodeMeta] = {}
        self._edges: Dict[str, Set[str]] = defaultdict(set)

        # sequence counter per step to account for multiple agents
        self._seq_by_step: Dict[int, int] = defaultdict(int)

        # a synthetic root node so dependencies from initial world state are visible
        self.INIT_NODE_ID = "INIT"
        self._nodes[self.INIT_NODE_ID] = NodeMeta(
            node_id=self.INIT_NODE_ID, step=-1, actor="env", rule="init", seq=0
        )

        # human-readable hints for visualization
        self._node_hints: Dict[str, str] = {}
        self._step_hints: DefaultDict[int, List[str]] = defaultdict(list)
        self._max_step_hint_lines = 8

    def bootstrap_from_state(self, env: "AgentOdysseyEnv", world: "World", node_id: str = None) -> None:
        node_id = self.INIT_NODE_ID if node_id is None else node_id

        for area_id, area in world.area_instances.items():
            loc = ("area", str(area_id))
            for obj_id, cnt in area.objects.items():
                self._spawn_tokens(obj_id=str(obj_id), amount=int(cnt), dst=loc, last=node_id)

        for agent in env.agents:
            aid = str(agent.id)
            for obj_id, cnt in agent.items_in_hands.items():
                self._spawn_tokens(str(obj_id), int(cnt), ("hand", aid), last=node_id)
            for obj_id, cnt in agent.equipped_items_in_limb.items():
                self._spawn_tokens(str(obj_id), int(cnt), ("equip", aid), last=node_id)
            # no need for inventory here as all will be tracked via containers

        for cid, cinst in world.container_instances.items():
            cloc = ("container", str(cid))
            for obj_id, cnt in cinst.inventory.items():
                self._spawn_tokens(str(obj_id), int(cnt), cloc, last=node_id)

    def process_rule_result(self, *, step: int, actor: str, rule: str, events: List[Any], hint: Optional[str] = None) -> str:
        node_id = self._new_node(step=int(step), actor=str(actor), rule=str(rule))

        if hint:
            self._node_hints[node_id] = str(hint)
            self._step_hints[int(step)].append(str(hint))

        self._consume_tracking_events(node_id, events)
        return node_id

    def edges(self) -> Dict[str, Set[str]]:
        return self._edges

    def nodes(self) -> Dict[str, NodeMeta]:
        return self._nodes

    @staticmethod
    def _esc_dot_label(s: str) -> str:
        # Graphviz label escaping
        return (
            s.replace("\\", "\\\\")
             .replace('"', '\\"')
             .replace("\n", "\\n")
        )

    def to_dict(self) -> Dict[str, Any]:
        edges: Dict[str, List[List[int]]] = {}

        for dst_nid, src_nids in self._edges.items():
            dst_meta = self._nodes.get(dst_nid)
            if dst_meta is None:
                continue
            if dst_nid == self.INIT_NODE_ID:
                continue
            for src_nid in src_nids:
                if src_nid == self.INIT_NODE_ID:
                    continue
                src_meta = self._nodes.get(src_nid)
                if src_meta is None:
                    continue
                key = f"{src_meta.rule} -> {dst_meta.rule}"
                edges.setdefault(key, []).append([src_meta.step, dst_meta.step])

        for key in edges:
            edges[key].sort()

        step_hints: Dict[str, List[str]] = {}
        for step, hints in self._step_hints.items():
            step_hints[str(step)] = list(hints)

        steps = sorted(self._steps_present())

        return {
            "edges": edges,
            "step_hints": step_hints,
            "steps": steps,
        }

    def to_dot(self, mode: str = "action") -> str:
        if mode not in ("action", "step"):
            raise ValueError("mode must be 'action' or 'step'")

        if mode == "action":
            return self._to_dot_action()
        else:
            return self._to_dot_step()

    def to_mermaid(self, mode: str = "step") -> str:
        if mode not in ("action", "step"):
            raise ValueError("mode must be 'action' or 'step'")

        def esc(s: str) -> str:
            # Mermaid node label escaping
            s = s.replace("\\", "\\\\")
            s = s.replace('"', '\\"')
            return s
        
        if mode == "action":
            edges = [(u, v) for u, vs in self._edges.items() for v in vs
                     if u != self.INIT_NODE_ID and v != self.INIT_NODE_ID]

            def label(nid: str) -> str:
                m = self._nodes[nid]
                base = f"{m.step}:{m.seq} {m.actor} {m.rule}"
                hint = self._node_hints.get(nid)
                if hint:
                    base += "<br/>" + hint
                return esc(base)

            def mid(nid: str) -> str:
                return "N_" + nid.replace(":", "_").replace("-", "_")

            lines = ["flowchart TD"]
            for nid in self._nodes:
                if nid == self.INIT_NODE_ID:
                    continue
                lines.append(f'  {mid(nid)}["{label(nid)}"]')
            for u, v in edges:
                lines.append(f"  {mid(v)} --> {mid(u)}")
            return "\n".join(lines)

        # step mode
        step_edges = self._collapse_edges_by_step()
        lines = ["flowchart TD"]

        steps = sorted({s for s in self._steps_present()})
        for s in steps:
            hints = self._step_hints.get(s, [])
            if hints:
                shown = hints[: getattr(self, "_max_step_hint_lines", 8)]
                extra = len(hints) - len(shown)
                hint_block = "<br/>".join(esc(h) for h in shown)
                if extra > 0:
                    hint_block += f"<br/>... (+{extra} more)"
                label = esc(f"step {s}<br/>{hint_block}")
            else:
                label = esc(f"step {s}")

            lines.append(f'  S_{s}["{label}"]')

        for u, vs in step_edges.items():
            if u == "S_INIT":
                continue
            for v in vs:
                if v == "S_INIT":
                    continue
                lines.append(f"  {v} --> {u}")

        return "\n".join(lines)

    def _new_node(self, step: int, actor: str, rule: str) -> str:
        seq = self._seq_by_step[step]
        self._seq_by_step[step] += 1

        node_id = f"{step}:{seq}:{actor}:{rule}"
        self._nodes[node_id] = NodeMeta(node_id=node_id, step=step, actor=actor, rule=rule, seq=seq)
        return node_id

    @staticmethod
    def _parse_loc(d: Dict[str, Any], field: str) -> Location:
        if not isinstance(d, dict) or "kind" not in d or "id" not in d:
            raise ValueError(f"Invalid location in {field}: {d}")
        return (str(d["kind"]), str(d["id"]))

    def _consume_tracking_events(self, node_id: str, events: List[Any]) -> None:
        for ev in events:
            if getattr(ev, "type", None) not in ("track.move", "track.spawn", "track.consume"):
                continue

            data = getattr(ev, "data", None) or {}
            obj_id = str(data.get("obj_id"))
            amount = int(data.get("amount", 0))
            if amount <= 0:
                continue

            if ev.type == "track.move":
                src = self._parse_loc(data.get("src"), "src")
                dst = self._parse_loc(data.get("dst"), "dst")
                self._apply_move(node_id, obj_id, amount, src, dst)

            elif ev.type == "track.spawn":
                dst = self._parse_loc(data.get("dst"), "dst")
                self._apply_spawn(node_id, obj_id, amount, dst)

            elif ev.type == "track.consume":
                src = self._parse_loc(data.get("src"), "src")
                self._apply_consume(node_id, obj_id, amount, src)
            
            elif ev.type == "track.utilize":
                src = self._parse_loc(data.get("src"), "src")
                self._apply_utilize(node_id, obj_id, amount, src)

    def _spawn_tokens(self, obj_id: str, amount: int, dst: Location, last: Optional[str]) -> None:
        q = self._loc_tokens[dst][obj_id]
        for _ in range(amount):
            tid = f"{obj_id}#{self._next_token}"
            self._next_token += 1
            self._token_last[tid] = last
            q.append(tid)

    def _pop_tokens(self, obj_id: str, amount: int, src: Location) -> List[str]:
        q = self._loc_tokens[src][obj_id]
        if len(q) < amount:
            if self.strict:
                raise RuntimeError(f"Not enough tokens: need {amount} {obj_id} from {src}, have {len(q)}")
            # recovery: fabricate missing units from INIT (keeps sim running)
            missing = amount - len(q)
            self._spawn_tokens(obj_id, missing, src, last=self.INIT_NODE_ID)

        out = [q.popleft() for _ in range(amount)]
        return out

    def _add_deps_from_tokens(self, node_id: str, token_ids: List[str]) -> None:
        for tid in token_ids:
            prev = self._token_last.get(tid)
            if prev is not None and prev != node_id:
                self._edges[node_id].add(prev)

    def _apply_move(self, node_id: str, obj_id: str, amount: int, src: Location, dst: Location) -> None:
        tids = self._pop_tokens(obj_id, amount, src)
        self._add_deps_from_tokens(node_id, tids)
        for tid in tids:
            self._token_last[tid] = node_id
        self._loc_tokens[dst][obj_id].extend(tids)

    def _apply_spawn(self, node_id: str, obj_id: str, amount: int, dst: Location) -> None:
        self._spawn_tokens(obj_id, amount, dst, last=node_id)

    def _apply_consume(self, node_id: str, obj_id: str, amount: int, src: Location) -> None:
        tids = self._pop_tokens(obj_id, amount, src)
        self._add_deps_from_tokens(node_id, tids)
        for tid in tids:
            self._token_last[tid] = node_id

    def _apply_utilize(self, node_id: str, obj_id: str, amount: int, src: Location) -> None:
        tids = self._pop_tokens(obj_id, amount, src)
        self._add_deps_from_tokens(node_id, tids)
        for tid in tids[::-1]:
            self._loc_tokens[src][obj_id].appendleft(tid)

    def _to_dot_action(self) -> str:
        lines = ["digraph deps {", '  rankdir="LR";', "  node [shape=box];"]
        for nid, meta in self._nodes.items():
            if nid == self.INIT_NODE_ID:
                continue
            base = f"step {meta.step}\\n{meta.actor}\\n{meta.rule} ({meta.seq})"
            hint = self._node_hints.get(nid)
            if hint:
                base += "\\n" + self._esc_dot_label(hint)
            label = base
            lines.append(f'  "{nid}" [label="{label}"];')

        for u, vs in self._edges.items():
            if u == self.INIT_NODE_ID:
                continue
            for v in vs:
                if v == self.INIT_NODE_ID:
                    continue
                lines.append(f'  "{v}" -> "{u}";')
        lines.append("}")
        return "\n".join(lines)

    def _steps_present(self) -> Set[int]:
        return {m.step for nid, m in self._nodes.items() if nid != self.INIT_NODE_ID and m.step >= 0}

    def _collapse_edges_by_step(self) -> Dict[str, Set[str]]:
        step_edges: Dict[str, Set[str]] = defaultdict(set)

        def step_node(s: int) -> str:
            return f"S_{s}"

        # map action node -> step node
        def to_step(nid: str) -> str:
            if nid == self.INIT_NODE_ID:
                return "S_INIT"
            return step_node(self._nodes[nid].step)

        for u, vs in self._edges.items():
            if u == self.INIT_NODE_ID:
                continue
            su = to_step(u)
            for v in vs:
                if v == self.INIT_NODE_ID:
                    continue
                sv = to_step(v)
                if su != sv:
                    step_edges[su].add(sv)

        return step_edges

    def _to_dot_step(self) -> str:
        step_edges = self._collapse_edges_by_step()
        lines = ["digraph deps {", '  rankdir="LR";', "  node [shape=box];"]

        for s in sorted(self._steps_present()):
            hints = self._step_hints.get(s, [])
            if hints:
                # cap lines so the graph doesn't explode
                shown = hints[: self._max_step_hint_lines]
                extra = len(hints) - len(shown)
                hint_block = "\\n".join(self._esc_dot_label(h) for h in shown)
                if extra > 0:
                    hint_block += f"\\n... (+{extra} more)"
                label = f"step {s}\\n{hint_block}"
            else:
                label = f"step {s}"

            lines.append(f'  "S_{s}" [label="{label}"];')

        for u, vs in step_edges.items():
            for v in vs:
                lines.append(f'  "{v}" -> "{u}";')

        lines.append("}")
        return "\n".join(lines)

    def to_axis_svg(
        self,
        *,
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
        present = sorted(self._steps_present())
        max_step = max(present) if present else 0

        if show_all_steps:
            axis_steps = list(range(0, max_step + 1))
        else:
            axis_steps = present

        step_to_idx = {s: i for i, s in enumerate(axis_steps)}

        def stepnode_to_step(sn: str) -> int:
            return int(sn.split("_", 1)[1])

        step_edges = self._collapse_edges_by_step()

        interval_info: Dict[tuple[int, int], Dict[str, Any]] = {}
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

        max_above = -1
        max_below = -1
        for _, _, lvl in assigned:
            if is_above(lvl):
                max_above = max(max_above, above_level(lvl))
            else:
                max_below = max(max_below, above_level(lvl))

        n = len(axis_steps)
        width = margin_x * 2 + (n - 1) * step_px + 60

        top_room = (max_above + 1) * level_gap + base_arc if max_above >= 0 else base_arc
        bottom_room = (max_below + 1) * level_gap + base_arc if max_below >= 0 else base_arc
        axis_y = margin_y + top_room
        height = axis_y + bottom_room + margin_y + 60

        def x_at(i: int) -> float:
            return float(margin_x + i * step_px)

        def esc(s: str) -> str:
            return (s.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;"))

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
                f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{arc_color}"/></marker>'
            )
            parts.append(
                f'<marker id="axisArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{axis_color}"/></marker>'
            )
        parts.append("</defs>")

        parts.append(f'<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>')

        if banded and n >= 2:
            for i in range(n - 1):
                if i % 2 == 1:
                    x_left = x_at(i)
                    parts.append(f'<rect class="band" x="{x_left}" y="0" width="{step_px}" height="{height}"/>')

        x0 = x_at(0)
        x1 = x_at(n - 1) + 40.0
        if arrows:
            parts.append(f'<line class="axis" x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" marker-end="url(#axisArrow)"/>')
        else:
            parts.append(f'<line class="axis" x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}"/>')

        if show_grid:
            for i in range(n):
                xi = x_at(i)
                parts.append(f'<line class="grid" x1="{xi}" y1="{margin_y/2}" x2="{xi}" y2="{height - margin_y/2}"/>')

        for i, s in enumerate(axis_steps):
            xi = x_at(i)
            parts.append(
                f'<line class="tick" x1="{xi}" y1="{axis_y - tick_len/2}" x2="{xi}" y2="{axis_y + tick_len/2}"/>'
            )
            label = f"{s}"
            parts.append(
                f'<text class="label" x="{xi}" y="{axis_y + tick_len/2 + font_size + 8}" text-anchor="middle">{esc(label)}</text>'
            )
            parts.append(f'<circle class="dot" cx="{xi}" cy="{axis_y}" r="{dot_radius}"/>')

        for a, b, lvl in assigned:
            info = interval_info[(a, b)]
            lr = bool(info["lr"])
            rl = bool(info["rl"])

            left_i, right_i = a, b
            left_x, right_x = x_at(left_i), x_at(right_i)

            hlevel = above_level(lvl)
            arc_h = base_arc + hlevel * level_gap
            cy = axis_y - arc_h if is_above(lvl) else axis_y + arc_h
            midx = (left_x + right_x) / 2.0

            draw_l2r = lr and not rl
            draw_r2l = rl and not lr

            if draw_l2r:
                sx, ex = left_x, right_x
            elif draw_r2l:
                sx, ex = right_x, left_x
            else:
                sx, ex = left_x, right_x

            cls = "arc arcBelow" if not is_above(lvl) else "arc"
            marker = ' marker-end="url(#arrow)"' if arrows and (draw_l2r or draw_r2l) else ""
            title_txt = ""
            if info["examples"]:
                ex0 = info["examples"][0]
                title_txt = f"{ex0[0]} -> {ex0[1]}"
            else:
                title_txt = "dependency"

            parts.append(f'<path class="{cls}" d="M {sx} {axis_y} Q {midx} {cy} {ex} {axis_y}"{marker}>'
                        f'<title>{esc(str(title_txt))}</title></path>')

        parts.append("</svg>")
        return "\n".join(parts)
