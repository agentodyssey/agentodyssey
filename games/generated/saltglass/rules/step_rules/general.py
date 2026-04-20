import copy
import math
from utils import *
from games.generated.saltglass.rule import BaseStepRule, RuleContext, RuleResult, Event
from typing import Dict, Optional, Set, Tuple, List, Any
from games.generated.saltglass.world import NPC, Object, Area, Place, Path, Container, Writable
from games.generated.saltglass.agent import Agent
from tools.logger import get_logger


class AgentAttackUpdateStepRule(BaseStepRule):
    name = "agent_attack_update_step"
    description = "Update each agent's attack attribute based on their items in hand."
    priority = 1

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        for agent in env.agents:
            if not agent.items_in_hands:
                agent.attack = agent.min_attack
            else:
                agent.attack = agent.min_attack + max(
                    world.objects[get_def_id(obj_id)].attack
                    for obj_id in agent.items_in_hands.keys()
                    if get_def_id(obj_id) in world.objects
                )

            if agent.attack <= 0:
                res.add_feedback(agent.id, "Your attack power is too weak to damage anything.\n")
                return


class SaltGlazeStepRule(BaseStepRule):
    name = "salt_glaze_step"
    description = (
        "Under the open salt sky, carrying glass may slowly draw a thin salt glaze over one held or "
        "equipped item, temporarily weakening it until the agent reaches an interior observatory area or dies."
    )
    priority = 4

    OPEN_SALT_AREAS = {"hall", "yard", "flats", "basin"}
    INTERIOR_OBSERVATORY_AREAS = {"library", "orrery", "spire"}
    GLASS_NAME_TOKENS = ("glass", "lens", "saltglass", "brine")
    GLAZE_CHANCE = 0.2

    def _is_open_salt_sky_area(self, area) -> bool:
        area_name = str(getattr(area, "name", "") or "").strip().lower()
        return area_name in self.OPEN_SALT_AREAS

    def _is_interior_observatory_area(self, area) -> bool:
        area_name = str(getattr(area, "name", "") or "").strip().lower()
        return area_name in self.INTERIOR_OBSERVATORY_AREAS

    def _get_object_instance(self, world, obj_id: str):
        base_id = get_def_id(obj_id)
        if obj_id in getattr(world, "container_instances", {}):
            return world.container_instances[obj_id]
        if obj_id in getattr(world, "writable_instances", {}):
            return world.writable_instances[obj_id]
        if base_id in getattr(world, "objects", {}):
            return world.objects[base_id]
        return None

    def _is_glasslike_object(self, world, obj_id: str) -> bool:
        obj = self._get_object_instance(world, obj_id)
        if obj is None:
            return False

        name = str(getattr(obj, "name", "") or "").lower()
        desc = str(getattr(obj, "description", "") or "").lower()
        oid = str(getattr(obj, "id", "") or "").lower()

        return any(tok in name or tok in desc or tok in oid for tok in self.GLASS_NAME_TOKENS)

    def _has_held_glass(self, world, agent) -> bool:
        for obj_id, count in (agent.items_in_hands or {}).items():
            if int(count) > 0 and self._is_glasslike_object(world, obj_id):
                return True
        return False

    def _restore_glaze(self, env, world, agent, res: RuleResult, reason: str) -> None:
        glaze_state = env.curr_agents_state.setdefault("salt_glaze", {})
        agent_state = glaze_state.get(agent.id)
        if not agent_state or not agent_state.get("active"):
            return

        obj_id = agent_state.get("obj_id")
        stat = agent_state.get("stat")
        amount = int(agent_state.get("amount", 0))

        if obj_id and stat and amount > 0:
            obj = self._get_object_instance(world, obj_id)
            if obj is not None and hasattr(obj, stat):
                setattr(obj, stat, getattr(obj, stat) + amount)

        glaze_state[agent.id] = {
            "active": False,
            "obj_id": None,
            "stat": None,
            "amount": 0,
        }

        if reason == "observatory":
            res.add_feedback(
                agent.id,
                "Away from the open salt sky, the thin glaze loosens and flakes away from your gear.\n"
            )
        elif reason == "death":
            res.add_feedback(
                agent.id,
                "Death breaks the crusted glaze; the salt slips from what you carried.\n"
            )

        res.events.append(Event(
            type="salt_glaze_cleared",
            agent_id=agent.id,
            data={"reason": reason, "obj_id": obj_id, "stat": stat, "amount": amount},
        ))

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        glaze_state = env.curr_agents_state.setdefault("salt_glaze", {})
        for agent in env.agents:
            glaze_state.setdefault(agent.id, {
                "active": False,
                "obj_id": None,
                "stat": None,
                "amount": 0,
            })

            current_area_id = env.curr_agents_state["area"][agent.id]
            current_area = world.area_instances.get(current_area_id)
            if current_area is None:
                continue

            if agent.hp <= 0:
                self._restore_glaze(env, world, agent, res, reason="death")
                continue

            if self._is_interior_observatory_area(current_area):
                self._restore_glaze(env, world, agent, res, reason="observatory")
                continue

            if glaze_state[agent.id].get("active"):
                continue

            if not self._is_open_salt_sky_area(current_area):
                continue

            if not self._has_held_glass(world, agent):
                continue

            candidates: List[Tuple[str, str, int]] = []

            for obj_id, count in (agent.items_in_hands or {}).items():
                if int(count) <= 0:
                    continue
                obj = self._get_object_instance(world, obj_id)
                if obj is None:
                    continue
                base_id = get_def_id(obj_id)
                if base_id in getattr(world, "objects", {}):
                    obj_def = world.objects[base_id]
                    attack_val = int(getattr(obj_def, "attack", 0) or 0)
                    defense_val = int(getattr(obj_def, "defense", 0) or 0)
                else:
                    attack_val = int(getattr(obj, "attack", 0) or 0)
                    defense_val = int(getattr(obj, "defense", 0) or 0)

                if attack_val > 0:
                    candidates.append((obj_id, "attack", attack_val))
                if defense_val > 0:
                    candidates.append((obj_id, "defense", defense_val))

            for obj_id, count in (agent.equipped_items_in_limb or {}).items():
                if int(count) <= 0:
                    continue
                obj = self._get_object_instance(world, obj_id)
                if obj is None:
                    continue
                base_id = get_def_id(obj_id)
                if base_id in getattr(world, "objects", {}):
                    obj_def = world.objects[base_id]
                    attack_val = int(getattr(obj_def, "attack", 0) or 0)
                    defense_val = int(getattr(obj_def, "defense", 0) or 0)
                else:
                    attack_val = int(getattr(obj, "attack", 0) or 0)
                    defense_val = int(getattr(obj, "defense", 0) or 0)

                if attack_val > 0:
                    candidates.append((obj_id, "attack", attack_val))
                if defense_val > 0:
                    candidates.append((obj_id, "defense", defense_val))

            if not candidates:
                continue

            if env.rng.random() >= self.GLAZE_CHANCE:
                continue

            obj_id, stat, current_value = env.rng.choice(candidates)
            obj = self._get_object_instance(world, obj_id)
            if obj is None or not hasattr(obj, stat):
                continue

            reduction = max(1, int(math.ceil(current_value * 0.25)))
            new_value = max(0, int(getattr(obj, stat, 0)) - reduction)
            actual_reduction = int(getattr(obj, stat, 0)) - new_value
            if actual_reduction <= 0:
                continue

            setattr(obj, stat, new_value)
            glaze_state[agent.id] = {
                "active": True,
                "obj_id": obj_id,
                "stat": stat,
                "amount": actual_reduction,
            }

            obj_name = (
                world.container_instances[obj_id].name if obj_id in world.container_instances
                else world.writable_instances[obj_id].name if obj_id in world.writable_instances
                else world.objects[get_def_id(obj_id)].name if get_def_id(obj_id) in world.objects
                else obj_id
            )

            if stat == "attack":
                stat_text = "attack"
            else:
                stat_text = "defense"

            res.add_feedback(
                agent.id,
                f"The open salt sky answers the glass in your hands. A thin glaze creeps across {obj_name}, "
                f"dulling its {stat_text} by {actual_reduction} until you reach shelter in an observatory.\n"
            )
            res.events.append(Event(
                type="salt_glaze_formed",
                agent_id=agent.id,
                data={
                    "area_id": current_area_id,
                    "obj_id": obj_id,
                    "stat": stat,
                    "amount": actual_reduction,
                },
            ))


class ForgeAlertStepRule(BaseStepRule):
    name = "forge_alert_step"
    description = (
        "When a crafting station (bench or kiln) is present in the current area and the agent "
        "carries all ingredients for at least one recipe, a short alert names the craftable item. "
        "If the agent has recently crafted in this area the alert is suppressed to avoid spam."
    )
    priority = 5

    STATION_IDS = {"obj_bench", "obj_kiln", "bench", "kiln"}
    COOLDOWN_STEPS = 8  # suppress repeated alerts for the same agent+area

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        fa = aux.setdefault("forge_alert", {})
        fa.setdefault("last_alert", {})  # (agent_id, area_id) -> step
        return fa

    def _area_has_station(self, world, area) -> bool:
        for obj_id in (getattr(area, "objects", {}) or {}):
            if get_def_id(obj_id) in self.STATION_IDS or obj_id in self.STATION_IDS:
                return True
        return False

    def _agent_total_items(self, world, agent) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for oid, cnt in (agent.items_in_hands or {}).items():
            base = get_def_id(oid)
            totals[base] = totals.get(base, 0) + int(cnt)
        if agent.inventory.container:
            for oid, cnt in (agent.inventory.items or {}).items():
                base = get_def_id(oid)
                totals[base] = totals.get(base, 0) + int(cnt)
        for oid in (agent.items_in_hands or {}):
            if oid in getattr(world, "container_instances", {}):
                for coid, cnt in (world.container_instances[oid].inventory or {}).items():
                    base = get_def_id(coid)
                    totals[base] = totals.get(base, 0) + int(cnt)
        return totals

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        fa = self._ensure_state(world)
        current_step = int(env.steps)

        for agent in env.agents:
            area_id = env.curr_agents_state["area"][agent.id]
            area = world.area_instances.get(area_id)
            if area is None:
                continue

            if not self._area_has_station(world, area):
                continue

            key = (agent.id, area_id)
            last = fa["last_alert"].get(str(key), -self.COOLDOWN_STEPS - 1)
            if current_step - last < self.COOLDOWN_STEPS:
                continue

            held = self._agent_total_items(world, agent)
            craftable_names: List[str] = []
            for oid, obj in world.objects.items():
                craft_ings = getattr(obj, "craft_ingredients", {}) or {}
                if not craft_ings:
                    continue
                if getattr(obj, "quest", False):
                    continue
                if all(held.get(ing_id, 0) >= int(cnt) for ing_id, cnt in craft_ings.items()):
                    craftable_names.append(obj.name)

            if not craftable_names:
                continue

            fa["last_alert"][str(key)] = current_step
            listing = ", ".join(craftable_names[:3])
            suffix = f" (+{len(craftable_names) - 3} more)" if len(craftable_names) > 3 else ""
            res.add_feedback(
                agent.id,
                f"The forge here hums — you carry enough to craft: {listing}{suffix}.\n"
            )


class CollectUnattendedStepRule(BaseStepRule):
    name = "collect_unattended_step"
    description = (
        "Items dropped on the ground in open salt areas slowly sink into the sand. "
        "After a threshold of idle steps with no agent present, a fraction of loose "
        "non-quest objects vanish. Merchant areas are exempt."
    )
    priority = 6

    OPEN_SALT_AREAS = {"hall", "yard", "flats", "basin"}

    def __init__(self) -> None:
        super().__init__()
        self._min_idle = 20
        self._max_idle = 35
        self._collect_frac = 0.35  # fraction of each stack to remove

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        cu = aux.setdefault("collect_unattended", {})
        cu.setdefault("idle_steps", {})
        cu.setdefault("thresholds", {})
        return cu

    def _is_open_salt(self, area) -> bool:
        return str(getattr(area, "name", "") or "").strip().lower() in self.OPEN_SALT_AREAS

    @staticmethod
    def _has_merchant(world, area) -> bool:
        for npc_id in list(getattr(area, "npcs", []) or []):
            inst = world.npc_instances.get(npc_id)
            if inst is not None and getattr(inst, "role", None) == "merchant":
                return True
        return False

    @staticmethod
    def _agents_by_area(env) -> Dict[str, list]:
        by_area: Dict[str, list] = {}
        for agent in env.agents:
            aid = env.curr_agents_state["area"][agent.id]
            by_area.setdefault(aid, []).append(agent)
        return by_area

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        cu = self._ensure_state(world)
        agents_here = self._agents_by_area(env)

        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_active = bool(tutorial_room) and not bool(tutorial_room.get("removed", False))
        tutorial_area = tutorial_room.get("area_id")

        for area_id, area in world.area_instances.items():
            if not self._is_open_salt(area):
                continue
            if tutorial_active and area_id == tutorial_area:
                continue
            if self._has_merchant(world, area):
                continue

            # reset if agents present
            if area_id in agents_here and agents_here[area_id]:
                cu["idle_steps"][area_id] = 0
                continue

            idle = int(cu["idle_steps"].get(area_id, 0)) + 1
            cu["idle_steps"][area_id] = idle

            if area_id not in cu["thresholds"]:
                cu["thresholds"][area_id] = env.rng.randint(self._min_idle, self._max_idle)
            thresh = int(cu["thresholds"][area_id])

            if idle < thresh:
                continue

            removed_any = False
            objs = getattr(area, "objects", {}) or {}
            for oid in list(objs.keys()):
                cnt = int(objs.get(oid, 0))
                if cnt <= 0:
                    continue
                # skip quest objects
                base = get_def_id(oid)
                if base in world.objects and getattr(world.objects[base], "quest", False):
                    continue
                remove = max(1, int(math.floor(cnt * self._collect_frac)))
                objs[oid] = cnt - remove
                if objs[oid] <= 0:
                    del objs[oid]
                removed_any = True

            if removed_any:
                res.events.append(Event(
                    type="unattended_collected",
                    agent_id="env",
                    data={"area_id": area_id},
                ))

            cu["idle_steps"][area_id] = 0
            cu["thresholds"][area_id] = env.rng.randint(self._min_idle, self._max_idle)


class ResonanceImprintStepRule(BaseStepRule):
    name = "resonance_imprint_step"
    description = (
        "Observatory areas remember the last agent who used 'listen' or 'refract' there. "
        "When a different agent enters the same observatory, they receive a faint echo of the "
        "previous user's hint, creating an indirect information trail between agents."
    )
    priority = 5

    OBSERVATORY_AREAS = {"library", "orrery", "spire"}

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        ri = aux.setdefault("resonance_imprint", {})
        ri.setdefault("imprints", {})  # area_id -> {agent_id, hint_text, step}
        return ri

    def _is_observatory(self, area) -> bool:
        return str(getattr(area, "name", "") or "").strip().lower() in self.OBSERVATORY_AREAS

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        ri = self._ensure_state(world)
        imprints = ri["imprints"]

        # 1) Record new imprints from listen/refract events this step.
        for ev in list(res.events):
            et = getattr(ev, "type", None)
            if et not in ("saltglass_listened", "refracted"):
                continue
            data = getattr(ev, "data", {}) or {}
            area_id = data.get("area_id")
            if not area_id:
                continue
            area = world.area_instances.get(area_id)
            if area is None or not self._is_observatory(area):
                continue

            # build a terse echo from the event data
            hint_type = data.get("hint_type", "")
            if hint_type == "nearby_object":
                obj_id = data.get("obj_id", "something")
                base = get_def_id(obj_id)
                name = world.objects[base].name if base in world.objects else obj_id
                echo = f"a faint impression of {name} nearby"
            elif hint_type == "locked_path":
                to_area = data.get("to_area", "")
                neighbor = world.area_instances.get(to_area)
                n_name = getattr(neighbor, "name", to_area) if neighbor else to_area
                echo = f"a sealed road toward {n_name}"
            elif hint_type == "enemy_rhythm":
                npc_id = data.get("npc_id", "")
                npc = world.npc_instances.get(npc_id)
                n_name = getattr(npc, "name", npc_id) if npc else npc_id
                echo = f"a combat rhythm left by {n_name}"
            elif et == "refracted":
                echo = "refracted light traces still hanging in the glass"
            else:
                echo = "a whisper too faint to read"

            imprints[area_id] = {
                "agent_id": ev.agent_id,
                "echo": echo,
                "step": int(env.steps),
            }

        # 2) Deliver echoes to agents entering an observatory with a stored imprint.
        for agent in env.agents:
            area_id = env.curr_agents_state["area"][agent.id]
            area = world.area_instances.get(area_id)
            if area is None or not self._is_observatory(area):
                continue

            imp = imprints.get(area_id)
            if not imp:
                continue
            if imp["agent_id"] == agent.id:
                continue  # own imprint — skip

            age = int(env.steps) - int(imp.get("step", 0))
            if age > 80:
                # imprint has faded
                continue

            # only deliver once per entry
            delivered_key = f"resonance_delivered_{area_id}"
            delivered = env.curr_agents_state.setdefault("resonance_flags", {})
            last_delivered = delivered.get((agent.id, area_id))
            if last_delivered == imp["step"]:
                continue
            delivered[(agent.id, area_id)] = imp["step"]

            res.add_feedback(
                agent.id,
                f"The observatory glass still hums with a resonance left behind — {imp['echo']}.\n"
            )
            res.events.append(Event(
                type="resonance_echo_received",
                agent_id=agent.id,
                data={"area_id": area_id, "from_agent": imp["agent_id"], "echo": imp["echo"]},
            ))


class SaltMirageStepRule(BaseStepRule):
    name = "salt_mirage_step"
    description = (
        "Open salt-sky areas periodically inject phantom objects into their ground inventory "
        "that look real in area descriptions but vanish when the agent tries to pick them up."
    )
    priority = 6

    OPEN_SALT_AREAS = {"hall", "yard", "flats", "basin"}
    MIRAGE_CHANCE = 0.12  # per eligible area per step
    MIRAGE_LIFETIME = (8, 18)  # steps before a mirage fades on its own
    MIRAGE_PREFIX = "_mirage_"

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        sm = aux.setdefault("salt_mirage", {})
        sm.setdefault("active", {})  # area_id -> {obj_id, fake_name, born_step, lifetime}
        return sm

    def _is_open_salt(self, area) -> bool:
        return str(getattr(area, "name", "") or "").strip().lower() in self.OPEN_SALT_AREAS

    def _pick_phantom_name(self, world, env) -> Optional[str]:
        """Choose a plausible object name from world definitions to use as a mirage."""
        candidates = []
        for oid, obj in world.objects.items():
            if getattr(obj, "quest", False):
                continue
            candidates.append(obj.name)
        if not candidates:
            return None
        return env.rng.choice(candidates)

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        sm = self._ensure_state(world)
        active = sm["active"]
        current_step = int(env.steps)

        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_active = bool(tutorial_room) and not bool(tutorial_room.get("removed", False))
        tutorial_area = tutorial_room.get("area_id")

        # 1) Expire old mirages.
        for area_id in list(active.keys()):
            m = active[area_id]
            if current_step - int(m.get("born_step", 0)) >= int(m.get("lifetime", 999)):
                # remove phantom from area inventory
                area = world.area_instances.get(area_id)
                if area is not None:
                    objs = getattr(area, "objects", {})
                    fake_id = m.get("obj_id")
                    if fake_id and fake_id in objs:
                        del objs[fake_id]
                del active[area_id]

        # 2) Spawn new mirages in eligible areas.
        agents_by_area: Dict[str, list] = {}
        for agent in env.agents:
            aid = env.curr_agents_state["area"][agent.id]
            agents_by_area.setdefault(aid, []).append(agent)

        for area_id, area in world.area_instances.items():
            if not self._is_open_salt(area):
                continue
            if tutorial_active and area_id == tutorial_area:
                continue
            if area_id in active:
                continue  # one mirage at a time per area

            if env.rng.random() >= self.MIRAGE_CHANCE:
                continue

            fake_name = self._pick_phantom_name(world, env)
            if not fake_name:
                continue

            fake_id = f"{self.MIRAGE_PREFIX}{area_id}_{current_step}"
            lifetime = env.rng.randint(*self.MIRAGE_LIFETIME)

            # inject into area inventory so it shows up in observations
            objs = getattr(area, "objects", {})
            objs[fake_id] = 1

            active[area_id] = {
                "obj_id": fake_id,
                "fake_name": fake_name,
                "born_step": current_step,
                "lifetime": lifetime,
            }

        # 3) Intercept pick-up attempts on mirages.
        #    Because action rules run before step rules, we check events for
        #    failed pick-ups via the mirage prefix.  However, the simpler
        #    approach is to register the fake IDs so the PickUpRule's
        #    resolution will fail naturally (no base obj_id match).
        #    We add feedback when an agent is in an area with an active mirage.
        for area_id, m in list(active.items()):
            agents = agents_by_area.get(area_id, [])
            if not agents:
                continue
            # Check if any pick_up event targeted the mirage this step.
            for ev in list(res.events):
                if getattr(ev, "type", None) != "object_picked_up":
                    continue
                data = getattr(ev, "data", {}) or {}
                if data.get("obj_id") == m["obj_id"]:
                    # This shouldn't normally happen since there's no base def,
                    # but clean up just in case.
                    area = world.area_instances.get(area_id)
                    if area is not None:
                        objs = getattr(area, "objects", {})
                        if m["obj_id"] in objs:
                            del objs[m["obj_id"]]
                    del active[area_id]
                    for a in agents:
                        res.add_feedback(
                            a.id,
                            f"Your hand closes on empty air — the {m['fake_name']} was only a salt mirage.\n"
                        )
                    res.events.append(Event(
                        type="mirage_dispelled",
                        agent_id=ev.agent_id,
                        data={"area_id": area_id, "fake_name": m["fake_name"]},
                    ))
                    break


class CombatRhythmStepRule(BaseStepRule):
    name = "combat_rhythm_step"
    description = "Process NPC attack rhythm actions for agents in active combat."
    priority = 2

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        _tut_area = (world.auxiliary.get("tutorial_room") or {}).get("area_id")

        for agent in env.agents:
            if agent.hp <= 0:
                continue
            
            agent_area_id = env.curr_agents_state["area"][agent.id]
            if _tut_area and agent_area_id == _tut_area:
                continue
            active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
            agent_is_defending = env.curr_agents_state.get("agent_defending", {}).get(agent.id, False)
            
            if not active_combats:
                continue
            
            npcs_to_remove = []
            for npc_id, combat_state in list(active_combats.items()):
                if combat_state.get("area_id") != agent_area_id:
                    npcs_to_remove.append(npc_id)
                    continue
                
                if npc_id not in world.npc_instances:
                    npcs_to_remove.append(npc_id)
                    continue
                    
                npc_instance = world.npc_instances[npc_id]
                
                if npc_instance.hp <= 0:
                    npcs_to_remove.append(npc_id)
                    continue
                
                current_area = world.area_instances.get(agent_area_id)
                if current_area is None or npc_id not in current_area.npcs:
                    npcs_to_remove.append(npc_id)
                    continue
                
                rhythm = npc_instance.combat_pattern
                if not rhythm:
                    continue
                    
                rhythm_index = combat_state.get("rhythm_index", 0)
                current_action = rhythm[rhythm_index % len(rhythm)]
                
                if current_action == "defend":
                    res.add_feedback(agent.id, f"{npc_instance.name} is defending.\n")
                elif current_action == "wait":
                    res.add_feedback(agent.id, f"{npc_instance.name} is waiting.\n")
                elif current_action == "attack":
                    res.add_feedback(agent.id, f"{npc_instance.name} is attacking.\n")
                
                if current_action == "attack":
                    base_damage = max(0, npc_instance.attack_power - agent.defense)
                    
                    if agent_is_defending:
                        damage = int(base_damage * 0.1)  # 10% damage if defending 
                        res.add_feedback(
                            agent.id,
                            f"{env.person_verbalized['subject_pronoun']} was attacked by {npc_instance.name} "
                            f"and lost {damage} HP.\n"
                        )
                    else:
                        damage = base_damage
                        res.add_feedback(
                            agent.id,
                            f"{env.person_verbalized['subject_pronoun']} was attacked by {npc_instance.name} "
                            f"and lost {damage} HP.\n"
                        )
                    
                    agent.hp -= damage
                    
                    res.events.append(Event(
                        type="agent_damaged",
                        agent_id=agent.id,
                        data={
                            "npc_id": npc_id,
                            "damage": damage,
                            "area_id": agent_area_id,
                            "rhythm_action": "attack",
                            "agent_defended": agent_is_defending,
                        },
                    ))
                    
                    if agent.hp <= 0:
                        res.add_feedback(
                            agent.id,
                            f"{env.person_verbalized['subject_pronoun']} have been defeated by {npc_instance.name}!\n"
                        )
                        res.events.append(Event(
                            type="agent_defeated_in_combat",
                            agent_id=agent.id,
                            data={"npc_id": npc_id, "area_id": agent_area_id},
                        ))
                        env.curr_agents_state["active_combats"][agent.id] = {}
                        # reset stamina when leaving combat
                        env.curr_agents_state["agent_stamina"][agent.id] = 1.0
                        env.curr_agents_state["agent_consecutive_attacks"][agent.id] = 0
                        break
                
                combat_state["rhythm_index"] = (rhythm_index + 1) % len(rhythm)
            
            for npc_id in npcs_to_remove:
                if npc_id in env.curr_agents_state["active_combats"][agent.id]:
                    del env.curr_agents_state["active_combats"][agent.id][npc_id]
            
            # reset stamina when no more active combats
            if not env.curr_agents_state["active_combats"].get(agent.id, {}):
                env.curr_agents_state["agent_stamina"][agent.id] = 1.0
                env.curr_agents_state["agent_consecutive_attacks"][agent.id] = 0


class ActiveAttackStepRule(BaseStepRule):
    name = "active_attack_step"
    description = "Enemy NPCs may randomly attack agents between steps, causing HP loss. Does not apply to NPCs already in rhythm-based combat."
    priority = 6

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        _tut_area = (world.auxiliary.get("tutorial_room") or {}).get("area_id")

        # build enemy NPC list per area
        enemy_npcs: Dict[str, list[str]] = {area_id: [] for area_id in world.area_instances}
        for area_id, area in world.area_instances.items():
            for npc_id in area.npcs:
                npc_instance = world.npc_instances[npc_id]
                if npc_instance.enemy:
                    enemy_npcs[area_id].append(npc_id)
        area_levels = set(area.level for area in world.area_instances.values())
        base_attack_prob, max_attack_prob = 0.05, 0.15
        if max(area_levels) <= 1:
            attack_prob_by_area_level = {1: base_attack_prob}
        else:
            attack_prob_by_area_level = {i : base_attack_prob + (i - 1) * 
                                         (max_attack_prob - base_attack_prob) / (max(area_levels) - 1) for i in area_levels}

        for agent in env.agents:
            if agent.hp <= 0:
                continue
            agent_area = env.curr_agents_state["area"][agent.id]
            if _tut_area and agent_area == _tut_area:
                continue
            
            active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
            npcs_in_combat = set(active_combats.keys())

            for npc_id in enemy_npcs[agent_area].copy():
                # skip NPCs already in rhythm-based combat
                if npc_id in npcs_in_combat:
                    continue
                
                # if agent is already in combat with other NPCs, reduce attack rate
                agent_in_combat = len(npcs_in_combat) > 0
                if agent_in_combat:
                    base_prob = 0.025
                else:
                    base_prob = attack_prob_by_area_level[world.area_instances[agent_area].level]
                
                if env.rng.random() > base_prob:
                    continue
                
                npc_instance = world.npc_instances[npc_id]
                # active NPC attack has less damage than NPC attack in combat
                damage = max(0, math.floor(npc_instance.attack_power / 2) - agent.defense)
                
                # apply defending modifier if agent is defending
                agent_is_defending = env.curr_agents_state.get("agent_defending", {}).get(agent.id, False)
                if agent_is_defending:
                    damage = int(damage * 0.1)  # 10% damage if defending
                
                agent.hp -= damage
                enemy_npcs[agent_area].remove(npc_id)

                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} was attacked by {npc_instance.name} "
                    f"and lost {damage} HP.\n"
                )
                res.events.append(Event(
                    type="agent_damaged",
                    agent_id=agent.id,
                    data={"npc_id": npc_id, "damage": damage, "area_id": agent_area},
                ))
        
        # reset defending state for all agents (after both combat and active attacks have used it)
        for agent in env.agents:
            env.curr_agents_state["agent_defending"][agent.id] = False


class NewCraftableFeedbackStepRule(BaseStepRule):
    name = "new_craftable_feedback_step"
    description = "Provides feedback on crafting recipes unlocked by newly acquired items."
    priority = 5

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        ing_to_obj_map = world.auxiliary.get("ing_to_obj_map", {})

        for agent in env.agents:
            tutorial_room = world.auxiliary.get("tutorial_room") or {}
            tutorial_removed = bool(tutorial_room.get("removed", False))
            if tutorial_room and not tutorial_removed:
                continue

            objects_acquired = env.curr_agents_state["objects_acquired"][agent.id]
            new_objects_acquired = set()
            container_instances = [
                world.container_instances[oid] 
                for oid in agent.items_in_hands.keys()
                if oid in world.container_instances
            ]

            for obj_id in agent.items_in_hands:
                if not get_def_id(obj_id) in objects_acquired:
                    new_objects_acquired.add(get_def_id(obj_id))
            # check inventory and container instances in hands
            for obj_id in agent.equipped_items_in_limb:
                if not get_def_id(obj_id) in objects_acquired:
                    new_objects_acquired.add(get_def_id(obj_id))
            if agent.inventory.container:
                for obj_id in agent.inventory.items:
                    if not get_def_id(obj_id) in objects_acquired:
                        new_objects_acquired.add(get_def_id(obj_id))
            for container in container_instances:
                for obj_id in container.inventory:
                    if not get_def_id(obj_id) in objects_acquired:
                        new_objects_acquired.add(get_def_id(obj_id))

            show_crafting_recipes = env.world_definition.get("features", {}).get("show_crafting_recipes", True)
            lines = ""
            for obj_id in new_objects_acquired:
                if obj_id in ing_to_obj_map:
                    lines += (
                        f"{env.person_verbalized['subject_pronoun']} have acquired a new item: "
                        f"{world.objects[obj_id].name}. It can be used to craft: "
                    )
                    if show_crafting_recipes:
                        for craftable_obj_id in ing_to_obj_map[obj_id]:
                            craftable_obj = world.objects[craftable_obj_id]
                            ingredient_str = f"{craftable_obj.name} ("
                            for ingredient_id, qty in craftable_obj.craft_ingredients.items():
                                ingredient_name = world.objects[ingredient_id].name if ingredient_id in world.objects else "Unknown"
                                ingredient_str += f"{qty} {ingredient_name}, "
                            lines += ingredient_str.rstrip(", ") + "), "
                    else:
                        obj_names = [world.objects[oid].name for oid in ing_to_obj_map[obj_id]]
                        lines += ", ".join(obj_names)
            
            lines = lines.rstrip(", ") + "\n"
            env.curr_agents_state["objects_acquired"][agent.id].extend(new_objects_acquired)
            res.add_feedback(agent.id, lines)
            res.events.append(Event(
                type="new_crafting_recipe_seen",
                agent_id=agent.id,
                data={"object_ids": list(new_objects_acquired)},
            ))


# class MerchantInfoStepRule(BaseStepRule):
#     name = "merchant_info_step"
#     description = "Tells agents what nearby merchants offer when they enter an area with merchants."
#     priority = 4

#     def __init__(self):
#         super().__init__()
#         self.agents_last_visited = {}

#     def apply(self, ctx: RuleContext, res: RuleResult) -> None:
#         env, world = ctx.env, ctx.world

#         for agent in env.agents:
#             agent_area = env.curr_agents_state["area"][agent.id]
#             if agent.id not in self.agents_last_visited:
#                 self.agents_last_visited[agent.id] = agent_area
#             elif self.agents_last_visited[agent.id] == agent_area:
#                 continue
            
#             self.agents_last_visited[agent.id] = agent_area
#             area = world.area_instances[agent_area]
#             merchant_npcs = [
#                 world.npc_instances[npc_id] 
#                 for npc_id in area.npcs 
#                 if npc_id in world.npc_instances and world.npc_instances[npc_id].role == "merchant"
#             ]
#             if not merchant_npcs:
#                 continue

#             lines = ""
#             for merchant in merchant_npcs:
#                 lines += f"Merchant {merchant.name} offers the following items for sale: "
#                 for item_id, count in merchant.inventory.items():
#                     if item_id in world.objects:
#                         item_name = world.objects[item_id].name
#                         value_each = world.objects[item_id].value
#                     else:
#                         item_name = "Unknown"
#                         value_each = 0
#                     lines += f"{count} {item_name} (each for {value_each} coins), "
#                 lines = lines.rstrip(", ") + "\n"
#             lines += "\n"

#             res.add_feedback(agent.id, lines)
#             res.events.append(Event(
#                 type="merchant_info_provided",
#                 agent_id=agent.id,
#                 data={"area_id": agent_area, "merchant_ids": [m.id for m in merchant_npcs]},
#             ))


class DeathAndRespawnStepRule(BaseStepRule):
    name = "death_and_respawn_step"
    description = "Agents who die (HP <= 0) drop all items at the place they died and respawn at the starting point with full health."
    priority: int = 7
    
    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        for agent in env.agents:
            if agent.hp > 0:
                continue

            agent_area = env.curr_agents_state["area"][agent.id]
            current_area = world.area_instances[agent_area]

            res.add_feedback(agent.id, f"{env.person_verbalized['subject_pronoun']} have died.\n")

            # drop items
            for obj_id, count in agent.items_in_hands.items():
                if count > 0:
                    current_area.objects[obj_id] = current_area.objects.get(obj_id, 0) + count
                    res.track_move(
                        "env", obj_id, count, 
                        src=res.tloc("hand", agent.id),
                        dst=res.tloc("area", current_area.id),
                    )
            for obj_id, count in agent.equipped_items_in_limb.items():
                if count > 0:
                    current_area.objects[obj_id] = current_area.objects.get(obj_id, 0) + count
                    res.track_move(
                        "env", obj_id, count, 
                        src=res.tloc("hand", agent.id),
                        dst=res.tloc("area", current_area.id),
                    )

            agent.inventory.container = None
            agent.items_in_hands = {}
            agent.equipped_items_in_limb = {}
            agent.defense = 0

            # respawn
            agent.hp = agent.max_hp
            env.curr_agents_state["area"][agent.id] = env.world_definition["initializations"]["spawn"]["area"]

            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} revived at the starting point with full health but "
                f"lost all items in hand, equipped items, and inventory items at the place "
                f"{env.person_verbalized['subject_pronoun']} died.\n"
            )
            res.events.append(Event(
                type="agent_died",
                agent_id=agent.id,
                data={"area_id": agent_area},
            ))


class SideQuestStepRule(BaseStepRule):
    name = "side_quest_step"
    description = "Four-scenario side quests (collect, craft, talk, trade) with objective guide NPC."
    priority = 3

    OBJECTIVE_NPC_BASE_ID = "npc_side_quest_guide"
    OBJECTIVE_NPC_NAME_POOL = [
        "michael_hart", "sarah_cole", "james_owen", "jessica_pine",
        "daniel_cross", "emily_frost", "william_brooks", "olivia_grant",
    ]
    OBJECTIVE_NPC_MAX_HP = 10**9
    OBJECTIVE_NPC_ATK = 10**6

    required_npcs: List[Dict] = [
        {
            "type": "npc", "id": OBJECTIVE_NPC_BASE_ID, "name": "side_quest_guide",
            "enemy": False, "unique": True, "role": "guide", "quest": True,
            "base_attack_power": OBJECTIVE_NPC_ATK, "slope_attack_power": 0,
            "base_hp": OBJECTIVE_NPC_MAX_HP, "slope_hp": 0, "objects": [],
            "description": "a helpful guide who tracks your side quest progress.",
        },
    ]

    REWARD_COINS = {
        "collect": (1, 3),
        "craft": (3, 6),
        "talk": (1, 3),
        "trade": (3, 6),
    }

    # minimum areas explored (outside spawn) to unlock side quests
    MIN_EXPLORED_AREAS = 2

    def __init__(self) -> None:
        super().__init__()
        self._initialized = False
        self._world_ref = None
        self._area_to_place: Dict[str, str] = {}
        self._progress: Dict[str, Dict[str, Any]] = {}  # agent_id -> progress dict

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        if "side_quest" not in env.world_definition.get("custom_events", []):
            return

        tutorial_enabled = "tutorial" in env.world_definition.get("custom_events", [])
        tutorial_room = world.auxiliary.get("tutorial_room") or {}

        env.curr_agents_state.setdefault("side_quest_progress", {})
        world.auxiliary.setdefault("side_quest", {})
        sq = world.auxiliary["side_quest"]
        sq.setdefault("guide_names", {})
        sq.setdefault("run_seed", None)

        if not self._initialized:
            self._world_ref = world
            self._area_to_place = world.auxiliary.get("area_to_place", {}) or {}
            if not self._area_to_place:
                for place_id, place in world.place_instances.items():
                    for area_id in getattr(place, "areas", []):
                        self._area_to_place[area_id] = place_id
                world.auxiliary["area_to_place"] = self._area_to_place

            self._register_required_npcs(world)
            self._ensure_guide_name_map_for_agents(env, world)
            self._initialized = True

        for agent in env.agents:
            aid = agent.id

            if tutorial_enabled and tutorial_room and not bool(tutorial_room.get("removed", False)):
                continue

            env.curr_agents_state["side_quest_progress"].setdefault(aid, None)

            if aid not in self._progress:
                persisted_prog = env.curr_agents_state["side_quest_progress"].get(aid)
                if isinstance(persisted_prog, dict) and persisted_prog:
                    self._progress[aid] = persisted_prog
                else:
                    self._init_agent_progress(aid)

            prog = self._progress[aid]
            guide_name = self._ensure_agent_guide_name(world, prog, aid)
            current_area_id = env.curr_agents_state["area"][aid]

            visited_areas = env.curr_agents_state.get("areas_visited", {}).get(aid, [])
            spawn_area = env.world_definition["initializations"]["spawn"]["area"]
            explored_non_spawn = [a for a in visited_areas if a != spawn_area]

            if not prog.get("unlocked", False):
                if len(explored_non_spawn) >= self.MIN_EXPLORED_AREAS:
                    prog["unlocked"] = True
                    prog["just_unlocked"] = True
                    prog["last_visited_areas_count"] = len(visited_areas)
                    # IMPORTANT: Ensure guide is created BEFORE generating tasks
                    # so that sq_guide_area is set and tasks can exclude guide's area
                    self._ensure_objective_guide(env, world, agent, prog, res, force_here=True, announce=True)
                    self._generate_all_tasks(env, world, agent, prog)
                    res.add_feedback(
                        aid,
                        f"\n=== SIDE QUESTS UNLOCKED ===\n"
                        f"Side quests are now available! Completing them will grant you {world.objects['obj_coin'].name if 'obj_coin' in world.objects else 'coin'}s.\n"
                        f"Talk to {guide_name} to see your current tasks.\n"
                        f"============================\n"
                    )
                else:
                    env.curr_agents_state["side_quest_progress"][aid] = prog
                    continue

            last_count = prog.get("last_visited_areas_count", 0)
            current_count = len(visited_areas)
            if current_count > last_count:
                prog["last_visited_areas_count"] = current_count
                self._update_task_pools_on_new_area(env, world, agent, prog)

            aevents = [e for e in res.events if getattr(e, "agent_id", None) == aid]
            completed_scenarios = self._check_task_completions(env, world, agent, prog, aevents, res)

            if completed_scenarios:
                self._ensure_objective_guide(env, world, agent, prog, res, force_here=True, announce=True)
                for scenario in completed_scenarios:
                    task = prog["completed_task_info"].get(scenario)
                    if task:
                        reward = task.get("reward_coins", 10)
                        self._spawn_coins(world, current_area_id, reward, aid, res)
                        res.add_feedback(
                            aid,
                            f"\nSide quest completed: {task['description']}\n"
                            f"You earned {reward} {world.objects['obj_coin'].name if 'obj_coin' in world.objects else 'coin'}s!\n"
                            f"Talk to {guide_name} to see your updated tasks.\n"
                        )
                        res.events.append(Event(
                            type="side_quest_completed",
                            agent_id=aid,
                            data={"task_type": scenario, "reward_coins": reward, "description": task["description"]},
                        ))

                        # check if agent has explored new areas since task was published
                        # only generate new task if so
                        old_published_areas = prog.get("task_published_areas", {}).get(scenario, [])
                        current_visited = list(env.curr_agents_state.get("areas_visited", {}).get(aid, []))
                        if set(current_visited) - set(old_published_areas):
                            self._generate_task_for_scenario(env, world, agent, prog, scenario)

                prog["completed_task_info"] = {}

            self._update_guide_dialogue(world, prog, guide_name)

            env.curr_agents_state["side_quest_progress"][aid] = prog

    def _init_agent_progress(self, agent_id: str) -> None:
        self._progress[agent_id] = {
            "unlocked": False,
            "just_unlocked": False,
            "tasks": {
                "collect": None,
                "craft": None,
                "talk": None,
                "trade": None,
            },
            "task_progress": {},  # for tracking partial completion (e.g., crafted count)
            "completed_count": {"collect": 0, "craft": 0, "talk": 0, "trade": 0},
            "completed_task_info": {},
            "sq_guide_name": None,
            "sq_guide_inst": None,
            "sq_guide_area": None,
            "task_published_areas": {
                "collect": [],
                "craft": [],
                "talk": [],
                "trade": [],
            },
            "last_visited_areas_count": 0,
        }

    def _register_required_npcs(self, world) -> None:
        aux = world.auxiliary
        aux.setdefault("npc_id_to_count", {})
        aux.setdefault("npc_name_to_id", {})

        for d in self.required_npcs:
            nid = d["id"]
            if nid in world.npcs:
                continue

            atk = d.get("attack_power", d.get("base_attack_power", 0))
            hp = d.get("hp", d.get("base_hp", 100))

            # Convert 'objects' list to inventory dict
            inventory = {}
            for obj_id in d.get("objects", []):
                if obj_id:
                    inventory[obj_id] = inventory.get(obj_id, 0) + 1

            world.npcs[nid] = NPC(
                type=d.get("type", "npc"),
                id=nid,
                name=d["name"],
                enemy=bool(d.get("enemy", False)),
                unique=bool(d.get("unique", True)),
                role=d.get("role", "guide"),
                level=int(d.get("level", 1) or 1),
                description=d.get("description", ""),
                attack_power=int(atk or 0),
                hp=int(hp or 0),
                coins=int(d.get("coins", 0) or 0),
                base_hp=int(d.get("base_hp", 100) or 100),
                base_attack_power=int(d.get("base_attack_power", 0) or 0),
                slope_hp=int(d.get("slope_hp", 0) or 0),
                slope_attack_power=int(d.get("slope_attack_power", 0) or 0),
                quest=bool(d.get("quest", True)),
                inventory=inventory,
            )

    def _ensure_guide_name_map_for_agents(self, env, world) -> None:
        sq = world.auxiliary.setdefault("side_quest", {})
        name_map = sq.setdefault("guide_names", {})

        run_seed = int(getattr(env, "seed", 0))
        sq["run_seed"] = run_seed

        agent_ids = sorted([str(a.id) for a in getattr(env, "agents", [])])
        if agent_ids and all(aid in name_map for aid in agent_ids):
            return

        import random
        base_pool = list(self.OBJECTIVE_NPC_NAME_POOL)
        r = random.Random(run_seed ^ 0x5A5A5A5A)
        pool = base_pool[:]
        r.shuffle(pool)

        used = set(v for v in name_map.values() if isinstance(v, str))
        idx = 0
        for aid in agent_ids:
            if aid in name_map and isinstance(name_map[aid], str) and name_map[aid].strip():
                used.add(name_map[aid])
                continue
            while idx < len(pool) and pool[idx] in used:
                idx += 1
            if idx < len(pool):
                name_map[aid] = pool[idx]
                used.add(pool[idx])
                idx += 1
            else:
                name_map[aid] = "guide_ember"
                used.add("guide_ember")

        sq["guide_names"] = name_map

    def _ensure_agent_guide_name(self, world, prog: Dict[str, Any], agent_id: str) -> str:
        sq = world.auxiliary.setdefault("side_quest", {})
        name_map = sq.setdefault("guide_names", {})

        existing = prog.get("sq_guide_name")
        if isinstance(existing, str) and existing.strip():
            guide_name = existing.strip().lower()
            name_map[agent_id] = guide_name
            return guide_name

        if agent_id not in name_map:
            pool = list(self.OBJECTIVE_NPC_NAME_POOL)
            used = set(v for v in name_map.values() if isinstance(v, str))
            for cand in pool:
                if cand not in used:
                    name_map[agent_id] = cand
                    break
            else:
                name_map[agent_id] = pool[0] if pool else "guide_ember"

        prog["sq_guide_name"] = name_map[agent_id]
        return name_map[agent_id]

    def _generate_all_tasks(self, env, world, agent, prog: Dict[str, Any]) -> None:
        for scenario in ["collect", "craft", "talk", "trade"]:
            self._generate_task_for_scenario(env, world, agent, prog, scenario)

    def _generate_task_for_scenario(self, env, world, agent, prog: Dict[str, Any], scenario: str) -> None:
        rng = env.rng
        visited_areas = list(env.curr_agents_state.get("areas_visited", {}).get(agent.id, []))
        current_area_id = env.curr_agents_state["area"][agent.id]

        task = None

        if scenario == "collect":
            task = self._generate_collect_task(env, world, agent, prog, set(visited_areas), rng)
        elif scenario == "craft":
            task = self._generate_craft_task(env, world, agent, prog, set(visited_areas), rng)
        elif scenario == "talk":
            task = self._generate_talk_task(env, world, agent, prog, set(visited_areas), current_area_id, rng)
        elif scenario == "trade":
            task = self._generate_trade_task(env, world, agent, prog, set(visited_areas), rng)

        if task:
            task["scenario"] = scenario
            prog["tasks"][scenario] = task
            prog["task_progress"].setdefault(scenario, {})
            prog.setdefault("task_published_areas", {})[scenario] = list(visited_areas)

    def _generate_collect_task(self, env, world, agent, prog, visited_areas: Set[str], rng) -> Optional[Dict]:
        guide_area_id = prog.get("sq_guide_area")

        eligible_areas = set(a for a in visited_areas if a != guide_area_id)

        guide_area_objects: Set[str] = set()
        if guide_area_id:
            guide_area = world.area_instances.get(guide_area_id)
            if guide_area:
                objs = getattr(guide_area, "objects", {}) or {}
                if isinstance(objs, dict):
                    for oid in objs.keys():
                        guide_area_objects.add(get_def_id(oid))

        available_objects: Set[str] = set()
        for area_id in eligible_areas:
            area = world.area_instances.get(area_id)
            if not area:
                continue
            objs = getattr(area, "objects", {}) or {}
            if isinstance(objs, dict):
                for oid, cnt in objs.items():
                    if cnt > 0:
                        available_objects.add(get_def_id(oid))

        eligible_base_ids = available_objects - guide_area_objects

        non_compositional_objs = []
        for base_id in eligible_base_ids:
            if base_id not in world.objects:
                continue
            obj = world.objects[base_id]
            if getattr(obj, "quest", False):
                continue
            craft_ings = getattr(obj, "craft_ingredients", {}) or {}
            if not craft_ings:  # non-compositional
                non_compositional_objs.append((base_id, obj))

        if not non_compositional_objs:
            return None

        # filter objects to only those with area_total > 0 in explored regions
        valid_objs_with_area_count = []
        for base_id, obj in non_compositional_objs:
            area_total = 0
            for area_id in eligible_areas:
                area = world.area_instances.get(area_id)
                if area:
                    objs = getattr(area, "objects", {}) or {}
                    if isinstance(objs, dict):
                        for oid, cnt in objs.items():
                            if get_def_id(oid) == base_id:
                                area_total += int(cnt)
            if area_total > 0:
                valid_objs_with_area_count.append((base_id, obj, area_total))

        if not valid_objs_with_area_count:
            return None

        obj_id, obj, area_total = rng.choice(valid_objs_with_area_count)

        agent_count = self._count_agent_item(world, agent, obj_id)
        target_count = agent_count + max(1, area_total // 2)

        reward_range = self.REWARD_COINS["collect"]
        reward = rng.randint(reward_range[0], reward_range[1])

        return {
            "type": "collect",
            "target_obj": obj_id,
            "target_obj_name": obj.name,
            "target_count": target_count,
            "description": f"collect {target_count} {obj.name}",
            "reward_coins": reward,
        }

    def _generate_craft_task(self, env, world, agent, prog, visited_areas: Set[str], rng) -> Optional[Dict]:
        guide_area_id = prog.get("sq_guide_area")

        objects_crafted = env.curr_agents_state.get("objects_crafted", {}).get(agent.id, {})
        crafted_obj_ids = set(objects_crafted.keys())
        agent_items = self._get_all_agent_items(world, agent)

        guide_area_objects: Dict[str, int] = {}
        if guide_area_id:
            guide_area = world.area_instances.get(guide_area_id)
            if guide_area:
                objs = getattr(guide_area, "objects", {}) or {}
                if isinstance(objs, dict):
                    for oid, cnt in objs.items():
                        base = get_def_id(oid)
                        guide_area_objects[base] = guide_area_objects.get(base, 0) + int(cnt)

        eligible_areas = set(a for a in visited_areas if a != guide_area_id)
        area_objects_excl_guide = self._get_total_objects_in_areas(world, eligible_areas)
        area_objects_all = self._get_total_objects_in_areas(world, visited_areas)

        craftable_objs = []
        for oid, obj in world.objects.items():
            if getattr(obj, "quest", False):
                continue

            if oid in crafted_obj_ids:
                continue

            craft_ings = getattr(obj, "craft_ingredients", {}) or {}
            if not craft_ings:
                continue

            all_ingredients_available = True
            has_ingredient_requiring_travel = False

            for ing_id, required_count in craft_ings.items():
                available = agent_items.get(ing_id, 0) + area_objects_all.get(ing_id, 0)
                if available < required_count:
                    all_ingredients_available = False
                    break

                if ing_id in world.objects:
                    ing_obj = world.objects[ing_id]
                    ing_craft = getattr(ing_obj, "craft_ingredients", {}) or {}
                    if not ing_craft:
                        available_local = agent_items.get(ing_id, 0) + guide_area_objects.get(ing_id, 0)
                        if available_local < required_count:
                            has_ingredient_requiring_travel = True

            if all_ingredients_available and has_ingredient_requiring_travel:
                craftable_objs.append((oid, obj))

        if not craftable_objs:
            return None

        obj_id, obj = rng.choice(craftable_objs)

        reward_range = self.REWARD_COINS["craft"]
        reward = rng.randint(reward_range[0], reward_range[1])

        return {
            "type": "craft",
            "target_obj": obj_id,
            "target_obj_name": obj.name,
            "description": f"craft a {obj.name}",
            "reward_coins": reward,
        }

    def _generate_talk_task(self, env, world, agent, prog, visited_areas: Set[str], current_area_id: str, rng) -> Optional[Dict]:
        guide_area_id = prog.get("sq_guide_area")

        eligible_npcs = []
        for area_id in visited_areas:
            if area_id == guide_area_id:
                continue
            area = world.area_instances.get(area_id)
            if not area:
                continue
            for npc_id in getattr(area, "npcs", []):
                npc = world.npc_instances.get(npc_id)
                if not npc:
                    continue
                if getattr(npc, "quest", False):
                    continue
                if getattr(npc, "enemy", False):
                    continue
                if get_def_id(npc_id) == self.OBJECTIVE_NPC_BASE_ID:
                    continue
                eligible_npcs.append((npc_id, npc, area_id))

        if not eligible_npcs:
            return None

        npc_id, npc, npc_area = rng.choice(eligible_npcs)

        reward_range = self.REWARD_COINS["talk"]
        reward = rng.randint(reward_range[0], reward_range[1])

        return {
            "type": "talk",
            "target_npc": npc_id,
            "target_npc_name": npc.name,
            "target_area": npc_area,
            "description": f"talk to {npc.name}",
            "reward_coins": reward,
        }

    def _generate_trade_task(self, env, world, agent, prog, visited_areas: Set[str], rng) -> Optional[Dict]:
        guide_area_id = prog.get("sq_guide_area")

        eligible_areas = set(a for a in visited_areas if a != guide_area_id)

        eligible_trades = []
        for area_id in eligible_areas:
            area = world.area_instances.get(area_id)
            if not area:
                continue
            for npc_id in getattr(area, "npcs", []):
                npc = world.npc_instances.get(npc_id)
                if not npc:
                    continue
                if getattr(npc, "quest", False):
                    continue
                if getattr(npc, "enemy", False):
                    continue
                if getattr(npc, "role", None) != "merchant":
                    continue
                inv = getattr(npc, "inventory", {}) or {}
                for obj_id, count in inv.items():
                    if count > 0 and obj_id in world.objects:
                        obj = world.objects[obj_id]
                        if not getattr(obj, "quest", False):
                            eligible_trades.append((obj_id, obj, npc_id, npc))

        if not eligible_trades:
            return None

        obj_id, obj, npc_id, npc = rng.choice(eligible_trades)

        reward_range = self.REWARD_COINS["trade"]
        reward = rng.randint(reward_range[0], reward_range[1])

        return {
            "type": "trade",
            "target_obj": obj_id,
            "target_obj_name": obj.name,
            "target_npc": npc_id,
            "target_npc_name": npc.name,
            "description": f"trade {obj.name} with {npc.name}",
            "reward_coins": reward,
        }

    def _check_task_completions(self, env, world, agent, prog: Dict, events: list, res: RuleResult) -> List[str]:
        completed = []
        tasks = prog.get("tasks", {})
        task_progress = prog.setdefault("task_progress", {})
        aid = agent.id

        for scenario, task in tasks.items():
            if not task:
                continue

            is_complete = False

            if scenario == "collect":
                target_obj = task.get("target_obj")
                target_count = task.get("target_count", 1)
                current_count = self._count_agent_item(world, agent, target_obj)
                if current_count >= target_count:
                    is_complete = True

            elif scenario == "craft":
                target_obj = task.get("target_obj")
                for e in events:
                    if getattr(e, "type", None) == "object_crafted":
                        d = getattr(e, "data", {}) or {}
                        crafted_id = d.get("obj_id", "")
                        if get_def_id(crafted_id) == target_obj:
                            is_complete = True
                            break

            elif scenario == "talk":
                target_npc = task.get("target_npc")
                for e in events:
                    if getattr(e, "type", None) == "npc_talked_to":
                        d = getattr(e, "data", {}) or {}
                        talked_npc = d.get("npc_id", "")
                        if talked_npc == target_npc:
                            is_complete = True
                            break

            elif scenario == "trade":
                target_obj = task.get("target_obj")
                target_npc = task.get("target_npc")
                for e in events:
                    if getattr(e, "type", None) == "object_bought":
                        d = getattr(e, "data", {}) or {}
                        bought_obj = d.get("obj_id", "")
                        from_npc = d.get("npc_id", "")
                        if get_def_id(bought_obj) == target_obj and from_npc == target_npc:
                            is_complete = True
                            break

            if is_complete:
                completed.append(scenario)
                prog["completed_count"][scenario] = prog["completed_count"].get(scenario, 0) + 1
                prog["completed_task_info"][scenario] = task
                prog["tasks"][scenario] = None

        return completed

    def _spawn_coins(self, world, area_id: str, count: int, agent_id: str, res: RuleResult) -> None:
        area = world.area_instances.get(area_id)
        if not area or count <= 0:
            return
        coin_id = "obj_coin"
        objs = getattr(area, "objects", None)
        if objs is not None and isinstance(objs, dict):
            objs[coin_id] = int(objs.get(coin_id, 0) or 0) + count
            res.track_spawn(agent_id, coin_id, count, res.tloc("area", area_id))

    def _ensure_objective_guide(
        self, env, world, agent, prog: Dict[str, Any], res: RuleResult,
        force_here: bool = False, announce: bool = False
    ) -> None:
        if self.OBJECTIVE_NPC_BASE_ID not in getattr(world, "npcs", {}):
            return

        guide_name = self._ensure_agent_guide_name(world, prog, agent.id)
        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances.get(current_area_id)
        if not current_area:
            return

        inst_id = prog.get("sq_guide_inst")
        if inst_id and inst_id not in world.npc_instances:
            inst_id = None
            prog["sq_guide_inst"] = None
            prog["sq_guide_area"] = None

        entered = False

        if not inst_id:
            inst_id = self._create_npc_instance(world, self.OBJECTIVE_NPC_BASE_ID, current_area_id, env.rng)
            if not inst_id:
                return
            prog["sq_guide_inst"] = inst_id
            prog["sq_guide_area"] = current_area_id
            entered = True
        else:
            if force_here:
                where = prog.get("sq_guide_area")
                if where != current_area_id or not self._area_contains_npc(world, where, inst_id):
                    old_area_id = self._find_area_containing_npc(world, inst_id, hint=where)
                    if old_area_id and old_area_id in world.area_instances:
                        old_area = world.area_instances[old_area_id]
                        try:
                            if inst_id in getattr(old_area, "npcs", []):
                                old_area.npcs.remove(inst_id)
                        except Exception:
                            pass

                    try:
                        if inst_id not in getattr(current_area, "npcs", []):
                            current_area.npcs.append(inst_id)
                    except Exception:
                        return

                    prog["sq_guide_area"] = current_area_id
                    entered = True

        self._set_npc_name(world, inst_id, guide_name)
        self._force_npc_invincible(world, inst_id)

        if entered and announce:
            res.add_feedback(agent.id, f"{guide_name} enters the area.\n")

    def _update_guide_dialogue(self, world, prog: Dict[str, Any], guide_name: str) -> None:
        inst_id = prog.get("sq_guide_inst")
        if not inst_id or inst_id not in world.npc_instances:
            return

        tasks = prog.get("tasks", {})
        lines = ["Your current tasks are:\n"]
        for scenario in ["collect", "craft", "talk", "trade"]:
            task = tasks.get(scenario)
            if task:
                reward = task.get("reward_coins", 10)
                coin_name = world.objects["obj_coin"].name if "obj_coin" in world.objects else "coin"
                lines.append(f"- {task['description']} (reward: {reward} {coin_name}s)\n")
            else:
                lines.append(f"- [{scenario}]: No task available\n")
        dialogue = "".join(lines)

        self._set_npc_dialogue(world, inst_id, dialogue)

    def _create_npc_instance(self, world, base_npc_id: str, area_id: str, rng) -> Optional[str]:
        if base_npc_id not in world.npcs:
            return None
        area = world.area_instances.get(area_id)
        if not area:
            return None

        for npc_id in getattr(area, "npcs", []):
            if get_def_id(npc_id) == base_npc_id:
                return npc_id

        aux = world.auxiliary
        proto = world.npcs[base_npc_id]
        level = int(getattr(area, "level", 1) or 1)

        if getattr(proto, "unique", False):
            idx = None
        else:
            idx = int(aux.get("npc_id_to_count", {}).get(base_npc_id, 0))
            aux.setdefault("npc_id_to_count", {})[base_npc_id] = idx + 1

        inst_obj = proto.create_instance(idx, level=level, objects=world.objects, rng=rng)
        if isinstance(inst_obj, tuple):
            inst_obj = inst_obj[0]
        inst = inst_obj

        world.npc_instances[inst.id] = inst
        getattr(area, "npcs", []).append(inst.id)
        world.auxiliary.setdefault("npc_name_to_id", {})[inst.name] = inst.id
        return inst.id

    def _set_npc_dialogue(self, world, npc_inst_id: str, dialogue: str) -> None:
        npc = world.npc_instances.get(npc_inst_id)
        if not npc:
            return
        try:
            npc.dialogue = str(dialogue or "")
        except Exception:
            try:
                setattr(npc, "dialogue", str(dialogue or ""))
            except Exception:
                pass

    def _set_npc_name(self, world, npc_inst_id: str, new_name: str) -> None:
        npc = world.npc_instances.get(npc_inst_id)
        if not npc:
            return
        new_name = (new_name or "").strip()
        if not new_name:
            return

        aux = world.auxiliary
        aux.setdefault("npc_name_to_id", {})
        old_name = getattr(npc, "name", None)

        if old_name and aux["npc_name_to_id"].get(old_name) == npc_inst_id:
            try:
                del aux["npc_name_to_id"][old_name]
            except Exception:
                pass

        try:
            npc.name = new_name
        except Exception:
            try:
                setattr(npc, "name", new_name)
            except Exception:
                return

        aux["npc_name_to_id"][new_name] = npc_inst_id

    def _force_npc_invincible(self, world, npc_inst_id: str) -> None:
        npc = world.npc_instances.get(npc_inst_id)
        if not npc:
            return
        try:
            npc.enemy = False
        except Exception:
            pass
        try:
            npc.hp = max(int(getattr(npc, "hp", 0) or 0), int(self.OBJECTIVE_NPC_MAX_HP))
        except Exception:
            try:
                setattr(npc, "hp", int(self.OBJECTIVE_NPC_MAX_HP))
            except Exception:
                pass
        try:
            npc.attack_power = max(int(getattr(npc, "attack_power", 0) or 0), int(self.OBJECTIVE_NPC_ATK))
        except Exception:
            try:
                setattr(npc, "attack_power", int(self.OBJECTIVE_NPC_ATK))
            except Exception:
                pass

    def _area_contains_npc(self, world, area_id: Optional[str], npc_inst_id: str) -> bool:
        if not area_id:
            return False
        area = world.area_instances.get(area_id)
        if not area:
            return False
        try:
            return npc_inst_id in getattr(area, "npcs", [])
        except Exception:
            return False

    def _find_area_containing_npc(self, world, npc_inst_id: str, hint: Optional[str] = None) -> Optional[str]:
        if hint and hint in world.area_instances and self._area_contains_npc(world, hint, npc_inst_id):
            return hint
        for aid, area in getattr(world, "area_instances", {}).items():
            try:
                if npc_inst_id in getattr(area, "npcs", []):
                    return aid
            except Exception:
                continue
        return None

    def _count_agent_item(self, world, agent, base_obj_id: str) -> int:
        total = 0
        if not base_obj_id:
            return total

        items_in_hands = getattr(agent, "items_in_hands", {}) or {}
        equipped = getattr(agent, "equipped_items_in_limb", {}) or {}

        if isinstance(items_in_hands, dict):
            for oid, cnt in items_in_hands.items():
                if get_def_id(oid) == base_obj_id:
                    total += int(cnt)
        else:
            for oid in items_in_hands:
                if get_def_id(oid) == base_obj_id:
                    total += 1

        if isinstance(equipped, dict):
            for oid, cnt in equipped.items():
                if get_def_id(oid) == base_obj_id:
                    total += int(cnt)
        else:
            for oid in equipped:
                if get_def_id(oid) == base_obj_id:
                    total += 1

        inv = getattr(agent, "inventory", None)
        if inv and getattr(inv, "container", None):
            inv_items = getattr(inv, "items", {}) or {}
            for oid, cnt in inv_items.items():
                if get_def_id(oid) == base_obj_id:
                    total += int(cnt)

        # check containers held in hands
        hand_keys = items_in_hands.keys() if isinstance(items_in_hands, dict) else items_in_hands
        for oid in hand_keys:
            if oid in getattr(world, "container_instances", {}):
                cinv = world.container_instances[oid].inventory
                for coid, cnt in (cinv or {}).items():
                    if get_def_id(coid) == base_obj_id:
                        total += int(cnt)

        return total

    def _get_all_agent_items(self, world, agent) -> Dict[str, int]:
        items = {}

        items_in_hands = getattr(agent, "items_in_hands", {}) or {}
        equipped = getattr(agent, "equipped_items_in_limb", {}) or {}

        if isinstance(items_in_hands, dict):
            for oid, cnt in items_in_hands.items():
                base = get_def_id(oid)
                items[base] = items.get(base, 0) + int(cnt)

        if isinstance(equipped, dict):
            for oid, cnt in equipped.items():
                base = get_def_id(oid)
                items[base] = items.get(base, 0) + int(cnt)

        inv = getattr(agent, "inventory", None)
        if inv and getattr(inv, "container", None):
            inv_items = getattr(inv, "items", {}) or {}
            for oid, cnt in inv_items.items():
                base = get_def_id(oid)
                items[base] = items.get(base, 0) + int(cnt)

        hand_keys = items_in_hands.keys() if isinstance(items_in_hands, dict) else items_in_hands
        for oid in hand_keys:
            if oid in getattr(world, "container_instances", {}):
                cinv = world.container_instances[oid].inventory
                for coid, cnt in (cinv or {}).items():
                    base = get_def_id(coid)
                    items[base] = items.get(base, 0) + int(cnt)

        return items

    def _get_total_objects_in_areas(self, world, visited_areas: Set[str]) -> Dict[str, int]:
        area_objects: Dict[str, int] = {}
        for area_id in visited_areas:
            area = world.area_instances.get(area_id)
            if not area:
                continue
            objs = getattr(area, "objects", {}) or {}
            for oid, cnt in objs.items():
                base = get_def_id(oid)
                area_objects[base] = area_objects.get(base, 0) + int(cnt)
        return area_objects

    def _update_task_pools_on_new_area(self, env, world, agent, prog: Dict[str, Any]) -> None:
        rng = env.rng
        visited_areas = list(env.curr_agents_state.get("areas_visited", {}).get(agent.id, []))
        current_area_id = env.curr_agents_state["area"][agent.id]
        tasks = prog.get("tasks", {})
        task_published_areas = prog.get("task_published_areas", {})

        for scenario in ["collect", "craft", "talk", "trade"]:
            if tasks.get(scenario) is not None:
                continue  # already has an active task

            # check if agent has visited new areas since last task of this scenario was published
            old_published_areas = task_published_areas.get(scenario, [])
            if old_published_areas and not (set(visited_areas) - set(old_published_areas)):
                continue  # no new areas visited, cannot generate new task

            task = None
            if scenario == "collect":
                task = self._generate_collect_task(env, world, agent, prog, set(visited_areas), rng)
            elif scenario == "craft":
                task = self._generate_craft_task(env, world, agent, prog, set(visited_areas), rng)
            elif scenario == "talk":
                task = self._generate_talk_task(env, world, agent, prog, set(visited_areas), current_area_id, rng)
            elif scenario == "trade":
                task = self._generate_trade_task(env, world, agent, prog, set(visited_areas), rng)

            if task:
                task["scenario"] = scenario
                prog["tasks"][scenario] = task
                prog["task_progress"].setdefault(scenario, {})
                # record published areas for this scenario
                prog.setdefault("task_published_areas", {})[scenario] = list(visited_areas)

    def _get_area_display_name(self, area_id: str) -> str:
        if not area_id or not self._world_ref:
            return area_id or "unknown"
        area = self._world_ref.area_instances.get(area_id)
        if not area:
            return area_id
        area_name = getattr(area, "name", area_id)
        place_id = self._area_to_place.get(area_id)
        if place_id and place_id in self._world_ref.place_instances:
            place_name = getattr(self._world_ref.place_instances[place_id], "name", place_id)
            return f"{place_name}, {area_name}"
        return str(area_name)


class WorldExpansionStepRule(BaseStepRule):
    name = "world_expansion_step"
    description = (
        "Dynamically expands the world when the player has explored all "
        "accessible areas."
    )
    priority = 100

    _SYSTEM_PROMPT = (
        "You are a game world entity generator for a text-based RPG called "
        "Agent Odyssey. You create entities (places, objects, NPCs) that are:\n"
        "1. USEFUL - Every entity must serve a purpose in the game mechanics\n"
        "2. BALANCED - Difficulty increases progressively with levels\n"
        "3. INTERCONNECTED - Objects should be craftable from materials, "
        "NPCs should drop useful items\n\n"
        "GAME MECHANICS AWARENESS:\n"
        "- Players can: pick up, drop, store, craft, equip, attack, defend, "
        "buy/sell, enter areas, inspect, write, wait\n"
        "- Enemies attack players and drop loot when defeated\n"
        "- Objects have levels that determine when they become relevant\n"
        "- Areas have levels that determine enemy strength and available "
        "resources\n"
        "- Crafting requires ingredients (consumed) and may require "
        "dependency stations (not consumed, must be in the same area)\n"
        "- Weapons provide attack_power when equipped; armor provides "
        "defense\n"
        "- Merchants buy/sell items using currency\n\n"
        "DIFFICULTY PROGRESSION RULES:\n"
        "- Level 1-2: Basic materials, weak enemies, simple crafts\n"
        "- Level 3-4: Intermediate materials, moderate enemies, useful "
        "equipment\n"
        "- Level 5+: Rare materials, dangerous enemies, powerful equipment\n\n"
        "ENTITY DESIGN PRINCIPLES:\n"
        "1. Materials should be used in multiple craft recipes - every raw "
        "material with usage='craft' MUST be referenced as an ingredient in "
        "at least one other object's crafting recipe\n"
        "2. Crafting chains: raw material → processed material → equipment\n"
        "3. NPCs: Merchants should sell useful items; Enemies should drop "
        "crafting materials\n"
        "4. Areas: higher-level areas should have higher-level resources and "
        "tougher enemies\n"
        "5. Area names must be SHORT and CONCISE — a single generic word "
        "like 'entrance', 'depths', 'clearing', 'ridge', NOT compound "
        "names like 'catacomb_entrance' or 'dark_depths'. The place name "
        "already provides thematic context.\n"
        "6. Each area should have 1-3 crafting materials matching its level\n"
        "7. Every object should either be: craftable, a crafting ingredient, "
        "purchasable, or dropped by enemies\n\n"
        "Return ONLY valid JSON (no markdown fences, no explanations)."
    )

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if world is None:
            return

        features = world.world_definition.get("features", {})
        if not features.get("online_expansion", False):
            return

        logger = get_logger("ExpansionLogger")
        expansion_state = env.curr_agents_state.setdefault(
            "world_expansion",
            {"pending": False, "count": 0, "total_areas": len(world.area_instances)},
        )

        # --- phase 1: check whether a pending expansion is ready -----------
        if expansion_state.get("pending", False):
            future = getattr(env, "_expansion_future", None)
            if future is not None and future.done():
                try:
                    expansion_def = future.result(timeout=0)
                    if expansion_def and expansion_def.get("places"):
                        new_places, new_areas = world.expand(
                            expansion_def,
                            seed=env.rng.randint(0, 10**9),
                        )
                        expansion_state["count"] = (
                            expansion_state.get("count", 0) + 1
                        )
                        expansion_state["total_areas"] = len(
                            world.area_instances
                        )
                        place_list = (
                            ", ".join(new_places) if new_places else "unknown lands"
                        )
                        for agent in env.agents:
                            res.add_feedback(
                                agent.id,
                                f"The ground trembles and the horizon "
                                f"shifts. New lands have emerged."
                                f"Unfamiliar paths now "
                                f"extend from the edges of the known "
                                f"world.\n",
                            )
                        logger.info(
                            f"\u2705 Expansion #{expansion_state['count']} "
                            f"integrated: {place_list}"
                        )
                    else:
                        logger.warning(
                            "\u26a0\ufe0f Expansion generation returned "
                            "empty or invalid result."
                        )
                except Exception as e:
                    logger.error(
                        f"\u274c Expansion generation failed: {e}"
                    )

                expansion_state["pending"] = False
                env._expansion_future = None

            return

        # --- phase 2: check whether expansion should trigger ---------------
        for agent in env.agents:
            visited = set(
                env.curr_agents_state.get("areas_visited", {}).get(
                    agent.id, []
                )
            )
            all_areas = set(world.area_instances.keys())

            if all_areas and all_areas.issubset(visited):
                expansion_state["pending"] = True

                context = self._build_context(world)
                model_name = features.get(
                    "expansion_model", "gpt-4o-mini"
                )
                existing_obj_ids = set(world.objects.keys())

                import concurrent.futures

                executor = getattr(env, "_expansion_executor", None)
                if executor is None:
                    env._expansion_executor = (
                        concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    )
                    executor = env._expansion_executor

                env._expansion_future = executor.submit(
                    WorldExpansionStepRule._generate_expansion,
                    context,
                    model_name,
                    existing_obj_ids,
                )

                for a in env.agents:
                    res.add_feedback(
                        a.id,
                        "As you explore the last corners of the known world, "
                        "the environment around you begins to change in subtle ways.\n"
                    )
                logger.info(
                    "\U0001f30d Expansion triggered — generating new "
                    "content in background..."
                )
                break  # trigger once per step

    @staticmethod
    def _analyze_difficulty(world) -> dict:
        # analyze live world for stat ranges and level distribution
        analysis = {
            "max_level": 1,
            "min_level": 1,
            "level_distribution": {},
            "stat_ranges": {
                "attack": {"min": float("inf"), "max": 0},
                "defense": {"min": float("inf"), "max": 0},
                "hp": {"min": float("inf"), "max": 0},
                "value": {"min": float("inf"), "max": 0},
            },
        }

        for area in world.area_instances.values():
            lvl = area.level
            analysis["level_distribution"][lvl] = (
                analysis["level_distribution"].get(lvl, 0) + 1
            )

        all_levels = [a.level for a in world.area_instances.values()]
        if all_levels:
            analysis["max_level"] = max(all_levels)
            analysis["min_level"] = min(all_levels)

        for obj in world.objects.values():
            if getattr(obj, "attack", 0):
                analysis["stat_ranges"]["attack"]["min"] = min(
                    analysis["stat_ranges"]["attack"]["min"], obj.attack
                )
                analysis["stat_ranges"]["attack"]["max"] = max(
                    analysis["stat_ranges"]["attack"]["max"], obj.attack
                )
            if getattr(obj, "defense", 0):
                analysis["stat_ranges"]["defense"]["min"] = min(
                    analysis["stat_ranges"]["defense"]["min"], obj.defense
                )
                analysis["stat_ranges"]["defense"]["max"] = max(
                    analysis["stat_ranges"]["defense"]["max"], obj.defense
                )
            if obj.value is not None:
                analysis["stat_ranges"]["value"]["min"] = min(
                    analysis["stat_ranges"]["value"]["min"], obj.value
                )
                analysis["stat_ranges"]["value"]["max"] = max(
                    analysis["stat_ranges"]["value"]["max"], obj.value
                )

        for npc in world.npcs.values():
            if getattr(npc, "enemy", False):
                bhp = getattr(npc, "hp", 0) or 0
                batk = getattr(npc, "attack_power", 0) or 0
                if bhp:
                    analysis["stat_ranges"]["hp"]["min"] = min(
                        analysis["stat_ranges"]["hp"]["min"], bhp
                    )
                    analysis["stat_ranges"]["hp"]["max"] = max(
                        analysis["stat_ranges"]["hp"]["max"], bhp
                    )
                if batk:
                    analysis["stat_ranges"]["attack"]["min"] = min(
                        analysis["stat_ranges"]["attack"]["min"], batk
                    )
                    analysis["stat_ranges"]["attack"]["max"] = max(
                        analysis["stat_ranges"]["attack"]["max"], batk
                    )

        # replace sentinel inf with 0
        for stat in analysis["stat_ranges"].values():
            if stat["min"] == float("inf"):
                stat["min"] = 0

        return analysis

    @staticmethod
    def _build_context(world) -> str:
        import json as _json

        analysis = WorldExpansionStepRule._analyze_difficulty(world)
        max_level = analysis["max_level"]
        new_min_level = max_level + 1
        new_max_level = max_level + 2

        # --- existing world summary ---
        place_lines = []
        for pid, p in world.place_instances.items():
            area_info = []
            for aid in p.areas:
                a = world.area_instances.get(aid)
                if a:
                    area_info.append(
                        f"    {{\"type\": \"area\", \"id\": \"{a.id}\", "
                        f"\"name\": \"{a.name}\", \"level\": {a.level}}}"
                    )
            place_lines.append(
                f"  {{\"id\": \"{pid}\", \"name\": \"{p.name}\", "
                f"\"unlocked\": {str(p.unlocked).lower()}, \"areas\": [\n"
                + ",\n".join(area_info)
                + "\n  ]}}"
            )

        # objects categorised
        obj_by_cat: dict[str, list] = {}
        for oid, obj in world.objects.items():
            cat = getattr(obj, "category", "misc")
            obj_by_cat.setdefault(cat, []).append(
                f"    {{\"id\": \"{oid}\", \"name\": \"{obj.name}\", "
                f"\"level\": {obj.level}, \"usage\": \"{obj.usage}\""
                + (f", \"attack\": {obj.attack}" if getattr(obj, "attack", 0) else "")
                + (f", \"defense\": {obj.defense}" if getattr(obj, "defense", 0) else "")
                + "}"
            )

        obj_summary_lines = []
        for cat, items in sorted(obj_by_cat.items()):
            obj_summary_lines.append(f"  [{cat}] ({len(items)} total)")
            for line in items[:8]:  # cap per-category
                obj_summary_lines.append(line)
            if len(items) > 8:
                obj_summary_lines.append(f"    ... and {len(items) - 8} more")

        npc_lines = []
        for nid, npc in world.npcs.items():
            enemy_str = "enemy" if getattr(npc, "enemy", False) else "friendly"
            npc_lines.append(
                f"  {{\"id\": \"{nid}\", \"name\": \"{npc.name}\", "
                f"\"role\": \"{npc.role}\", \"{enemy_str}\"}}"
            )

        # Build distributable objects list for LLM to assign to new areas
        undistributable = set(
            world.world_definition.get("initializations", {})
            .get("undistributable_objects", [])
        )
        distributable_lines = []
        for oid, obj in world.objects.items():
            if oid in undistributable:
                continue
            if obj.category in ("currency", "container"):
                continue
            # Only objects that were originally area-distributed are eligible
            if not getattr(obj, "areas", []):
                continue
            distributable_lines.append(
                f"  {{\"id\": \"{oid}\", \"name\": \"{obj.name}\", "
                f"\"category\": \"{obj.category}\", "
                f"\"level\": {obj.level}, "
                f"\"usage\": \"{obj.usage}\"}}"
            )

        prompt = (
            f"EXISTING WORLD (live state):\n"
            f"Places:\n" + "\n".join(place_lines) + "\n\n"
            f"Objects by category:\n" + "\n".join(obj_summary_lines) + "\n\n"
            f"NPCs:\n" + "\n".join(npc_lines) + "\n\n"

            f"DISTRIBUTABLE EXISTING OBJECTS (assign these to new areas):\n"
            + "\n".join(distributable_lines) + "\n\n"

            f"EXISTING IDS (do NOT reuse any of these):\n"
            f"  Place IDs: {list(world.place_instances.keys())}\n"
            f"  Area IDs:  {list(world.area_instances.keys())}\n"
            f"  Object IDs: {list(world.objects.keys())}\n"
            f"  NPC IDs:   {list(world.npcs.keys())}\n\n"

            f"CURRENT DIFFICULTY ANALYSIS:\n"
            f"  Level range: {analysis['min_level']} to {max_level}\n"
            f"  Level distribution: "
            f"{_json.dumps(analysis['level_distribution'])}\n"
            f"  Stat ranges:\n"
            f"    Attack: {analysis['stat_ranges']['attack']['min']}"
            f"-{analysis['stat_ranges']['attack']['max']}\n"
            f"    Defense: {analysis['stat_ranges']['defense']['min']}"
            f"-{analysis['stat_ranges']['defense']['max']}\n"
            f"    HP (enemies): {analysis['stat_ranges']['hp']['min']}"
            f"-{analysis['stat_ranges']['hp']['max']}\n\n"

            f"GENERATION REQUIREMENTS:\n"
            f"Generate entities for levels {new_min_level} to "
            f"{new_max_level}:\n"
            f"- 1-2 NEW places, each with 2-3 areas (levels "
            f"{new_min_level}-{new_max_level})\n"
            f"  For EACH area, include an 'existing_objects' list of "
            f"3-5 IDs from DISTRIBUTABLE EXISTING OBJECTS above that "
            f"are thematically appropriate for that area's setting "
            f"(e.g. a forge area should get anvils/metal, a forest "
            f"clearing should get wood/herbs). Pick objects whose level "
            f"is close to the area's level.\n"
            f"- 4-8 NEW objects with progressive difficulty:\n"
            f"  - ~50% materials (raw & processed, usage='craft', "
            f"category='material')\n"
            f"    Raw materials MUST have empty craft.ingredients and "
            f"craft.dependencies\n"
            f"    Raw materials MUST list your NEW area IDs in 'areas'\n"
            f"    Every raw material MUST be used as an ingredient in at "
            f"least one other object\n"
            f"  - ~25% weapons (category='weapon', usage='attack', "
            f"attack ~{max(15, analysis['stat_ranges']['attack']['max'] + 5)}"
            f"-{max(25, analysis['stat_ranges']['attack']['max'] + 15)} "
            f"scaling with level)\n"
            f"  - ~25% armor (category='armor', usage='defend', "
            f"defense ~{max(10, analysis['stat_ranges']['defense']['max'] + 5)}"
            f"-{max(20, analysis['stat_ranges']['defense']['max'] + 10)} "
            f"scaling with level)\n"
            f"  - Weapons & armor should require crafting from the new "
            f"materials\n"
            f"- 1-2 NEW NPCs:\n"
            f"  - At least 1 enemy with stats scaling above current max: "
            f"base_hp ~{max(50, analysis['stat_ranges']['hp']['max'] + 20)}, "
            f"base_attack_power "
            f"~{max(15, analysis['stat_ranges']['attack']['max'] + 5)}, "
            f"slope_hp ~20-30, slope_attack_power ~10-15\n"
            f"  - Optionally 1 merchant (enemy=false, role='merchant', "
            f"unique=true) with 'objects' listing items they sell\n\n"

            f"BALANCE GUIDELINES:\n"
            f"1. Crafting chains: raw material → processed material → "
            f"equipment\n"
            f"2. Every material with usage='craft' must appear as an "
            f"ingredient in at least one recipe\n"
            f"3. Weapons attack_power should exceed current max "
            f"({analysis['stat_ranges']['attack']['max']})\n"
            f"4. Armor defense should exceed current max "
            f"({analysis['stat_ranges']['defense']['max']})\n"
            f"5. Enemy base_hp should exceed current max "
            f"({analysis['stat_ranges']['hp']['max']})\n"
            f"6. Keep the theme consistent: dark fantasy, survival, "
            f"exploration\n\n"

            f"Return as JSON with structure:\n"
            f"{{\n"
            f"  \"places\": [\n"
            f"    {{\"type\": \"place\", \"id\": \"place_X\", "
            f"\"name\": \"Place Name\", \"unlocked\": true, \"areas\": [\n"
            f"      {{\"type\": \"area\", \"id\": \"area_X\", "
            f"\"name\": \"entrance\", \"level\": N, "
            f"\"existing_objects\": [\"obj_existing_1\", \"obj_existing_2\", \"...\"]}}\n"
            f"    ]}}\n"
            f"  ],\n"
            f"  \"objects\": [\n"
            f"    {{\"type\": \"object\", \"id\": \"obj_X\", "
            f"\"name\": \"object_name\", \"category\": \"...\", "
            f"\"usage\": \"...\", \"value\": N, \"size\": N, "
            f"\"description\": \"...\", \"text\": \"\", "
            f"\"attack\": 0, \"defense\": 0, \"level\": N, "
            f"\"craft\": {{\"ingredients\": {{}}, \"dependencies\": []}}, "
            f"\"areas\": [\"area_ids\"]}}\n"
            f"  ],\n"
            f"  \"npcs\": [\n"
            f"    {{\"type\": \"npc\", \"id\": \"npc_X\", "
            f"\"name\": \"npc_name\", \"enemy\": bool, \"unique\": bool, "
            f"\"description\": \"...\", \"role\": \"...\", "
            f"\"base_hp\": N, \"slope_hp\": N, "
            f"\"base_attack_power\": N, \"slope_attack_power\": N, "
            f"\"objects\": [\"obj_ids\"]}}\n"
            f"  ]\n"
            f"}}\n\n"
            f"ALL IDs must be globally unique and snake_case. "
            f"Object and NPC names must be snake_case (no spaces). "
            f"Area names must be SHORT single words (e.g. 'clearing', "
            f"'depths', 'ridge'). "
            f"Only return valid JSON."
        )

        return prompt

    @staticmethod
    def _validate_expansion(expansion_def: dict, existing_obj_ids: set) -> tuple:
        errors = []
        new_obj_ids = {
            obj.get("id") for obj in expansion_def.get("objects", [])
        }
        all_obj_ids = existing_obj_ids | new_obj_ids

        for obj in expansion_def.get("objects", []):
            oid = obj.get("id", "?")
            if not obj.get("name"):
                errors.append(f"Object {oid} missing 'name'")
            if not obj.get("category"):
                errors.append(f"Object {oid} missing 'category'")
            craft = obj.get("craft", {})
            for ing_id in craft.get("ingredients", {}).keys():
                if ing_id not in all_obj_ids:
                    errors.append(
                        f"Object {oid} references unknown ingredient "
                        f"{ing_id}"
                    )
            for dep_id in craft.get("dependencies", []):
                if dep_id not in all_obj_ids:
                    errors.append(
                        f"Object {oid} references unknown dependency "
                        f"{dep_id}"
                    )

        for npc in expansion_def.get("npcs", []):
            nid = npc.get("id", "?")
            if not npc.get("name"):
                errors.append(f"NPC {nid} missing 'name'")
            if npc.get("role") == "merchant" and npc.get("enemy", False):
                errors.append(
                    f"NPC {nid} cannot be both merchant and enemy"
                )

        for place in expansion_def.get("places", []):
            pid = place.get("id", "?")
            if not place.get("areas"):
                errors.append(f"Place {pid} has no areas")

        return (len(errors) == 0, errors)

    @staticmethod
    def _generate_expansion(
        context: str, model_name: str, existing_obj_ids: set
    ) -> dict:
        import json as _json
        import os
        import re

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; cannot generate world expansion."
            )

        from providers.openai_api import OpenAIClient

        client = OpenAIClient(api_key)

        response = client.run_prompt(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": WorldExpansionStepRule._SYSTEM_PROMPT,
                },
                {"role": "user", "content": context},
            ],
        )

        raw = response["response"].strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            raw = json_match.group()

        expansion_def = _json.loads(raw)

        valid, errors = WorldExpansionStepRule._validate_expansion(
            expansion_def, existing_obj_ids
        )
        if not valid:
            from tools.logger import get_logger

            logger = get_logger("ExpansionLogger")
            for err in errors:
                logger.warning(f"\u26a0\ufe0f Expansion validation: {err}")

        return expansion_def
