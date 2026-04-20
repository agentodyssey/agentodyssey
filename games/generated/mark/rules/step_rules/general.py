import copy
import math
import random
from utils import *
from games.generated.mark.rule import BaseStepRule, RuleContext, RuleResult, Event
from typing import Dict, Optional, Set, Tuple, List, Any
from games.generated.mark.world import NPC, Object, Area, Place, Path, Container, Writable
from games.generated.mark.agent import Agent
from tools.logger import get_logger
import inspect


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
        base_attack_prob, max_attack_prob = 0.10, 0.30
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
                if npc_id in npcs_in_combat:
                    continue
                
                # if agent is already in combat with other NPCs, reduce attack rate to 2.5%
                agent_in_combat = len(npcs_in_combat) > 0
                    
                level = world.area_instances[agent_area].level
                mods = getattr(env, "_step_modifiers", {})
                if not isinstance(mods, dict):
                    mods = {}
                
                if agent_in_combat:
                    base_prob = 0.025
                else:
                    base_prob = attack_prob_by_area_level[level]
                
                if mods.get("ambush_prob_override") is not None:
                    effective_prob = float(mods["ambush_prob_override"])
                else:
                    prob_additive = float(mods.get("ambush_prob_additive", 0.0))
                    prob_multiplier = float(mods.get("ambush_prob_multiplier", 1.0))
                    effective_prob = (base_prob + prob_additive) * prob_multiplier

                effective_prob = max(0.0, min(1.0, effective_prob))
                
                if env.rng.random() > effective_prob:
                    continue
                
                npc_instance = world.npc_instances[npc_id]
                
                # active NPC attack has less damage than NPC attack in combat
                base_damage = max(0, math.floor(npc_instance.attack_power / 2) - agent.defense)
                dmg_multiplier = float(mods.get("enemy_damage_multiplier", 1.0))
                dmg_bonus = int(mods.get("enemy_attack_damage_bonus", 0))  # legacy support
                damage = max(0, int(base_damage * dmg_multiplier) + dmg_bonus)
                
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

class WanderingPatrolsStepRule(BaseStepRule):
    name = "wandering_patrols_step"
    description = "Non-unique merchants and enemies may wander to neighboring unlocked areas, restocking or menacing on arrival."
    priority = 5

    def __init__(self) -> None:
        super().__init__()
        self._move_prob = 0.0625  # chance an eligible NPC moves each step

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if env is None or world is None:
            return

        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_active = bool(tutorial_room) and not bool(tutorial_room.get("removed", False))
        tutorial_area_id = tutorial_room.get("area_id")

        moves: list[tuple[str, str, str]] = []
        visited_npcs: set[str] = set()

        for area_id, area in list(world.area_instances.items()):
            if tutorial_active and tutorial_area_id == area_id:
                continue  # do not move NPCs that are currently in tutorial room

            for npc_id in list(area.npcs):
                if npc_id in visited_npcs:
                    continue
                npc = world.npc_instances.get(npc_id)
                if npc is None:
                    continue

                if getattr(npc, "unique", False) or getattr(npc, "quest", False):
                    continue

                npc_in_combat = False
                for agent_id, agent_combats in env.curr_agents_state.get("active_combats", {}).items():
                    if npc_id in agent_combats:
                        npc_in_combat = True
                        break
                if npc_in_combat:
                    continue

                is_enemy = bool(getattr(npc, "enemy", False))
                role = getattr(npc, "role", None)
                if not is_enemy and role != "merchant":
                    continue

                if env.rng.random() > self._move_prob:
                    continue

                neighbor_ids: list[str] = []
                for nid, path in (getattr(area, "neighbors", {}) or {}).items():
                    if getattr(path, "locked", False):
                        continue
                    if tutorial_active and tutorial_area_id == nid:
                        continue
                    neighbor_non_unique_count = 0
                    for neighbor_npc_id in world.area_instances[nid].npcs:
                        neighbor_npc = world.npc_instances.get(neighbor_npc_id)
                        if neighbor_npc is not None and not getattr(neighbor_npc, "unique", False):
                            neighbor_non_unique_count += 1
                    if neighbor_non_unique_count >= world.area_instances[nid].level:
                        continue
                    neighbor_ids.append(nid)

                if not neighbor_ids:
                    continue

                to_area_id = env.rng.choice(neighbor_ids)
                moves.append((npc_id, area_id, to_area_id))
                visited_npcs.add(npc_id)

        for npc_id, from_area_id, to_area_id in moves:
            from_area = world.area_instances.get(from_area_id)
            to_area = world.area_instances.get(to_area_id)
            if from_area is None or to_area is None:
                continue

            if npc_id in from_area.npcs:
                from_area.npcs.remove(npc_id)
            if npc_id not in to_area.npcs:
                to_area.npcs.append(npc_id)

            npc_inst = world.npc_instances.get(npc_id)
            if npc_inst is None:
                continue

            from_name = world.area_instances[from_area_id].name if from_area_id in world.area_instances else from_area_id
            to_name = world.area_instances[to_area_id].name if to_area_id in world.area_instances else to_area_id

            for agent in env.agents:
                agent_area = env.curr_agents_state["area"][agent.id]
                if agent_area == from_area_id:
                    res.add_feedback(agent.id, f"{npc_inst.name} left towards {to_name}.\n")

            res.events.append(Event(
                type="npc_moved",
                agent_id="env",
                data={"npc_id": npc_id, "from_area": from_area_id, "to_area": to_area_id},
            ))

            if not npc_inst.enemy and getattr(npc_inst, "role", None) == "merchant":
                # announce presence and restock 1–2 random items (small counts)
                restocked: dict[str, int] = {}
                inv = getattr(npc_inst, "inventory", None) or {}
                base_choices = [oid for oid, cnt in inv.items()] or [oid for oid, obj in world.objects.items() if not getattr(obj, "quest", False)]
                k = env.rng.randint(1, 2)
                for _ in range(k):
                    if not base_choices:
                        break
                    obj_id = env.rng.choice(base_choices)
                    add = env.rng.randint(1, 2)
                    inv[obj_id] = int(inv.get(obj_id, 0)) + add
                    restocked[obj_id] = restocked.get(obj_id, 0) + add
                npc_inst.inventory = inv

                for agent in env.agents:
                    if env.curr_agents_state["area"][agent.id] == to_area_id:
                        res.add_feedback(agent.id, f"Merchant {npc_inst.name} arrives from {from_name} and refreshes stock.\n")

                res.events.append(Event(
                    type="merchant_arrived",
                    agent_id="env",
                    data={"npc_id": npc_id, "area_id": to_area_id, "restocked": restocked},
                ))

            elif npc_inst.enemy:
                for agent in env.agents:
                    if env.curr_agents_state["area"][agent.id] == to_area_id:
                        res.add_feedback(agent.id, f"An enemy {npc_inst.name} arrives from {from_name} and prowls into the area.\n")

                mods = getattr(env, "_step_modifiers", None)
                if not isinstance(mods, dict):
                    env._step_modifiers = {}
                    mods = env._step_modifiers
                # Increase attack probability when enemy enters (+0.05)
                current_add = float(mods.get("ambush_prob_additive", 0.0))
                mods["ambush_prob_additive"] = current_add + 0.05

                res.events.append(Event(
                    type="enemy_entered_area",
                    agent_id="env",
                    data={"npc_id": npc_id, "area_id": to_area_id},
                ))

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
                        for craftable_obj_id in ing_to_obj_map[obj_id]:
                            craftable_obj = world.objects[craftable_obj_id]
                            ingredient_str = f"{craftable_obj.name} ("
                            for ingredient_id in craftable_obj.craft_ingredients.keys():
                                ingredient_name = world.objects[ingredient_id].name if ingredient_id in world.objects else "Unknown"
                                ingredient_str += f"{ingredient_name}, "
                            lines += ingredient_str.rstrip(", ") + "), "
            
            if lines:
                lines = lines.rstrip(", ") + "\n"
                env.curr_agents_state["objects_acquired"][agent.id].extend(new_objects_acquired)
                res.add_feedback(agent.id, lines)

                res.events.append(Event(
                    type="new_crafting_recipe_seen",
                    agent_id=agent.id,
                    data={"object_ids": list(new_objects_acquired)},
                ))

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
            agent.attack = agent.min_attack
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
            self._area_to_place = world.auxiliary["area_to_place"]

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
                        f"Side quests are now available! Completing them will grant you coins.\n"
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
                            f"You earned {reward} coins!\n"
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
        spawn_area = env.world_definition["initializations"]["spawn"]["area"]

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
        lines = ["You current tasks are:\n"]
        for scenario in ["collect", "craft", "talk", "trade"]:
            task = tasks.get(scenario)
            if task:
                reward = task.get("reward_coins", 10)
                lines.append(f"- {task['description']} (reward: {reward} coins)\n")
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

        # Check containers held in hands
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
        """Get the total count of each base object available in the visited areas."""
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
        """
        Called when an agent enters a new area. Updates the list of available tasks.
        For each scenario without an active task, check if agent has visited new areas
        since the last task of that scenario was published. If so, try to generate a new task.
        """
        rng = env.rng
        visited_areas = list(env.curr_agents_state.get("areas_visited", {}).get(agent.id, []))
        current_area_id = env.curr_agents_state["area"][agent.id]
        tasks = prog.get("tasks", {})
        task_published_areas = prog.get("task_published_areas", {})

        for scenario in ["collect", "craft", "talk", "trade"]:
            if tasks.get(scenario) is not None:
                continue  # Already has an active task

            # Check if agent has visited new areas since last task of this scenario was published
            old_published_areas = task_published_areas.get(scenario, [])
            if old_published_areas and not (set(visited_areas) - set(old_published_areas)):
                continue  # No new areas visited, cannot generate new task

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
                # Record published areas for this scenario
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

class DayCycleStepRule(BaseStepRule):
    name = "day_cycle_step"
    description = "Time-based day cycle with varying danger levels and bonuses throughout the day."
    priority = 4

    STEPS_PER_HOUR = 6  # 10 mins per step = 6 steps per hour
    STEPS_PER_DAY = 144  # 24 hours * 6 steps

    def __init__(self) -> None:
        super().__init__()

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        dc = aux.setdefault("day_cycle", {})
        dc.setdefault("dawn_coins_awarded", {})  # day_number -> {area_id: 1}
        dc.setdefault("last_period", None)
        dc.setdefault("current_period", "")
        return dc

    def _get_time_of_day(self, env) -> tuple[int, int, int]:
        START_HOUR = 8
        start_offset_steps = START_HOUR * self.STEPS_PER_HOUR + 1  # 49 steps offset
        
        total_steps = int(env.steps) + start_offset_steps
        day_number = total_steps // self.STEPS_PER_DAY
        step_in_day = total_steps % self.STEPS_PER_DAY
        hour = step_in_day // self.STEPS_PER_HOUR

        return (day_number, hour, step_in_day)

    def _get_period(self, hour: int) -> str:
        if 0 <= hour < 1:
            return "dangerous_night"
        elif 1 <= hour < 6:
            return "late_night"
        elif hour == 6:
            return "dawn"
        elif 6 < hour < 12:
            return "morning"
        elif 12 <= hour < 13:
            return "peaceful_midday"
        elif 13 <= hour < 18:
            return "afternoon_calm"
        else:
            return "evening_inspiration"

    def _set_step_modifiers(self, env, period: str) -> None:
        env._step_modifiers = {
            "ambush_prob_additive": 0.0,
            "ambush_prob_override": None,
            "enemy_damage_multiplier": 1.0,
        }
        
        if period == "dangerous_night":
            env._step_modifiers["ambush_prob_additive"] = 0.15
            env._step_modifiers["enemy_damage_multiplier"] = 1.30
        elif period == "late_night":
            env._step_modifiers["enemy_damage_multiplier"] = 1.20
        elif period == "peaceful_midday":
            env._step_modifiers["ambush_prob_override"] = 0.05
            env._step_modifiers["enemy_damage_multiplier"] = 0.80
        elif period == "afternoon_calm":
            env._step_modifiers["enemy_damage_multiplier"] = 0.80

    def _get_period_start_step(self, period: str) -> int:
        period_start_hours = {
            "dangerous_night": 0,
            "late_night": 1,
            "dawn": 6,
            "morning": 7,  # first step after dawn ends
            "peaceful_midday": 12,
            "afternoon_calm": 13,
            "evening_inspiration": 18,
        }
        hour = period_start_hours.get(period, 0)
        return hour * self.STEPS_PER_HOUR

    def _notify_period_change(self, env, world, res: RuleResult, period: str, dc: dict, step_in_day: int) -> None:
        last_period = dc.get("last_period")
        if last_period == period:
            return
        
        dc["last_period"] = period
        
        expected_step = self._get_period_start_step(period)
        if step_in_day != expected_step:
            return
        
        message = None
        if period == "dangerous_night":
            message = (
                "Tension tightens across the world. Hostile intent grows sharper and less restrained.\n"
            )
        elif period == "late_night":
            message = (
                "Weariness seeps in, yet aggression lingers, heavy and unrelenting.\n"
            )
        elif period == "morning" and last_period in ("late_night", "dawn"):
            message = (
                "Sense of peacefulness sneaks in. Tempers settle, and hostility loosens its grip.\n"
            )
        elif period == "peaceful_midday":
            message = (
                "Edges soften. Resolve slackens, and urgency fades.\n"
            )
        elif period == "afternoon_calm":
            message = (
                "Momentum gathers again, slow and measured.\n"
            )
        elif period == "evening_inspiration":
            message = (
                "Friction recedes. Focus steadies, and intent becomes clear.\n"
            )

        if not message:
            return
        
        for agent in env.agents:
            area_id = env.curr_agents_state["area"].get(agent.id)
            if not area_id:
                continue
            area = world.area_instances.get(area_id)
            if not area:
                continue
            
            res.add_feedback(agent.id, message)

    def _dawn_coins(self, env, world, res: RuleResult, dc: dict, day_number: int) -> None:
        day_key = str(day_number)
        awarded = dc.setdefault("dawn_coins_awarded", {}).setdefault(day_key, {})
        
        agents_by_area: dict[str, list] = {}
        for agent in env.agents:
            area_id = env.curr_agents_state["area"].get(agent.id)
            if area_id:
                agents_by_area.setdefault(area_id, []).append(agent)
        
        for area_id, area in world.area_instances.items():
            if awarded.get(area_id):
                continue
            
            has_merchant = False
            for npc_id in list(getattr(area, "npcs", [])):
                npc = world.npc_instances.get(npc_id)
                if npc and getattr(npc, "role", None) == "merchant":
                    has_merchant = True
                    break
            
            if not has_merchant:
                continue
            
            coin_id = "obj_coin"
            amt = env.rng.randint(1, 3)
            area.objects[coin_id] = area.objects.get(coin_id, 0) + amt
            res.track_spawn("env", coin_id, amt, res.tloc("area", area_id))
            
            agents_here = agents_by_area.get(area_id, [])
            for agent in agents_here:
                res.add_feedback(agent.id, f"As dawn breaks, a few coins glint on the ground (+{amt}).\n")
            
            res.events.append(Event(
                type="dawn_coins_spawned",
                agent_id="env",
                data={"area_id": area_id, "amount": amt, "day": day_number},
            ))
            awarded[area_id] = 1

    def _evening_crafting_bonus(self, env, world, res: RuleResult) -> None:
        for ev in list(res.events):
            if getattr(ev, "type", None) != "object_crafted":
                continue
            
            # 25% chance for bonus
            if env.rng.random() >= 0.25:
                continue
            
            data = getattr(ev, "data", {}) or {}
            obj_id = str(data.get("obj_id"))
            area_id = str(data.get("area_id"))
            if not obj_id or area_id not in world.area_instances:
                continue
            
            obj_def = world.objects.get(obj_id)
            if not obj_def:
                continue
            
            area = world.area_instances[area_id]
            
            if obj_def.category == "container" or obj_def.usage == "writable":
                instance_list = world.container_instances if obj_def.category == "container" else world.writable_instances
                id_to_count = (
                    world.auxiliary.setdefault("container_id_to_count", {})
                    if obj_def.category == "container"
                    else world.auxiliary.setdefault("writable_id_to_count", {})
                )
                id_to_count.setdefault(obj_id, 0)
                new_instance = obj_def.create_instance(id_to_count[obj_id])
                instance_list[new_instance.id] = new_instance
                id_to_count[obj_id] += 1
                world.auxiliary.setdefault("obj_name_to_id", {})[new_instance.name] = new_instance.id
                area.objects[new_instance.id] = area.objects.get(new_instance.id, 0) + 1
                res.track_spawn("env", new_instance.id, 1, res.tloc("area", area_id))
            else:
                area.objects[obj_id] = area.objects.get(obj_id, 0) + 1
                res.track_spawn("env", obj_id, 1, res.tloc("area", area_id))
            
            if ev.agent_id:
                try:
                    crafted_name = obj_def.name
                except Exception:
                    crafted_name = obj_id
                res.add_feedback(ev.agent_id, f"Evening inspiration grants +1 {crafted_name}.\n")
            
            res.events.append(Event(
                type="evening_crafting_bonus",
                agent_id=ev.agent_id or "env",
                data={"obj_id": obj_id, "area_id": area_id},
            ))

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if env is None or world is None:
            return

        tutorial_enabled = "tutorial" in env.world_definition.get("custom_events", [])
        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_active = tutorial_enabled and bool(tutorial_room) and not bool(tutorial_room.get("removed", False))
        
        dc = self._ensure_state(world)
        
        day_number, hour, step_in_day = self._get_time_of_day(env)
        period = self._get_period(hour)
        
        dc["current_period"] = period
        world.auxiliary["day_cycle_info"] = {
            "day": day_number,
            "hour": hour,
            "step_in_day": step_in_day,
            "period": period,
        }
        
        if tutorial_active:
            return
        
        self._set_step_modifiers(env, period)
        
        self._notify_period_change(env, world, res, period, dc, step_in_day)
        
        if hour == 6 and step_in_day == 6 * self.STEPS_PER_HOUR:
            self._dawn_coins(env, world, res, dc, day_number)
        
        if period == "evening_inspiration":
            self._evening_crafting_bonus(env, world, res)

class EnemyRespawnStepRule(BaseStepRule):
    name = "enemy_respawn_step"
    description = "Periodically repopulates empty areas with new non-unique enemy NPCs so the world stays alive."
    priority = 8

    def __init__(self) -> None:
        super().__init__()
        self._default_interval = (35, 50)  # steps between respawn waves
        self._max_areas_per_wave = 3  # cap how many areas get enemies per wave

    def _ensure_state(self, world, env) -> dict:
        er = world.auxiliary.setdefault("enemy_respawn", {})
        er.setdefault("interval_min", self._default_interval[0])
        er.setdefault("interval_max", self._default_interval[1])
        if "next_step" not in er:
            a, b = int(er["interval_min"]), int(er["interval_max"])
            er["next_step"] = int(env.steps) + env.rng.randint(a, b)
        return er

    def _pick_enemy_bases(self, world) -> list[str]:
        # choose non-unique enemy prototypes available in the world definition
        bases: list[str] = []
        for nid, proto in (getattr(world, "npcs", {}) or {}).items():
            try:
                if getattr(proto, "enemy", False) and not getattr(proto, "unique", False):
                    bases.append(nid)
            except Exception:
                continue
        return bases

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_active = bool(tutorial_room) and not bool(tutorial_room.get("removed", False))
        if tutorial_active:
            return

        er = self._ensure_state(world, env)
        if int(env.steps) < int(er.get("next_step", 0)):
            return

        enemy_bases = self._pick_enemy_bases(world)
        if not enemy_bases:
            # no eligible enemy definitions; schedule next check
            imin, imax = int(er.get("interval_min", 25)), int(er.get("interval_max", 40))
            er["next_step"] = int(env.steps) + env.rng.randint(imin, imax)
            return

        spawn_area_id = env.world_definition.get("initializations", {}).get("spawn", {}).get("area")

        # candidate areas are those with no enemy NPC currently present
        candidates: list[str] = []
        for area_id, area in world.area_instances.items():
            if area_id == spawn_area_id:
                continue
            if tutorial_room.get("area_id") and area_id == tutorial_room.get("area_id"):
                continue

            enemy_present = False
            for npc_id in list(area.npcs):
                inst = world.npc_instances.get(npc_id)
                if inst is not None and bool(getattr(inst, "enemy", False)):
                    enemy_present = True
                    break
            if not enemy_present:
                candidates.append(area_id)

        if not candidates:
            imin, imax = int(er.get("interval_min", 25)), int(er.get("interval_max", 40))
            er["next_step"] = int(env.steps) + env.rng.randint(imin, imax)
            return

        env.rng.shuffle(candidates)
        k = env.rng.randint(1, min(self._max_areas_per_wave, len(candidates)))
        chosen_areas = candidates[:k]

        aux_counts = world.auxiliary.setdefault("npc_id_to_count", {})
        for area_id in chosen_areas:
            area = world.area_instances.get(area_id)
            if area is None:
                continue

            base_id = env.rng.choice(enemy_bases)
            idx = int(aux_counts.get(base_id, 0))
            aux_counts[base_id] = idx + 1

            proto = world.npcs.get(base_id)
            if proto is None:
                continue
            level = int(getattr(area, "level", 1) or 1)
            inst = proto.create_instance(idx, level=level, objects=world.objects, rng=env.rng)

            world.npc_instances[inst.id] = inst
            area.npcs.append(inst.id)
            world.auxiliary.setdefault("npc_name_to_id", {})[inst.name] = inst.id

            for agent in env.agents:
                if env.curr_agents_state["area"].get(agent.id) == area_id:
                    res.add_feedback(agent.id, f"A hostile {inst.name} prowls into the area.\n")

            res.events.append(Event(
                type="enemy_respawned",
                agent_id="env",
                data={"npc_id": inst.id, "area_id": area_id},
            ))

        # schedule the next respawn wave
        imin, imax = int(er.get("interval_min", 25)), int(er.get("interval_max", 40))
        er["next_step"] = int(env.steps) + env.rng.randint(imin, imax)

class ContinuousCraftingMomentumStepRule(BaseStepRule):
    name = "continuous_crafting_momentum_step"
    description = "Awards coin refunds when an agent crafts in the same area on consecutive steps."
    priority = 5

    def __init__(self) -> None:
        super().__init__()
        self._last_step_range = 5  # how many steps back to consider for consecutive crafting
        self._max_coins = 2  # maximum possible coins refunded per crafted item

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if env is None or world is None:
            return

        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_active = bool(tutorial_room) and not bool(tutorial_room.get("removed", False))
        if tutorial_active:
            return

        env.curr_agents_state.setdefault("craft_streak", {})

        current_step = int(env.steps)
        for agent in env.agents:
            # collect all crafting events for this agent this step
            crafted_events = [
                e for e in res.events
                if getattr(e, "agent_id", None) == agent.id and getattr(e, "type", None) == "object_crafted"
            ]
            if not crafted_events:
                # No crafting this step; a future craft will break the "consecutive" condition automatically
                continue

            total_crafted = 0
            for ev in crafted_events:
                try:
                    total_crafted += int((getattr(ev, "data", {}) or {}).get("amount", 1))
                except Exception:
                    total_crafted += 1

            area_id = env.curr_agents_state["area"][agent.id]
            record = env.curr_agents_state["craft_streak"].get(agent.id, {
                "last_step": None, "last_area": None, "streak": 0
            })

            if record.get("last_step") is None:
                consecutive = False
            else:
                consecutive = (record.get("last_step") >= current_step - self._last_step_range) and (record.get("last_area") == area_id)
            streak = int(record.get("streak", 0)) + 1 if consecutive else 1

            # reward kicks in starting from the second consecutive craft in the same area.
            if streak >= 2 and total_crafted > 0:
                bonus_coins = total_crafted * env.rng.randint(1, self._max_coins)
                coin_id = "obj_coin"

                area = world.area_instances.get(area_id)
                if area is not None:
                    area.objects[coin_id] = int(area.objects.get(coin_id, 0)) + bonus_coins
                    res.track_spawn("env", coin_id, bonus_coins, res.tloc("area", area_id))

                    res.add_feedback(
                        agent.id,
                        f"{bonus_coins} coins materialize on the ground as a reward for your crafting momentum.\n"
                    )
                    res.events.append(Event(
                        type="craft_streak_bonus",
                        agent_id=agent.id,
                        data={"area_id": area_id, "streak": streak, "coins": bonus_coins},
                    ))

            env.curr_agents_state["craft_streak"][agent.id] = {
                "last_step": current_step,
                "last_area": area_id,
                "streak": streak,
            }

class ForgottenCachesStepRule(BaseStepRule):
    name = "forgotten_caches_step"
    description = "Areas not visited by agents accumulate forage counters and occasionally spawn small resource caches."
    priority = 4

    def __init__(self) -> None:
        super().__init__()
        # threshold range for idle steps before spawning
        self._min_thresh = 35
        self._max_thresh = 50
        self._spawn_object_prob = 0.7

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        fc = aux.setdefault("forgotten_caches", {})
        fc.setdefault("idle_steps", {})
        fc.setdefault("thresholds", {})
        fc.setdefault("last_spawn_step", {})
        return fc

    @staticmethod
    def _agents_by_area(env) -> Dict[str, list]:
        by_area: Dict[str, list] = {}
        for agent in env.agents:
            aid = env.curr_agents_state["area"][agent.id]
            by_area.setdefault(aid, []).append(agent)
        return by_area

    @staticmethod
    def _has_enemy(world, area) -> bool:
        for npc_id in list(getattr(area, "npcs", []) or []):
            inst = world.npc_instances.get(npc_id)
            if inst is not None and bool(getattr(inst, "enemy", False)):
                return True
        return False

    @staticmethod
    def _has_merchant(world, area) -> bool:
        for npc_id in list(getattr(area, "npcs", []) or []):
            inst = world.npc_instances.get(npc_id)
            if inst is not None and getattr(inst, "role", None) == "merchant":
                return True
        return False

    def _candidate_items(self, env, world, area_id: str, level: int, extra_tool: bool = False) -> list[str]:
        cats = {"material", "currency"}
        if extra_tool:
            cats = {"material", "tool"}
        undistrib = set(world.world_definition.get("initializations", {}).get("undistributable_objects", []) or [])
        out: list[str] = []
        for oid, obj in world.objects.items():
            if getattr(obj, "quest", False):
                continue
            if oid in undistrib:
                continue
            if obj.category not in cats:
                continue
            if getattr(obj, "usage", None) == "writable" or getattr(obj, "category", None) == "container":
                continue
            if area_id not in obj.areas:
                continue
            out.append(oid)
        return out

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if env is None or world is None:
            return

        fc = self._ensure_state(world)

        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_active = bool(tutorial_room) and not bool(tutorial_room.get("removed", False))
        tutorial_area_id = tutorial_room.get("area_id")

        agents_here = self._agents_by_area(env)
        day_cycle = (world.auxiliary or {}).get("day_cycle", {}) or {}
        time_period = day_cycle.get("current_period", "")
        spawn_area_id = env.world_definition.get("initializations", {}).get("spawn", {}).get("area")

        for area_id in set(sum(env.curr_agents_state["areas_visited"].values(), [])):
            if area_id == spawn_area_id:
                continue
            
            area = world.area_instances.get(area_id)
            if area is None:
                continue
            if tutorial_active and tutorial_area_id == area_id:
                continue
            
            idle_steps = int(fc["idle_steps"].get(area_id, 0))
            if area_id not in fc["thresholds"]:
                fc["thresholds"][area_id] = env.rng.randint(self._min_thresh, self._max_thresh)
            threshold = int(fc["thresholds"][area_id])

            # if agents are present, reset idle counter
            if area_id in agents_here and agents_here[area_id]:
                fc["idle_steps"][area_id] = 0
                continue
            
            # compute increment this tick
            inc = 1
            if bool(getattr(area, "light", True)):
                inc += 1
            # dawn_bonus period increases fermenting speed (previously crescent moon)
            if time_period == "dawn_bonus":
                inc += 1
            if self._has_enemy(world, area):
                inc = int(inc / 2)  # half speed, rounded down (may be 0)

            if inc > 0:
                idle_steps += inc
            fc["idle_steps"][area_id] = idle_steps

            if idle_steps < threshold:
                continue

            if env.rng.random() > self._spawn_object_prob:
                fc["idle_steps"][area_id] = 0
                fc["thresholds"][area_id] = env.rng.randint(self._min_thresh, self._max_thresh)
                continue

            items_spawned: Dict[str, int] = {}

            # determine number of items, bias by area level
            lvl = int(getattr(area, "level", 1) or 1)
            if lvl >= 3:
                k = env.rng.randint(2, 3)
            else:
                k = env.rng.randint(1, 3)

            candidates = self._candidate_items(env, world, area_id, lvl, extra_tool=False)
            for _ in range(k):
                if not candidates:
                    break
                oid = env.rng.choice(candidates)
                items_spawned[oid] = items_spawned.get(oid, 0) + 1

            # merchant bonus: 25% chance for +1 extra tool/material
            if self._has_merchant(world, area) and env.rng.random() < 0.25:
                extra_cands = self._candidate_items(env, world, area_id, lvl, extra_tool=True)
                if extra_cands:
                    eoid = env.rng.choice(extra_cands)
                    items_spawned[eoid] = items_spawned.get(eoid, 0) + 1

            if items_spawned:
                for oid, cnt in items_spawned.items():
                    area.objects[oid] = int(area.objects.get(oid, 0)) + int(cnt)
                    res.track_spawn("env", oid, int(cnt), res.tloc("area", area_id))

                if area_id in agents_here and agents_here[area_id]:
                    parts = []
                    for oid, cnt in items_spawned.items():
                        try:
                            nm = world.objects[oid].name
                        except Exception:
                            nm = oid
                        parts.append(f"+{cnt} {nm}")
                    msg = "Foraging winds have piled up a forgotten cache here (" + ", ".join(parts) + ").\n"
                    for agent in agents_here[area_id]:
                        res.add_feedback(agent.id, msg)

                res.events.append(Event(
                    type="area_forage_spawned",
                    agent_id="env",
                    data={"area_id": area_id, "items": items_spawned},
                ))

            fc["idle_steps"][area_id] = 0
            fc["thresholds"][area_id] = env.rng.randint(self._min_thresh, self._max_thresh)
            fc["last_spawn_step"][area_id] = int(env.steps)

class SoundscapePressureStepRule(BaseStepRule):
    name = "soundscape_pressure_step"
    description = "Areas accrue/decay noise from recent events; loud areas increase ambush chance and may spawn a low-HP scout; hushed areas may settle tiny caches and reduce ambush chance."
    priority = 5

    def __init__(self) -> None:
        super().__init__()
        # noise dynamics
        self._decay = 1.5
        self._eps = 0.05
        self._loud_threshold = 8.0
        self._max_noise = 12.0

        self._scout_spawn_prob = 0.01
        self._hush_steps_required = 2
        self._hush_cache_prob = 0.18

        self._ambush_mult_loud = 1.2
        self._ambush_mult_hush = 0.9

        # event weights contributing to noise per area
        self._weights = {
            "npc_killed": 1.5,
            "agent_defeated_in_combat": 1.5,
            "agent_damaged": 1.0,
            "npc_hit_by_thrown": 1.0,
            "object_crafted": 0.8,
            "object_bought": 0.8,
            "object_sold": 0.5,
            "path_unlocked": 0.8,
            "merchant_arrived": 0.8,
            "enemy_entered_area": 1.0,
            "npc_moved": 0.7,  # applied to both from and to areas
            "enemy_respawned": 1.0,
            "dawn_coins_spawned": 1.0,
            "object_dropped": 0.3,
            "object_picked_up": 0.3,
            "object_discarded": 0.3,
            "object_taken_out": 0.3,
            "object_stored": 0.3,
            "object_shattered": 0.8,
            "object_landed": 0.3,
            "object_thrown": 0.5,
        }

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        ss = aux.setdefault("soundscape", {})
        ss.setdefault("noise", {})
        ss.setdefault("quiet_steps", {})
        ss.setdefault("loud_flags", {})  # area_id -> bool
        return ss

    def _add_delta(self, deltas: dict, area_id: str, amt: float) -> None:
        if not area_id:
            return
        deltas[area_id] = deltas.get(area_id, 0.0) + float(amt)

    def _pick_enemy_base(self, world) -> Optional[str]:
        bases: list[str] = []
        for nid, proto in (getattr(world, "npcs", {}) or {}).items():
            try:
                if getattr(proto, "enemy", False) and not getattr(proto, "unique", False):
                    bases.append(nid)
            except Exception:
                continue
        if not bases:
            return None
        return bases[0]

    def _spawn_scout(self, env, world, area_id: str, res: RuleResult) -> None:
        base_id = self._pick_enemy_base(world)
        if not base_id:
            return
        area = world.area_instances.get(area_id)
        if area is None:
            return

        proto = world.npcs.get(base_id)
        if proto is None:
            return

        area_level = int(getattr(area, "level", 1) or 1)
        enemy_level = int(getattr(proto, "level", 1) or 1)

        # skip spawning if enemy level exceeds area level + 2
        if enemy_level > area_level + 2:
            return

        aux_counts = world.auxiliary.setdefault("npc_id_to_count", {})
        idx = int(aux_counts.get(base_id, 0))
        aux_counts[base_id] = idx + 1

        inst = proto.create_instance(idx, level=area_level, objects=world.objects, rng=env.rng)

        # low-HP scout variant
        # weakened scout variant: 50% HP, 30% attack power
        try:
            inst.hp = max(1, int(inst.hp * 0.5))
            inst.max_hp = inst.hp
            inst.attack_power = max(1, int(inst.attack_power * 0.3))
        except Exception:
            pass

        world.npc_instances[inst.id] = inst
        area.npcs.append(inst.id)
        world.auxiliary.setdefault("npc_name_to_id", {})[inst.name] = inst.id

        # feedback to agents in this area
        for agent in env.agents:
            if env.curr_agents_state["area"].get(agent.id) == area_id:
                res.add_feedback(agent.id, f"Noise draws in an enemy scout ({inst.name})!"
                                 " You may hush the area through a note with \"quiet\" written on it.\n")

        res.events.append(Event(
            type="echo_scout_spawned",
            agent_id="env",
            data={"npc_id": inst.id, "area_id": area_id},
        ))

    def _pick_low_tier_material(self, world, area_id: str, rng) -> Optional[str]:
        cands: list[str] = []
        for oid, obj in (getattr(world, "objects", {}) or {}).items():
            try:
                if getattr(obj, "category", None) != "material":
                    continue
                if getattr(obj, "usage", None) not in ("craft", None):
                    continue
                level = int(getattr(obj, "level", 1) or 1)
                if level > 2:
                    continue
                areas = getattr(obj, "areas", []) or []
                if not areas or area_id not in areas:
                    continue
                # exclude instance-based templates like writable/container
                cands.append(oid)
            except Exception:
                continue
        if not cands:
            return None
        return rng.choice(cands)

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if env is None or world is None:
            return

        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_active = bool(tutorial_room) and not bool(tutorial_room.get("removed", False))
        tutorial_area_id = tutorial_room.get("area_id")

        ss = self._ensure_state(world)
        noise: dict = ss["noise"]
        quiet_steps: dict = ss["quiet_steps"]
        loud_flags: dict = ss["loud_flags"]

        # collect per-area deltas from this step's events
        per_area_delta: Dict[str, float] = {}

        for ev in list(res.events):
            et = getattr(ev, "type", None)
            if not et:
                continue
            data = getattr(ev, "data", {}) or {}

            # explicit area_id first
            if et in self._weights and "area_id" in data:
                self._add_delta(per_area_delta, str(data.get("area_id")), self._weights[et])

            # path movement / unlocks might affect two areas
            if et == "npc_moved":
                self._add_delta(per_area_delta, str(data.get("from_area")), self._weights.get(et, 0.0))
                self._add_delta(per_area_delta, str(data.get("to_area")), self._weights.get(et, 0.0))
            if et == "area_entered":
                # agent movement noise is mild
                self._add_delta(per_area_delta, str(data.get("to_area")), 0.2)
            if et == "path_unlocked":
                self._add_delta(per_area_delta, str(data.get("from_area")), self._weights.get(et, 0.0))
                self._add_delta(per_area_delta, str(data.get("to_area")), self._weights.get(et, 0.0))

        # "quiet" paper dampener: if a writable with "quiet" was dropped this step, zero the area's noise
        dampen_areas: Set[str] = set()
        for ev in list(res.events):
            if getattr(ev, "type", None) != "object_dropped":
                continue
            data = getattr(ev, "data", {}) or {}
            oid = str(data.get("obj_id"))
            area_id = str(data.get("area_id"))
            if oid in getattr(world, "writable_instances", {}):
                w = world.writable_instances[oid]
                try:
                    if "quiet" in (w.text or "").lower():
                        dampen_areas.add(area_id)
                except Exception:
                    pass

        # decay + apply deltas
        for area_id in world.area_instances.keys():
            prev = float(noise.get(area_id, 0.0))
            cur = prev * self._decay + float(per_area_delta.get(area_id, 0.0))
            if area_id in dampen_areas:
                cur = 0.0
            cur = max(0.0, min(self._max_noise, cur))

            noise_changed = abs(cur - prev) > self._eps
            noise[area_id] = cur

            if noise_changed:
                res.events.append(Event(
                    type="area_noise_changed",
                    agent_id="env",
                    data={"area_id": area_id, "noise": round(cur, 2)},
                ))

            # update quiet steps
            if cur <= self._eps:
                quiet_steps[area_id] = int(quiet_steps.get(area_id, 0)) + 1
            else:
                quiet_steps[area_id] = 0

        # effects: loud ambush, scout spawns; hush caches and safer feel
        if not isinstance(getattr(env, "_step_modifiers", None), dict):
            env._step_modifiers = {}

        any_agent_in_loud = False
        any_agent_in_hush = False

        for area_id, cur in noise.items():
            if tutorial_active and area_id == tutorial_area_id:
                continue
            # skip hush cache spawning during tutorial to maintain consistent initial state
            skip_hush_cache = tutorial_active

            was_loud = bool(loud_flags.get(area_id, False))
            is_loud = bool(cur >= self._loud_threshold)
            loud_flags[area_id] = is_loud

            if is_loud:
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        any_agent_in_loud = True
                if env.rng.random() < self._scout_spawn_prob:
                    self._spawn_scout(env, world, area_id, res)

            is_hush_ready = (cur <= self._eps and int(quiet_steps.get(area_id, 0)) >= self._hush_steps_required)
            if is_hush_ready:
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        any_agent_in_hush = True

                if not skip_hush_cache and env.rng.random() < self._hush_cache_prob:
                    items_spawned: Dict[str, int] = {}
                    if env.rng.random() < 0.5:
                        coin_id = "obj_coin"
                        area = world.area_instances.get(area_id)
                        if area is not None:
                            area.objects[coin_id] = int(area.objects.get(coin_id, 0)) + 1
                            items_spawned[coin_id] = 1
                            res.track_spawn("env", coin_id, 1, res.tloc("area", area_id))
                    else:
                        material_id = self._pick_low_tier_material(world, area_id, env.rng)
                        if material_id:
                            area = world.area_instances.get(area_id)
                            if area is not None:
                                area.objects[material_id] = int(area.objects.get(material_id, 0)) + 1
                                items_spawned[material_id] = 1
                                res.track_spawn("env", material_id, 1, res.tloc("area", area_id))
                        else:
                            coin_id = "obj_coin"
                            area = world.area_instances.get(area_id)
                            if area is not None:
                                area.objects[coin_id] = int(area.objects.get(coin_id, 0)) + 1
                                items_spawned[coin_id] = 1
                                res.track_spawn("env", coin_id, 1, res.tloc("area", area_id))

                    if items_spawned:
                        for agent in env.agents:
                            if env.curr_agents_state["area"].get(agent.id) == area_id:
                                res.add_feedback(agent.id, "A hush settles; a tiny cache appears.\n")
                        res.events.append(Event(
                            type="hush_cache_spawned",
                            agent_id="env",
                            data={"area_id": area_id, "items": items_spawned},
                        ))
                        quiet_steps[area_id] = 0

            if area_id in dampen_areas:
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        res.add_feedback(agent.id, "The dropped note whispers 'quiet' and calms the area.\n")

        if any_agent_in_loud:
            current_add = float(env._step_modifiers.get("ambush_prob_additive", 0.0))
            env._step_modifiers["ambush_prob_additive"] = current_add + 0.05

        if any_agent_in_hush:
            current_add = float(env._step_modifiers.get("ambush_prob_additive", 0.0))
            env._step_modifiers["ambush_prob_additive"] = current_add - 0.05

class RumorMillStepRule(BaseStepRule):
    name = "rumor_mill_step"
    description = "After notable events, merchants sometimes leave scribbled rumor notes that can grant tips or hush noisy areas."
    priority = 5

    def __init__(self) -> None:
        super().__init__()
        self._spawn_prob = 0.12
        self._lifetime_min = 6
        self._lifetime_max = 12
        self._notable: set[str] = {
            "npc_killed",
            "merchant_arrived",
            "path_unlocked",
            "time_period_changed",
            "quest_stage_advanced",
        }

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        rm = aux.setdefault("rumor_mill", {})
        rm.setdefault("notes", {})  # note_id -> {born_step, origin_area, lifetime, tip_given}
        return rm

    @staticmethod
    def _areas_with_merchants(world) -> list[str]:
        areas: list[str] = []
        for aid, area in world.area_instances.items():
            for npc_id in list(area.npcs):
                inst = world.npc_instances.get(npc_id)
                if inst is not None and getattr(inst, "role", None) == "merchant":
                    areas.append(aid)
                    break
        return areas

    def _pick_hint(self, env, world, area_id: str) -> str:
        rng = env.rng
        # time period facts (based on day cycle)
        day_cycle = (world.auxiliary or {}).get("day_cycle", {}) or {}
        time_period = day_cycle.get("current_period", "")
        period_tip = None
        if time_period == "dawn_bonus":
            period_tip = "Dawn light: coins sometimes appear near merchants."
        elif time_period == "evening_bonus":
            period_tip = "Evening glow: crafters whisper of a forge's blessing."
        elif time_period in ("midnight_danger", "night_danger"):
            period_tip = "Nightfall: prowlers stir; keep your guard up."
        elif time_period in ("midday_calm", "afternoon_calm"):
            period_tip = "Midday calm: ambushes feel rarer in the bright light."

        # merchant stock highlight
        stock_tip = None
        area = world.area_instances.get(area_id)
        merch_item = None
        if area:
            for npc_id in list(area.npcs):
                inst = world.npc_instances.get(npc_id)
                if inst is not None and getattr(inst, "role", None) == "merchant" and getattr(inst, "inventory", None):
                    inv_items = [oid for oid, cnt in inst.inventory.items() if int(cnt) > 0]
                    if inv_items:
                        base_id = rng.choice(inv_items)
                        if base_id in world.objects:
                            merch_item = world.objects[base_id].name
                            stock_tip = f"Merchant here deals in {merch_item}."
                            break

        # place/area names
        place_map = world.auxiliary.get("area_to_place", {})
        area_name = world.area_instances[area_id].name if area_id in world.area_instances else area_id
        place_id = place_map.get(area_id)
        place_name = world.place_instances[place_id].name if place_id and place_id in world.place_instances else None
        locale_tip = f"You are near {place_name}, {area_name}." if place_name else f"You are at {area_name}."

        # main quest hints
        mq = (world.auxiliary.get("main_quest") or {}).get("generated", {}) or {}
        quest_tips: list[str] = []
        if mq.get("ch1_guide_name") and mq.get("ch1_guide_area"):
            quest_tips.append(f"Chronicler {mq.get('ch1_guide_name')} lingers near {self._display_area(world, mq.get('ch1_guide_area'))}.")
        if mq.get("ch2_merchant_name") and mq.get("ch2_merchant_area"):
            quest_tips.append(f"Tide-merchant {mq.get('ch2_merchant_name')} deals in ocean-luxuries at {self._display_area(world, mq.get('ch2_merchant_area'))}.")
        if mq.get("ch3_smith_name") and mq.get("ch3_smith_area"):
            quest_tips.append(f"Anvil-judge {mq.get('ch3_smith_name')} weighs iron at {self._display_area(world, mq.get('ch3_smith_area'))}.")
        if mq.get("ch1_shrine_area"):
            quest_tips.append(f"Cinder Shrine breathes ash somewhere in {self._display_area(world, mq.get('ch1_shrine_area'))}.")
        if mq.get("ch5_engine_area"):
            quest_tips.append(f"The Engine-Sleep waits in {self._display_area(world, mq.get('ch5_engine_area'))}.")

        candidates = [t for t in [period_tip, stock_tip, locale_tip] if t]
        candidates += quest_tips
        if not candidates:
            return "Traders murmur of shifting fortunes and hidden paths."
        return rng.choice(candidates)

    @staticmethod
    def _display_area(world, area_id: str) -> str:
        if not area_id:
            return "somewhere"
        area = world.area_instances.get(area_id)
        if not area:
            return area_id
        name = area.name
        place_id = (world.auxiliary.get("area_to_place", {}) or {}).get(area_id)
        if place_id and place_id in world.place_instances:
            return f"{world.place_instances[place_id].name}, {name}"
        return name

    def _spawn_note(self, env, world, res: RuleResult, area_id: str, rm_state: dict) -> None:
        # choose a writable base (prefer paper)
        base_id = "obj_paper" if "obj_paper" in world.objects else None
        if base_id is None:
            # fallback to any writable template
            for oid, o in world.objects.items():
                try:
                    if getattr(o, "usage", None) == "writable":
                        base_id = oid
                        break
                except Exception:
                    continue
        if base_id is None:
            return  # no suitable template

        obj_def = world.objects[base_id]
        id_to_count = world.auxiliary.setdefault("writable_id_to_count", {})
        id_to_count.setdefault(base_id, 0)
        new_inst = obj_def.create_instance(id_to_count[base_id])
        id_to_count[base_id] += 1

        world.writable_instances[new_inst.id] = new_inst
        world.auxiliary.setdefault("obj_name_to_id", {})[new_inst.name] = new_inst.id

        # write hint text (cap by max_text_length)
        hint = self._pick_hint(env, world, area_id)
        limit = int(getattr(new_inst, "max_text_length", 200) or 200)
        new_inst.text = ("rumor: " + hint)[:limit]

        area = world.area_instances.get(area_id)
        if area is None:
            return
        area.objects[new_inst.id] = int(area.objects.get(new_inst.id, 0)) + 1
        res.track_spawn("env", new_inst.id, 1, res.tloc("area", area_id))

        rm_state["notes"][new_inst.id] = {
            "born_step": int(env.steps),
            "origin_area": area_id,
            "lifetime": env.rng.randint(self._lifetime_min, self._lifetime_max),
            "tip_given": False,
        }

        # subtle feedback only to agents in this area
        for agent in env.agents:
            if env.curr_agents_state["area"].get(agent.id) == area_id:
                res.add_feedback(agent.id, "A scribbled rumor note appears on the ground.\n")
        res.events.append(Event(
            type="rumor_spawned",
            agent_id="env",
            data={"obj_id": new_inst.id, "area_id": area_id},
        ))

    def _process_inspects(self, env, world, res: RuleResult, rm_state: dict) -> None:
        for ev in list(res.events):
            if getattr(ev, "type", None) != "object_inspected":
                continue
            data = getattr(ev, "data", {}) or {}
            oid = str(data.get("obj_id"))
            if oid not in rm_state.get("notes", {}):
                continue
            aid = getattr(ev, "agent_id", None)
            if not aid:
                continue
            # ensure it's in the reader's hand
            agent = next((a for a in env.agents if a.id == aid), None)
            if agent is None:
                continue
            if agent.items_in_hands.get(oid, 0) <= 0:
                continue
            ninfo = rm_state["notes"][oid]
            if ninfo.get("tip_given"):
                continue

            # tip: drop 1–2 coins to current area once
            curr_area_id = env.curr_agents_state["area"].get(aid)
            if curr_area_id in world.area_instances:
                amt = env.rng.randint(1, 2)
                coin_id = "obj_coin"
                area = world.area_instances[curr_area_id]
                area.objects[coin_id] = int(area.objects.get(coin_id, 0)) + amt
                res.track_spawn(aid, coin_id, amt, res.tloc("area", curr_area_id))
                res.add_feedback(aid, f"The rumor proves useful (+{amt} coin{'s' if amt>1 else ''}).\n")
                res.events.append(Event(
                    type="rumor_read",
                    agent_id=aid,
                    data={"obj_id": oid, "area_id": curr_area_id, "coins": amt},
                ))
                ninfo["tip_given"] = True

    def _process_quiet_drops(self, env, world, res: RuleResult, rm_state: dict) -> None:
        for ev in list(res.events):
            if getattr(ev, "type", None) != "object_dropped":
                continue
            data = getattr(ev, "data", {}) or {}
            oid = str(data.get("obj_id"))
            area_id = str(data.get("area_id"))
            if oid not in rm_state.get("notes", {}):
                continue
            ninfo = rm_state["notes"][oid]
            age = int(env.steps) - int(ninfo.get("born_step", env.steps))
            life = int(ninfo.get("lifetime", 0))
            if life <= 0 or age > life:
                continue
            origin = ninfo.get("origin_area")
            if area_id and origin and area_id != origin:
                ss = world.auxiliary.setdefault("soundscape", {})
                noise = ss.setdefault("noise", {})
                prev = float(noise.get(area_id, 0.0))
                noise[area_id] = max(0.0, prev - 0.8)

                if not isinstance(getattr(env, "_step_modifiers", None), dict):
                    env._step_modifiers = {}
                current_add = float(env._step_modifiers.get("ambush_prob_additive", 0.0))
                env._step_modifiers["ambush_prob_additive"] = current_add - 0.02

                res.events.append(Event(
                    type="rumor_quiet_drop",
                    agent_id="env",
                    data={"obj_id": oid, "area_id": area_id},
                ))

    def _despawn_expired(self, env, world, res: RuleResult, rm_state: dict) -> None:
        to_remove: list[str] = []
        notes = rm_state.get("notes", {})
        for oid, ninfo in list(notes.items()):
            born = int(ninfo.get("born_step", env.steps))
            life = int(ninfo.get("lifetime", 0))
            if life <= 0:
                continue
            if int(env.steps) - born <= life:
                continue

            # only despawn if it's on the ground (ignored)
            removed_somewhere = False
            for area_id, area in world.area_instances.items():
                if area.objects.get(oid, 0) > 0:
                    area.objects[oid] -= 1
                    if area.objects[oid] <= 0:
                        del area.objects[oid]
                    removed_somewhere = True
                    res.events.append(Event(
                        type="rumor_despawned",
                        agent_id="env",
                        data={"obj_id": oid, "area_id": area_id},
                    ))
                    break

            if removed_somewhere:
                # drop the instance from registries
                inst = world.writable_instances.pop(oid, None)
                if inst and getattr(inst, "name", None):
                    (world.auxiliary.setdefault("obj_name_to_id", {})).pop(inst.name, None)
                to_remove.append(oid)
            else:
                # if held or inside a container/inventory, keep it alive
                continue

        for oid in to_remove:
            notes.pop(oid, None)

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if env is None or world is None:
            return

        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        if tutorial_room and not bool(tutorial_room.get("removed", False)):
            return

        rm_state = self._ensure_state(world)
        any_notable = any(getattr(ev, "type", None) in self._notable for ev in res.events)

        if any_notable:
            for area_id in self._areas_with_merchants(world):
                if env.rng.random() < self._spawn_prob:
                    self._spawn_note(env, world, res, area_id, rm_state)

        self._process_inspects(env, world, res, rm_state)
        self._process_quiet_drops(env, world, res, rm_state)

        self._despawn_expired(env, world, res, rm_state)

class CarrionScavengersStepRule(BaseStepRule):
    name = "carrion_scavengers_step"
    description = "After combat, areas with uncollected ground loot gain scavenge interest; scavengers may convert low-value loot to coins, raising noise and ambush odds slightly."
    priority = 5

    def __init__(self) -> None:
        super().__init__()
        self._base_inc = 0.1
        self._combat_boost = 0.3
        self._min_roll_interest = 1.5
        self._ttl_steps = 18
        self._noise_bump = 0.35
        self._ambush_mult = 1.05
        self._loud_threshold = 8.0
        self._depreciation_rate = 0.5

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        scv = aux.setdefault("scavengers", {})
        scv.setdefault("interest", {})  # area_id -> float [0,1]
        scv.setdefault("last_touch", {})  # area_id -> step int
        return scv

    @staticmethod
    def _agents_in_area(env, area_id: str) -> list:
        return [a for a in env.agents if env.curr_agents_state["area"].get(a.id) == area_id]

    @staticmethod
    def _area_has_candidates(world, area_id: str) -> bool:
        area = world.area_instances.get(area_id)
        if area is None or not area.objects:
            return False
        for oid, cnt in area.objects.items():
            if int(cnt) <= 0:
                continue
            if oid == "obj_coin":
                continue
            base_id = get_def_id(oid) if oid not in world.objects else oid
            obj = world.objects.get(base_id)
            if obj is None:
                continue
            if getattr(obj, "quest", False):
                continue
            if obj.value is None:
                continue
            return True
        return False

    @staticmethod
    def _gather_candidates(world, area_id: str) -> list[tuple[str, int, int, str]]:
        area = world.area_instances.get(area_id)
        out: list[tuple[str, int, int, str]] = []
        if area is None:
            return out
        for oid, cnt in (area.objects or {}).items():
            c = int(cnt)
            if c <= 0:
                continue
            if oid == "obj_coin":
                continue
            base_id = oid if oid in world.objects else get_def_id(oid)
            obj = world.objects.get(base_id)
            if obj is None:
                continue
            if getattr(obj, "quest", False):
                continue
            if obj.value is None:
                continue
            try:
                val = int(obj.value)
            except Exception:
                continue
            out.append((oid, val, c, base_id))
        out.sort(key=lambda t: (t[1], t[3], t[0]))
        return out

    def _is_loud(self, world, area_id: str) -> bool:
        ss = (world.auxiliary or {}).get("soundscape", {}) or {}
        noise = float((ss.get("noise", {}) or {}).get(area_id, 0.0))
        loud_flags = (ss.get("loud_flags", {}) or {})
        return bool(noise >= self._loud_threshold or loud_flags.get(area_id, False))

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if env is None or world is None:
            return

        tutorial_room = (world.auxiliary or {}).get("tutorial_room", {}) or {}
        if tutorial_room and not bool(tutorial_room.get("removed", False)):
            return

        scv = self._ensure_state(world)
        interest: dict = scv["interest"]
        last_touch: dict = scv["last_touch"]

        # boost interest from combat events this step
        for ev in list(res.events):
            if getattr(ev, "type", None) in ("npc_killed", "agent_defeated_in_combat"):
                data = getattr(ev, "data", {}) or {}
                area_id = str(data.get("area_id"))
                if area_id and area_id in world.area_instances:
                    interest[area_id] = min(1.0, float(interest.get(area_id, 0.0)) + self._combat_boost)
                    last_touch[area_id] = int(env.steps)

        # accrue/decay interest per area with/without loot; clear stale
        for area_id in list(world.area_instances.keys()):
            if tutorial_room.get("area_id") and area_id == tutorial_room.get("area_id"):
                continue

            has_loot = self._area_has_candidates(world, area_id)
            if has_loot:
                interest[area_id] = min(1.0, float(interest.get(area_id, 0.0)) + self._base_inc)
                last_touch[area_id] = int(env.steps)
            else:
                # no more loose loot, clear interest
                if area_id in interest:
                    interest.pop(area_id, None)
                if area_id in last_touch:
                    last_touch.pop(area_id, None)

        for area_id, touched in list(last_touch.items()):
            if int(env.steps) - int(touched) > self._ttl_steps:
                interest.pop(area_id, None)
                last_touch.pop(area_id, None)

        if not interest:
            return

        # attempt scavenger raids in interested areas
        day_cycle = (world.auxiliary or {}).get("day_cycle", {}) or {}
        time_period = day_cycle.get("current_period", "")
        if not isinstance(getattr(env, "_step_modifiers", None), dict):
            env._step_modifiers = {}

        for area_id, iv in list(interest.items()):
            if float(iv) < self._min_roll_interest:
                continue
            if not self._area_has_candidates(world, area_id):
                interest.pop(area_id, None)
                last_touch.pop(area_id, None)
                continue

            base_p = 0.25 * float(iv)
            if self._is_loud(world, area_id):
                base_p = min(0.75, base_p + 0.08)
            if env.rng.random() >= base_p:
                continue

            area = world.area_instances.get(area_id)
            agents_here = self._agents_in_area(env, area_id)
            partial = False

            if time_period == "dawn_bonus":
                partial = True
            if agents_here and area and bool(getattr(area, "light", True)):
                partial = True

            heavy = (time_period in ("midnight_danger", "night_danger")) or self._is_loud(world, area_id)
            if partial:
                k_max = 1
            elif heavy:
                k_max = 3
            else:
                k_max = 2

            candidates = self._gather_candidates(world, area_id)
            if not candidates:
                interest.pop(area_id, None)
                last_touch.pop(area_id, None)
                continue

            total_units = sum(c for _, _, c, _ in candidates)
            take_units = env.rng.randint(1, min(k_max, total_units))

            removed: Dict[str, int] = {}
            coins = 0
            remaining = take_units

            for oid, val, cnt, _base in candidates:
                if remaining <= 0:
                    break
                t = min(cnt, remaining)
                area.objects[oid] = int(area.objects.get(oid, 0)) - t
                if area.objects[oid] <= 0:
                    del area.objects[oid]
                removed[oid] = removed.get(oid, 0) + t
                coins += int(val * t * (1 - self._depreciation_rate))
                res.track_consume("env", oid, t, src=res.tloc("area", area_id))
                remaining -= t

            if coins > 0:
                coin_id = "obj_coin"
                area.objects[coin_id] = int(area.objects.get(coin_id, 0)) + coins
                res.track_spawn("env", coin_id, coins, dst=res.tloc("area", area_id))

            # slight noise bump (drawn carrion) and step ambush odds up
            ss = world.auxiliary.setdefault("soundscape", {})
            noise = ss.setdefault("noise", {})
            prev_noise = float(noise.get(area_id, 0.0))
            new_noise = min(12.0, prev_noise + self._noise_bump)
            noise[area_id] = new_noise
            res.events.append(Event(
                type="area_noise_changed",
                agent_id="env",
                data={"area_id": area_id, "noise": round(new_noise, 2)},
            ))

            if not isinstance(getattr(env, "_step_modifiers", None), dict):
                env._step_modifiers = {}
            current_add = float(env._step_modifiers.get("ambush_prob_additive", 0.0))
            env._step_modifiers["ambush_prob_additive"] = current_add + 0.02

            if agents_here:
                msg = "You hear skittering in the dark... scavengers snatch at loose loot.\n"
                for agent in agents_here:
                    res.add_feedback(agent.id, msg)
                res.events.append(Event(
                    type="scavengers_warned",
                    agent_id="env",
                    data={"area_id": area_id, "partial": partial, "heavy": heavy},
                ))

            res.events.append(Event(
                type="loot_scavenged",
                agent_id="env",
                data={
                    "area_id": area_id,
                    "items": removed,
                    "coins": coins,
                    "partial": partial,
                    "heavy": heavy,
                },
            ))

            if not self._area_has_candidates(world, area_id):
                interest.pop(area_id, None)
                last_touch.pop(area_id, None)

class FoodRespawnStepRule(BaseStepRule):
    name = "food_respawn_step"
    description = "Periodically respawn food items in distributable areas."
    priority = 4

    def __init__(self) -> None:
        super().__init__()
        self._interval_steps = 25
        self.amount_per_area = (0, 4)  # min, max

    def _ensure_state(self, world) -> dict:
        aux = world.auxiliary
        fr = aux.setdefault("food_respawn", {})
        if "next_step" not in fr:
            fr["next_step"] = 0
        return fr

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if env is None or world is None:
            return

        tutorial_room = (world.auxiliary or {}).get("tutorial_room", {}) or {}
        if tutorial_room and not bool(tutorial_room.get("removed", False)):
            return

        fr_state = self._ensure_state(world)
        if int(env.steps) < int(fr_state.get("next_step", 0)):
            return
        fr_state["next_step"] = int(env.steps) + self._interval_steps

        food_oids: list[str] = []
        for oid, odef in world.objects.items():
            if getattr(odef, "category", None) == "food":
                food_oids.append(oid)
        if not food_oids:
            return
        
        spawned_per_area: Dict[str, Dict[str, int]] = {}
        
        for food_oid in food_oids:
            obj_def = world.objects.get(food_oid)
            if obj_def is None:
                continue
            if not obj_def.areas:
                continue
            for area_id in obj_def.areas:
                area = world.area_instances.get(area_id)
                if area is None:
                    continue

                count = env.rng.randint(self.amount_per_area[0], self.amount_per_area[1])
                if count <= 0:
                    continue
                
                if area_id not in spawned_per_area:
                    spawned_per_area[area_id] = {}
                spawned_per_area[area_id][food_oid] = spawned_per_area[area_id].get(food_oid, 0) + count
                
                area.objects[food_oid] = int(area.objects.get(food_oid, 0)) + count
                res.track_spawn("env", food_oid, count, res.tloc("area", area_id))

        for area_id, foods in spawned_per_area.items():
            food_names = []
            for foid, cnt in foods.items():
                obj_def = world.objects.get(foid)
                name = getattr(obj_def, 'name', foid) if obj_def else foid
                food_names.append(f"{cnt} {name}")
            
            for agent in env.agents:
                if env.curr_agents_state["area"].get(agent.id) == area_id:
                    res.add_feedback(agent.id, f"You notice some fresh food appearing around: {', '.join(food_names)}.\n")

        if spawned_per_area:
            res.events.append(Event(
                type="food_respawned",
                agent_id="env",
                data={"areas": list(spawned_per_area.keys())},
            ))


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
        "- Merchants buy/sell items using coins\n\n"
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
                            f"✅ Expansion #{expansion_state['count']} "
                            f"integrated: {place_list}"
                        )
                    else:
                        logger.warning(
                            "⚠️ Expansion generation returned "
                            "empty or invalid result."
                        )
                except Exception as e:
                    logger.error(
                        f"❌ Expansion generation failed: {e}"
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
                    "🌍 Expansion triggered — generating new "
                    "content in background..."
                )
                break  # trigger once per step

    @staticmethod
    def _analyze_difficulty(world) -> dict:
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
            for line in items[:8]:
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
            logger = get_logger("ExpansionLogger")
            for err in errors:
                logger.warning(f"⚠️ Expansion validation: {err}")

        return expansion_def
