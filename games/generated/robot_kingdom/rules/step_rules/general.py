import copy
import math
from utils import *
from games.generated.robot_kingdom.rule import BaseStepRule, RuleContext, RuleResult, Event
from typing import Dict, Optional, Set, Tuple, List, Any
from games.generated.robot_kingdom.world import NPC, Object, Area, Place, Path, Container, Writable
from games.generated.robot_kingdom.agent import Agent
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

                    # --- overload aura counter-damage ---
                    overload_aura = env.curr_agents_state.get("overload_aura", {}).get(agent.id, {})
                    aura_pct = overload_aura.get("aura_pct", 0)
                    if aura_pct > 0 and agent.defense > 0 and npc_instance.hp > 0:
                        counter_damage = max(1, int(agent.defense * aura_pct / 100))
                        npc_instance.hp -= counter_damage
                        res.add_feedback(
                            agent.id,
                            f"{env.person_verbalized['possessive_adjective'].capitalize()} overloaded armor "
                            f"crackles with devil-forged energy, dealing {counter_damage} counter-damage "
                            f"to {npc_instance.name}! "
                            f"{npc_instance.name} has {max(0, npc_instance.hp)} HP remaining.\n"
                        )
                        res.events.append(Event(
                            type="overload_aura_damage",
                            agent_id=agent.id,
                            data={
                                "npc_id": npc_id,
                                "counter_damage": counter_damage,
                                "aura_pct": aura_pct,
                                "area_id": agent_area_id,
                            },
                        ))
                        if npc_instance.hp <= 0:
                            npc_instance.hp = 0
                            res.add_feedback(
                                agent.id,
                                f"The reactive aura destroyed {npc_instance.name}!\n"
                            )
                            npcs_to_remove.append(npc_id)
                            current_area = world.area_instances.get(agent_area_id)
                            if current_area and npc_id in current_area.npcs:
                                current_area.npcs.remove(npc_id)
                                env.curr_agents_state["npcs_killed"][agent.id].append(npc_id)
                                target_npc_str = f"{get_def_id(npc_id)}_{npc_instance.level}"
                                if target_npc_str not in env.curr_agents_state["unique_npcs_killed"][agent.id]:
                                    env.curr_agents_state["unique_npcs_killed"][agent.id].append(target_npc_str)
                                # drop loot
                                for loot_id, loot_count in npc_instance.inventory.items():
                                    if loot_count > 0:
                                        loot_def = world.objects.get(loot_id)
                                        if loot_def:
                                            current_area.objects[loot_id] = current_area.objects.get(loot_id, 0) + loot_count
                                            res.track_spawn(agent.id, loot_id, loot_count, dst=res.tloc("area", current_area.id))
                            res.events.append(Event(
                                type="npc_killed",
                                agent_id=agent.id,
                                data={"npc_id": npc_id, "area_id": agent_area_id},
                            ))
                            continue
                    
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

            # clear rallied ally on death
            if "rallied_allies" in env.curr_agents_state:
                env.curr_agents_state["rallied_allies"][agent.id] = {}

            # clear overload aura on death
            if "overload_aura" in env.curr_agents_state:
                env.curr_agents_state["overload_aura"][agent.id] = {}

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

class CorruptionDecayStepRule(BaseStepRule):
    name = "corruption_decay_step"
    description = (
        "Areas where devil-forged war machines were defeated leak corrupting energy that "
        "gradually damages objects left on the ground. Objects accumulate decay each step "
        "and are eventually destroyed if not picked up in time."
    )
    priority = 8

    # How many steps of corruption exposure before an object is destroyed
    DECAY_THRESHOLD = 8
    # Corruption only ticks every N steps to avoid being too punishing
    TICK_INTERVAL = 2
    # Categories immune to corruption decay
    IMMUNE_CATEGORIES = {"station", "currency", "container"}
    # Usages immune to corruption decay
    IMMUNE_USAGES = {"writable", "unlock"}

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        # --- track areas where enemy NPCs have been killed ---
        corrupted_areas = env.curr_agents_state.setdefault("corrupted_areas", {})
        # Mark areas from npc_killed events this step
        for event in res.events:
            if event.type == "npc_killed":
                area_id = event.data.get("area_id")
                npc_id = event.data.get("npc_id")
                if area_id and npc_id and npc_id in world.npc_instances:
                    npc = world.npc_instances[npc_id]
                    if npc.enemy:
                        # Record the step when corruption started (keep earliest)
                        if area_id not in corrupted_areas:
                            corrupted_areas[area_id] = env.steps

        # Also check historical kills to seed corrupted areas on first run
        if not corrupted_areas:
            for agent in env.agents:
                for npc_id in env.curr_agents_state.get("npcs_killed", {}).get(agent.id, []):
                    if npc_id in world.npc_instances and world.npc_instances[npc_id].enemy:
                        # Find which area the NPC *was* in — we can't know exactly,
                        # so we rely on event-based tracking going forward
                        pass

        # --- only tick decay on interval ---
        if env.steps < 0 or env.steps % self.TICK_INTERVAL != 0:
            return

        # --- object decay tracking: {area_id: {obj_id: decay_counter}} ---
        decay_state = env.curr_agents_state.setdefault("corruption_object_decay", {})

        areas_to_clean = []
        for area_id, corruption_start_step in list(corrupted_areas.items()):
            area = world.area_instances.get(area_id)
            if area is None:
                continue

            # Check if there are still any enemy NPC remains (i.e., no living enemies)
            # Corruption persists as long as the area was a battlefield
            area_decay = decay_state.setdefault(area_id, {})

            # Identify vulnerable objects currently on the ground
            vulnerable_objs = []
            for obj_id, count in list(area.objects.items()):
                if count <= 0:
                    continue
                base_id = get_def_id(obj_id)
                obj_def = world.objects.get(base_id)
                if obj_def is None:
                    continue
                if obj_def.category in self.IMMUNE_CATEGORIES:
                    continue
                if obj_def.usage in self.IMMUNE_USAGES:
                    continue
                # Writable and container instances are also immune
                if obj_id in world.container_instances or obj_id in world.writable_instances:
                    continue
                vulnerable_objs.append((obj_id, base_id, count))

            # Increment decay for objects present, remove tracking for objects no longer there
            current_obj_ids = set(oid for oid, _, _ in vulnerable_objs)
            for tracked_id in list(area_decay.keys()):
                if tracked_id not in current_obj_ids:
                    del area_decay[tracked_id]

            destroyed_items = []
            for obj_id, base_id, count in vulnerable_objs:
                area_decay[obj_id] = area_decay.get(obj_id, 0) + 1
                current_decay = area_decay[obj_id]

                if current_decay >= self.DECAY_THRESHOLD:
                    # Destroy the objects
                    destroyed_count = count
                    del area.objects[obj_id]
                    del area_decay[obj_id]

                    res.track_consume(
                        "env", obj_id, destroyed_count,
                        src=res.tloc("area", area_id),
                    )
                    obj_name = world.objects.get(base_id, None)
                    display_name = obj_name.name if obj_name else obj_id
                    destroyed_items.append((display_name, destroyed_count))

                    res.events.append(Event(
                        type="corruption_destroyed",
                        data={
                            "area_id": area_id,
                            "obj_id": obj_id,
                            "count": destroyed_count,
                        },
                    ))

                elif current_decay == self.DECAY_THRESHOLD - 2:
                    # Warn agents in this area that objects are about to be destroyed
                    obj_name = world.objects.get(base_id, None)
                    display_name = obj_name.name if obj_name else obj_id
                    for agent in env.agents:
                        if env.curr_agents_state["area"].get(agent.id) == area_id:
                            res.add_feedback(
                                agent.id,
                                f"Warning: {count} {display_name} on the ground "
                                f"is corroding from corrupting energy leaking from "
                                f"war machine remains and will soon be destroyed!\n"
                            )

            # Notify agents in the area about destroyed items
            if destroyed_items:
                destruction_text = ", ".join(
                    f"{cnt} {name}" for name, cnt in destroyed_items
                )
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        res.add_feedback(
                            agent.id,
                            f"The corrupting energy from defeated war machine remains "
                            f"has destroyed: {destruction_text}. "
                            f"Loot battlefields quickly to avoid losing valuable salvage!\n"
                        )

            # Clean up area decay if no more vulnerable objects
            if not area_decay:
                areas_to_clean.append(area_id)

        for area_id in areas_to_clean:
            if area_id in decay_state:
                del decay_state[area_id]


class IntimidationStepRule(BaseStepRule):
    name = "intimidation_step"
    description = (
        "Friendly NPCs sharing an area with enemy war machines gradually lose HP from "
        "intimidation each step. If a friendly NPC is defeated this way, the strongest "
        "enemy in the area absorbs their strength, permanently gaining increased attack "
        "power and HP — pressuring players to clear enemies from populated areas before "
        "their allies fall."
    )
    priority = 9  # after combat and death rules

    # HP lost per step per enemy present in the same area
    INTIMIDATION_DAMAGE_PER_ENEMY = 3
    # Multiplier for the bonus the absorbing enemy gains
    ABSORB_ATTACK_BONUS = 5
    ABSORB_HP_BONUS = 15

    # NPC base IDs that are immune to intimidation (quest guides, etc.)
    IMMUNE_NPC_BASE_IDS = {"npc_quest_wayfarer_guide", "npc_side_quest_guide"}

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        intimidation_state = env.curr_agents_state.setdefault("intimidation_state", {})

        for area_id, area in world.area_instances.items():
            if not area.npcs:
                continue

            # Separate friendly and enemy NPCs in this area
            friendly_npcs = []
            enemy_npcs = []
            for npc_id in area.npcs:
                npc = world.npc_instances.get(npc_id)
                if npc is None:
                    continue
                if npc.hp <= 0:
                    continue
                if npc.enemy:
                    enemy_npcs.append(npc)
                else:
                    # Skip immune NPCs (quest guides)
                    if get_def_id(npc_id) in self.IMMUNE_NPC_BASE_IDS:
                        continue
                    friendly_npcs.append(npc)

            if not friendly_npcs or not enemy_npcs:
                # Clean up intimidation state for friendlies no longer threatened
                for npc_id in list(intimidation_state.keys()):
                    npc = world.npc_instances.get(npc_id)
                    if npc and not npc.enemy and npc_id in [n.id for n in friendly_npcs if not enemy_npcs]:
                        pass  # keep state, they're just not threatened now
                continue

            num_enemies = len(enemy_npcs)
            damage = self.INTIMIDATION_DAMAGE_PER_ENEMY * num_enemies

            defeated_friendlies = []
            for friendly in friendly_npcs:
                friendly.hp -= damage

                # Track cumulative intimidation damage for feedback throttling
                intimidation_state.setdefault(friendly.id, {"total_damage": 0})
                intimidation_state[friendly.id]["total_damage"] = (
                    intimidation_state[friendly.id].get("total_damage", 0) + damage
                )

                # Notify agents in this area
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        res.add_feedback(
                            agent.id,
                            f"{friendly.name} cowers under the presence of {num_enemies} "
                            f"war machine(s), losing {damage} HP from intimidation "
                            f"({max(0, friendly.hp)} HP remaining).\n"
                        )

                res.events.append(Event(
                    type="npc_intimidated",
                    data={
                        "npc_id": friendly.id,
                        "area_id": area_id,
                        "damage": damage,
                        "remaining_hp": max(0, friendly.hp),
                        "num_enemies": num_enemies,
                    },
                ))

                if friendly.hp <= 0:
                    friendly.hp = 0
                    defeated_friendlies.append(friendly)

            # Process defeated friendly NPCs
            for fallen in defeated_friendlies:
                # Find the strongest enemy in the area to absorb the strength
                absorber = max(enemy_npcs, key=lambda e: e.attack_power + e.hp)

                old_atk = absorber.attack_power
                old_hp = absorber.hp
                absorber.attack_power += self.ABSORB_ATTACK_BONUS
                absorber.hp += self.ABSORB_HP_BONUS

                # Remove the fallen friendly from the area
                if fallen.id in area.npcs:
                    area.npcs.remove(fallen.id)

                # Clean up intimidation tracking
                if fallen.id in intimidation_state:
                    del intimidation_state[fallen.id]

                # Notify agents in this area
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        res.add_feedback(
                            agent.id,
                            f"{fallen.name} has been overwhelmed by the presence of devil-forged "
                            f"war machines and collapsed! {absorber.name} absorbs their lingering "
                            f"strength, gaining +{self.ABSORB_ATTACK_BONUS} attack power and "
                            f"+{self.ABSORB_HP_BONUS} HP "
                            f"(now {absorber.attack_power} ATK, {absorber.hp} HP).\n"
                        )

                res.events.append(Event(
                    type="npc_fallen_from_intimidation",
                    data={
                        "fallen_npc_id": fallen.id,
                        "fallen_npc_name": fallen.name,
                        "absorber_npc_id": absorber.id,
                        "absorber_npc_name": absorber.name,
                        "absorb_attack_bonus": self.ABSORB_ATTACK_BONUS,
                        "absorb_hp_bonus": self.ABSORB_HP_BONUS,
                        "area_id": area_id,
                    },
                ))

                # Also remove from npc_name_to_id if present
                npc_name = getattr(fallen, "name", None)
                if npc_name and world.auxiliary.get("npc_name_to_id", {}).get(npc_name) == fallen.id:
                    del world.auxiliary["npc_name_to_id"][npc_name]


class TerritoryReclamationStepRule(BaseStepRule):
    name = "territory_reclamation_step"
    description = (
        "Areas left unvisited by any agent for a prolonged number of steps are gradually "
        "reclaimed by dormant devil-forged war machines. Enemy NPCs spontaneously spawn and "
        "existing objects slowly diminish, turning previously cleared safe zones back into "
        "dangerous territory if left unpatrolled."
    )
    priority = 10  # after intimidation and corruption rules

    # How many consecutive unvisited steps before reclamation begins
    DORMANCY_THRESHOLD = 8
    # Reclamation effects tick every N steps of dormancy beyond the threshold
    TICK_INTERVAL = 3
    # Maximum number of enemies that can accumulate from reclamation per area
    MAX_RECLAIMED_ENEMIES_PER_AREA = 3
    # Fraction of objects removed per reclamation tick (rounded down, min 1 if any exist)
    OBJECT_DECAY_FRACTION = 0.15
    # Categories immune to reclamation decay
    IMMUNE_CATEGORIES = {"station", "currency"}
    # Usages immune to reclamation decay
    IMMUNE_USAGES = {"writable", "unlock"}

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if world is None:
            return

        # --- initialise tracking state ---
        area_last_visited = env.curr_agents_state.setdefault("area_last_visited", {})
        reclamation_state = env.curr_agents_state.setdefault("reclamation_state", {})

        # Update last-visited step for every area that currently has an agent
        for agent in env.agents:
            agent_area = env.curr_agents_state["area"].get(agent.id)
            if agent_area:
                area_last_visited[agent_area] = env.steps

        # Ensure every area has an initial last-visited entry
        for area_id in world.area_instances:
            if area_id not in area_last_visited:
                area_last_visited[area_id] = env.steps

        spawn_area = env.world_definition.get("initializations", {}).get("spawn", {}).get("area")

        # --- process each area ---
        for area_id, area in world.area_instances.items():
            # Never reclaim the spawn area
            if area_id == spawn_area:
                continue

            last_visit = area_last_visited.get(area_id, env.steps)
            dormancy = env.steps - last_visit

            # If an agent is currently here, reset reclamation progress
            agent_present = any(
                env.curr_agents_state["area"].get(a.id) == area_id
                for a in env.agents
            )
            if agent_present:
                if area_id in reclamation_state:
                    del reclamation_state[area_id]
                continue

            if dormancy < self.DORMANCY_THRESHOLD:
                continue

            # Initialise per-area reclamation tracking
            rec = reclamation_state.setdefault(area_id, {
                "ticks": 0,
                "last_tick_step": last_visit + self.DORMANCY_THRESHOLD,
                "spawned_enemies": 0,
            })

            # Only tick at intervals
            steps_since_last_tick = env.steps - rec["last_tick_step"]
            if steps_since_last_tick < self.TICK_INTERVAL:
                continue

            rec["ticks"] += 1
            rec["last_tick_step"] = env.steps

            # --- 1. Object decay ---
            vulnerable_objs = []
            for obj_id, count in list(area.objects.items()):
                if count <= 0:
                    continue
                base_id = get_def_id(obj_id)
                obj_def = world.objects.get(base_id)
                if obj_def is None:
                    continue
                if obj_def.category in self.IMMUNE_CATEGORIES:
                    continue
                if obj_def.usage in self.IMMUNE_USAGES:
                    continue
                if obj_id in world.container_instances or obj_id in world.writable_instances:
                    continue
                vulnerable_objs.append((obj_id, base_id, count))

            decayed_items = []
            for obj_id, base_id, count in vulnerable_objs:
                decay_count = max(1, int(count * self.OBJECT_DECAY_FRACTION))
                decay_count = min(decay_count, count)
                area.objects[obj_id] -= decay_count
                if area.objects[obj_id] <= 0:
                    del area.objects[obj_id]

                res.track_consume(
                    "env", obj_id, decay_count,
                    src=res.tloc("area", area_id),
                )
                obj_name = world.objects[base_id].name if base_id in world.objects else obj_id
                decayed_items.append((obj_name, decay_count))

            # --- 2. Enemy NPC spawning ---
            spawned_npc_name = None
            if rec["spawned_enemies"] < self.MAX_RECLAIMED_ENEMIES_PER_AREA:
                # Pick a non-unique enemy NPC prototype to spawn
                enemy_protos = [
                    npc for npc in world.npcs.values()
                    if npc.enemy and not npc.unique and not getattr(npc, "quest", False)
                ]
                if enemy_protos:
                    proto = env.rng.choice(enemy_protos)
                    npc_level = max(1, area.level)
                    npc_id_to_count = world.auxiliary.get("npc_id_to_count", {})
                    idx = npc_id_to_count.get(proto.id, 0)
                    new_npc = proto.create_instance(idx, npc_level, world.objects, env.rng)
                    npc_id_to_count[proto.id] = idx + 1
                    world.auxiliary["npc_id_to_count"] = npc_id_to_count

                    world.npc_instances[new_npc.id] = new_npc
                    world.auxiliary.setdefault("npc_name_to_id", {})[new_npc.name] = new_npc.id
                    area.npcs.append(new_npc.id)
                    rec["spawned_enemies"] += 1
                    spawned_npc_name = new_npc.name

            # --- 3. Emit events ---
            if decayed_items or spawned_npc_name:
                res.events.append(Event(
                    type="territory_reclamation_tick",
                    data={
                        "area_id": area_id,
                        "tick": rec["ticks"],
                        "dormancy_steps": dormancy,
                        "decayed_items": {name: cnt for name, cnt in decayed_items},
                        "spawned_npc": spawned_npc_name,
                    },
                ))

                # Provide dep_tracker hint
                hint_parts = []
                if decayed_items:
                    decay_str = ", ".join(f"{cnt} {name}" for name, cnt in decayed_items)
                    hint_parts.append(f"objects decayed: {decay_str}")
                if spawned_npc_name:
                    hint_parts.append(f"enemy spawned: {spawned_npc_name}")
                res.events.append(Event(
                    type="dep_tracker_hint",
                    data={"hint": f"reclamation@{area.name}: " + "; ".join(hint_parts)},
                ))

            # --- 4. Notify agents in neighboring areas ---
            if spawned_npc_name:
                for neighbor_id in area.neighbors:
                    for agent in env.agents:
                        if env.curr_agents_state["area"].get(agent.id) == neighbor_id:
                            res.add_feedback(
                                agent.id,
                                f"Ominous mechanical grinding echoes from the nearby {area.name}. "
                                f"Dormant war machines are reclaiming the abandoned territory — "
                                f"a {spawned_npc_name} has emerged from the wreckage.\n"
                            )


class UnitedPresenceStepRule(BaseStepRule):
    name = "united_presence_step"
    description = (
        "When two or more agents occupy the same area, the remnants of the old alliance "
        "are emboldened. Each co-located agent receives a stacking defense bonus (+2 per "
        "additional ally present), and enemy NPCs in that area hesitate — skipping their "
        "next combat rhythm action — reflecting the tactical advantage of the united "
        "kingdoms fighting together against the devil-forged war machines."
    )
    priority = 1  # run early, before combat rhythm processing

    DEFENSE_BONUS_PER_ALLY = 2

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        united_state = env.curr_agents_state.setdefault("united_presence", {})

        # --- 1. Build a map of area_id -> list of living agents in that area ---
        area_agents: Dict[str, List] = {}
        for agent in env.agents:
            if agent.hp <= 0:
                continue
            area_id = env.curr_agents_state["area"].get(agent.id)
            if area_id:
                area_agents.setdefault(area_id, []).append(agent)

        # --- 2. Determine which agents are co-located (2+ agents in same area) ---
        co_located_agents: Set[str] = set()
        co_located_areas: Set[str] = set()
        for area_id, agents_in_area in area_agents.items():
            if len(agents_in_area) >= 2:
                for agent in agents_in_area:
                    co_located_agents.add(agent.id)
                co_located_areas.add(area_id)

        # --- 3. Remove defense bonus from agents who are no longer co-located ---
        for agent in env.agents:
            prev_bonus = united_state.get(agent.id, {}).get("defense_bonus", 0)
            if agent.id not in co_located_agents and prev_bonus > 0:
                agent.defense -= prev_bonus
                agent.defense = max(0, agent.defense)
                res.add_feedback(
                    agent.id,
                    f"The spirit of the old alliance fades as "
                    f"{env.person_verbalized['subject_pronoun']} stand alone. "
                    f"Lost {prev_bonus} alliance defense bonus.\n"
                )
                united_state[agent.id] = {"defense_bonus": 0}

        # --- 4. Apply / update defense bonus for co-located agents ---
        for area_id in co_located_areas:
            agents_in_area = area_agents[area_id]
            num_allies = len(agents_in_area) - 1  # allies = others in the area

            for agent in agents_in_area:
                new_bonus = self.DEFENSE_BONUS_PER_ALLY * num_allies
                prev_bonus = united_state.get(agent.id, {}).get("defense_bonus", 0)
                delta = new_bonus - prev_bonus

                if delta != 0:
                    agent.defense += delta
                    agent.defense = max(0, agent.defense)
                    united_state[agent.id] = {"defense_bonus": new_bonus}

                    if delta > 0:
                        ally_names = [a.name for a in agents_in_area if a.id != agent.id]
                        res.add_feedback(
                            agent.id,
                            f"The presence of {', '.join(ally_names)} rekindles the spirit of "
                            f"the united kingdoms! Alliance defense bonus: +{new_bonus}.\n"
                        )
                    elif delta < 0:
                        res.add_feedback(
                            agent.id,
                            f"An ally has departed. Alliance defense bonus reduced to "
                            f"+{new_bonus}.\n"
                        )
                elif new_bonus > 0 and prev_bonus == new_bonus:
                    # Bonus unchanged — no feedback needed each step
                    pass

        # --- 5. Enemy hesitation: skip next rhythm action for enemies in co-located areas ---
        for area_id in co_located_areas:
            agents_in_area = area_agents[area_id]
            area = world.area_instances.get(area_id)
            if area is None:
                continue

            # Collect all enemy NPCs in this area that are in active combat with any agent
            hesitated_npcs: Set[str] = set()
            for agent in agents_in_area:
                active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
                for npc_id, combat_state in list(active_combats.items()):
                    if combat_state.get("area_id") != area_id:
                        continue
                    if npc_id in hesitated_npcs:
                        continue
                    npc = world.npc_instances.get(npc_id)
                    if npc is None or npc.hp <= 0 or not npc.enemy:
                        continue
                    rhythm = npc.combat_pattern
                    if not rhythm:
                        continue

                    # Advance the rhythm index by 1 to skip the next action
                    old_index = combat_state.get("rhythm_index", 0)
                    skipped_action = rhythm[old_index % len(rhythm)]
                    combat_state["rhythm_index"] = (old_index + 1) % len(rhythm)
                    hesitated_npcs.add(npc_id)

                    # Also advance for other agents in combat with the same NPC
                    for other_agent in agents_in_area:
                        if other_agent.id == agent.id:
                            continue
                        other_combats = env.curr_agents_state.get("active_combats", {}).get(other_agent.id, {})
                        if npc_id in other_combats and other_combats[npc_id].get("area_id") == area_id:
                            other_combats[npc_id]["rhythm_index"] = combat_state["rhythm_index"]

            # Notify agents about enemy hesitation
            if hesitated_npcs:
                npc_names = [
                    world.npc_instances[nid].name
                    for nid in hesitated_npcs
                    if nid in world.npc_instances
                ]
                for agent in agents_in_area:
                    res.add_feedback(
                        agent.id,
                        f"Faced with the united strength of the old alliance, "
                        f"{', '.join(npc_names)} hesitate{'s' if len(npc_names) == 1 else ''} "
                        f"— skipping their next action!\n"
                    )

                res.events.append(Event(
                    type="united_presence_hesitation",
                    data={
                        "area_id": area_id,
                        "hesitated_npcs": list(hesitated_npcs),
                        "agent_ids": [a.id for a in agents_in_area],
                    },
                ))


class TacticalWarningStepRule(BaseStepRule):
    name = "tactical_warning_step"
    description = (
        "Friendly NPCs in the same area as an agent engaged in active combat "
        "occasionally shout tactical warnings, revealing the next action in an "
        "enemy war machine's combat rhythm pattern."
    )
    priority = 4  # after CombatRhythmStepRule (2) so rhythm_index already advanced

    # Probability that a friendly NPC shouts a warning per enemy per step
    WARNING_PROBABILITY = 0.30

    # Thematic shout templates per predicted action
    WARNING_TEMPLATES = {
        "attack": [
            "{ally} shouts: \"Watch out! {enemy} is winding up to strike!\"",
            "{ally} warns: \"Brace yourself! {enemy} is preparing an attack!\"",
            "{ally} yells: \"{enemy}'s gears are spinning — it's going to attack next!\"",
        ],
        "defend": [
            "{ally} calls out: \"{enemy} is locking its plates — it will defend next!\"",
            "{ally} shouts: \"Hold your swing! {enemy} is about to raise its guard!\"",
            "{ally} warns: \"{enemy}'s armor plates are shifting into a defensive stance!\"",
        ],
        "wait": [
            "{ally} shouts: \"{enemy}'s engine is sputtering — it will hesitate next!\"",
            "{ally} calls out: \"{enemy} is venting steam — it won't act next turn!\"",
            "{ally} warns: \"Now's your chance! {enemy} is about to stall!\"",
        ],
    }

    # NPC base IDs that should never shout warnings (quest guides, etc.)
    EXCLUDED_NPC_BASE_IDS = {"npc_quest_wayfarer_guide", "npc_side_quest_guide"}

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        for agent in env.agents:
            if agent.hp <= 0:
                continue

            active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
            if not active_combats:
                continue

            agent_area_id = env.curr_agents_state["area"][agent.id]
            current_area = world.area_instances.get(agent_area_id)
            if current_area is None:
                continue

            # Gather friendly NPCs in the same area (non-enemy, non-quest-guide)
            friendly_npcs = []
            for npc_id in current_area.npcs:
                npc = world.npc_instances.get(npc_id)
                if npc is None:
                    continue
                if npc.enemy:
                    continue
                if get_def_id(npc_id) in self.EXCLUDED_NPC_BASE_IDS:
                    continue
                friendly_npcs.append(npc)

            if not friendly_npcs:
                continue

            # For each active combat, a friendly NPC may shout a warning
            for npc_id, combat_state in active_combats.items():
                if combat_state.get("area_id") != agent_area_id:
                    continue

                enemy_npc = world.npc_instances.get(npc_id)
                if enemy_npc is None or enemy_npc.hp <= 0:
                    continue

                rhythm = enemy_npc.combat_pattern
                if not rhythm:
                    continue

                # rhythm_index already points to the NEXT action after
                # CombatRhythmStepRule incremented it
                rhythm_index = combat_state.get("rhythm_index", 0)
                next_action = rhythm[rhythm_index % len(rhythm)]

                # Each friendly NPC has an independent chance to warn
                for ally_npc in friendly_npcs:
                    if env.rng.random() > self.WARNING_PROBABILITY:
                        continue

                    templates = self.WARNING_TEMPLATES.get(next_action)
                    if not templates:
                        continue

                    shout = env.rng.choice(templates).format(
                        ally=ally_npc.name,
                        enemy=enemy_npc.name,
                    )

                    res.add_feedback(agent.id, f"{shout}\n")
                    res.events.append(Event(
                        type="tactical_warning",
                        agent_id=agent.id,
                        data={
                            "ally_npc_id": ally_npc.id if hasattr(ally_npc, "id") else "",
                            "enemy_npc_id": npc_id,
                            "predicted_action": next_action,
                            "area_id": agent_area_id,
                        },
                    ))

                    # Only one friendly NPC warns per enemy per step
                    break


class ForgeEchoMigrationStepRule(BaseStepRule):
    name = "forge_echo_migration_step"
    description = (
        "Areas where agents have recently crafted items emit resonant forge-echoes that "
        "attract nearby merchant NPCs from adjacent connected areas. Merchants automatically "
        "migrate one step closer toward the crafting site each turn, transforming active "
        "workshops into evolving trade hubs."
    )
    priority = 11  # after territory reclamation and other world-state rules

    # How many steps the forge-echo persists after the last craft in that area
    ECHO_DURATION = 6
    # Merchants only migrate once every N steps to avoid instant teleportation
    MIGRATION_TICK_INTERVAL = 2
    # Maximum number of merchants that can be attracted to a single area
    MAX_MERCHANTS_PER_AREA = 3

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if world is None:
            return

        forge_echo_state = env.curr_agents_state.setdefault("forge_echo_state", {})
        # forge_echo_state structure:
        # {
        #   "active_echoes": { area_id: last_craft_step },
        #   "merchant_origin": { npc_inst_id: original_area_id },
        #   "migration_targets": { npc_inst_id: target_area_id },
        # }
        active_echoes = forge_echo_state.setdefault("active_echoes", {})
        merchant_origin = forge_echo_state.setdefault("merchant_origin", {})
        migration_targets = forge_echo_state.setdefault("migration_targets", {})

        # --- 1. Record new crafting events from this step ---
        for event in res.events:
            if event.type == "object_crafted" and event.agent_id:
                area_id = event.data.get("area_id")
                if area_id:
                    active_echoes[area_id] = env.steps

        # --- 2. Expire old echoes ---
        expired = [
            area_id for area_id, last_step in list(active_echoes.items())
            if env.steps - last_step > self.ECHO_DURATION
        ]
        for area_id in expired:
            del active_echoes[area_id]

        # --- 3. Only migrate on tick intervals ---
        if env.steps < 0 or env.steps % self.MIGRATION_TICK_INTERVAL != 0:
            return

        # --- 4. For each active echo, attract merchants from neighboring areas ---
        for echo_area_id, last_craft_step in list(active_echoes.items()):
            echo_area = world.area_instances.get(echo_area_id)
            if echo_area is None:
                continue

            # Count merchants already in the echo area
            merchants_in_echo = 0
            for npc_id in echo_area.npcs:
                npc = world.npc_instances.get(npc_id)
                if npc and not npc.enemy and getattr(npc, "role", None) == "merchant":
                    merchants_in_echo += 1

            if merchants_in_echo >= self.MAX_MERCHANTS_PER_AREA:
                continue

            # Look for merchants in neighboring areas
            for neighbor_id in list(echo_area.neighbors.keys()):
                if merchants_in_echo >= self.MAX_MERCHANTS_PER_AREA:
                    break

                # Don't pull merchants through locked paths
                path = echo_area.neighbors.get(neighbor_id)
                if path and path.locked:
                    continue

                neighbor_area = world.area_instances.get(neighbor_id)
                if neighbor_area is None:
                    continue

                for npc_id in list(neighbor_area.npcs):
                    if merchants_in_echo >= self.MAX_MERCHANTS_PER_AREA:
                        break

                    npc = world.npc_instances.get(npc_id)
                    if npc is None:
                        continue
                    if npc.enemy:
                        continue
                    if getattr(npc, "role", None) != "merchant":
                        continue
                    # Don't move quest NPCs
                    if getattr(npc, "quest", False):
                        continue
                    # Don't move guide NPCs
                    base_npc_id = get_def_id(npc_id)
                    if base_npc_id in ("npc_quest_wayfarer_guide", "npc_side_quest_guide"):
                        continue

                    # Skip if this merchant is already being targeted to move elsewhere
                    if npc_id in migration_targets and migration_targets[npc_id] != echo_area_id:
                        continue

                    # Record the merchant's original area if not already tracked
                    if npc_id not in merchant_origin:
                        merchant_origin[npc_id] = neighbor_id

                    # Move the merchant from neighbor to echo area
                    if npc_id in neighbor_area.npcs:
                        neighbor_area.npcs.remove(npc_id)
                    if npc_id not in echo_area.npcs:
                        echo_area.npcs.append(npc_id)

                    migration_targets[npc_id] = echo_area_id
                    merchants_in_echo += 1

                    # Update npc_name_to_id in case it matters for lookups
                    npc_name = getattr(npc, "name", None)
                    if npc_name:
                        world.auxiliary.setdefault("npc_name_to_id", {})[npc_name] = npc_id

                    # Notify agents in the echo area
                    echo_area_name = echo_area.name
                    neighbor_area_name = neighbor_area.name
                    for agent in env.agents:
                        agent_area = env.curr_agents_state["area"].get(agent.id)
                        if agent_area == echo_area_id:
                            res.add_feedback(
                                agent.id,
                                f"The resonant echoes of the forge draw {npc.name} from "
                                f"the nearby {neighbor_area_name}. "
                                f"A new merchant has arrived, attracted by the sound of "
                                f"industry!\n"
                            )
                        elif agent_area == neighbor_id:
                            res.add_feedback(
                                agent.id,
                                f"{npc.name} packs up their wares and heads toward "
                                f"{echo_area_name}, drawn by the echoes of a working forge.\n"
                            )

                    res.events.append(Event(
                        type="forge_echo_merchant_migrated",
                        data={
                            "npc_id": npc_id,
                            "npc_name": npc.name,
                            "from_area": neighbor_id,
                            "to_area": echo_area_id,
                            "echo_step": last_craft_step,
                        },
                    ))

                    res.events.append(Event(
                        type="dep_tracker_hint",
                        data={
                            "hint": f"forge_echo: {npc.name} migrated to {echo_area_name}",
                        },
                    ))

        # --- 5. Return merchants to origin when echoes expire ---
        merchants_to_return = []
        for npc_id, target_area_id in list(migration_targets.items()):
            # If the echo for the target area has expired, send merchant back
            if target_area_id not in active_echoes:
                origin_area_id = merchant_origin.get(npc_id)
                if origin_area_id is None:
                    # Can't return, just clean up tracking
                    merchants_to_return.append(npc_id)
                    continue

                npc = world.npc_instances.get(npc_id)
                if npc is None:
                    merchants_to_return.append(npc_id)
                    continue

                # Find where the merchant currently is
                current_area_id = None
                for aid, area in world.area_instances.items():
                    if npc_id in area.npcs:
                        current_area_id = aid
                        break

                if current_area_id is None:
                    merchants_to_return.append(npc_id)
                    continue

                if current_area_id == origin_area_id:
                    # Already home
                    merchants_to_return.append(npc_id)
                    continue

                # Move merchant one step toward origin using BFS shortest path
                next_step = self._next_step_toward(
                    world, current_area_id, origin_area_id
                )
                if next_step is None:
                    # Can't find path, just teleport home
                    next_step = origin_area_id

                current_area = world.area_instances.get(current_area_id)
                next_area = world.area_instances.get(next_step)
                if current_area and next_area:
                    if npc_id in current_area.npcs:
                        current_area.npcs.remove(npc_id)
                    if npc_id not in next_area.npcs:
                        next_area.npcs.append(npc_id)

                    # Notify agents
                    for agent in env.agents:
                        agent_area = env.curr_agents_state["area"].get(agent.id)
                        if agent_area == current_area_id:
                            res.add_feedback(
                                agent.id,
                                f"The forge-echoes have faded. {npc.name} packs up and "
                                f"begins the journey back to {next_area.name}.\n"
                            )
                        elif agent_area == next_step:
                            res.add_feedback(
                                agent.id,
                                f"{npc.name} arrives from {current_area.name}, returning "
                                f"to their usual post.\n"
                            )

                    if next_step == origin_area_id:
                        merchants_to_return.append(npc_id)
                    # else: still in transit, keep tracking

        for npc_id in merchants_to_return:
            if npc_id in migration_targets:
                del migration_targets[npc_id]
            if npc_id in merchant_origin:
                del merchant_origin[npc_id]

    @staticmethod
    def _next_step_toward(world, from_area_id: str, to_area_id: str) -> Optional[str]:
        """BFS to find the next area on the shortest unlocked path from -> to."""
        from collections import deque

        if from_area_id == to_area_id:
            return None

        visited = {from_area_id}
        # queue entries: (current_area_id, first_step_area_id)
        queue = deque()
        from_area = world.area_instances.get(from_area_id)
        if from_area is None:
            return None

        for neighbor_id, path in from_area.neighbors.items():
            if path.locked:
                continue
            if neighbor_id not in world.area_instances:
                continue
            if neighbor_id == to_area_id:
                return neighbor_id
            visited.add(neighbor_id)
            queue.append((neighbor_id, neighbor_id))

        while queue:
            current_id, first_step = queue.popleft()
            current_area = world.area_instances.get(current_id)
            if current_area is None:
                continue
            for neighbor_id, path in current_area.neighbors.items():
                if path.locked:
                    continue
                if neighbor_id in visited:
                    continue
                if neighbor_id not in world.area_instances:
                    continue
                if neighbor_id == to_area_id:
                    return first_step
                visited.add(neighbor_id)
                queue.append((neighbor_id, first_step))

        return None


class WarMachineCannibalizeStepRule(BaseStepRule):
    name = "war_machine_cannibalize_step"
    description = (
        "Enemy NPCs currently engaged in active combat gradually repair themselves by "
        "cannibalizing objects lying on the ground in their area. Each turn, a war machine "
        "consumes one available object to restore HP proportional to the object's coin value, "
        "forcing players to clear loose items from battlefields or finish fights quickly."
    )
    priority = 3  # after combat rhythm (2), before tactical warning (4)

    # Fraction of the object's value converted to HP restoration
    HP_RESTORE_FRACTION = 0.5
    # Minimum HP restored per consumed object (if value > 0)
    MIN_HP_RESTORE = 2
    # Object categories immune to cannibalization
    IMMUNE_CATEGORIES = {"station", "currency", "container"}
    # Object usages immune to cannibalization
    IMMUNE_USAGES = {"writable", "unlock"}

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        # Collect all NPC IDs currently in active combat across all agents,
        # along with the area they are fighting in.
        npc_combat_areas: Dict[str, str] = {}  # npc_id -> area_id
        for agent in env.agents:
            active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
            for npc_id, combat_state in active_combats.items():
                if npc_id not in npc_combat_areas:
                    npc_combat_areas[npc_id] = combat_state.get("area_id")

        if not npc_combat_areas:
            return

        # Process each NPC in combat
        for npc_id, area_id in npc_combat_areas.items():
            if not area_id:
                continue

            npc = world.npc_instances.get(npc_id)
            if npc is None or npc.hp <= 0:
                continue
            if not npc.enemy:
                continue

            area = world.area_instances.get(area_id)
            if area is None:
                continue

            # Check if NPC is already at full base HP (use slope formula to estimate max)
            # We don't have a stored max_hp for NPCs, so we allow healing up to
            # the original HP at their level. We'll track the original HP in state.
            npc_base_id = get_def_id(npc_id)
            npc_proto = world.npcs.get(npc_base_id)
            if npc_proto is None:
                continue
            npc_max_hp = npc_proto.hp + npc_proto.slope_hp * (npc.level - 1)
            if npc.hp >= npc_max_hp:
                continue

            # Find a vulnerable object on the ground to consume
            candidate_objs = []
            for obj_id, count in list(area.objects.items()):
                if count <= 0:
                    continue
                base_obj_id = get_def_id(obj_id)
                obj_def = world.objects.get(base_obj_id)
                if obj_def is None:
                    continue
                if obj_def.category in self.IMMUNE_CATEGORIES:
                    continue
                if obj_def.usage in self.IMMUNE_USAGES:
                    continue
                # Skip container and writable instances
                if obj_id in world.container_instances or obj_id in world.writable_instances:
                    continue
                # Must have a positive value to provide HP
                if obj_def.value is None or int(obj_def.value) <= 0:
                    continue
                candidate_objs.append((obj_id, base_obj_id, obj_def))

            if not candidate_objs:
                continue

            # Pick the most valuable object (war machines are smart scavengers)
            candidate_objs.sort(key=lambda x: int(x[2].value), reverse=True)
            chosen_obj_id, chosen_base_id, chosen_def = candidate_objs[0]

            # Consume one unit of the chosen object
            area.objects[chosen_obj_id] -= 1
            if area.objects[chosen_obj_id] <= 0:
                del area.objects[chosen_obj_id]

            res.track_consume(
                "env", chosen_obj_id, 1,
                src=res.tloc("area", area_id),
            )

            # Restore HP
            hp_restore = max(self.MIN_HP_RESTORE, int(int(chosen_def.value) * self.HP_RESTORE_FRACTION))
            old_hp = npc.hp
            npc.hp = min(npc_max_hp, npc.hp + hp_restore)
            actual_restore = npc.hp - old_hp

            # Notify all agents in the same area
            for agent in env.agents:
                if env.curr_agents_state["area"].get(agent.id) == area_id:
                    res.add_feedback(
                        agent.id,
                        f"{npc.name} cannibalizes a {chosen_def.name} from the ground, "
                        f"welding it into its damaged hull and restoring {actual_restore} HP "
                        f"({npc.hp}/{npc_max_hp} HP). "
                        f"Clear the battlefield to deny war machines repair material!\n"
                    )

            res.events.append(Event(
                type="war_machine_cannibalize",
                data={
                    "npc_id": npc_id,
                    "area_id": area_id,
                    "consumed_obj_id": chosen_obj_id,
                    "consumed_obj_name": chosen_def.name,
                    "hp_restored": actual_restore,
                    "npc_hp": npc.hp,
                    "npc_max_hp": npc_max_hp,
                },
            ))

            res.events.append(Event(
                type="dep_tracker_hint",
                data={
                    "hint": f"cannibalize: {npc.name} consumed {chosen_def.name} for {actual_restore} HP",
                },
            ))


class DevilForgedCorruptionStepRule(BaseStepRule):
    name = "devil_forged_corruption_step"
    description = (
        "Agents carrying more than three devil-forged objects simultaneously suffer "
        "creeping corruption that drains HP each step, proportional to the number of "
        "such items held. The dark machines' malevolent influence overwhelms mortal "
        "bearers who hoard too much accursed technology."
    )
    priority = 12  # after most combat / world-state rules

    # How many devil-forged items an agent can safely carry before corruption kicks in
    SAFE_THRESHOLD = 3
    # HP drained per devil-forged item beyond the threshold each step
    HP_DRAIN_PER_EXCESS = 2

    # Base object IDs considered devil-forged (raw war-machine materials and their derivatives)
    DEVIL_FORGED_BASE_IDS = {
        # raw drops from war machines
        "obj_scrap_iron",
        "obj_devil_oil",
        "obj_hellcoal",
        "obj_cog",
        "obj_fiend_plate",
        "obj_demon_core_shard",
        "obj_brimite_ore",
        "obj_war_engine_heart",
        # processed devil-forged materials
        "obj_iron_nugget",
        "obj_iron_bar",
        "obj_iron_sheet",
        "obj_iron_scale",
        "obj_scale_bundle",
        "obj_hellcoal_concentrate",
        "obj_fiend_strip",
        "obj_fiend_rivet",
        "obj_brimite_nugget",
        "obj_brimite_ingot",
        "obj_brimite_blade_blank",
        "obj_brimite_blade",
        "obj_fiend_alloy_ingot",
        "obj_fiend_alloy_sheet",
        "obj_fiend_alloy_scale",
        "obj_fiend_scale_bundle",
        "obj_oil_flask",
        "obj_cog_assembly",
        "obj_gearwork_frame",
    }

    def _count_devil_forged(self, world, agent) -> int:
        """Count total devil-forged items across hands, equipped, inventory, and held containers."""
        total = 0

        for oid, cnt in agent.items_in_hands.items():
            if get_def_id(oid) in self.DEVIL_FORGED_BASE_IDS:
                total += int(cnt)

        for oid, cnt in agent.equipped_items_in_limb.items():
            if get_def_id(oid) in self.DEVIL_FORGED_BASE_IDS:
                total += int(cnt)

        inv = getattr(agent, "inventory", None)
        if inv and getattr(inv, "container", None):
            for oid, cnt in inv.items.items():
                if get_def_id(oid) in self.DEVIL_FORGED_BASE_IDS:
                    total += int(cnt)

        # containers held in hand
        for oid in list(agent.items_in_hands.keys()):
            if oid in world.container_instances:
                for coid, cnt in world.container_instances[oid].inventory.items():
                    if get_def_id(coid) in self.DEVIL_FORGED_BASE_IDS:
                        total += int(cnt)

        return total

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        for agent in env.agents:
            if agent.hp <= 0:
                continue

            devil_count = self._count_devil_forged(world, agent)
            excess = devil_count - self.SAFE_THRESHOLD

            if excess <= 0:
                continue

            damage = excess * self.HP_DRAIN_PER_EXCESS
            agent.hp -= damage

            res.add_feedback(
                agent.id,
                f"The malevolent energy of {devil_count} devil-forged objects seeps into "
                f"{env.person_verbalized['possessive_adjective']} flesh, draining {damage} HP. "
                f"Carrying more than {self.SAFE_THRESHOLD} accursed items invites corruption! "
                f"({agent.hp} HP remaining)\n"
            )

            res.events.append(Event(
                type="devil_forged_corruption",
                agent_id=agent.id,
                data={
                    "devil_forged_count": devil_count,
                    "excess": excess,
                    "damage": damage,
                    "remaining_hp": max(0, agent.hp),
                    "area_id": env.curr_agents_state["area"].get(agent.id),
                },
            ))

            if agent.hp <= 0:
                agent.hp = 0
                res.add_feedback(
                    agent.id,
                    f"The creeping corruption of the devil-forged machines has overwhelmed "
                    f"{env.person_verbalized['object_pronoun']}!\n"
                )
                res.events.append(Event(
                    type="agent_defeated_by_corruption",
                    agent_id=agent.id,
                    data={
                        "devil_forged_count": devil_count,
                        "area_id": env.curr_agents_state["area"].get(agent.id),
                    },
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
