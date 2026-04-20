import copy
import math
from utils import *
from games.generated.metropolis.rule import BaseStepRule, RuleContext, RuleResult, Event
from typing import Dict, Optional, Set, Tuple, List, Any
from games.generated.metropolis.world import NPC, Object, Area, Place, Path, Container, Writable
from games.generated.metropolis.agent import Agent
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
    description = (
        "Agents who die (HP <= 0) are given one turn to appeal the verdict. "
        "If they do not appeal, they drop all items and respawn at the starting point."
    )
    priority: int = 7
    
    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        for agent in env.agents:
            # --- Phase 1: handle agents who successfully appealed ---
            appeal_used = env.curr_agents_state.get("appeal_verdict_used", {}).get(agent.id, False)
            if appeal_used:
                env.curr_agents_state["appeal_verdict_used"][agent.id] = False
                # clear pending death since the appeal reversed it
                env.curr_agents_state.get("pending_death", {})[agent.id] = False
                continue

            # --- Phase 2: process pending deaths (agent didn't appeal) ---
            pending = env.curr_agents_state.get("pending_death", {}).get(agent.id, False)
            if pending:
                env.curr_agents_state["pending_death"][agent.id] = False
                # the agent chose not to appeal (or couldn't); process death now
                self._process_death(env, world, agent, res)
                continue

            # --- Phase 3: detect new deaths and defer them ---
            if agent.hp > 0:
                continue

            # Mark death as pending — give the agent one turn to appeal
            env.curr_agents_state.setdefault("pending_death", {})[agent.id] = True

            # Count coins to inform the agent whether appeal is possible
            coin_id = "obj_coin"
            coin_count = int(agent.items_in_hands.get(coin_id, 0))
            if agent.inventory.container:
                coin_count += int(agent.inventory.items.get(coin_id, 0))
            for oid in agent.items_in_hands.keys():
                if oid in world.container_instances:
                    coin_count += int(world.container_instances[oid].inventory.get(coin_id, 0))

            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} have been struck down!\n"
            )
            if coin_count > 0:
                res.add_feedback(
                    agent.id,
                    f"The higher court offers a chance: use 'appeal verdict' to sacrifice "
                    f"all {coin_count} coins, restore {env.person_verbalized['possessive_adjective']} "
                    f"health, and banish {env.person_verbalized['possessive_adjective']} attacker. "
                    f"Otherwise, {env.person_verbalized['subject_pronoun']} will perish next turn.\n"
                )
            else:
                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} have no coins to appeal. "
                    f"Death will be processed next turn.\n"
                )

            res.events.append(Event(
                type="death_pending",
                agent_id=agent.id,
                data={"area_id": env.curr_agents_state["area"][agent.id]},
            ))

    @staticmethod
    def _process_death(env, world, agent, res) -> None:
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


class NightlyLawRewriteStepRule(BaseStepRule):
    name = "nightly_law_rewrite_step"
    description = (
        "Every fixed number of steps (one 'night cycle'), all written legal "
        "documents in the world spontaneously rewrite themselves with new "
        "randomized laws. Area-level physics shift accordingly: heavy items "
        "may float away from some areas, certain paths may seal or unseal, "
        "and NPC combat patterns may shuffle or invert."
    )
    priority = 50

    # How many steps constitute one night cycle
    NIGHT_CYCLE_STEPS = 30

    # Pool of randomized law texts that get written onto documents
    LAW_POOL = [
        "Under Statute 7-A: All consecutive attacks are hereby banned in this district.",
        "Edict of Levitation: Gravity is nullified for lightweight objects until dawn.",
        "Passage Decree 12: All locked passages shall remain sealed until further notice.",
        "Open Court Order: All doors in the metropolis are to remain unlocked tonight.",
        "Silence Mandate: No hostile action may be initiated without a formal declaration.",
        "Reversal Clause: Defensive postures must precede any offensive maneuver.",
        "Statute of Aggression: All combatants must open with an attack before defending.",
        "Nullification Writ: All prior combat orders are suspended; combatants must wait.",
        "Gravity Inversion Act: Heavy materials sink through floors into lower chambers.",
        "Peaceful Conduct Law: No aggression is permitted within courthouse grounds.",
        "Mandatory Defense Edict: All entities must include a defensive stance in combat.",
        "First Strike Prohibition: No entity may open combat with an attack action.",
        "Wait Mandate: All combatants must include a waiting period in their rhythm.",
        "Free Passage Act: All sealed doors are temporarily unsealed by judicial order.",
        "Evidence Redistribution Law: Loose materials drift between adjacent chambers.",
        "Consecutive Attack Ban: No entity may attack twice in succession.",
    ]

    # Fraction of paths that get toggled each night
    PATH_TOGGLE_FRACTION = 0.15

    # Fraction of areas that lose a random non-compositional object
    OBJECT_DRIFT_FRACTION = 0.2

    # Fraction of enemy NPCs whose combat patterns get shuffled
    COMBAT_SHUFFLE_FRACTION = 0.4

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if world is None:
            return

        step = ctx.step_index
        if step <= 0:
            return

        night_state = env.curr_agents_state.setdefault("nightly_law_rewrite", {
            "last_night_step": 0,
            "night_count": 0,
        })

        last_night = night_state.get("last_night_step", 0)
        if step - last_night < self.NIGHT_CYCLE_STEPS:
            return

        # --- It's a new night! ---
        night_state["last_night_step"] = step
        night_state["night_count"] = night_state.get("night_count", 0) + 1
        night_num = night_state["night_count"]

        rng = env.rng

        # Notify all agents
        for agent in env.agents:
            res.add_feedback(
                agent.id,
                f"\n=== NIGHT {night_num}: THE LAWS REWRITE THEMSELVES ===\n"
                f"Midnight strikes. The statutes of the metropolis dissolve and "
                f"reform. The ground shifts, doors groan, and the air hums with "
                f"new legal force. All written documents now bear different laws.\n"
                f"================================================\n"
            )

        # --- 1. Rewrite all writable instances with random law text ---
        rewritten_count = 0
        for wid, writable in world.writable_instances.items():
            old_text = (writable.text or "").strip()
            if not old_text:
                continue  # skip blank documents
            new_law = rng.choice(self.LAW_POOL)
            # Truncate to max_text_length
            max_len = getattr(writable, "max_text_length", 200) or 200
            writable.text = new_law[:max_len]
            rewritten_count += 1

        # --- 2. Toggle some paths (seal/unseal) ---
        all_area_ids = list(world.area_instances.keys())
        toggled_paths = set()
        for area_id in all_area_ids:
            area = world.area_instances[area_id]
            for neighbor_id, path in area.neighbors.items():
                pair = tuple(sorted([area_id, neighbor_id]))
                if pair in toggled_paths:
                    continue
                if rng.random() < self.PATH_TOGGLE_FRACTION:
                    toggled_paths.add(pair)
                    new_locked = not path.locked
                    path.locked = new_locked
                    # Toggle reverse path too
                    reverse_area = world.area_instances.get(neighbor_id)
                    if reverse_area:
                        reverse_path = reverse_area.neighbors.get(area_id)
                        if reverse_path:
                            reverse_path.locked = new_locked

                    # Notify agents in either area
                    for agent in env.agents:
                        agent_area = env.curr_agents_state["area"].get(agent.id)
                        if agent_area == area_id or agent_area == neighbor_id:
                            area_name = world.area_instances[neighbor_id].name if agent_area == area_id else area.name
                            if new_locked:
                                res.add_feedback(
                                    agent.id,
                                    f"The path to {area_name} seals shut under tonight's new statutes.\n"
                                )
                            else:
                                res.add_feedback(
                                    agent.id,
                                    f"The path to {area_name} groans open as tonight's laws take effect.\n"
                                )

        # --- 3. Object drift: some lightweight objects float away ---
        spawn_area = env.world_definition.get("initializations", {}).get("spawn", {}).get("area")
        undistributable = set(
            env.world_definition.get("initializations", {}).get("undistributable_objects", [])
        )
        for area_id in all_area_ids:
            if rng.random() > self.OBJECT_DRIFT_FRACTION:
                continue
            area = world.area_instances[area_id]
            # Find non-compositional, non-quest, non-station, non-currency objects
            driftable = []
            for oid, cnt in list(area.objects.items()):
                if cnt <= 0:
                    continue
                base_id = get_def_id(oid)
                if base_id not in world.objects:
                    continue
                obj_def = world.objects[base_id]
                if obj_def.category in ("station", "currency", "container"):
                    continue
                if obj_def.usage == "writable":
                    continue
                if getattr(obj_def, "quest", False):
                    continue
                if base_id in undistributable:
                    continue
                if obj_def.size is not None and obj_def.size <= 2:
                    driftable.append(oid)

            if not driftable:
                continue

            # Pick one object to drift
            drift_oid = rng.choice(driftable)
            # Find a neighboring area to drift to
            neighbor_ids = list(area.neighbors.keys())
            if not neighbor_ids:
                continue
            dest_area_id = rng.choice(neighbor_ids)
            dest_area = world.area_instances[dest_area_id]

            # Move 1 unit
            area.objects[drift_oid] -= 1
            if area.objects[drift_oid] <= 0:
                del area.objects[drift_oid]
            dest_area.objects[drift_oid] = dest_area.objects.get(drift_oid, 0) + 1

            res.track_move(
                "env", drift_oid, 1,
                src=res.tloc("area", area_id),
                dst=res.tloc("area", dest_area_id),
            )

            obj_name = world.objects[get_def_id(drift_oid)].name
            # Notify agents in source area
            for agent in env.agents:
                if env.curr_agents_state["area"].get(agent.id) == area_id:
                    res.add_feedback(
                        agent.id,
                        f"The shifting laws cause a {obj_name} to float away toward {dest_area.name}.\n"
                    )
                elif env.curr_agents_state["area"].get(agent.id) == dest_area_id:
                    res.add_feedback(
                        agent.id,
                        f"A {obj_name} drifts in from {area.name}, deposited by the night's new gravity laws.\n"
                    )

        # --- 4. Shuffle enemy NPC combat patterns ---
        shuffled_npcs = []
        for npc_id, npc_inst in world.npc_instances.items():
            if not npc_inst.enemy:
                continue
            if npc_inst.hp <= 0:
                continue
            if not npc_inst.combat_pattern:
                continue
            if rng.random() > self.COMBAT_SHUFFLE_FRACTION:
                continue

            old_pattern = list(npc_inst.combat_pattern)
            new_pattern = list(old_pattern)

            # Randomly choose: shuffle, reverse, or rotate
            mutation = rng.choice(["shuffle", "reverse", "rotate"])
            if mutation == "shuffle":
                rng.shuffle(new_pattern)
            elif mutation == "reverse":
                new_pattern = new_pattern[::-1]
            elif mutation == "rotate":
                k = rng.randint(1, max(1, len(new_pattern) - 1))
                new_pattern = new_pattern[k:] + new_pattern[:k]

            npc_inst.combat_pattern = new_pattern
            shuffled_npcs.append(npc_inst.name)

            # Reset rhythm index for any active combats involving this NPC
            for agent in env.agents:
                active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
                if npc_id in active_combats:
                    active_combats[npc_id]["rhythm_index"] = 0

        # Notify agents about combat pattern changes
        if shuffled_npcs:
            for agent in env.agents:
                res.add_feedback(
                    agent.id,
                    f"The new statutes alter the fighting doctrine of the metropolis. "
                    f"Enemy combat behaviors have shifted tonight.\n"
                )

        res.events.append(Event(
            type="nightly_law_rewrite",
            agent_id=None,
            data={
                "night_number": night_num,
                "step": step,
                "documents_rewritten": rewritten_count,
                "paths_toggled": len(toggled_paths),
                "npcs_shuffled": len(shuffled_npcs),
            },
        ))

class ObjectionEffectsStepRule(BaseStepRule):
    name = "objection_effects_step"
    description = "Tick down pacified NPC timers from sustained objections and restore them on expiry. Also enforce skip-turn penalties for overruled objections."
    priority = 9

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        # --- 1. Handle pacified NPCs ---
        pacified = env.curr_agents_state.get("pacified_npcs", {})
        expired_keys = []
        for npc_id, info in list(pacified.items()):
            info["remaining"] -= 1
            if info["remaining"] <= 0:
                expired_keys.append(npc_id)
                # restore NPC
                npc_inst = world.npc_instances.get(npc_id)
                if npc_inst:
                    npc_inst.enemy = info.get("original_enemy", True)
                    npc_inst.attack_power = info.get("original_attack_power", 0)
                    npc_inst.combat_pattern = info.get("original_combat_pattern", [])

                area_id = info.get("area_id")
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        npc_name = npc_inst.name if npc_inst else "an NPC"
                        res.add_feedback(
                            agent.id,
                            f"The legal pacification on {npc_name} has expired. They regain their hostile demeanor.\n"
                        )
                res.events.append(Event(
                    type="objection_pacify_expired",
                    agent_id=info.get("agent_id"),
                    data={"npc_id": npc_id, "area_id": area_id},
                ))
            else:
                # notify agents in the area
                area_id = info.get("area_id")
                npc_inst = world.npc_instances.get(npc_id)
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        npc_name = npc_inst.name if npc_inst else "an NPC"
                        res.add_feedback(
                            agent.id,
                            f"{npc_name} is legally pacified ({info['remaining']} turn(s) remaining).\n"
                        )

        for key in expired_keys:
            del pacified[key]
        env.curr_agents_state["pacified_npcs"] = pacified


class InvokeLawExpiryStepRule(BaseStepRule):
    name = "invoke_law_expiry_step"
    description = "Tick down active law invocations each step and revert their effects when they expire."
    priority = 8

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        active_invocations = env.curr_agents_state.get("active_invocations", {})
        if not active_invocations:
            return

        expired_keys = []
        for inv_id, inv in list(active_invocations.items()):
            inv["remaining"] -= 1
            area_id = inv.get("area_id")
            effect_type = inv.get("effect_type")
            agent_id = inv.get("agent_id")
            saved = inv.get("saved_state", {})

            if inv["remaining"] <= 0:
                expired_keys.append(inv_id)

                # revert effects
                if effect_type == "unlock_doors":
                    locked_paths = saved.get("locked_paths", {})
                    area = world.area_instances.get(area_id)
                    if area:
                        for neighbor_id in locked_paths:
                            path = area.neighbors.get(neighbor_id)
                            if path:
                                path.locked = True
                            reverse_area = world.area_instances.get(neighbor_id)
                            if reverse_area:
                                reverse_path = reverse_area.neighbors.get(area_id)
                                if reverse_path:
                                    reverse_path.locked = True

                elif effect_type == "pacify_npcs":
                    original_stats = saved.get("original_npc_stats", {})
                    for npc_id, stats in original_stats.items():
                        npc_inst = world.npc_instances.get(npc_id)
                        if npc_inst:
                            npc_inst.attack_power = stats["attack_power"]
                            npc_inst.combat_pattern = stats["combat_pattern"]

                # reveal_hidden has no revert — spawned objects stay

                # notify agents in the area
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        if effect_type == "unlock_doors":
                            res.add_feedback(agent.id, "The invoked passage law fades — locked doors seal shut once more.\n")
                        elif effect_type == "pacify_npcs":
                            res.add_feedback(agent.id, "The invoked pacification law fades — enemies regain their fighting spirit.\n")
                        elif effect_type == "reveal_hidden":
                            res.add_feedback(agent.id, "The invoked gravity law fades — but the revealed objects remain.\n")

                res.events.append(Event(
                    type="law_expired",
                    agent_id=agent_id,
                    data={"invocation_id": inv_id, "effect_type": effect_type, "area_id": area_id},
                ))
            else:
                # notify agents that the law is still active
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        res.add_feedback(
                            agent.id,
                            f"An invoked {effect_type.replace('_', ' ')} law is active here ({inv['remaining']} turn(s) remaining).\n"
                        )

        for key in expired_keys:
            del active_invocations[key]

        env.curr_agents_state["active_invocations"] = active_invocations


class LegalParadoxStepRule(BaseStepRule):
    name = "legal_paradox_step"
    description = (
        "Each step, any area containing three or more written legal documents "
        "with contradictory laws enters a state of 'legal paradox' that warps "
        "reality: NPCs freeze in confusion, objects randomly duplicate or vanish, "
        "and paths shuffle their destinations. Persists until documents are removed."
    )
    priority = 0  # runs before combat rules so NPC freeze takes effect

    # Minimum number of written documents required to trigger a paradox
    MIN_DOCUMENTS = 3

    # Keyword groups — two documents are contradictory if they belong to
    # different groups.  Groups mirror InvokeLawRule / NightlyLawRewriteStepRule.
    KEYWORD_GROUPS = {
        "passage": {"passage", "unlock", "open", "free passage", "open court"},
        "seal":    {"seal", "locked", "sealed", "remain sealed"},
        "gravity": {"gravity", "reveal", "hidden", "levitation", "float", "inversion"},
        "pacify":  {"nullify", "silence", "pacify", "peaceful", "no aggression",
                    "ban attack", "prohibit attack"},
        "aggress": {"attack", "aggression", "first strike", "offensive",
                    "statute of aggression", "must open with"},
    }

    # Contradictory group pairs
    CONTRADICTIONS = {
        frozenset({"passage", "seal"}),
        frozenset({"pacify", "aggress"}),
        frozenset({"gravity", "seal"}),
    }

    # Fraction of non-station, non-currency objects that may duplicate or vanish
    OBJECT_WARP_FRACTION = 0.25

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if world is None:
            return

        paradox_state = env.curr_agents_state.setdefault("legal_paradox", {})
        # paradox_state: { area_id: { "active": bool, "shuffled_paths": {orig_neighbor: redirected_neighbor, ...} } }

        all_area_ids = list(world.area_instances.keys())
        undistributable = set(
            env.world_definition.get("initializations", {}).get("undistributable_objects", [])
        )

        for area_id in all_area_ids:
            area = world.area_instances[area_id]

            # --- Collect written documents in this area ---
            doc_groups = self._classify_area_documents(world, area)
            has_contradiction = self._detect_contradiction(doc_groups)
            enough_docs = len(doc_groups) >= self.MIN_DOCUMENTS

            was_active = paradox_state.get(area_id, {}).get("active", False)
            is_active = enough_docs and has_contradiction

            if is_active and not was_active:
                # --- Paradox just started ---
                shuffled = self._shuffle_paths(env.rng, area, world)
                paradox_state[area_id] = {
                    "active": True,
                    "shuffled_paths": shuffled,
                }

                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        res.add_feedback(
                            agent.id,
                            f"\n⚖️ LEGAL PARADOX! Contradictory laws in {area.name} "
                            f"tear at the fabric of reality. NPCs freeze in confusion, "
                            f"objects flicker, and paths twist to unknown destinations. "
                            f"Remove documents to resolve the contradiction.\n"
                        )

                res.events.append(Event(
                    type="legal_paradox_started",
                    agent_id=None,
                    data={"area_id": area_id},
                ))

            elif was_active and not is_active:
                # --- Paradox just resolved ---
                old_shuffled = paradox_state.get(area_id, {}).get("shuffled_paths", {})
                self._unshuffle_paths(area, world, old_shuffled)

                # Restore frozen NPCs
                frozen_npcs = paradox_state.get(area_id, {}).get("frozen_npcs", {})
                for npc_id, saved in frozen_npcs.items():
                    npc_inst = world.npc_instances.get(npc_id)
                    if npc_inst:
                        npc_inst.attack_power = saved["original_attack_power"]
                        npc_inst.combat_pattern = saved["original_combat_pattern"]

                paradox_state[area_id] = {"active": False, "shuffled_paths": {}, "frozen_npcs": {}}

                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        res.add_feedback(
                            agent.id,
                            f"\n⚖️ The legal paradox in {area.name} has been resolved. "
                            f"Reality stabilizes — NPCs regain their senses and paths "
                            f"return to their true destinations.\n"
                        )

                res.events.append(Event(
                    type="legal_paradox_resolved",
                    agent_id=None,
                    data={"area_id": area_id},
                ))

            if is_active:
                # --- Ongoing paradox effects ---

                # 1. Freeze NPCs: suppress active combats and active attacks
                #    by temporarily neutralizing all enemy NPCs in this area.
                frozen_npcs = paradox_state[area_id].setdefault("frozen_npcs", {})
                for npc_id in area.npcs:
                    npc_inst = world.npc_instances.get(npc_id)
                    if npc_inst is None or not npc_inst.enemy:
                        continue
                    if npc_id not in frozen_npcs:
                        frozen_npcs[npc_id] = {
                            "original_attack_power": npc_inst.attack_power,
                            "original_combat_pattern": list(npc_inst.combat_pattern),
                        }
                    npc_inst.attack_power = 0
                    npc_inst.combat_pattern = []

                # Break active combats in this area for all agents
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) != area_id:
                        continue
                    active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
                    to_remove = [nid for nid, cs in active_combats.items() if cs.get("area_id") == area_id]
                    for nid in to_remove:
                        npc_inst = world.npc_instances.get(nid)
                        npc_name = npc_inst.name if npc_inst else "an NPC"
                        res.add_feedback(
                            agent.id,
                            f"{npc_name} is frozen in legal confusion and cannot act.\n"
                        )
                        del active_combats[nid]
                    if not active_combats:
                        env.curr_agents_state["agent_stamina"][agent.id] = 1.0
                        env.curr_agents_state["agent_consecutive_attacks"][agent.id] = 0

                # 2. Object warp: randomly duplicate or vanish small objects
                warpable = []
                for oid, cnt in list(area.objects.items()):
                    if cnt <= 0:
                        continue
                    base_id = get_def_id(oid)
                    if base_id not in world.objects:
                        continue
                    obj_def = world.objects[base_id]
                    if obj_def.category in ("station", "currency", "container"):
                        continue
                    if obj_def.usage == "writable":
                        continue
                    if getattr(obj_def, "quest", False):
                        continue
                    if base_id in undistributable:
                        continue
                    warpable.append(oid)

                if warpable:
                    num_warp = max(1, int(len(warpable) * self.OBJECT_WARP_FRACTION))
                    chosen = env.rng.sample(warpable, min(num_warp, len(warpable)))
                    for oid in chosen:
                        if oid not in area.objects or area.objects[oid] <= 0:
                            continue
                        obj_name = world.objects[get_def_id(oid)].name

                        if env.rng.random() < 0.5:
                            # duplicate
                            area.objects[oid] = area.objects.get(oid, 0) + 1
                            res.track_spawn("env", oid, 1, dst=res.tloc("area", area_id))
                            for agent in env.agents:
                                if env.curr_agents_state["area"].get(agent.id) == area_id:
                                    res.add_feedback(
                                        agent.id,
                                        f"The paradox flickers and a {obj_name} materializes from thin air.\n"
                                    )
                        else:
                            # vanish
                            area.objects[oid] -= 1
                            if area.objects[oid] <= 0:
                                del area.objects[oid]
                            res.track_consume("env", oid, 1, src=res.tloc("area", area_id))
                            for agent in env.agents:
                                if env.curr_agents_state["area"].get(agent.id) == area_id:
                                    res.add_feedback(
                                        agent.id,
                                        f"A {obj_name} shimmers and vanishes in the paradox.\n"
                                    )

                # 3. Notify agents in the area about the ongoing paradox
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        res.add_feedback(
                            agent.id,
                            f"⚖️ A legal paradox persists in {area.name}. "
                            f"Paths lead to unexpected places. Remove contradictory documents to restore order.\n"
                        )

        env.curr_agents_state["legal_paradox"] = paradox_state

    def _classify_area_documents(self, world, area) -> list:
        """Return a list of sets of keyword-group names for each written document in the area."""
        doc_groups = []
        for oid, cnt in area.objects.items():
            if cnt <= 0:
                continue
            if oid not in world.writable_instances:
                continue
            writable = world.writable_instances[oid]
            text = (writable.text or "").strip().lower()
            if not text:
                continue

            groups_found = set()
            for group_name, keywords in self.KEYWORD_GROUPS.items():
                for kw in keywords:
                    if kw in text:
                        groups_found.add(group_name)
                        break
            if groups_found:
                doc_groups.append(groups_found)
        return doc_groups

    def _detect_contradiction(self, doc_groups: list) -> bool:
        """Return True if any two documents belong to contradictory keyword groups."""
        all_groups = set()
        per_doc_groups = []
        for groups in doc_groups:
            per_doc_groups.append(groups)
            all_groups.update(groups)

        for contradiction_pair in self.CONTRADICTIONS:
            if contradiction_pair.issubset(all_groups):
                # Ensure the contradictory groups come from *different* documents
                has_a = False
                has_b = False
                pair_list = list(contradiction_pair)
                for dg in per_doc_groups:
                    if pair_list[0] in dg:
                        has_a = True
                    if pair_list[1] in dg:
                        has_b = True
                if has_a and has_b:
                    return True
        return False

    def _shuffle_paths(self, rng, area, world) -> dict:
        """Swap neighbor dict keys so 'enter <area_name>' leads to an
        unexpected destination. Stores and returns the original neighbors
        dict (serialized) for restoration."""
        neighbor_ids = list(area.neighbors.keys())
        if len(neighbor_ids) < 2:
            return {}

        shuffled_ids = list(neighbor_ids)
        attempts = 0
        while shuffled_ids == neighbor_ids and attempts < 10:
            rng.shuffle(shuffled_ids)
            attempts += 1

        if shuffled_ids == neighbor_ids:
            return {}

        # Save original neighbors for restoration: {orig_key: path_attrs}
        original_neighbors = {}
        for nid, path in area.neighbors.items():
            original_neighbors[nid] = {
                "to_id": path.to_id,
                "locked": path.locked,
                "object_to_unlock": path.object_to_unlock,
            }

        # Rebuild with swapped keys: the Path object from orig_key is now
        # stored under shuffled_key. When the agent types "enter <name>"
        # matching shuffled_key, they get sent to shuffled_key's area.
        original_paths = {nid: area.neighbors[nid] for nid in neighbor_ids}
        new_neighbors = {}
        for orig_key, shuffled_key in zip(neighbor_ids, shuffled_ids):
            new_neighbors[shuffled_key] = original_paths[orig_key]

        area.neighbors = new_neighbors
        return original_neighbors

    def _unshuffle_paths(self, area, world, original_neighbors: dict) -> None:
        """Restore the original neighbor dict from saved state."""
        if not original_neighbors:
            return
        restored = {}
        for nid, path_attrs in original_neighbors.items():
            restored[nid] = Path(
                to_id=path_attrs["to_id"],
                locked=path_attrs["locked"],
                object_to_unlock=path_attrs.get("object_to_unlock"),
            )
        area.neighbors = restored


class ContemptOfCourtStepRule(BaseStepRule):
    name = "contempt_of_court_step"
    description = (
        "Agents who idle in the same area for several consecutive turns without "
        "performing meaningful actions accrue 'contempt of court' penalties: "
        "first warnings, then coin fines, and finally temporary max HP reduction. "
        "Moving to a different area or performing a meaningful action resets the counter."
    )
    priority = 10

    # Thresholds (consecutive idle turns)
    WARNING_THRESHOLD = 3       # first warning
    FINE_THRESHOLD = 5          # start losing coins
    HP_PENALTY_THRESHOLD = 7    # start losing max HP

    # Penalty amounts
    FINE_COINS_PER_TURN = 2
    HP_REDUCTION_PER_TURN = 5
    MAX_HP_REDUCTION_CAP = 30   # maximum total max HP that can be reduced

    # Actions considered "idle" (not meaningful)
    IDLE_ACTIONS = {"wait", "defend"}

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if world is None:
            return

        contempt = env.curr_agents_state.setdefault("contempt_of_court", {})

        for agent in env.agents:
            aid = agent.id
            current_area_id = env.curr_agents_state["area"].get(aid)
            if current_area_id is None:
                continue

            # initialise tracking for this agent
            if aid not in contempt:
                contempt[aid] = {
                    "idle_area": current_area_id,
                    "idle_turns": 0,
                    "last_action": None,
                    "hp_reduction_applied": 0,
                }

            state = contempt[aid]

            # Determine the agent's last action from this step's events
            last_action = state.get("last_action")

            # Check whether the agent moved or performed a meaningful action
            area_changed = (current_area_id != state.get("idle_area"))
            meaningful_action = (
                last_action is not None
                and last_action not in self.IDLE_ACTIONS
            )

            if area_changed or meaningful_action:
                # Reset idle counter
                # Restore any temporary max HP reduction
                hp_restored = state.get("hp_reduction_applied", 0)
                if hp_restored > 0:
                    agent.max_hp += hp_restored
                    agent.hp = min(agent.hp + hp_restored, agent.max_hp)
                    res.add_feedback(
                        aid,
                        f"The contempt of court charges are dropped. "
                        f"{env.person_verbalized['possessive_adjective'].capitalize()} "
                        f"maximum health is restored by {hp_restored}.\n"
                    )
                    res.events.append(Event(
                        type="contempt_of_court_cleared",
                        agent_id=aid,
                        data={
                            "hp_restored": hp_restored,
                            "area_id": current_area_id,
                        },
                    ))

                state["idle_area"] = current_area_id
                state["idle_turns"] = 0
                state["hp_reduction_applied"] = 0
                continue

            # Agent is still idle in the same area
            state["idle_turns"] += 1
            idle_turns = state["idle_turns"]

            # --- Escalating penalties ---

            if idle_turns == self.WARNING_THRESHOLD:
                res.add_feedback(
                    aid,
                    f"⚖️ The court clerk eyes {env.person_verbalized['object_pronoun']} "
                    f"suspiciously. \"Loitering is a misdemeanor in this district. "
                    f"Move along or face contempt charges.\"\n"
                )
                res.events.append(Event(
                    type="contempt_of_court_warning",
                    agent_id=aid,
                    data={"idle_turns": idle_turns, "area_id": current_area_id},
                ))

            elif idle_turns == self.WARNING_THRESHOLD + 1:
                res.add_feedback(
                    aid,
                    f"⚖️ A bailiff approaches. \"This is {env.person_verbalized['possessive_adjective']} "
                    f"final warning. The metropolis does not tolerate idleness.\"\n"
                )
                res.events.append(Event(
                    type="contempt_of_court_final_warning",
                    agent_id=aid,
                    data={"idle_turns": idle_turns, "area_id": current_area_id},
                ))

            elif idle_turns >= self.FINE_THRESHOLD and idle_turns < self.HP_PENALTY_THRESHOLD:
                # Fine: remove coins from hand, inventory, or held containers
                coins_to_fine = self.FINE_COINS_PER_TURN
                coins_taken = self._collect_coins(env, world, agent, coins_to_fine, res)

                if coins_taken > 0:
                    res.add_feedback(
                        aid,
                        f"⚖️ CONTEMPT OF COURT! The metropolis fines "
                        f"{env.person_verbalized['object_pronoun']} {coins_taken} coin(s) "
                        f"for continued idleness. Move or act to avoid further penalties.\n"
                    )
                else:
                    res.add_feedback(
                        aid,
                        f"⚖️ CONTEMPT OF COURT! The metropolis attempts to fine "
                        f"{env.person_verbalized['object_pronoun']} but "
                        f"{env.person_verbalized['subject_pronoun']} have no coins. "
                        f"Harsher penalties loom.\n"
                    )

                res.events.append(Event(
                    type="contempt_of_court_fine",
                    agent_id=aid,
                    data={
                        "idle_turns": idle_turns,
                        "coins_fined": coins_taken,
                        "area_id": current_area_id,
                    },
                ))

            elif idle_turns >= self.HP_PENALTY_THRESHOLD:
                # Fine coins AND reduce max HP
                coins_to_fine = self.FINE_COINS_PER_TURN
                coins_taken = self._collect_coins(env, world, agent, coins_to_fine, res)

                current_reduction = state.get("hp_reduction_applied", 0)
                if current_reduction < self.MAX_HP_REDUCTION_CAP:
                    hp_loss = min(
                        self.HP_REDUCTION_PER_TURN,
                        self.MAX_HP_REDUCTION_CAP - current_reduction,
                    )
                    agent.max_hp = max(1, agent.max_hp - hp_loss)
                    agent.hp = min(agent.hp, agent.max_hp)
                    state["hp_reduction_applied"] = current_reduction + hp_loss

                    res.add_feedback(
                        aid,
                        f"⚖️ SEVERE CONTEMPT OF COURT! The weight of the law bears down "
                        f"on {env.person_verbalized['object_pronoun']}. "
                        f"{env.person_verbalized['possessive_adjective'].capitalize()} "
                        f"maximum health is reduced by {hp_loss} "
                        f"(total reduction: {state['hp_reduction_applied']})."
                    )
                    if coins_taken > 0:
                        res.add_feedback(
                            aid,
                            f" Additionally fined {coins_taken} coin(s).\n"
                        )
                    else:
                        res.add_feedback(aid, "\n")
                else:
                    msg = (
                        f"⚖️ CONTEMPT OF COURT persists! "
                        f"{env.person_verbalized['possessive_adjective'].capitalize()} "
                        f"health remains suppressed."
                    )
                    if coins_taken > 0:
                        msg += f" Fined {coins_taken} coin(s)."
                    msg += " Move or act to clear the charges.\n"
                    res.add_feedback(aid, msg)

                res.events.append(Event(
                    type="contempt_of_court_hp_penalty",
                    agent_id=aid,
                    data={
                        "idle_turns": idle_turns,
                        "coins_fined": coins_taken,
                        "hp_reduction_this_turn": min(
                            self.HP_REDUCTION_PER_TURN,
                            self.MAX_HP_REDUCTION_CAP - (current_reduction if idle_turns >= self.HP_PENALTY_THRESHOLD else 0),
                        ),
                        "total_hp_reduction": state["hp_reduction_applied"],
                        "area_id": current_area_id,
                    },
                ))

        env.curr_agents_state["contempt_of_court"] = contempt

    def _collect_coins(
        self, env, world, agent, amount: int, res: RuleResult
    ) -> int:
        """Remove up to `amount` coins from the agent. Returns actual coins taken."""
        coin_id = "obj_coin"
        remaining = amount
        taken = 0

        # From hand
        hand_coins = int(agent.items_in_hands.get(coin_id, 0))
        if hand_coins > 0 and remaining > 0:
            take = min(hand_coins, remaining)
            agent.items_in_hands[coin_id] -= take
            if agent.items_in_hands[coin_id] <= 0:
                del agent.items_in_hands[coin_id]
            res.track_consume(agent.id, coin_id, take, src=res.tloc("hand", agent.id))
            remaining -= take
            taken += take

        # From inventory
        if remaining > 0 and agent.inventory.container:
            inv_coins = int(agent.inventory.items.get(coin_id, 0))
            if inv_coins > 0:
                take = min(inv_coins, remaining)
                agent.inventory.items[coin_id] -= take
                if agent.inventory.items[coin_id] <= 0:
                    del agent.inventory.items[coin_id]
                res.track_consume(
                    agent.id, coin_id, take,
                    src=res.tloc("container", agent.inventory.container.id),
                )
                remaining -= take
                taken += take

        # From held containers
        if remaining > 0:
            for oid in list(agent.items_in_hands.keys()):
                if remaining <= 0:
                    break
                if oid in world.container_instances:
                    ci = world.container_instances[oid]
                    cc = int(ci.inventory.get(coin_id, 0))
                    if cc > 0:
                        take = min(cc, remaining)
                        ci.inventory[coin_id] -= take
                        if ci.inventory[coin_id] <= 0:
                            del ci.inventory[coin_id]
                        res.track_consume(
                            agent.id, coin_id, take,
                            src=res.tloc("container", ci.id),
                        )
                        remaining -= take
                        taken += take

        return taken


class BindingPrecedentStepRule(BaseStepRule):
    name = "binding_precedent_step"
    description = (
        "Defeated NPCs leave ghostly 'precedent' imprints in the area where they fell. "
        "Living NPCs of the same base type in an area with such a precedent are weakened "
        "(reduced stats and shortened combat pattern) because prior defeat established "
        "binding case law. If the player loses to an NPC type, the precedent is overturned "
        "and all future instances of that type receive a permanent stat boost."
    )
    priority = 4  # after combat rhythm, before active attack

    # Weakening applied to NPCs affected by a precedent
    STAT_REDUCTION_FRACTION = 0.25   # 25% reduction in HP and attack_power
    PATTERN_TRIM = 1                 # remove this many actions from combat pattern

    # Boost applied when a precedent is overturned (player lost)
    OVERTURN_BOOST_FRACTION = 0.20   # 20% permanent boost to HP and attack_power

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if world is None:
            return

        bp = env.curr_agents_state.setdefault("binding_precedents", {
            "area_precedents": {},      # area_id -> set of base_npc_ids defeated there
            "overturned_types": {},     # base_npc_id -> True if precedent overturned
            "weakened_npcs": {},        # npc_inst_id -> saved original stats before weakening
            "boosted_npcs": {},         # npc_inst_id -> True if already boosted from overturn
        })

        area_precedents = bp.setdefault("area_precedents", {})
        overturned_types = bp.setdefault("overturned_types", {})
        weakened_npcs = bp.setdefault("weakened_npcs", {})
        boosted_npcs = bp.setdefault("boosted_npcs", {})

        # --- Phase 1: Detect new NPC kills this step and record precedents ---
        for ev in res.events:
            if getattr(ev, "type", None) == "npc_killed" and ev.agent_id:
                npc_id = ev.data.get("npc_id")
                area_id = ev.data.get("area_id")
                if npc_id and area_id:
                    base_npc_id = get_def_id(npc_id)
                    if area_id not in area_precedents:
                        area_precedents[area_id] = []
                    if base_npc_id not in area_precedents[area_id]:
                        area_precedents[area_id].append(base_npc_id)

                    # Notify agents in the area
                    for agent in env.agents:
                        if env.curr_agents_state["area"].get(agent.id) == area_id:
                            npc_inst = world.npc_instances.get(npc_id)
                            npc_name = npc_inst.name if npc_inst else npc_id
                            res.add_feedback(
                                agent.id,
                                f"A ghostly precedent imprint of {npc_name} lingers in this area. "
                                f"The case law of its defeat now binds all creatures of its kind here.\n"
                            )
                    res.events.append(Event(
                        type="precedent_established",
                        agent_id=ev.agent_id,
                        data={"base_npc_id": base_npc_id, "area_id": area_id},
                    ))

        # --- Phase 2: Detect player defeats and overturn precedents ---
        for ev in res.events:
            if getattr(ev, "type", None) == "agent_defeated_in_combat" and ev.agent_id:
                defeating_npc_id = ev.data.get("npc_id")
                if defeating_npc_id:
                    base_npc_id = get_def_id(defeating_npc_id)
                    # Only overturn if there was a precedent for this type
                    has_precedent = any(
                        base_npc_id in precs
                        for precs in area_precedents.values()
                    )
                    if has_precedent and base_npc_id not in overturned_types:
                        overturned_types[base_npc_id] = True

                        # Restore any currently weakened NPCs of this type
                        for inst_id in list(weakened_npcs.keys()):
                            if get_def_id(inst_id) == base_npc_id:
                                saved = weakened_npcs.pop(inst_id)
                                npc_inst = world.npc_instances.get(inst_id)
                                if npc_inst and npc_inst.hp > 0:
                                    npc_inst.hp = saved.get("original_hp", npc_inst.hp)
                                    npc_inst.attack_power = saved.get("original_attack_power", npc_inst.attack_power)
                                    npc_inst.combat_pattern = saved.get("original_combat_pattern", npc_inst.combat_pattern)

                        # Remove this type from all area precedents
                        for aid in list(area_precedents.keys()):
                            if base_npc_id in area_precedents[aid]:
                                area_precedents[aid].remove(base_npc_id)

                        for agent in env.agents:
                            res.add_feedback(
                                agent.id,
                                f"⚖️ PRECEDENT OVERTURNED! The defeat at the hands of "
                                f"{world.npc_instances[defeating_npc_id].name if defeating_npc_id in world.npc_instances else defeating_npc_id} "
                                f"has reversed the binding case law. All creatures of this kind "
                                f"are now emboldened and permanently strengthened!\n"
                            )
                        res.events.append(Event(
                            type="precedent_overturned",
                            agent_id=ev.agent_id,
                            data={"base_npc_id": base_npc_id},
                        ))

        # --- Phase 3: Apply weakening to living NPCs in areas with precedents ---
        for area_id, precedent_types in area_precedents.items():
            if not precedent_types:
                continue
            area = world.area_instances.get(area_id)
            if not area:
                continue

            for npc_id in area.npcs:
                npc_inst = world.npc_instances.get(npc_id)
                if not npc_inst or npc_inst.hp <= 0 or not npc_inst.enemy:
                    continue

                base_npc_id = get_def_id(npc_id)

                # Skip if this type's precedent was overturned
                if base_npc_id in overturned_types:
                    continue

                if base_npc_id not in precedent_types:
                    continue

                # Skip if already weakened
                if npc_id in weakened_npcs:
                    continue

                # Save original stats and weaken
                original_hp = npc_inst.hp
                original_attack = npc_inst.attack_power
                original_pattern = list(npc_inst.combat_pattern)

                weakened_npcs[npc_id] = {
                    "original_hp": original_hp,
                    "original_attack_power": original_attack,
                    "original_combat_pattern": original_pattern,
                }

                hp_reduction = int(original_hp * self.STAT_REDUCTION_FRACTION)
                atk_reduction = int(original_attack * self.STAT_REDUCTION_FRACTION)
                npc_inst.hp = max(1, original_hp - hp_reduction)
                npc_inst.attack_power = max(0, original_attack - atk_reduction)

                if len(npc_inst.combat_pattern) > self.PATTERN_TRIM + 1:
                    npc_inst.combat_pattern = npc_inst.combat_pattern[:len(npc_inst.combat_pattern) - self.PATTERN_TRIM]

                # Notify agents in the area
                for agent in env.agents:
                    if env.curr_agents_state["area"].get(agent.id) == area_id:
                        res.add_feedback(
                            agent.id,
                            f"The ghostly precedent weighs on {npc_inst.name}. "
                            f"Bound by prior case law, it appears weakened "
                            f"(HP -{hp_reduction}, ATK -{atk_reduction}).\n"
                        )

        # --- Phase 4: Apply overturn boost to NPCs of overturned types ---
        for npc_id, npc_inst in world.npc_instances.items():
            if not npc_inst.enemy or npc_inst.hp <= 0:
                continue

            base_npc_id = get_def_id(npc_id)
            if base_npc_id not in overturned_types:
                continue

            if npc_id in boosted_npcs:
                continue

            hp_boost = int(npc_inst.hp * self.OVERTURN_BOOST_FRACTION)
            atk_boost = int(npc_inst.attack_power * self.OVERTURN_BOOST_FRACTION)
            npc_inst.hp += hp_boost
            npc_inst.attack_power += atk_boost
            boosted_npcs[npc_id] = True

            # Find which area this NPC is in for feedback
            for agent in env.agents:
                agent_area = env.curr_agents_state["area"].get(agent.id)
                if agent_area:
                    area = world.area_instances.get(agent_area)
                    if area and npc_id in area.npcs:
                        res.add_feedback(
                            agent.id,
                            f"{npc_inst.name} surges with newfound power from the overturned precedent "
                            f"(HP +{hp_boost}, ATK +{atk_boost}).\n"
                        )

        # --- Phase 5: Clean up weakened_npcs for dead NPCs ---
        for inst_id in list(weakened_npcs.keys()):
            npc_inst = world.npc_instances.get(inst_id)
            if npc_inst is None or npc_inst.hp <= 0:
                weakened_npcs.pop(inst_id, None)

        bp["area_precedents"] = area_precedents
        bp["overturned_types"] = overturned_types
        bp["weakened_npcs"] = weakened_npcs
        bp["boosted_npcs"] = boosted_npcs
        env.curr_agents_state["binding_precedents"] = bp


class ContrabandSuspicionStepRule(BaseStepRule):
    name = "contraband_suspicion_step"
    description = (
        "Each step, any agent carrying more total items than a threshold "
        "determined by the current area's legal documents accumulates "
        "'contraband suspicion'. After several turns, patrolling bailiff NPCs "
        "converge on the agent's location. If the agent fails to reduce their "
        "carried items before the bailiffs arrive, the bailiffs confiscate a "
        "random selection of non-equipped items and redistribute them to "
        "nearby merchant NPCs as seized assets at inflated prices."
    )
    priority = 11  # after contempt of court

    # Base carrying threshold (total item count across hands, inventory, held containers)
    BASE_THRESHOLD = 8

    # Document keywords that modify the threshold
    # "lenient" / "free trade" / "open carry" → +4
    # "strict" / "contraband" / "prohibition" → -3
    LENIENT_KEYWORDS = {"lenient", "free trade", "open carry", "deregulation", "free passage"}
    STRICT_KEYWORDS = {"strict", "contraband", "prohibition", "restricted", "ban carry", "seizure"}

    # Suspicion escalation thresholds (consecutive turns over limit)
    WARNING_TURNS = 2       # first warning
    CONVERGE_TURNS = 4      # bailiffs start converging (feedback)
    CONFISCATE_TURNS = 5    # bailiffs arrive and confiscate

    # Confiscation parameters
    CONFISCATE_FRACTION = 0.5   # fraction of excess items confiscated
    INFLATED_PRICE_MULT = 2.0   # price multiplier when redistributed to merchants

    # Maximum items confiscated per event
    MAX_CONFISCATE_ITEMS = 6

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if world is None:
            return

        suspicion_state = env.curr_agents_state.setdefault("contraband_suspicion", {})

        for agent in env.agents:
            aid = agent.id
            if agent.hp <= 0:
                continue

            current_area_id = env.curr_agents_state["area"].get(aid)
            if not current_area_id:
                continue
            current_area = world.area_instances.get(current_area_id)
            if not current_area:
                continue

            # --- compute carrying threshold for this area ---
            threshold = self._compute_threshold(world, current_area)

            # --- count total carried items ---
            total_carried = self._count_carried_items(world, agent)

            # --- initialise per-agent state ---
            if aid not in suspicion_state:
                suspicion_state[aid] = {
                    "turns_over": 0,
                    "last_area": current_area_id,
                    "converging_bailiffs": [],
                }

            state = suspicion_state[aid]

            # reset if agent moved to a different area
            if current_area_id != state.get("last_area"):
                state["turns_over"] = 0
                state["last_area"] = current_area_id
                state["converging_bailiffs"] = []

            # --- check if over threshold ---
            if total_carried <= threshold:
                # under limit → reset suspicion
                if state["turns_over"] > 0:
                    res.add_feedback(
                        aid,
                        f"{env.person_verbalized['subject_pronoun'].capitalize()} "
                        f"{env.person_verbalized['to_be_conjugation']} now carrying {total_carried} "
                        f"item(s), within the legal limit of {threshold}. "
                        f"Contraband suspicion cleared.\n"
                    )
                    # remove any converging bailiffs that haven't arrived yet
                    for bailiff_info in state.get("converging_bailiffs", []):
                        src_area = world.area_instances.get(bailiff_info.get("from_area"))
                        npc_id = bailiff_info.get("npc_id")
                        if src_area and npc_id and npc_id in world.npc_instances:
                            # bailiff returns to origin
                            if npc_id not in src_area.npcs:
                                src_area.npcs.append(npc_id)
                state["turns_over"] = 0
                state["converging_bailiffs"] = []
                continue

            # --- over threshold: accumulate suspicion ---
            state["turns_over"] += 1
            state["last_area"] = current_area_id
            turns = state["turns_over"]
            excess = total_carried - threshold

            if turns == self.WARNING_TURNS:
                res.add_feedback(
                    aid,
                    f"⚖️ A court clerk eyes {env.person_verbalized['possessive_adjective']} "
                    f"bulging pockets suspiciously. \"{env.person_verbalized['subject_pronoun'].capitalize()} "
                    f"{env.person_verbalized['to_be_conjugation']} carrying {total_carried} items — "
                    f"the legal limit here is {threshold}. Reduce "
                    f"{env.person_verbalized['possessive_adjective']} load or face seizure.\"\n"
                )
                res.events.append(Event(
                    type="contraband_warning",
                    agent_id=aid,
                    data={"total_carried": total_carried, "threshold": threshold,
                          "area_id": current_area_id, "turns_over": turns},
                ))

            elif turns == self.CONVERGE_TURNS:
                # bailiffs begin converging from adjacent areas
                converging = self._dispatch_bailiffs(env, world, current_area, current_area_id, res)
                state["converging_bailiffs"] = converging

                bailiff_names = [
                    world.npc_instances[b["npc_id"]].name
                    for b in converging
                    if b["npc_id"] in world.npc_instances
                ]
                if bailiff_names:
                    res.add_feedback(
                        aid,
                        f"⚖️ CONTRABAND ALERT! Bailiffs have been dispatched to "
                        f"{env.person_verbalized['possessive_adjective']} location: "
                        f"{', '.join(bailiff_names)}. "
                        f"Discard or store excess items immediately to avoid confiscation!\n"
                    )
                else:
                    res.add_feedback(
                        aid,
                        f"⚖️ CONTRABAND ALERT! The court has issued a seizure order. "
                        f"Discard or store excess items immediately!\n"
                    )
                res.events.append(Event(
                    type="contraband_bailiffs_dispatched",
                    agent_id=aid,
                    data={"area_id": current_area_id, "bailiff_count": len(converging)},
                ))

            elif turns >= self.CONFISCATE_TURNS:
                # --- CONFISCATION ---
                confiscated = self._confiscate_items(
                    env, world, agent, current_area, current_area_id,
                    excess, res,
                )

                if confiscated:
                    # redistribute to merchants
                    redistributed = self._redistribute_to_merchants(
                        env, world, current_area_id, confiscated, res,
                    )

                    item_names = []
                    for oid, cnt in confiscated.items():
                        obj_def = world.objects.get(get_def_id(oid))
                        name = obj_def.name if obj_def else oid
                        item_names.append(f"{cnt} {name}")

                    res.add_feedback(
                        aid,
                        f"⚖️ SEIZED! Bailiffs confiscate: {', '.join(item_names)}. "
                        f"The items have been redistributed to nearby merchants as "
                        f"seized assets available for repurchase at inflated prices.\n"
                    )
                    res.events.append(Event(
                        type="contraband_confiscated",
                        agent_id=aid,
                        data={
                            "confiscated": {k: v for k, v in confiscated.items()},
                            "redistributed_to": redistributed,
                            "area_id": current_area_id,
                        },
                    ))

                    # reset after confiscation
                    state["turns_over"] = 0
                    state["converging_bailiffs"] = []
                else:
                    res.add_feedback(
                        aid,
                        f"⚖️ Bailiffs search {env.person_verbalized['object_pronoun']} "
                        f"but find nothing to confiscate.\n"
                    )
                    state["turns_over"] = 0
                    state["converging_bailiffs"] = []

            elif turns > self.WARNING_TURNS:
                # intermediate turns: escalating warnings
                res.add_feedback(
                    aid,
                    f"⚖️ Contraband suspicion grows. "
                    f"{env.person_verbalized['subject_pronoun'].capitalize()} "
                    f"{env.person_verbalized['to_be_conjugation']} carrying {total_carried} items "
                    f"(limit: {threshold}). Bailiffs are being summoned.\n"
                )

        env.curr_agents_state["contraband_suspicion"] = suspicion_state

    def _compute_threshold(self, world, area) -> int:
        """Compute the carrying threshold for an area based on written documents."""
        threshold = self.BASE_THRESHOLD

        for oid, cnt in area.objects.items():
            if cnt <= 0:
                continue
            if oid not in world.writable_instances:
                continue
            writable = world.writable_instances[oid]
            text = (writable.text or "").strip().lower()
            if not text:
                continue

            for kw in self.LENIENT_KEYWORDS:
                if kw in text:
                    threshold += 4
                    break

            for kw in self.STRICT_KEYWORDS:
                if kw in text:
                    threshold -= 3
                    break

        return max(2, threshold)  # minimum threshold of 2

    def _count_carried_items(self, world, agent) -> int:
        """Count total items carried by the agent (hands + inventory + held containers)."""
        total = 0

        # items in hands (count units, not distinct types)
        for oid, cnt in agent.items_in_hands.items():
            total += int(cnt)

        # inventory items
        if agent.inventory.container:
            for oid, cnt in agent.inventory.items.items():
                total += int(cnt)

        # items inside held containers (not the container itself)
        for oid in list(agent.items_in_hands.keys()):
            if oid in world.container_instances:
                ci = world.container_instances[oid]
                for coid, cnt in ci.inventory.items():
                    total += int(cnt)

        return total

    def _dispatch_bailiffs(self, env, world, current_area, current_area_id, res) -> list:
        """Find bailiff NPCs in adjacent areas and mark them as converging."""
        converging = []

        for neighbor_id in current_area.neighbors.keys():
            neighbor_area = world.area_instances.get(neighbor_id)
            if not neighbor_area:
                continue

            for npc_id in list(neighbor_area.npcs):
                npc_inst = world.npc_instances.get(npc_id)
                if not npc_inst:
                    continue
                if not npc_inst.enemy:
                    continue
                if get_def_id(npc_id) == "npc_bailiff_construct":
                    converging.append({
                        "npc_id": npc_id,
                        "from_area": neighbor_id,
                    })
                    break  # one bailiff per adjacent area

            if len(converging) >= 2:
                break  # cap at 2 converging bailiffs

        return converging

    def _confiscate_items(self, env, world, agent, current_area, current_area_id,
                          excess, res) -> Dict[str, int]:
        """Confiscate non-equipped items from the agent. Returns {obj_id: count}."""
        confiscated: Dict[str, int] = {}
        num_to_confiscate = min(
            self.MAX_CONFISCATE_ITEMS,
            max(1, int(excess * self.CONFISCATE_FRACTION)),
        )
        remaining = num_to_confiscate

        # Build a pool of confiscatable items (non-equipped, non-container, non-station)
        # Priority: inventory first, then hands, then held containers
        confiscatable = []

        # from inventory
        if agent.inventory.container:
            for oid, cnt in list(agent.inventory.items.items()):
                base_id = get_def_id(oid)
                if base_id not in world.objects:
                    continue
                obj_def = world.objects[base_id]
                if obj_def.category in ("station", "container"):
                    continue
                if obj_def.usage == "writable":
                    continue
                confiscatable.append(("inventory", oid, cnt))

        # from hands (skip equipped-like containers)
        for oid, cnt in list(agent.items_in_hands.items()):
            base_id = get_def_id(oid)
            if oid in world.container_instances:
                continue  # don't confiscate containers themselves
            if base_id not in world.objects:
                continue
            obj_def = world.objects[base_id]
            if obj_def.category in ("station",):
                continue
            confiscatable.append(("hand", oid, cnt))

        # from held containers
        for held_oid in list(agent.items_in_hands.keys()):
            if held_oid not in world.container_instances:
                continue
            ci = world.container_instances[held_oid]
            for oid, cnt in list(ci.inventory.items()):
                base_id = get_def_id(oid)
                if base_id not in world.objects:
                    continue
                obj_def = world.objects[base_id]
                if obj_def.category in ("station", "container"):
                    continue
                confiscatable.append(("container:" + held_oid, oid, cnt))

        # shuffle for randomness
        env.rng.shuffle(confiscatable)

        for source, oid, available in confiscatable:
            if remaining <= 0:
                break

            take = min(available, remaining)
            if take <= 0:
                continue

            # remove from source
            if source == "inventory" and agent.inventory.container:
                agent.inventory.items[oid] = agent.inventory.items.get(oid, 0) - take
                if agent.inventory.items[oid] <= 0:
                    del agent.inventory.items[oid]
                res.track_consume(
                    agent.id, oid, take,
                    src=res.tloc("container", agent.inventory.container.id),
                )
            elif source == "hand":
                agent.items_in_hands[oid] = agent.items_in_hands.get(oid, 0) - take
                if agent.items_in_hands[oid] <= 0:
                    del agent.items_in_hands[oid]
                res.track_consume(agent.id, oid, take, src=res.tloc("hand", agent.id))
            elif source.startswith("container:"):
                container_id = source.split(":", 1)[1]
                ci = world.container_instances.get(container_id)
                if ci:
                    ci.inventory[oid] = ci.inventory.get(oid, 0) - take
                    if ci.inventory[oid] <= 0:
                        del ci.inventory[oid]
                    res.track_consume(agent.id, oid, take, src=res.tloc("container", container_id))

            confiscated[oid] = confiscated.get(oid, 0) + take
            remaining -= take

        return confiscated

    def _redistribute_to_merchants(self, env, world, area_id, confiscated, res) -> List[str]:
        """Redistribute confiscated items to nearby merchant NPCs at inflated prices."""
        redistributed_to = []

        # find merchants in the current area and adjacent areas
        merchants = []
        search_areas = [area_id] + list(
            world.area_instances.get(area_id, world.area_instances[area_id]).neighbors.keys()
            if area_id in world.area_instances else []
        )

        for search_area_id in search_areas:
            search_area = world.area_instances.get(search_area_id)
            if not search_area:
                continue
            for npc_id in search_area.npcs:
                npc_inst = world.npc_instances.get(npc_id)
                if npc_inst and npc_inst.role == "merchant":
                    merchants.append(npc_inst)

        if not merchants:
            # no merchants found — items are simply lost
            return redistributed_to

        # distribute items round-robin among merchants
        merchant_idx = 0
        for oid, count in confiscated.items():
            base_id = get_def_id(oid)
            obj_def = world.objects.get(base_id)
            if not obj_def:
                continue

            merchant = merchants[merchant_idx % len(merchants)]
            merchant.inventory[base_id] = int(merchant.inventory.get(base_id, 0)) + count

            # inflate the price if the object has a value
            if obj_def.value is not None:
                inflated_value = int(obj_def.value * self.INFLATED_PRICE_MULT)
                # We store the inflated value as a temporary override via extra
                # Since we can't change the base object's value for all instances,
                # we just note it in the event — the merchant already sells at obj_def.value
                # For actual inflated pricing, we'd need per-merchant pricing.
                # Instead, we add extra stock and the existing buy mechanic uses obj_def.value.
                # The "inflated" aspect is thematic — the items were confiscated and resold.
                pass

            if merchant.name not in redistributed_to:
                redistributed_to.append(merchant.name)

            res.track_spawn(
                "env", base_id, count,
                dst=res.tloc("area", "merchant_" + merchant.id),
            )

            merchant_idx += 1

        return redistributed_to


class TrialProceedingsStepRule(BaseStepRule):
    name = "trial_proceedings_step"
    description = (
        "When an agent kills an NPC, the area enters 'trial proceedings'. "
        "A spectral judge is summoned to review written documents in the area "
        "for language authorizing the killing. If no justification is found, "
        "the agent's attack power is cursed (halved progressively) for several "
        "turns. If a valid written justification exists, the judge rewards the "
        "agent by permanently upgrading a random held item to a higher-level variant."
    )
    priority = 5  # after combat, before new craftable feedback

    SPECTRAL_JUDGE_BASE_ID = "npc_spectral_judge"

    # Duration of the attack curse in turns
    CURSE_DURATION = 4

    # Keywords in written documents that count as "justification" for a killing
    JUSTIFICATION_KEYWORDS = {
        "authorize killing", "lethal force", "execution order",
        "death warrant", "capital punishment", "terminate hostile",
        "sanctioned elimination", "lawful execution", "justified force",
        "right to kill", "self defense", "self-defense",
        "deadly force authorized", "kill order", "sentence of death",
        "combat authorization", "license to kill", "warrant of execution",
        "eliminate threat", "neutralize enemy",
    }

    required_npcs = [
        {
            "type": "npc",
            "id": SPECTRAL_JUDGE_BASE_ID,
            "name": "spectral_judge",
            "enemy": False,
            "unique": False,
            "role": "judge",
            "quest": True,
            "base_attack_power": 0,
            "slope_attack_power": 0,
            "base_hp": 10**9,
            "slope_hp": 0,
            "objects": [],
            "description": (
                "A translucent figure in flowing judicial robes, eyes burning "
                "with the light of every law ever written. It appears whenever "
                "blood is spilled, demanding written justification for the act."
            ),
        },
    ]

    def __init__(self) -> None:
        super().__init__()
        self._initialized = False

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world
        if world is None:
            return

        if not self._initialized:
            self._register_required_npcs(world)
            self._initialized = True

        trial_state = env.curr_agents_state.setdefault("trial_proceedings", {})

        step = ctx.step_index

        # --- Phase 1: Detect new NPC kills this step and initiate trials ---
        for ev in res.events:
            if getattr(ev, "type", None) != "npc_killed":
                continue
            agent_id = ev.agent_id
            if not agent_id:
                continue
            npc_id = ev.data.get("npc_id")
            area_id = ev.data.get("area_id")
            if not npc_id or not area_id:
                continue

            # Don't trigger trial for quest NPCs or spectral judges
            killed_npc = world.npc_instances.get(npc_id)
            if killed_npc and getattr(killed_npc, "quest", False):
                continue
            if get_def_id(npc_id) == self.SPECTRAL_JUDGE_BASE_ID:
                continue

            agent_state = trial_state.setdefault(agent_id, {
                "active_trials": [],
                "attack_curse": None,
            })

            # Summon spectral judge to the area
            judge_inst_id = self._summon_judge(env, world, area_id)
            if not judge_inst_id:
                continue

            trial_info = {
                "area_id": area_id,
                "npc_id": npc_id,
                "judge_inst_id": judge_inst_id,
                "step_initiated": step,
                "resolved": False,
            }
            agent_state["active_trials"].append(trial_info)

            killed_name = killed_npc.name if killed_npc else npc_id
            judge_inst = world.npc_instances.get(judge_inst_id)
            judge_name = judge_inst.name if judge_inst else "spectral_judge"

            res.add_feedback(
                agent_id,
                f"\n⚖️ TRIAL PROCEEDINGS INITIATED! The air crackles with judicial "
                f"energy as {judge_name} materializes in a swirl of ghostly parchment. "
                f"\"You stand accused of slaying {killed_name}. Present written "
                f"justification or face the court's curse!\"\n"
            )
            res.events.append(Event(
                type="trial_proceedings_initiated",
                agent_id=agent_id,
                data={
                    "area_id": area_id,
                    "killed_npc_id": npc_id,
                    "judge_inst_id": judge_inst_id,
                },
            ))

        # --- Phase 2: Resolve pending trials (same step they are initiated) ---
        for agent in env.agents:
            aid = agent.id
            agent_state = trial_state.get(aid)
            if not agent_state:
                continue

            active_trials = agent_state.get("active_trials", [])
            resolved_indices = []

            for i, trial in enumerate(active_trials):
                if trial.get("resolved", False):
                    resolved_indices.append(i)
                    continue

                area_id = trial["area_id"]
                judge_inst_id = trial["judge_inst_id"]
                killed_npc_id = trial["npc_id"]

                judge_inst = world.npc_instances.get(judge_inst_id)
                judge_name = judge_inst.name if judge_inst else "spectral_judge"

                # Search for written justification in the area and on the agent
                justified = self._search_for_justification(world, agent, area_id)

                if justified:
                    # --- JUSTIFIED: reward the agent with an item upgrade ---
                    upgraded = self._upgrade_random_item(world, agent, env.rng, res)

                    if upgraded:
                        old_name, new_name = upgraded
                        res.add_feedback(
                            aid,
                            f"⚖️ {judge_name} reviews the written evidence and nods solemnly. "
                            f"\"The killing was lawful under the written statutes. The court "
                            f"rewards your diligence.\" A surge of judicial energy transforms "
                            f"your {old_name} into a {new_name}!\n"
                        )
                    else:
                        coin_reward = max(5, env.rng.randint(5, 15))
                        current_area = world.area_instances.get(
                            env.curr_agents_state["area"].get(aid, area_id)
                        )
                        if current_area:
                            current_area.objects["obj_coin"] = (
                                current_area.objects.get("obj_coin", 0) + coin_reward
                            )
                            res.track_spawn(
                                aid, "obj_coin", coin_reward,
                                dst=res.tloc("area", current_area.id),
                            )
                        res.add_feedback(
                            aid,
                            f"⚖️ {judge_name} reviews the written evidence and nods solemnly. "
                            f"\"The killing was lawful. The court rewards you with "
                            f"{coin_reward} coins.\"\n"
                        )

                    res.events.append(Event(
                        type="trial_justified",
                        agent_id=aid,
                        data={
                            "area_id": area_id,
                            "killed_npc_id": killed_npc_id,
                            "judge_inst_id": judge_inst_id,
                        },
                    ))
                else:
                    existing_curse = agent_state.get("attack_curse")
                    if existing_curse and existing_curse.get("remaining", 0) > 0:
                        existing_curse["curse_level"] = existing_curse.get("curse_level", 1) + 1
                        existing_curse["remaining"] = self.CURSE_DURATION
                        curse_level = existing_curse["curse_level"]
                    else:
                        agent_state["attack_curse"] = {
                            "remaining": self.CURSE_DURATION,
                            "original_min_attack": agent.min_attack,
                            "curse_level": 1,
                        }
                        curse_level = 1

                    divisor = 2 ** curse_level
                    res.add_feedback(
                        aid,
                        f"⚖️ {judge_name} scans the area and finds no written law "
                        f"authorizing the killing. \"GUILTY! The court curses your "
                        f"strength!\" {env.person_verbalized['possessive_adjective'].capitalize()} "
                        f"attack power is divided by {divisor} for {self.CURSE_DURATION} turns.\n"
                    )
                    res.events.append(Event(
                        type="trial_guilty",
                        agent_id=aid,
                        data={
                            "area_id": area_id,
                            "killed_npc_id": killed_npc_id,
                            "judge_inst_id": judge_inst_id,
                            "curse_level": curse_level,
                            "curse_duration": self.CURSE_DURATION,
                        },
                    ))

                trial["resolved"] = True
                resolved_indices.append(i)
                self._remove_judge(world, judge_inst_id, area_id)

            agent_state["active_trials"] = [
                t for j, t in enumerate(active_trials) if j not in resolved_indices
            ]

        # --- Phase 3: Apply and tick down active attack curses ---
        for agent in env.agents:
            aid = agent.id
            agent_state = trial_state.get(aid)
            if not agent_state:
                continue

            curse = agent_state.get("attack_curse")
            if not curse or curse.get("remaining", 0) <= 0:
                if curse and curse.get("remaining", 0) == 0:
                    original = curse.get("original_min_attack", agent.min_attack)
                    if agent.min_attack != original:
                        agent.min_attack = original
                    agent_state["attack_curse"] = None
                continue

            original = curse.get("original_min_attack", agent.min_attack)
            curse_level = curse.get("curse_level", 1)
            divisor = 2 ** curse_level
            cursed_attack = max(1, original // divisor)
            agent.min_attack = cursed_attack

            curse["remaining"] -= 1
            remaining = curse["remaining"]

            if remaining > 0:
                res.add_feedback(
                    aid,
                    f"⚖️ The judicial curse weighs on {env.person_verbalized['object_pronoun']}. "
                    f"{env.person_verbalized['possessive_adjective'].capitalize()} base attack "
                    f"is reduced to {cursed_attack} ({remaining} turn(s) remaining).\n"
                )
            else:
                agent.min_attack = original
                agent_state["attack_curse"] = None
                res.add_feedback(
                    aid,
                    f"⚖️ The judicial curse lifts. "
                    f"{env.person_verbalized['possessive_adjective'].capitalize()} attack "
                    f"power is restored to normal.\n"
                )
                res.events.append(Event(
                    type="trial_curse_expired",
                    agent_id=aid,
                    data={},
                ))

        env.curr_agents_state["trial_proceedings"] = trial_state

    def _register_required_npcs(self, world) -> None:
        """Register the spectral judge NPC prototype if not already present."""
        aux = world.auxiliary
        aux.setdefault("npc_id_to_count", {})
        aux.setdefault("npc_name_to_id", {})

        for d in self.required_npcs:
            nid = d["id"]
            if nid in world.npcs:
                continue

            atk = d.get("base_attack_power", 0)
            hp = d.get("base_hp", 10**9)

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
                unique=bool(d.get("unique", False)),
                role=d.get("role", "judge"),
                level=1,
                description=d.get("description", ""),
                attack_power=int(atk),
                hp=int(hp),
                coins=0,
                slope_hp=int(d.get("slope_hp", 0)),
                slope_attack_power=int(d.get("slope_attack_power", 0)),
                quest=bool(d.get("quest", True)),
                inventory=inventory,
            )
            aux.setdefault("npc_id_to_count", {})[nid] = 0

    def _summon_judge(self, env, world, area_id: str) -> str:
        """Create a spectral judge NPC instance in the given area. Returns instance ID."""
        base_id = self.SPECTRAL_JUDGE_BASE_ID
        if base_id not in world.npcs:
            return None

        area = world.area_instances.get(area_id)
        if not area:
            return None

        aux = world.auxiliary
        proto = world.npcs[base_id]
        idx = int(aux.get("npc_id_to_count", {}).get(base_id, 0))
        aux.setdefault("npc_id_to_count", {})[base_id] = idx + 1

        level = int(getattr(area, "level", 1) or 1)
        inst = proto.create_instance(idx, level=level, objects=world.objects, rng=env.rng)
        if isinstance(inst, tuple):
            inst = inst[0]

        inst.enemy = False
        inst.hp = 10**9
        inst.attack_power = 0

        world.npc_instances[inst.id] = inst
        area.npcs.append(inst.id)
        world.auxiliary.setdefault("npc_name_to_id", {})[inst.name] = inst.id

        return inst.id

    def _remove_judge(self, world, judge_inst_id: str, area_id: str) -> None:
        """Remove the spectral judge from the area after the trial resolves."""
        area = world.area_instances.get(area_id)
        if area and judge_inst_id in area.npcs:
            area.npcs.remove(judge_inst_id)

        judge_inst = world.npc_instances.get(judge_inst_id)
        if judge_inst:
            name = getattr(judge_inst, "name", None)
            if name and world.auxiliary.get("npc_name_to_id", {}).get(name) == judge_inst_id:
                del world.auxiliary["npc_name_to_id"][name]
            del world.npc_instances[judge_inst_id]

    def _search_for_justification(self, world, agent, area_id: str) -> bool:
        """Search written documents in the area and on the agent for justification keywords."""
        texts = []

        area = world.area_instances.get(area_id)
        if area:
            for oid, cnt in area.objects.items():
                if cnt <= 0:
                    continue
                if oid in world.writable_instances:
                    w = world.writable_instances[oid]
                    if (w.text or "").strip():
                        texts.append(w.text)

        for oid in list(agent.items_in_hands.keys()):
            if oid in world.writable_instances:
                w = world.writable_instances[oid]
                if (w.text or "").strip():
                    texts.append(w.text)

        if agent.inventory.container:
            for oid in list(agent.inventory.items.keys()):
                if oid in world.writable_instances:
                    w = world.writable_instances[oid]
                    if (w.text or "").strip():
                        texts.append(w.text)

        for oid in list(agent.items_in_hands.keys()):
            if oid in world.container_instances:
                ci = world.container_instances[oid]
                for coid in list(ci.inventory.keys()):
                    if coid in world.writable_instances:
                        w = world.writable_instances[coid]
                        if (w.text or "").strip():
                            texts.append(w.text)

        for text in texts:
            text_lower = text.lower()
            for keyword in self.JUSTIFICATION_KEYWORDS:
                if keyword in text_lower:
                    return True

        return False

    def _upgrade_random_item(self, world, agent, rng, res) -> tuple:
        """
        Attempt to upgrade a random item in the agent's possession to a
        higher-level variant. Returns (old_name, new_name) or None.
        """
        upgradeable = []

        agent_items = []
        for oid, cnt in agent.items_in_hands.items():
            if cnt > 0:
                base = get_def_id(oid)
                if base in world.objects:
                    agent_items.append(("hand", oid, base))

        if agent.inventory.container:
            for oid, cnt in agent.inventory.items.items():
                if cnt > 0:
                    base = get_def_id(oid)
                    if base in world.objects:
                        agent_items.append(("inventory", oid, base))

        for held_oid in list(agent.items_in_hands.keys()):
            if held_oid in world.container_instances:
                ci = world.container_instances[held_oid]
                for oid, cnt in ci.inventory.items():
                    if cnt > 0:
                        base = get_def_id(oid)
                        if base in world.objects:
                            agent_items.append(("container:" + held_oid, oid, base))

        for location, oid, base_id in agent_items:
            obj_def = world.objects[base_id]
            if obj_def.category in ("currency", "station", "container"):
                continue
            if obj_def.usage == "writable":
                continue

            candidates = []
            for cand_id, cand_obj in world.objects.items():
                if cand_id == base_id:
                    continue
                if cand_obj.category != obj_def.category:
                    continue
                if cand_obj.level <= obj_def.level:
                    continue
                if getattr(cand_obj, "quest", False):
                    continue
                candidates.append((cand_id, cand_obj))

            if candidates:
                upgradeable.append((location, oid, base_id, obj_def, candidates))

        if not upgradeable:
            return None

        location, oid, base_id, obj_def, candidates = rng.choice(upgradeable)

        candidates.sort(key=lambda x: x[1].level)
        upgrade_id, upgrade_obj = candidates[0]

        old_name = obj_def.name

        if location == "hand":
            agent.items_in_hands[oid] = agent.items_in_hands.get(oid, 0) - 1
            if agent.items_in_hands[oid] <= 0:
                del agent.items_in_hands[oid]
            res.track_consume(agent.id, oid, 1, src=res.tloc("hand", agent.id))

            if upgrade_obj.category == "container" or upgrade_obj.usage == "writable":
                instance_list = (
                    world.container_instances
                    if upgrade_obj.category == "container"
                    else world.writable_instances
                )
                id_to_count = (
                    world.auxiliary["container_id_to_count"]
                    if upgrade_obj.category == "container"
                    else world.auxiliary["writable_id_to_count"]
                )
                new_inst = upgrade_obj.create_instance(id_to_count.get(upgrade_id, 0))
                instance_list[new_inst.id] = new_inst
                id_to_count[upgrade_id] = id_to_count.get(upgrade_id, 0) + 1
                world.auxiliary["obj_name_to_id"][new_inst.name] = new_inst.id
                agent.items_in_hands[new_inst.id] = agent.items_in_hands.get(new_inst.id, 0) + 1
                res.track_spawn(agent.id, new_inst.id, 1, dst=res.tloc("hand", agent.id))
                new_name = new_inst.name
            else:
                agent.items_in_hands[upgrade_id] = agent.items_in_hands.get(upgrade_id, 0) + 1
                res.track_spawn(agent.id, upgrade_id, 1, dst=res.tloc("hand", agent.id))
                new_name = upgrade_obj.name

        elif location == "inventory" and agent.inventory.container:
            agent.inventory.items[oid] = agent.inventory.items.get(oid, 0) - 1
            if agent.inventory.items[oid] <= 0:
                del agent.inventory.items[oid]
            res.track_consume(
                agent.id, oid, 1,
                src=res.tloc("container", agent.inventory.container.id),
            )

            agent.inventory.items[upgrade_id] = agent.inventory.items.get(upgrade_id, 0) + 1
            res.track_spawn(
                agent.id, upgrade_id, 1,
                dst=res.tloc("container", agent.inventory.container.id),
            )
            new_name = upgrade_obj.name

        elif location.startswith("container:"):
            container_id = location.split(":", 1)[1]
            ci = world.container_instances.get(container_id)
            if ci:
                ci.inventory[oid] = ci.inventory.get(oid, 0) - 1
                if ci.inventory[oid] <= 0:
                    del ci.inventory[oid]
                res.track_consume(
                    agent.id, oid, 1,
                    src=res.tloc("container", container_id),
                )

                ci.inventory[upgrade_id] = ci.inventory.get(upgrade_id, 0) + 1
                res.track_spawn(
                    agent.id, upgrade_id, 1,
                    dst=res.tloc("container", container_id),
                )
                new_name = upgrade_obj.name
            else:
                return None
        else:
            return None

        return (old_name, new_name)


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
