import os
import gymnasium as gym
import math
import random
import json
from datetime import datetime, timedelta
import inspect
import shlex

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from utils import *

# from torch import obj
from games.generated.saltglass.world import World, Object, NPC, Path, Area, Place, Container
from games.generated.saltglass.agent import Agent, Inventory, Action
from tools.logger import get_logger
from games.generated.saltglass.rule import RuleResult, RuleContext, ActionRuleEngine, BaseStepRule, BaseActionRule, DependencyTracker
from games.generated.saltglass.rules.reward_functions import RewardFunction, RewardBreakdown, DefaultRewardFunction
import games.generated.saltglass.rules.action_rules as action_rules
import games.generated.saltglass.rules.step_rules as step_rules


class AgentOdysseyEnv(gym.Env):

    def __init__(self, 
        seed: int, 
        agents, 
        world_definition_path: str, 
        run_dir: str, 
        config_path: str, 
        enable_obs_valid_actions: bool = False, 
        from_step: Optional[int] = None,
        save_dep_graph_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.world_definition = json.load(open(world_definition_path))
        if "main_quest" in self.world_definition.get("custom_events", []):
            self._add_main_quest_entities()
        if "side_quest" in self.world_definition.get("custom_events", []):
            self._add_side_quest_entities()
        self.run_dir = run_dir
        self.config_path = config_path
        self.config = load_config(self.config_path, from_step=from_step)
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.rng = random.Random(self.seed)
        # self.world = None if self.config["world"] is None else World.dict_to_world(self.config["world"])
        if isinstance(agents, list):
            self.agents = agents
        else:
            self.agents = [agents]
        self.obs = {a.id: "" for a in self.agents}
        self.person_verbalized = {"subject_pronoun": "I", 
                                  "object_pronoun": "me", 
                                  "possessive_adjective": "my", 
                                  "possessive_pronoun": "mine", 
                                  "reflexive_pronoun": "myself", 
                                  "to_be_conjugation": "am"}  # first person by default
        self.step_delta_time = 60 * 10  # 10 minutes per step
        self.logger = get_logger("EnvLogger")
        self.steps = self.config["step"]
        self.curr_time = datetime.strptime(self.config["curr_time"], "%Y-%m-%d %H:%M:%S")
        self.curr_agents_state = self.config["curr_agents_state"]
        self.scores = self.config["scores"]
        self.enable_obs_valid_actions = enable_obs_valid_actions
        self.save_dep_graph_steps = save_dep_graph_steps

        self.world: World = None
        self.action_rule_engine: ActionRuleEngine = ActionRuleEngine(rules=[
                cls() for _, cls in inspect.getmembers(action_rules, inspect.isclass)
                if issubclass(cls, BaseActionRule) and cls is not BaseActionRule and not inspect.isabstract(cls)
            ])
        self.step_rules: list[BaseStepRule] = sorted([
            cls() for _, cls in inspect.getmembers(step_rules, inspect.isclass)
            if issubclass(cls, BaseStepRule) and cls is not BaseStepRule and not inspect.isabstract(cls)
        ], key=lambda rule: rule.priority)
        self.reward_function: RewardFunction = DefaultRewardFunction()
        
        for a in self.agents:
          a._action_rules_module = action_rules
          a._base_action_rule_class = BaseActionRule

    def _add_main_quest_entities(self) -> None:
        entities = self.world_definition.setdefault("entities", {})
        
        # Add required objects
        objects_list = entities.setdefault("objects", [])
        existing_obj_ids = {obj["id"] for obj in objects_list}
        for quest_obj in step_rules.MainQuestStepRule.required_objects:
            if quest_obj["id"] not in existing_obj_ids:
                objects_list.append(quest_obj)
        self.world_definition["initializations"]["undistributable_objects"].extend(step_rules.MainQuestStepRule.undistributable_objects)
        
        # Add required NPCs
        npcs_list = entities.setdefault("npcs", [])
        existing_npc_ids = {npc["id"] for npc in npcs_list}
        for quest_npc in step_rules.MainQuestStepRule.required_npcs:
            if quest_npc["id"] not in existing_npc_ids:
                npcs_list.append(quest_npc)

    def _add_side_quest_entities(self) -> None:
        """Add side quest NPCs (quest givers) to the world definition."""
        entities = self.world_definition.setdefault("entities", {})
        
        # add required NPCs for side quests
        npcs_list = entities.setdefault("npcs", [])
        existing_npc_ids = {npc["id"] for npc in npcs_list}
        for quest_npc in step_rules.SideQuestStepRule.required_npcs:
            if quest_npc["id"] not in existing_npc_ids:
                npcs_list.append(quest_npc)
    
    def __str__(self):
        return "AgentOdysseyEnv"
    
    def get_all_valid_actions(self, agent: Agent) -> list[str]:
        actions = defaultdict(list)
        curr_area_id = self.curr_agents_state["area"][agent.id]
        curr_area = self.world.area_instances[curr_area_id]
        area_objects = list(curr_area.objects.keys())
        neighboring_areas = [self.world.area_instances[nid].name for nid in curr_area.neighbors.keys()]
        hand_objects = list(agent.items_in_hands.keys())
        equipped_objects = list(agent.equipped_items_in_limb.keys())
        inventory_items = list(agent.inventory.items.keys()) if agent.inventory.container else []
        container_names = [container.name for container in self.world.container_instances.values()]

        obj_id_to_name = {v: k for k, v in self.world.auxiliary["obj_name_to_id"].items()}

        # Filter out dynamically-injected phantom IDs (e.g. mirage objects) that
        # are not registered in obj_name_to_id and cannot be acted upon.
        area_objects = [oid for oid in area_objects if oid in obj_id_to_name]

        # --- Movement ---
        for area in neighboring_areas:
            actions["enter"].append(f"enter {area}")

        # --- Pick up / Drop ---
        for obj_id in area_objects:
            actions["pick up"].append(f"pick up {obj_id_to_name[obj_id]}")
        for obj_id in hand_objects:
            actions["drop"].append(f"drop {obj_id_to_name[obj_id]}")

        # --- Combat ---
        for npc_id in curr_area.npcs:
            actions["attack"].append(f"attack {self.world.npc_instances[npc_id].name}")

        # --- Store (in inventory or containers) ---
        all_storable_objects = area_objects + hand_objects
        for obj_id in all_storable_objects:
            obj_name = obj_id_to_name[obj_id]
            obj_amount = range(1, curr_area.objects.get(obj_id, 0) + agent.items_in_hands.get(obj_id, 0) + 1)
            for amount in obj_amount:
                actions["store"].append(f"store {amount} {obj_name} inventory")
                for container_name in container_names:
                    container_id = self.world.auxiliary["obj_name_to_id"][container_name]
                    if container_id in hand_objects or container_id in equipped_objects or container_id in area_objects:
                      actions["store"].append(f"store {amount} {obj_name} {container_name}")

        # --- Take out / Discard ---
        for container_name in container_names:
            container_id = self.world.auxiliary["obj_name_to_id"][container_name]
            if container_id in hand_objects or container_id in equipped_objects or container_id in area_objects:
                actions["store"].append(f"store {amount} {obj_name} {container_name}")
            for obj_id in self.world.objects.keys():
                container_instance = self.world.container_instances[container_id]
                obj_name = obj_id_to_name[obj_id]
                obj_amount = range(1, container_instance.inventory.get(obj_id, 0) + 1)
                for amount in obj_amount:
                    actions["take out"].append(f"take out {obj_name} {container_name}")
                    actions["discard"].append(f"discard {amount} {obj_name} {container_name}")

        if agent.inventory.container:
            for obj_id in agent.inventory.items.keys():
                obj_name = obj_id_to_name[obj_id]
                obj_amount = range(1, agent.inventory.items.get(obj_id, 0) + 1)
                actions["take out"].append(f"take out {obj_name} inventory")
                for amount in obj_amount:
                    actions["discard"].append(f"discard {amount} {obj_name} inventory")

        # --- Equip / Unequip ---
        equiptable_categories = ["weapon", "armor", "container"]
        equiptable_objects = [obj_id for obj_id in hand_objects if obj_id in self.world.objects 
                              and self.world.objects[obj_id].category in equiptable_categories]
        for obj_id in equiptable_objects:
            obj_name = obj_id_to_name[obj_id]
            actions["equip"].append(f"equip {obj_name}")
        for obj_id in equipped_objects:
            obj_name = obj_id_to_name[obj_id]
            actions["unequip"].append(f"unequip {obj_name}")

        # --- Inspect / Craft ---
        inspectable_objects = hand_objects + inventory_items + area_objects + equipped_objects
        for obj_id in inspectable_objects:
            if obj_id in self.world.objects:
                actions["inspect"].append(f"inspect {obj_id_to_name[obj_id]}")
        hand_containers = [obj_id for obj_id in hand_objects if obj_id in container_names]
        for obj_id, obj in self.world.objects.items():
            if obj.craft_ingredients:
                min_amount = None
                for obj_id, count in obj.craft_ingredients.items():
                    available_count = 0
                    if obj_id in agent.items_in_hands:
                        available_count += agent.items_in_hands[obj_id]
                    if agent.inventory.container and obj_id in agent.inventory.items:
                        available_count += agent.inventory.items[obj_id]
                    for container_name in hand_containers:
                        container_instance = self.world.container_instances[container_name]
                        if obj_id in container_instance.inventory:
                            available_count += container_instance.inventory[obj_id]
                    min_amount = min(min_amount, math.floor(available_count / count)) \
                                    if min_amount is not None else math.floor(available_count / count)
                for craft_amount in range(1, min_amount + 1):
                    actions["craft"].append(f"craft {craft_amount} {obj.name}")

        # --- Trade ---
        coin_id = "obj_coin"
        coin_have = int(agent.items_in_hands.get(coin_id, 0))
        if agent.inventory.container:
            coin_have += int(agent.inventory.items.get(coin_id, 0))
        held_containers = [
            self.world.container_instances[oid]
            for oid in agent.items_in_hands.keys()
            if oid in self.world.container_instances
        ]
        for ci in held_containers:
            coin_have += int(ci.inventory.get(coin_id, 0))
        for npc_id in curr_area.npcs:
            npc_instance = self.world.npc_instances[npc_id]
            if npc_instance.role == "merchant":
                for obj_id in npc_instance.inventory:
                    obj_def = self.world.objects[obj_id]
                    if obj_def.value is None:
                        continue
                    if obj_def.value <= coin_have:
                        for cnt in range(1, int(coin_have // obj_def.value)):
                            actions["buy"].append(f"buy {cnt} {obj_def.name} {npc_instance.name}")
                for obj_id in (list(agent.items_in_hands.keys()) + list(agent.equipped_items_in_limb.keys())):
                    if obj_id not in self.world.objects:  # get rid of notes and containers
                        continue
                    obj_def = self.world.objects[obj_id]
                    if obj_def.value is None or npc_instance.coins is None:
                        continue
                    if obj_def.value <= npc_instance.coins:
                        for cnt in range(1, int(npc_instance.coins // obj_def.value)):
                            actions["sell"].append(f"sell {cnt} {obj_def.name} {npc_instance.name}")
                for ci in held_containers:
                    for obj_id in ci.inventory.items():
                        if obj_id not in self.world.objects:
                          continue
                        obj_def = self.world.objects[obj_id]
                        if obj_def.value is None or npc_instance.coins is None:
                            continue
                        if obj_def.value <= npc_instance.coins:
                            for cnt in range(1, int(npc_instance.coins // obj_def.value)):
                                actions["sell"].append(f"sell {cnt} {obj_def.name} {npc_instance.name}")
        # --- Write ---
        writable_objects = [obj_id for obj_id in hand_objects if obj_id in self.world.writable_instances]
        have_pen = any(obj_id for obj_id in hand_objects 
                       if get_def_id(obj_id) in self.world.objects 
                       and self.world.objects[get_def_id(obj_id)].usage == "write")
        if have_pen:
            for obj_id in writable_objects:
                obj_name = obj_id_to_name[obj_id]
                actions["write"].append(f'write "a placeholder text" {obj_name}')

        # --- Listen ---
        can_listen = False
        area_name = str(getattr(curr_area, "name", "") or "").strip().lower()
        if area_name in {"library", "orrery", "spire"}:
            can_listen = True
        else:
            for obj_id in hand_objects:
                base_id = get_def_id(obj_id)
                obj = None
                if base_id in self.world.objects:
                    obj = self.world.objects[base_id]
                elif obj_id in self.world.writable_instances:
                    obj = self.world.writable_instances[obj_id]
                elif obj_id in self.world.container_instances:
                    obj = self.world.container_instances[obj_id]
                if obj is None:
                    continue
                name = str(getattr(obj, "name", "") or "").lower()
                desc = str(getattr(obj, "description", "") or "").lower()
                oid = str(getattr(obj, "id", "") or "").lower()
                if any(tok in name or tok in desc or tok in oid for tok in ("glass", "lens", "saltglass", "brine")):
                    can_listen = True
                    break
        if can_listen:
            actions["listen"].append("listen saltglass")

        actions["wait"].append("wait")

        return actions

    def parse_action(self, agent: Agent, action_str: str) -> Tuple[Optional[str], Optional[List[str]]]:
        # test: print(f"Parsing action string: {action_str}")
        if action_str is None or action_str.strip() == "":
            return None, None
        s = action_str.strip()
        s_lower = s.lower()
        actions = sorted((a.verb for a in agent.available_actions), key=len, reverse=True)

        action = None
        action_end = None

        for a in actions:
            a_lower = a.lower()
            if s_lower.startswith(a_lower):
              if len(s_lower) == len(a_lower) or s_lower[len(a_lower)].isspace():
                  action = a
                  action_end = len(a_lower)
                  break
            
        if action is None:
            return None, None
        
        remainder = s[action_end:].strip()
        if remainder == "":
            return action, []
        
        try:
            params = shlex.split(remainder)
        except ValueError:  # e.g. unmatched quotes
            return None, None
        
        return action, params

    def verbalize_objects(self, objects: Dict[str, int]) -> str:
        if not objects:
            return "no objects"
        obj_strs = []
        for oid, count in objects.items():
            if oid in self.world.container_instances:
                obj_strs.append(f"{count} {self.world.container_instances[oid].name}")
            elif oid in self.world.writable_instances:
                obj_strs.append(f"{count} {self.world.writable_instances[oid].name}")
            elif oid in self.world.objects:
                obj_strs.append(f"{count} {self.world.objects[oid].name}")
            else:
                # Handle phantom/mirage objects injected by step rules
                sm = self.world.auxiliary.get("salt_mirage", {})
                mirage_name = None
                for m in sm.get("active", {}).values():
                    if m.get("obj_id") == oid:
                        mirage_name = m.get("fake_name")
                        break
                if mirage_name:
                    obj_strs.append(f"{count} {mirage_name}")
        return ", ".join(obj_strs)
    
    def verbalize_npcs(self, npc_ids: str) -> str:
        if not npc_ids:
            return "no one"
        # print({npc_id: self.world.npc_instances[npc_id].level for npc_id in npc_ids})
        return ", ".join([self.world.npc_instances[npc_id].name for npc_id in npc_ids])

    def verbalize_areas(self, areas: List[str]) -> str:
        if not areas:
            return "no neighboring areas"
        return ", ".join(areas)
    
    def get_nearby_agent_names(self, agent: Agent) -> List[str]:
        agent_area = self.curr_agents_state["area"][agent.id]
        return [
            a.name for a in self.agents
            if a.id != agent.id and self.curr_agents_state["area"].get(a.id) == agent_area
        ]

    def verbalize_npcs_and_agents(self, npc_ids: list, agent_names: List[str]) -> str:
        names = [self.world.npc_instances[npc_id].name for npc_id in npc_ids] + agent_names
        if not names:
            return "no one"
        return ", ".join(names)

    def verbalize_obs(self, obs_time: str, obs_current_location: str, neighboring_areas: str, obs_items_in_hand: str, 
        obs_equipped_items_in_limb: str, obs_objects: str, obs_npcs_and_agents: str, obs_agent_hp: int, obs_agent_xp: int,
        obs_agent_level: int, obs_agent_attack: int, obs_agent_defense: int, feedback: str = "",
    ) -> str:
        p = self.person_verbalized
        feedback = feedback.strip() + "\n" if feedback else ""
        return f"""
Current Time: {obs_time}
Current Location: {obs_current_location}

{feedback}
{p['subject_pronoun'].capitalize()} {p['to_be_conjugation']} holding {obs_items_in_hand}.
{p['subject_pronoun'].capitalize()} have equipped {obs_equipped_items_in_limb}.
{p['subject_pronoun'].capitalize()} see {obs_objects} near {p['object_pronoun']}.
{p['subject_pronoun'].capitalize()} see {obs_npcs_and_agents} nearby.

{p['possessive_adjective'].capitalize()} level is {obs_agent_level}.
{p['possessive_adjective'].capitalize()} attack is at {obs_agent_attack}.
{p['possessive_adjective'].capitalize()} defense is at {obs_agent_defense}.
{p['possessive_adjective'].capitalize()} health is at {obs_agent_hp}.
{p['possessive_adjective'].capitalize()} experience is at {obs_agent_xp}.

Neighboring areas: {neighboring_areas}.
"""

    def get_curr_info(self, curr_area_id: str):
        area = self.world.area_instances[curr_area_id]
        area_info = {
            "name": area.name,
            "objects": area.objects,
            "npcs": area.npcs,
            "place": self.world.auxiliary["area_to_place"][curr_area_id],
            "neighbors": [self.world.area_instances[neighbor_area_id].name for neighbor_area_id in area.neighbors.keys()]
        }
        return {
            "time": f"{self.curr_time.year:04d}-{self.curr_time.strftime('%m-%d %H:%M:%S')}",
            "area": area_info
        }

    def reset(self, from_config: bool = False) -> Dict[str, str]:
        if not from_config:
            self.world = World.generate(self.world_definition, seed=self.seed)
            self.steps = -1
            spawn_area = self.world_definition["initializations"]["spawn"]["area"]
            for agent in self.agents:
                self.curr_agents_state["area"][agent.id] = spawn_area
                self.curr_agents_state["areas_visited"][agent.id] = [spawn_area]
                self.curr_agents_state["objects_crafted"][agent.id] = {}
                self.curr_agents_state["objects_traded"][agent.id] = {}
                self.curr_agents_state["npcs_killed"][agent.id] = []
                self.curr_agents_state["unique_npcs_killed"][agent.id] = []
                self.curr_agents_state["objects_acquired"][agent.id] = []

            self.curr_agents_state.setdefault("tutorial_passed", {})
            self.curr_agents_state.setdefault("tutorial_stage", {})
            for agent in self.agents:
                self.curr_agents_state["tutorial_passed"].setdefault(agent.id, False)
                self.curr_agents_state["tutorial_stage"].setdefault(agent.id, 0)

            self.curr_agents_state.setdefault("main_quest_progress", {})
            self.curr_agents_state.setdefault("side_quest_current_task", {})
            self.curr_agents_state.setdefault("side_quest_completed_tasks", {})
            self.curr_agents_state.setdefault("side_quest_task_progress", {})
            for agent in self.agents:
                self.curr_agents_state["main_quest_progress"].setdefault(agent.id, None)
                self.curr_agents_state["side_quest_current_task"].setdefault(agent.id, None)
                self.curr_agents_state["side_quest_completed_tasks"].setdefault(agent.id, [])
                self.curr_agents_state["side_quest_task_progress"].setdefault(agent.id, {})

            # combat tracking: {agent_id: {npc_id: {"rhythm_index": int, "area_id": str}}}
            self.curr_agents_state.setdefault("active_combats", {})
            # agent defending state: {agent_id: bool} - True if agent is defending this step
            self.curr_agents_state.setdefault("agent_defending", {})
            # agent stamina: {agent_id: float} - 1.0 = full, decreases with consecutive attacks
            self.curr_agents_state.setdefault("agent_stamina", {})
            # agent consecutive attacks: {agent_id: int} - count of consecutive attack actions
            self.curr_agents_state.setdefault("agent_consecutive_attacks", {})
            # salt glaze state
            self.curr_agents_state.setdefault("salt_glaze", {})
            for agent in self.agents:
                self.curr_agents_state["active_combats"].setdefault(agent.id, {})
                self.curr_agents_state["agent_defending"].setdefault(agent.id, False)
                self.curr_agents_state["agent_stamina"].setdefault(agent.id, 1.0)
                self.curr_agents_state["agent_consecutive_attacks"].setdefault(agent.id, 0)
                self.curr_agents_state["salt_glaze"].setdefault(agent.id, {
                    "active": False,
                    "obj_id": None,
                    "stat": None,
                    "amount": 0,
                })

            # dynamic world expansion state
            online_expansion = self.world_definition.get("features", {}).get("online_expansion", False)
            if online_expansion:
                import os
                if not os.environ.get("OPENAI_API_KEY"):
                    self.logger.warning(
                        "online_expansion is enabled but OPENAI_API_KEY is not set. "
                        "Disabling online_expansion to avoid runtime errors.\n"
                    )
                    self.world_definition.setdefault("features", {})["online_expansion"] = False
            self.curr_agents_state.setdefault("world_expansion", {
                "pending": False,
                "count": 0,
                "total_areas": len(self.world.area_instances),
            })

            self.update_config(update_env_config=True, update_file=True)
        else:
            self.world = World.from_dict(self.config["world"])

            self.curr_agents_state.setdefault("tutorial_passed", {})
            self.curr_agents_state.setdefault("tutorial_stage", {})
            for agent in self.agents:
                self.curr_agents_state["tutorial_passed"].setdefault(agent.id, False)
                self.curr_agents_state["tutorial_stage"].setdefault(agent.id, 0)

            self.curr_agents_state.setdefault("main_quest_progress", {})
            self.curr_agents_state.setdefault("side_quest_current_task", {})
            self.curr_agents_state.setdefault("side_quest_completed_tasks", {})
            self.curr_agents_state.setdefault("side_quest_task_progress", {})
            for agent in self.agents:
                self.curr_agents_state["main_quest_progress"].setdefault(agent.id, None)
                self.curr_agents_state["side_quest_current_task"].setdefault(agent.id, None)
                self.curr_agents_state["side_quest_completed_tasks"].setdefault(agent.id, [])
                self.curr_agents_state["side_quest_task_progress"].setdefault(agent.id, {})

            # combat tracking for resuming from config
            self.curr_agents_state.setdefault("active_combats", {})
            self.curr_agents_state.setdefault("agent_defending", {})
            self.curr_agents_state.setdefault("agent_stamina", {})
            self.curr_agents_state.setdefault("agent_consecutive_attacks", {})
            self.curr_agents_state.setdefault("salt_glaze", {})
            for agent in self.agents:
                self.curr_agents_state["active_combats"].setdefault(agent.id, {})
                self.curr_agents_state["agent_defending"].setdefault(agent.id, False)
                self.curr_agents_state["agent_stamina"].setdefault(agent.id, 1.0)
                self.curr_agents_state["agent_consecutive_attacks"].setdefault(agent.id, 0)
                self.curr_agents_state["salt_glaze"].setdefault(agent.id, {
                    "active": False,
                    "obj_id": None,
                    "stat": None,
                    "amount": 0,
                })

            # dynamic world expansion: initialise / reset pending flag
            # (background thread does not survive a process restart)
            online_expansion = self.world_definition.get("features", {}).get("online_expansion", False)
            if online_expansion:
                import os
                if not os.environ.get("OPENAI_API_KEY"):
                    self.logger.warning(
                        "online_expansion is enabled but OPENAI_API_KEY is not set. "
                        "Disabling online_expansion to avoid runtime errors.\n"
                    )
                    self.world_definition.setdefault("features", {})["online_expansion"] = False
            expansion = self.curr_agents_state.setdefault("world_expansion", {
                "pending": False,
                "count": 0,
                "total_areas": len(self.world.area_instances),
            })
            expansion["pending"] = False

        self._intro = (f"\n{self.person_verbalized['subject_pronoun']} awake in a world unknown. "
            "There's a paper written in an ancient script: 'The past is gone, the future unwritten. "
            "Every step, every decision, shapes what lies ahead. "
            "Adventure, danger, and discovery await. "
            "Step forward and let the journey begin.' "
            "Suddenly, the paper bursts into flames, leaving no trace behind.\n")

        obs_str = {}
        self.obs = {}
        for agent in self.agents:
            if from_config:
                for agent_data in self.config["agents"]:
                    if agent_data["id"] == agent.id:
                        agent.from_dict(agent_data, self.world.container_instances)
                        break
                else:
                    self.logger.error(f"Corrupted config: Agent ID {agent.id} not found in config agents.")
                    raise ValueError(f"Agent ID {agent.id} not found in config agents.")

            current_info = self.get_curr_info(spawn_area) if not from_config else self.get_curr_info(self.curr_agents_state["area"][agent.id])
            obs_str = self.verbalize_obs(current_info["time"],
                                                f"{self.world.place_instances[current_info['area']['place']].name}, "
                                                f"{current_info['area']['name']}",
                                                self.verbalize_areas(current_info["area"]["neighbors"]),
                                                self.verbalize_objects(agent.items_in_hands),
                                                self.verbalize_objects(agent.equipped_items_in_limb),
                                                self.verbalize_objects(current_info["area"]["objects"]),
                                                self.verbalize_npcs_and_agents(current_info["area"]["npcs"], self.get_nearby_agent_names(agent)),
                                                agent.hp,
                                                agent.xp,
                                                agent.level,
                                                agent.attack,
                                                agent.defense
                                                )
            self.obs[agent.id] = {"step": self.steps, "text": obs_str} 
            if self.enable_obs_valid_actions:
                self.obs[agent.id]["valid_actions"] = self.get_all_valid_actions(agent)

        if self.save_dep_graph_steps is not None:
            self.dep_tracker = DependencyTracker(strict=False)
            self.dep_tracker.bootstrap_from_state(self, self.world)
        else:
            self.dep_tracker = None

        if self.steps == -1: # bootstrap step, do not delete
            self.step({agent.id: "wait" for agent in self.agents})

        return self.obs
    
    def export_depdency_graph(self, prefix: str = None):
        if prefix is None:
            prefix = f"{self.run_dir}/dep_graphs/dependency_graph_step_{self.steps}"
        os.makedirs(os.path.dirname(prefix), exist_ok=True)

        dep_dict = self.dep_tracker.to_dict()

        out_path = prefix + "_dep.json"
        with open(out_path, "w", encoding="utf-8") as f:
            import json
            json.dump(dep_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Dependency graph exported to {out_path}")

    def update_config(self, update_env_config: Optional[bool] = True,
                      hardware: Optional[Dict] = None, update_file: bool = True, cumulative: bool = False):
        if update_env_config:
            self.config["step"] = self.steps
            self.config["curr_time"] = f"{self.curr_time.year:04d}-{self.curr_time.strftime('%m-%d %H:%M:%S')}"
            self.config["curr_agents_state"] = self.curr_agents_state
            self.config["scores"] = self.scores
            self.config["world"] = self.world.to_dict() if self.world is not None else None
            self.config["agents"] = [a.to_dict() for a in self.agents]
        if hardware is not None:
            self.config["hardware"] = hardware
        if update_file:
            if cumulative:
                with open(self.config_path, "a") as f:
                    f.write(json.dumps(self.config) + "\n")
            else:
                atomic_write(self.config_path, json.dumps(self.config, indent=2))

    def update_scores(self, reward: Dict):
        for agent_id in [a.id for a in self.agents]:
            if agent_id not in self.scores:
                self.scores[agent_id] = {}
            for key, value in reward[agent_id].__dict__.items():
                if key not in self.scores[agent_id]:
                    self.scores[agent_id][key] = 0
                self.scores[agent_id][key] += value

    def step(self, action_strs: Dict[str, str]):
        step_index = self.steps

        feedback: Dict[str, str] = {a.id: "" for a in self.agents}
        info: Dict[str, Dict[str, Any]] = {"step_invalid_action": {a.id: False for a in self.agents}}

        res = RuleResult(feedback=feedback, info_flags=info)
        if not hasattr(res, "events") or res.events is None:
            res.events = []

        # skip dependency tracking while any agent is still in the tutorial
        _tut_passed = self.curr_agents_state.get("tutorial_passed", {})
        _tutorial_active = any(
            not _tut_passed.get(a.id, True) for a in self.agents
        )

        # snapshot previous state for reward diffs
        prev_state = {
            "areas_visited": {
                aid: visited.copy()
                for aid, visited in self.curr_agents_state["areas_visited"].items()
            },
            "npcs_killed": {
                aid: kills.copy()
                for aid, kills in self.curr_agents_state["npcs_killed"].items()
            },
            "unique_npcs_killed": {
                aid: unique_kills.copy()
                for aid, unique_kills in self.curr_agents_state["unique_npcs_killed"].items()
            },
            "objects_crafted": {
                aid: crafted.copy()
                for aid, crafted in self.curr_agents_state["objects_crafted"].items()
            },
            "objects_traded": {
                aid: traded.copy()
                for aid, traded in self.curr_agents_state["objects_traded"].items()
            },
            "objects_acquired": {
                aid: acquired.copy()
                for aid, acquired in self.curr_agents_state["objects_acquired"].items()
            },
        }

        terminated = False

        # apply agent-level actions via rule engine
        for agent in self.agents:
            action_str = action_strs[agent.id]
            action, params = self.parse_action(agent, action_str)
            hint = f"{agent.name}: {action}"
            if params:
                hint += " " + " ".join(str(p) for p in params)

            if action is None:
                msg = (
                    f"Invalid action. Failed to parse action from input: '{action_str}'. "
                    f"The available actions are {', '.join([a.verb for a in agent.available_actions])}. \n"
                    f"Default to wait.\n"
                )
                res.add_feedback(agent.id, msg)
                self.logger.warning(
                    f"❌ [Step {self.steps}] {agent.name} provided an unparsable action: '{action_str}'"
                )
                res.info_flags["step_invalid_action"][agent.id] = True

                action = "wait"
                params = []

            if self.steps != -1:
                self.logger.info(f"✅ [Step {self.steps}] {agent.name} performs action {action} with params {params}")
                
            ctx = RuleContext(
                env=self,
                world=self.world,
                agent=agent,
                action=action,
                params=params,
                step_index=step_index,
            )

            # event slicing for dependency tracking
            ev_start = len(res.events)
            self.action_rule_engine.dispatch(ctx, res)
            new_events = res.events[ev_start:]

            if hasattr(self, "dep_tracker") and self.dep_tracker is not None and not _tutorial_active:
                self.dep_tracker.process_rule_result(
                    step=step_index,
                    actor=agent.id,
                    rule=str(action),
                    events=new_events,
                    hint=hint,
                )

        # apply env level step rules
        ctx = RuleContext(
            env=self,
            world=self.world,
            agent=None,
            action="",
            params=[],
            step_index=self.steps,
        )
        for rule in self.step_rules:
            ev_start = len(res.events)
            rule.apply(ctx, res)
            new_events = res.events[ev_start:]
            hint_events = [ev for ev in new_events if ev.type == "dep_tracker_hint"]
            hint = "\n".join([ev.data.get("hint", "") for ev in hint_events])

            if hasattr(self, "dep_tracker") and self.dep_tracker is not None and not _tutorial_active:
                self.dep_tracker.process_rule_result(
                    step=step_index,
                    actor="env",
                    rule=rule.name,
                    events=new_events,
                    hint=hint,
                )
            
        bootstrap = (self.steps == -1) # fastforward from step -1 to step 0
        self.steps += 1
        if not bootstrap:
            self.curr_time += timedelta(seconds=self.step_delta_time)
        reward = self.reward_function.compute(self, prev_state, res) if not bootstrap else {a.id: RewardBreakdown() for a in self.agents}
        intro = self._intro if bootstrap and self._intro else ""

        self.obs = {}
        for agent in self.agents:
            agent_area = self.curr_agents_state["area"][agent.id]
            current_info = self.get_curr_info(agent_area) 

            obs_text = intro + self.verbalize_obs(
                current_info["time"],
                f"{self.world.place_instances[current_info['area']['place']].name}, {current_info['area']['name']}",
                self.verbalize_areas(current_info["area"]["neighbors"]),
                self.verbalize_objects(agent.items_in_hands),
                self.verbalize_objects(agent.equipped_items_in_limb),
                self.verbalize_objects(current_info["area"]["objects"]),
                self.verbalize_npcs_and_agents(current_info["area"]["npcs"], self.get_nearby_agent_names(agent)),
                agent.hp,
                agent.xp,
                agent.level,
                agent.attack,
                agent.defense,
                res.feedback[agent.id],
            )

            self.obs[agent.id] = {"step": self.steps, "text": obs_text}
            if self.enable_obs_valid_actions: # cannot skip even in bootstrap step because RandomAgent requires it
                self.obs[agent.id]["valid_actions"] = self.get_all_valid_actions(agent)
            
            if not bootstrap and self.save_dep_graph_steps is not None and self.steps % self.save_dep_graph_steps == 0:
                self.export_depdency_graph()

        # Terminate the game when any agent completes the main quest
        if any(e.type == "main_quest_complete" for e in res.events):
            terminated = True
            for agent in self.agents:
                self.obs[agent.id]["text"] += (
                    "\n=== CONGRATULATIONS! You have completed the main quest and passed the game! ===\n"
                )

        return self.obs, reward, terminated, res.info_flags
