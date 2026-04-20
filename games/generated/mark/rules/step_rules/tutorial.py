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


class TutorialRoomStepRule(BaseStepRule):
    name = "tutorial_room_step"
    description = "Guides agents through a multi-step tutorial in a dedicated Tutorial room connected to spawn."
    priority = 1

    # tutorial-only entities. Unlike main/side quest entities, these are not added to world_definition.
    # They are instantiated into the runtime World and later removed.
    required_objects: List[Dict] = [
        {"type": "object", "id": "obj_small_pouch"},
        {"type": "object", "id": "obj_paper"},
        {"type": "object", "id": "obj_coin"},
        {"type": "object", "id": "obj_pen"},
        {"type": "object", "id": "obj_oak_log"},
        {"type": "object", "id": "obj_iron_bar"},
        {"type": "object", "id": "obj_workbench"},
        {"type": "object", "id": "obj_furnace"},
        {"type": "object", "id": "obj_key"},
        {"type": "object", "id": "obj_copper_bar"},
        {"type": "object", "id": "obj_fiber_bundle"},
    ]

    required_npcs: List[Dict] = [
        {
            "type": "npc",
            "id": "npc_tutorial_merchant",
            "name": "selene_wyatt",
            "enemy": False,
            "unique": True,
            "role": "merchant",
            "level": 1,
            "attack_power": 0,
            "hp": 1,
            "coins": 1000,
            "slope_hp": 0,
            "slope_attack_power": 0,
            "description": "a calm merchant with a tidy ledger and an unhurried gaze.",
        },
        {
            "type": "npc",
            "id": "npc_tutorial_enemy",
            "name": "goblin_raider",
            "enemy": True,
            "unique": False,
            "role": "enemy",
            "level": 1,
            "attack_power": 1,
            "hp": 1,
            "coins": 0,
            "slope_hp": 0,
            "slope_attack_power": 0,
            "description": "A crude raider with a chipped dagger and a bad temper.",
            "dialogue": "",
        },
    ]

    # to change tutorial content, edit these blocks only
    STEPS: List[dict] = [
        {
            "id": "pick_bag",
            "setup": {"instances_on_ground_if_missing": ["bag"]},
            "instruction": (
                "Welcome to the tutorial of this game!\n"
                "You'll be instructed to learn different actions that you are available to take.\n"
                "Tutorial Step 1 — Pick up the bag.\n"
                "Required action: `pick up {bag}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_picked_up", "base_obj": "obj_small_pouch"},
                {"kind": "state", "type": "in_hands_base", "base_obj": "obj_small_pouch"},
            ],
        },
        {
            "id": "equip_bag_inventory",
            "setup": {},
            "instruction": (
                "Tutorial Step 2 — Equip the bag as inventory.\n"
                "Required action: `equip {bag}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "item_equipped", "base_obj": "obj_small_pouch"},
                {"kind": "state", "type": "inventory_container_is_base", "base_obj": "obj_small_pouch"},
            ],
        },
        {
            "id": "inspect_inventory",
            "setup": {},
            "instruction": (
                "Tutorial Step 3 — Inspect your inventory.\n"
                "Required action: `inspect inventory`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "inventory_inspected"},
            ],
        },
        {
            "id": "unequip_bag_to_hand",
            "setup": {},
            "instruction": (
                "Tutorial Step 4 — Unequip the bag.\n"
                "Required action: `unequip {bag}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "item_unequipped", "base_obj": "obj_small_pouch"},
                {"kind": "state", "type": "bag_in_hand_and_no_inventory"},
            ],
        },
        {
            "id": "store_coin_into_held_bag",
            "setup": {"ensure_ground_base_if_agent_missing": {"obj_coin": 1}},
            "instruction": (
                "Tutorial Step 5 — Store a coin into the bag.\n"
                "Required action: `store 1 {coin} {bag}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_stored", "base_obj": "obj_coin", "data": {"container": "{bag}"}},
            ],
        },
        {
            "id": "inspect_bag",
            "setup": {},
            "instruction": (
                "Tutorial Step 6 — Inspect the bag.\n"
                "Required action: `inspect {bag}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "container_inspected", "base_obj": "obj_small_pouch"},
            ],
        },
        {
            "id": "take_out_coin_from_bag",
            "setup": {},
            "instruction": (
                "Tutorial Step 7 — Take the coin out of the bag.\n"
                "Required action: `take out {coin} {bag}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_taken_out", "base_obj": "obj_coin", "data": {"from": "{bag}"}},
                {"kind": "state", "type": "in_hands_base", "base_obj": "obj_coin"},
            ],
        },
        {
            "id": "drop_coin",
            "setup": {},
            "instruction": (
                "Tutorial Step 8 — Drop the coin.\n"
                "Required action: `drop {coin}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_dropped", "base_obj": "obj_coin"},
            ],
        },
        {
            "id": "equip_bag_inventory_again",
            "setup": {"instances_on_ground_if_missing": ["bag"]},
            "instruction": (
                "Tutorial Step 9 — Equip the bag as inventory again.\n"
                "Required action: `equip {bag}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "item_equipped", "base_obj": "obj_small_pouch"},
                {"kind": "state", "type": "inventory_container_is_base", "base_obj": "obj_small_pouch"},
            ],
        },
        {
            "id": "store_coin_into_inventory",
            "setup": {"ensure_ground_base_if_agent_missing": {"obj_coin": 1}},
            "instruction": (
                "Tutorial Step 10 — Store a coin into your inventory.\n"
                "Now you may refer to {bag} as 'inventory' in actions.\n"
                "Required action: `store 1 {coin} inventory`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_stored", "base_obj": "obj_coin", "data": {"container": "inventory"}},
            ],
        },
        {
            "id": "discard_coin_from_inventory",
            "setup": {},
            "instruction": (
                "Tutorial Step 11 — Discard a coin from your inventory.\n"
                "Required action: `discard 1 {coin} inventory`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_discarded", "base_obj": "obj_coin", "data": {"from": "inventory"}},
            ],
        },
        {
            "id": "pick_pen",
            "setup": {"ensure_ground_base_if_agent_missing": {"obj_pen": 1}},
            "instruction": (
                "Tutorial Step 12 — Pick up a pen.\n"
                "Required action: `pick up {pen}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_picked_up", "base_obj": "obj_pen"},
                {"kind": "state", "type": "in_hands_base", "base_obj": "obj_pen"},
            ],
        },
        {
            "id": "pick_writable_instance",
            "setup": {"instances_on_ground_if_missing": ["writable"]},
            "instruction": (
                "Tutorial Step 13 — Pick up the writable instance.\n"
                "Required action: `pick up {writable}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_picked_up", "base_obj": "obj_paper"},
                {"kind": "state", "type": "in_hands_base", "base_obj": "obj_paper"},
            ],
        },
        {
            "id": "write_writable",
            "setup": {},
            "instruction": (
                "Tutorial Step 14 — Write on the writable.\n"
                "Required action: `write \"hello\" {writable}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "writable_written", "base_obj": "obj_paper"},
            ],
        },
        {
            "id": "trade_prep_get_2_coins",
            "setup": {"ensure_ground_base_if_agent_base_count_lt": {"obj_coin": 2}},
            "instruction": (
                "Tutorial Step 15 — Trading prep.\n"
                "Buying a wood_log costs 2 coins.\n"
                "Required action: `store 2 {coin} inventory`\n"
            ),
            "done_any": [
                {"kind": "state", "type": "base_count_at_least", "base_obj": "obj_coin", "count": 2},
            ],
        },
        {
            "id": "talk_to_merchant",
            "setup": {
                "ensure_npcs": [
                    {
                        "kind": "merchant",
                        "stock": {"obj_oak_log": 1},
                        "description": "a calm merchant with a tidy ledger and an unhurried gaze.",
                        "dialogue": (
                            "I can trade with you based on what's in my inventory. "
                        ),
                    }
                ],
            },
            "instruction": (
                "Tutorial Step 16 — Talk to an NPC.\n"
                "Required action: `talk to {merchant}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "npc_talked_to", "data": {"npc_id": "{merchant_id}"}}  # strict match
            ],
        },
        {
            "id": "buy_from_merchant",
            "setup": {
                "ensure_npcs": [
                    {
                        "kind": "merchant",
                        "stock": {
                            "obj_oak_log": 1,
                        },
                    }
                ],
            },
            "instruction": (
                "Tutorial Step 17 — Purchase from a merchant.\n"
                "Required action: `buy 1 {oak_log} {merchant}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_bought", "base_obj": "obj_oak_log"},
            ],
        },
        {
            "id": "inspect_oak_log_crafting",
            "setup": {
                "ensure_npcs": [
                    {
                        "kind": "merchant",
                        "stock": {"obj_oak_log": 1},
                    }
                ],
                "ensure_ground_base_if_agent_missing": {"obj_oak_log": 1},
            },
            "instruction": (
                "Tutorial Step 18 — Inspect non-container objects.\n"
                "Required action: `inspect {oak_log}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_inspected", "base_obj": "obj_oak_log"},
            ],
        },
        {
            "id": "sell_to_merchant_store",
            "setup": {
                "ensure_npcs": [
                    {
                        "kind": "merchant",
                        "stock": {
                            "obj_oak_log": 1,
                        },
                    }
                ],
            },
            "instruction": (
                "Tutorial Step 19 — Empty a hand.\n"
                "Required action: `store 1 {writable} inventory`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_stored", "base_obj": "obj_paper"},
                {"kind": "event", "type": "object_dropped", "base_obj": "obj_paper"},
            ],
        },
        {
            "id": "sell_to_merchant_pick_up",
            "setup": {
                "ensure_npcs": [
                    {
                        "kind": "merchant",
                        "stock": {
                            "obj_oak_log": 1,
                        },
                    }
                ],
            },
            "instruction": (
                "Tutorial Step 20 — Pick up the object to sell.\n"
                "Required action: `pick up {oak_log}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_stored", "base_obj": "obj_oak_log"},
                {"kind": "event", "type": "object_picked_up", "base_obj": "obj_oak_log"},
                {"kind": "state", "type": "base_count_at_least", "base_obj": "obj_oak_log", "count": 1},
            ],
        },
        {
            "id": "sell_to_merchant_sell",
            "setup": {
                "ensure_npcs": [
                    {
                        "kind": "merchant",
                        "stock": {
                            "obj_oak_log": 1,
                        },
                    }
                ],
            },
            "instruction": (
                "Tutorial Step 21 — Sell to a merchant.\n"
                "Required action: `sell 1 {oak_log} {merchant}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_sold", "base_obj": "obj_oak_log"},
            ],
        },
        {
            "id": "npc_attack_tutorial",
            "setup": {
                "ensure_npcs": [{
                    "kind": "enemy",
                    "tune": {"attack_power": 0, "hp": 1},
                }],
            },
            "instruction": (
                "Tutorial Step 22 — Combat an enemy.\n"
                "Required action: `attack {enemy}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "npc_killed"},
            ],
        },
        {
            "id": "craft_prep_key_ingredients_and_stations",
            "setup": {"ensure_ground_base": {"obj_copper_bar": 1, "obj_fiber_bundle": 1, "obj_workbench": 1}},
            "instruction": (
                "Tutorial Step 23 — Craft prep (ingredients + dependencies).\n"
                "I placed:\n"
                "  - {copper_bar} + {fiber_bundle} (ingredients)\n"
                "  - {workbench} (dependency)\n\n"
                "Required action: `pick up {copper_bar}`\n"
            ),
            "done_any": [
                {"kind": "state", "type": "in_hands_base", "base_obj": "obj_copper_bar"},
                {"kind": "state", "type": "inventory_has_base", "base_obj": "obj_copper_bar"},
                {"kind": "state", "type": "held_containers_have_base", "base_obj": "obj_copper_bar"},
                {"kind": "event", "type": "object_picked_up", "base_obj": "obj_copper_bar"},
                {"kind": "event", "type": "object_stored", "base_obj": "obj_copper_bar"},
            ],
        },
                {
            "id": "craft_prep_key_ingredients_and_stations",
            "setup": {"ensure_ground_base": {"obj_copper_bar": 1, "obj_fiber_bundle": 1, "obj_workbench": 1}},
            "instruction": (
                "Tutorial Step 24 — Craft prep (ingredients + dependencies).\n"
                "Required action: `drop {pen}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_dropped", "base_obj": "obj_pen"}
            ],
        },
        {
            "id": "craft_prep_key_ingredients_and_stations",
            "setup": {"ensure_ground_base": {"obj_copper_bar": 1, "obj_fiber_bundle": 1, "obj_workbench": 1}},
            "instruction": (
                "Tutorial Step 25 — Craft prep (ingredients + dependencies).\n"
                "Required action: `pick up {fiber_bundle}`\n"
            ),
            "done_any": [
                {"kind": "state", "type": "in_hands_base", "base_obj": "obj_fiber_bundle"},
                {"kind": "state", "type": "inventory_has_base", "base_obj": "obj_fiber_bundle"},
                {"kind": "state", "type": "held_containers_have_base", "base_obj": "obj_fiber_bundle"},
                {"kind": "event", "type": "object_picked_up", "base_obj": "obj_fiber_bundle"},
                {"kind": "event", "type": "object_stored", "base_obj": "obj_fiber_bundle"},
            ],
        },
        {
            "id": "craft_key",
            "setup": {"ensure_ground_base": {"obj_workbench": 1}},
            "instruction": (
                "Tutorial Step 26 — Craft a key.\n"
                "Required action: `craft 1 {key}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_crafted", "base_obj": "obj_key"},
            ],
        },
        {
            "id": "enter_with_key_in_hand_pick_up",
            "setup": {"lock_exit": True},
            "instruction": (
                "Tutorial Step 27 — Pick up the key.\n"
                "Required action: `pick up {key}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "object_picked_up", "base_obj": "obj_key"},
            ],
        },
        {
            "id": "enter_with_key_in_hand_enter",
            "setup": {"lock_exit": True},
            "instruction": (
                "Tutorial Step 28 — Enter an area.\n"
                "Required action: `enter {exit}`\n"
            ),
            "done_any": [
                {"kind": "event", "type": "area_entered"},
                {"kind": "state", "type": "area_changed_from_spawn"},
            ],
        },
    ]

    def __init__(self) -> None:
        super().__init__()
        self._initialized = False
        self._stage: Dict[str, int] = {}
        self._misses: Dict[str, int] = {}

        self._spawn_area_id: Optional[str] = None
        self._tutorial_place_id: Optional[str] = None
        self._tutorial_area_id: Optional[str] = None
        self._exit_area_id: Optional[str] = None
        self._exit_area_name: Optional[str] = None
        self._exit_locked: bool = False

        self._bag_inst_id: Optional[str] = None
        self._bag_name: Optional[str] = None
        self._writable_inst_id: Optional[str] = None
        self._writable_name: Optional[str] = None

        self._npc_setup_stage_done: Dict[str, int] = {}
        self._merchant_npc_inst_id: Optional[str] = None
        self._merchant_npc_inst_name: Optional[str] = None
        self._enemy_npc_inst_id: Optional[str] = None
        self._enemy_npc_inst_name: Optional[str] = None

        self.agents_passed_tutorial: Dict[str, bool] = {}

        self._tutorial_obj_ids: Set[str] = set()
        self._tutorial_npc_ids: Set[str] = set()
        self._removed: bool = False

    def _emit_completion_feedback(self, aid: str, res: RuleResult) -> None:
        from games.base.agent import Agent as base_agent
        from games.generated.mark.agent import Agent as gen_agent

        base_actions = base_agent(None, None).available_actions
        gen_actions = gen_agent(None, None).available_actions

        res.add_feedback(aid, "Tutorial Finished. You have exited to spawn.\n")

        extra_actions = [a for a in gen_actions if a.name not in {ba.name for ba in base_actions}]
        if extra_actions:
            actions_formatted = [
                f"- {action.verb} " + " ".join(f"<{p}>" for p in action.params)
                for action in extra_actions
            ]
            res.add_feedback(
                aid,
                "\nIn addition to the actions in the tutorial, there are some other actions available:\n"
                + "\n".join(actions_formatted)
                + "\n\n",
            )

        res.events.append(Event(
            type="tutorial_completed",
            agent_id=aid,
            data={"spawn_area_id": self._spawn_area_id},
        ))

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        if "tutorial" not in (env.world_definition.get("custom_events") or []):
            return

        spawn_area_id = ((env.world_definition.get("initializations") or {}).get("spawn") or {}).get("area")
        if not spawn_area_id:
            return

        env.curr_agents_state.setdefault("tutorial_passed", {})
        env.curr_agents_state.setdefault("tutorial_stage", {})
        for a in env.agents:
            env.curr_agents_state["tutorial_passed"].setdefault(a.id, False)
            env.curr_agents_state["tutorial_stage"].setdefault(a.id, 0)

        tutorial_room_aux = world.auxiliary.get("tutorial_room") or {}
        tutorial_removed = bool(tutorial_room_aux.get("removed", False))
        all_passed_persisted = bool(env.agents) and all(
            env.curr_agents_state["tutorial_passed"].get(a.id, False) for a in env.agents
        )

        if tutorial_removed or all_passed_persisted:
            self._removed = True
            self._initialized = True
            return

        if not self._initialized:
            tut_place_id = tutorial_room_aux.get("place_id")
            tut_area_id = tutorial_room_aux.get("area_id")
            spawn_area_id = ((env.world_definition.get("initializations") or {}).get("spawn") or {}).get("area")
            if (
                tut_place_id
                and tut_area_id
                and tut_place_id in world.place_instances
                and tut_area_id in world.area_instances
                and spawn_area_id
                and spawn_area_id in world.area_instances
            ):
                self._spawn_area_id = spawn_area_id
                self._tutorial_place_id = tut_place_id
                self._tutorial_area_id = tut_area_id
                self._exit_area_id = self._spawn_area_id
                self._exit_area_name = world.area_instances[self._spawn_area_id].name

                area = world.area_instances[self._tutorial_area_id]
                if self._bag_inst_id is None:
                    for oid in area.objects.keys():
                        if oid in world.container_instances and get_def_id(oid) == "obj_small_pouch":
                            self._bag_inst_id = oid
                            self._bag_name = world.container_instances[oid].name
                            break
                if self._bag_inst_id is None:
                    for oid, inst in world.container_instances.items():
                        if get_def_id(oid) == "obj_small_pouch":
                            self._bag_inst_id = oid
                            self._bag_name = inst.name
                            break
                if self._writable_inst_id is None:
                    for oid in area.objects.keys():
                        if oid in world.writable_instances and get_def_id(oid) == "obj_paper":
                            self._writable_inst_id = oid
                            self._writable_name = world.writable_instances[oid].name
                            break
                if self._writable_inst_id is None:
                    for oid, inst in world.writable_instances.items():
                        if get_def_id(oid) == "obj_paper":
                            self._writable_inst_id = oid
                            self._writable_name = inst.name
                            break

                self._initialized = True
            else:
                self._bootstrap(env, world, res)

        if not (self._spawn_area_id and self._tutorial_area_id):
            return

        for agent in env.agents:
            aid = agent.id

            if aid not in self.agents_passed_tutorial:
                self.agents_passed_tutorial[aid] = bool(env.curr_agents_state["tutorial_passed"].get(aid, False))
            if self.agents_passed_tutorial[aid]:
                continue

            stage = self._stage.setdefault(aid, int(env.curr_agents_state["tutorial_stage"].get(aid, 0)))

            if env.curr_agents_state["area"][aid] != self._tutorial_area_id:
                if stage == len(self.STEPS) - 1:
                    last_step = self.STEPS[stage]
                    aevents = [e for e in res.events if getattr(e, "agent_id", None) == aid]
                    if self._is_step_done(env, world, agent, last_step, aevents):
                        self._stage[aid] = stage + 1
                        env.curr_agents_state["tutorial_stage"][aid] = self._stage[aid]
                        self._misses[aid] = 0

                        self.agents_passed_tutorial[aid] = True
                        env.curr_agents_state["tutorial_passed"][aid] = True
                        self._emit_completion_feedback(aid, res)
                        continue

                if env.curr_agents_state["area"][aid] == self._spawn_area_id:
                    self.agents_passed_tutorial[aid] = True
                    env.curr_agents_state["tutorial_passed"][aid] = True
                continue

            if stage >= len(self.STEPS):
                continue

            step = self.STEPS[stage]
            self._apply_setup(env, world, agent, res, step, stage)
            res.events.append(Event(
                type="dep_tracker_hint",
                agent_id=aid,
                data={"hint": f"{agent.name}: tutorial step {stage}"},
            ))

            world.auxiliary.setdefault("tutorial_room", {})
            world.auxiliary["tutorial_room"].update({
                "place_id": self._tutorial_place_id,
                "area_id": self._tutorial_area_id,
            })

            aevents = [e for e in res.events if getattr(e, "agent_id", None) == aid]
            done = self._is_step_done(env, world, agent, step, aevents)

            if done:
                self._stage[aid] = stage + 1
                env.curr_agents_state["tutorial_stage"][aid] = self._stage[aid]
                self._misses[aid] = 0

                res.add_feedback(aid, f"\nTutorial Step {stage + 1} complete.\n")
                res.events.append(Event(
                    type="tutorial_stage_advanced",
                    agent_id=aid,
                    data={"from": stage, "to": stage + 1},
                ))

                if self._stage[aid] < len(self.STEPS):
                    next_step = self.STEPS[self._stage[aid]]
                    self._apply_setup(env, world, agent, res, next_step, self._stage[aid])
                    res.add_feedback(aid, self._render_instruction(world, next_step))
                else:
                    self.agents_passed_tutorial[aid] = True
                    env.curr_agents_state["tutorial_passed"][aid] = True
                    self._emit_completion_feedback(aid, res)
                continue

            self._misses[aid] = self._misses.get(aid, 0) + 1
            if self._misses[aid] > 1:
                res.add_feedback(aid, "You are not following the tutorial instructions.\n")
            res.add_feedback(aid, self._render_instruction(world, step))

        if (
            not self._removed
            and env.agents
            and all(self.agents_passed_tutorial.get(a.id, False) for a in env.agents)
        ):
            self._cleanup_and_remove(env, world, res)

    def _bootstrap(self, env, world, res: RuleResult) -> None:
        self._spawn_area_id = ((env.world_definition.get("initializations") or {}).get("spawn") or {}).get("area")
        if not self._spawn_area_id or self._spawn_area_id not in world.area_instances:
            return

        self._tutorial_place_id, self._tutorial_area_id = self._ensure_tutorial_room(world)
        if not self._tutorial_area_id:
            return

        self._exit_area_id = self._spawn_area_id
        self._exit_area_name = world.area_instances[self._spawn_area_id].name

        passed = env.curr_agents_state.get("tutorial_passed") or {}
        for agent in env.agents:
            if passed.get(agent.id, False):
                continue
            env.curr_agents_state["area"][agent.id] = self._tutorial_area_id

        self._bag_inst_id, self._bag_name = self._create_instance(world, "obj_small_pouch")
        self._writable_inst_id, self._writable_name = self._create_instance(world, "obj_paper")

        if self._bag_inst_id:
            self._tutorial_obj_ids.add(self._bag_inst_id)
        if self._writable_inst_id:
            self._tutorial_obj_ids.add(self._writable_inst_id)

        if self._bag_name and self._bag_inst_id:
            world.auxiliary.setdefault("obj_name_to_id", {})[self._bag_name] = self._bag_inst_id
        if self._writable_name and self._writable_inst_id:
            world.auxiliary.setdefault("obj_name_to_id", {})[self._writable_name] = self._writable_inst_id

        aux = world.auxiliary.setdefault("obj_name_to_id", {})
        for base_id in ("obj_coin", "obj_pen", "obj_iron_bar", "obj_workbench", "obj_furnace", "obj_key", "obj_oak_log", "obj_copper_bar", "obj_fiber_bundle"):
            if base_id in world.objects:
                aux[world.objects[base_id].name] = base_id

        world.auxiliary.setdefault("tutorial_room", {})
        world.auxiliary["tutorial_room"].update({
            "place_id": self._tutorial_place_id,
            "area_id": self._tutorial_area_id,
        })

        self._initialized = True

    def _ensure_tutorial_room(self, world) -> tuple[Optional[str], Optional[str]]:
        if self._tutorial_place_id and self._tutorial_area_id:
            if self._tutorial_place_id in world.place_instances and self._tutorial_area_id in world.area_instances:
                return self._tutorial_place_id, self._tutorial_area_id

        place_id = self._alloc_id(world.place_instances, "place_tutorial")
        area_id = self._alloc_id(world.area_instances, "area_tutorial_room")

        place = Place(type="place", id=place_id, name="Tutorial", unlocked=True, areas=[area_id], neighbors=[])
        area = Area(type="area", id=area_id, name="room", light=True, level=1, objects={}, npcs=[], neighbors={})

        spawn_area = world.area_instances.get(self._spawn_area_id)
        if spawn_area is None:
            return None, None

        area.neighbors[self._spawn_area_id] = Path(to_id=self._spawn_area_id, locked=True, object_to_unlock="obj_key")
        spawn_area.neighbors[area_id] = Path(to_id=area_id, locked=False, object_to_unlock="obj_key")

        world.place_instances[place_id] = place
        world.area_instances[area_id] = area
        world.auxiliary.setdefault("area_to_place", {})[area_id] = place_id

        return place_id, area_id

    @staticmethod
    def _alloc_id(existing: Dict[str, Any], base: str) -> str:
        if base not in existing:
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def _pick_exit(self, world) -> tuple[Optional[str], Optional[str]]:
        if not self._spawn_area_id or self._spawn_area_id not in world.area_instances:
            return None, None
        spawn_area = world.area_instances[self._spawn_area_id]
        for nid in (getattr(spawn_area, "neighbors", {}) or {}).keys():
            name = world.area_instances[nid].name if nid in world.area_instances else None
            return nid, name
        return None, None

    def _create_instance(self, world, base_id: str) -> tuple[Optional[str], Optional[str]]:
        if base_id not in world.objects:
            return None, None
        obj_def = world.objects[base_id]
        cat = getattr(obj_def, "category", None)
        usage = getattr(obj_def, "usage", None)
        if (cat != "container") and (usage != "writable"):
            return None, None

        aux = world.auxiliary
        cnt_key = "container_id_to_count" if cat == "container" else "writable_id_to_count"
        aux.setdefault(cnt_key, {})
        aux[cnt_key].setdefault(base_id, 0)

        idx = int(aux[cnt_key][base_id])
        inst = obj_def.create_instance(idx)
        aux[cnt_key][base_id] = idx + 1

        if cat == "container":
            world.container_instances[inst.id] = inst
        else:
            world.writable_instances[inst.id] = inst

        aux.setdefault("obj_name_to_id", {})[inst.name] = inst.id
        return inst.id, inst.name

    def _render_instruction(self, world, step: dict) -> str:
        bag = self._bag_name or "small_pouch"
        writable = self._writable_name or "paper"
        fmt = {
            "bag": bag,
            "writable": writable,
            "exit": self._exit_area_name or "the next area",
            "coin": world.objects["obj_coin"].name if "obj_coin" in world.objects else "coin",
            "pen": world.objects["obj_pen"].name if "obj_pen" in world.objects else "pen",
            "oak_log": world.objects["obj_oak_log"].name if "obj_oak_log" in world.objects else "oak_log",
            "iron_bar": world.objects["obj_iron_bar"].name if "obj_iron_bar" in world.objects else "iron_bar",
            "workbench": world.objects["obj_workbench"].name if "obj_workbench" in world.objects else "workbench",
            "furnace": world.objects["obj_furnace"].name if "obj_furnace" in world.objects else "furnace",
            "key": world.objects["obj_key"].name if "obj_key" in world.objects else "key",
            "copper_bar": world.objects["obj_copper_bar"].name if "obj_copper_bar" in world.objects else "copper_bar",
            "fiber_bundle": world.objects["obj_fiber_bundle"].name if "obj_fiber_bundle" in world.objects else "fiber_bundle",
            "merchant": self._merchant_npc_inst_name or "merchant",
            "enemy": self._enemy_npc_inst_name or "enemy",
        }
        return step["instruction"].format(**fmt) + "\n"

    def _agent_base_count_in_possession(self, env, world, agent, base_obj_id: str) -> int:
        total = 0

        for oid, cnt in (agent.items_in_hands or {}).items():
            if int(cnt) > 0 and get_def_id(oid) == base_obj_id:
                total += int(cnt)

        for oid, cnt in (agent.equipped_items_in_limb or {}).items():
            if int(cnt) > 0 and get_def_id(oid) == base_obj_id:
                total += int(cnt)

        if agent.inventory.container is not None:
            for oid, cnt in (agent.inventory.items or {}).items():
                if int(cnt) > 0 and get_def_id(oid) == base_obj_id:
                    total += int(cnt)

        for oid in list((agent.items_in_hands or {}).keys()):
            if oid in world.container_instances:
                inv = world.container_instances[oid].inventory or {}
                for x, cnt in inv.items():
                    if int(cnt) > 0 and get_def_id(x) == base_obj_id:
                        total += int(cnt)

        return total

    def _apply_setup(self, env, world, agent, res: RuleResult, step: dict, stage: int) -> None:
        if not self._tutorial_area_id:
            return
        area = world.area_instances[self._tutorial_area_id]
        setup = step.get("setup", {})

        if setup.get("lock_exit"):
            self._lock_exit(world)

        aid = agent.id
        do_npc_setup = (self._npc_setup_stage_done.get(aid, -1) != stage)

        if do_npc_setup:
            for spec in (setup.get("ensure_npcs", []) or []):
                kind = (spec or {}).get("kind")
                if not kind:
                    continue

                npc_id = self._ensure_npc_in_area(
                    env, world, area,
                    kind=kind,
                    tune=(spec or {}).get("tune"),
                    stock=(spec or {}).get("stock"),
                    dialogue=(spec or {}).get("dialogue"),
                    description=(spec or {}).get("description"),
                )
                if not npc_id:
                    continue

                if not hasattr(area, "npcs") or area.npcs is None:
                    area.npcs = []
                if npc_id not in area.npcs:
                    area.npcs.append(npc_id)

                inst = world.npc_instances.get(npc_id)
                name = getattr(inst, "name", None) if inst else None
                if kind == "merchant":
                    self._merchant_npc_inst_id = npc_id
                    self._merchant_npc_inst_name = name or npc_id
                elif kind == "enemy":
                    self._enemy_npc_inst_id = npc_id
                    self._enemy_npc_inst_name = name or npc_id

            self._npc_setup_stage_done[aid] = stage

        for obj_id, cnt in (setup.get("ensure_ground_base", {}) or {}).items():
            want = int(cnt)
            have = int(area.objects.get(obj_id, 0))
            if want > have:
                area.objects[obj_id] = want
                self._tutorial_obj_ids.add(obj_id)
                res.track_spawn(agent.id, obj_id, want - have, res.tloc("area", self._tutorial_area_id))

        for obj_id, cnt in (setup.get("ensure_ground_base_if_agent_missing", {}) or {}).items():
            if not self._agent_has_base_anywhere(env, world, agent, obj_id):
                want = int(cnt)
                have = int(area.objects.get(obj_id, 0))
                if want > have:
                    area.objects[obj_id] = want
                    self._tutorial_obj_ids.add(obj_id)
                    res.track_spawn(agent.id, obj_id, want - have, res.tloc("area", self._tutorial_area_id))

        for obj_id, need in (setup.get("ensure_ground_base_if_agent_base_count_lt", {}) or {}).items():
            need = int(need)
            if need <= 0:
                continue
            have_agent = int(self._agent_base_count_in_possession(env, world, agent, obj_id))
            if have_agent >= need:
                continue
            deficit = int(need - have_agent)
            have_ground = int(area.objects.get(obj_id, 0))
            if have_ground < deficit:
                area.objects[obj_id] = deficit
                self._tutorial_obj_ids.add(obj_id)
                res.track_spawn(agent.id, obj_id, deficit - have_ground, res.tloc("area", self._tutorial_area_id))

        for key in (setup.get("instances_on_ground", []) or []):
            inst_id = self._resolve_instance_id(key)
            if inst_id:
                have = int(area.objects.get(inst_id, 0))
                if have < 1:
                    area.objects[inst_id] = 1
                    self._tutorial_obj_ids.add(inst_id)
                    res.track_spawn(agent.id, inst_id, 1, res.tloc("area", self._tutorial_area_id))

        for key in (setup.get("instances_on_ground_if_missing", []) or []):
            inst_id = self._resolve_instance_id(key)
            if inst_id and not self._agent_has_exact(agent, inst_id):
                have = int(area.objects.get(inst_id, 0))
                if have < 1:
                    area.objects[inst_id] = 1
                    self._tutorial_obj_ids.add(inst_id)
                    res.track_spawn(agent.id, inst_id, 1, res.tloc("area", self._tutorial_area_id))

    def _resolve_instance_id(self, key: str) -> Optional[str]:
        if key == "bag":
            return self._bag_inst_id
        if key == "writable":
            return self._writable_inst_id
        return None

    def _lock_exit(self, world) -> None:
        if self._exit_locked:
            return
        if not (self._tutorial_area_id and self._spawn_area_id):
            return
        tut_area = world.area_instances[self._tutorial_area_id]
        path = (getattr(tut_area, "neighbors", {}) or {}).get(self._spawn_area_id)
        if path is not None and hasattr(path, "locked"):
            path.locked = True
            rev_area = world.area_instances.get(self._spawn_area_id)
            if rev_area is not None:
                rev_path = (getattr(rev_area, "neighbors", {}) or {}).get(self._tutorial_area_id)
                if rev_path is not None and hasattr(rev_path, "locked"):
                    rev_path.locked = True
        self._exit_locked = True

    def _cleanup_and_remove(self, env, world, res: RuleResult) -> None:
        if self._removed:
            return
        if not (self._tutorial_area_id and self._tutorial_place_id and self._spawn_area_id):
            return

        for agent in env.agents:
            if env.curr_agents_state["area"].get(agent.id) == self._tutorial_area_id:
                return

        tutorial_base_ids = {
            "obj_coin",
            "obj_pen",
            "obj_oak_log",
            "obj_iron_bar",
            "obj_workbench",
            "obj_furnace",
            "obj_key",
            "obj_copper_bar",
            "obj_fiber_bundle",
            "obj_small_pouch",
            "obj_paper",
        }
        for agent in env.agents:
            self._remove_items_from_agent(world, agent, tutorial_base_ids)

        for npc_id in list(self._tutorial_npc_ids):
            self._remove_npc_instance(world, npc_id)

        for oid in list(self._tutorial_obj_ids):
            self._remove_object_instance(world, oid)

        spawn_area = world.area_instances.get(self._spawn_area_id)
        if spawn_area is not None:
            spawn_area.neighbors.pop(self._tutorial_area_id, None)

        world.area_instances.pop(self._tutorial_area_id, None)
        world.place_instances.pop(self._tutorial_place_id, None)
        world.auxiliary.get("area_to_place", {}).pop(self._tutorial_area_id, None)

        world.auxiliary.get("tutorial_room", {}).update({"removed": True})
        self._removed = True
        res.events.append(Event(
            type="tutorial_room_removed",
            agent_id="env",
            data={"area_id": self._tutorial_area_id, "place_id": self._tutorial_place_id},
        ))

    @staticmethod
    def _remove_npc_instance(world, npc_id: str) -> None:
        for a in (getattr(world, "area_instances", {}) or {}).values():
            if hasattr(a, "npcs") and a.npcs:
                while npc_id in a.npcs:
                    a.npcs.remove(npc_id)
        npc = (getattr(world, "npc_instances", {}) or {}).pop(npc_id, None)
        name = getattr(npc, "name", None) if npc is not None else None
        if name:
            (getattr(world, "auxiliary", {}) or {}).get("npc_name_to_id", {}).pop(name, None)

    @staticmethod
    def _remove_object_instance(world, obj_id: str) -> None:
        tutorial_area = world.area_instances.get(world.auxiliary.get("tutorial_room", {}).get("area_id"))
        if tutorial_area is not None:
            if obj_id in (tutorial_area.objects or {}):
                tutorial_area.objects.pop(obj_id, None)

        if obj_id in (getattr(world, "container_instances", {}) or {}):
            inst = world.container_instances.pop(obj_id, None)
            name = getattr(inst, "name", None) if inst is not None else None
            if name:
                (getattr(world, "auxiliary", {}) or {}).get("obj_name_to_id", {}).pop(name, None)
            return

        if obj_id in (getattr(world, "writable_instances", {}) or {}):
            inst = world.writable_instances.pop(obj_id, None)
            name = getattr(inst, "name", None) if inst is not None else None
            if name:
                (getattr(world, "auxiliary", {}) or {}).get("obj_name_to_id", {}).pop(name, None)
            return

    @staticmethod
    def _remove_items_from_agent(world, agent, base_ids: Set[str]) -> None:
        def should_remove(oid: str) -> bool:
            try:
                return get_def_id(oid) in base_ids
            except Exception:
                return oid in base_ids

        for oid in list((agent.items_in_hands or {}).keys()):
            if should_remove(oid):
                agent.items_in_hands.pop(oid, None)

        for oid in list((agent.equipped_items_in_limb or {}).keys()):
            if should_remove(oid):
                agent.equipped_items_in_limb.pop(oid, None)

        if agent.inventory.container is not None:
            if should_remove(agent.inventory.container.id):
                try:
                    agent.inventory.container.inventory = {}
                except Exception:
                    pass
                agent.inventory.container = None
            else:
                for oid in list((agent.inventory.items or {}).keys()):
                    if should_remove(oid):
                        agent.inventory.items.pop(oid, None)

        for oid in list((agent.items_in_hands or {}).keys()):
            if oid in (getattr(world, "container_instances", {}) or {}):
                ci = world.container_instances[oid]
                for x in list((ci.inventory or {}).keys()):
                    if should_remove(x):
                        ci.inventory.pop(x, None)

    def _ensure_npc_in_area(self, env, world, area, kind: str, tune=None, stock=None, dialogue=None, description=None) -> Optional[str]:
        spec = None
        if kind == "merchant":
            spec = next((d for d in self.required_npcs if d.get("role") == "merchant"), None)
        elif kind == "enemy":
            spec = next((d for d in self.required_npcs if bool(d.get("enemy", False)) is True), None)
        if not spec:
            return None

        def _apply_dialogue_and_desc(npc_obj, dtext, ddesc):
            if npc_obj is None:
                return
            if ddesc is not None:
                try:
                    npc_obj.description = str(ddesc)
                except Exception:
                    pass
            if dtext is not None:
                try:
                    setattr(npc_obj, "dialogue", str(dtext))
                except Exception:
                    pass

        if kind == "merchant" and self._merchant_npc_inst_id and self._merchant_npc_inst_id in world.npc_instances:
            if stock:
                self._append_stock_to_npc_inventory(world, self._merchant_npc_inst_id, stock)
            inst = world.npc_instances.get(self._merchant_npc_inst_id)
            _apply_dialogue_and_desc(inst, dialogue if dialogue is not None else spec.get("dialogue"), description if description is not None else spec.get("description"))
            return self._merchant_npc_inst_id

        if kind == "enemy" and self._enemy_npc_inst_id and self._enemy_npc_inst_id in world.npc_instances:
            if tune:
                self._tune_enemy_npc_for_tutorial(world, self._enemy_npc_inst_id, tune)
            inst = world.npc_instances.get(self._enemy_npc_inst_id)
            _apply_dialogue_and_desc(inst, dialogue if dialogue is not None else spec.get("dialogue"), description if description is not None else spec.get("description"))
            return self._enemy_npc_inst_id

        base_id = str(spec.get("id"))
        base_name = str(spec.get("name", base_id))

        if kind == "merchant":
            inst_id = self._alloc_id(world.npc_instances, base_id)
            inst_name = self._alloc_name(world.auxiliary.setdefault("npc_name_to_id", {}), base_name)
        else:
            aux = world.auxiliary.setdefault("npc_id_to_count", {})
            idx = int(aux.get(base_id, 0))
            aux[base_id] = idx + 1
            inst_id = f"{base_id}_{idx}"
            inst_name = f"{base_name}_{idx}"

        npc = NPC(
            type=spec.get("type", "npc"),
            id=inst_id,
            name=inst_name,
            enemy=bool(spec.get("enemy", False)),
            unique=bool(spec.get("unique", False)),
            role=str(spec.get("role", kind)),
            level=int(spec.get("level", getattr(area, "level", 1) or 1)),
            description=str(description if description is not None else spec.get("description", "")),
            attack_power=int(spec.get("attack_power", 0)),
            hp=int(spec.get("hp", 1)),
            base_hp=int(spec.get("base_hp", 1)),
            base_attack_power=int(spec.get("base_attack_power", 0)),
            coins=int(spec.get("coins", 0)),
            slope_hp=int(spec.get("slope_hp", 0)),
            slope_attack_power=int(spec.get("slope_attack_power", 0)),
            quest=bool(spec.get("quest", False)),
        )
        try:
            setattr(npc, "dialogue", str(dialogue if dialogue is not None else spec.get("dialogue", "")))
        except Exception:
            pass

        world.npc_instances[npc.id] = npc
        world.auxiliary.setdefault("npc_name_to_id", {})[npc.name] = npc.id
        self._tutorial_npc_ids.add(npc.id)

        if kind == "merchant":
            self._merchant_npc_inst_id = npc.id
            self._merchant_npc_inst_name = npc.name
        elif kind == "enemy":
            self._enemy_npc_inst_id = npc.id
            self._enemy_npc_inst_name = npc.name

        if kind == "merchant" and stock:
            self._append_stock_to_npc_inventory(world, npc.id, stock)
        if kind == "enemy" and tune:
            self._tune_enemy_npc_for_tutorial(world, npc.id, tune)

        return npc.id

    @staticmethod
    def _alloc_name(name_to_id: Dict[str, str], base: str) -> str:
        if base not in name_to_id:
            return base
        i = 1
        while f"{base}_{i}" in name_to_id:
            i += 1
        return f"{base}_{i}"

    def _select_npc_def(self, env, kind: str) -> Optional[dict]:
        npc_defs = env.world_definition.get("entities", {}).get("npcs", []) or []
        if kind == "enemy":
            cands = [d for d in npc_defs if d.get("enemy") is True]
        elif kind == "merchant":
            cands = [d for d in npc_defs if (d.get("enemy") is False and d.get("role") == "merchant")]
        else:
            return None
        if not cands:
            return None
        cands.sort(key=lambda d: (not bool(d.get("unique", False)), str(d.get("id", ""))))
        return cands[0]

    def _ensure_unique_npc_instance(self, env, world, npc_def: dict, area_level: int) -> Optional[str]:
        base_id = npc_def.get("id")
        base_name = npc_def.get("name", base_id)
        if not base_id or not base_name:
            return None

        if base_id in getattr(world, "npc_instances", {}):
            npc_id = base_id
        else:
            npc_id = None
            for nid, inst in (getattr(world, "npc_instances", {}) or {}).items():
                try:
                    if get_def_id(nid) == base_id:
                        npc_id = nid
                        break
                except Exception:
                    pass
                if inst is not None and getattr(inst, "name", None) == base_name:
                    npc_id = nid
                    break

            if npc_id is None:
                npc_obj = self._clone_npc_template(world)
                if npc_obj is None:
                    class _NPC:  # noqa: N801
                        pass
                    npc_obj = _NPC()

                npc_obj.id = base_id
                npc_obj.name = base_name
                npc_obj.enemy = bool(npc_def.get("enemy", False))
                npc_obj.unique = True
                npc_obj.role = npc_def.get("role", None)

                inv: dict[str, int] = {}
                for oid in (npc_def.get("objects", []) or []):
                    if isinstance(oid, str):
                        rid = self._resolve_object_def_id(world, oid) or oid
                        inv[rid] = inv.get(rid, 0) + 1
                npc_obj.inventory = inv

                if npc_obj.enemy:
                    base_ap = float(npc_def.get("base_attack_power", 0))
                    slope_ap = float(npc_def.get("slope_attack_power", 0))
                    base_hp = float(npc_def.get("base_hp", 1))
                    slope_hp = float(npc_def.get("slope_hp", 0))
                    npc_obj.attack_power = int(round(base_ap + max(area_level - 1, 0) * slope_ap))
                    npc_obj.hp = int(round(base_hp + max(area_level - 1, 0) * slope_hp))
                    npc_obj.max_hp = max(int(getattr(npc_obj, "max_hp", npc_obj.hp)), npc_obj.hp)
                else:
                    npc_obj.attack_power = int(getattr(npc_obj, "attack_power", 0))
                    npc_obj.hp = int(getattr(npc_obj, "hp", 1))
                    npc_obj.max_hp = int(getattr(npc_obj, "max_hp", npc_obj.hp))

                world.npc_instances[base_id] = npc_obj
                npc_id = base_id

        inst = world.npc_instances.get(npc_id)
        if inst is not None:
            if not hasattr(inst, "inventory") or inst.inventory is None:
                inst.inventory = {}
            for oid in (npc_def.get("objects", []) or []):
                if isinstance(oid, str):
                    rid = self._resolve_object_def_id(world, oid) or oid
                    inst.inventory[rid] = inst.inventory.get(rid, 0) + 1

        return npc_id

    def _create_nonunique_npc_instance_for_tutorial(self, env, world, npc_def: dict, area_level: int) -> Optional[str]:
        base_id = npc_def.get("id")
        base_name = npc_def.get("name", base_id)
        if not base_id or not base_name:
            return None

        aux = getattr(world, "auxiliary", None)
        if aux is None:
            return None
        aux.setdefault("npc_id_to_count", {})
        idx = int(aux["npc_id_to_count"].get(base_id, 0))
        aux["npc_id_to_count"][base_id] = idx + 1

        inst_id = f"{base_id}_{idx}"
        inst_name = f"{base_name}_{idx}"

        npc_obj = self._clone_npc_template(world)
        if npc_obj is None:
            class _NPC:  # noqa: N801
                pass
            npc_obj = _NPC()

        npc_obj.id = inst_id
        npc_obj.name = inst_name
        npc_obj.enemy = bool(npc_def.get("enemy", False))
        npc_obj.unique = False
        npc_obj.role = npc_def.get("role", None)

        inv: dict[str, int] = {}
        for oid in (npc_def.get("objects", []) or []):
            if isinstance(oid, str):
                rid = self._resolve_object_def_id(world, oid) or oid
                inv[rid] = inv.get(rid, 0) + 1
        npc_obj.inventory = inv

        if npc_obj.enemy:
            base_ap = float(npc_def.get("base_attack_power", 0))
            slope_ap = float(npc_def.get("slope_attack_power", 0))
            base_hp = float(npc_def.get("base_hp", 1))
            slope_hp = float(npc_def.get("slope_hp", 0))
            npc_obj.attack_power = int(round(base_ap + max(area_level - 1, 0) * slope_ap))
            npc_obj.hp = int(round(base_hp + max(area_level - 1, 0) * slope_hp))
            npc_obj.max_hp = max(int(getattr(npc_obj, "max_hp", npc_obj.hp)), npc_obj.hp)
        else:
            npc_obj.attack_power = int(getattr(npc_obj, "attack_power", 0))
            npc_obj.hp = int(getattr(npc_obj, "hp", 1))
            npc_obj.max_hp = int(getattr(npc_obj, "max_hp", npc_obj.hp))

        world.npc_instances[inst_id] = npc_obj
        return inst_id

    def _detach_npc_from_all_areas(self, world, npc_id: str) -> None:
        for a in (getattr(world, "area_instances", {}) or {}).values():
            if hasattr(a, "npcs") and a.npcs:
                while npc_id in a.npcs:
                    a.npcs.remove(npc_id)

    def _register_npc_name_map(self, world, npc_id: str) -> None:
        aux = getattr(world, "auxiliary", None)
        if aux is None:
            return
        aux.setdefault("npc_name_to_id", {})
        npc = (getattr(world, "npc_instances", {}) or {}).get(npc_id)
        name = getattr(npc, "name", None) if npc is not None else None
        if name:
            aux["npc_name_to_id"][name] = npc_id

    def _clone_npc_template(self, world):
        insts = getattr(world, "npc_instances", None)
        if isinstance(insts, dict) and insts:
            try:
                return copy.deepcopy(next(iter(insts.values())))
            except Exception:
                return None
        return None

    def _create_dummy_npc_instance(self, world, area_level: int, kind: str, area) -> Optional[str]:
        aux = getattr(world, "auxiliary", None)
        if aux is None:
            return None
        aux.setdefault("npc_id_to_count", {})
        base_id = f"npc_tutorial_{kind}"
        idx = int(aux["npc_id_to_count"].get(base_id, 0))
        aux["npc_id_to_count"][base_id] = idx + 1

        inst_id = f"{base_id}_{idx}"
        inst_name = f"{base_id}_{idx}"

        npc_obj = self._clone_npc_template(world)
        if npc_obj is None:
            class _NPC:  # noqa: N801
                pass
            npc_obj = _NPC()

        npc_obj.id = inst_id
        npc_obj.name = inst_name

        if kind == "enemy":
            npc_obj.enemy = True
            npc_obj.unique = False
            npc_obj.role = "tutorial_enemy"
            npc_obj.attack_power = 3 + max(area_level - 1, 0)
            npc_obj.hp = 10 + 2 * max(area_level - 1, 0)
            npc_obj.max_hp = npc_obj.hp
            npc_obj.inventory = {}
            try:
                setattr(npc_obj, "dialogue", "")
            except Exception:
                pass
        else:
            npc_obj.enemy = False
            npc_obj.unique = True
            npc_obj.role = "merchant"
            npc_obj.attack_power = 0
            npc_obj.hp = 1
            npc_obj.max_hp = 1
            npc_obj.inventory = {}
            try:
                setattr(npc_obj, "dialogue", "")
            except Exception:
                pass

        world.npc_instances[inst_id] = npc_obj
        self._register_npc_name_map(world, inst_id)
        return inst_id

    def _append_stock_to_npc_inventory(self, world, npc_id: str, stock) -> None:
        npc = (getattr(world, "npc_instances", {}) or {}).get(npc_id)
        if npc is None:
            return
        if not hasattr(npc, "inventory") or npc.inventory is None:
            npc.inventory = {}

        pairs: list[tuple[str, int]] = []
        if isinstance(stock, dict):
            for k, v in stock.items():
                try:
                    c = int(v)
                except Exception:
                    continue
                if c > 0:
                    pairs.append((str(k), c))
        elif isinstance(stock, list):
            tmp = {}
            for x in stock:
                if isinstance(x, str):
                    tmp[x] = tmp.get(x, 0) + 1
            pairs = list(tmp.items())
        else:
            return

        for obj_id, cnt in pairs:
            self._add_stock_item(world, npc, obj_id, cnt)

    def _add_stock_item(self, world, npc, obj_id: str, cnt: int) -> None:
        resolved = self._resolve_object_def_id(world, obj_id)
        if not resolved:
            return

        obj_def = (getattr(world, "objects", {}) or {}).get(resolved)
        if obj_def is None:
            return

        cat = getattr(obj_def, "category", None)

        if cat in ("container", "note"):
            for _ in range(int(cnt)):
                inst_id, inst_name = self._create_indexed_obj_instance(world, resolved)
                if inst_id:
                    npc.inventory[inst_id] = npc.inventory.get(inst_id, 0) + 1
                if inst_name:
                    (getattr(world, "auxiliary", {}) or {}).setdefault("obj_name_to_id", {})[inst_name] = inst_id
            return

        npc.inventory[resolved] = npc.inventory.get(resolved, 0) + int(cnt)

    def _create_indexed_obj_instance(self, world, base_id: str) -> tuple[Optional[str], Optional[str]]:
        if base_id not in (getattr(world, "objects", {}) or {}):
            return None, None
        obj_def = world.objects[base_id]
        cat = getattr(obj_def, "category", None)
        usage = getattr(obj_def, "usage", None)

        aux = getattr(world, "auxiliary", None)
        if aux is None:
            return None, None

        if cat == "container":
            key = "container_id_to_count"
            store = getattr(world, "container_instances", None)
        elif cat == "note":
            if hasattr(world, "note_instances"):
                key = "note_id_to_count"
                store = getattr(world, "note_instances", None)
            else:
                key = "writable_id_to_count"
                store = getattr(world, "writable_instances", None)
        elif usage == "writable":
            key = "writable_id_to_count"
            store = getattr(world, "writable_instances", None) or getattr(world, "note_instances", None)
        else:
            return None, None

        aux.setdefault(key, {})
        aux[key].setdefault(base_id, 0)
        idx = int(aux[key][base_id])
        inst = obj_def.create_instance(idx)
        aux[key][base_id] = idx + 1

        if isinstance(store, dict):
            store[inst.id] = inst

        aux.setdefault("obj_name_to_id", {})[inst.name] = inst.id
        return inst.id, inst.name

    def _resolve_object_def_id(self, world, obj_id: str) -> Optional[str]:
        if obj_id in (getattr(world, "objects", {}) or {}):
            return obj_id

        try:
            base = get_def_id(obj_id)
            if base in world.objects:
                return base
        except Exception:
            pass

        aux = getattr(world, "auxiliary", {}) or {}
        name_map = aux.get("obj_name_to_id", {}) or {}
        if obj_id in name_map:
            rid = name_map[obj_id]
            try:
                rb = get_def_id(rid)
                if rb in world.objects:
                    return rb
            except Exception:
                if rid in world.objects:
                    return rid

        for oid, odef in (getattr(world, "objects", {}) or {}).items():
            if getattr(odef, "name", None) == obj_id:
                return oid

        return None

    def _tune_enemy_npc_for_tutorial(self, world, npc_id: str, tune: dict) -> None:
        npc = (getattr(world, "npc_instances", {}) or {}).get(npc_id)
        if npc is None:
            return
        if "attack_power" in tune:
            npc.attack_power = int(tune["attack_power"])
        if "hp" in tune:
            npc.hp = int(tune["hp"])
            npc.max_hp = max(int(getattr(npc, "max_hp", npc.hp)), npc.hp)

    def _is_step_done(self, env, world, agent, step: dict, events: list[Event]) -> bool:
        for cond in step.get("done_any", []):
            if self._cond_true(env, world, agent, cond, events):
                return True
        return False

    def _cond_true(self, env, world, agent, cond: dict, events: list[Event]) -> bool:
        kind = cond.get("kind")

        if kind == "event":
            etype = cond.get("type")
            base_obj = cond.get("base_obj")
            data_match = cond.get("data") or {}

            if data_match:
                fmt = {
                    "bag": self._bag_name or "small_pouch",
                    "writable": self._writable_name or "paper",
                    "merchant": self._merchant_npc_inst_name or "merchant",
                    "enemy": self._enemy_npc_inst_name or "enemy",
                    "merchant_id": self._merchant_npc_inst_id or "",
                    "enemy_id": self._enemy_npc_inst_id or "",
                    "tutorial_area_id": self._tutorial_area_id or "",
                    "spawn_area_id": self._spawn_area_id or "",
                }
                tmp = {}
                for k, v in data_match.items():
                    if isinstance(v, str):
                        try:
                            tmp[k] = v.format(**fmt)
                        except Exception:
                            tmp[k] = v
                    else:
                        tmp[k] = v
                data_match = tmp

            for e in events:
                if getattr(e, "type", None) != etype:
                    continue
                if base_obj is not None:
                    d = getattr(e, "data", None) or {}
                    oid = d.get("obj_id")
                    if oid is None or get_def_id(oid) != base_obj:
                        continue
                if data_match:
                    d = getattr(e, "data", None) or {}
                    if any(d.get(k) != v for k, v in data_match.items()):
                        continue
                return True
            return False

        if kind == "state":
            stype = cond.get("type")
            base_obj = cond.get("base_obj")

            if stype == "in_hands_base":
                return self._has_base(agent.items_in_hands, base_obj)

            if stype == "inventory_container_is_base":
                return (
                    agent.inventory.container is not None
                    and get_def_id(agent.inventory.container.id) == base_obj
                )

            if stype == "inventory_has_base":
                if agent.inventory.container is None:
                    return False
                return any(cnt > 0 and get_def_id(oid) == base_obj for oid, cnt in agent.inventory.items.items())

            if stype == "held_containers_have_base":
                for oid in list((agent.items_in_hands or {}).keys()):
                    if oid in world.container_instances:
                        inv = world.container_instances[oid].inventory
                        if any(cnt > 0 and get_def_id(x) == base_obj for x, cnt in inv.items()):
                            return True
                return False

            if stype == "base_count_at_least":
                need = int(cond.get("count", 1))
                if need <= 0:
                    return True
                total = int(self._agent_base_count_in_possession(env, world, agent, base_obj))
                return total >= need

            if stype == "inventory_has_and_not_in_hands":
                in_inv = False
                if agent.inventory.container is not None:
                    in_inv = any(cnt > 0 and get_def_id(oid) == base_obj for oid, cnt in agent.inventory.items.items())
                return in_inv and (not self._has_base(agent.items_in_hands, base_obj))

            if stype == "area_changed_from_spawn":
                return self._tutorial_area_id is not None and env.curr_agents_state["area"][agent.id] != self._tutorial_area_id

            if stype == "bag_in_hand_and_no_inventory":
                return self._has_base(agent.items_in_hands, "obj_small_pouch") and (agent.inventory.container is None)

            return False

        return False

    @staticmethod
    def _agent_has_exact(agent, obj_id: str) -> bool:
        return ((agent.items_in_hands or {}).get(obj_id, 0) > 0) or ((agent.equipped_items_in_limb or {}).get(obj_id, 0) > 0)

    @staticmethod
    def _has_base(items: dict[str, int], base_obj_id: str) -> bool:
        if not base_obj_id:
            return False
        for oid, cnt in (items or {}).items():
            if cnt > 0 and get_def_id(oid) == base_obj_id:
                return True
        return False

    def _agent_has_base_anywhere(self, env, world, agent, base_obj_id: str) -> bool:
        if self._has_base(agent.items_in_hands, base_obj_id):
            return True
        if self._has_base(agent.equipped_items_in_limb, base_obj_id):
            return True
        if agent.inventory.container is not None:
            if any(cnt > 0 and get_def_id(oid) == base_obj_id for oid, cnt in agent.inventory.items.items()):
                return True
        for oid in list((agent.items_in_hands or {}).keys()):
            if oid in world.container_instances:
                inv = world.container_instances[oid].inventory
                if any(cnt > 0 and get_def_id(x) == base_obj_id for x, cnt in inv.items()):
                    return True
        if self._tutorial_area_id:
            area = world.area_instances[self._tutorial_area_id]
            if any(cnt > 0 and get_def_id(oid) == base_obj_id for oid, cnt in area.objects.items()):
                return True
        return False

