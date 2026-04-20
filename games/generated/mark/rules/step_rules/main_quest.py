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


class MainQuestStepRule(BaseStepRule):

    name = "main_quest_step_v2"
    description = "Two-chapter main quest (simple): talk + write/drop paper + craft + explore + bosses."
    priority = 2

    CH1_LIBRARY_AREA = "area_castle_library"
    CH1_BOSS_CANDIDATE_AREAS = ["area_castle_armory", "area_market_stalls", "area_market_backstreets"]
    CH1_BOSS_AREA = "area_market_backstreets"
    CH1_EXPLORE_MIN = 3

    CH2_CONTACT_AREA = "area_mines_crystal_grotto"
    CH2_BOSS_CANDIDATE_AREAS = ["area_mines_shafts", "area_woods_ridge", "area_flats_farm"]
    CH2_BOSS_AREA = "area_flats_farm"
    CH2_EXPLORE_MIN = 3

    ANY_WEAPON_BASE_IDS = ["obj_oak_dagger", "obj_copper_sword", "obj_steel_sword", "obj_oak_sword", "obj_goblin_sword"]
    ANY_ARMOR_BASE_IDS = [
        "obj_cloth_tunic", "obj_copper_mail", "obj_steel_plate_armor",
        "obj_leather_vest", "obj_wooden_buckler",
    ]

    PAPER_BASE_IDS = ["obj_paper"]

    TEAR_OBJ_ID = "obj_tear_of_forest"
    TEAR_VALUE = 10

    LIBRARIAN_BASE_NPC_ID = "npc_quest_librarian_v2"
    CONTACT_BASE_NPC_ID = "npc_quest_contact_v2"

    CH1_BOSS_BASE_NPC_ID = "npc_boss_page_gnawer_v2"
    CH2_BOSS_BASE_NPC_ID = "npc_boss_root_warden_v2"

    # Boss stats (verified with combat_simulator using correct agent stats)
    # Level 2 agent: HP=120, base_atk=15 + oak_dagger(6) = 21, def=0
    # Level 4 agent: HP=160, base_atk=25 + oak_dagger(6) = 31, def=12 (copper_mail)
    
    # CH1: AWAW wins in 11 turns with 18 HP left, AAAA loses in 12 turns
    CH1_BOSS_HP = 80
    CH1_BOSS_ATK = 15

    # CH2: AWAW wins in 11 turns with 16 HP left, AAAA loses in 12 turns
    CH2_BOSS_HP = 160
    CH2_BOSS_ATK = 36

    CH2_DEBUFFED_BOSS_ATK = 30
    CH2_DEBUFFED_BOSS_HP_CAP = 120  # also used as lowered "curr_max_hp" when debuffed

    OBJECTIVE_NPC_BASE_ID = "npc_quest_wayfarer_guide_v2"
    OBJECTIVE_NPC_NAME_POOL = [
        "mira", "rowan", "sera", "jude", "ivy", "remy", "nina", "oren", "kian", "lena",
        "cass", "elio", "syl", "tess", "vale", "bram", "arlen", "dara", "noel", "faye",
    ]
    OBJECTIVE_NPC_MAX_HP = 10**9
    OBJECTIVE_NPC_ATK = 10**6

    # If agent is stuck on a stage for this many steps, teleport the guide to remind them
    GUIDE_REMINDER_INTERVAL_STEPS = 30

    QUEST_CONFIG: Dict[str, Any] = {
        "title": "The Borrowed Mark",
        "intro": (
            "=== MAIN QUEST: The Borrowed Mark ===\n"
            "A rule in the old library is simple: if you take something, you leave something behind.\n"
            "A small note can be a promise.\n"
            "A promise can be a key.\n"
            "====================================\n"
            "\n"
        ),
        "chapters": [
            {
                "id": "chapter_1",
                "title": "The Library Debt",
                "intro": (
                    "Chapter 1: The Library Debt\n"
                    "There is a library that keeps more than books.\n"
                ),
                "stages": [
                    {
                        "id": "ch1_craft_oak_rod",
                        "objective": (
                            "Craft an oak rod.\n"
                            "You'll need ingredient: 1 oak_log."
                        ),
                        "done_any": [
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_oak_rod"]},
                        ],
                        "on_complete_feedback": (
                            "The oak rod is smooth and sturdy in your hand.\n"
                            "A useful tool for crafting more advanced items.\n"
                        ),
                    },
                    {
                        "id": "ch1_craft_copper_sword",
                        "objective": (
                            "Craft a copper sword and hold it in your hands.\n"
                        ),
                        "done_any": [
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_copper_sword"]},
                        ],
                        "on_complete_feedback": (
                            "The copper blade gleams faintly in the light.\n"
                            "With this weapon, you're ready for the challenges ahead.\n"
                        ),
                    },
                    {
                        "id": "ch1_library_meet",
                        "objective": "Enter the area where {ch1_librarian_name} is at.",
                        "done_any": [
                            {"kind": "state", "type": "in_area_with_npc", "npc_key": "ch1_librarian_npc"},
                        ],
                        "on_complete_feedback": (
                            "{ch1_librarian_name} shuts a book with a soft thud.\n"
                            "'Something was taken from the back shelves,' they say.\n"
                            "'The shelves don’t open for empty hands, and they don’t open for liars.'\n"
                            "\n"
                            "They slide a scrap of paper toward you.\n"
                            "'Write this exactly:\n"
                            "  {ch1_promise_line}\n"
                            "It’s not magic. It’s a record.\n"
                            "The library remembers what you admit.'\n"
                        ),
                    },
                    {
                        "id": "ch1_write_promise",
                        "objective": (
                            "Write the word you were given on a paper and hold it:\n"
                            "  {ch1_promise_line}\n"
                            "(hint: you may empty your hands by storing or dropping items.)"
                        ),
                        "done_all": [
                            {"kind": "state", "type": "paper_text_in_possession", "text_key": "ch1_promise_line"},
                        ],
                        "on_complete_feedback": (
                            "You hold out the paper.\n"
                            "For a moment, the room feels quieter, like the building noticed.\n"
                            "\n"
                            "{ch1_librarian_name} nods.\n"
                            "'Good. Keep that paper with you. Now you can go get it back.'\n"
                        ),
                    },
                    {
                        "id": "ch1_explore_armory",
                        "objective": (
                            "The missing book was last traced somewhere nearby.\n"
                            "First, search {ch1_area_1} for any signs."
                        ),
                        "done_all": [
                            {"kind": "state", "type": "in_area", "area_key": "ch1_area_1"},
                        ],
                        "on_complete_feedback": (
                            "Dust and old equipment. No sign of torn pages here.\n"
                            "The trail must lead elsewhere.\n"
                        ),
                    },
                    {
                        "id": "ch1_craft_key",
                        "objective": (
                            "The path to the stalls is locked.\n"
                            "First, craft a key and pick it up.\n"
                            "You'll need ingredients: 1 copper_bar and 1 fiber_bundle.\n"
                            "(hint: you may empty your hands by storing or dropping items.)"
                        ),
                        "done_any": [
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_key"]},
                        ],
                        "on_complete_feedback": (
                            "The key clicks together in your hand.\n"
                            "Now you can unlock the way forward.\n"
                        ),
                    },
                    {
                        "id": "ch1_explore_stalls",
                        "objective": "With the key in hand, continue your search. Head to {ch1_area_2}.",
                        "done_all": [
                            {"kind": "state", "type": "in_area", "area_key": "ch1_area_2"},
                        ],
                        "on_complete_feedback": (
                            "Busy stalls and haggling merchants. A few people mention seeing\n"
                            "something scurrying toward the back alleys with scraps in its mouth.\n"
                        ),
                    },
                    {
                        "id": "ch1_craft_torch",
                        "objective": (
                            "The back alleys are dark. You'll need light.\n"
                            "Craft a torch using 1 oak_rod and 1 coal_chunk before heading in.\n"
                            "Pick it up after crafting.\n"
                        ),
                        "done_any": [
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_torch"]},
                        ],
                        "on_complete_feedback": (
                            "The torch flickers to life.\n"
                            "A nearby crate tips over, revealing a couple more torches.\n"
                            "Now you can follow the trail into the shadows.\n"
                        ),
                        "spawns_objects": {"obj_torch": 2},
                    },
                    {
                        "id": "ch1_explore_backstreets",
                        "objective": (
                            "The trail leads to {ch1_area_3}.\n"
                            "Go there and find where the pages have been carried off."
                        ),
                        "done_all": [
                            {"kind": "state", "type": "in_area", "area_key": "ch1_area_3"},
                        ],
                        "on_complete_feedback": (
                            "You find scraps of paper stuck to the ground like dead leaves.\n"
                            "A trail of shredded pages leads into a cramped corner.\n"
                            "Something inside is chewing slowly.\n"
                        ),
                    },
                    # {
                    #     "id": "ch1_weapon_gate",
                    #     "objective": "Craft an oak dagger. ",
                    #     "done_any": [
                    #         {"kind": "state", "type": "has_any_of", "base_objs": ANY_WEAPON_BASE_IDS},
                    #     ],
                    #     "on_complete_feedback": (
                    #         "The balance feels real in your hand.\n"
                    #         "Whatever took the missing book won't give it back politely.\n"
                    #         "Make sure you have your blade equipped before confronting it.\n"
                    #     ),
                    #     "spawns_boss_key": "ch1_boss_npc",
                    # },
                    {
                        "id": "ch1_boss",
                        "objective": "Attack and defeat {ch1_boss_name} in {ch1_area_3}, the monster that has been eating what the library keeps.",
                        "done_any": [
                            {"kind": "event", "type": "npc_killed", "npc_key": "ch1_boss_npc"},
                            {"kind": "state", "type": "boss_already_defeated", "boss_key": "ch1_boss_npc"},
                        ],
                        "on_complete_feedback": (
                            "The chewing stops.\n"
                            "Between torn pages, you find a clean slip of paper—untouched.\n"
                            "\n"
                            "=== Chapter 1 Complete ===\n"
                            "You faced down the hunger and reclaimed what was lost.\n"
                        ),
                        "chapter_complete": True,
                    },
                ],
            },
            {
                "id": "chapter_2",
                "title": "The Forest Receipt",
                "intro": (
                    "Chapter 2: The Forest Receipt\n"
                    "The forest keeps its own ledgers.\n"
                    "A promise written down can travel farther than you can.\n"
                ),
                "stages": [
                    {
                        "id": "ch2_show_note_to_contact",
                        "objective": (
                            "Hold the note you've written \"kindness\" on and find {ch2_contact_name} in {ch2_contact_area}.\n"
                        ),
                        "done_all": [
                            {"kind": "state", "type": "in_area", "area_key": "ch2_contact_area"},
                            {"kind": "state", "type": "in_area_with_npc", "npc_key": "ch2_contact_npc"},
                            {"kind": "state", "type": "paper_text_in_possession", "text_key": "ch1_promise_line"},
                        ],
                        "on_complete_feedback": (
                            "{ch2_contact_name} glances at the note in your hand and nods.\n"
                            "'Alright,' they say. 'You’re not just making noise.'\n"
                            "\n"
                            "They lower their voice.\n"
                            "'If the thing you’re hunting gets wild, this line can dull its bite.\n"
                            "But I know why you’re really here.\n"
                            "'If you want the Tear of Forest, you pay fair and you listen first.'\n"
                        ),
                    },
                    {
                        "id": "ch2_trade_tear",
                        "objective": "Trade with {ch2_contact_name} and obtain the Tear of Forest.",
                        "done_all": [
                            {"kind": "state", "type": "in_area_with_npc", "npc_key": "ch2_contact_npc"},
                            {"kind": "event", "type": "object_bought", "npc_key": "ch2_contact_npc", "obj_id": TEAR_OBJ_ID},
                        ],
                        "on_complete_feedback": (
                            "The Tear of Forest is cold even through your palm.\n"
                            "{ch2_contact_name} points away from the path.\n"
                            "'The trouble you’re heading toward doesn’t like being named.\n"
                            "So don’t name it. Just be ready.'\n"
                        ),
                    },
                    {
                        "id": "ch2_explore_shafts",
                        "objective": (
                            "The lair is somewhere in the wilds.\n"
                            "First, search {ch2_area_1} for any signs."
                        ),
                        "done_all": [
                            {"kind": "state", "type": "in_area", "area_key": "ch2_area_1"},
                        ],
                        "on_complete_feedback": (
                            "Dark tunnels and dripping stone. Nothing but echoes here.\n"
                            "The creature's lair must be elsewhere.\n"
                        ),
                    },
                    {
                        "id": "ch2_armor_gate",
                        "objective": "Craft any armor. You’ll need something that can take a real hit.",
                        "done_any": [
                            {"kind": "state", "type": "has_any_of", "base_objs": ANY_ARMOR_BASE_IDS},
                        ],
                        "on_complete_feedback": (
                            "Straps tighten. Weight settles.\n"
                            "Good.\n"
                        ),
                    },
                    {
                        "id": "ch2_explore_ridge",
                        "objective": "Continue your search. Head to {ch2_area_2}.",
                        "done_all": [
                            {"kind": "state", "type": "in_area", "area_key": "ch2_area_2"},
                        ],
                        "on_complete_feedback": (
                            "Wind through the trees. You spot claw marks on the bark,\n"
                            "leading toward the flatlands below.\n"
                        ),
                    },
                    {
                        "id": "ch2_explore_farm",
                        "objective": (
                            "The trail leads to {ch2_area_3}.\n"
                            "Go there and find the lair."
                        ),
                        "done_all": [
                            {"kind": "state", "type": "in_area", "area_key": "ch2_area_3"},
                        ],
                        "on_complete_feedback": (
                            "The air smells like crushed leaves and damp bark.\n"
                            "The ground is scarred, like something heavy dragged itself in circles.\n"
                            "\n"
                            "Something inside the lair shifts.\n"
                        ),
                        "spawns_boss_key": "ch2_boss_npc",
                    },
                    {
                        "id": "ch2_boss",
                        "objective": "Defeat {ch2_boss_name}, the thing guarding the lair.",
                        "done_any": [
                            {"kind": "event", "type": "npc_killed", "npc_key": "ch2_boss_npc"},
                            {"kind": "state", "type": "boss_already_defeated", "boss_key": "ch2_boss_npc"},
                        ],
                        "on_complete_feedback": (
                            "The lair goes still.\n"
                            "Your armor is scratched. Your paper is smudged. You’re still standing.\n"
                            "\n"
                            "=== QUEST COMPLETE ===\n"
                            "You paid your debt in ink, then in effort.\n"
                        ),
                        "quest_complete": True,
                    },
                ],
            },
        ],
    }

    required_objects: List[Dict[str, Any]] = [
        {
            "type": "object",
            "id": TEAR_OBJ_ID,
            "name": "tear_of_forest",
            "category": "material",
            "usage": "quest",
            "value": TEAR_VALUE,
            "size": 1,
            "description": "A clear drop sealed in glass. It smells like wet moss and new bark.",
            "craft": {"ingredients": {}, "dependencies": []},
            "level": 2,
            "quest": True,
        },
    ]

    required_npcs: List[Dict[str, Any]] = [
        {
            "type": "npc",
            "id": LIBRARIAN_BASE_NPC_ID,
            "name": "librarian_rowan",
            "enemy": False,
            "unique": True,
            "role": "librarian",
            "quest": True,
            "description": "A quiet librarian who watches hands more than faces.",
            "base_attack_power": 0,
            "slope_attack_power": 0,
            "base_hp": 200,
            "slope_hp": 0,
            "objects": [],
        },
        {
            "type": "npc",
            "id": CONTACT_BASE_NPC_ID,
            "name": "broker_sera",
            "enemy": False,
            "unique": True,
            "role": "merchant",
            "quest": True,
            "description": "A careful trader who deals in odd receipts and rarer materials.",
            "base_attack_power": 0,
            "slope_attack_power": 0,
            "base_hp": 200,
            "slope_hp": 0,
            "objects": [TEAR_OBJ_ID],
        },
        {
            "type": "npc",
            "id": CH1_BOSS_BASE_NPC_ID,
            "name": "page_gnawer",
            "enemy": True,
            "unique": True,
            "role": "boss",
            "quest": True,
            "description": "A hunched thing with ink-stained teeth. It eats paper like hunger is a hobby.",
            "base_attack_power": CH1_BOSS_ATK,
            "slope_attack_power": 0,
            "base_hp": CH1_BOSS_HP,
            "slope_hp": 0,
            "combat_pattern": ["attack", "attack", "wait", "attack", "defend"],
            "objects": [],
        },
        {
            "type": "npc",
            "id": CH2_BOSS_BASE_NPC_ID,
            "name": "root_warden",
            "enemy": True,
            "unique": True,
            "role": "boss",
            "quest": True,
            "description": "A knot of bark and muscle that moves like a slow decision.",
            "base_attack_power": CH2_BOSS_ATK,
            "slope_attack_power": 0,
            "base_hp": CH2_BOSS_HP,
            "slope_hp": 0,
            "combat_pattern": ["wait", "attack", "attack", "attack", "wait"],
            "objects": [],
        },
        # objective guide (invincible) — ported from v1
        {
            "type": "npc",
            "id": OBJECTIVE_NPC_BASE_ID,
            "name": "mira",
            "enemy": False,
            "unique": True,
            "role": "guide",
            "quest": True,
            "description": "",
            "base_attack_power": OBJECTIVE_NPC_ATK,
            "slope_attack_power": 0,
            "base_hp": OBJECTIVE_NPC_MAX_HP,
            "slope_hp": 0,
            "objects": [],
        },
    ]

    undistributable_objects: List[str] = [
        TEAR_OBJ_ID,
    ]

    def __init__(self) -> None:
        super().__init__()
        self._initialized = False
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._generated: Dict[str, Any] = {}
        self._world_ref = None
        self._area_to_place: Dict[str, str] = {}

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world = ctx.env, ctx.world

        if "main_quest" not in env.world_definition.get("custom_events", []):
            return

        tutorial_enabled = "tutorial" in env.world_definition.get("custom_events", [])
        tutorial_room = world.auxiliary.get("tutorial_room") or {}

        env.curr_agents_state.setdefault("main_quest_progress", {})
        world.auxiliary.setdefault("main_quest", {})
        mq = world.auxiliary["main_quest"]
        mq.setdefault("generated", {})
        mq.setdefault("boss_max_hp", {})
        mq.setdefault("boss_defeated", {})
        mq.setdefault("boss_mods", {})
        mq.setdefault("objective_npc_names", {})  # agent_id -> guide_name
        mq.setdefault("run_seed", None)

        # v1: deterministic guide-name assignment for multi-agent
        self._ensure_objective_name_map_for_agents(env, world)

        if not self._initialized:
            self._world_ref = world
            self._area_to_place = world.auxiliary.get("area_to_place", {}) or {}
            if not self._area_to_place:
                for place_id, place in world.place_instances.items():
                    for area_id in getattr(place, "areas", []):
                        self._area_to_place[area_id] = place_id
                world.auxiliary["area_to_place"] = self._area_to_place

            persisted = mq.get("generated")
            if isinstance(persisted, dict) and persisted:
                self._generated = persisted
            else:
                self._bootstrap(env, world)
                mq["generated"] = self._generated

            self._register_required_objects(world)
            self._register_required_npcs(world)

            self._ensure_static_quest_npcs(env, world)
            self._ensure_tear_has_value(world)

            self._initialized = True

        # v1 robustness: if something deletes static quest NPCs mid-run, heal them
        self._ensure_static_quest_npcs(env, world)

        for agent in env.agents:
            aid = agent.id

            if tutorial_enabled and tutorial_room and not bool(tutorial_room.get("removed", False)):
                continue

            env.curr_agents_state["main_quest_progress"].setdefault(aid, None)

            # (re)load per-agent progress
            if aid not in self._progress:
                persisted_prog = env.curr_agents_state["main_quest_progress"].get(aid)
                if isinstance(persisted_prog, dict) and persisted_prog:
                    self._progress[aid] = persisted_prog
                    prog = self._progress[aid]
                    guide_name = self._ensure_agent_guide_name(world, prog, aid)
                    env.curr_agents_state["main_quest_progress"][aid] = prog
                else:
                    self._init_agent_progress(aid)
                    prog = self._progress[aid]
                    guide_name = self._ensure_agent_guide_name(world, prog, aid)
                    env.curr_agents_state["main_quest_progress"][aid] = prog

                    # show intro once
                    res.add_feedback(aid, self._render_text(self.QUEST_CONFIG["intro"], extra={"mq_guide_name": guide_name}))
                    ch0 = self.QUEST_CONFIG["chapters"][0]
                    res.add_feedback(aid, self._render_text(ch0["intro"]) + "\n")

                    self._set_stage_key(prog, 0, 0)
                    stage0 = ch0["stages"][0]
                    self._ensure_objective_guide_for_stage(
                        env, world, agent, prog, stage0, res, force_here=True, announce=True
                    )
                    continue

            prog = self._progress[aid]
            guide_name = self._ensure_agent_guide_name(world, prog, aid)

            if prog.get("complete"):
                env.curr_agents_state["main_quest_progress"][aid] = prog
                continue

            # v1: per-agent event slice
            aevents = [e for e in res.events if getattr(e, "agent_id", None) == aid]

            # v1: metrics tracking (kills, lit areas, etc.)
            self._update_metrics_from_events(world, prog, aevents)

            # v1: boss HP reset on agent defeat
            self._reset_boss_hp_if_needed(world, aevents)

            # keep bosses spawned when needed
            self._ensure_bosses_for_progress(env, world, prog)

            # v2 modifier: debuff boss if binding verse paper is dropped at lair
            self._apply_ch2_boss_debuff_if_present(env, world, agent)

            ch_idx = int(prog.get("chapter", 0))
            st_idx = int(prog.get("stage", 0))

            if ch_idx >= len(self.QUEST_CONFIG["chapters"]):
                prog["complete"] = True
                env.curr_agents_state["main_quest_progress"][aid] = prog
                continue

            chapter = self.QUEST_CONFIG["chapters"][ch_idx]

            # if stage overflow, advance chapter
            if st_idx >= len(chapter["stages"]):
                prog["chapter"] = ch_idx + 1
                prog["stage"] = 0
                env.curr_agents_state["main_quest_progress"][aid] = prog

                if prog["chapter"] < len(self.QUEST_CONFIG["chapters"]):
                    next_ch = self.QUEST_CONFIG["chapters"][prog["chapter"]]
                    res.add_feedback(aid, "\n" + self._render_text(next_ch["intro"]) + "\n")

                    self._set_stage_key(prog, prog["chapter"], 0)
                    next_stage = next_ch["stages"][0]
                    self._ensure_objective_guide_for_stage(
                        env, world, agent, prog, next_stage, res, force_here=True, announce=True
                    )
                continue

            stage = chapter["stages"][st_idx]

            current_step = int(env.steps)
            stage_started = self._mark_stage_started(prog, ch_idx, st_idx, current_step)

            reminder_triggered = self._check_guide_reminder(env, world, agent, prog, stage, res, current_step)

            self._ensure_objective_guide_for_stage(
                env, world, agent, prog, stage, res,
                force_here=stage_started or reminder_triggered,
                announce=stage_started or reminder_triggered
            )

            if self._is_stage_done(env, world, agent, stage, aevents, prog):
                completion = self._render_text(stage.get("on_complete_feedback", ""), extra={"mq_guide_name": guide_name})
                completion = self._append_talk_hint(completion, guide_name)
                res.add_feedback(aid, f"\nStage completed. {completion}")

                # spawn boss if stage says so
                if stage.get("spawns_boss_key"):
                    self._ensure_named_boss_spawned(env, world, stage["spawns_boss_key"])

                # coins reward
                if stage.get("gives_coins"):
                    current_area_id = env.curr_agents_state["area"][aid]
                    self._spawn_object_in_area(aid, world, current_area_id, "obj_coin", int(stage["gives_coins"]), res)

                # object spawns on stage complete
                if stage.get("spawns_objects"):
                    current_area_id = env.curr_agents_state["area"][aid]
                    for obj_base_id, qty in stage["spawns_objects"].items():
                        self._spawn_object_in_area(aid, world, current_area_id, obj_base_id, int(qty), res)

                # chapter complete
                if stage.get("chapter_complete"):
                    prog["chapter"] = ch_idx + 1
                    prog["stage"] = 0
                    env.curr_agents_state["main_quest_progress"][aid] = prog

                    res.events.append(Event(
                        type="quest_stage_advanced",
                        agent_id=aid,
                        data={"chapter": prog["chapter"], "stage": prog["stage"], "chapter_complete": True},
                    ))

                    if prog["chapter"] < len(self.QUEST_CONFIG["chapters"]):
                        next_ch = self.QUEST_CONFIG["chapters"][prog["chapter"]]
                        res.add_feedback(aid, "\n" + self._render_text(next_ch["intro"]) + "\n")

                        self._set_stage_key(prog, prog["chapter"], 0)
                        next_stage = next_ch["stages"][0]
                        self._ensure_objective_guide_for_stage(
                            env, world, agent, prog, next_stage, res, force_here=True, announce=True
                        )
                    continue

                # quest complete
                if stage.get("quest_complete"):
                    prog["complete"] = True
                    env.curr_agents_state["main_quest_progress"][aid] = prog
                    res.events.append(Event(
                        type="quest_stage_advanced",
                        agent_id=aid,
                        data={"chapter": prog.get("chapter", 0), "stage": prog.get("stage", 0), "quest_complete": True},
                    ))
                    res.events.append(Event(type="main_quest_complete", agent_id=aid, data={}))
                    continue

                # advance stage
                prog["stage"] = st_idx + 1
                self._set_stage_key(prog, ch_idx, prog["stage"])
                env.curr_agents_state["main_quest_progress"][aid] = prog

                res.events.append(Event(
                    type="quest_stage_advanced",
                    agent_id=aid,
                    data={"chapter": prog["chapter"], "stage": prog["stage"]},
                ))

                # v1: stage transition guide teleport+announce
                if prog["stage"] < len(chapter["stages"]):
                    next_stage = chapter["stages"][prog["stage"]]
                    self._ensure_objective_guide_for_stage(
                        env, world, agent, prog, next_stage, res, force_here=True, announce=True
                    )

            env.curr_agents_state["main_quest_progress"][aid] = prog

    def _bootstrap(self, env, world) -> None:
        self._generated["ch1_library_area"] = self.CH1_LIBRARY_AREA
        self._generated["ch1_boss_candidate_areas"] = list(self.CH1_BOSS_CANDIDATE_AREAS)
        self._generated["ch1_boss_area"] = self.CH1_BOSS_AREA
        self._generated["ch1_explore_min"] = int(self.CH1_EXPLORE_MIN)
        # Individual Ch1 exploration areas
        self._generated["ch1_area_1"] = self.CH1_BOSS_CANDIDATE_AREAS[0]  # area_castle_armory
        self._generated["ch1_area_2"] = self.CH1_BOSS_CANDIDATE_AREAS[1]  # area_market_stalls
        self._generated["ch1_area_3"] = self.CH1_BOSS_CANDIDATE_AREAS[2]  # area_market_backstreets

        self._generated["ch2_contact_area"] = self.CH2_CONTACT_AREA
        self._generated["ch2_boss_candidate_areas"] = list(self.CH2_BOSS_CANDIDATE_AREAS)
        self._generated["ch2_boss_area"] = self.CH2_BOSS_AREA
        self._generated["ch2_explore_min"] = int(self.CH2_EXPLORE_MIN)
        # Individual Ch2 exploration areas
        self._generated["ch2_area_1"] = self.CH2_BOSS_CANDIDATE_AREAS[0]  # area_mines_shafts
        self._generated["ch2_area_2"] = self.CH2_BOSS_CANDIDATE_AREAS[1]  # area_woods_ridge
        self._generated["ch2_area_3"] = self.CH2_BOSS_CANDIDATE_AREAS[2]  # area_flats_farm

        self._generated["ch1_promise_line"] = "kindness"

    def _register_required_objects(self, world) -> None:
        if not hasattr(world, "objects") or not isinstance(world.objects, dict):
            return

        ObjCls = None
        for v in world.objects.values():
            if v is not None:
                ObjCls = v.__class__
                break

        for d in self.required_objects:
            oid = d.get("id")
            if not oid or oid in world.objects:
                continue
            try:
                obj = self._instantiate_object_like(ObjCls, d)
                world.objects[oid] = obj
            except Exception:
                world.objects[oid] = d

        # v1: build ingredient -> craftable mapping (if your object class supports craft_ingredients)
        ing_to_obj_map: Dict[str, set] = {}
        for oid, obj in world.objects.items():
            if not hasattr(obj, "craft_ingredients"):
                continue
            for ing_id in getattr(obj, "craft_ingredients", {}).keys():
                ing_to_obj_map.setdefault(ing_id, set()).add(oid)
        world.auxiliary["ing_to_obj_map"] = {k: list(v) for k, v in ing_to_obj_map.items()}

    def _instantiate_object_like(self, ObjCls, d: Dict[str, Any]):
        import inspect

        if ObjCls is None:
            raise RuntimeError("No object prototype class found.")

        craft = d.get("craft", {}) or {}
        normalized = dict(d)
        normalized.setdefault("level", 1)

        # v1 normalization
        if "attack" in normalized and "attack_power" not in normalized:
            normalized["attack_power"] = normalized["attack"]
        if "hp_restore" in normalized and "hp_increase" not in normalized:
            normalized["hp_increase"] = normalized["hp_restore"]

        if "craft_ingredients" not in normalized and isinstance(craft, dict):
            normalized["craft_ingredients"] = dict(craft.get("ingredients", {}) or {})
        if "craft_dependencies" not in normalized and isinstance(craft, dict):
            normalized["craft_dependencies"] = list(craft.get("dependencies", []) or [])

        sig = inspect.signature(ObjCls.__init__)
        params = {k for k in sig.parameters.keys() if k != "self"}

        kwargs = {}
        for k in params:
            if k in normalized:
                kwargs[k] = normalized[k]

        if "text" in params and "text" not in kwargs:
            kwargs["text"] = str(normalized.get("text", ""))

        if "quest" in params and "quest" not in kwargs:
            kwargs["quest"] = bool(normalized.get("quest", False))

        return ObjCls(**kwargs)

    def _register_required_npcs(self, world) -> None:
        aux = world.auxiliary
        aux.setdefault("npc_id_to_count", {})
        aux.setdefault("npc_name_to_id", {})
        aux.setdefault("container_id_to_count", {})
        aux.setdefault("writable_id_to_count", {})

        for d in self.required_npcs:
            nid = d["id"]
            if nid in world.npcs:
                continue

            atk = d.get("attack_power", d.get("base_attack_power", 0))
            hp = d.get("hp", d.get("base_hp", 100))

            inv = {}
            raw_inv = d.get("inventory", None)
            if isinstance(raw_inv, dict):
                inv = dict(raw_inv)
            else:
                objs = d.get("objects", None)
                if isinstance(objs, list):
                    for oid in objs:
                        if oid:
                            inv[oid] = int(inv.get(oid, 0) or 0) + 1

            world.npcs[nid] = NPC(
                type=d.get("type", "npc"),
                id=nid,
                name=d["name"],
                enemy=bool(d.get("enemy", False)),
                unique=bool(d.get("unique", True)),
                role=d.get("role", "quest"),
                level=int(d.get("level", 1) or 1),
                description=d.get("description", ""),
                attack_power=int(atk or 0),
                hp=int(hp or 0),
                base_hp=int(d.get("base_hp", 100) or 100),
                base_attack_power=int(d.get("base_attack_power", 0) or 0),
                coins=int(d.get("coins", 0) or 0),
                slope_hp=int(d.get("slope_hp", 0) or 0),
                slope_attack_power=int(d.get("slope_attack_power", 0) or 0),
                quest=bool(d.get("quest", True)),
                inventory=inv,
            )

    def _ensure_static_quest_npcs(self, env, world) -> None:
        rng = env.rng

        # Librarian
        if "ch1_librarian_npc" not in self._generated or self._generated.get("ch1_librarian_npc") not in world.npc_instances:
            inst = self._create_npc_instance(world, self.LIBRARIAN_BASE_NPC_ID, self._generated["ch1_library_area"], rng)
            if inst:
                self._generated["ch1_librarian_npc"] = inst
                self._generated["ch1_librarian_name"] = world.npc_instances[inst].name

        # Contact (merchant)
        if "ch2_contact_npc" not in self._generated or self._generated.get("ch2_contact_npc") not in world.npc_instances:
            inst = self._create_npc_instance(world, self.CONTACT_BASE_NPC_ID, self._generated["ch2_contact_area"], rng)
            if inst:
                self._generated["ch2_contact_npc"] = inst
                self._generated["ch2_contact_name"] = world.npc_instances[inst].name
                self._setup_contact_stock(world, inst)

        world.auxiliary["main_quest"]["generated"] = self._generated

    def _setup_contact_stock(self, world, npc_inst_id: str) -> None:
        npc = world.npc_instances.get(npc_inst_id)
        if not npc:
            return
        npc.role = "merchant"
        npc.coins = max(int(getattr(npc, "coins", 0)), 300)
        npc.inventory[self.TEAR_OBJ_ID] = max(int(npc.inventory.get(self.TEAR_OBJ_ID, 0)), 1)

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
            name_counts = aux.setdefault("npc_name_to_count", {})
            idx = int(name_counts.get(proto.name, 0))
            name_map = aux.get("npc_name_to_id", {})
            while f"{proto.name}_{idx}" in name_map:
                idx += 1
            name_counts[proto.name] = idx + 1

        inst_obj = proto.create_instance(idx, level=level, objects=world.objects, rng=rng)
        if isinstance(inst_obj, tuple):
            inst_obj = inst_obj[0]
        inst = inst_obj

        # Guarantee prototype inventory items survive create_instance randomization
        for oid, orig_count in proto.inventory.items():
            if orig_count > 0 and inst.inventory.get(oid, 0) < 1:
                inst.inventory[oid] = orig_count

        world.npc_instances[inst.id] = inst
        getattr(area, "npcs", []).append(inst.id)
        world.auxiliary.setdefault("npc_name_to_id", {})
        world.auxiliary["npc_name_to_id"][inst.name] = inst.id
        return inst.id

    def _ensure_bosses_for_progress(self, env, world, prog: Dict[str, Any]) -> None:
        ch = int(prog.get("chapter", 0))
        st = int(prog.get("stage", 0))
        # Ch1: stages 0-1 craft oak_rod/copper_sword, stages 2-8 explore/craft/write, stage 9 is boss
        if ch == 0 and st >= 9:
            self._ensure_named_boss_spawned(env, world, "ch1_boss_npc")
        # Ch2: stages 0-2 are show/trade/armor, stages 3-5 are explore, stage 6 is boss
        if ch == 1 and st >= 6:
            self._ensure_named_boss_spawned(env, world, "ch2_boss_npc")

    def _ensure_named_boss_spawned(self, env, world, boss_key: str) -> None:
        mq = world.auxiliary["main_quest"]
        if mq["boss_defeated"].get(boss_key, False):
            return

        inst_id = self._generated.get(boss_key)
        if inst_id and inst_id in world.npc_instances:
            return

        rng = env.rng

        if boss_key == "ch1_boss_npc":
            base_id = self.CH1_BOSS_BASE_NPC_ID
            area_id = self._generated["ch1_boss_area"]
            base_hp, base_atk = self.CH1_BOSS_HP, self.CH1_BOSS_ATK
        elif boss_key == "ch2_boss_npc":
            base_id = self.CH2_BOSS_BASE_NPC_ID
            area_id = self._generated["ch2_boss_area"]
            base_hp, base_atk = self.CH2_BOSS_HP, self.CH2_BOSS_ATK
        else:
            return

        inst = self._create_npc_instance(world, base_id, area_id, rng)
        if not inst:
            return

        boss = world.npc_instances[inst]
        boss.hp = int(base_hp)
        boss.attack_power = int(base_atk)

        self._generated[boss_key] = inst
        mq["boss_max_hp"][inst] = int(base_hp)
        mq["boss_mods"].setdefault(inst, {"base_max_hp": int(base_hp), "curr_max_hp": int(base_hp), "debuffed": False})
        world.auxiliary["main_quest"]["generated"] = self._generated

    def _reset_boss_hp_if_needed(self, world, events: list) -> None:
        mq = world.auxiliary.get("main_quest", {}) or {}
        boss_max_hp = mq.get("boss_max_hp", {}) or {}
        boss_mods = mq.get("boss_mods", {}) or {}

        for e in events:
            if getattr(e, "type", None) != "agent_defeated_in_combat":
                continue
            d = getattr(e, "data", {}) or {}
            npc_id = d.get("npc_id")
            if not npc_id or npc_id not in world.npc_instances:
                continue
            if npc_id not in boss_max_hp and npc_id not in boss_mods:
                continue

            if npc_id in boss_mods and "curr_max_hp" in boss_mods[npc_id]:
                world.npc_instances[npc_id].hp = int(boss_mods[npc_id]["curr_max_hp"])
            else:
                world.npc_instances[npc_id].hp = int(boss_max_hp.get(npc_id, world.npc_instances[npc_id].hp))

    def _apply_ch2_boss_debuff_if_present(self, env, world, agent: Agent) -> None:
        boss_id = self._generated.get("ch2_boss_npc")
        if not boss_id or boss_id not in world.npc_instances:
            return
        boss_area = self._generated.get("ch2_boss_area")
        if not boss_area:
            return
        if env.curr_agents_state["area"][agent.id] != boss_area:
            return

        area = world.area_instances.get(boss_area)
        if not area:
            return

        mq = world.auxiliary["main_quest"]
        mods = mq["boss_mods"].setdefault(
            boss_id,
            {"base_max_hp": int(self.CH2_BOSS_HP), "curr_max_hp": int(self.CH2_BOSS_HP), "debuffed": False},
        )
        if mods.get("debuffed", False):
            return

        target = str(self._generated.get("ch1_promise_line", "")).strip().lower()
        if not target:
            return

        # check if agent holds a paper with the binding verse
        found = False
        for oid in self._iter_agent_item_ids(world, agent):
            if oid in getattr(world, "writable_instances", {}):
                writable = world.writable_instances[oid]
                paper_text = str(getattr(writable, "text", "")).strip().lower()
                if target in paper_text:
                    found = True
                    break
        if not found:
            return

        boss = world.npc_instances[boss_id]
        boss.attack_power = min(int(getattr(boss, "attack_power", self.CH2_BOSS_ATK) or self.CH2_BOSS_ATK),
                                int(self.CH2_DEBUFFED_BOSS_ATK))

        # Treat the cap as the new "curr max hp" for reset logic (v1-style)
        mods["curr_max_hp"] = int(self.CH2_DEBUFFED_BOSS_HP_CAP)
        boss.hp = min(int(getattr(boss, "hp", self.CH2_BOSS_HP) or self.CH2_BOSS_HP),
                      int(mods["curr_max_hp"]))
        mods["debuffed"] = True

    def _init_agent_progress(self, agent_id: str) -> None:
        self._progress[agent_id] = {
            "chapter": 0,
            "stage": 0,
            "complete": False,
            # v1 metrics
            "mq_kills": [],
            "mq_lit_dark_areas": [],
            # v1 objective guide state
            "mq_guide_name": None,
            "mq_guide_inst": None,
            "mq_guide_area": None,
            "mq_stage_key": None,
        }

    def _update_metrics_from_events(self, world, prog: Dict[str, Any], events: list) -> None:
        if not events:
            return

        kills = prog.setdefault("mq_kills", [])
        lit = prog.setdefault("mq_lit_dark_areas", [])

        def add_lit(aid: str) -> None:
            if aid and aid not in lit:
                lit.append(aid)

        for e in events:
            etype = getattr(e, "type", None)

            if etype == "area_lighted":
                d = getattr(e, "data", {}) or {}
                area_id = d.get("area_id")
                area = world.area_instances.get(area_id) if area_id else None
                if area and (not bool(getattr(area, "light", True))):
                    add_lit(area_id)
                continue

            if etype == "npc_killed":
                d = getattr(e, "data", {}) or {}
                npc_id = d.get("npc_id")
                area_id = d.get("area_id")
                if not npc_id:
                    continue

                npc = world.npc_instances.get(npc_id)
                enemy = bool(getattr(npc, "enemy", False)) if npc else False
                role = str(getattr(npc, "role", "")) if npc else ""
                area = world.area_instances.get(area_id) if area_id else None
                lvl = int(getattr(area, "level", 1) or 1) if area else 1
                is_light = bool(getattr(area, "light", True)) if area else True

                kills.append({
                    "npc_id": npc_id,
                    "enemy": enemy,
                    "role": role,
                    "area_id": area_id,
                    "area_level": lvl,
                    "area_light": is_light,
                })
                continue

    def _set_stage_key(self, prog: Dict[str, Any], ch: int, st: int) -> None:
        prog["mq_stage_key"] = f"{int(ch)}:{int(st)}"

    def _mark_stage_started(self, prog: Dict[str, Any], ch: int, st: int, current_step: int) -> bool:
        key = f"{int(ch)}:{int(st)}"
        prev = str(prog.get("mq_stage_key", "") or "")
        if prev != key:
            prog["mq_stage_key"] = key
            prog["mq_stage_started_step"] = current_step
            prog["mq_last_reminder_step"] = current_step
            return True
        return False

    def _check_guide_reminder(
        self,
        env,
        world,
        agent,
        prog: Dict[str, Any],
        stage: Dict[str, Any],
        res: RuleResult,
        current_step: int,
    ) -> bool:
        last_reminder = int(prog.get("mq_last_reminder_step", 0))
        steps_since_reminder = current_step - last_reminder

        if steps_since_reminder < self.GUIDE_REMINDER_INTERVAL_STEPS:
            return False

        prog["mq_last_reminder_step"] = current_step
        guide_name = self._ensure_agent_guide_name(world, prog, agent.id)
        res.add_feedback(
            agent.id,
            f"\n{guide_name} appears nearby.\n"
            f"{guide_name} says: \"You seem stuck. Talk to me if you need a reminder of your current task.\"\n"
        )

        return True

    def _ensure_objective_name_map_for_agents(self, env, world) -> None:
        import random

        mq = world.auxiliary.setdefault("main_quest", {})
        name_map = mq.setdefault("objective_npc_names", {})

        run_seed = int(getattr(env, "seed", 0) or 0)
        mq["run_seed"] = run_seed

        agent_ids = [a.id for a in getattr(env, "agents", [])]
        agent_ids_sorted = sorted([str(aid) for aid in agent_ids])
        if agent_ids_sorted and all(aid in name_map for aid in agent_ids_sorted):
            return

        base_pool = list(self.OBJECTIVE_NPC_NAME_POOL)
        r = random.Random(run_seed ^ 0xA5A5A5A5)
        pool = base_pool[:]
        r.shuffle(pool)

        used = set(v for v in name_map.values() if isinstance(v, str))

        idx = 0
        for aid in agent_ids_sorted:
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
                name_map[aid] = "mira"
                used.add("mira")

        mq["objective_npc_names"] = name_map

    def _ensure_agent_guide_name(self, world, prog: Dict[str, Any], agent_id: str) -> str:
        mq = world.auxiliary.setdefault("main_quest", {})
        name_map = mq.setdefault("objective_npc_names", {})

        existing = prog.get("mq_guide_name")
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
                name_map[agent_id] = pool[0] if pool else "mira"

        prog["mq_guide_name"] = name_map[agent_id]
        return name_map[agent_id]

    def _ensure_objective_guide_for_stage(
        self,
        env,
        world,
        agent: Agent,
        prog: Dict[str, Any],
        stage: Dict[str, Any],
        res: RuleResult,
        force_here: bool,
        announce: bool,
    ) -> None:
        if self.OBJECTIVE_NPC_BASE_ID not in getattr(world, "npcs", {}):
            return

        guide_name = self._ensure_agent_guide_name(world, prog, agent.id)
        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances.get(current_area_id)
        if not current_area:
            return

        inst_id = prog.get("mq_guide_inst")
        if inst_id and inst_id not in world.npc_instances:
            inst_id = None
            prog["mq_guide_inst"] = None
            prog["mq_guide_area"] = None

        entered = False

        if not inst_id:
            inst_id = self._create_npc_instance(world, self.OBJECTIVE_NPC_BASE_ID, current_area_id, env.rng)
            if not inst_id:
                return
            prog["mq_guide_inst"] = inst_id
            prog["mq_guide_area"] = current_area_id
            entered = True
        else:
            if force_here:
                where = prog.get("mq_guide_area")
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

                    prog["mq_guide_area"] = current_area_id
                    entered = True

        self._set_npc_name(world, inst_id, guide_name)
        self._force_npc_invincible(world, inst_id)

        obj_text = self._render_text(stage.get("objective", ""), extra={"mq_guide_name": guide_name})
        self._set_npc_dialogue(world, inst_id, obj_text)

        if entered and announce:
            res.add_feedback(agent.id, f"{guide_name} enters the area.\n")

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

    def _append_talk_hint(self, feedback: str, guide_name: str) -> str:
        guide_name = (guide_name or "").strip() or "mira"
        hint = f"Talk to {guide_name} about what to do next.\n\n"
        if not feedback:
            return hint
        low = feedback.lower()
        if f"talk to {guide_name}".lower() in low:
            return feedback
        return feedback + hint

    # ============================================================
    # Rendering + conditions (ported superset from v1)
    # ============================================================
    def _render_text(self, text: str, extra: Optional[Dict[str, Any]] = None) -> str:
        def area_disp(key: str) -> str:
            return self._get_area_display_name_from_id(self._generated.get(key))

        def area_list_disp(list_key: str) -> str:
            ids = self._generated.get(list_key, [])
            if not isinstance(ids, list):
                return "unknown"
            names = [self._get_area_display_name_from_id(aid) for aid in ids]
            return "; ".join(names) if names else "unknown"

        fmt = {
            "ch1_library_area": area_disp("ch1_library_area"),
            "ch1_librarian_name": self._generated.get("ch1_librarian_name", "the librarian"),
            "ch1_promise_line": str(self._generated.get("ch1_promise_line", "???")),
            "ch1_boss_candidates": area_list_disp("ch1_boss_candidate_areas"),
            "ch1_explore_min": str(self._generated.get("ch1_explore_min", 2)),
            "ch1_boss_area": area_disp("ch1_boss_area"),
            "ch1_boss_name": "page_gnawer",
            # Individual Ch1 exploration areas
            "ch1_area_1": area_disp("ch1_area_1"),
            "ch1_area_2": area_disp("ch1_area_2"),
            "ch1_area_3": area_disp("ch1_area_3"),

            "ch2_contact_area": area_disp("ch2_contact_area"),
            "ch2_contact_name": self._generated.get("ch2_contact_name", "the contact"),
            "ch2_boss_candidates": area_list_disp("ch2_boss_candidate_areas"),
            "ch2_explore_min": str(self._generated.get("ch2_explore_min", 2)),
            "ch2_boss_area": area_disp("ch2_boss_area"),
            "ch2_boss_name": "root_warden",
            # Individual Ch2 exploration areas
            "ch2_area_1": area_disp("ch2_area_1"),
            "ch2_area_2": area_disp("ch2_area_2"),
            "ch2_area_3": area_disp("ch2_area_3"),
        }
        if isinstance(extra, dict):
            fmt.update(extra)

        try:
            return str(text or "").format(**fmt)
        except Exception:
            return str(text or "")

    def _get_area_display_name_from_id(self, area_id: str) -> str:
        if not area_id:
            return "unknown"
        if not self._world_ref:
            return str(area_id)
        area = self._world_ref.area_instances.get(area_id)
        if not area:
            return str(area_id)
        area_name = getattr(area, "name", area_id)
        place_id = self._area_to_place.get(area_id)
        if place_id and place_id in self._world_ref.place_instances:
            place_name = getattr(self._world_ref.place_instances[place_id], "name", place_id)
            return f"{place_name}, {area_name}"
        return str(area_name)

    def _is_stage_done(self, env, world, agent, stage: dict, events: list, prog: dict) -> bool:
        if "done_all" in stage:
            for cond in stage.get("done_all", []):
                if not self._cond_true(env, world, agent, cond, events, prog):
                    return False
            return True
        for cond in stage.get("done_any", []):
            if self._cond_true(env, world, agent, cond, events, prog):
                return True
        return False

    def _cond_true(self, env, world, agent, cond: dict, events: list, prog: dict) -> bool:
        kind = cond.get("kind")

        if kind == "event":
            etype = cond.get("type")
            for e in events:
                if getattr(e, "type", None) != etype:
                    continue
                d = getattr(e, "data", {}) or {}

                if etype == "npc_killed":
                    npc_key = cond.get("npc_key")
                    want = self._generated.get(npc_key) if npc_key else None
                    got = d.get("npc_id")
                    if want and got == want:
                        world.auxiliary["main_quest"]["boss_defeated"][npc_key] = True
                        return True
                    continue

                if etype == "object_bought":
                    npc_key = cond.get("npc_key")
                    want_npc = self._generated.get(npc_key) if npc_key else None
                    if want_npc and d.get("npc_id") != want_npc:
                        continue
                    want_obj = cond.get("obj_id")
                    if want_obj and d.get("obj_id") != want_obj:
                        continue
                    return True

                return True
            return False

        if kind == "state":
            stype = cond.get("type")

            if stype == "in_area":
                target_area = self._generated.get(cond.get("area_key"))
                return env.curr_agents_state["area"][agent.id] == target_area

            if stype == "in_area_with_npc":
                target_npc = self._generated.get(cond.get("npc_key"))
                if not target_npc:
                    return False
                current_area = env.curr_agents_state["area"][agent.id]
                area = world.area_instances.get(current_area)
                return bool(area and target_npc in getattr(area, "npcs", []))

            if stype == "visited_k_of_area_set":
                ids = self._generated.get(cond.get("area_set_key"), [])
                if not isinstance(ids, list):
                    return False
                k = self._generated.get(cond.get("k_key"), cond.get("k", 1))
                try:
                    k = int(k)
                except Exception:
                    k = 1
                visited = set(env.curr_agents_state.get("areas_visited", {}).get(agent.id, []))
                # Also include the current area (areas_visited is updated after step rules)
                current_area = env.curr_agents_state["area"].get(agent.id)
                if current_area:
                    visited.add(current_area)
                got = sum(1 for aid in ids if aid in visited)
                return got >= k

            if stype == "paper_text_equals_in_area":
                text_key = cond.get("text_key")
                target = str(self._generated.get(text_key, "")).strip().lower()
                if not target:
                    return False
                current_area_id = env.curr_agents_state["area"][agent.id]
                area = world.area_instances.get(current_area_id)
                if not area:
                    return False
                for oid in self._iter_area_object_ids(area):
                    if oid in getattr(world, "writable_instances", {}):
                        writable = world.writable_instances[oid]
                        paper_text = str(getattr(writable, "text", "")).strip().lower()
                        if target in paper_text:
                            return True
                return False

            if stype == "paper_text_in_possession":
                text_key = cond.get("text_key")
                target = str(self._generated.get(text_key, "")).strip().lower()
                if not target:
                    return False
                for oid in self._iter_agent_item_ids(world, agent):
                    if oid in getattr(world, "writable_instances", {}):
                        writable = world.writable_instances[oid]
                        paper_text = str(getattr(writable, "text", "")).strip().lower()
                        if target in paper_text:
                            return True
                return False

            if stype == "boss_already_defeated":
                boss_key = cond.get("boss_key") or cond.get("npc_key")
                return bool(world.auxiliary.get("main_quest", {}).get("boss_defeated", {}).get(boss_key, False))

            if stype == "has_any_of":
                base_objs = cond.get("base_objs", [])
                if not isinstance(base_objs, list) or not base_objs:
                    return False
                return any(self._count_base_item(world, agent, b) > 0 for b in base_objs)

            # --- v1 superset conditions kept for compatibility/future use ---
            if stype == "equipped_any_of":
                base_objs = cond.get("base_objs", [])
                if not isinstance(base_objs, list) or not base_objs:
                    return False
                return any(self._count_base_equipped(agent, b) > 0 for b in base_objs)

            if stype == "killed_n_enemies":
                n = int(cond.get("n", 1) or 1)
                enemy_only = bool(cond.get("enemy_only", False))
                roles = cond.get("roles")
                roles = set(roles) if isinstance(roles, list) else None
                in_dark_only = bool(cond.get("in_dark_only", False))
                min_area_level = int(cond.get("min_area_level", 0) or 0)

                kills = prog.get("mq_kills", []) or []
                count = 0
                for krec in kills:
                    if enemy_only and not bool(krec.get("enemy", False)):
                        continue
                    if roles is not None and str(krec.get("role", "")) not in roles:
                        continue
                    if in_dark_only and bool(krec.get("area_light", True)):
                        continue
                    if min_area_level and int(krec.get("area_level", 1) or 1) < min_area_level:
                        continue
                    count += 1
                    if count >= n:
                        return True
                return False

            if stype == "killed_one_of_each":
                roles = cond.get("roles", [])
                if not isinstance(roles, list) or not roles:
                    return False
                enemy_only = bool(cond.get("enemy_only", False))
                kills = prog.get("mq_kills", []) or []
                found = set()
                for krec in kills:
                    if enemy_only and not bool(krec.get("enemy", False)):
                        continue
                    r = str(krec.get("role", ""))
                    if r in roles:
                        found.add(r)
                return all(r in found for r in roles)

            if stype == "lit_k_dark_areas":
                k = int(cond.get("k", 1) or 1)
                lit = prog.get("mq_lit_dark_areas", []) or []
                uniq = set([a for a in lit if a])
                return len(uniq) >= k

            if stype == "writable_text_equals_in_inventory":
                text_key = cond.get("text_key")
                target = str(self._generated.get(text_key, "")).strip().lower()
                if not target:
                    return False
                for oid in self._iter_agent_item_ids(world, agent):
                    if oid in getattr(world, "writable_instances", {}):
                        writable = world.writable_instances[oid]
                        paper_text = str(getattr(writable, "text", "")).strip().lower()
                        if target in paper_text:
                            return True
                return False

            if stype == "writable_text_equals_in_area":
                text_key = cond.get("text_key")
                target = str(self._generated.get(text_key, "")).strip().lower()
                if not target:
                    return False
                current_area_id = env.curr_agents_state["area"][agent.id]
                area = world.area_instances.get(current_area_id)
                if not area:
                    return False
                for oid in self._iter_area_object_ids(area):
                    if oid in getattr(world, "writable_instances", {}):
                        writable = world.writable_instances[oid]
                        paper_text = str(getattr(writable, "text", "")).strip().lower()
                        if target in paper_text:
                            return True
                return False

        return False

    # ============================================================
    # Object spawning + iter helpers (v1 superset)
    # ============================================================
    def _spawn_object_in_area(self, agent_id: str, world, area_id: str, obj_id: str, count: int, res) -> None:
        area = world.area_instances.get(area_id)
        if not area or count <= 0:
            return
        if obj_id not in world.objects:
            return

        objs = getattr(area, "objects", None)
        if objs is None:
            return

        if isinstance(objs, dict):
            objs[obj_id] = int(objs.get(obj_id, 0) or 0) + int(count)
        else:
            try:
                for _ in range(int(count)):
                    objs.append(obj_id)
            except Exception:
                return

        try:
            res.track_spawn(agent_id, obj_id, count, dst=res.tloc("area", area_id))
        except Exception:
            pass
        res.events.append(Event(type="quest_spawn", agent_id=agent_id, data={"obj_id": obj_id, "count": count, "area_id": area_id}))

    def _iter_area_object_ids(self, area) -> List[str]:
        objs = getattr(area, "objects", None)
        if objs is None:
            return []
        if isinstance(objs, dict):
            return list(objs.keys())
        try:
            return list(objs)
        except Exception:
            return []

    def _iter_agent_item_ids(self, world, agent: Agent):
        seen = set()

        items_in_hands = getattr(agent, "items_in_hands", {}) or {}
        equipped = getattr(agent, "equipped_items_in_limb", {}) or {}

        def add(oid: str):
            if oid and oid not in seen:
                seen.add(oid)

        if isinstance(items_in_hands, dict):
            for oid in items_in_hands.keys():
                add(oid)
        else:
            for oid in items_in_hands:
                add(oid)

        if isinstance(equipped, dict):
            for oid in equipped.keys():
                add(oid)
        else:
            for oid in equipped:
                add(oid)

        inv = getattr(agent, "inventory", None)
        if inv and getattr(inv, "container", None):
            inv_items = getattr(inv, "items", {}) or {}
            if isinstance(inv_items, dict):
                for oid in inv_items.keys():
                    add(oid)

        hand_keys = items_in_hands.keys() if isinstance(items_in_hands, dict) else items_in_hands
        for oid in hand_keys:
            if oid in getattr(world, "container_instances", {}):
                cinv = world.container_instances[oid].inventory
                for coid in (cinv or {}).keys():
                    add(coid)

        return seen

    def _count_base_equipped(self, agent: Agent, base_obj_id: str) -> int:
        equipped = getattr(agent, "equipped_items_in_limb", {}) or {}
        total = 0
        if isinstance(equipped, dict):
            for oid, cnt in equipped.items():
                if get_def_id(oid) == base_obj_id:
                    total += int(cnt)
        else:
            for oid in equipped:
                if get_def_id(oid) == base_obj_id:
                    total += 1
        return total

    def _count_base_in_hands(self, agent: Agent, base_obj_id: str) -> int:
        items_in_hands = getattr(agent, "items_in_hands", {}) or {}
        total = 0
        if isinstance(items_in_hands, dict):
            for oid, cnt in items_in_hands.items():
                if get_def_id(oid) == base_obj_id:
                    total += int(cnt)
        else:
            for oid in items_in_hands:
                if get_def_id(oid) == base_obj_id:
                    total += 1
        return total

    def _count_base_item(self, world, agent, base_obj_id: str) -> int:
        if not base_obj_id:
            return 0
        total = 0

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
            for oid, cnt in (inv_items or {}).items():
                if get_def_id(oid) == base_obj_id:
                    total += int(cnt)

        # containers held in hands
        hand_keys = items_in_hands.keys() if isinstance(items_in_hands, dict) else items_in_hands
        for oid in hand_keys:
            if oid in getattr(world, "container_instances", {}):
                cinv = world.container_instances[oid].inventory
                for coid, cnt in (cinv or {}).items():
                    if get_def_id(coid) == base_obj_id:
                        total += int(cnt)

        return total

    # Optional v1 helpers kept (useful if you add stage rewards later)
    def _ensure_reward_available(self, agent_id: str, world, area_id: str, obj_id: str, res) -> None:
        if self._area_has_base_object(world, area_id, obj_id):
            return
        self._spawn_object_in_area(agent_id, world, area_id, obj_id, 1, res)

    def _area_has_base_object(self, world, area_id: str, base_obj_id: str) -> bool:
        area = world.area_instances.get(area_id)
        if not area:
            return False
        objs = getattr(area, "objects", None)
        if isinstance(objs, dict):
            if base_obj_id in objs and int(objs.get(base_obj_id, 0) or 0) > 0:
                return True
            for oid in objs.keys():
                if get_def_id(oid) == base_obj_id:
                    return True
            return False
        try:
            for oid in list(objs) if objs is not None else []:
                if get_def_id(oid) == base_obj_id:
                    return True
        except Exception:
            pass
        return False

    def _ensure_tear_has_value(self, world) -> None:
        if self.TEAR_OBJ_ID in getattr(world, "objects", {}):
            obj = world.objects[self.TEAR_OBJ_ID]
            try:
                if getattr(obj, "value", None) is None:
                    obj.value = int(self.TEAR_VALUE)
            except Exception:
                pass

