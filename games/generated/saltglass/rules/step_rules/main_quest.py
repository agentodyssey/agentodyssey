import copy
import math
from utils import *
from games.generated.saltglass.rule import BaseStepRule, RuleContext, RuleResult, Event
from typing import Dict, Optional, Set, Tuple, List, Any
from games.generated.saltglass.world import NPC, Object, Area, Place, Path, Container, Writable
from games.generated.saltglass.agent import Agent
from tools.logger import get_logger


class MainQuestStepRule(BaseStepRule):
    name = "main_quest_step"
    description = "Main quest that provides main storyline and goals with objective guide NPC."
    priority = 2

    OBJECTIVE_NPC_BASE_ID = "npc_quest_wayfarer_guide"
    OBJECTIVE_NPC_NAME_POOL = [
        "alex_smith", "bob_johnson", "chris_brown",
        "david_miller", "emma_wilson", "james_taylor",
        "lucas_anderson", "mia_thompson", "noah_martin", "olivia_clark",
    ]
    OBJECTIVE_NPC_MAX_HP = 10**9
    OBJECTIVE_NPC_ATK = 10**6

    required_objects: List[Dict[str, Any]] = [
        {'id': 'obj_survey_seal', 'name': 'survey_seal', 'category': 'material', 'usage': 'quest', 'value': 40, 'size': 1, 'description': 'A stamped brass token used by old observatory surveyors to certify a completed chart.', 'level': 2},
    ]

    required_npcs: List[Dict[str, Any]] = [
        {'id': 'npc_quest_guide', 'name': 'sere', 'enemy': False, 'unique': True, 'role': 'guide', 'quest': True, 'description': 'A weathered guide wrapped in saltcloth, keeping notes on the angles of fallen towers.', 'base_attack_power': 0, 'base_hp': 40, 'objects': []},

        # objective guide (invincible)
        {"type": "npc", "id": OBJECTIVE_NPC_BASE_ID, "name": "mira",
        "enemy": False, "unique": True, "role": "guide", "quest": True,
        "description": "",
        "base_attack_power": OBJECTIVE_NPC_ATK, "slope_attack_power": 0,
        "base_hp": OBJECTIVE_NPC_MAX_HP, "slope_hp": 0,
        "objects": []},
    ]

    undistributable_objects: List[str] = ['obj_survey_seal']

    QUEST_CONFIG: Dict[str, Any] = {
        'title': 'The Salt Oath',
        'intro': (
            'Across the endless salt flat, toppled observatories lie half-buried like broken compasses. {mq_guide_name} has heard that one surviving chart can still align the fallen spire, but only if an oath is carried by hand and spoken through deed. Seek the scholar who waits among the ruins, learn the old phrase, and restore direction to Saltreach.'
        ),
        'chapters': [
            {
                'id': 'chapter_1',
                'title': 'Map of the Fallen Sky',
                'intro': (
                    '{mq_guide_name} sends you to find {ch1_guide_name}, a patient scholar in the near ruins of Saltreach. She claims one chart, properly marked and delivered, can tell the salt flat where north still lives.'
                ),
                'stages': [
                    {
                        'id': 'ch1_restore_chart',
                        'objective': (
                            'Help {ch1_guide_name} prepare and deliver a certified chart to the toppled observatory.'
                        ),
                        'chapter_complete': True,
                        'quest_complete': True,
                        'requires': ['ch1_learn_oath', 'ch1_prepare_delivery'],
                        'children': ['ch1_learn_oath', 'ch1_prepare_delivery'],
                    },
                    {
                        'id': 'ch1_learn_oath',
                        'objective': "Learn the old surveyor's oath from {ch1_guide_name}.",
                        'requires': ['ch1_find_guide', 'ch1_record_oath'],
                        'children': ['ch1_find_guide', 'ch1_record_oath'],
                    },
                    {
                        'id': 'ch1_find_guide',
                        'objective': 'Find {ch1_guide_name} and speak with her in the near ruins.',
                        'hint': 'Stay close to the spawn-side ruins and look for a non-hostile figure.',
                        'done_any': [
                            {
                                'kind': 'state',
                                'type': 'in_area_with_npc',
                                'npc_key': 'ch1_guide_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            '{ch1_guide_name} brushes salt from a cracked lens and recites the oath used by the old surveyors: "THE SKY REMEMBERS". She tells you to write it down exactly before carrying any chart onward.'
                        ),
                    },
                    {
                        'id': 'ch1_record_oath',
                        'objective': "Write the surveyor's oath onto chart paper.",
                        'hint': 'Use a pen and paper. Record the exact phrase you learned from {ch1_guide_name}.',
                        'done_any': [
                            {
                                'kind': 'state',
                                'type': 'paper_text_equals_in_hands',
                                'text_key': 'ch1_oath_text',
                            },
                        ],
                        'on_complete_feedback': (
                            'The words sit dark on the page. {ch1_guide_name} nods: a chart without the oath is only scrap, but a chart bearing it can still be witnessed.'
                        ),
                    },
                    {
                        'id': 'ch1_prepare_delivery',
                        'objective': 'Gather what is needed to certify the chart and place it in the observatory.',
                        'requires': ['ch1_get_supplies', 'ch1_deliver_chart'],
                        'children': ['ch1_get_supplies', 'ch1_deliver_chart'],
                    },
                    {
                        'id': 'ch1_get_supplies',
                        'objective': 'Acquire the tools and seal needed for the certified chart.',
                        'requires': ['ch1_buy_paper', 'ch1_buy_seal'],
                        'children': ['ch1_buy_paper', 'ch1_buy_seal'],
                    },
                    {
                        'id': 'ch1_buy_paper',
                        'objective': 'Buy chart paper from vesin.',
                        'hint': "Find the merchant in the near area and purchase the writable sheet you'll need.",
                        'done_any': [
                            {
                                'kind': 'event',
                                'type': 'object_bought',
                                'obj_id': 'obj_paper',
                            },
                        ],
                        'on_complete_feedback': 'You secure a sheet of chart paper fit for the old notation.',
                    },
                    {
                        'id': 'ch1_buy_seal',
                        'objective': 'Obtain the survey seal from vesin.',
                        'hint': 'The merchant keeps odd salvage. If he has a brass survey token, buy it.',
                        'done_any': [
                            {
                                'kind': 'custom',
                                'type': 'has_quest_item_by_base_or_instance',
                                'description': 'Player possesses the survey_seal quest item.',
                                'implementation_hint': (
                                    'Check player hands, inventory, and equipped container contents for any object with id obj_survey_seal.'
                                ),
                            },
                        ],
                        'on_complete_feedback': 'The heavy brass seal rests in your palm, stamped with a faded observatory ring.',
                    },
                    {
                        'id': 'ch1_deliver_chart',
                        'objective': (
                            'Bring the written chart and the survey seal to the toppled observatory and leave them there together.'
                        ),
                        'hint': (
                            'Travel to the observatory-side ruin after preparing both items. The chart must still bear the oath.'
                        ),
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'paper_with_text_dropped_in_area',
                                'description': (
                                    'A chart paper bearing the learned oath has been dropped in the target observatory area.'
                                ),
                                'implementation_hint': (
                                    'Inspect objects on the ground in the target area; succeed if any writable instance based on obj_paper has text exactly equal to generated text ch1_oath_text.'
                                ),
                                'text_key': 'ch1_oath_text',
                                'area_key': 'ch1_delivery_area',
                            },
                            {
                                'kind': 'custom',
                                'type': 'object_in_area',
                                'description': 'The survey seal has been dropped in the target observatory area.',
                                'implementation_hint': (
                                    'Check whether an object instance with base id obj_survey_seal exists among the objects in the target area.'
                                ),
                                'area_key': 'ch1_delivery_area',
                                'obj_key': 'ch1_survey_seal_obj',
                            },
                        ],
                        'on_complete_feedback': (
                            'You leave the sworn chart beside the brass seal in the shadow of the fallen observatory. Salt wind turns, hissing through broken rings like a compass finding its first true line in years.'
                        ),
                    },
                ],
                'root_goal': 'ch1_restore_chart',
            },
        ],
    }

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
        mq.setdefault("boss_max_hp", {})  # boss_inst_id -> int
        mq.setdefault("boss_defeated", {})  # boss_key -> bool
        mq.setdefault("boss_mods", {})  # boss_inst_id -> dict

        if not self._initialized:
            self._world_ref = world
            self._area_to_place = world.auxiliary.get("area_to_place", {}) or {}
            if not self._area_to_place:
                for place_id, place in world.place_instances.items():
                    for area_id in place.areas:
                        self._area_to_place[area_id] = place_id
                world.auxiliary["area_to_place"] = self._area_to_place

            persisted = mq.get("generated")
            if isinstance(persisted, dict) and persisted:
                self._generated = persisted
            else:
                self._bootstrap(env, world)
                mq["generated"] = self._generated

            self._register_required_npcs(world)
            self._ensure_static_quest_npcs(env, world)
            self._ensure_heart_has_value(world)
            self._initialized = True

        for agent in env.agents:
            aid = agent.id

            if tutorial_enabled and tutorial_room and not bool(tutorial_room.get("removed", False)):
                continue

            env.curr_agents_state["main_quest_progress"].setdefault(aid, None)

            if aid not in self._progress:
                persisted_prog = env.curr_agents_state["main_quest_progress"].get(aid)
                if isinstance(persisted_prog, dict) and persisted_prog:
                    self._progress[aid] = persisted_prog
                    prog = self._progress[aid]
                    # ensure DAG fields exist for older saves
                    prog.setdefault("completed_stages", {})
                    prog.setdefault("active_stages", {})
                    prog.setdefault("dag_announced", {})
                    guide_name = self._ensure_agent_guide_name(world, prog, aid)
                    env.curr_agents_state["main_quest_progress"][aid] = prog
                else:
                    self._init_agent_progress(aid)

                    prog = self._progress[aid]
                    guide_name = self._ensure_agent_guide_name(world, prog, aid)
                    env.curr_agents_state["main_quest_progress"][aid] = prog

                    res.add_feedback(aid, self._render_text(self.QUEST_CONFIG["intro"], extra={"mq_guide_name": guide_name}))
                    ch = self.QUEST_CONFIG["chapters"][0]
                    res.add_feedback(aid, self._render_text(ch["intro"] + "\n", extra={"mq_guide_name": guide_name}))

                    if self._is_chapter_dag(ch):
                        # DAG mode: activate root stages
                        ch_id = ch["id"]
                        prog["completed_stages"].setdefault(ch_id, [])
                        prog["dag_announced"].setdefault(ch_id, [])
                        active_ids = self._compute_active_stages(ch, set(prog["completed_stages"][ch_id]))
                        prog["active_stages"][ch_id] = active_ids

                        # Announce root goal if present (goal tree mode)
                        root_goal_id = ch.get("root_goal")
                        if root_goal_id:
                            root_stage = self._get_stage_by_id(ch, root_goal_id)
                            if root_stage:
                                root_obj = self._render_text(root_stage.get("objective", ""), extra={"mq_guide_name": guide_name})
                                res.add_feedback(aid, f"\nChapter goal: {root_obj}\n")

                        active_stage_dicts = [self._get_stage_by_id(ch, sid) for sid in active_ids]
                        active_stage_dicts = [s for s in active_stage_dicts if s]
                        self._ensure_objective_guide_for_stages(env, world, agent, prog, active_stage_dicts, res, force_here=True, announce=True)
                        prog["dag_announced"][ch_id] = list(active_ids)
                    else:
                        self._set_stage_key(prog, 0, 0)
                        stage0 = ch["stages"][0]
                        self._ensure_objective_guide_for_stage(env, world, agent, prog, stage0, res, force_here=True, announce=True)
                    continue

            prog = self._progress[aid]
            guide_name = self._ensure_agent_guide_name(world, prog, aid)

            if prog.get("complete"):
                env.curr_agents_state["main_quest_progress"][aid] = prog
                continue

            current_area_id = env.curr_agents_state["area"][aid]
            self._update_visited(prog, current_area_id)

            aevents = [e for e in res.events if getattr(e, "agent_id", None) == aid]

            self._reset_boss_hp_if_needed(world, aevents)
            self._ensure_bosses_for_progress(env, world, prog)
            self._apply_fire_boss_modifier(env, world, agent)

            ch_idx = int(prog.get("chapter", 0))

            if ch_idx >= len(self.QUEST_CONFIG["chapters"]):
                prog["complete"] = True
                env.curr_agents_state["main_quest_progress"][aid] = prog
                continue

            chapter = self.QUEST_CONFIG["chapters"][ch_idx]

            if self._is_chapter_dag(chapter):
                self._apply_dag_chapter(env, world, agent, prog, chapter, ch_idx, aevents, current_area_id, guide_name, res)
                # Save goal tree visualization after DAG processing
                completed_set = set(prog.get("completed_stages", {}).get(chapter["id"], []))
                active_ids = prog.get("active_stages", {}).get(chapter["id"], [])
                self._save_goal_tree_png(env, chapter, ch_idx, completed_set, active_ids, aid, env.steps)
            else:
                self._apply_linear_chapter(env, world, agent, prog, chapter, ch_idx, aevents, current_area_id, guide_name, res)

            env.curr_agents_state["main_quest_progress"][aid] = prog

    # Linear chapter logic (original behaviour, fully preserved)
    def _apply_linear_chapter(self, env, world, agent, prog, chapter, ch_idx, aevents, current_area_id, guide_name, res) -> None:
        aid = agent.id
        st_idx = int(prog.get("stage", 0))

        if st_idx >= len(chapter["stages"]):
            self._advance_to_next_chapter(env, world, agent, prog, ch_idx, guide_name, res)
            return

        stage = chapter["stages"][st_idx]

        stage_started = self._mark_stage_started(prog, ch_idx, st_idx)
        self._ensure_objective_guide_for_stage(
            env, world, agent, prog, stage, res,
            force_here=stage_started,
            announce=stage_started
        )

        if self._is_stage_done(env, world, agent, stage, aevents, prog):
            self._handle_stage_completion(env, world, agent, prog, chapter, ch_idx, st_idx, stage, aevents, current_area_id, guide_name, res)

    # DAG chapter logic (multi-goal / nested prerequisites)
    def _apply_dag_chapter(self, env, world, agent, prog, chapter, ch_idx, aevents, current_area_id, guide_name, res) -> None:
        aid = agent.id
        ch_id = chapter["id"]

        prog["completed_stages"].setdefault(ch_id, [])
        prog["active_stages"].setdefault(ch_id, [])
        prog["dag_announced"].setdefault(ch_id, [])

        completed_set = set(prog["completed_stages"][ch_id])

        # recompute active stages (in case of resume)
        active_ids = self._compute_active_stages(chapter, completed_set)
        prog["active_stages"][ch_id] = active_ids

        # announce any newly active stages
        announced = set(prog["dag_announced"][ch_id])
        new_active = [sid for sid in active_ids if sid not in announced]
        if new_active:
            new_stage_dicts = [self._get_stage_by_id(chapter, sid) for sid in new_active]
            new_stage_dicts = [s for s in new_stage_dicts if s]
            # update guide with ALL active objectives
            all_active_dicts = [self._get_stage_by_id(chapter, sid) for sid in active_ids]
            all_active_dicts = [s for s in all_active_dicts if s]
            self._ensure_objective_guide_for_stages(env, world, agent, prog, all_active_dicts, res, force_here=True, announce=True)
            prog["dag_announced"][ch_id] = list(set(prog["dag_announced"][ch_id]) | set(new_active))
        else:
            # keep guide dialogue updated even when no new stages
            all_active_dicts = [self._get_stage_by_id(chapter, sid) for sid in active_ids]
            all_active_dicts = [s for s in all_active_dicts if s]
            if all_active_dicts:
                self._ensure_objective_guide_for_stages(env, world, agent, prog, all_active_dicts, res, force_here=False, announce=False)

        # check all active stages for completion (skip conditionless — cascade below)
        newly_completed: List[dict] = []
        for sid in list(active_ids):
            stage = self._get_stage_by_id(chapter, sid)
            if not stage:
                continue
            # Conditionless stages (goal tree internal nodes) are handled by
            # the cascade loop below so they get uniform "Sub-goal completed"
            # messages and avoid double-processing.
            if not stage.get("done_any") and not stage.get("done_all"):
                continue
            if self._is_stage_done(env, world, agent, stage, aevents, prog):
                newly_completed.append(stage)

        # process completions
        for stage in newly_completed:
            sid = stage["id"]
            obj_text = self._render_text(stage.get("objective", sid), extra={"mq_guide_name": guide_name})
            completion = self._render_text(stage.get("on_complete_feedback", ""), extra={"mq_guide_name": guide_name})
            completion = self._append_talk_hint(completion, guide_name)
            res.add_feedback(aid, f"\nStage completed: {obj_text[:60]}\n{completion}")

            if stage.get("spawns_boss_key"):
                self._ensure_named_boss_spawned(env, world, stage["spawns_boss_key"])

            if stage.get("gives_coins"):
                self._spawn_object_in_area(aid, world, current_area_id, "obj_coin", int(stage["gives_coins"]), res)

            # mark completed
            if sid not in prog["completed_stages"][ch_id]:
                prog["completed_stages"][ch_id].append(sid)

            res.events.append(Event(
                type="quest_stage_advanced",
                agent_id=aid,
                data={"chapter": ch_idx, "stage": sid, "dag": True},
            ))

        # recompute active set (always — needed for cascade even when nothing
        # completed this step, e.g. conditionless nodes active from start)
        completed_set = set(prog["completed_stages"][ch_id])
        active_ids = self._compute_active_stages(chapter, completed_set)
        prog["active_stages"][ch_id] = active_ids

        # check for chapter / quest completion
        for stage in newly_completed:
            if stage.get("quest_complete"):
                prog["complete"] = True
                env.curr_agents_state["main_quest_progress"][aid] = prog
                res.events.append(Event(type="main_quest_complete", agent_id=aid, data={}))
                return

            if stage.get("chapter_complete"):
                self._advance_to_next_chapter(env, world, agent, prog, ch_idx, guide_name, res)
                return

        # Cascade: auto-complete conditionless parent nodes whose children
        # are all done.  Also runs on first step to handle conditionless
        # nodes with no requires.
        MAX_CASCADE = 50  # safety guard
        for _ in range(MAX_CASCADE):
            auto_completed: List[dict] = []
            for sid in list(active_ids):
                stage = self._get_stage_by_id(chapter, sid)
                if not stage:
                    continue
                if not stage.get("done_any") and not stage.get("done_all"):
                    auto_completed.append(stage)

            if not auto_completed:
                break

            for stage in auto_completed:
                sid = stage["id"]
                obj_text = self._render_text(stage.get("objective", ""), extra={"mq_guide_name": guide_name})
                feedback = self._render_text(stage.get("on_complete_feedback", ""), extra={"mq_guide_name": guide_name})
                msg = f"\nAll sub-tasks done — goal achieved: {obj_text}"
                if feedback:
                    msg += f" {feedback}"
                res.add_feedback(aid, msg)

                if stage.get("gives_coins"):
                    self._spawn_object_in_area(aid, world, current_area_id, "obj_coin", int(stage["gives_coins"]), res)

                if sid not in prog["completed_stages"][ch_id]:
                    prog["completed_stages"][ch_id].append(sid)

                res.events.append(Event(
                    type="quest_stage_advanced",
                    agent_id=aid,
                    data={"chapter": ch_idx, "stage": sid, "dag": True, "auto_complete": True},
                ))

                if stage.get("quest_complete"):
                    prog["complete"] = True
                    env.curr_agents_state["main_quest_progress"][aid] = prog
                    res.events.append(Event(type="main_quest_complete", agent_id=aid, data={}))
                    return

                if stage.get("chapter_complete"):
                    self._advance_to_next_chapter(env, world, agent, prog, ch_idx, guide_name, res)
                    return

            # recompute after cascade step
            completed_set = set(prog["completed_stages"][ch_id])
            active_ids = self._compute_active_stages(chapter, completed_set)
            prog["active_stages"][ch_id] = active_ids

        if not newly_completed:
            return

        # always refresh guide dialogue after completions (remove completed stages)
        all_active_dicts = [self._get_stage_by_id(chapter, sid) for sid in active_ids]
        all_active_dicts = [s for s in all_active_dicts if s]

        # announce newly unlocked stages
        new_unlocked = [sid for sid in active_ids if sid not in set(prog["dag_announced"][ch_id])]
        if new_unlocked:
            # update guide dialogue + move guide here
            self._ensure_objective_guide_for_stages(env, world, agent, prog, all_active_dicts, res, force_here=True, announce=False)
            prog["dag_announced"][ch_id] = list(set(prog["dag_announced"][ch_id]) | set(new_unlocked))
            for sid in new_unlocked:
                unlocked_stage = self._get_stage_by_id(chapter, sid)
                if unlocked_stage:
                    obj_text = self._render_text(unlocked_stage.get("objective", ""), extra={"mq_guide_name": guide_name})
                    res.add_feedback(aid, f"\nNew objective unlocked: {obj_text}")
        else:
            # no new stages, but still refresh guide to remove completed ones
            self._ensure_objective_guide_for_stages(env, world, agent, prog, all_active_dicts, res, force_here=False, announce=False)

    # Stage completion handler (shared by linear mode)
    def _handle_stage_completion(self, env, world, agent, prog, chapter, ch_idx, st_idx, stage, aevents, current_area_id, guide_name, res) -> None:
        aid = agent.id
        completion = self._render_text(stage.get("on_complete_feedback", ""), extra={"mq_guide_name": guide_name})
        completion = self._append_talk_hint(completion, guide_name)
        res.add_feedback(aid, f"\nStage completed. {completion}")

        if stage.get("spawns_boss_key"):
            self._ensure_named_boss_spawned(env, world, stage["spawns_boss_key"])

        if stage.get("gives_coins"):
            self._spawn_object_in_area(aid, world, current_area_id, "obj_coin", int(stage["gives_coins"]), res)

        if stage.get("chapter_complete"):
            self._advance_to_next_chapter(env, world, agent, prog, ch_idx, guide_name, res)
            return

        if stage.get("quest_complete"):
            prog["complete"] = True
            env.curr_agents_state["main_quest_progress"][aid] = prog
            res.events.append(Event(type="main_quest_complete", agent_id=aid, data={}))
            return

        prog["stage"] = st_idx + 1
        env.curr_agents_state["main_quest_progress"][aid] = prog
        if prog["stage"] < len(chapter["stages"]):
            next_stage = chapter["stages"][prog["stage"]]
            self._set_stage_key(prog, ch_idx, prog["stage"])
            self._ensure_objective_guide_for_stage(env, world, agent, prog, next_stage, res, force_here=True, announce=True)

        res.events.append(Event(
            type="quest_stage_advanced",
            agent_id=aid,
            data={"chapter": prog["chapter"], "stage": prog["stage"]},
        ))

    # Chapter advancement (shared by linear and DAG modes)
    def _advance_to_next_chapter(self, env, world, agent, prog, ch_idx, guide_name, res) -> None:
        aid = agent.id
        prog["chapter"] = ch_idx + 1
        prog["stage"] = 0
        env.curr_agents_state["main_quest_progress"][aid] = prog

        if prog["chapter"] < len(self.QUEST_CONFIG["chapters"]):
            next_ch = self.QUEST_CONFIG["chapters"][prog["chapter"]]
            res.add_feedback(aid, f"\n{self._render_text(next_ch['intro'], extra={'mq_guide_name': guide_name})}")

            if self._is_chapter_dag(next_ch):
                ch_id = next_ch["id"]
                prog["completed_stages"].setdefault(ch_id, [])
                prog["dag_announced"].setdefault(ch_id, [])
                active_ids = self._compute_active_stages(next_ch, set(prog["completed_stages"][ch_id]))
                prog["active_stages"][ch_id] = active_ids

                # Announce root goal if present (goal tree mode)
                root_goal_id = next_ch.get("root_goal")
                if root_goal_id:
                    root_stage = self._get_stage_by_id(next_ch, root_goal_id)
                    if root_stage:
                        root_obj = self._render_text(root_stage.get("objective", ""), extra={"mq_guide_name": guide_name})
                        res.add_feedback(aid, f"Chapter goal: {root_obj}")

                active_stage_dicts = [self._get_stage_by_id(next_ch, sid) for sid in active_ids]
                active_stage_dicts = [s for s in active_stage_dicts if s]
                self._ensure_objective_guide_for_stages(env, world, agent, prog, active_stage_dicts, res, force_here=True, announce=True)
                prog["dag_announced"][ch_id] = list(active_ids)
            else:
                self._set_stage_key(prog, prog["chapter"], 0)
                next_stage = next_ch["stages"][0]
                self._ensure_objective_guide_for_stage(env, world, agent, prog, next_stage, res, force_here=True, announce=True)

    def _bootstrap(self, env, world) -> None:
        from collections import deque

        rng = env.rng

        area_ids = list(world.area_instances.keys())
        spawn_area = env.world_definition["initializations"]["spawn"]["area"]
        non_spawn = [a for a in area_ids if a != spawn_area]

        # Exclude tutorial area from quest NPC placement
        _tut_room = world.auxiliary.get("tutorial_room") or {}
        _tut_area_id = _tut_room.get("area_id")
        if _tut_area_id:
            non_spawn = [a for a in non_spawn if a != _tut_area_id]

        # BFS from spawn to compute distances
        dist: Dict[str, int] = {spawn_area: 0}
        queue: deque = deque([spawn_area])
        while queue:
            cur = queue.popleft()
            cur_area = world.area_instances.get(cur)
            if cur_area is None:
                continue
            for nbr_id in cur_area.neighbors:
                if nbr_id not in dist and nbr_id in world.area_instances:
                    dist[nbr_id] = dist[cur] + 1
                    queue.append(nbr_id)

        # sort non-spawn areas by BFS distance (closest first),
        # with ~15% chance to bump an area one
        # layer further,
        NOISE_PROB = 0.25
        rng.shuffle(non_spawn)
        non_spawn.sort(key=lambda a: dist.get(a, len(area_ids))
                       + (1 if rng.random() < NOISE_PROB else 0))

        used: set = set()
        all_area_set = set(area_ids)

        def pick_progressive(n: int = 1, exclude: Optional[set] = None) -> List[str]:
            # pick *n* closest unused areas, respecting progressive expansion.
            exclude = exclude or set()
            picked: List[str] = []

            # pick from the distance-sorted non_spawn list
            for aid in non_spawn:
                if aid in used or aid in exclude:
                    continue
                picked.append(aid)
                used.add(aid)
                if len(picked) >= n:
                    return picked

            # all non-spawn areas exhausted — BFS expand from used areas
            frontier = deque()
            visited_expand: Set[str] = set(used) | exclude | {spawn_area}
            for u in used:
                u_area = world.area_instances.get(u)
                if u_area is None:
                    continue
                for nbr_id in u_area.neighbors:
                    if nbr_id in all_area_set and nbr_id not in visited_expand:
                        frontier.append(nbr_id)
                        visited_expand.add(nbr_id)

            while frontier and len(picked) < n:
                cand = frontier.popleft()
                if cand not in used and cand not in exclude:
                    picked.append(cand)
                    used.add(cand)
                    if len(picked) >= n:
                        return picked
                cand_area = world.area_instances.get(cand)
                if cand_area is None:
                    continue
                for nbr_id in cand_area.neighbors:
                    if nbr_id in all_area_set and nbr_id not in visited_expand:
                        frontier.append(nbr_id)
                        visited_expand.add(nbr_id)

            # still short — allow reusing already-used areas
            if len(picked) < n:
                reuse_src = picked[-1] if picked else (list(used)[0] if used else spawn_area)
                reuse_dist: Dict[str, int] = {reuse_src: 0}
                rq: deque = deque([reuse_src])
                while rq:
                    rc = rq.popleft()
                    rc_area = world.area_instances.get(rc)
                    if rc_area is None:
                        continue
                    for nbr_id in rc_area.neighbors:
                        if nbr_id in all_area_set and nbr_id not in reuse_dist:
                            reuse_dist[nbr_id] = reuse_dist[rc] + 1
                            rq.append(nbr_id)
                reuse_pool = sorted(
                    (a for a in all_area_set if a not in exclude and a not in set(picked)),
                    key=lambda a: reuse_dist.get(a, len(area_ids)),
                )
                for aid in reuse_pool:
                    picked.append(aid)
                    if len(picked) >= n:
                        break

            return picked

        self._generated['ch1_oath_text'] = 'THE SKY REMEMBERS'

        # chapter_1: Map of the Fallen Sky
        _picks_ch1_guide_area = pick_progressive(1)
        self._generated['ch1_guide_area'] = _picks_ch1_guide_area[0] if _picks_ch1_guide_area else spawn_area
        # ch1_guide_npc: NPC npc_quest_guide spawned in _ensure_static_quest_npcs
        self._generated['ch1_guide_name'] = 'sere'
        _picks_ch1_market_area = pick_progressive(1)
        self._generated['ch1_market_area'] = _picks_ch1_market_area[0] if _picks_ch1_market_area else spawn_area
        _picks_ch1_delivery_area = pick_progressive(1)
        self._generated['ch1_delivery_area'] = _picks_ch1_delivery_area[0] if _picks_ch1_delivery_area else spawn_area
        self._generated['ch1_survey_seal_obj'] = 'obj_survey_seal'

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
            # Convert 'objects' list to inventory dict
            inventory = dict(d.get("inventory", {}) or {})
            for obj_id in d.get("objects", []):
                if obj_id:
                    inventory[obj_id] = inventory.get(obj_id, 0) + 1
            world.npcs[nid] = NPC(
                type=d.get("type", "npc"),
                id=nid,
                name=d["name"],
                enemy=bool(d.get("enemy", False)),
                unique=bool(d.get("unique", True)),
                role=d.get("role", "quest"),
                level=1,
                description=d.get("description", ""),
                attack_power=int(d.get("base_attack_power", d.get("attack_power", 0))),
                hp=int(d.get("base_hp", d.get("hp", 100))),
                coins=int(d.get("coins", 0)),
                slope_hp=int(d.get("slope_hp", 0)),
                slope_attack_power=int(d.get("slope_attack_power", 0)),
                quest=bool(d.get("quest", True)),
                inventory=inventory,
                combat_pattern=list(d.get("combat_pattern", []) or []),
            )

    def _ensure_static_quest_npcs(self, env, world) -> None:
        rng = env.rng

        if 'ch1_guide_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_quest_guide', self._generated['ch1_guide_area'], rng)
            if inst:
                self._generated['ch1_guide_npc'] = inst
                self._generated['ch1_guide_name'] = world.npc_instances[inst].name

        if 'ch1_merchant_npc' not in self._generated:
            # Find the existing npc_vesin instance placed by world generation
            for npc_inst_id, npc_inst in world.npc_instances.items():
                if get_def_id(npc_inst_id) == 'npc_vesin':
                    self._generated['ch1_merchant_npc'] = npc_inst_id
                    self._generated['ch1_merchant_name'] = npc_inst.name
                    self._setup_ch1_merchant_stock(world, npc_inst_id)
                    break

        world.auxiliary["main_quest"]["generated"] = self._generated

    def _setup_ch1_merchant_stock(self, world, npc_inst_id: str) -> None:
        npc = world.npc_instances.get(npc_inst_id)
        if not npc:
            return
        npc.inventory["obj_survey_seal"] = max(int(npc.inventory.get("obj_survey_seal", 0)), 1)

    def _create_npc_instance(self, world, base_npc_id: str, area_id: str, rng) -> Optional[str]:
        if base_npc_id not in world.npcs:
            return None
        area = world.area_instances.get(area_id)
        if not area:
            return None

        for npc_id in area.npcs:
            if get_def_id(npc_id) == base_npc_id:
                return npc_id

        aux = world.auxiliary
        proto = world.npcs[base_npc_id]
        level = int(getattr(area, "level", 1) or 1)
        if proto.unique:
            inst_obj = proto.create_instance(None, level=level, objects=world.objects, rng=rng)
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
        area.npcs.append(inst.id)
        world.auxiliary["npc_name_to_id"][inst.name] = inst.id
        return inst.id

    def _ensure_bosses_for_progress(self, env, world, prog: Dict[str, Any]) -> None:
        ch = int(prog.get('chapter', 0))
        st = int(prog.get('stage', 0))

        pass  # no bosses to spawn

    def _ensure_named_boss_spawned(self, env, world, boss_key: str) -> None:
        mq = world.auxiliary["main_quest"]
        if mq["boss_defeated"].get(boss_key, False):
            return

        inst_id = self._generated.get(boss_key)
        if inst_id and inst_id in world.npc_instances:
            return

        rng = env.rng
        if boss_key == "ch1_boss_npc":
            base_id = "npc_boss_cinder_reaver"
            area_id = self._generated["ch1_shrine_area"]
            base_hp, base_atk = 115, 20
        elif boss_key == "ch2_boss_npc":
            base_id = "npc_boss_ember_warden"
            area_id = self._generated["ch2_boss_area"]
            base_hp, base_atk = 225, 20
        else:
            return

        inst = self._create_npc_instance(world, base_id, area_id, rng)
        if not inst:
            return

        boss = world.npc_instances[inst]
        boss.hp = base_hp
        boss.attack_power = base_atk

        self._generated[boss_key] = inst
        mq["boss_max_hp"][inst] = base_hp
        mq["boss_mods"].setdefault(inst, {"base_max_hp": base_hp, "curr_max_hp": base_hp, "drenched": False})
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

    def _apply_fire_boss_modifier(self, env, world, agent: Agent) -> None:
        boss_id = self._generated.get("ch2_boss_npc")
        if not boss_id or boss_id not in world.npc_instances:
            return
        boss_area = self._generated.get("ch2_boss_area")
        if not boss_area:
            return
        if env.curr_agents_state["area"][agent.id] != boss_area:
            return
        area = world.area_instances.get(boss_area)
        if not area or boss_id not in area.npcs:
            return

        mq = world.auxiliary["main_quest"]
        mods = mq["boss_mods"].setdefault(boss_id, {"base_max_hp": 200, "curr_max_hp": 200, "drenched": False})

        has_water_sword = "obj_water_sword" in agent.items_in_hands
        boss = world.npc_instances[boss_id]

        has_wooden_sword = "obj_wooden_sword" in agent.items_in_hands

        if has_water_sword and not has_wooden_sword:
            # huge advantage
            mods["drenched"] = True
            mods["curr_max_hp"] = 150
            boss.attack_power = 20
            boss.hp = min(int(boss.hp), int(mods["curr_max_hp"]))
        if not has_water_sword and has_wooden_sword:
            # huge disadvantage
            boss.attack_power = 45
        else:
            if mods.get("drenched", False):
                boss.attack_power = 20
                boss.hp = min(int(boss.hp), int(mods.get("curr_max_hp", 150)))

    def _ensure_heart_has_value(self, world) -> None:
        if "obj_heart_of_the_ocean" in world.objects:
            obj = world.objects["obj_heart_of_the_ocean"]
            if getattr(obj, "value", None) is None:
                obj.value = 25

    def _init_agent_progress(self, agent_id: str) -> None:
        self._progress[agent_id] = {
            "chapter": 0,
            "stage": 0,
            "visited_areas": [],
            "complete": False,
            "mq_guide_name": None,
            "mq_guide_inst": None,
            "mq_guide_area": None,
            "mq_stage_key": None,
            # DAG progression (used when chapter has stages with "requires")
            "completed_stages": {},   # chapter_id -> list of completed stage IDs
            "active_stages": {},      # chapter_id -> list of currently active stage IDs
            "dag_announced": {},      # chapter_id -> set of stage IDs already announced
        }

    @staticmethod
    def _is_chapter_dag(chapter: dict) -> bool:
        """Return True if any stage in the chapter has a ``requires`` field."""
        return any("requires" in s for s in chapter.get("stages", []))

    @staticmethod
    def _compute_active_stages(chapter: dict, completed: set) -> List[str]:
        """Return IDs of stages whose prerequisites are met and not yet done.

        Stages without a ``requires`` field are treated as having no
        prerequisites (roots).  A stage is activatable when every ID in
        its ``requires`` list is in *completed*.
        """
        active: List[str] = []
        for stage in chapter.get("stages", []):
            sid = stage["id"]
            if sid in completed:
                continue
            reqs = stage.get("requires", [])
            if all(r in completed for r in reqs):
                active.append(sid)
        return active

    def _get_stage_by_id(self, chapter: dict, stage_id: str) -> Optional[dict]:
        for s in chapter.get("stages", []):
            if s["id"] == stage_id:
                return s
        return None

    def _save_goal_tree_png(self, env, chapter: dict, ch_idx: int,
                            completed_set: set, active_ids: list,
                            agent_id: str, step: int) -> None:
        """Render the goal tree for a DAG chapter and save as PNG."""
        try:
            import networkx as nx
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import textwrap
        except ImportError:
            return

        stages = chapter.get("stages", [])
        if not stages:
            return

        G = nx.DiGraph()
        labels = {}
        for s in stages:
            sid = s["id"]
            obj = self._render_text(s.get("objective", sid))
            # Wrap long labels with newlines
            label = "\n".join(textwrap.wrap(obj, width=20))
            G.add_node(sid)
            labels[sid] = label
            for req in s.get("requires", []):
                G.add_edge(req, sid)

        # Assign colors: green=completed, gold=active, lightgray=locked
        colors = []
        for sid in G.nodes():
            if sid in completed_set:
                colors.append("#66bb6a")   # green
            elif sid in active_ids:
                colors.append("#ffca28")   # gold
            else:
                colors.append("#bdbdbd")   # gray

        # Use graphviz layout if available, else fall back to spring
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            try:
                pos = nx.planar_layout(G)
            except Exception:
                pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.2), max(6, len(stages) * 0.8)))
        nx.draw(
            G, pos, ax=ax,
            labels=labels,
            with_labels=True,
            node_color=colors,
            node_size=3600,
            font_size=7,
            font_weight="bold",
            arrows=True,
            arrowsize=15,
            edge_color="#888888",
            node_shape="s",
        )

        # Legend
        import matplotlib.patches as mpatches
        legend_items = [
            mpatches.Patch(color="#66bb6a", label="Completed"),
            mpatches.Patch(color="#ffca28", label="Active"),
            mpatches.Patch(color="#bdbdbd", label="Locked"),
        ]
        ax.legend(handles=legend_items, loc="upper left", fontsize=8)

        ch_title = self._render_text(chapter.get("title", f"chapter_{ch_idx}"))
        ax.set_title(f"Goal Tree — {ch_title}  (step {step})", fontsize=10)

        import os
        out_dir = os.path.join(env.run_dir, "goal_tree")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"goal_tree_ch{ch_idx}_{agent_id}_step{step:04d}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _update_visited(self, prog: Dict[str, Any], area_id: str) -> None:
        if not area_id:
            return
        visited = prog.setdefault("visited_areas", [])
        if area_id not in visited:
            visited.append(area_id)

    def _render_text(self, text: str, extra: Optional[Dict[str, str]] = None) -> str:
        def area_disp(key: str) -> str:
            return self._get_area_display_name(key)

        def area_list_disp(list_key: str) -> str:
            ids = self._generated.get(list_key, [])
            if not isinstance(ids, list):
                return "unknown"
            names = [self._get_area_display_name_from_id(aid) for aid in ids]
            return ", ".join(names) if names else "unknown"

        fmt = {
            'ch1_guide_name': self._generated.get('ch1_guide_name', 'ch1_guide_name'),

            "mq_guide_name": (extra or {}).get("mq_guide_name", "your guide"),
        }
        if extra:
            fmt.update(extra)
        try:
            return text.format(**fmt)
        except Exception:
            return text

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

    def _get_area_display_name(self, key: str) -> str:
        return self._get_area_display_name_from_id(self._generated.get(key))

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
                    # BuyRule: data={"npc_id": npc_id, "obj_id": base_id, ...}
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
                return bool(area and target_npc in area.npcs)

            if stype == "visited_k_of_area_set":
                ids = self._generated.get(cond.get("area_set_key"), [])
                if not isinstance(ids, list):
                    return False
                k = self._generated.get(cond.get("k_key"), cond.get("k", 1))
                try:
                    k = int(k)
                except Exception:
                    k = 1
                visited = set(prog.get("visited_areas", []))
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
                # expects writable instances to be in area.objects (same as your current MainQuest code)
                for oid in area.objects:
                    if oid in world.writable_instances:
                        writable = world.writable_instances[oid]
                        if str(getattr(writable, "text", "")).strip().lower() == target:
                            return True
                return False

            if stype == "paper_text_equals_in_hands":
                text_key = cond.get("text_key")
                target = str(self._generated.get(text_key, "")).strip().lower()
                if not target:
                    return False
                held_ids = list(getattr(agent, "items_in_hands", {}) or {})
                held_ids += list(getattr(agent, "equipped_items_in_limb", {}) or {})
                for oid in held_ids:
                    if oid in world.writable_instances:
                        writable = world.writable_instances[oid]
                        if str(getattr(writable, "text", "")).strip().lower() == target:
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

        if kind == "custom":
            ctype = cond.get("type")

            if ctype == "paper_with_text_in_area":
                text_key = cond.get("text_key")
                area_key = cond.get("area_key")
                if not text_key or not area_key:
                    return False
                target_text = str(self._generated.get(text_key, "")).strip().lower()
                area_id = self._generated.get(area_key)
                if not target_text or not area_id:
                    return False
                area = world.area_instances.get(area_id)
                if not area:
                    return False
                for oid in area.objects.keys():
                    if oid in world.writable_instances and get_def_id(oid) == "obj_paper":
                        writable = world.writable_instances[oid]
                        wtxt = str(getattr(writable, "text", "")).strip().lower()
                        if wtxt == target_text:
                            return True
                return False

            if ctype == "has_quest_item_by_base_or_instance":
                target_base_id = cond.get("obj_id") or "obj_survey_seal"
                return self._count_base_item(world, agent, target_base_id) > 0

            if ctype == "paper_with_text_dropped_in_area":
                text_key = cond.get("text_key")
                area_key = cond.get("area_key")
                if not text_key or not area_key:
                    return False
                target_text = str(self._generated.get(text_key, "")).strip().lower()
                area_id = self._generated.get(area_key)
                if not target_text or not area_id:
                    return False
                area = world.area_instances.get(area_id)
                if not area:
                    return False
                for oid in area.objects.keys():
                    if oid in world.writable_instances and get_def_id(oid) == "obj_paper":
                        writable = world.writable_instances[oid]
                        wtxt = str(getattr(writable, "text", "")).strip().lower()
                        if wtxt == target_text:
                            return True
                return False

            if ctype == "object_in_area":
                area_key = cond.get("area_key")
                obj_key = cond.get("obj_key")
                if not area_key or not obj_key:
                    return False
                area_id = self._generated.get(area_key)
                target_obj = self._generated.get(obj_key)
                if not area_id or not target_obj:
                    return False
                area = world.area_instances.get(area_id)
                if not area:
                    return False
                target_base_id = get_def_id(target_obj)
                for oid in area.objects.keys():
                    if get_def_id(oid) == target_base_id:
                        return True
                return False

            if ctype == "stamped_areas_in_order":
                area_keys = cond.get("area_keys", [])
                if not isinstance(area_keys, list) or not area_keys:
                    return False
                target_areas: list[str] = []
                for k in area_keys:
                    aid = self._generated.get(k)
                    if not aid:
                        return False
                    target_areas.append(aid)

                obj_key = cond.get("obj_key")
                stamper_base = None
                if obj_key:
                    stamper_base = self._generated.get(obj_key)
                if not stamper_base:
                    stamper_base = "obj_rule_stamper"

                history = []
                qhist_all = env.curr_agents_state.get("quest_event_history", {})
                qhist = qhist_all.get(agent.id) if isinstance(qhist_all, dict) else None
                if isinstance(qhist, list):
                    history = qhist
                else:
                    history = [
                        {"type": getattr(e, "type", None), **(getattr(e, "data", {}) or {})}
                        for e in (events or [])
                    ]

                drops: list[str] = []
                for ev in history:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("type") == "object_dropped":
                        obj_id = ev.get("obj_id")
                        area_id = ev.get("area_id")
                        if obj_id and area_id and get_def_id(str(obj_id)) == stamper_base:
                            drops.append(area_id)

                if len(drops) < len(target_areas):
                    return False
                i = 0
                for area_id in drops:
                    if area_id == target_areas[i]:
                        i += 1
                        if i == len(target_areas):
                            return True
                return False

            if ctype == "stage_completed":
                stage_id = cond.get("stage_id") or ""
                if not stage_id:
                    return False

                completed: set[str] = set()
                prog_all = env.curr_agents_state.get("main_quest_progress", {})
                aprog = prog_all.get(agent.id) if isinstance(prog_all, dict) else None
                if isinstance(aprog, dict):
                    cs = aprog.get("completed_stages")
                    if isinstance(cs, dict):
                        for v in cs.values():
                            if isinstance(v, (list, set, tuple)):
                                completed.update(v)
                    elif isinstance(cs, (list, set, tuple)):
                        completed.update(cs)

                return stage_id in completed

            print(f"WARNING: Unknown custom condition type={ctype}")
            return False

        print(f"WARNING: Unknown condition kind={kind} type={cond.get('type')} — needs custom implementation")
        return False

    def _spawn_object_in_area(self, agent_id: str, world, area_id: str, obj_id: str, count: int, res) -> None:
        area = world.area_instances.get(area_id)
        if not area or count <= 0:
            return
        if obj_id not in world.objects:
            return
        area.objects[obj_id] = area.objects.get(obj_id, 0) + count
        res.track_spawn(agent_id, obj_id, count, dst=res.tloc("area", area_id))
        res.events.append(Event(type="quest_spawn", agent_id=agent_id, data={"obj_id": obj_id, "count": count, "area_id": area_id}))

    def _count_base_item(self, world, agent, base_obj_id: str) -> int:
        total = 0
        if not base_obj_id:
            return 0

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

        # containers held in hands
        hand_keys = items_in_hands.keys() if isinstance(items_in_hands, dict) else items_in_hands
        for oid in hand_keys:
            if oid in getattr(world, "container_instances", {}):
                cinv = world.container_instances[oid].inventory
                for coid, cnt in (cinv or {}).items():
                    if get_def_id(coid) == base_obj_id:
                        total += int(cnt)

        return total

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

    def _set_stage_key(self, prog: Dict[str, Any], ch: int, st: int) -> None:
        prog["mq_stage_key"] = f"{int(ch)}:{int(st)}"

    def _mark_stage_started(self, prog: Dict[str, Any], ch: int, st: int) -> bool:
        key = f"{int(ch)}:{int(st)}"
        prev = str(prog.get("mq_stage_key", "") or "")
        if prev != key:
            prog["mq_stage_key"] = key
            return True
        return False

    def _ensure_objective_guide_for_stage(
        self,
        env,
        world,
        agent,
        prog: Dict[str, Any],
        stage: Dict[str, Any],
        res,
        force_here: bool,
        announce: bool,
    ) -> None:
        # ensure guide prototype exists
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
                already_here = self._area_contains_npc(world, current_area_id, inst_id)
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
                    entered = not already_here

        self._set_npc_name(world, inst_id, guide_name)
        self._force_npc_invincible(world, inst_id)

        obj_text = self._render_text(stage.get("objective", ""), extra={"mq_guide_name": guide_name})
        self._set_npc_dialogue(world, inst_id, obj_text)

        if entered and announce:
            res.add_feedback(agent.id, f"{guide_name} enters the area.\n")

    def _ensure_objective_guide_for_stages(
        self,
        env,
        world,
        agent,
        prog: Dict[str, Any],
        stages: List[Dict[str, Any]],
        res,
        force_here: bool,
        announce: bool,
    ) -> None:
        """DAG mode: set the guide's dialogue to a numbered list of all
        active objectives."""
        if not stages:
            return
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
                already_here = self._area_contains_npc(world, current_area_id, inst_id)
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
                    entered = not already_here

        self._set_npc_name(world, inst_id, guide_name)
        self._force_npc_invincible(world, inst_id)

        # build multi-objective dialogue
        # filter to actionable stages only (those with conditions the player can work toward)
        actionable = [s for s in stages if s.get("done_any") or s.get("done_all")]
        if not actionable:
            actionable = stages  # fallback: show all if none have conditions

        # Root goal header (tree mode)
        header = ""
        ch_idx = int(prog.get("chapter", 0))
        chapters = self.QUEST_CONFIG.get("chapters", [])
        if ch_idx < len(chapters):
            ch = chapters[ch_idx]
            root_id = ch.get("root_goal")
            if root_id:
                root_stage = self._get_stage_by_id(ch, root_id)
                if root_stage:
                    root_text = self._render_text(root_stage.get("objective", ""), extra={"mq_guide_name": guide_name})
                    header = f"Main goal: {root_text}\n\n"

        lines: List[str] = []
        for i, stage in enumerate(actionable, 1):
            obj_text = self._render_text(stage.get("objective", ""), extra={"mq_guide_name": guide_name})
            lines.append(f"[{i}] {obj_text}")
        dialogue = header + "Current tasks:\n" + "\n".join(lines)
        self._set_npc_dialogue(world, inst_id, dialogue)

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

    def _ensure_objective_name_map_for_agents(self, env, world) -> None:
        mq = world.auxiliary.setdefault("main_quest", {})
        name_map = mq.setdefault("objective_npc_names", {})

        run_seed = int(getattr(env, "seed", 0))
        mq["run_seed"] = run_seed

        agent_ids = [a.id for a in getattr(env, "agents", [])]
        agent_ids_sorted = sorted([str(aid) for aid in agent_ids])
        if agent_ids_sorted and all(aid in name_map for aid in agent_ids_sorted):
            return

        # name pool to reduce collision risk for multi-agent settings
        base_pool = list(self.OBJECTIVE_NPC_NAME_POOL)

        import random
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

    def _append_talk_hint(self, text: str, guide_name: str) -> str:
        if not text.strip():
            return text
        hint = f"\nTalk to {guide_name} about what to do next.\n"
        return text.rstrip() + hint
