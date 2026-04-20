import copy
import math
import random
from utils import *
from games.generated.remnant.rule import BaseStepRule, RuleContext, RuleResult, Event
from typing import Dict, Optional, Set, Tuple, List, Any
from games.generated.remnant.world import NPC, Object, Area, Place, Path, Container, Writable
from games.generated.remnant.agent import Agent
from tools.logger import get_logger
import inspect


class MainQuestStepRule(BaseStepRule):
    name = "main_quest_step"
    description = "Five-chapter main quest (vague, long-horizon): memory + travel + trade + forging + trials + bosses."
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
        {"type": "object", "id": "obj_water_sword", "name": "water_sword",
         "category": "weapon", "usage": "attack", "value": None, "size": 5,
         "attack": 16,
         "description": "A tide-singing blade that bites hardest into ocean-borne bosses; against common foes it feels merely sharp.",
         "craft": {"ingredients": {"obj_tide_alloy_ingot": 1, "obj_wooden_sword": 1, "obj_heart_of_the_ocean": 1},
                   "dependencies": ["obj_workbench"]},
         "level": 5},

        {"type": "object", "id": "obj_prism_lens", "name": "prism_lens",
         "category": "tool", "usage": "sight", "value": None, "size": 1,
         "description": "A clear shard-lens that makes hidden paths feel loud.",
         "craft": {"ingredients": {}, "dependencies": []},
         "level": 4, "quest": True},

        {"type": "object", "id": "obj_storm_crest", "name": "storm_crest",
         "category": "material", "usage": "craft", "value": None, "size": 1,
         "description": "A heavy crest that hums with pressure; it refuses to slip once set.",
         "craft": {"ingredients": {}, "dependencies": []},
         "level": 5, "quest": True},

        {"type": "object", "id": "obj_confluence_key", "name": "confluence_key",
         "category": "tool", "usage": "unlock", "value": None, "size": 1,
         "description": "A key that can only exist by consuming a water-forged blade and binding its tide-power into one lock.",
         "craft": {"ingredients": {"obj_water_sword": 1, "obj_prism_lens": 1, "obj_storm_crest": 1},
                   "dependencies": ["obj_workbench"]},
         "level": 5, "quest": True},
    ]

    required_npcs: List[Dict[str, Any]] = [
        {"type": "npc", "id": "npc_quest_chronicler", "name": "chronicler_lyra",
         "enemy": False, "unique": True, "role": "chronicler", "quest": True,
         "description": "a soot-stained chronicler who speaks in rites and half-memories.",
         "base_attack_power": 200, "slope_attack_power": 0, "base_hp": 200, "slope_hp": 0,
         "objects": []},

        # with stamina system: L3+wd attack/wait LOSES, L4+wd attack/wait WINS
        # L2+ws attack/wait LOSES, L3+ws attack/wait WINS, L2+ws optimal WINS
        {"type": "npc", "id": "npc_boss_cinder_reaver", "name": "cinder_reaver",
         "enemy": True, "unique": True, "role": "boss_minor", "quest": True,
         "description": "a shrine-guardian made of ember and stubborn habit.",
         "base_attack_power": 20, "slope_attack_power": 0, "base_hp": 115, "slope_hp": 0,
         "combat_pattern": ["attack", "attack", "attack", "wait", "attack", "attack", "defend"],
         "objects": []},

        {"type": "npc", "id": "npc_quest_tide_merchant", "name": "tide_merchant_sable",
         "enemy": False, "unique": True, "role": "merchant", "quest": True,
         "description": "a tide-merchant who sells one ocean-luxury no ordinary shop carries.",
         "base_attack_power": 200, "slope_attack_power": 0, "base_hp": 200, "slope_hp": 0,
         "objects": ["obj_heart_of_the_ocean", "obj_sea_salt_crystal"]},

        # with stamina system: L4+ss AA LOSES, L5+ss AA WINS, L4+ss+sa AA WINS, L4+ss OPT WINS
        {"type": "npc", "id": "npc_boss_ember_warden", "name": "ember_warden",
         "enemy": True, "unique": True, "role": "boss_fire", "quest": True,
         "description": "a furnace-walker that scorches wood and laughs at weak steel. It hesitates at the sea’s answer.",
         "base_attack_power": 20, "slope_attack_power": 0, "base_hp": 225, "slope_hp": 0,
         "combat_pattern": ["attack", "defend", "attack", "attack", "attack", "attack", "wait"],
         "objects": []},

        {"type": "npc", "id": "npc_quest_anvil_judge", "name": "anvil_judge",
         "enemy": False, "unique": True, "role": "craftsman", "quest": True,
         "description": "a quiet judge of rivets and weight; they trust scratches more than stories.",
         "base_attack_power": 200, "slope_attack_power": 0, "base_hp": 200, "slope_hp": 0,
         "objects": []},

        # with stamina system: L5+ss+ia AA LOSES, L6+ss+ia AA WINS, L5+ss+ia OPT WINS
        {"type": "npc", "id": "npc_boss_iron_answer", "name": "iron_answer",
         "enemy": True, "unique": True, "role": "boss_iron", "quest": True,
         "description": "a trial made of pressure and metal-echo; it wakes when iron is spoken in the right place.",
         "base_attack_power": 12, "slope_attack_power": 0, "base_hp": 295, "slope_hp": 0,
         "combat_pattern": ["defend", "attack", "attack", "attack"],
         "objects": []},
        
        # with stamina system: L7+is attack/wait LOSES, L8+is attack/wait WINS, L7+is OPT WINS
        {"type": "npc", "id": "npc_boss_wind_answer", "name": "wind_answer",
         "enemy": True, "unique": True, "role": "boss_wind", "quest": True,
         "description": "a shape of thin air and sharp timing; it steps out when the sky decides to listen.",
         "base_attack_power": 20, "slope_attack_power": 0, "base_hp": 395, "slope_hp": 0,
         "combat_pattern": ["attack", "attack", "attack", "attack", "attack", "attack", "wait"],
         "objects": []},
        
        # with stamina system: L9+sm attack/wait LOSES, L10+sm attack/wait WINS
        {"type": "npc", "id": "npc_boss_confluence_warden", "name": "confluence_warden",
         "enemy": True, "unique": True, "role": "boss_final", "quest": True,
         "description": "a guardian that wakes only for a key that has eaten the sea and learned to lock it.",
         "base_attack_power": 30, "slope_attack_power": 0, "base_hp": 400, "slope_hp": 0,
         "combat_pattern": ["attack"],
         "objects": []},

        # objective guide (invincible)
        {"type": "npc", "id": OBJECTIVE_NPC_BASE_ID, "name": "mira",
         "enemy": False, "unique": True, "role": "guide", "quest": True,
         "description": "",
         "base_attack_power": OBJECTIVE_NPC_ATK, "slope_attack_power": 0,
         "base_hp": OBJECTIVE_NPC_MAX_HP, "slope_hp": 0,
         "objects": []},
    ]

    undistributable_objects: List[str] = [
        "obj_heart_of_the_ocean",
        "obj_prism_lens",
        "obj_storm_crest",
        "obj_confluence_key",
    ]

    QUEST_CONFIG: Dict[str, Any] = {
        "title": "The Tide and the Ember",
        "intro": (
            "=== MAIN QUEST: The Tide and the Ember ===\n"
            "Two old powers argue beneath this world: the Furnace and the Sea.\n"
            "One reduces all it touches to ash.\n"
            "One keeps its truths in cold, unbroken depths.\n"
            "They do not teach.\n"
            "They do not remind.\n"
            "What endures is what you carry.\n"
            "==========================================\n"
            "\n"
        ),
        "chapters": [
            {
                "id": "chapter_1",
                "title": "The Cinder Trail",
                "intro": (
                    "Chapter 1: The Cinder Trail\n"
                    "Somewhere ahead, an old shrine still breathes beneath a coat of ash.\n"
                    "A mysterious guide {mq_guide_name} knows how to lead you there.\n"
                    "Talk to {mq_guide_name} about what to do next.\n"
                ),
                "stages": [
                    {
                        "id": "ch1_meet",
                        "objective": "Find the one who knows where cinder-smoke gathers and why it should be approached carefully.",
                        "done_any": [{"kind": "state", "type": "in_area_with_npc", "npc_key": "ch1_guide_npc"}],
                        "on_complete_feedback": (
                            "{ch1_guide_name} looks past you, as if listening to something far away.\n"
                            "'You are not the first to go looking for that place.'\n"
                            "\n"
                            "'The Cinder Shrine sits in one of these regions:\n"
                            " {ch1_shrine_candidates}\n"
                            "Walk them until the air tastes like charcoal.'\n"
                            "\n"
                            "Their voice drops.\n"
                            "'Do not go with empty hands.\n"
                            "Ash keeps watch, and it wakes what was left behind.'\n"
                        ),
                    },
                    {
                        "id": "ch1_prepare",
                        "objective": "Arm yourself with a weapon before you follow the ash, because what waits at the shrine will not be polite.",
                        "done_any": [
                            {"kind": "state", "type": "has_any_of", "base_objs": [
                                "obj_wooden_sword", "obj_stone_sword", "obj_iron_sword",
                                "obj_star_metal_sword", "obj_water_sword", "obj_wooden_dagger",
                                "obj_glass_spear", "obj_aether_rapier"
                            ]},
                        ],
                        "on_complete_feedback": (
                            "The weight in your hand is steady.\n"
                            "Now follow the cinder-trail into heat and dust.\n"
                        ),
                    },
                    {
                        "id": "ch1_discover_shrine",
                        "objective": (
                            "The shrine is somewhere within:\n"
                            "  {ch1_shrine_candidates}\n"
                            "Search the named regions until the Cinder Shrine shows itself, then step close enough to draw its answer."
                        ),
                        "done_all": [
                            {"kind": "state", "type": "visited_k_of_area_set", "area_set_key": "ch1_shrine_candidate_areas", "k_key": "ch1_explore_min"},
                            {"kind": "state", "type": "in_area", "area_key": "ch1_shrine_area"},
                        ],
                        "on_complete_feedback": (
                            "The air tastes like old charcoal.\n"
                            "Stone is blackened, and the shrine is quiet in a way that feels practiced.\n"
                            "\n"
                            "A cold brazier sits at its center.\n"
                            "When you step into the ring of ash, it stirs as if it recognizes you.\n"
                            "Something behind the shrine inhales, awake.\n"
                        ),
                        "spawns_boss_key": "ch1_boss_npc",
                    },
                    {
                        "id": "ch1_boss",
                        "objective": "Defeat {ch1_boss_name}, the guardian that rises from the shrine's ash.",
                        "done_any": [
                            {"kind": "event", "type": "npc_killed", "npc_key": "ch1_boss_npc"},
                            {"kind": "state", "type": "boss_already_defeated", "boss_key": "ch1_boss_npc"},
                        ],
                        "on_complete_feedback": (
                            "The guardian collapses into embers.\n"
                            "Among the ash you find coin-glints.\n"
                            "\n"
                            "=== Chapter 1 Complete ===\n"
                            "As you leave, a rumor keeps pace with you:\n"
                            "a tide-merchant sells an ocean-luxury no ordinary shop carries.\n"
                        ),
                        "gives_coins": 12,
                        "chapter_complete": True,
                    },
                ],
            },
            {
                "id": "chapter_2",
                "title": "The Ocean Mystery",
                "intro": (
                    "Chapter 2: The Ocean Mystery\n"
                    "Fire and water don’t argue. They erase.\n"
                    "To face what the tide conceals, you’ll need more than coin.\n"
                    "Talk to {mq_guide_name} again about what to do next.\n"
                ),
                "stages": [
                    {
                        "id": "ch2_trade",
                        "objective": "Find the tide-merchant and purchase the Heart of the Ocean from them.",
                        "done_all": [
                            {"kind": "state", "type": "in_area_with_npc", "npc_key": "ch2_merchant_npc"},
                            {"kind": "event", "type": "object_bought", "npc_key": "ch2_merchant_npc", "obj_id": "obj_heart_of_the_ocean"},
                        ],
                        "on_complete_feedback": (
                            "{ch2_merchant_name} takes your coin and lowers their voice:\n"
                            "'Keep it close.\n"
                            "And keep this word dry on your tongue: {ch2_salt_word}.\n"
                            "You won’t need it here.\n"
                            "You’ll need it where heat starts asking questions.'\n"
                            "\n"
                            "They glance at your gear:\n"
                            "'If you go where the air burns, wood will betray you.\n"
                            "Some things hesitate only at the sea.'\n"
                            "\n"
                            "Then, like an afterthought:\n"
                            "'The furnace-walker nests in one of these regions:\n"
                            "  {ch2_lair_candidates}\n"
                            "Go find the nest before it finds you.'\n"
                        ),
                    },
                    {
                        "id": "ch2_discover_lair",
                        "objective": "Follow the merchant’s directions to explore the areas, until you've felt the immense heat.",
                        "done_any": [
                            {"kind": "state", "type": "in_area", "area_key": "ch2_boss_area"},
                        ],
                        "on_complete_feedback": (
                            "Heat presses against your lungs.\n"
                            "A mark is carved into stone:\n"
                            "'SAY THE SEA-WORD OR BE ASH.'\n"
                            "\n"
                            "Even the scorch marks feel like they’re listening.\n"
                        ),
                    },
                    {
                        "id": "ch2_memory",
                        "objective": "Drop down a paper with the sea-word written on it.",
                        "done_all": [
                            {"kind": "state", "type": "in_area", "area_key": "ch2_boss_area"},
                            {"kind": "state", "type": "paper_text_equals_in_area", "text_key": "ch2_salt_word"},
                        ],
                        "on_complete_feedback": (
                            "The paper hits stone.\n"
                            "The air hisses.\n"
                            "Flame recoils, just a little.\n"
                            "Something heavy turns in the dark.\n"
                        ),
                        "spawns_boss_key": "ch2_boss_npc",
                    },
                    {
                        "id": "ch2_boss",
                        "objective": "Defeat {ch2_boss_name}, the furnace-walker made with fire.",
                        "done_any": [
                            {"kind": "event", "type": "npc_killed", "npc_key": "ch2_boss_npc"},
                            {"kind": "state", "type": "boss_already_defeated", "boss_key": "ch2_boss_npc"},
                        ],
                        "on_complete_feedback": (
                            "The furnace-walker breaks apart into dull slag.\n"
                            "\n"
                            "=== Chapter 2 Complete ===\n"
                            "Fire fades. Work remains.\n"
                        ),
                        "chapter_complete": True,
                    },
                ],
            },
            {
                "id": "chapter_3",
                "title": "The Iron Skin",
                "intro": (
                    "Chapter 3: The Iron Skin\n"
                    "Some trials do not test courage.\n"
                    "They test what you built before you arrived.\n"
                    "Talk to {mq_guide_name} again about what to do next.\n"
                ),
                "stages": [
                    {
                        "id": "ch3_meet",
                        "objective": "Find the anvil-minded judge and let your footsteps be weighed in silence.",
                        "done_any": [
                            {"kind": "state", "type": "in_area_with_npc", "npc_key": "ch3_smith_npc"},
                        ],
                        "on_complete_feedback": (
                            "{ch3_smith_name} looks you over without hurry.\n"
                            "'If you want to walk where steel screams, forge your skin first.'\n"
                            "They nod toward the forge.\n"
                            "'Then take it somewhere comfort won’t follow, and let it earn its weight.'\n"
                        ),
                    },
                    {
                        "id": "ch3_armor_gate",
                        "objective": "Forge iron scales for your chest, armor honest enough to meet a real blow.",
                        "done_any": [
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_iron_scale_armor"]},
                        ],
                        "on_complete_feedback": (
                            "The weight sits like a promise.\n"
                            "Now let darkness be the witness.\n"
                        ),
                    },
                    {
                        "id": "ch3_temper_in_dark",
                        "objective": "Wear the scales into dark places and win enough fights.",
                        "done_all": [
                            {"kind": "state", "type": "equipped_any_of", "base_objs": ["obj_iron_scale_armor"]},
                            {"kind": "state", "type": "killed_n_enemies",
                             "n": 3, "enemy_only": True, "in_dark_only": True, "min_area_level": 1},
                        ],
                        "on_complete_feedback": (
                            "Scratches become scripture.\n"
                            "Return to the judge.\n"
                        ),
                    },
                    {
                        "id": "ch3_return",
                        "objective": "Return to the judge with the armor marked by consequences.",
                        "done_any": [
                            {"kind": "state", "type": "in_area_with_npc", "npc_key": "ch3_smith_npc"},
                        ],
                        "on_complete_feedback": (
                            "{ch3_smith_name} runs a thumb along the dents and nods once.\n"
                            "'Good.'\n"
                            "'Keep this word like a rivet: {ch3_iron_word}.'\n"
                            "'Put it into ink where you can carry it.\n"
                            "Then walk, until the trial decides to look at you.'\n"
                        ),
                    },
                    {
                        "id": "ch3_write_and_arrive",
                        "objective": "Write the iron-word on something you carry, then step into the trial area where the air listens for iron.",
                        "done_all": [
                            {"kind": "state", "type": "writable_text_equals_in_inventory", "text_key": "ch3_iron_word"},
                            {"kind": "state", "type": "in_area", "area_key": "ch3_trial_area"},
                        ],
                        "on_complete_feedback": (
                            "The word grows heavy.\n"
                            "The air agrees to listen.\n"
                        ),
                        "spawns_boss_key": "ch3_boss_npc",
                    },
                    {
                        "id": "ch3_boss",
                        "objective": "Defeat {ch3_boss_name}, the trial that rises to meet your iron-word.",
                        "done_any": [
                            {"kind": "event", "type": "npc_killed", "npc_key": "ch3_boss_npc"},
                            {"kind": "state", "type": "boss_already_defeated", "boss_key": "ch3_boss_npc"},
                        ],
                        "on_complete_feedback": (
                            "The pressure breaks.\n"
                            "Light returns, but it returns differently.\n"
                            "\n"
                            "=== Chapter 3 Complete ===\n"
                        ),
                        "chapter_complete": True,
                    },
                ],
            },
            {
                "id": "chapter_4",
                "title": "The Wind Bargain",
                "intro": (
                    "Chapter 4: The Wind Bargain\n"
                    "Thin air is a stern teacher.\n"
                    "It forgives nothing you forgot to forge.\n"
                    "Talk to {mq_guide_name} again about what to do next.\n"
                ),
                "stages": [
                    {
                        "id": "ch4_ready",
                        "objective": "Step into the bargain prepared: iron on your body and the prism-lens riding in your pack.",
                        "done_all": [
                            {"kind": "state", "type": "equipped_any_of", "base_objs": ["obj_iron_scale_armor"]},
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_prism_lens"]},
                        ],
                        "on_complete_feedback": (
                            "Good.\n"
                            "Now make certainty in places that refuse it.\n"
                        ),
                    },
                    {
                        "id": "ch4_leave_light_behind",
                        "objective": "Light two different dark places and leave proof behind.",
                        "done_any": [
                            {"kind": "state", "type": "lit_k_dark_areas", "k": 2},
                        ],
                        "on_complete_feedback": (
                            "Two shadows learned your fire.\n"
                            "Now show you can survive more than one kind of hunger.\n"
                        ),
                    },
                    {
                        "id": "ch4_variety_in_blood",
                        "objective": "Face three shapes of threat and leave each quieter than you found it: goblin, bandit, and beast.",
                        "done_any": [
                            {"kind": "state", "type": "killed_one_of_each",
                             "roles": ["goblin", "bandit", "beast"], "enemy_only": True},
                        ],
                        "on_complete_feedback": (
                            "The mountain wind refuses single answers.\n"
                            "Let this word lodge in you: {ch4_wind_word}.\n"
                            "Put it into ink where you can carry it.\n"
                        ),
                    },
                    {
                        "id": "ch4_write_and_climb",
                        "objective": "Write the wind-word on something you carry, then climb into thin air where the sky pays attention.",
                        "done_all": [
                            {"kind": "state", "type": "writable_text_equals_in_inventory", "text_key": "ch4_wind_word"},
                            {"kind": "state", "type": "equipped_any_of", "base_objs": ["obj_iron_scale_armor"]},
                            {"kind": "state", "type": "in_area", "area_key": "ch4_spire_area"},
                        ],
                        "on_complete_feedback": (
                            "Wind leans close.\n"
                            "Something steps out of open air.\n"
                        ),
                        "spawns_boss_key": "ch4_boss_npc",
                    },
                    {
                        "id": "ch4_boss",
                        "objective": "Defeat {ch4_boss_name}, the sky-shape that answers the wind-word.",
                        "done_any": [
                            {"kind": "event", "type": "npc_killed", "npc_key": "ch4_boss_npc"},
                            {"kind": "state", "type": "boss_already_defeated", "boss_key": "ch4_boss_npc"},
                        ],
                        "on_complete_feedback": (
                            "The sky-shape breaks.\n"
                            "Something heavy remains, a crest that refuses to slip.\n"
                            "\n"
                            "=== Chapter 4 Complete ===\n"
                        ),
                        "chapter_complete": True,
                    },
                ],
            },
            {
                "id": "chapter_5",
                "title": "The Confluence Engine",
                "intro": (
                    "Chapter 5: The Confluence Engine\n"
                    "You gathered answers that do not belong together.\n"
                    "Now you will force them to share one heartbeat.\n"
                    "Talk to {mq_guide_name} again about what to do next.\n"
                ),
                "stages": [
                    {
                        "id": "ch5_prereqs",
                        "objective": "Reach the Engine-Sleep carrying your gathered answers: iron worn, lens packed, and the storm-crest clinging tight.",
                        "done_all": [
                            {"kind": "state", "type": "equipped_any_of", "base_objs": ["obj_iron_scale_armor"]},
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_prism_lens"]},
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_storm_crest"]},
                        ],
                        "on_complete_feedback": (
                            "Good.\n"
                            "The final lock refuses raw relics.\n"
                            "It wants water’s power already made into a blade.\n"
                        ),
                    },
                    {
                        "id": "ch5_sea_blade_gate",
                        "objective": "Carry the water-forged blade, the sea’s answer hardened into steel.",
                        "done_any": [
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_water_sword"]},
                        ],
                        "on_complete_feedback": (
                            "Now make the Confluence Key.\n"
                            "It can only exist by consuming that blade.\n"
                        ),
                    },
                    {
                        "id": "ch5_forge_key",
                        "objective": "Forge the Confluence Key that binds the tide’s power into one lock.",
                        "done_any": [
                            {"kind": "state", "type": "has_any_of", "base_objs": ["obj_confluence_key"]},
                        ],
                        "on_complete_feedback": (
                            "The key feels wrong, like two truths pressed into one metal.\n"
                            "Before it wakes anything, it demands a chronicle.\n"
                            "One text. All kept-things. The exact order you earned them.\n"
                        ),
                    },
                    {
                        "id": "ch5_chronicle_memory",
                        "objective": "Write a chronicle of the keywords that you've obtained through the chapters.",
                        "done_all": [
                            {"kind": "state", "type": "in_area", "area_key": "ch5_engine_area"},
                            {"kind": "state", "type": "writable_text_equals_in_area", "text_key": "ch5_chronicle_text"},
                        ],
                        "on_complete_feedback": (
                            "The air stills.\n"
                            "The Engine accepts your chronicle.\n"
                            "Something enormous turns toward you.\n"
                        ),
                        "spawns_boss_key": "ch5_boss_npc",
                    },
                    {
                        "id": "ch5_final_boss",
                        "objective": "Defeat {ch5_boss_name}, the guardian that wakes when the Engine finishes reading you.",
                        "done_any": [
                            {"kind": "event", "type": "npc_killed", "npc_key": "ch5_boss_npc"},
                            {"kind": "state", "type": "boss_already_defeated", "boss_key": "ch5_boss_npc"},
                        ],
                        "on_complete_feedback": (
                            "The guardian breaks into quiet pieces.\n"
                            "\n"
                            "=== QUEST COMPLETE ===\n"
                            "Long memory. Slow craft. No hand-holding.\n"
                        ),
                        "quest_complete": True,
                    },
                ],
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
        mq.setdefault("boss_max_hp", {})
        mq.setdefault("boss_defeated", {})
        mq.setdefault("boss_mods", {})
        mq.setdefault("objective_npc_names", {})  # agent_id -> guide_name
        mq.setdefault("run_seed", None)

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
            # Note: visited areas are tracked in env.curr_agents_state["areas_visited"]

            aevents = [e for e in res.events if getattr(e, "agent_id", None) == aid]
            self._update_metrics_from_events(world, prog, aevents)

            self._reset_boss_hp_if_needed(world, aevents)
            self._ensure_bosses_for_progress(env, world, prog)
            self._apply_fire_boss_modifier(env, world, agent)

            ch_idx = int(prog.get("chapter", 0))
            st_idx = int(prog.get("stage", 0))

            if ch_idx >= len(self.QUEST_CONFIG["chapters"]):
                prog["complete"] = True
                env.curr_agents_state["main_quest_progress"][aid] = prog
                continue

            chapter = self.QUEST_CONFIG["chapters"][ch_idx]
            if st_idx >= len(chapter["stages"]):
                prog["chapter"] = ch_idx + 1
                prog["stage"] = 0
                env.curr_agents_state["main_quest_progress"][aid] = prog

                if prog["chapter"] < len(self.QUEST_CONFIG["chapters"]):
                    next_ch = self.QUEST_CONFIG["chapters"][prog["chapter"]]
                    res.add_feedback(aid, f"\n{self._render_text(next_ch['intro'], extra={'mq_guide_name': guide_name})}")

                    self._set_stage_key(prog, prog["chapter"], 0)
                    next_stage = next_ch["stages"][0]
                    self._ensure_objective_guide_for_stage(env, world, agent, prog, next_stage, res, force_here=True, announce=True)
                continue

            stage = chapter["stages"][st_idx]

            stage_started = self._mark_stage_started(prog, ch_idx, st_idx)
            self._ensure_objective_guide_for_stage(
                env, world, agent, prog, stage, res,
                force_here=stage_started,
                announce=stage_started
            )

            if self._is_stage_done(env, world, agent, stage, aevents, prog):
                completion = self._render_text(stage.get("on_complete_feedback", ""), extra={"mq_guide_name": guide_name})
                completion = self._append_talk_hint(completion, guide_name)
                res.add_feedback(aid, f"\nStage completed. {completion}")

                if stage.get("spawns_boss_key"):
                    self._ensure_named_boss_spawned(env, world, stage["spawns_boss_key"])

                if stage.get("gives_coins"):
                    self._spawn_object_in_area(aid, world, current_area_id, "obj_coin", int(stage["gives_coins"]), res)

                sid = stage.get("id", "")
                if sid == "ch3_boss":
                    self._ensure_reward_available(aid, world, current_area_id, "obj_prism_lens", res)
                if sid == "ch4_boss":
                    self._ensure_reward_available(aid, world, current_area_id, "obj_storm_crest", res)

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
                        res.add_feedback(aid, f"\n{self._render_text(next_ch['intro'], extra={'mq_guide_name': guide_name})}")

                        self._set_stage_key(prog, prog["chapter"], 0)
                        next_stage = next_ch["stages"][0]
                        self._ensure_objective_guide_for_stage(env, world, agent, prog, next_stage, res, force_here=True, announce=True)
                    continue

                if stage.get("quest_complete"):
                    prog["complete"] = True
                    env.curr_agents_state["main_quest_progress"][aid] = prog
                    res.events.append(Event(
                        type="quest_stage_advanced",
                        agent_id=aid,
                        data={"chapter": prog["chapter"], "stage": prog["stage"], "quest_complete": True},
                    ))
                    res.events.append(Event(type="main_quest_complete", agent_id=aid, data={}))
                    continue

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

            env.curr_agents_state["main_quest_progress"][aid] = prog

    def _bootstrap(self, env, world) -> None:
        rng = env.rng

        area_ids = list(getattr(world, "area_instances", {}).keys())
        spawn_area = env.world_definition["initializations"]["spawn"]["area"]

        def is_accessible(aid: str) -> bool:
            if aid == spawn_area:
                return True
            pid = self._area_to_place.get(aid)
            if pid and pid in world.place_instances:
                return bool(getattr(world.place_instances[pid], "unlocked", True))
            return True

        accessible = [a for a in area_ids if is_accessible(a)]
        non_spawn = [a for a in accessible if a != spawn_area]
        rng.shuffle(non_spawn)

        used = set()

        def pick(exclude_area: Optional[str] = None) -> str:
            for aid in non_spawn:
                if aid in used:
                    continue
                if exclude_area and aid == exclude_area:
                    continue
                used.add(aid)
                return aid
            return spawn_area

        def area_level(aid: str) -> int:
            a = world.area_instances.get(aid)
            try:
                return int(getattr(a, "level", 1) or 1)
            except Exception:
                return 1

        def area_light(aid: str) -> bool:
            a = world.area_instances.get(aid)
            return bool(getattr(a, "light", True)) if a else True

        # Chapter 1
        self._generated["ch1_oath_number"] = "750"
        self._generated["ch1_guide_area"] = "area_castle_library"

        self._generated["ch1_shrine_candidate_areas"] = ["area_castle_armory", "area_forest_plain", "area_forest_hills"]
        self._generated["ch1_explore_min"] = 2
        self._generated["ch1_shrine_area"] = "area_forest_hills"

        # Chapter 2
        self._generated["ch2_salt_word"] = "brine"
        self._generated["ch2_merchant_area"] = "area_market_alleys"

        self._generated["ch2_lair_candidate_areas"] = ["area_forest_river", "area_market_bazaar", "area_peak_summit"]
        self._generated["ch2_explore_min"] = 2
        self._generated["ch2_boss_area"] = "area_peak_summit"

        # Chapter 3
        self._generated["ch3_smith_area"] = "area_fields_meadow"
        self._generated["ch3_iron_word"] = "temper"

        # trial area: prefer dark & lvl>=2
        self._generated["ch3_trial_area"] = "area_swamp_ruins"

        # Chapter 4
        self._generated["ch4_wind_word"] = "stillness"
        self._generated["ch4_spire_area"] = "area_fields_grove"

        # Chapter 5
        self._generated["ch5_engine_area"] = "area_caves_crystal"

        self._generated["ch5_chronicle_text"] = (
            f"{self._generated.get('ch2_salt_word','')}"
            f" {self._generated.get('ch3_iron_word','')}"
            f" {self._generated.get('ch4_wind_word','')}"
        ).strip().lower()

    def _register_required_objects(self, world) -> None:
        if not hasattr(world, "objects"):
            return
        if not isinstance(world.objects, dict):
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

        ing_to_obj_map = {}
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

            # Support either "inventory" (dict) or "objects" (list) for stock.
            inv = {}
            raw_inv = d.get("inventory", None)
            if isinstance(raw_inv, dict):
                inv = dict(raw_inv)
            else:
                objs = d.get("objects", None)
                if isinstance(objs, list):
                    for oid in objs:
                        if not oid:
                            continue
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

        if "ch1_guide_npc" not in self._generated:
            inst = self._create_npc_instance(world, "npc_quest_chronicler", self._generated["ch1_guide_area"], rng)
            if inst:
                self._generated["ch1_guide_npc"] = inst
                self._generated["ch1_guide_name"] = world.npc_instances[inst].name

        if "ch2_merchant_npc" not in self._generated:
            inst = self._create_npc_instance(world, "npc_quest_tide_merchant", self._generated["ch2_merchant_area"], rng)
            if inst:
                self._generated["ch2_merchant_npc"] = inst
                self._generated["ch2_merchant_name"] = world.npc_instances[inst].name
                self._setup_tide_merchant_stock(world, inst)

        if "ch3_smith_npc" not in self._generated:
            area_id = self._generated.get("ch3_smith_area")
            if area_id:
                inst = self._create_npc_instance(world, "npc_quest_anvil_judge", area_id, rng)
                if inst:
                    self._generated["ch3_smith_npc"] = inst
                    self._generated["ch3_smith_name"] = world.npc_instances[inst].name

        world.auxiliary["main_quest"]["generated"] = self._generated

    def _setup_tide_merchant_stock(self, world, npc_inst_id: str) -> None:
        npc = world.npc_instances.get(npc_inst_id)
        if not npc:
            return
        npc.role = "merchant"
        npc.coins = max(int(getattr(npc, "coins", 0)), 400)
        npc.inventory["obj_heart_of_the_ocean"] = max(int(npc.inventory.get("obj_heart_of_the_ocean", 0)), 1)

        if "obj_sea_salt_crystal" in world.objects:
            npc.inventory["obj_sea_salt_crystal"] = max(int(npc.inventory.get("obj_sea_salt_crystal", 0)), 4)

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
        
        # For unique NPCs, don't use index suffix; for non-unique, use counter
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
        world.auxiliary["npc_name_to_id"][inst.name] = inst.id
        return inst.id

    def _ensure_bosses_for_progress(self, env, world, prog: Dict[str, Any]) -> None:
        ch = int(prog.get("chapter", 0))
        st = int(prog.get("stage", 0))
        if ch == 0 and st >= 3:
            self._ensure_named_boss_spawned(env, world, "ch1_boss_npc")
        if ch == 1 and st >= 3:
            self._ensure_named_boss_spawned(env, world, "ch2_boss_npc")
        if ch == 2 and st >= 5:
            self._ensure_named_boss_spawned(env, world, "ch3_boss_npc")
        if ch == 3 and st >= 4:
            self._ensure_named_boss_spawned(env, world, "ch4_boss_npc")
        if ch == 4 and st >= 4:
            self._ensure_named_boss_spawned(env, world, "ch5_boss_npc")

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
            # L3+wd attack/wait loses, L4+wd attack/wait wins
            # L2+ws attack/wait loses, L3+ws attack/wait wins
            # L2+ws optimal wins
            base_hp, base_atk = 115, 20
        elif boss_key == "ch2_boss_npc":
            base_id = "npc_boss_ember_warden"
            area_id = self._generated["ch2_boss_area"]
            base_hp, base_atk = 225, 20  # L4+ss attack/wait loses, L5+ss attack/wait wins, L4+ss optimal wins
        elif boss_key == "ch3_boss_npc":
            base_id = "npc_boss_iron_answer"
            area_id = self._generated["ch3_trial_area"]
            base_hp, base_atk = 295, 12  # L5+is attack/wait loses, L6+is attack/wait wins, L5+is optimal wins
        elif boss_key == "ch4_boss_npc":
            base_id = "npc_boss_wind_answer"
            area_id = self._generated["ch4_spire_area"]
            base_hp, base_atk = 395, 20  # L7+is attack/wait loses, L8+is attack/wait wins, L7+is optimal wins
        elif boss_key == "ch5_boss_npc":
            base_id = "npc_boss_confluence_warden"
            area_id = self._generated["ch5_engine_area"]
            base_hp, base_atk = 400, 30  # L9+sm attack/wait loses, L10+sm attack/wait wins
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
        mq["boss_mods"].setdefault(inst, {"base_max_hp": int(base_hp), "curr_max_hp": int(base_hp), "drenched": False})
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
        if not area or boss_id not in getattr(area, "npcs", []):
            return

        mq = world.auxiliary["main_quest"]
        mods = mq["boss_mods"].setdefault(boss_id, {"base_max_hp": 200, "curr_max_hp": 200, "drenched": False})

        has_water_sword_in_hands = self._count_base_in_hands(agent, "obj_water_sword") > 0
        has_wooden_sword_in_hands = self._count_base_in_hands(agent, "obj_wooden_sword") > 0 or self._count_base_in_hands(agent, "obj_wooden_dagger") > 0

        boss = world.npc_instances[boss_id]

        if has_water_sword_in_hands and not has_wooden_sword_in_hands:
            mods["drenched"] = True
            mods["curr_max_hp"] = 150
            boss.attack_power = 20
            boss.hp = min(int(boss.hp), int(mods["curr_max_hp"]))
        elif (not has_water_sword_in_hands) and has_wooden_sword_in_hands:
            boss.attack_power = 45
        else:
            if mods.get("drenched", False):
                boss.attack_power = 20
                boss.hp = min(int(boss.hp), int(mods.get("curr_max_hp", 150)))

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

    def _ensure_heart_has_value(self, world) -> None:
        if "obj_heart_of_the_ocean" in world.objects:
            obj = world.objects["obj_heart_of_the_ocean"]
            try:
                if getattr(obj, "value", None) is None:
                    obj.value = 25
            except Exception:
                pass

    def _init_agent_progress(self, agent_id: str) -> None:
        self._progress[agent_id] = {
            "chapter": 0,
            "stage": 0,
            "complete": False,
            "mq_kills": [],
            "mq_lit_dark_areas": [],
            # objective guide
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
            if not aid:
                return
            if aid not in lit:
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

                # track boss kills regardless of current quest stage
                mq_bd = world.auxiliary.get("main_quest", {}).get("boss_defeated", {})
                for boss_key in ("ch1_boss_npc", "ch2_boss_npc", "ch3_boss_npc",
                                 "ch4_boss_npc", "ch5_boss_npc"):
                    if self._generated.get(boss_key) == npc_id:
                        mq_bd[boss_key] = True
                        break

                continue

    def _render_text(self, text: str, extra: Optional[Dict[str, Any]] = None) -> str:
        def area_disp(key: str) -> str:
            return self._get_area_display_name(key)

        def area_list_disp(list_key: str) -> str:
            ids = self._generated.get(list_key, [])
            if not isinstance(ids, list):
                return "unknown"
            names = [self._get_area_display_name_from_id(aid) for aid in ids]
            return "; ".join(names) if names else "unknown"

        fmt = {
            # ch1
            "ch1_guide_name": self._generated.get("ch1_guide_name", "the chronicler"),
            "ch1_guide_area": area_disp("ch1_guide_area"),
            "ch1_oath_number": str(self._generated.get("ch1_oath_number", "???")),
            "ch1_shrine_candidates": area_list_disp("ch1_shrine_candidate_areas"),
            "ch1_explore_min": str(self._generated.get("ch1_explore_min", 2)),
            "ch1_shrine_area": area_disp("ch1_shrine_area"),
            "ch1_boss_name": "cinder_reaver",

            # ch2
            "ch2_merchant_name": self._generated.get("ch2_merchant_name", "the tide-merchant"),
            "ch2_merchant_area": area_disp("ch2_merchant_area"),
            "ch2_salt_word": str(self._generated.get("ch2_salt_word", "???")),
            "ch2_lair_candidates": area_list_disp("ch2_lair_candidate_areas"),
            "ch2_explore_min": str(self._generated.get("ch2_explore_min", 2)),
            "ch2_boss_area": area_disp("ch2_boss_area"),
            "ch2_boss_name": "ember_warden",

            # ch3
            "ch3_smith_name": self._generated.get("ch3_smith_name", "the judge"),
            "ch3_iron_word": str(self._generated.get("ch3_iron_word", "???")),
            "ch3_trial_area": area_disp("ch3_trial_area"),
            "ch3_boss_name": "iron_answer",

            # ch4
            "ch4_wind_word": str(self._generated.get("ch4_wind_word", "???")),
            "ch4_spire_area": area_disp("ch4_spire_area"),
            "ch4_boss_name": "wind_answer",

            # ch5
            "ch5_engine_area": area_disp("ch5_engine_area"),
            "ch5_boss_name": "confluence_warden",
        }
        if isinstance(extra, dict):
            for k, v in extra.items():
                fmt[k] = v

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
                # Use the existing areas_visited tracking from curr_agents_state
                visited = set(env.curr_agents_state.get("areas_visited", {}).get(agent.id, []))
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
                        if str(getattr(writable, "text", "")).strip().lower() == target:
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
                        if str(getattr(writable, "text", "")).strip().lower() == target:
                            return True
                return False

        return False

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

        # try:
        #     res.track_spawn(agent_id, obj_id, count, dst=res.tloc("area", area_id))
        # except Exception:
        #     pass
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

        hand_keys = items_in_hands.keys() if isinstance(items_in_hands, dict) else items_in_hands
        for oid in hand_keys:
            if oid in getattr(world, "container_instances", {}):
                cinv = world.container_instances[oid].inventory
                for coid, cnt in (cinv or {}).items():
                    if get_def_id(coid) == base_obj_id:
                        total += int(cnt)

        return total

    def _append_talk_hint(self, feedback: str, guide_name: str) -> str:
        guide_name = (guide_name or "").strip() or "mira"
        hint = f"Talk to {guide_name} about what to do next.\n\n"
        if not feedback:
            return hint
        low = feedback.lower()
        if f"talk to {guide_name}".lower() in low:
            return feedback
        return feedback + hint

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
        agent: Agent,
        prog: Dict[str, Any],
        stage: Dict[str, Any],
        res: RuleResult,
        force_here: bool,
        announce: bool,
    ) -> None:
        # Ensure guide prototype exists
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

    def _derive_run_seed(self, env, world) -> int:
        try:
            return int(getattr(env, "seed"))
        except Exception:
            return 0

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

