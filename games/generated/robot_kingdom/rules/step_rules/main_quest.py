import copy
import math
from utils import *
from games.generated.robot_kingdom.rule import BaseStepRule, RuleContext, RuleResult, Event
from typing import Dict, Optional, Set, Tuple, List, Any
from games.generated.robot_kingdom.world import NPC, Object, Area, Place, Path, Container, Writable
from games.generated.robot_kingdom.agent import Agent
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
        {'id': 'obj_quest_recon_report', 'name': 'recon_report', 'category': 'tool', 'usage': 'writable', 'value': None, 'size': 1, 'attack': 0, 'description': 'A blank war parchment prepared for recording reconnaissance findings.', 'text': '', 'max_text_length': 200, 'level': 1, 'craft': {'ingredients': {}, 'dependencies': []}},
        {'id': 'obj_quest_gate_cipher', 'name': 'gate_cipher', 'category': 'tool', 'usage': 'unlock', 'value': 50, 'size': 2, 'attack': 0, 'description': 'A mechanical cipher device that can decode the locking mechanisms of demon-forged gates.', 'level': 3, 'craft': {'ingredients': {'obj_cog': 2, 'obj_scrap_iron': 1}, 'dependencies': ['obj_workbench']}},
        {'id': 'obj_quest_war_banner', 'name': 'war_banner', 'category': 'tool', 'usage': 'craft', 'value': 30, 'size': 3, 'attack': 0, 'description': 'A rallying banner stitched from salvage cloth and charred timber. Inspires allied NPCs when planted.', 'level': 2, 'craft': {'ingredients': {'obj_salvage_cloth': 2, 'obj_wooden_rod': 1}, 'dependencies': []}},
        {'id': 'obj_quest_demon_core', 'name': 'demon_core', 'category': 'material', 'usage': 'craft', 'value': 200, 'size': 5, 'attack': 0, 'description': 'The pulsing infernal core ripped from the heart of a war engine. Radiates malevolent heat.', 'level': 5, 'craft': {'ingredients': {}, 'dependencies': []}},
        {'id': 'obj_quest_suppression_charge', 'name': 'suppression_charge', 'category': 'weapon', 'usage': 'attack', 'value': 150, 'size': 4, 'attack': 50, 'description': 'An explosive device designed to overload demon robot power systems. Requires hellcoal concentrate and cog assemblies.', 'level': 5, 'craft': {'ingredients': {'obj_hellcoal_concentrate': 2, 'obj_cog_assembly': 1, 'obj_oil_flask': 1}, 'dependencies': ['obj_workbench']}},
        {'id': 'obj_quest_marshals_orders', 'name': 'marshals_orders', 'category': 'tool', 'usage': 'writable', 'value': 10, 'size': 1, 'attack': 0, 'description': 'Sealed orders from the Old Marshal containing strategic directives.', 'text': '', 'max_text_length': 200, 'level': 1, 'craft': {'ingredients': {}, 'dependencies': []}},
    ]

    required_npcs: List[Dict[str, Any]] = [
        {'id': 'npc_quest_scout_lyra', 'name': 'scout_lyra', 'enemy': False, 'unique': True, 'role': 'guide', 'quest': True, 'description': 'A wiry survivor with ash-stained goggles and keen eyes. She knows every ruin and crater in the Scorched Vale.', 'base_attack_power': 100, 'base_hp': 150, 'objects': []},
        {'id': 'npc_quest_forgemaster_dren', 'name': 'forgemaster_dren', 'enemy': False, 'unique': True, 'role': 'quest', 'quest': True, 'description': 'A burly smith who once built the demon machines before defecting. He knows their weaknesses intimately.', 'base_attack_power': 150, 'base_hp': 200, 'objects': []},
        {'id': 'npc_boss_scrap_warden', 'name': 'scrap_warden', 'enemy': True, 'unique': True, 'role': 'boss', 'quest': True, 'description': 'A towering demon robot assembled from the wreckage of a dozen fallen machines. Its body crackles with residual hellfire.', 'base_attack_power': 18, 'base_hp': 120, 'combat_pattern': ['attack', 'defend', 'attack', 'attack', 'wait'], 'objects': ['obj_quest_gate_cipher']},
        {'id': 'npc_boss_infernal_architect', 'name': 'infernal_architect', 'enemy': True, 'unique': True, 'role': 'boss', 'quest': True, 'description': 'The master war engine that designs and births all other demon robots. A colossal machine of brimite and hellfire, it guards the Foundry Core.', 'base_attack_power': 30, 'base_hp': 250, 'combat_pattern': ['attack', 'attack', 'defend', 'wait', 'attack', 'attack', 'defend'], 'objects': ['obj_quest_demon_core']},
        {'id': 'npc_quest_ember_pack_1', 'name': 'ember_scavenger', 'enemy': True, 'unique': False, 'role': 'enemy', 'quest': True, 'description': 'A skittering demon robot that picks through ruins for scrap to feed the Foundry.', 'base_attack_power': 8, 'base_hp': 40, 'combat_pattern': ['attack', 'wait', 'attack'], 'objects': ['obj_scrap_iron']},
        {'id': 'npc_quest_iron_ravager_1', 'name': 'iron_ravager', 'enemy': True, 'unique': False, 'role': 'enemy', 'quest': True, 'description': 'A heavy assault demon robot with iron-plated limbs and grinding jaw mechanisms.', 'base_attack_power': 15, 'base_hp': 80, 'combat_pattern': ['attack', 'attack', 'defend', 'attack'], 'objects': ['obj_cog']},
        {'id': 'npc_quest_iron_ravager_2', 'name': 'iron_ravager', 'enemy': True, 'unique': False, 'role': 'enemy', 'quest': True, 'description': 'Another heavy assault demon robot patrolling the gulch perimeter.', 'base_attack_power': 15, 'base_hp': 80, 'combat_pattern': ['defend', 'attack', 'attack', 'wait', 'attack'], 'objects': ['obj_hellcoal', 'obj_key']},
        {'id': 'npc_quest_scrap_auto_1', 'name': 'scrap_automaton', 'enemy': True, 'unique': False, 'role': 'enemy', 'quest': True, 'description': 'A crude demon robot cobbled from charred timber and scrap iron. Weak but persistent.', 'base_attack_power': 5, 'base_hp': 25, 'combat_pattern': ['attack', 'wait', 'attack'], 'objects': ['obj_charcoal_brick']},
        {'id': 'npc_quest_colossus_guard', 'name': 'infernal_colossus', 'enemy': True, 'unique': False, 'role': 'enemy', 'quest': True, 'description': 'A massive demon robot stationed at the Foundry gate. Its fiend-plated body resists most weapons.', 'base_attack_power': 22, 'base_hp': 150, 'combat_pattern': ['defend', 'attack', 'attack', 'defend', 'attack', 'wait'], 'objects': ['obj_fiend_plate']},

        # objective guide (invincible)
        {"type": "npc", "id": OBJECTIVE_NPC_BASE_ID, "name": "mira",
        "enemy": False, "unique": True, "role": "guide", "quest": True,
        "description": "",
        "base_attack_power": OBJECTIVE_NPC_ATK, "slope_attack_power": 0,
        "base_hp": OBJECTIVE_NPC_MAX_HP, "slope_hp": 0,
        "objects": []},
    ]

    undistributable_objects: List[str] = ['obj_quest_recon_report', 'obj_quest_gate_cipher', 'obj_quest_war_banner', 'obj_quest_demon_core', 'obj_quest_suppression_charge', 'obj_quest_marshals_orders']

    QUEST_CONFIG: Dict[str, Any] = {
        'title': 'The Siege of Broken Gears',
        'intro': (
            "The world lies in ash and iron. Demon robots — infernal machines fused with hellfire and rusted malice — stalk the ruins of civilization. You are a scavenger turned soldier, recruited by {mq_guide_name} to push back the mechanical tide. Begin at the Ashen Bastion, humanity's last outpost, and forge a path through the scorched wastes to the Devils Foundry, where the war engines are born. Destroy their source, or perish in the attempt."
        ),
        'chapters': [
            {
                'id': 'chapter_1',
                'title': 'Ashes and Orders',
                'intro': (
                    'You arrive at the Ashen Bastion, the last fortified outpost standing against the demon robot horde. {mq_guide_name} told you to report to {ch1_marshal_name} for your first assignment. The bastion is battered but still holds. Explore, gather your bearings, and prepare for what lies beyond the walls.'
                ),
                'stages': [
                    {
                        'id': 'ch1_root',
                        'objective': 'Complete your orientation at the Ashen Bastion.',
                        'chapter_complete': True,
                        'requires': ['ch1_explore_bastion', 'ch1_receive_orders'],
                        'children': ['ch1_explore_bastion', 'ch1_receive_orders'],
                    },
                    {
                        'id': 'ch1_explore_bastion',
                        'objective': 'Explore the Ashen Bastion and learn its layout.',
                        'on_complete_feedback': 'You now know the bastion well — its courtyard, barracks, cellar, and watchtower.',
                        'requires': ['ch1_visit_areas', 'ch1_find_equipment'],
                        'children': ['ch1_visit_areas', 'ch1_find_equipment'],
                    },
                    {
                        'id': 'ch1_visit_areas',
                        'objective': 'Visit at least 3 areas within the Ashen Bastion to familiarize yourself.',
                        'hint': 'Move through the bastion — try the barracks, cellar, and watchtower.',
                        'done_all': [
                            {
                                'kind': 'state',
                                'type': 'visited_k_of_area_set',
                                'area_set_key': 'ch1_bastion_areas',
                                'k_key': 'ch1_visit_count',
                            },
                        ],
                        'on_complete_feedback': (
                            "You've scouted the bastion thoroughly. In the watchtower, you noticed strange inscriptions on the wall: 'IRON-VEIL-7'. They look like an old military cipher code."
                        ),
                    },
                    {
                        'id': 'ch1_find_equipment',
                        'objective': 'Pick up a weapon and a piece of armor from the bastion stores.',
                        'hint': (
                            'Search the bastion areas for basic equipment — a club and a buckler should be available.'
                        ),
                        'done_all': [
                            {
                                'kind': 'state',
                                'type': 'has_category',
                                'category': 'weapon',
                            },
                            {
                                'kind': 'state',
                                'type': 'has_category',
                                'category': 'armor',
                            },
                        ],
                        'on_complete_feedback': (
                            "You're armed and armored. Not much, but better than bare fists against demon steel."
                        ),
                    },
                    {
                        'id': 'ch1_receive_orders',
                        'objective': 'Report to the Old Marshal and receive your mission orders.',
                        'on_complete_feedback': 'The Marshal nods approvingly. You are ready to venture beyond the bastion walls.',
                        'requires': ['ch1_talk_marshal', 'ch1_record_cipher'],
                        'children': ['ch1_talk_marshal', 'ch1_record_cipher'],
                    },
                    {
                        'id': 'ch1_talk_marshal',
                        'objective': 'Find and speak with {ch1_marshal_name} in the bastion.',
                        'hint': (
                            'The Old Marshal is stationed somewhere in the bastion. Seek him out and talk to him.'
                        ),
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'talked_to_npc',
                                'description': 'Player has talked to the Old Marshal NPC at least once.',
                                'implementation_hint': (
                                    'Track in agent memory or check dialogue history — set a flag when talk_to action targets this NPC. Check if agent has ever performed talk_to on npc_old_marshal.'
                                ),
                                'npc_key': 'ch1_marshal_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            "The Old Marshal speaks gravely: 'The demon machines are massing in the Scorched Vale. I need you to scout the ruins and report back. Take this parchment — write down the cipher code from the watchtower wall before you leave. We'll need it to decode enemy gate locks later. The code was inscribed by our last scout team: IRON-VEIL-7.'"
                        ),
                    },
                    {
                        'id': 'ch1_record_cipher',
                        'objective': 'Write down the cipher code on the recon report before departing.',
                        'hint': (
                            'You learned a code from the watchtower inscriptions and the Marshal confirmed it. Write it on your recon report using a charcoal stylus.'
                        ),
                        'done_all': [
                            {
                                'kind': 'state',
                                'type': 'paper_text_equals_in_hands',
                                'text_key': 'ch1_cipher_code',
                            },
                        ],
                        'on_complete_feedback': (
                            'The cipher code is safely recorded. This will be crucial for breaching demon-locked gates in the field.'
                        ),
                    },
                ],
                'root_goal': 'ch1_root',
            },
            {
                'id': 'chapter_2',
                'title': 'Scorched Reconnaissance',
                'intro': (
                    'With orders in hand and cipher recorded, you venture beyond the bastion into the Scorched Vale — a blasted wasteland of craters and crumbling ruins. {ch2_lyra_name} is supposed to meet you at the trailhead. The demon robots have been spotted here in growing numbers. Scout the area, eliminate threats, and rally what allies you can find.'
                ),
                'stages': [
                    {
                        'id': 'ch2_root',
                        'objective': (
                            'Complete reconnaissance of the Scorched Vale and secure passage to Ironwreck Gulch.'
                        ),
                        'chapter_complete': True,
                        'requires': ['ch2_recon_ops', 'ch2_secure_passage'],
                        'children': ['ch2_recon_ops', 'ch2_secure_passage'],
                    },
                    {
                        'id': 'ch2_recon_ops',
                        'objective': 'Conduct reconnaissance operations in the Scorched Vale.',
                        'requires': ['ch2_meet_lyra', 'ch2_clear_vale'],
                        'children': ['ch2_meet_lyra', 'ch2_clear_vale'],
                    },
                    {
                        'id': 'ch2_meet_lyra',
                        'objective': 'Rendezvous with {ch2_lyra_name} at the Vale trailhead.',
                        'hint': 'Head into the Scorched Vale and look for Scout Lyra near the entrance.',
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'talked_to_npc_with_item',
                                'description': (
                                    'Player talks to Scout Lyra while holding the recon report with the cipher code written on it.'
                                ),
                                'implementation_hint': (
                                    'Check that the player has performed talk_to on this NPC AND currently holds a writable object (obj_quest_recon_report) whose text content equals the cipher code.'
                                ),
                                'npc_key': 'ch2_lyra_npc',
                                'text_key': 'ch1_cipher_code',
                            },
                        ],
                        'on_complete_feedback': (
                            "Lyra examines your recon report and nods. 'Good, you have the cipher. IRON-VEIL-7 — that's the prefix for all demon gate locks in this sector. I've been tracking a pack of scrap automatons in the ruins. Clear them out and I'll show you the route to the Gulch. Also, Kael Ashworth has set up a trading post nearby — buy supplies if you need them.'"
                        ),
                    },
                    {
                        'id': 'ch2_clear_vale',
                        'objective': 'Eliminate demon robot threats in the Vale.',
                        'requires': ['ch2_kill_scrap', 'ch2_craft_banner'],
                        'children': ['ch2_kill_scrap', 'ch2_craft_banner'],
                    },
                    {
                        'id': 'ch2_kill_scrap',
                        'objective': 'Destroy the scrap automaton lurking in the Vale ruins.',
                        'hint': (
                            'Head deeper into the Vale. A scrap automaton patrols the ruins — engage and destroy it.'
                        ),
                        'done_all': [
                            {
                                'kind': 'event',
                                'type': 'npc_killed',
                                'npc_key': 'ch2_scrap_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            'The scrap automaton collapses in a shower of sparks and charred timber. Among the wreckage you find salvageable materials. Lyra mentioned that crafting a war banner could rally any surviving scouts in the area.'
                        ),
                    },
                    {
                        'id': 'ch2_craft_banner',
                        'objective': 'Craft a war banner to signal allied scouts in the region.',
                        'hint': (
                            'You need salvage cloth and a wooden rod. Craft the war banner — no workbench required.'
                        ),
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'object_in_area',
                                'description': 'Player has crafted and dropped the war banner in the rally area to plant it.',
                                'implementation_hint': (
                                    'Check if an instance of obj_quest_war_banner exists in the objects list of the target area.'
                                ),
                                'obj_key': 'ch2_banner_obj',
                                'area_key': 'ch2_rally_area',
                            },
                        ],
                        'on_complete_feedback': (
                            'The war banner flutters in the ashen wind. Any allied scouts in range will see it and know the Vale is being reclaimed.'
                        ),
                    },
                    {
                        'id': 'ch2_secure_passage',
                        'objective': 'Secure passage from the Vale to Ironwreck Gulch.',
                        'on_complete_feedback': "The path to Ironwreck Gulch is open. The demon machines won't stop you now.",
                        'requires': ['ch2_buy_supplies', 'ch2_defeat_warden'],
                        'children': ['ch2_buy_supplies', 'ch2_defeat_warden'],
                    },
                    {
                        'id': 'ch2_buy_supplies',
                        'objective': 'Purchase essential supplies from {ch2_merchant_name} before the assault.',
                        'hint': (
                            'Find the merchant and sell salvage to earn crown shards, then buy what you need for tougher fights ahead.'
                        ),
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'npc_has_coins',
                                'description': (
                                    'The merchant NPC has accumulated at least 15 coins from player sales, indicating the player has traded enough.'
                                ),
                                'implementation_hint': 'Check world.npc_instances[npc_id].coins >= 15',
                                'npc_key': 'ch2_merchant_npc',
                                'threshold': 15,
                            },
                        ],
                        'on_complete_feedback': (
                            "Kael nods as he counts the crown shards. 'Good trading. You'll need strong gear where you're headed. The Scrap Warden guards the passage to the Gulch — it's no ordinary machine.'"
                        ),
                    },
                    {
                        'id': 'ch2_defeat_warden',
                        'objective': 'Defeat the Scrap Warden guarding the passage to Ironwreck Gulch.',
                        'hint': (
                            'Equip your best weapon and armor. The Scrap Warden is tough — watch its attack patterns and defend when it strikes hard.'
                        ),
                        'done_any': [
                            {
                                'kind': 'event',
                                'type': 'npc_killed',
                                'npc_key': 'ch2_warden_npc',
                            },
                            {
                                'kind': 'state',
                                'type': 'boss_already_defeated',
                                'npc_key': 'ch2_warden_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            'The Scrap Warden crashes to the ground, its hellfire eyes dimming. Among its remains you find a gate cipher device — it must have been guarding it. This will unlock the demon-forged gates ahead.'
                        ),
                        'spawns_boss_key': 'ch2_warden_npc',
                    },
                ],
                'root_goal': 'ch2_root',
            },
            {
                'id': 'chapter_3',
                'title': 'The Iron Gulch',
                'intro': (
                    "With the Scrap Warden destroyed and the gate cipher in hand, you push into Ironwreck Gulch — a canyon choked with the rusting carcasses of demon machines. {ch3_dren_name}, a defector who once built these abominations, is said to be hiding in the gulch forge. His knowledge is essential for crafting weapons capable of piercing the Foundry's defenses. But the gulch crawls with iron ravagers, and every step deeper is a step closer to the heart of the enemy."
                ),
                'stages': [
                    {
                        'id': 'ch3_root',
                        'objective': (
                            'Secure Ironwreck Gulch and prepare an arsenal for the assault on the Devils Foundry.'
                        ),
                        'chapter_complete': True,
                        'requires': ['ch3_infiltrate_gulch', 'ch3_forge_arsenal'],
                        'children': ['ch3_infiltrate_gulch', 'ch3_forge_arsenal'],
                    },
                    {
                        'id': 'ch3_infiltrate_gulch',
                        'objective': 'Infiltrate the gulch and establish contact with the Forgemaster.',
                        'requires': ['ch3_clear_entrance', 'ch3_find_dren'],
                        'children': ['ch3_clear_entrance', 'ch3_find_dren'],
                    },
                    {
                        'id': 'ch3_clear_entrance',
                        'objective': 'Clear the gulch entrance of demon robot patrols.',
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'area_cleared',
                                'description': 'All enemy NPCs in the gulch entrance area have been killed.',
                                'implementation_hint': 'Check that no alive enemy NPC instances remain in the target area.',
                                'area_key': 'ch3_entrance_area',
                            },
                        ],
                        'on_complete_feedback': (
                            'The gulch entrance is secured. The wreckage of demon robots litters the canyon floor.'
                        ),
                        'requires': ['ch3_kill_ravager_1', 'ch3_kill_ember'],
                        'children': ['ch3_kill_ravager_1', 'ch3_kill_ember'],
                    },
                    {
                        'id': 'ch3_kill_ravager_1',
                        'objective': 'Destroy the iron ravager patrolling the gulch entrance.',
                        'hint': (
                            'This is a tougher opponent than scrap automatons. Use your best weapon and defend strategically.'
                        ),
                        'done_all': [
                            {
                                'kind': 'event',
                                'type': 'npc_killed',
                                'npc_key': 'ch3_ravager1_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            "The iron ravager's grinding jaws seize and go still. Its cog mechanisms could be useful for crafting."
                        ),
                    },
                    {
                        'id': 'ch3_kill_ember',
                        'objective': 'Destroy the ember scavenger skulking in the gulch entrance.',
                        'hint': 'A smaller but fast demon robot. Attack quickly before it can flee.',
                        'done_all': [
                            {
                                'kind': 'event',
                                'type': 'npc_killed',
                                'npc_key': 'ch3_ember_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            "The ember scavenger's glowing eyes flicker out. It drops scrap iron that could be refined at a forge."
                        ),
                    },
                    {
                        'id': 'ch3_find_dren',
                        'objective': (
                            "Find {ch3_dren_name} in the gulch and learn how to build weapons against the Foundry's defenses."
                        ),
                        'hint': 'The Forgemaster is hiding deeper in the gulch. Explore further to find him.',
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'talked_to_npc_in_area',
                                'description': 'Player has talked to Forgemaster Dren in the forge area.',
                                'implementation_hint': (
                                    'Check that player is in the forge area AND has performed talk_to on this NPC (check dialogue/interaction flag for the NPC).'
                                ),
                                'npc_key': 'ch3_dren_npc',
                                'area_key': 'ch3_forge_area',
                            },
                        ],
                        'on_complete_feedback': (
                            "Dren looks you over with weary eyes. 'So the Marshal sent you. Good. Listen carefully — the Infernal Architect in the Foundry Core has a brimite-reinforced chassis. Normal weapons will barely scratch it. You need suppression charges — explosives that overload their power systems. I'll need you to gather hellcoal concentrate, cog assemblies, and oil flasks. Craft them at my war bench here. Also — the Foundry gate is locked with a fiend-alloy mechanism. The cipher you carry will help, but you also need a physical key. Kill the ravager in the scrapyard — it swallowed one. One more thing: the Architect's weak point is its central power conduit. When you fight it, look for moments when it pauses to recharge — that's when you strike hardest.'"
                        ),
                    },
                    {
                        'id': 'ch3_forge_arsenal',
                        'objective': 'Build an arsenal capable of breaching the Devils Foundry.',
                        'requires': ['ch3_obtain_key', 'ch3_craft_weapons'],
                        'children': ['ch3_obtain_key', 'ch3_craft_weapons'],
                    },
                    {
                        'id': 'ch3_obtain_key',
                        'objective': 'Obtain the siege key from the scrapyard ravager.',
                        'requires': ['ch3_kill_ravager_2', 'ch3_pickup_key'],
                        'children': ['ch3_kill_ravager_2', 'ch3_pickup_key'],
                    },
                    {
                        'id': 'ch3_kill_ravager_2',
                        'objective': 'Destroy the iron ravager in the scrapyard that swallowed the siege key.',
                        'hint': (
                            'Head to the scrapyard area. Another iron ravager patrols there — it carries the key you need.'
                        ),
                        'done_all': [
                            {
                                'kind': 'event',
                                'type': 'npc_killed',
                                'npc_key': 'ch3_ravager2_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            'The ravager collapses. Among the twisted metal of its gut, something glints — the siege key, still intact.'
                        ),
                    },
                    {
                        'id': 'ch3_pickup_key',
                        'objective': "Retrieve the siege key from the ravager's wreckage.",
                        'hint': "The key should be among the fallen ravager's remains. Pick it up.",
                        'requires': ['ch3_kill_ravager_2'],
                        'done_all': [
                            {
                                'kind': 'state',
                                'type': 'has_any_of',
                                'base_objs': ['obj_key'],
                            },
                        ],
                        'on_complete_feedback': (
                            'The siege key is warm to the touch, still radiating residual hellfire. This will open the locked passages ahead.'
                        ),
                    },
                    {
                        'id': 'ch3_craft_weapons',
                        'objective': 'Craft advanced weapons for the Foundry assault.',
                        'requires': ['ch3_craft_suppression', 'ch3_upgrade_armor'],
                        'children': ['ch3_craft_suppression', 'ch3_upgrade_armor'],
                    },
                    {
                        'id': 'ch3_craft_suppression',
                        'objective': 'Craft a suppression charge at the war bench in the forge.',
                        'hint': (
                            'Dren told you the recipe: hellcoal concentrate, cog assemblies, and an oil flask. Gather materials and craft at the war bench.'
                        ),
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'player_has_crafted_specific',
                                'description': 'Player has crafted at least one suppression charge.',
                                'implementation_hint': (
                                    "Check if obj_quest_suppression_charge exists in player's hands or inventory (agent.hands or agent.inventory contains an object with base_id == 'obj_quest_suppression_charge')."
                                ),
                                'obj_id': 'obj_quest_suppression_charge',
                            },
                        ],
                        'on_complete_feedback': (
                            "The suppression charge hums with contained energy. Dren inspects it approvingly. 'That'll crack the Architect's power grid. But you'll need to get close to use it.'"
                        ),
                    },
                    {
                        'id': 'ch3_upgrade_armor',
                        'objective': 'Equip iron-tier or better armor before assaulting the Foundry.',
                        'hint': (
                            "You can craft iron scale armor or buy it from merchants. The Foundry's defenders hit hard — you need proper protection."
                        ),
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'equipped_armor_level',
                                'description': 'Player has equipped armor of level 4 or higher.',
                                'implementation_hint': 'Check agent.equipped_armor — if it exists, check if its level >= 4.',
                                'min_level': 4,
                            },
                        ],
                        'on_complete_feedback': (
                            "Your armor gleams dully in the forge light. You're as ready as you'll ever be for the Foundry."
                        ),
                    },
                ],
                'root_goal': 'ch3_root',
            },
            {
                'id': 'chapter_4',
                'title': 'Heart of the Foundry',
                'intro': (
                    "The Devils Foundry looms ahead — a volcanic fortress of black iron and brimstone where the demon robots are manufactured. With siege key, suppression charge, and hard-won armor, you must breach the gate, fight through the furnace room, and reach the core where the Infernal Architect — the master war engine — orchestrates the production of all demon machines. {ch3_dren_name}'s words echo in your mind: strike when it pauses to recharge. This is the final assault. End the demon robot plague at its source."
                ),
                'stages': [
                    {
                        'id': 'ch4_root',
                        'objective': 'Destroy the Infernal Architect and end the demon robot threat.',
                        'quest_complete': True,
                        'requires': ['ch4_breach_foundry', 'ch4_destroy_architect'],
                        'children': ['ch4_breach_foundry', 'ch4_destroy_architect'],
                    },
                    {
                        'id': 'ch4_breach_foundry',
                        'objective': 'Breach the Devils Foundry defenses.',
                        'requires': ['ch4_enter_gate', 'ch4_clear_furnace'],
                        'children': ['ch4_enter_gate', 'ch4_clear_furnace'],
                    },
                    {
                        'id': 'ch4_enter_gate',
                        'objective': 'Use the siege key to enter the Foundry gate.',
                        'requires': ['ch4_reach_gate', 'ch4_defeat_colossus'],
                        'children': ['ch4_reach_gate', 'ch4_defeat_colossus'],
                    },
                    {
                        'id': 'ch4_reach_gate',
                        'objective': 'Navigate to the Devils Foundry gate.',
                        'hint': (
                            'Use the siege key to unlock the passage from Ironwreck Gulch to the Foundry. Keep moving forward.'
                        ),
                        'done_all': [
                            {
                                'kind': 'state',
                                'type': 'in_area',
                                'area_key': 'ch4_gate_area',
                            },
                        ],
                        'on_complete_feedback': (
                            'The Foundry gate towers before you — a massive portal of fiend-alloy and brimstone. The air reeks of sulfur and machine oil. A colossal guardian stands watch.'
                        ),
                    },
                    {
                        'id': 'ch4_defeat_colossus',
                        'objective': 'Defeat the infernal colossus guarding the Foundry gate.',
                        'hint': (
                            'This is a heavily armored foe. Use your strongest weapon and be patient — defend when it attacks in succession, then counter-strike.'
                        ),
                        'done_any': [
                            {
                                'kind': 'event',
                                'type': 'npc_killed',
                                'npc_key': 'ch4_colossus_npc',
                            },
                            {
                                'kind': 'state',
                                'type': 'boss_already_defeated',
                                'npc_key': 'ch4_colossus_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            'The infernal colossus topples with an earth-shaking crash. Fiend plates scatter across the ground. The gate mechanism groans — with the guardian destroyed, the way into the furnace room is open.'
                        ),
                        'spawns_boss_key': 'ch4_colossus_npc',
                    },
                    {
                        'id': 'ch4_clear_furnace',
                        'objective': 'Secure the furnace room and prepare for the final assault.',
                        'requires': ['ch4_sell_to_vessa', 'ch4_sabotage_furnace'],
                        'children': ['ch4_sell_to_vessa', 'ch4_sabotage_furnace'],
                    },
                    {
                        'id': 'ch4_sell_to_vessa',
                        'objective': (
                            'Trade with {ch4_vessa_name} who has infiltrated the Foundry to gather last-minute supplies.'
                        ),
                        'hint': (
                            'Vessa Ironhand has set up a hidden trading post inside the furnace room. Sell excess materials and buy what you need for the final fight.'
                        ),
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'bought_from_merchant',
                                'description': 'Player has purchased at least one item from Vessa Ironhand.',
                                'implementation_hint': (
                                    "Check if the player has performed a buy action targeting this NPC — track via event log or check if NPC's sold_items counter > 0."
                                ),
                                'npc_key': 'ch4_vessa_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            "Vessa hands over the goods with a grim smile. 'The Architect is in the core chamber beyond. It's surrounded by production lines — destroy them and it loses its ability to manufacture reinforcements. Good luck, soldier.'"
                        ),
                    },
                    {
                        'id': 'ch4_sabotage_furnace',
                        'objective': 'Sabotage the demon robot production line in the furnace room.',
                        'hint': (
                            'Dren mentioned production lines. Drop an oil bomb or hellfire bomb into the furnace to disable it. Alternatively, jam a cog assembly into the machinery.'
                        ),
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'object_in_area',
                                'description': (
                                    'Player has dropped an explosive or cog assembly in the furnace room to sabotage the production line.'
                                ),
                                'implementation_hint': (
                                    "Check if any object with base_id in ['obj_oil_bomb', 'obj_hellfire_bomb', 'obj_cog_assembly'] exists in the furnace room area's objects list."
                                ),
                                'target_objs': ['obj_oil_bomb', 'obj_hellfire_bomb', 'obj_cog_assembly'],
                                'area_key': 'ch4_furnace_area_sabotage',
                            },
                        ],
                        'on_complete_feedback': (
                            "The production machinery screeches and grinds to a halt. Sparks fly as gears lock and conveyor belts seize. The Architect won't be getting reinforcements. The path to the core is clear."
                        ),
                    },
                    {
                        'id': 'ch4_destroy_architect',
                        'objective': 'Confront and destroy the Infernal Architect in the Foundry Core.',
                        'done_all': [
                            {
                                'kind': 'custom',
                                'type': 'object_in_area',
                                'description': (
                                    "Player has dropped the demon core in the foundry core area after defeating the Architect, symbolizing the destruction of the Foundry's heart."
                                ),
                                'implementation_hint': "Check if obj_quest_demon_core exists in the target area's objects list.",
                                'obj_key': 'ch4_demon_core_obj',
                                'area_key': 'ch4_core_area_final',
                            },
                        ],
                        'on_complete_feedback': (
                            'You place the ruptured demon core on the ground. It pulses once, twice, then goes dark forever. The Foundry shudders and begins to collapse. The demon robot threat is ended.'
                        ),
                        'requires': ['ch4_enter_core', 'ch4_final_battle'],
                        'children': ['ch4_enter_core', 'ch4_final_battle'],
                    },
                    {
                        'id': 'ch4_enter_core',
                        'objective': 'Enter the Foundry Core where the Infernal Architect resides.',
                        'hint': 'Press deeper into the Foundry. The core is the innermost chamber.',
                        'done_all': [
                            {
                                'kind': 'state',
                                'type': 'in_area',
                                'area_key': 'ch4_core_area',
                            },
                        ],
                        'on_complete_feedback': (
                            "The Foundry Core is a vast, hellish cathedral of molten metal and grinding machinery. At its center, the Infernal Architect rises — a towering monstrosity of brimite and fiend alloy, its eyes blazing with infernal intelligence. Remember Dren's advice: it pauses to recharge. That's your opening."
                        ),
                    },
                    {
                        'id': 'ch4_final_battle',
                        'objective': 'Destroy the Infernal Architect — the master war engine.',
                        'hint': (
                            "Use everything you've prepared. The suppression charge deals massive damage. When the Architect defends or waits (recharges), that's your moment to attack with full force. Defend against its double-attack combos."
                        ),
                        'done_any': [
                            {
                                'kind': 'event',
                                'type': 'npc_killed',
                                'npc_key': 'ch4_architect_npc',
                            },
                            {
                                'kind': 'state',
                                'type': 'boss_already_defeated',
                                'npc_key': 'ch4_architect_npc',
                            },
                        ],
                        'on_complete_feedback': (
                            'The Infernal Architect lets out a mechanical death scream that echoes through the Foundry. Its brimite chassis cracks and splits, revealing the demon core within — now exposed and vulnerable. Rip it out and end this forever.'
                        ),
                        'spawns_boss_key': 'ch4_architect_npc',
                    },
                ],
                'root_goal': 'ch4_root',
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

            # Track talked-to NPCs and bought-from merchants for quest conditions
            mq_tracking = env.curr_agents_state.setdefault("main_quest_tracking", {})
            agent_tracking = mq_tracking.setdefault(aid, {"talked_to_npcs": [], "bought_from_npcs": []})
            for e in aevents:
                etype = getattr(e, "type", None)
                edata = getattr(e, "data", {}) or {}
                if etype == "npc_talked_to":
                    npc_id = edata.get("npc_id")
                    if npc_id and npc_id not in agent_tracking["talked_to_npcs"]:
                        agent_tracking["talked_to_npcs"].append(npc_id)
                elif etype == "object_bought":
                    npc_id = edata.get("npc_id")
                    if npc_id and npc_id not in agent_tracking["bought_from_npcs"]:
                        agent_tracking["bought_from_npcs"].append(npc_id)

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

        self._generated['ch1_cipher_code'] = 'IRON-VEIL-7'

        # chapter_1: Ashes and Orders
        _picks_ch1_bastion_areas = pick_progressive(3)
        self._generated['ch1_bastion_areas'] = _picks_ch1_bastion_areas if _picks_ch1_bastion_areas else [spawn_area]
        self._generated['ch1_visit_count'] = 3
        _picks_ch1_marshal_area = pick_progressive(1)
        self._generated['ch1_marshal_area'] = _picks_ch1_marshal_area[0] if _picks_ch1_marshal_area else spawn_area
        # ch1_marshal_npc: NPC npc_old_marshal spawned in _ensure_static_quest_npcs
        # ch1_marshal_npc: NPC npc_old_marshal spawned in _ensure_static_quest_npcs

        # chapter_2: Scorched Reconnaissance
        _picks_ch2_lyra_area = pick_progressive(1)
        self._generated['ch2_lyra_area'] = _picks_ch2_lyra_area[0] if _picks_ch2_lyra_area else spawn_area
        # ch2_lyra_npc: NPC npc_quest_scout_lyra spawned in _ensure_static_quest_npcs
        _picks_ch2_scrap_area = pick_progressive(2)
        self._generated['ch2_scrap_area'] = _picks_ch2_scrap_area[0] if _picks_ch2_scrap_area else spawn_area
        # ch2_scrap_npc: NPC npc_quest_scrap_auto_1 spawned in _ensure_static_quest_npcs
        self._generated['ch2_banner_obj'] = 'obj_quest_war_banner'
        self._generated['ch2_rally_area'] = self._generated['ch2_scrap_area']
        _picks_ch2_merchant_area = pick_progressive(2)
        self._generated['ch2_merchant_area'] = _picks_ch2_merchant_area[0] if _picks_ch2_merchant_area else spawn_area
        # ch2_merchant_npc: NPC npc_kael_ashworth spawned in _ensure_static_quest_npcs
        _picks_ch2_warden_area = pick_progressive(2)
        self._generated['ch2_warden_area'] = _picks_ch2_warden_area[0] if _picks_ch2_warden_area else spawn_area
        # ch2_warden_npc: NPC npc_boss_scrap_warden spawned in _ensure_static_quest_npcs

        # chapter_3: The Iron Gulch
        _picks_ch3_entrance_area = pick_progressive(3)
        self._generated['ch3_entrance_area'] = _picks_ch3_entrance_area[0] if _picks_ch3_entrance_area else spawn_area
        # ch3_ravager1_npc: NPC npc_quest_iron_ravager_1 spawned in _ensure_static_quest_npcs
        # ch3_ember_npc: NPC npc_quest_ember_pack_1 spawned in _ensure_static_quest_npcs
        _picks_ch3_forge_area = pick_progressive(4)
        self._generated['ch3_forge_area'] = _picks_ch3_forge_area[0] if _picks_ch3_forge_area else spawn_area
        # ch3_dren_npc: NPC npc_quest_forgemaster_dren spawned in _ensure_static_quest_npcs
        _picks_ch3_scrapyard_area = pick_progressive(4)
        self._generated['ch3_scrapyard_area'] = _picks_ch3_scrapyard_area[0] if _picks_ch3_scrapyard_area else spawn_area
        # ch3_ravager2_npc: NPC npc_quest_iron_ravager_2 spawned in _ensure_static_quest_npcs

        # chapter_4: Heart of the Foundry
        _picks_ch4_gate_area = pick_progressive(5)
        self._generated['ch4_gate_area'] = _picks_ch4_gate_area[0] if _picks_ch4_gate_area else spawn_area
        # ch4_colossus_npc: NPC npc_quest_colossus_guard spawned in _ensure_static_quest_npcs
        _picks_ch4_furnace_area = pick_progressive(6)
        self._generated['ch4_furnace_area'] = _picks_ch4_furnace_area[0] if _picks_ch4_furnace_area else spawn_area
        # ch4_vessa_npc: NPC npc_vessa_ironhand spawned in _ensure_static_quest_npcs
        self._generated['ch4_furnace_area_sabotage'] = self._generated['ch4_furnace_area']
        _picks_ch4_core_area = pick_progressive(6)
        self._generated['ch4_core_area'] = _picks_ch4_core_area[0] if _picks_ch4_core_area else spawn_area
        self._generated['ch4_core_area_final'] = self._generated['ch4_core_area']
        self._generated['ch4_demon_core_obj'] = 'obj_quest_demon_core'
        # ch4_architect_npc: NPC npc_boss_infernal_architect spawned in _ensure_static_quest_npcs

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

        if 'ch1_marshal_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_old_marshal', self._generated['ch1_marshal_area'], rng)
            if inst:
                self._generated['ch1_marshal_npc'] = inst
                self._generated['ch1_marshal_name'] = world.npc_instances[inst].name

        if 'ch2_lyra_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_quest_scout_lyra', self._generated['ch2_lyra_area'], rng)
            if inst:
                self._generated['ch2_lyra_npc'] = inst
                self._generated['ch2_lyra_name'] = world.npc_instances[inst].name

        if 'ch2_scrap_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_quest_scrap_auto_1', self._generated['ch2_scrap_area'], rng)
            if inst:
                self._generated['ch2_scrap_npc'] = inst
                self._generated['ch2_scrap_name'] = world.npc_instances[inst].name

        if 'ch2_merchant_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_kael_ashworth', self._generated['ch2_merchant_area'], rng)
            if inst:
                self._generated['ch2_merchant_npc'] = inst
                self._generated['ch2_merchant_name'] = world.npc_instances[inst].name

        if 'ch2_warden_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_boss_scrap_warden', self._generated['ch2_warden_area'], rng)
            if inst:
                self._generated['ch2_warden_npc'] = inst
                self._generated['ch2_warden_name'] = world.npc_instances[inst].name

        if 'ch3_ravager1_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_quest_iron_ravager_1', self._generated['ch3_entrance_area'], rng)
            if inst:
                self._generated['ch3_ravager1_npc'] = inst
                self._generated['ch3_ravager1_name'] = world.npc_instances[inst].name

        if 'ch3_ember_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_quest_ember_pack_1', self._generated['ch3_entrance_area'], rng)
            if inst:
                self._generated['ch3_ember_npc'] = inst
                self._generated['ch3_ember_name'] = world.npc_instances[inst].name

        if 'ch3_dren_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_quest_forgemaster_dren', self._generated['ch3_forge_area'], rng)
            if inst:
                self._generated['ch3_dren_npc'] = inst
                self._generated['ch3_dren_name'] = world.npc_instances[inst].name

        if 'ch3_ravager2_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_quest_iron_ravager_2', self._generated['ch3_scrapyard_area'], rng)
            if inst:
                self._generated['ch3_ravager2_npc'] = inst
                self._generated['ch3_ravager2_name'] = world.npc_instances[inst].name

        if 'ch4_colossus_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_quest_colossus_guard', self._generated['ch4_gate_area'], rng)
            if inst:
                self._generated['ch4_colossus_npc'] = inst
                self._generated['ch4_colossus_name'] = world.npc_instances[inst].name

        if 'ch4_vessa_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_vessa_ironhand', self._generated['ch4_furnace_area'], rng)
            if inst:
                self._generated['ch4_vessa_npc'] = inst
                self._generated['ch4_vessa_name'] = world.npc_instances[inst].name

        if 'ch4_architect_npc' not in self._generated:
            inst = self._create_npc_instance(world, 'npc_boss_infernal_architect', self._generated['ch4_core_area'], rng)
            if inst:
                self._generated['ch4_architect_npc'] = inst
                self._generated['ch4_architect_name'] = world.npc_instances[inst].name

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

    def _create_npc_instance(self, world, base_npc_id: str, area_id, rng) -> Optional[str]:
        if base_npc_id not in world.npcs:
            return None
        # area_id may be a list from pick_progressive; use the first element
        if isinstance(area_id, list):
            area_id = area_id[0] if area_id else None
        if not area_id:
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

        # DAG chapters: use completed_stages to determine boss spawning
        chapters = self.QUEST_CONFIG.get("chapters", [])
        if ch >= len(chapters):
            return

        chapter = chapters[ch]
        ch_id = chapter["id"]
        completed = set(prog.get("completed_stages", {}).get(ch_id, []))
        active = set(prog.get("active_stages", {}).get(ch_id, []))

        # chapter_2 (ch==1): spawn warden when ch2_defeat_warden becomes active
        if ch >= 1:
            if ch > 1 or 'ch2_defeat_warden' in active or 'ch2_defeat_warden' in completed:
                self._ensure_named_boss_spawned(env, world, 'ch2_warden_npc')

        # chapter_4 (ch==3): spawn colossus when ch4_defeat_colossus becomes active
        if ch >= 3:
            if ch > 3 or 'ch4_defeat_colossus' in active or 'ch4_defeat_colossus' in completed:
                self._ensure_named_boss_spawned(env, world, 'ch4_colossus_npc')

        # chapter_4 (ch==3): spawn architect when ch4_final_battle becomes active
        if ch >= 3:
            if ch > 3 or 'ch4_final_battle' in active or 'ch4_final_battle' in completed:
                self._ensure_named_boss_spawned(env, world, 'ch4_architect_npc')

    def _ensure_named_boss_spawned(self, env, world, boss_key: str) -> None:
        mq = world.auxiliary["main_quest"]
        if mq["boss_defeated"].get(boss_key, False):
            return

        inst_id = self._generated.get(boss_key)
        if inst_id and inst_id in world.npc_instances:
            # Boss already exists — ensure it's tracked in boss_max_hp/boss_mods
            boss = world.npc_instances[inst_id]
            if inst_id not in mq["boss_max_hp"]:
                mq["boss_max_hp"][inst_id] = int(boss.hp)
            if inst_id not in mq["boss_mods"]:
                mq["boss_mods"][inst_id] = {
                    "base_max_hp": int(boss.hp),
                    "curr_max_hp": int(boss.hp),
                    "drenched": False,
                }
            return

        # Map boss_key to base NPC ID and area key
        boss_key_to_info: Dict[str, Tuple[str, str]] = {
            'ch2_warden_npc': ('npc_boss_scrap_warden', 'ch2_warden_area'),
            'ch4_colossus_npc': ('npc_quest_colossus_guard', 'ch4_gate_area'),
            'ch4_architect_npc': ('npc_boss_infernal_architect', 'ch4_core_area'),
        }

        info = boss_key_to_info.get(boss_key)
        if not info:
            return

        base_id, area_key = info
        area_id = self._generated.get(area_key)
        if not area_id:
            return

        rng = env.rng
        inst = self._create_npc_instance(world, base_id, area_id, rng)
        if not inst:
            return

        boss = world.npc_instances[inst]
        self._generated[boss_key] = inst
        mq["boss_max_hp"][inst] = int(boss.hp)
        mq["boss_mods"].setdefault(inst, {
            "base_max_hp": int(boss.hp),
            "curr_max_hp": int(boss.hp),
            "drenched": False,
        })
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
        # The robot_kingdom quest does not use fire/water elemental boss
        # modifiers. Boss fights use standard combat mechanics.
        pass

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
            'ch1_marshal_name': self._generated.get('ch1_marshal_name', 'ch1_marshal_name'),
            'ch2_lyra_name': self._generated.get('ch2_lyra_name', 'ch2_lyra_name'),
            'ch2_merchant_name': self._generated.get('ch2_merchant_name', 'ch2_merchant_name'),
            'ch3_dren_name': self._generated.get('ch3_dren_name', 'ch3_dren_name'),
            'ch4_vessa_name': self._generated.get('ch4_vessa_name', 'ch4_vessa_name'),

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

            if stype == "boss_already_defeated":
                boss_key = cond.get("boss_key") or cond.get("npc_key")
                return bool(world.auxiliary.get("main_quest", {}).get("boss_defeated", {}).get(boss_key, False))

            if stype == "has_category":
                category = cond.get("category", "")
                if not category:
                    return False
                # Check hands, equipped, inventory, and held containers
                all_obj_ids = []
                for oid in (agent.items_in_hands or {}):
                    all_obj_ids.append(oid)
                for oid in (agent.equipped_items_in_limb or {}):
                    all_obj_ids.append(oid)
                if agent.inventory and getattr(agent.inventory, "container", None):
                    for oid in (agent.inventory.items or {}):
                        all_obj_ids.append(oid)
                for oid in all_obj_ids:
                    base = get_def_id(oid)
                    if base in world.objects and world.objects[base].category == category:
                        return True
                return False

            if stype == "paper_text_equals_in_hands":
                text_key = cond.get("text_key")
                target = str(self._generated.get(text_key, "")).strip().lower()
                if not target:
                    return False
                # Check writable instances held in agent's hands
                for oid in (agent.items_in_hands or {}):
                    if oid in world.writable_instances:
                        writable = world.writable_instances[oid]
                        if str(getattr(writable, "text", "")).strip().lower() == target:
                            return True
                return False

            if stype == "has_any_of":
                base_objs = cond.get("base_objs", [])
                if not isinstance(base_objs, list) or not base_objs:
                    return False
                return any(self._count_base_item(world, agent, b) > 0 for b in base_objs)

        if kind == "custom":
            ctype = cond.get("type")

            if ctype == "talked_to_npc":
                npc_key = cond.get("npc_key")
                want_npc = self._generated.get(npc_key) if npc_key else None
                if not want_npc:
                    return False
                mq_tracking = env.curr_agents_state.get("main_quest_tracking", {})
                agent_tracking = mq_tracking.get(agent.id, {})
                talked_list = agent_tracking.get("talked_to_npcs", [])
                return want_npc in talked_list

            if ctype == "talked_to_npc_with_item":
                npc_key = cond.get("npc_key")
                text_key = cond.get("text_key")
                want_npc = self._generated.get(npc_key) if npc_key else None
                target_text = str(self._generated.get(text_key, "")).strip().lower() if text_key else ""
                if not want_npc or not target_text:
                    return False
                # Check talked to NPC
                mq_tracking = env.curr_agents_state.get("main_quest_tracking", {})
                agent_tracking = mq_tracking.get(agent.id, {})
                talked_list = agent_tracking.get("talked_to_npcs", [])
                if want_npc not in talked_list:
                    return False
                # Check holding writable with matching text (obj_quest_recon_report instances)
                for oid in (agent.items_in_hands or {}):
                    if oid in world.writable_instances:
                        writable = world.writable_instances[oid]
                        if str(getattr(writable, "text", "")).strip().lower() == target_text:
                            return True
                # Also check inventory
                if agent.inventory and getattr(agent.inventory, "container", None):
                    for oid in (agent.inventory.items or {}):
                        if oid in world.writable_instances:
                            writable = world.writable_instances[oid]
                            if str(getattr(writable, "text", "")).strip().lower() == target_text:
                                return True
                return False

            if ctype == "object_in_area":
                # Variant 1: single object via obj_key + area_key
                obj_key = cond.get("obj_key")
                # Variant 2: multiple objects via target_objs + area_key
                target_objs = cond.get("target_objs")
                area_key = cond.get("area_key")

                if not area_key:
                    return False

                area_id = self._generated.get(area_key)
                if not area_id:
                    return False
                # area_id may be a list (pick_progressive returns lists)
                if isinstance(area_id, list):
                    area_ids_to_check = area_id
                else:
                    area_ids_to_check = [area_id]

                if target_objs and isinstance(target_objs, list):
                    # Check if any of the target objects exist in any of the target areas
                    for aid in area_ids_to_check:
                        area = world.area_instances.get(aid)
                        if not area:
                            continue
                        for oid in area.objects:
                            base = get_def_id(oid)
                            if base in target_objs:
                                return True
                    return False

                if obj_key:
                    target_obj_base = self._generated.get(obj_key)
                    if not target_obj_base:
                        return False
                    for aid in area_ids_to_check:
                        area = world.area_instances.get(aid)
                        if not area:
                            continue
                        for oid in area.objects:
                            base = get_def_id(oid)
                            if base == target_obj_base:
                                return True
                    return False

                return False

            if ctype == "npc_has_coins":
                npc_key = cond.get("npc_key")
                threshold = int(cond.get("threshold", 0))
                want_npc = self._generated.get(npc_key) if npc_key else None
                if not want_npc:
                    return False
                npc = world.npc_instances.get(want_npc)
                if not npc:
                    return False
                return int(getattr(npc, "coins", 0) or 0) >= threshold

            if ctype == "area_cleared":
                area_key = cond.get("area_key")
                if not area_key:
                    return False
                area_id = self._generated.get(area_key)
                if not area_id:
                    return False
                if isinstance(area_id, list):
                    area_ids_to_check = area_id
                else:
                    area_ids_to_check = [area_id]
                for aid in area_ids_to_check:
                    area = world.area_instances.get(aid)
                    if not area:
                        continue
                    for npc_id in area.npcs:
                        npc = world.npc_instances.get(npc_id)
                        if npc and getattr(npc, "enemy", False):
                            return False
                return True

            if ctype == "talked_to_npc_in_area":
                npc_key = cond.get("npc_key")
                area_key = cond.get("area_key")
                want_npc = self._generated.get(npc_key) if npc_key else None
                target_area = self._generated.get(area_key) if area_key else None
                if not want_npc or not target_area:
                    return False
                # Resolve target area (may be list)
                if isinstance(target_area, list):
                    target_area_ids = set(target_area)
                else:
                    target_area_ids = {target_area}
                # Check player is in one of the target areas
                current_area_id = env.curr_agents_state["area"][agent.id]
                if current_area_id not in target_area_ids:
                    return False
                # Check talked to NPC
                mq_tracking = env.curr_agents_state.get("main_quest_tracking", {})
                agent_tracking = mq_tracking.get(agent.id, {})
                talked_list = agent_tracking.get("talked_to_npcs", [])
                return want_npc in talked_list

            if ctype == "player_has_crafted_specific":
                obj_id = cond.get("obj_id")
                if not obj_id:
                    return False
                return self._count_base_item(world, agent, obj_id) > 0

            if ctype == "equipped_armor_level":
                min_level = int(cond.get("min_level", 1))
                equipped = getattr(agent, "equipped_items_in_limb", {}) or {}
                for oid in equipped:
                    base = get_def_id(oid)
                    if base in world.objects:
                        obj_def = world.objects[base]
                        if obj_def.category == "armor" and int(getattr(obj_def, "level", 0) or 0) >= min_level:
                            return True
                return False

            if ctype == "bought_from_merchant":
                npc_key = cond.get("npc_key")
                want_npc = self._generated.get(npc_key) if npc_key else None
                if not want_npc:
                    return False
                mq_tracking = env.curr_agents_state.get("main_quest_tracking", {})
                agent_tracking = mq_tracking.get(agent.id, {})
                bought_list = agent_tracking.get("bought_from_npcs", [])
                return want_npc in bought_list

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


