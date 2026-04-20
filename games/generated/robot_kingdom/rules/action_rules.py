import math
from games.generated.robot_kingdom.rule import BaseActionRule, RuleContext, RuleResult, Event
from collections import defaultdict
from utils import *


class WaitRule(BaseActionRule):
    name = "action_wait"
    verb = "wait"
    param_min = 0
    param_max = 0
    params = []
    description = "The agent waits for a while without taking any action. During combat, waiting fully recovers stamina."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, agent = ctx.env, ctx.agent
        
        env.curr_agents_state["agent_consecutive_attacks"][agent.id] = 0
        
        # check if agent is in active combat
        active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
        if active_combats:
            # fully recover stamina when waiting in combat
            old_stamina = env.curr_agents_state["agent_stamina"].get(agent.id, 1.0)
            env.curr_agents_state["agent_stamina"][agent.id] = 1.0
            
            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} waited for a while. "
                f"{env.person_verbalized['possessive_adjective'].capitalize()} stamina is now at 100%.\n"
            )
        else:
            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} waited for a while.\n"
            )


class DefendRule(BaseActionRule):
    name = "action_defend"
    verb = "defend"
    param_min = 0
    param_max = 0
    params = []
    description = "The agent takes a defensive stance. If attacked by an NPC while defending, damage received is reduced to 10%. Defending also partially recovers stamina."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, agent = ctx.env, ctx.agent
        
        env.curr_agents_state["agent_defending"][agent.id] = True
        
        env.curr_agents_state["agent_consecutive_attacks"][agent.id] = 0
        
        # recover stamina partially (50% of missing stamina)
        active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
        stamina_msg = ""
        if active_combats:
            old_stamina = env.curr_agents_state["agent_stamina"].get(agent.id, 1.0)
            missing = 1.0 - old_stamina
            new_stamina = min(1.0, old_stamina + missing * 0.5)
            env.curr_agents_state["agent_stamina"][agent.id] = new_stamina
            stamina_pct = int(new_stamina * 100)
            stamina_msg = f" {env.person_verbalized['possessive_adjective'].capitalize()} stamina is now at {stamina_pct}%."
        
        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} am defending.{stamina_msg}\n"
        )
        
        res.events.append(Event(
            type="agent_defending",
            agent_id=agent.id,
            data={"area_id": env.curr_agents_state["area"][agent.id]},
        ))


class PickUpRule(BaseActionRule):
    name = "action_pick_up"
    verb = "pick up"
    param_min = param_max = 1
    params = ["object_name"]
    description = "The agent picks up an object from the current area."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        obj_name = ctx.params[0]

        if obj_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot pick up {obj_name}, not found in current area.\n")
            return

        obj_id = world.auxiliary["obj_name_to_id"][obj_name]
        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]

        if obj_id in current_area.objects and current_area.objects[obj_id] > 0:
            # 2 hands max
            if sum(agent.items_in_hands.values()) <= 1:
                obj_count = min(1, current_area.objects[obj_id])
                current_area.objects[obj_id] -= obj_count
                if current_area.objects[obj_id] == 0:
                    del current_area.objects[obj_id]

                agent.items_in_hands[obj_id] = agent.items_in_hands.get(obj_id, 0) + obj_count

                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} picked up {obj_name}.\n"
                )
                res.track_move(
                    agent.id, obj_id, obj_count,
                    src=res.tloc("area", current_area.id),
                    dst=res.tloc("hand", agent.id),
                )
                res.events.append(Event(
                    type="object_picked_up",
                    agent_id=agent.id,
                    data={"obj_id": obj_id, "area_id": current_area.id},
                ))
            else:
                res.add_feedback(
                    agent.id,
                    f"Cannot pick up {obj_name}, not enough space in hand.\n"
                )
        else:
            res.add_feedback(
                agent.id,
                f"Cannot pick up {obj_name}, not found in current area.\n"
            )

class DropRule(BaseActionRule):
    name = "action_drop"
    verb = "drop"
    param_min = param_max = 1
    params = ["object_name"]
    description = "The agent drops an object from hand to the current area."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        obj_name = ctx.params[0]

        if obj_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot drop {obj_name}, not found in hand.\n")
            return

        obj_id = world.auxiliary["obj_name_to_id"][obj_name]

        if obj_id in agent.items_in_hands and agent.items_in_hands[obj_id] > 0:
            current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]
            obj_count = min(1, agent.items_in_hands[obj_id])
            agent.items_in_hands[obj_id] -= obj_count
            if agent.items_in_hands[obj_id] == 0:
                del agent.items_in_hands[obj_id]
            current_area.objects[obj_id] = current_area.objects.get(obj_id, 0) + obj_count

            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} dropped {obj_name}.\n"
            )
            res.track_move(
                agent.id, obj_id, obj_count,
                src=res.tloc("hand", agent.id),
                dst=res.tloc("area", current_area.id),
            )
            res.events.append(Event(
                type="object_dropped",
                agent_id=agent.id,
                data={"obj_id": obj_id, "area_id": current_area.id},
            ))
        else:
            res.add_feedback(agent.id, f"Cannot drop {obj_name}, not found in hand.\n")

class EnterRule(BaseActionRule):
    name = "action_enter"
    verb = "enter"
    param_min = param_max = 1
    params = ["area_name"]
    description = "The agent enters a designated neighboring area; if the path is locked, a key is required to unlock it. Cannot leave during active combat."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        area_name = ctx.params[0]

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        # check if agent is in active combat - cannot leave during combat
        active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
        if active_combats:
            blocking_npcs = []
            for npc_id, combat_state in active_combats.items():
                if combat_state.get("area_id") == current_area_id:
                    if npc_id in world.npc_instances:
                        blocking_npcs.append(world.npc_instances[npc_id].name)

            if blocking_npcs:
                npc_names = ", ".join(blocking_npcs)
                res.add_feedback(
                    agent.id,
                    f"Cannot enter {area_name}. {npc_names} has blocked the way out.\n"
                )
                return

        neighbors = current_area.neighbors

        connecting_path = None
        status = "success"

        for neighbor_id, path in neighbors.items():
            if world.area_instances[neighbor_id].name == area_name:
                connecting_path = path
                if path.locked:
                    if "obj_key" in agent.items_in_hands:
                        path.locked = False
                        # unlock reverse path too
                        world.area_instances[neighbor_id].neighbors[current_area_id].locked = False

                        res.add_feedback(
                            agent.id,
                            f"{env.person_verbalized['subject_pronoun']} used a key to unlock "
                            f"the path to {area_name}. {env.person_verbalized['subject_pronoun']} "
                            f"entered {area_name}.\n"
                        )
                        env.curr_agents_state["area"][agent.id] = neighbor_id
                        agent.items_in_hands["obj_key"] -= 1
                        if agent.items_in_hands["obj_key"] == 0:
                            del agent.items_in_hands["obj_key"]
                        res.track_consume(agent.id, "obj_key", 1, src=res.tloc("hand", agent.id))
                    else:
                        res.add_feedback(agent.id, f"Cannot enter {area_name}, the door is locked.\n")
                        status = "locked"
                else:
                    res.add_feedback(
                        agent.id,
                        f"{env.person_verbalized['subject_pronoun']} entered {area_name}.\n"
                    )
                    env.curr_agents_state["area"][agent.id] = neighbor_id

                break

        if connecting_path is None and status != "locked":
            res.add_feedback(agent.id, f"Cannot enter {area_name}. There's no path to {area_name}.\n")
            return

        if connecting_path is not None and status == "success":
            res.events.append(Event(
                type="area_entered",
                agent_id=agent.id,
                data={
                    "from_area": current_area_id,
                    "to_area": env.curr_agents_state["area"][agent.id],
                    "area_name": area_name,
                },
            ))

class AttackRule(BaseActionRule):
    name = "action_attack"
    verb = "attack"
    param_min = param_max = 1
    params = ["npc_name"]
    description = "The agent attacks a designated NPC in the current area; deals single-round damage based on agent's attack power. If NPC is in 'defend' state, damage is reduced to 25%."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        target_npc_name = ctx.params[0]

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        if target_npc_name not in world.auxiliary["npc_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot attack {target_npc_name}, not found in current area.\n")
            return

        target_npc_id = world.auxiliary["npc_name_to_id"][target_npc_name]
        if target_npc_id not in current_area.npcs:
            res.add_feedback(agent.id, f"Cannot attack {target_npc_name}, not found in current area.\n")
            return

        # Prevent attacking quest-critical non-enemy NPCs (guides, quest givers, etc.)
        TASK_NPC_BASE_ID = "npc_quest_wayfarer_guide"
        target_npc_instance = world.npc_instances[target_npc_id]
        if get_def_id(target_npc_id) == TASK_NPC_BASE_ID or (
            not getattr(target_npc_instance, "enemy", True)
            and getattr(target_npc_instance, "quest", False)
        ):
            res.add_feedback(agent.id, f"You cannot attack {target_npc_name}. They are here to guide you on your quest.\n")
            return
        agent_attack_power = agent.attack
        
        if agent_attack_power <= 0:
            res.add_feedback(agent.id, f"Cannot attack {target_npc_name}, {env.person_verbalized['subject_pronoun'].lower()} have no attack power.\n")
            return

        active_combats = env.curr_agents_state["active_combats"].get(agent.id, {})
        is_new_combat = target_npc_id not in active_combats
        
        npc_current_action = None
        if not is_new_combat:
            combat_state = active_combats[target_npc_id]
            rhythm = target_npc_instance.combat_pattern
            if rhythm:
                rhythm_index = combat_state.get("rhythm_index", 0)
                npc_current_action = rhythm[rhythm_index % len(rhythm)]
        
        stamina = env.curr_agents_state["agent_stamina"].get(agent.id, 1.0)
        consecutive_attacks = env.curr_agents_state["agent_consecutive_attacks"].get(agent.id, 0)
        
        base_damage = agent_attack_power
        damage_to_npc = int(base_damage * stamina)
        
        if npc_current_action == "defend":
            damage_to_npc = int(damage_to_npc * 0.1)
        
        consecutive_attacks += 1
        env.curr_agents_state["agent_consecutive_attacks"][agent.id] = consecutive_attacks
        
        # stamina decays by 20% per attack after the 1st consecutive attack
        stamina_decreased = False
        if consecutive_attacks > 1:
            stamina = max(0.2, stamina - 0.2)  # minimum stamina is 20%
            env.curr_agents_state["agent_stamina"][agent.id] = stamina
            stamina_decreased = True
        
        # --- rallied ally bonus damage ---
        rallied = env.curr_agents_state.get("rallied_allies", {}).get(agent.id)
        ally_bonus = 0
        if rallied and rallied.get("rounds_left", 0) > 0:
            ally_npc_id = rallied.get("npc_id")
            ally_area = rallied.get("area_id")
            # ally must still be in the same area and alive
            ally_valid = (
                ally_area == current_area_id
                and ally_npc_id in world.npc_instances
                and ally_npc_id in current_area.npcs
            )
            if ally_valid:
                ally_bonus = rallied.get("bonus_damage", 0)
                damage_to_npc += ally_bonus
                rallied["rounds_left"] -= 1
                ally_name = rallied.get("npc_name", "your ally")
                res.add_feedback(
                    agent.id,
                    f"{ally_name} strikes alongside {env.person_verbalized['object_pronoun']}, "
                    f"dealing {ally_bonus} bonus damage!\n"
                )
                if rallied["rounds_left"] <= 0:
                    res.add_feedback(
                        agent.id,
                        f"{ally_name}'s rallied fervor fades. They return to their post.\n"
                    )
                    env.curr_agents_state["rallied_allies"][agent.id] = {}
            else:
                # ally no longer valid (left area or dead)
                env.curr_agents_state["rallied_allies"][agent.id] = {}

        target_npc_instance.hp -= damage_to_npc
        
        if target_npc_instance.hp <= 0:
            target_npc_instance.hp = 0
            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} attacked {target_npc_name} for {damage_to_npc} damage and defeated {target_npc_name}!\n"
            )
            
            # show stamina feedback after attack feedback
            if stamina_decreased:
                stamina_pct = int(stamina * 100)
                res.add_feedback(agent.id, f"{env.person_verbalized['subject_pronoun']} feel tired. {env.person_verbalized['possessive_adjective'].capitalize()} stamina is now at {stamina_pct}%.\n")

            if target_npc_id in env.curr_agents_state["active_combats"].get(agent.id, {}):
                del env.curr_agents_state["active_combats"][agent.id][target_npc_id]
            
            # reset stamina when no more active combats
            if not env.curr_agents_state["active_combats"].get(agent.id, {}):
                env.curr_agents_state["agent_stamina"][agent.id] = 1.0
                env.curr_agents_state["agent_consecutive_attacks"][agent.id] = 0

            tutorial_room = world.auxiliary.get("tutorial_room") or {}
            tutorial_removed = bool(tutorial_room.get("removed", False))
            if not (tutorial_room and not tutorial_removed):
                env.curr_agents_state["npcs_killed"][agent.id].append(target_npc_id)
                target_npc_str = f"{get_def_id(target_npc_id)}_{target_npc_instance.level}"
                if not target_npc_str in env.curr_agents_state["unique_npcs_killed"][agent.id]:
                    env.curr_agents_state["unique_npcs_killed"][agent.id].append(target_npc_str)

            current_area.npcs.remove(target_npc_id)
            # drop loot
            for obj_id, count in target_npc_instance.inventory.items():
                if count > 0:
                    obj_def = world.objects[obj_id]
                    if obj_def.category == "container" or obj_def.usage == "writable":
                        # create a new instance for each new container/writable dropped
                        instance_list = world.container_instances if obj_def.category == "container" else world.writable_instances
                        id_to_count = (
                            world.auxiliary["container_id_to_count"]
                            if obj_def.category == "container" 
                            else world.auxiliary["writable_id_to_count"]
                        )
                        for _ in range(count):
                            new_instance = obj_def.create_instance(id_to_count[obj_id])
                            instance_list[new_instance.id] = new_instance
                            id_to_count[obj_id] += 1
                            world.auxiliary["obj_name_to_id"][new_instance.name] = new_instance.id
                            current_area.objects[new_instance.id] = current_area.objects.get(new_instance.id, 0) + 1
                            res.track_spawn(agent.id, new_instance.id, 1, dst=res.tloc("area", current_area.id))
                    else:
                        current_area.objects[obj_id] = current_area.objects.get(obj_id, 0) + count
                        res.track_spawn(agent.id, obj_id, count, dst=res.tloc("area", current_area.id))

            res.events.append(Event(
                type="npc_killed",
                agent_id=agent.id,
                data={"npc_id": target_npc_id, "area_id": current_area.id},
            ))
        else:
            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} attacked {target_npc_name} for {damage_to_npc} damage. "
                f"{target_npc_name} has {target_npc_instance.hp} HP remaining.\n"
            )
            
            # show stamina feedback after attack feedback
            if stamina_decreased:
                stamina_pct = int(stamina * 100)
                res.add_feedback(agent.id, f"{env.person_verbalized['subject_pronoun']} feel tired. {env.person_verbalized['possessive_adjective'].capitalize()} stamina is now at {stamina_pct}%.\n")
            
            # if this is an enemy NPC with an attack rhythm, initiate or continue combat
            if target_npc_instance.enemy and target_npc_instance.combat_pattern:
                if is_new_combat:
                    # start new combat - NPC will respond based on first rhythm action
                    env.curr_agents_state["active_combats"].setdefault(agent.id, {})
                    env.curr_agents_state["active_combats"][agent.id][target_npc_id] = {
                        "rhythm_index": 0,
                        "area_id": current_area_id,
                    }
                    rhythm_action = target_npc_instance.combat_pattern[0]
                    res.add_feedback(
                        agent.id,
                        f"{target_npc_name} has engaged in combat.\n"
                    )
                    res.events.append(Event(
                        type="combat_started",
                        agent_id=agent.id,
                        data={"npc_id": target_npc_id, "area_id": current_area_id},
                    ))

            res.events.append(Event(
                type="npc_attacked",
                agent_id=agent.id,
                data={"npc_id": target_npc_id, "damage": damage_to_npc, "area_id": current_area.id},
            ))

class StoreRule(BaseActionRule):
    name = "action_store"
    verb = "store"
    param_min = param_max = 3
    params = ["amount", "obj_name", "container_name"]
    description = "The agent stores a specified amount of an object into a designated container or inventory."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        amount_str, obj_name, container_name = ctx.params[0], ctx.params[1], ctx.params[2]

        if obj_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot store {obj_name}, not found in current area.\n")
            return
        obj_id = world.auxiliary["obj_name_to_id"][obj_name]
        # cannot store containers inside containers
        if obj_id in world.container_instances:
            res.add_feedback(
                agent.id,
                f"Cannot store {obj_name}. A container can't be stored in another container.\n"
            )
            return
        if obj_id in world.objects and world.objects[obj_id].size is None:
            res.add_feedback(
                agent.id,
                f"Cannot store {obj_name}. Objects of category {world.objects[obj_id].category} are not storable.\n"
            )
            return

        if not amount_str.isnumeric() or int(amount_str) <= 0:
            res.add_feedback(
                agent.id,
                f"Invalid amount '{amount_str}' for storing {obj_name}. "
                "Amount must be a positive integer.\n"
            )
            return

        amount = int(amount_str)
        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]

        # ----- case 1: store into inventory -----
        if container_name == "inventory":
            if agent.inventory.container is None:
                res.add_feedback(
                    agent.id,
                    "Cannot store {0}, no inventory found. A container needs to be equipped as inventory first.\n"
                    .format(obj_name)
                )
                return

            total_count = current_area.objects.get(obj_id, 0) + agent.items_in_hands.get(obj_id, 0)
            if (obj_id not in current_area.objects and obj_id not in agent.items_in_hands) or total_count == 0:
                res.add_feedback(
                    agent.id,
                    f"Cannot store {obj_name}, not found in current area or in hand.\n"
                )
                return

            object_size = world.writable_instances[obj_id].size if obj_id in world.writable_instances else world.objects[obj_id].size
            inventory_size = agent.inventory.container.current_load(world.objects)
            free_capacity = agent.inventory.capacity - inventory_size

            if free_capacity < object_size * amount:
                # reduce amount to max storable
                amount = free_capacity // object_size
            if amount <= 0:
                res.add_feedback(
                    agent.id,
                    f"Cannot store any number of {obj_name}, not enough space in inventory.\n"
                )
                return
            if amount > total_count:
                # not enough pieces to satisfy requested amount
                res.add_feedback(
                    agent.id,
                    f"Only found {total_count} {obj_name} to store in inventory.\n"
                )
                amount = total_count
            elif amount < int(amount_str):
                # capacity limited how many we could store
                res.add_feedback(
                    agent.id,
                    f"Only enough space to store {amount} {obj_name} in inventory.\n"
                )

            # first move from hands, then from ground
            obj_count = min(amount, agent.items_in_hands.get(obj_id, 0))
            if obj_count > 0:
                agent.inventory.items[obj_id] = agent.inventory.items.get(obj_id, 0) + obj_count
                agent.items_in_hands[obj_id] -= obj_count
                if agent.items_in_hands[obj_id] == 0:
                    del agent.items_in_hands[obj_id]
                res.track_move(
                    agent.id, obj_id, obj_count,
                    src=res.tloc("hand", agent.id),
                    dst=res.tloc("container", agent.inventory.container.id),
                )

            obj_count = min(amount - obj_count, current_area.objects.get(obj_id, 0))
            if obj_count > 0:
                agent.inventory.items[obj_id] = agent.inventory.items.get(obj_id, 0) + obj_count
                current_area.objects[obj_id] -= obj_count
                if current_area.objects[obj_id] == 0:
                    del current_area.objects[obj_id]
                res.track_move(
                    agent.id, obj_id, obj_count,
                    src=res.tloc("area", current_area.id),
                    dst=res.tloc("container", agent.inventory.container.id),
                )

            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} stored {amount} {obj_name} in "
                f"{env.person_verbalized['possessive_adjective']} inventory.\n"
            )
            res.events.append(Event(
                type="object_stored",
                agent_id=agent.id,
                data={"obj_id": obj_id, "container": "inventory", "amount": amount,
                      "area_id": current_area.id},
            ))
            return

        # ----- case 2: store into some other container -----
        if container_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot find {container_name} to store {obj_name}.\n")
            return
        container_id = world.auxiliary["obj_name_to_id"][container_name]
        if container_id not in world.container_instances:
            res.add_feedback(agent.id, f"Cannot find {container_name} to store {obj_name}.\n")
            return
        container_instance = world.container_instances[container_id]

        if (
            container_id not in current_area.objects
            and (agent.inventory.container is None or container_id != agent.inventory.container.id)
            and container_id not in agent.items_in_hands
        ):
            res.add_feedback(
                agent.id,
                f"Cannot find {container_name} in the current area, in hand or equipped to store {obj_name}.\n"
            )
            return

        total_count = current_area.objects.get(obj_id, 0) + agent.items_in_hands.get(obj_id, 0)
        if (obj_id not in current_area.objects and obj_id not in agent.items_in_hands) or total_count == 0:
            res.add_feedback(
                agent.id,
                f"Cannot store {obj_name}, not found in the current area or in hand.\n"
            )
            return

        object_size = world.writable_instances[obj_id].size if obj_id in world.writable_instances else world.objects[obj_id].size
        container_size = sum(
            world.writable_instances[oid].size * count if oid in world.writable_instances else world.objects[oid].size * count
            for oid, count in container_instance.inventory.items()
        )
        free_capacity = container_instance.capacity - container_size

        if free_capacity < object_size * amount:
            amount = free_capacity // object_size
        if amount == 0:
            res.add_feedback(
                agent.id,
                f"Cannot store any number of {obj_name}, not enough space in {container_name}.\n"
            )
            return
        if amount > total_count:
            res.add_feedback(
                agent.id,
                f"Only found {total_count} {obj_name} to store in {container_name}.\n"
            )
            amount = total_count
        elif amount < int(amount_str):
            res.add_feedback(
                agent.id,
                f"Only enough space to store {amount} {obj_name} in {container_name}.\n"
            )

        # first move from hands, then from ground
        obj_count = min(amount, agent.items_in_hands.get(obj_id, 0))
        if obj_count > 0:
            container_instance.inventory[obj_id] = container_instance.inventory.get(obj_id, 0) + obj_count
            agent.items_in_hands[obj_id] -= obj_count
            if agent.items_in_hands[obj_id] == 0:
                del agent.items_in_hands[obj_id]
            res.track_move(
                agent.id, obj_id, obj_count,
                src=res.tloc("hand", agent.id),
                dst=res.tloc("container", container_id),
            )

        obj_count = min(amount - obj_count, current_area.objects.get(obj_id, 0))
        if obj_count > 0:
            container_instance.inventory[obj_id] = container_instance.inventory.get(obj_id, 0) + obj_count
            current_area.objects[obj_id] -= obj_count
            if current_area.objects[obj_id] == 0:
                del current_area.objects[obj_id]
            res.track_move(
                agent.id, obj_id, obj_count,
                src=res.tloc("area", current_area.id),
                dst=res.tloc("container", container_id),
            )

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} stored {amount} {obj_name} in {container_name}.\n"
        )
        res.events.append(Event(
            type="object_stored",
            agent_id=agent.id,
            data={"obj_id": obj_id, "container": container_name, "amount": amount,
                  "area_id": current_area.id},
        ))

class DiscardRule(BaseActionRule):
    name = "action_discard"
    verb = "discard"
    param_min = param_max = 3
    params = ["amount", "obj_name", "container_name"]
    description = "The agent discards a specified amount of an object from a designated container or inventory."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        amount_str, obj_name, container_name = ctx.params[0], ctx.params[1], ctx.params[2]

        if not amount_str.isdigit() or int(amount_str) <= 0:
            res.add_feedback(
                agent.id,
                f"Invalid amount '{amount_str}' for discarding {obj_name}. "
                "Amount must be a positive integer.\n"
            )
            return
        amount = int(amount_str)

        if obj_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(
                agent.id,
                f"Cannot discard {obj_name}, not found in {container_name}.\n"
            )
            return

        obj_id = world.auxiliary["obj_name_to_id"][obj_name]
        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]

        # ----- case 1: discard from inventory -----
        if container_name == "inventory":
            if agent.inventory.container is None:
                res.add_feedback(
                    agent.id,
                    f"Cannot discard {obj_name}, no inventory found. "
                    "A container needs to be equipped as inventory first.\n"
                )
                return

            if obj_id in agent.inventory.items and agent.inventory.items[obj_id] > 0:
                available = agent.inventory.items[obj_id]
                if amount > available:
                    obj_count = available
                    res.add_feedback(
                        agent.id,
                        f"Only found {available} {obj_name} to discard from inventory.\n"
                    )
                else:
                    obj_count = amount

                agent.inventory.items[obj_id] -= obj_count
                if agent.inventory.items[obj_id] == 0:
                    del agent.inventory.items[obj_id]
                current_area.objects[obj_id] = current_area.objects.get(obj_id, 0) + obj_count
                res.track_move(
                    agent.id, obj_id, obj_count,
                    src=res.tloc("container", agent.inventory.container.id),
                    dst=res.tloc("area", current_area.id),
                )

                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} discarded {obj_count} {obj_name} "
                    f"from {env.person_verbalized['possessive_adjective']} inventory.\n"
                )
                res.events.append(Event(
                    type="object_discarded",
                    agent_id=agent.id,
                    data={"obj_id": obj_id, "from": "inventory",
                          "amount": obj_count, "area_id": current_area.id},
                ))
            else:
                res.add_feedback(agent.id, f"Cannot discard {obj_name}, not found in inventory.\n")

            return

        # ----- case 2: discard from some other container -----
        if container_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot find {container_name} to discard {obj_name}.\n")
            return
        container_id = world.auxiliary["obj_name_to_id"][container_name]
        if container_id not in world.container_instances:
            res.add_feedback(agent.id, f"Cannot find {container_name} to discard {obj_name}.\n")
            return
        container_instance = world.container_instances[container_id]

        if (
            container_id not in agent.items_in_hands
            and (agent.inventory.container is None or container_id != agent.inventory.container.id)
            and container_id not in current_area.objects
        ):
            res.add_feedback(
                agent.id,
                f"Cannot find {container_name} equipped, held in hands, or in the current area "
                f"to discard {obj_name}.\n"
            )
            return

        if obj_id in container_instance.inventory and container_instance.inventory[obj_id] > 0:
            available = container_instance.inventory[obj_id]
            if amount > available:
                obj_count = available
                res.add_feedback(
                    agent.id,
                    f"Only found {available} {obj_name} to discard from {container_name}.\n"
                )
            else:
                obj_count = amount

            container_instance.inventory[obj_id] -= obj_count
            if container_instance.inventory[obj_id] == 0:
                del container_instance.inventory[obj_id]
            current_area.objects[obj_id] = current_area.objects.get(obj_id, 0) + obj_count
            res.track_move(
                agent.id, obj_id, obj_count,
                src=res.tloc("container", container_id),
                dst=res.tloc("area", current_area.id),
            )

            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} discarded {obj_count} {obj_name} "
                f"from {container_name}.\n"
            )
            res.events.append(Event(
                type="object_discarded",
                agent_id=agent.id,
                data={"obj_id": obj_id, "from": container_name,
                      "amount": obj_count, "area_id": current_area.id},
            ))
        else:
            res.add_feedback(
                agent.id,
                f"Cannot discard {obj_name}, not found in {container_name}.\n"
            )

class TakeOutRule(BaseActionRule):
    name = "action_take_out"
    verb = "take out"
    param_min = param_max = 2
    params = ["obj_name", "container_name"]
    description = "The agent takes out an object from a designated container or inventory into hand."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        obj_name, container_name = ctx.params[0], ctx.params[1]

        if obj_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(
                agent.id,
                f"Cannot take out {obj_name}, no such object found in inventory or {container_name}.\n"
            )
            return
        obj_id = world.auxiliary["obj_name_to_id"][obj_name]

        # check hand capacity (2 hands max)
        if sum(agent.items_in_hands.values()) > 1:
            res.add_feedback(agent.id, f"Cannot take out {obj_name}, not enough space in hand.\n")
            return

        # ----- case 1: take out from inventory -----
        if container_name == "inventory":
            if agent.inventory.container is None:
                res.add_feedback(
                    agent.id,
                    f"Cannot take out {obj_name}, no inventory found. "
                    "A container needs to be equipped as inventory first.\n"
                )
                return

            if obj_id in agent.inventory.items and agent.inventory.items[obj_id] > 0:
                obj_count = min(1, agent.inventory.items[obj_id])
                agent.items_in_hands[obj_id] = agent.items_in_hands.get(obj_id, 0) + obj_count
                agent.inventory.items[obj_id] -= obj_count
                if agent.inventory.items[obj_id] == 0:
                    del agent.inventory.items[obj_id]
                res.track_move(
                    agent.id, obj_id, obj_count,
                    src=res.tloc("container", agent.inventory.container.id),
                    dst=res.tloc("hand", agent.id),
                )

                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} took out {obj_name} from "
                    f"{env.person_verbalized['possessive_adjective']} inventory.\n"
                )
                current_area_id = env.curr_agents_state["area"][agent.id]
                res.events.append(Event(
                    type="object_taken_out",
                    agent_id=agent.id,
                    data={"obj_id": obj_id, "from": "inventory", "amount": obj_count,
                          "area_id": current_area_id},
                ))
            else:
                res.add_feedback(agent.id, f"Cannot take out {obj_name}, not found in inventory.\n")
            return

        # ----- case 2: take out from some other container -----
        if container_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot find {container_name} to take out {obj_name}.\n")
            return
        container_id = world.auxiliary["obj_name_to_id"][container_name]
        if container_id not in world.container_instances:
            res.add_feedback(agent.id, f"Cannot find {container_name} to take out {obj_name}.\n")
            return
        container_instance = world.container_instances[container_id]
        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]

        if (
            container_id not in current_area.objects
            and (agent.inventory.container is None or container_id != agent.inventory.container.id)
            and container_id not in agent.items_in_hands
        ):
            res.add_feedback(
                agent.id,
                f"Cannot find {container_name} equipped, held in hands, or in the current area "
                f"to take out {obj_name}.\n"
            )
            return

        if obj_id in container_instance.inventory and container_instance.inventory[obj_id] > 0:
            obj_count = min(1, container_instance.inventory[obj_id])
            agent.items_in_hands[obj_id] = agent.items_in_hands.get(obj_id, 0) + obj_count
            container_instance.inventory[obj_id] -= obj_count
            if container_instance.inventory[obj_id] == 0:
                del container_instance.inventory[obj_id]
            res.track_move(
                agent.id, obj_id, obj_count,
                src=res.tloc("container", container_id),
                dst=res.tloc("hand", agent.id),
            )

            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} took out {obj_name} from "
                f"{container_name}.\n"
            )
            res.events.append(Event(
                type="object_taken_out",
                agent_id=agent.id,
                data={"obj_id": obj_id, "from": container_name, "amount": obj_count,
                      "area_id": current_area.id},
            ))
        else:
            res.add_feedback(agent.id, f"Cannot take out {obj_name}, not found in {container_name}.\n")

class TalkToRule(BaseActionRule):
    name = "action_talk_to"
    verb = "talk to"
    param_min = param_max = 1
    params = ["npc_name"]
    description = "The agent talks to a designated NPC in the current area to receive dialogue."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        target_npc_name = ctx.params[0]

        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]

        if target_npc_name not in world.auxiliary["npc_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot talk to {target_npc_name}, not found in current area.\n")
            return

        target_npc_id = world.auxiliary["npc_name_to_id"][target_npc_name]
        if target_npc_id not in current_area.npcs:
            res.add_feedback(agent.id, f"Cannot talk to {target_npc_name}, not found in current area.\n")
            return

        target_npc_instance = world.npc_instances[target_npc_id]
        description = target_npc_instance.description
        dialogue = target_npc_instance.dialogue + "\n"

        feedback_str = f"{env.person_verbalized['subject_pronoun']} talked to {target_npc_name}.\n"
        if description:
            feedback_str += f"{target_npc_instance.name} says: I'm {description} {dialogue}"
        else:
            feedback_str += f"{target_npc_instance.name} says: {dialogue}"

        if target_npc_instance.role == "merchant":
            lines = f"{target_npc_instance.name} offers the following items for sale: "
            for item_id, count in target_npc_instance.inventory.items():
                if item_id in world.objects:
                    item_name = world.objects[item_id].name
                    value_each = world.objects[item_id].value
                else:
                    item_name = "Unknown"
                    value_each = 0
                lines += f"{count} {item_name} (each for {value_each} coins), "
            lines = lines.rstrip(", ") + "\n"
            feedback_str += lines

        res.add_feedback(agent.id, feedback_str)
            
        res.events.append(Event(
            type="npc_talked_to",
            agent_id=agent.id,
            data={"npc_id": target_npc_id, "area_id": current_area.id},
        ))

class InspectRule(BaseActionRule):
    name = "action_inspect"
    verb = "inspect"
    param_min = param_max = 1
    params = ["object_name"]
    description = "The agent inspects an object to obtain its text content, or inspects inventory to see its contents."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        target = ctx.params[0]

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        # ----- case 1: inspect inventory -----
        if target == "inventory":
            if agent.inventory.container is None:
                res.add_feedback(
                    agent.id,
                    "Cannot inspect inventory, no inventory found. "
                    "A container needs to be equipped as inventory first.\n"
                )
                return

            inv_text = env.verbalize_objects(agent.inventory.items)
            inv_capacity = agent.inventory.container.current_load(world.objects)
            feedback_str = f"{env.person_verbalized['subject_pronoun']} inspected " \
                              f"{env.person_verbalized['possessive_adjective']} inventory.\n" \
                              f"It contains: {inv_text} (capacity: {inv_capacity}/" \
                              f"{agent.inventory.capacity}).\n"

            res.track_utilize(
                agent.id, agent.inventory.container.id, 1, 
                src=res.tloc("equip", agent.id),
            )
            res.add_feedback(agent.id, feedback_str)
            res.events.append(Event(
                type="inventory_inspected",
                agent_id=agent.id,
                data={"area_id": current_area_id},
            ))
            return

        # ----- case 2: inspect object -----
        if target in world.auxiliary["obj_name_to_id"]:
            obj_name = target
            obj_id = world.auxiliary["obj_name_to_id"][obj_name]
            obj_description = world.objects[get_def_id(obj_id)].description
            obj_description = f"{obj_description}\n" if obj_description else ""

            # container inspection
            if obj_id in world.container_instances:
                if (
                    obj_id not in current_area.objects
                    and (agent.inventory.container is None or obj_id != agent.inventory.container.id)
                    and obj_id not in agent.items_in_hands
                ):
                    res.add_feedback(
                        agent.id,
                        f"Cannot inspect {obj_name}, the object not found in hand, arms, inventory, or the current area.\n"
                    )
                    return

                container_instance = world.container_instances[obj_id]
                contents_text = env.verbalize_objects(container_instance.inventory)
                container_capacity = sum(
                    world.objects[get_def_id(oid)].size * count
                    for oid, count in container_instance.inventory.items()
                    if world.objects[get_def_id(oid)].size is not None
                )
                feedback_str = f"{env.person_verbalized['subject_pronoun']} inspected " \
                               f"{obj_name}: {obj_description}" \
                               f"It contains: {contents_text} " \
                               f"(capacity: {container_capacity}/" \
                               f"{container_instance.capacity}).\n"

                if obj_id in agent.items_in_hands:
                    res.track_utilize(agent.id, obj_id, 1, src=res.tloc("hand", agent.id))
                elif obj_id in agent.equipped_items_in_limb:
                    res.track_utilize(agent.id, obj_id, 1, src=res.tloc("equip", agent.id))
                elif obj_id in current_area.objects:
                    res.track_utilize(agent.id, obj_id, 1, src=res.tloc("area", current_area.id))

                res.add_feedback(agent.id, feedback_str)
                res.events.append(Event(
                    type="container_inspected",
                    agent_id=agent.id,
                    data={"obj_id": obj_id, "area_id": current_area_id},
                ))
                return

            # regular object inspection: ensure it's present somewhere
            if not (
                (obj_id in agent.items_in_hands and agent.items_in_hands[obj_id] > 0)
                or (obj_id in agent.equipped_items_in_limb and agent.equipped_items_in_limb[obj_id] > 0)
                or (agent.inventory.container is not None
                and obj_id in agent.inventory.items
                and agent.inventory.items[obj_id] > 0)
                or (obj_id in current_area.objects and current_area.objects[obj_id] > 0)
            ):
                res.add_feedback(
                    agent.id,
                    f"Cannot inspect {obj_name}, the object not found in hand, arms, inventory, or the current area.\n"
                )
                return

            # show text if any
            object_text = world.writable_instances[obj_id].text if obj_id in world.writable_instances else world.objects[obj_id].text
            craftable_items_str = ""
            for craftable_obj_id in world.auxiliary["ing_to_obj_map"].get(obj_id, []):
                craftable_obj = world.objects[craftable_obj_id]
                ingredient_str = f"{craftable_obj.name} (made of "
                for ingredient_id in craftable_obj.craft_ingredients.keys():
                    ingredient_name = world.objects[ingredient_id].name if ingredient_id in world.objects else "Unknown"
                    ingredient_str += f"{ingredient_name}, "
                craftable_items_str += ingredient_str.rstrip(", ") + "), "
            craftable_items_str = craftable_items_str.rstrip(", ") if craftable_items_str else "None"

            if object_text:
                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} inspected {obj_name}: {obj_description}\n"
                    f"It can be used to craft: {craftable_items_str}.\n"
                    f"It writes: {object_text}\n"
                )
            else:
                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} inspected {obj_name}: {obj_description}"
                    f"It can be used to craft: {craftable_items_str}.\n"
                    f"It has nothing written on it.\n"
                )
            
            if obj_id in agent.items_in_hands:
                res.track_utilize(agent.id, obj_id, 1, src=res.tloc("hand", agent.id))
            elif obj_id in agent.equipped_items_in_limb:
                res.track_utilize(agent.id, obj_id, 1, src=res.tloc("equip", agent.id))
            elif obj_id in current_area.objects:
                res.track_utilize(agent.id, obj_id, 1, src=res.tloc("area", current_area.id))

            res.events.append(Event(
                type="object_inspected",
                agent_id=agent.id,
                data={"obj_id": obj_id, "area_id": current_area_id},
            ))
            
            return

        # ----- unknown name -----
        res.add_feedback(
            agent.id,
            f"Cannot inspect {target}, the object not found in hand, arms, inventory, or the current area.\n"
        )

class EquipRule(BaseActionRule):
    name = "action_equip"
    verb = "equip"
    param_min = param_max = 1
    params = ["obj_name"]
    description = "The agent equips an object from hand, either as armor or as the inventory container."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        obj_name = ctx.params[0]

        if obj_name in world.auxiliary["obj_name_to_id"]:
            obj_id = world.auxiliary["obj_name_to_id"][obj_name]
        else:
            res.add_feedback(agent.id, f"Cannot equip {obj_name}, not found in hand.\n")
            return
        
        if obj_id not in agent.items_in_hands or agent.items_in_hands[obj_id] <= 0:
            res.add_feedback(agent.id, f"Cannot equip {obj_name}, not found in hand.\n")
            return

        # ----- case 1: equipping a container as inventory -----
        if obj_id in world.container_instances:
            for item in agent.equipped_items_in_limb.keys():
                if item in world.container_instances:
                    res.add_feedback(
                        agent.id,
                        f"Cannot equip {obj_name} as inventory, already have {item} "
                        f"equipped as inventory.\n"
                    )
                    break
            else:
                container_instance = world.container_instances[obj_id]
                agent.inventory.container = container_instance
                obj_count = min(1, agent.items_in_hands[obj_id])
                agent.equipped_items_in_limb[obj_id] = (
                    agent.equipped_items_in_limb.get(obj_id, 0) + obj_count
                )
                agent.items_in_hands[obj_id] -= obj_count
                if agent.items_in_hands[obj_id] == 0:
                    del agent.items_in_hands[obj_id]
                res.track_move(
                    agent.id, obj_id, obj_count,
                    src=res.tloc("hand", agent.id),
                    dst=res.tloc("equip", agent.id),
                )

                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} equipped {obj_name} as inventory.\n"
                )
                res.events.append(Event(
                    type="item_equipped",
                    agent_id=agent.id,
                    data={"obj_id": obj_id, "mode": "inventory"},
                ))
            return

        # ----- case 2: writables cannot be equipped ----
        if obj_id in world.writable_instances:
            res.add_feedback(agent.id, f"Cannot equip {obj_name}, it is not equippable.\n")
            return

        # ----- case 3: equipping armor -----
        obj_def = world.objects[obj_id]
        if obj_def.category in ["armor"]:
            for item in agent.equipped_items_in_limb.keys():
                if world.objects[get_def_id(item)].category == obj_def.category:
                    res.add_feedback(
                        agent.id,
                        f"Cannot equip {obj_name}, already have {world.objects[item].name} "
                        f"equipped as {world.objects[item].category}.\n"
                    )
                    break
            else:
                obj_count = min(1, agent.items_in_hands[obj_id])
                agent.equipped_items_in_limb[obj_id] = (
                    agent.equipped_items_in_limb.get(obj_id, 0) + obj_count
                )
                agent.items_in_hands[obj_id] -= obj_count
                if agent.items_in_hands[obj_id] == 0:
                    del agent.items_in_hands[obj_id]
                res.track_move(
                    agent.id, obj_id, obj_count,
                    src=res.tloc("hand", agent.id),
                    dst=res.tloc("armor", agent.id),
                )

                if hasattr(obj_def, "defense"):
                    agent.defense += obj_def.defense

                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} equipped {obj_name}.\n"
                )
                res.events.append(Event(
                    type="item_equipped",
                    agent_id=agent.id,
                    data={"obj_id": obj_id, "mode": "armor", "category": obj_def.category},
                ))
        else:
            res.add_feedback(agent.id, f"Cannot equip {obj_name}, it is not equippable.\n")

class UnequipRule(BaseActionRule):
    name = "action_unequip"
    verb = "unequip"
    param_min = param_max = 1
    params = ["obj_name"]
    description = "The agent unequips an equipped object back into hand."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        obj_name = ctx.params[0]

        if obj_name in world.auxiliary["obj_name_to_id"]:
            obj_id = world.auxiliary["obj_name_to_id"][obj_name]
        else:
            res.add_feedback(agent.id, f"Cannot unequip {obj_name}, not found in equipped items.\n")
            return

        if obj_id not in agent.equipped_items_in_limb or agent.equipped_items_in_limb[obj_id] <= 0:
            res.add_feedback(agent.id, f"Cannot unequip {obj_name}, not found in equipped items.\n")
            return

        # check hand capacity (2 hands max)
        if sum(agent.items_in_hands.values()) > 1:
            res.add_feedback(agent.id, f"Cannot unequip {obj_name}, not enough space in hand.\n")
            return

        obj_count = min(1, agent.equipped_items_in_limb[obj_id])
        agent.items_in_hands[obj_id] = agent.items_in_hands.get(obj_id, 0) + obj_count
        agent.equipped_items_in_limb[obj_id] -= obj_count
        if agent.equipped_items_in_limb[obj_id] == 0:
            del agent.equipped_items_in_limb[obj_id]
        res.track_move(
            agent.id, obj_id, obj_count,
            src=res.tloc("equip", agent.id),
            dst=res.tloc("hand", agent.id),
        )

        # if it's a container, also unequip from inventory
        if obj_id in world.container_instances:
            if agent.inventory.container is not None and agent.inventory.container.id == obj_id:
                agent.inventory.container = None
        # writable objects are not equippable, so don't need to check them here
        # if the object increases defense, reduce defense
        elif hasattr(world.objects[obj_id], "defense"):
            agent.defense -= world.objects[obj_id].defense

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} unequipped {obj_name}.\n"
        )
        res.events.append(Event(
            type="item_unequipped",
            agent_id=agent.id,
            data={"obj_id": obj_id},
        ))

class CraftRule(BaseActionRule):
    name = "action_craft"
    verb = "craft"
    param_min = param_max = 2
    params = ["amount","obj_name"]
    description = "The agent crafts a specified amount of an object if all necessary ingredients and dependencies are met."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        amount_str, obj_name = ctx.params[0], ctx.params[1]

        if not amount_str.isdigit() or int(amount_str) < 1:
            res.add_feedback(
                agent.id,
                f"Invalid amount '{amount_str}' for crafting {obj_name}. Amount must be a positive integer.\n"
            )
            return

        if obj_name in world.auxiliary["obj_name_to_id"]:
            obj_id = world.auxiliary["obj_name_to_id"][obj_name]
        else:
            res.add_feedback(
                agent.id,
                f"Cannot craft {obj_name}. There's no such object in the world.\n"
            )
            return

        obj_id = get_def_id(obj_id)
        obj_def = world.objects[obj_id]
        ingredients = obj_def.craft_ingredients
        dependencies = obj_def.craft_dependencies
        if not ingredients:
            res.add_feedback(
                agent.id,
                f"Cannot craft {obj_name}. There's no way to craft this object.\n"
            )
            return

        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]
        # check whether all dependency objects are present in the current area
        more_deps = []
        for dep_id in dependencies:
            if dep_id not in current_area.objects or current_area.objects[dep_id] <= 0:
                more_deps.append(dep_id)
        if len(more_deps) > 0:
            dep_names = [world.objects[dep_id].name for dep_id in more_deps]
            dep_name_str = ", ".join(dep_names)
            res.add_feedback(
                agent.id,
                f"Cannot craft {obj_name}. Missing necessary {dep_name_str} in the current area.\n"
            )
            return

        # collect ingredient counts from inventory, hands, and any containers held in hand
        container_instances = [
            world.container_instances[oid]
            for oid in agent.items_in_hands.keys()
            if oid in world.container_instances
        ]
        hand_counts = {}
        inv_counts = {}
        container_counts = [{} for _ in container_instances]
        more_counts = {}

        hands_writable = self.build_writable_index(agent.items_in_hands)
        inv_writable = self.build_writable_index(agent.inventory.items) if agent.inventory.container else {}
        container_writables = [self.build_writable_index(ci.inventory) for ci in container_instances]

        for ing_id, ing_count in ingredients.items():
            hand_count, inv_count, container_count = 0, 0, 0

            if obj_def.usage == "writable":
                hand_count = hands_writable.get(ing_id, 0)
                inv_count = inv_writable.get(ing_id, 0)
                for i, container_writable in enumerate(container_writables):
                    container_counts[i][ing_id] = container_writable.get(ing_id, 0)
                    container_count += container_counts[i][ing_id]
                
            else:
                hand_count = agent.items_in_hands.get(ing_id, 0)
                if agent.inventory.container:
                    inv_count = agent.inventory.items.get(ing_id, 0)
                for i, container_instance in enumerate(container_instances):
                    container_counts[i][ing_id] = container_instance.inventory.get(ing_id, 0)
                    container_count += container_counts[i][ing_id]

            if hand_count + inv_count + container_count < ing_count:
                more_counts[ing_id] = ing_count - hand_count - inv_count - container_count

            hand_counts[ing_id] = hand_count
            inv_counts[ing_id] = inv_count

        if len(more_counts) > 0:
            res.add_feedback(
                agent.id,
                f"Cannot craft a single {obj_name}. Do not have sufficient amount of necessary ingredients.\n"
            )
            return

        craft_amount = int(amount_str)
        # how many copies can we craft given available ingredients
        available_amount = int(min([
            math.floor(
                hand_counts[ing_id]
                + inv_counts[ing_id]
                + sum(container_counts[i][ing_id] for i in range(len(container_instances)))
            ) / ing_count
            for ing_id, ing_count in ingredients.items()
        ]))

        # at this point, available_amount >= 1
        if craft_amount > available_amount:
            craft_amount = available_amount
            res.add_feedback(
                agent.id,
                f"Only enough ingredients to craft {available_amount} {obj_name}.\n"
            )

        # consume ingredients; float counts will be floored
        for ing_id, base_ing_count in ingredients.items():
            ing_count = base_ing_count * craft_amount

            if obj_def.usage == "writable":
                consumed = self.consume_writable_instances(
                    res, agent.id, agent.items_in_hands, ing_id, ing_count, 
                    src_loc=res.tloc("hand", agent.id),
                )
                ing_count -= consumed
                if ing_count > 0:
                    consumed = self.consume_writable_instances(
                        res, agent.id, agent.inventory.items, ing_id, ing_count, 
                        src_loc=res.tloc("container", agent.inventory.container.id),
                    ) if agent.inventory.container else 0
                    ing_count -= consumed
                for i, container_instance in enumerate(container_instances):
                    if ing_count > 0:
                        consumed = self.consume_writable_instances(
                            res, agent.id, container_instance.inventory, ing_id, ing_count, 
                            src_loc=res.tloc("container", container_instance.id),
                        )
                        ing_count -= consumed
                continue

            if hand_counts[ing_id] != 0:
                use_count = min(hand_counts[ing_id], ing_count)
                if use_count > 0:
                    agent.items_in_hands[ing_id] = math.floor(agent.items_in_hands[ing_id] - use_count)
                    if agent.items_in_hands[ing_id] == 0:
                        del agent.items_in_hands[ing_id]
                    ing_count -= use_count
                    res.track_consume(agent.id, ing_id, use_count, src=res.tloc("hand", agent.id))

            if inv_counts[ing_id] != 0 and ing_count > 0:
                use_count = min(inv_counts[ing_id], ing_count)
                if use_count > 0:
                    agent.inventory.items[ing_id] = math.floor(agent.inventory.items[ing_id] - use_count)
                    if agent.inventory.items[ing_id] == 0:
                        del agent.inventory.items[ing_id]
                    ing_count -= use_count
                    res.track_consume(agent.id, ing_id, use_count, src=res.tloc("container", agent.inventory.container.id))

            for i, container_instance in enumerate(container_instances):
                if container_counts[i][ing_id] != 0 and ing_count > 0:
                    use_count = min(container_counts[i][ing_id], ing_count)
                    if use_count > 0:
                        container_instance.inventory[ing_id] = math.floor(
                            container_instance.inventory[ing_id] - use_count
                        )
                        if container_instance.inventory[ing_id] == 0:
                            del container_instance.inventory[ing_id]
                        ing_count -= use_count
                        res.track_consume(agent.id, ing_id, use_count, src=res.tloc("container", container_instance.id))

        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_removed = bool(tutorial_room.get("removed", False))
        if not (tutorial_room and not tutorial_removed):
            env.curr_agents_state["objects_crafted"][agent.id][obj_id] = (
                env.curr_agents_state["objects_crafted"][agent.id].get(obj_id, 0) + craft_amount
            )

        # crafted objects are dropped on the ground
        if obj_def.category == "container" or obj_def.usage == "writable":
            # create a new instance for each new container/writable dropped
            instance_list = world.container_instances if obj_def.category == "container" else world.writable_instances
            id_to_count = (
                world.auxiliary["container_id_to_count"]
                if obj_def.category == "container" 
                else world.auxiliary["writable_id_to_count"]
            )
            for _ in range(craft_amount):
                new_instance = obj_def.create_instance(id_to_count[obj_id])
                instance_list[new_instance.id] = new_instance
                id_to_count[obj_id] += 1
                world.auxiliary["obj_name_to_id"][new_instance.name] = new_instance.id
                current_area.objects[new_instance.id] = current_area.objects.get(new_instance.id, 0) + 1
                res.track_spawn(agent.id, new_instance.id, 1, dst=res.tloc("area", current_area.id))
        else:
            current_area.objects[obj_id] = current_area.objects.get(obj_id, 0) + craft_amount
            res.track_spawn(agent.id, obj_id, craft_amount, dst=res.tloc("area", current_area.id))

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} successfully crafted {craft_amount} {obj_name}. "
            f"It is now on the ground.\n"
        )
        res.events.append(Event(
            type="object_crafted",
            agent_id=agent.id,
            data={"obj_id": obj_id, "amount": craft_amount, "area_id": current_area.id},
        ))
    
    @staticmethod
    def build_writable_index(items: dict[str, int]) -> dict[str, int]:
        idx = defaultdict(int)
        for oid, cnt in items.items():
            base, sep, suffix = oid.rpartition("_")
            if sep and suffix.isdigit():
                idx[base] += cnt
        return idx
    
    @staticmethod
    def consume_writable_instances(res: RuleResult, agent_id: str, items: dict[str, int], base_id: str, amount: int, src_loc: dict) -> int:
        pref = base_id + "_"
        consumed = 0

        for oid in list(items.keys()):
            if consumed >= amount:
                break
            if oid.startswith(pref) and oid[len(pref):].isdigit():
                take = min(items[oid], amount - consumed)
                res.track_consume(agent_id, oid, take, src_loc)
                items[oid] -= take
                if items[oid] <= 0:
                    del items[oid]
                consumed += take

        return consumed

class WriteRule(BaseActionRule):
    name = "action_write"
    verb = "write"
    param_min = param_max = 2
    params = ["text", "writable_name"]
    description = "The agent writes text on a writable object, if holding writing tools."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        text, writable_name = ctx.params[0], ctx.params[1]
        if writable_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(
                agent.id,
                f"Cannot write on {writable_name}. It is not a writable object.\n"
            )
            return
        writable_id = world.auxiliary["obj_name_to_id"][writable_name]
        if writable_id not in world.writable_instances:
            res.add_feedback(
                agent.id,
                f"Cannot write on {writable_id}. It is not a writable object.\n"
            )
            return

        can_write = False
        for obj_id in agent.items_in_hands.keys():
            if world.objects[get_def_id(obj_id)].usage == "write":
                can_write = True
                break
        if not can_write:
            res.add_feedback(
                agent.id,
                f"Cannot write on {writable_name}. No writing tools held in hand.\n"
            )
            return

        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]
        if writable_id not in agent.items_in_hands:
            res.add_feedback(
                agent.id,
                f"Cannot write on {writable_name}. It is not held in hand.\n"
            )
            return

        writable_instance = world.writable_instances[writable_id]
        writable_length = writable_instance.max_text_length - len(writable_instance.text)
        if writable_length == 0:
            res.add_feedback(
                agent.id,
                f"Cannot write more on {writable_name} as there is no more space.\n"
            )
            return

        writable_text = text[:writable_length]
        writable_instance.text += writable_text
        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} wrote on {writable_name}: {writable_text}\n"
        )
        if len(text) > writable_length:
            res.add_feedback(
                agent.id,
                f"Cannot write more on {writable_name} as there is no more space.\n"
            )
        # track usage of the writing tool and the writable object
        res.track_utilize(agent.id, obj_id, 1, src=res.tloc("hand", agent.id))
        res.track_utilize(agent.id, writable_id, 1, src=res.tloc("hand", agent.id))
        res.events.append(Event(
            type="writable_written",
            agent_id=agent.id,
            data={"obj_id": writable_id, "written_text": writable_text, "area_id": current_area.id},
        ))

class SalvageRule(BaseActionRule):
    name = "action_salvage"
    verb = "salvage"
    param_min = param_max = 1
    params = ["material_name"]
    description = (
        "Salvage defeated war machine remains from the current area using a repair_kit held in hand. "
        "Consumes scrap materials to extract devil-forged components that permanently upgrade your stats. "
        "Salvageable materials: scrap_iron (x2) or cog (x2) → +3 attack; fiend_plate (x1) or demon_core_shard (x1) → +2 defense."
    )

    # mapping: material obj_id -> (required_count, stat_name, stat_bonus, feedback_label)
    SALVAGE_RECIPES = {
        "obj_scrap_iron":       (2, "attack",  3, "devil-forged attack component"),
        "obj_cog":              (2, "attack",  3, "devil-forged attack component"),
        "obj_fiend_plate":      (1, "defense", 2, "devil-forged defense component"),
        "obj_demon_core_shard": (1, "defense", 2, "devil-forged defense component"),
    }

    TOOL_ID = "obj_repair_kit"

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        material_name = ctx.params[0]

        # --- resolve material ---
        if material_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(
                agent.id,
                f"Cannot salvage {material_name}. Unknown material.\n"
            )
            return

        material_id = world.auxiliary["obj_name_to_id"][material_name]
        base_material_id = get_def_id(material_id)

        if base_material_id not in self.SALVAGE_RECIPES:
            valid_names = []
            for rid in self.SALVAGE_RECIPES:
                if rid in world.objects:
                    valid_names.append(world.objects[rid].name)
            res.add_feedback(
                agent.id,
                f"Cannot salvage {material_name}. "
                f"Salvageable materials: {', '.join(valid_names)}.\n"
            )
            return

        required_count, stat_name, stat_bonus, component_label = self.SALVAGE_RECIPES[base_material_id]

        # --- check tool in hand ---
        tool_in_hand = False
        for oid in agent.items_in_hands:
            if get_def_id(oid) == self.TOOL_ID:
                tool_in_hand = True
                break

        if not tool_in_hand:
            tool_name = world.objects[self.TOOL_ID].name if self.TOOL_ID in world.objects else "repair_kit"
            res.add_feedback(
                agent.id,
                f"Cannot salvage {material_name}. "
                f"A {tool_name} must be held in hand.\n"
            )
            return

        # --- check material availability in current area ---
        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]
        area_count = current_area.objects.get(base_material_id, 0)

        if area_count < required_count:
            res.add_feedback(
                agent.id,
                f"Cannot salvage {material_name}. "
                f"Need {required_count} in the current area but only found {area_count}.\n"
            )
            return

        # --- consume materials from area ---
        current_area.objects[base_material_id] -= required_count
        if current_area.objects[base_material_id] <= 0:
            del current_area.objects[base_material_id]

        res.track_consume(
            agent.id, base_material_id, required_count,
            src=res.tloc("area", current_area.id),
        )

        # --- track tool usage (not consumed) ---
        for oid in agent.items_in_hands:
            if get_def_id(oid) == self.TOOL_ID:
                res.track_utilize(agent.id, oid, 1, src=res.tloc("hand", agent.id))
                break

        # --- apply permanent stat upgrade ---
        if stat_name == "attack":
            agent.min_attack += stat_bonus
            agent.attack += stat_bonus
        elif stat_name == "defense":
            agent.defense += stat_bonus

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} salvaged {required_count} {material_name} "
            f"and extracted a {component_label}. "
            f"{env.person_verbalized['possessive_adjective'].capitalize()} {stat_name} permanently increased by {stat_bonus}!\n"
        )

        res.events.append(Event(
            type="material_salvaged",
            agent_id=agent.id,
            data={
                "material_id": base_material_id,
                "material_count": required_count,
                "stat_name": stat_name,
                "stat_bonus": stat_bonus,
                "area_id": current_area.id,
            },
        ))


class RallyRule(BaseActionRule):
    name = "action_rally"
    verb = "rally"
    param_min = param_max = 1
    params = ["npc_name"]
    description = (
        "Rally a friendly NPC in the current area by spending coins, temporarily converting them "
        "into a combat ally. The rallied ally automatically deals bonus damage to any enemy the "
        "agent attacks for the next 3 combat rounds. Cost scales with the NPC's level."
    )

    BASE_COST = 8          # coins per rally at level 1
    COST_PER_LEVEL = 4     # additional coins per NPC level
    ALLY_ROUNDS = 3        # how many attack rounds the ally assists
    BASE_BONUS_DAMAGE = 6  # bonus damage at level 1
    BONUS_PER_LEVEL = 3    # additional bonus damage per NPC level

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        target_npc_name = ctx.params[0]

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        # --- resolve NPC ---
        if target_npc_name not in world.auxiliary["npc_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot rally {target_npc_name}, not found in current area.\n")
            return

        target_npc_id = world.auxiliary["npc_name_to_id"][target_npc_name]
        if target_npc_id not in current_area.npcs:
            res.add_feedback(agent.id, f"Cannot rally {target_npc_name}, not found in current area.\n")
            return

        target_npc = world.npc_instances[target_npc_id]

        # --- must be friendly ---
        if target_npc.enemy:
            res.add_feedback(agent.id, f"Cannot rally {target_npc_name}. They are hostile!\n")
            return

        # --- cannot rally quest guide NPCs ---
        GUIDE_BASE_IDS = {"npc_quest_wayfarer_guide", "npc_side_quest_guide"}
        if get_def_id(target_npc_id) in GUIDE_BASE_IDS:
            res.add_feedback(agent.id, f"Cannot rally {target_npc_name}. They are a quest guide, not a fighter.\n")
            return

        # --- check if already rallied ---
        rallied = env.curr_agents_state.get("rallied_allies", {}).get(agent.id)
        if rallied and rallied.get("rounds_left", 0) > 0:
            existing_name = rallied.get("npc_name", "an ally")
            res.add_feedback(
                agent.id,
                f"Cannot rally {target_npc_name}. {existing_name} is already fighting alongside "
                f"{env.person_verbalized['object_pronoun']} ({rallied['rounds_left']} rounds remaining).\n"
            )
            return

        # --- compute cost ---
        npc_level = max(1, getattr(target_npc, "level", 1) or 1)
        cost = self.BASE_COST + self.COST_PER_LEVEL * npc_level

        # --- count agent coins ---
        coin_id = "obj_coin"
        coin_have = int(agent.items_in_hands.get(coin_id, 0))
        if agent.inventory.container:
            coin_have += int(agent.inventory.items.get(coin_id, 0))
        held_containers = [
            world.container_instances[oid]
            for oid in agent.items_in_hands.keys()
            if oid in world.container_instances
        ]
        for ci in held_containers:
            coin_have += int(ci.inventory.get(coin_id, 0))

        if coin_have < cost:
            res.add_feedback(
                agent.id,
                f"Cannot rally {target_npc_name}. Need {cost} coins but only have {coin_have}.\n"
            )
            return

        # --- consume coins (hands → inventory → held containers) ---
        remain = cost

        if remain > 0 and agent.items_in_hands.get(coin_id, 0) > 0:
            take = min(int(agent.items_in_hands[coin_id]), remain)
            agent.items_in_hands[coin_id] -= take
            if agent.items_in_hands[coin_id] <= 0:
                del agent.items_in_hands[coin_id]
            res.track_consume(agent.id, coin_id, take, src=res.tloc("hand", agent.id))
            remain -= take

        if remain > 0 and agent.inventory.container and agent.inventory.items.get(coin_id, 0) > 0:
            take = min(int(agent.inventory.items[coin_id]), remain)
            agent.inventory.items[coin_id] -= take
            if agent.inventory.items[coin_id] <= 0:
                del agent.inventory.items[coin_id]
            res.track_consume(agent.id, coin_id, take, src=res.tloc("container", agent.inventory.container.id))
            remain -= take

        for ci in held_containers:
            if remain <= 0:
                break
            if ci.inventory.get(coin_id, 0) > 0:
                take = min(int(ci.inventory[coin_id]), remain)
                ci.inventory[coin_id] -= take
                if ci.inventory[coin_id] <= 0:
                    del ci.inventory[coin_id]
                res.track_consume(agent.id, coin_id, take, src=res.tloc("container", ci.id))
                remain -= take

        # --- register rallied ally ---
        bonus_damage = self.BASE_BONUS_DAMAGE + self.BONUS_PER_LEVEL * npc_level
        env.curr_agents_state.setdefault("rallied_allies", {})
        env.curr_agents_state["rallied_allies"][agent.id] = {
            "npc_id": target_npc_id,
            "npc_name": target_npc_name,
            "rounds_left": self.ALLY_ROUNDS,
            "bonus_damage": bonus_damage,
            "area_id": current_area_id,
        }

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} rallied {target_npc_name} to "
            f"{env.person_verbalized['possessive_adjective']} cause for {cost} coins! "
            f"{target_npc_name} will fight alongside {env.person_verbalized['object_pronoun']} "
            f"for the next {self.ALLY_ROUNDS} combat rounds, dealing {bonus_damage} bonus damage.\n"
        )
        res.events.append(Event(
            type="npc_rallied",
            agent_id=agent.id,
            data={
                "npc_id": target_npc_id,
                "cost": cost,
                "rounds": self.ALLY_ROUNDS,
                "bonus_damage": bonus_damage,
                "area_id": current_area_id,
            },
        ))


class SiphonRule(BaseActionRule):
    name = "action_siphon"
    verb = "siphon"
    param_min = param_max = 2
    params = ["object_name", "npc_name"]
    description = (
        "Siphon residual dark energy from an enemy NPC mid-combat by sacrificing a held object as a conduit. "
        "The object is consumed. HP is restored equal to the object's coin value, and the NPC's attack power "
        "is reduced by a third of that value for the rest of the encounter. "
        "Requires active combat with the target NPC and a valued object held in hand."
    )

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        obj_name, target_npc_name = ctx.params[0], ctx.params[1]

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        # --- resolve NPC ---
        if target_npc_name not in world.auxiliary["npc_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot siphon from {target_npc_name}, not found in current area.\n")
            return

        target_npc_id = world.auxiliary["npc_name_to_id"][target_npc_name]
        if target_npc_id not in current_area.npcs:
            res.add_feedback(agent.id, f"Cannot siphon from {target_npc_name}, not found in current area.\n")
            return

        target_npc = world.npc_instances[target_npc_id]

        # --- must be an enemy ---
        if not target_npc.enemy:
            res.add_feedback(agent.id, f"Cannot siphon from {target_npc_name}. They are not a hostile war machine.\n")
            return

        # --- must be in active combat with this NPC ---
        active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
        if target_npc_id not in active_combats:
            res.add_feedback(
                agent.id,
                f"Cannot siphon from {target_npc_name}. "
                f"{env.person_verbalized['subject_pronoun']} must be in active combat with them first.\n"
            )
            return

        # --- resolve conduit object ---
        if obj_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot siphon using {obj_name}, not found in hand.\n")
            return

        obj_id = world.auxiliary["obj_name_to_id"][obj_name]

        if obj_id not in agent.items_in_hands or agent.items_in_hands[obj_id] <= 0:
            res.add_feedback(agent.id, f"Cannot siphon using {obj_name}, not found in hand.\n")
            return

        # --- object must not be a container, writable, or station ---
        if obj_id in world.container_instances:
            res.add_feedback(agent.id, f"Cannot siphon using {obj_name}. Containers cannot serve as conduits.\n")
            return
        if obj_id in world.writable_instances:
            res.add_feedback(agent.id, f"Cannot siphon using {obj_name}. Writable objects cannot serve as conduits.\n")
            return

        base_obj_id = get_def_id(obj_id)
        if base_obj_id not in world.objects:
            res.add_feedback(agent.id, f"Cannot siphon using {obj_name}. Unknown object.\n")
            return

        obj_def = world.objects[base_obj_id]

        if obj_def.category == "station":
            res.add_feedback(agent.id, f"Cannot siphon using {obj_name}. Stations cannot serve as conduits.\n")
            return

        if obj_def.value is None or int(obj_def.value) <= 0:
            res.add_feedback(
                agent.id,
                f"Cannot siphon using {obj_name}. The object has no material value to channel dark energy through.\n"
            )
            return

        conduit_value = int(obj_def.value)

        # --- consume the conduit object from hand ---
        agent.items_in_hands[obj_id] -= 1
        if agent.items_in_hands[obj_id] <= 0:
            del agent.items_in_hands[obj_id]
        res.track_consume(agent.id, obj_id, 1, src=res.tloc("hand", agent.id))

        # --- restore HP ---
        hp_restored = min(conduit_value, agent.max_hp - agent.hp)
        agent.hp += hp_restored

        # --- reduce NPC attack power ---
        atk_reduction = max(1, math.floor(conduit_value / 3))
        old_atk = target_npc.attack_power
        target_npc.attack_power = max(0, target_npc.attack_power - atk_reduction)
        actual_reduction = old_atk - target_npc.attack_power

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} channeled dark energy from {target_npc_name} "
            f"through {obj_name}, consuming it as a conduit. "
            f"{env.person_verbalized['subject_pronoun']} restored {hp_restored} HP "
            f"and weakened {target_npc_name}'s attack power by {actual_reduction} "
            f"(now {target_npc.attack_power}).\n"
        )

        res.events.append(Event(
            type="energy_siphoned",
            agent_id=agent.id,
            data={
                "npc_id": target_npc_id,
                "obj_id": obj_id,
                "conduit_value": conduit_value,
                "hp_restored": hp_restored,
                "atk_reduction": actual_reduction,
                "area_id": current_area_id,
            },
        ))


class JamRule(BaseActionRule):
    name = "action_jam"
    verb = "jam"
    param_min = param_max = 2
    params = ["object_name", "npc_name"]
    description = (
        "Jam a held object into an enemy war machine's exposed mechanisms during active combat. "
        "The object is destroyed, but it permanently disables one action from the NPC's combat "
        "pattern cycle. The NPC must have more than one action remaining in its pattern. "
        "Heavier and more valuable objects have a higher chance of disabling an 'attack' action."
    )

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        obj_name, target_npc_name = ctx.params[0], ctx.params[1]

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        # --- resolve NPC ---
        if target_npc_name not in world.auxiliary["npc_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot jam into {target_npc_name}, not found in current area.\n")
            return

        target_npc_id = world.auxiliary["npc_name_to_id"][target_npc_name]
        if target_npc_id not in current_area.npcs:
            res.add_feedback(agent.id, f"Cannot jam into {target_npc_name}, not found in current area.\n")
            return

        target_npc = world.npc_instances[target_npc_id]

        # --- must be an enemy ---
        if not target_npc.enemy:
            res.add_feedback(agent.id, f"Cannot jam into {target_npc_name}. They are not a hostile war machine.\n")
            return

        # --- must be in active combat ---
        active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
        if target_npc_id not in active_combats:
            res.add_feedback(
                agent.id,
                f"Cannot jam into {target_npc_name}. "
                f"{env.person_verbalized['subject_pronoun']} must be in active combat with them first.\n"
            )
            return

        # --- NPC must have more than 1 action in combat pattern ---
        pattern = target_npc.combat_pattern
        if not pattern or len(pattern) <= 1:
            res.add_feedback(
                agent.id,
                f"Cannot jam into {target_npc_name}. Their mechanisms are already too damaged to disrupt further.\n"
            )
            return

        # --- resolve object ---
        if obj_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot jam {obj_name}, not found in hand.\n")
            return

        obj_id = world.auxiliary["obj_name_to_id"][obj_name]

        if obj_id not in agent.items_in_hands or agent.items_in_hands[obj_id] <= 0:
            res.add_feedback(agent.id, f"Cannot jam {obj_name}, not found in hand.\n")
            return

        # --- object must not be a container, writable, or station ---
        if obj_id in world.container_instances:
            res.add_feedback(agent.id, f"Cannot jam {obj_name}. Containers cannot be jammed into machinery.\n")
            return
        if obj_id in world.writable_instances:
            res.add_feedback(agent.id, f"Cannot jam {obj_name}. Writable objects cannot be jammed into machinery.\n")
            return

        base_obj_id = get_def_id(obj_id)
        if base_obj_id not in world.objects:
            res.add_feedback(agent.id, f"Cannot jam {obj_name}. Unknown object.\n")
            return

        obj_def = world.objects[base_obj_id]

        if obj_def.category == "station":
            res.add_feedback(agent.id, f"Cannot jam {obj_name}. Stations cannot be jammed into machinery.\n")
            return

        if obj_def.category == "currency":
            res.add_feedback(agent.id, f"Cannot jam {obj_name}. Coins are too small to disrupt machinery.\n")
            return

        # --- consume the object from hand ---
        agent.items_in_hands[obj_id] -= 1
        if agent.items_in_hands[obj_id] <= 0:
            del agent.items_in_hands[obj_id]
        res.track_consume(agent.id, obj_id, 1, src=res.tloc("hand", agent.id))

        # --- determine which action to remove ---
        # Heavier/more valuable objects bias toward removing "attack" actions
        obj_value = int(obj_def.value) if obj_def.value is not None else 0
        obj_size = int(obj_def.size) if obj_def.size is not None else 1
        weight_score = obj_value + obj_size * 3  # higher = better chance to remove attack

        # Collect indices of each action type
        attack_indices = [i for i, a in enumerate(pattern) if a == "attack"]
        other_indices = [i for i, a in enumerate(pattern) if a != "attack"]

        if attack_indices and other_indices:
            # probability of removing an attack action scales with weight_score
            # base 60% chance, up to 90% with high-value objects
            attack_prob = min(0.9, 0.6 + weight_score * 0.005)
            if env.rng.random() < attack_prob:
                remove_idx = env.rng.choice(attack_indices)
            else:
                remove_idx = env.rng.choice(other_indices)
        elif attack_indices:
            remove_idx = env.rng.choice(attack_indices)
        else:
            remove_idx = env.rng.choice(other_indices)

        removed_action = pattern[remove_idx]
        # Use list.pop to permanently remove from the pattern
        pattern.pop(remove_idx)

        # Adjust the rhythm_index in combat state if needed
        combat_state = active_combats[target_npc_id]
        rhythm_index = combat_state.get("rhythm_index", 0)
        if rhythm_index >= len(pattern):
            combat_state["rhythm_index"] = rhythm_index % len(pattern) if pattern else 0
        elif remove_idx < rhythm_index:
            combat_state["rhythm_index"] = max(0, rhythm_index - 1)

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} jammed {obj_name} into {target_npc_name}'s "
            f"exposed mechanisms, destroying it! The obstruction permanently disabled the machine's "
            f"'{removed_action}' action. {target_npc_name} now cycles through {len(pattern)} "
            f"action(s): {', '.join(pattern)}.\n"
        )

        res.events.append(Event(
            type="mechanism_jammed",
            agent_id=agent.id,
            data={
                "npc_id": target_npc_id,
                "obj_id": obj_id,
                "removed_action": removed_action,
                "remaining_pattern": list(pattern),
                "area_id": current_area_id,
            },
        ))


class OverloadRule(BaseActionRule):
    name = "action_overload"
    verb = "overload"
    param_min = param_max = 1
    params = ["component_name"]
    description = (
        "Overload a held devil-forged component onto your equipped armor, permanently sacrificing "
        "the component to grant a reactive damage aura. Each combat round when an enemy NPC attacks "
        "you, the aura automatically deals counter-damage equal to a percentage of your defense stat. "
        "Eligible components: hellcoal, devil_oil, demon_core_shard, brimite_ore, hellcoal_concentrate, fiend_plate. "
        "Stronger components grant a higher aura percentage. The aura stacks across multiple overloads."
    )

    # component base_obj_id -> aura percentage of defense dealt as counter-damage
    ELIGIBLE_COMPONENTS = {
        "obj_hellcoal":              10,
        "obj_devil_oil":             8,
        "obj_brimite_ore":           12,
        "obj_fiend_plate":           15,
        "obj_demon_core_shard":      25,
        "obj_hellcoal_concentrate":  18,
    }

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        component_name = ctx.params[0]

        # --- resolve component ---
        if component_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot overload with {component_name}. Unknown object.\n")
            return

        obj_id = world.auxiliary["obj_name_to_id"][component_name]
        base_obj_id = get_def_id(obj_id)

        if base_obj_id not in self.ELIGIBLE_COMPONENTS:
            eligible_names = []
            for eid in self.ELIGIBLE_COMPONENTS:
                if eid in world.objects:
                    eligible_names.append(world.objects[eid].name)
            res.add_feedback(
                agent.id,
                f"Cannot overload with {component_name}. "
                f"Eligible devil-forged components: {', '.join(eligible_names)}.\n"
            )
            return

        # --- component must be in hand ---
        if obj_id not in agent.items_in_hands or agent.items_in_hands[obj_id] <= 0:
            res.add_feedback(agent.id, f"Cannot overload with {component_name}. Not found in hand.\n")
            return

        # --- must have armor equipped (not containers) ---
        equipped_armor_id = None
        for eq_id in agent.equipped_items_in_limb:
            eq_base = get_def_id(eq_id)
            if eq_base in world.objects and world.objects[eq_base].category == "armor":
                equipped_armor_id = eq_id
                break

        if equipped_armor_id is None:
            res.add_feedback(
                agent.id,
                f"Cannot overload with {component_name}. "
                f"No armor is currently equipped to receive the aura.\n"
            )
            return

        armor_name = world.objects[get_def_id(equipped_armor_id)].name

        # --- consume the component from hand ---
        agent.items_in_hands[obj_id] -= 1
        if agent.items_in_hands[obj_id] <= 0:
            del agent.items_in_hands[obj_id]
        res.track_consume(agent.id, obj_id, 1, src=res.tloc("hand", agent.id))

        # --- apply aura ---
        aura_pct = self.ELIGIBLE_COMPONENTS[base_obj_id]
        env.curr_agents_state.setdefault("overload_aura", {})
        current_aura = env.curr_agents_state["overload_aura"].get(agent.id, {})
        old_pct = current_aura.get("aura_pct", 0)
        new_pct = old_pct + aura_pct
        env.curr_agents_state["overload_aura"][agent.id] = {"aura_pct": new_pct}

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} overloaded {component_name} into "
            f"{env.person_verbalized['possessive_adjective']} {armor_name}, sacrificing it to "
            f"infuse the armor with devil-forged energy! "
            f"Reactive damage aura increased by {aura_pct}% of defense "
            f"(total aura: {new_pct}% of defense as counter-damage per attack received).\n"
        )

        res.events.append(Event(
            type="armor_overloaded",
            agent_id=agent.id,
            data={
                "component_id": obj_id,
                "armor_id": equipped_armor_id,
                "aura_pct_added": aura_pct,
                "total_aura_pct": new_pct,
                "area_id": env.curr_agents_state["area"][agent.id],
            },
        ))


class ScavengeRule(BaseActionRule):
    name = "action_scavenge"
    verb = "scavenge"
    param_min = 0
    param_max = 0
    params = []
    description = (
        "Scavenge the current area to discover hidden objects buried in the ruins. "
        "Your knowledge of war machine hiding spots — gained from defeating unique enemy types — "
        "increases both the chance and quality of findings. "
        "Each area can only be scavenged once every few steps."
    )

    COOLDOWN_STEPS = 5  # steps before the same area can be scavenged again

    # Tier thresholds based on number of unique enemy types killed
    # (min_unique_kills, discovery_probability, max_items, level_bonus)
    TIERS = [
        (0,  0.30, 1, 0),   # novice: 30% chance, 1 item, no level bonus
        (2,  0.50, 2, 0),   # apprentice: 50% chance, up to 2 items
        (4,  0.65, 2, 1),   # journeyman: 65% chance, up to 2 items, +1 level
        (6,  0.80, 3, 1),   # veteran: 80% chance, up to 3 items, +1 level
        (10, 0.95, 3, 2),   # master: 95% chance, up to 3 items, +2 level
    ]

    def _get_tier(self, unique_kills_count: int):
        """Return the best matching tier for the given unique kill count."""
        best = self.TIERS[0]
        for tier in self.TIERS:
            if unique_kills_count >= tier[0]:
                best = tier
        return best

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        rng = env.rng

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        # --- check active combat ---
        active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
        if active_combats:
            blocking_npcs = [
                world.npc_instances[nid].name
                for nid, cs in active_combats.items()
                if cs.get("area_id") == current_area_id and nid in world.npc_instances
            ]
            if blocking_npcs:
                res.add_feedback(
                    agent.id,
                    f"Cannot scavenge while in combat with {', '.join(blocking_npcs)}.\n"
                )
                return

        # --- cooldown check ---
        cooldowns = env.curr_agents_state.setdefault("scavenge_cooldowns", {})
        agent_cooldowns = cooldowns.setdefault(agent.id, {})
        last_step = agent_cooldowns.get(current_area_id, -999)
        if env.steps - last_step < self.COOLDOWN_STEPS:
            remaining = self.COOLDOWN_STEPS - (env.steps - last_step)
            res.add_feedback(
                agent.id,
                f"This area was recently scavenged. Wait {remaining} more step(s) before scavenging here again.\n"
            )
            return

        # --- record cooldown ---
        agent_cooldowns[current_area_id] = env.steps

        # --- determine tier from unique kills ---
        unique_kills = env.curr_agents_state.get("unique_npcs_killed", {}).get(agent.id, [])
        unique_kills_count = len(unique_kills)
        _, discovery_prob, max_items, level_bonus = self._get_tier(unique_kills_count)

        # --- roll for discovery ---
        if rng.random() > discovery_prob:
            if unique_kills_count == 0:
                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} searched the ruins but found nothing. "
                    f"Defeating different types of war machines might reveal knowledge of hidden caches.\n"
                )
            else:
                res.add_feedback(
                    agent.id,
                    f"{env.person_verbalized['subject_pronoun']} searched the ruins but found nothing this time.\n"
                )
            res.events.append(Event(
                type="scavenge_failed",
                agent_id=agent.id,
                data={"area_id": current_area_id, "unique_kills": unique_kills_count},
            ))
            return

        # --- build loot pool based on area level + level_bonus ---
        area_level = current_area.level
        max_loot_level = min(area_level + level_bonus, 5)  # cap at level 5

        undistributable = set(
            env.world_definition.get("initializations", {})
            .get("undistributable_objects", [])
        )

        loot_pool = []
        loot_weights = []
        for oid, obj in world.objects.items():
            if obj.category in ("station", "currency"):
                continue
            if obj.category == "container" or obj.usage == "writable":
                continue
            if oid in undistributable:
                continue
            if getattr(obj, "quest", False):
                continue
            if obj.level > max_loot_level:
                continue
            # weight: objects closer to area level are more likely
            level_diff = abs(obj.level - area_level)
            if level_diff == 0:
                w = 1.0
            elif level_diff == 1:
                w = 0.5
            elif level_diff == 2:
                w = 0.2
            else:
                w = 0.05
            # raw materials (no craft ingredients) are more common finds
            if not obj.craft_ingredients:
                w *= 1.5
            loot_pool.append(oid)
            loot_weights.append(w)

        if not loot_pool:
            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} searched the ruins but found nothing of value.\n"
            )
            return

        # --- determine how many items to find ---
        num_items = rng.randint(1, max_items)

        # --- select and spawn items ---
        found_items = {}
        for _ in range(num_items):
            chosen_id = rng.choices(loot_pool, weights=loot_weights, k=1)[0]
            found_items[chosen_id] = found_items.get(chosen_id, 0) + 1

        for oid, count in found_items.items():
            current_area.objects[oid] = current_area.objects.get(oid, 0) + count
            res.track_spawn(agent.id, oid, count, dst=res.tloc("area", current_area.id))

        # --- build feedback ---
        item_strs = []
        for oid, count in found_items.items():
            obj_name = world.objects[oid].name
            item_strs.append(f"{count} {obj_name}")
        items_text = ", ".join(item_strs)

        if unique_kills_count > 0:
            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} scavenged the ruins, drawing on "
                f"knowledge gained from defeating {unique_kills_count} unique war machine type(s). "
                f"{env.person_verbalized['subject_pronoun']} discovered: {items_text}! "
                f"The items are now on the ground.\n"
            )
        else:
            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} scavenged the ruins and got lucky! "
                f"{env.person_verbalized['subject_pronoun']} discovered: {items_text}! "
                f"The items are now on the ground.\n"
            )

        res.events.append(Event(
            type="area_scavenged",
            agent_id=agent.id,
            data={
                "area_id": current_area_id,
                "found_items": found_items,
                "unique_kills": unique_kills_count,
                "tier_discovery_prob": discovery_prob,
            },
        ))


class BuyRule(BaseActionRule):
    name = "action_buy"
    verb = "buy"
    param_min = param_max = 3
    params = ["amount", "obj_name", "npc_name"]
    description = "Buy a specified amount of an object from a merchant NPC using coins."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        amount_str, obj_name, npc_name = ctx.params[0], ctx.params[1], ctx.params[2]

        if not amount_str.isdigit() or int(amount_str) < 1:
            res.add_feedback(agent.id, f"Invalid amount '{amount_str}' for buying {obj_name}. Amount must be a positive integer.\n")
            return
        buy_amount = int(amount_str)

        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]

        if npc_name not in world.auxiliary["npc_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot buy from {npc_name}, not found in current area.\n")
            return
        npc_id = world.auxiliary["npc_name_to_id"][npc_name]
        if npc_id not in current_area.npcs:
            res.add_feedback(agent.id, f"Cannot buy from {npc_name}, not found in current area.\n")
            return

        npc = world.npc_instances[npc_id]
        if npc.role != "merchant":
            res.add_feedback(agent.id, f"Cannot buy from {npc_name}; they are not a merchant.\n")
            return

        if obj_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot buy {obj_name}. The merchant doesn't have it.\n")
            return
        requested_id = world.auxiliary["obj_name_to_id"][obj_name]
        base_id = get_def_id(requested_id)

        if base_id not in world.objects:
            res.add_feedback(agent.id, f"Cannot buy {obj_name}. The merchant doesn't have it.\n")
            return

        obj_def = world.objects[base_id]
        if obj_def.value is None:
            res.add_feedback(agent.id, f"Cannot buy {obj_name}. The merchant doesn't have it.\n")
            return

        unit_price = int(obj_def.value)
        if unit_price < 0:
            res.add_feedback(agent.id, f"Cannot buy {obj_name}. Invalid value offered.\n")
            return

        total_stock = int(npc.inventory.get(base_id, 0))
        if total_stock <= 0:
            res.add_feedback(agent.id, f"Cannot buy {obj_name}. {npc_name} is out of stock.\n")
            return

        if buy_amount > total_stock:
            buy_amount = total_stock
            res.add_feedback(agent.id, f"Only enough stock to buy {total_stock} {obj_name}.\n")

        total_cost = unit_price * buy_amount

        coin_id = "obj_coin"
        coin_have = int(agent.items_in_hands.get(coin_id, 0))
        if agent.inventory.container:
            coin_have += int(agent.inventory.items.get(coin_id, 0))

        held_containers = [
            world.container_instances[oid]
            for oid in agent.items_in_hands.keys()
            if oid in world.container_instances
        ]
        for ci in held_containers:
            coin_have += int(ci.inventory.get(coin_id, 0))

        if coin_have < total_cost:
            res.add_feedback(
                agent.id,
                f"Cannot buy {buy_amount} {obj_name}. Need {total_cost} coins but only have {coin_have}.\n"
            )
            return

        # consume coins
        remain_pay = total_cost
        
        if remain_pay > 0 and agent.items_in_hands.get(coin_id, 0) > 0:
            take = min(int(agent.items_in_hands[coin_id]), remain_pay)
            agent.items_in_hands[coin_id] -= take
            if agent.items_in_hands[coin_id] <= 0:
                del agent.items_in_hands[coin_id]
            res.track_consume(agent.id, coin_id, take, src=res.tloc("hand", agent.id))
            remain_pay -= take

        if remain_pay > 0 and agent.inventory.container and agent.inventory.items.get(coin_id, 0) > 0:
            take = min(int(agent.inventory.items[coin_id]), remain_pay)
            agent.inventory.items[coin_id] -= take
            if agent.inventory.items[coin_id] <= 0:
                del agent.inventory.items[coin_id]
            res.track_consume(agent.id, coin_id, take, src=res.tloc("container", agent.inventory.container.id))
            remain_pay -= take

        for ci in held_containers:
            if remain_pay <= 0:
                break
            if ci.inventory.get(coin_id, 0) > 0:
                take = min(int(ci.inventory[coin_id]), remain_pay)
                ci.inventory[coin_id] -= take
                if ci.inventory[coin_id] <= 0:
                    del ci.inventory[coin_id]
                res.track_consume(agent.id, coin_id, take, src=res.tloc("container", ci.id))
                remain_pay -= take

        if remain_pay != 0:
            res.add_feedback(agent.id, f"Cannot buy {obj_name}. Failed to pay coins.\n")
            return

        npc.coins = int(getattr(npc, "coins", 0)) + total_cost

        # deliver purchased items and drop to the ground
        npc.inventory[base_id] = int(npc.inventory.get(base_id, 0)) - buy_amount
        if npc.inventory[base_id] <= 0:
            del npc.inventory[base_id]
        if obj_def.category == "container" or obj_def.usage == "writable":
            # create a new instance for each new container/writable dropped
            instance_list = world.container_instances if obj_def.category == "container" else world.writable_instances
            id_to_count = (
                world.auxiliary["container_id_to_count"]
                if obj_def.category == "container" 
                else world.auxiliary["writable_id_to_count"]
            )
            for _ in range(buy_amount):
                new_instance = obj_def.create_instance(id_to_count[base_id])
                instance_list[new_instance.id] = new_instance
                id_to_count[base_id] += 1
                world.auxiliary["obj_name_to_id"][new_instance.name] = new_instance.id
                current_area.objects[new_instance.id] = current_area.objects.get(new_instance.id, 0) + 1
                res.track_spawn(agent.id, new_instance.id, 1, dst=res.tloc("area", current_area.id))
        else:
            current_area.objects[base_id] = current_area.objects.get(base_id, 0) + buy_amount
            res.track_spawn(agent.id, base_id, buy_amount, dst=res.tloc("area", current_area.id))
        
        tutorial_room = world.auxiliary.get("tutorial_room") or {}
        tutorial_removed = bool(tutorial_room.get("removed", False))
        if not (tutorial_room and not tutorial_removed):
            env.curr_agents_state["objects_traded"][agent.id][base_id] = (
                env.curr_agents_state["objects_traded"][agent.id].get(base_id, 0) + buy_amount
            )
        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} bought {buy_amount} {obj_name} from {npc_name} for {total_cost} coins. "
            f"It is now on the ground.\n"
        )
        res.events.append(Event(
            type="object_bought",
            agent_id=agent.id,
            data={"npc_id": npc_id, "obj_id": base_id, "amount": buy_amount, "unit_price": unit_price, "area_id": current_area.id},
        ))

class SellRule(BaseActionRule):
    name = "action_sell"
    verb = "sell"
    param_min = param_max = 3
    params = ["amount", "obj_name", "npc_name"]
    description = "Sell a specified amount of an object to a merchant NPC. Payout is 0.5 times the real value of the objects."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        amount_str, obj_name, npc_name = ctx.params[0], ctx.params[1], ctx.params[2]

        if not amount_str.isdigit() or int(amount_str) < 1:
            res.add_feedback(agent.id, f"Invalid amount '{amount_str}' for selling {obj_name}. Amount must be a positive integer.\n")
            return
        sell_amount_req = int(amount_str)

        current_area = world.area_instances[env.curr_agents_state["area"][agent.id]]

        if npc_name not in world.auxiliary["npc_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot sell to {npc_name}, not found in current area.\n")
            return
        npc_id = world.auxiliary["npc_name_to_id"][npc_name]
        if npc_id not in current_area.npcs:
            res.add_feedback(agent.id, f"Cannot sell to {npc_name}, not found in current area.\n")
            return

        npc = world.npc_instances[npc_id]
        if npc.role != "merchant":
            res.add_feedback(agent.id, f"Cannot sell to {npc_name}; they are not a merchant.\n")
            return

        if obj_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot sell {obj_name}. It is not in {env.person_verbalized['possessive_adjective']} hand, inventory, or held containers.\n")
            return

        obj_id = world.auxiliary["obj_name_to_id"][obj_name]
        if obj_id in world.writable_instances or obj_id in world.container_instances:
            res.add_feedback(agent.id, f"Cannot sell {obj_name}. This object is not tradable.\n")
            return
        
        obj_def = world.objects[obj_id]
        if obj_def.category == "container" or obj_def.usage == "writable":
            res.add_feedback(agent.id, f"Cannot sell {obj_name}. This object is not tradable.\n")
            return
        if obj_def.value is None:
            res.add_feedback(agent.id, f"Cannot sell {obj_name}. This object is not tradable.\n")
            return

        buy_value = int(obj_def.value)
        sell_unit_price = int(math.floor(buy_value * 0.5))

        if sell_unit_price <= 0:
            res.add_feedback(agent.id, f"Cannot sell {obj_name}. This object has no value for selling.\n")
            return

        # count sellable items
        held_containers = [
            world.container_instances[oid]
            for oid in agent.items_in_hands.keys()
            if oid in world.container_instances
        ]

        sellable_total = 0
        sellable_total += int(agent.items_in_hands.get(obj_id, 0))
        if agent.inventory.container:
            sellable_total += int(agent.inventory.items.get(obj_id, 0))
        for ci in held_containers:
            sellable_total += int(ci.inventory.get(obj_id, 0))
 
        if sellable_total <= 0:
            res.add_feedback(agent.id, f"Cannot sell {obj_name}. Not found in hand, inventory, or held containers.\n")
            return

        sell_amount = sell_amount_req
        if sell_amount > sellable_total:
            sell_amount = sellable_total
            res.add_feedback(agent.id, f"Only enough stock to sell {sellable_total} {obj_name}.\n")

        merchant_coins = int(getattr(npc, "coins", 0))
        if sell_unit_price > 0:
            max_affordable = merchant_coins // sell_unit_price
            if max_affordable <= 0:
                res.add_feedback(agent.id, f"Cannot sell {obj_name}. {npc_name} does not have enough coins to buy.\n")
                return
            if sell_amount > max_affordable:
                sell_amount = max_affordable
                res.add_feedback(agent.id, f"{npc_name} can only afford to buy {max_affordable} {obj_name}.\n")

        total_gain = sell_unit_price * sell_amount

        remaining = sell_amount
        if remaining > 0:
            have_count = int(agent.items_in_hands.get(obj_id, 0))
            take = min(have_count, remaining)
            if take > 0:
                agent.items_in_hands[obj_id] -= take
                if agent.items_in_hands[obj_id] <= 0:
                    del agent.items_in_hands[obj_id]
                remaining -= take
                res.track_consume(
                    agent.id, obj_id, take,
                    src=res.tloc("hand", agent.id),
                )
        if remaining > 0 and agent.inventory.container:
            have_count = int(agent.inventory.items.get(obj_id, 0))
            take = min(have_count, remaining)
            if take > 0:
                agent.inventory.items[obj_id] -= take
                if agent.inventory.items[obj_id] <= 0:
                    del agent.inventory.items[obj_id]
                remaining -= take
                res.track_consume(
                    agent.id, obj_id, take,
                    src=res.tloc("container", agent.inventory.container.id),
                )
        for ci in held_containers:
            if remaining <= 0:
                break
            have_count = int(ci.inventory.get(obj_id, 0))
            take = min(have_count, remaining)
            if take > 0:
                ci.inventory[obj_id] -= take
                if ci.inventory[obj_id] <= 0:
                    del ci.inventory[obj_id]
                remaining -= take
                res.track_consume(
                    agent.id, obj_id, take,
                    src=res.tloc("container", ci.id),
                )

        npc.coins = merchant_coins - total_gain
        coin_id = "obj_coin"
        if total_gain > 0:
            current_area.objects[coin_id] = current_area.objects.get(coin_id, 0) + total_gain
            res.track_spawn(agent.id, coin_id, total_gain, dst=res.tloc("area", current_area.id))

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} sold {sell_amount} {obj_name} to {npc_name} for {total_gain} coins. "
            f"The coins are now on the ground.\n"
        )
        res.events.append(Event(
            type="object_sold",
            agent_id=agent.id,
            data={"npc_id": npc_id, "obj_id": obj_id, "amount": sell_amount, "unit_price": sell_unit_price, "area_id": current_area.id},
        ))
