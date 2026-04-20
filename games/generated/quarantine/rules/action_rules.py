import math
from games.generated.quarantine.rule import BaseActionRule, RuleContext, RuleResult, Event
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

class BarricadeRule(BaseActionRule):
    name = "action_barricade"
    verb = "barricade"
    param_min = param_max = 1
    params = ["area_name"]
    description = "Barricade the path to a neighboring area using wooden materials (requires 3 wood units from hand/inventory: wood_log or wood_plank each count as 1). The path is permanently blocked for everyone — players and NPCs alike."

    WOOD_IDS = {"obj_wood_log", "obj_wood_plank"}
    REQUIRED_WOOD = 3

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        area_name = ctx.params[0]

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        # check if agent is in active combat - cannot barricade during combat
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
                    f"Cannot barricade while in combat with {npc_names}.\n"
                )
                return

        # find the neighboring area matching the name
        target_area_id = None
        for neighbor_id in current_area.neighbors:
            if world.area_instances[neighbor_id].name == area_name:
                target_area_id = neighbor_id
                break

        if target_area_id is None:
            res.add_feedback(
                agent.id,
                f"Cannot barricade path to {area_name}. There's no path to {area_name} from here.\n"
            )
            return

        # count available wood from hand and inventory
        wood_sources = []  # list of (location, obj_id, available_count)
        for wood_id in self.WOOD_IDS:
            hand_count = agent.items_in_hands.get(wood_id, 0)
            if hand_count > 0:
                wood_sources.append(("hand", wood_id, hand_count))
            if agent.inventory.container:
                inv_count = agent.inventory.items.get(wood_id, 0)
                if inv_count > 0:
                    wood_sources.append(("inventory", wood_id, inv_count))

        total_wood = sum(count for _, _, count in wood_sources)
        if total_wood < self.REQUIRED_WOOD:
            res.add_feedback(
                agent.id,
                f"Cannot barricade path to {area_name}. Need {self.REQUIRED_WOOD} wood units "
                f"(wood_log or wood_plank) but only have {total_wood}.\n"
            )
            return

        # consume wood: prefer hand first, then inventory
        remaining = self.REQUIRED_WOOD
        for location, wood_id, available in wood_sources:
            if remaining <= 0:
                break
            take = min(available, remaining)
            if location == "hand":
                agent.items_in_hands[wood_id] -= take
                if agent.items_in_hands[wood_id] <= 0:
                    del agent.items_in_hands[wood_id]
                res.track_consume(agent.id, wood_id, take, src=res.tloc("hand", agent.id))
            elif location == "inventory":
                agent.inventory.items[wood_id] -= take
                if agent.inventory.items[wood_id] <= 0:
                    del agent.inventory.items[wood_id]
                res.track_consume(
                    agent.id, wood_id, take,
                    src=res.tloc("container", agent.inventory.container.id),
                )
            remaining -= take

        # remove the path in both directions (permanent)
        if target_area_id in current_area.neighbors:
            del current_area.neighbors[target_area_id]
        target_area = world.area_instances[target_area_id]
        if current_area_id in target_area.neighbors:
            del target_area.neighbors[current_area_id]

        # record the barricade
        barricaded = env.curr_agents_state.setdefault("barricaded_paths", [])
        barricaded.append({
            "from": current_area_id,
            "to": target_area_id,
            "agent_id": agent.id,
            "step": ctx.step_index,
        })

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} barricaded the path to {area_name}. "
            f"The passage is now permanently blocked.\n"
        )
        res.events.append(Event(
            type="path_barricaded",
            agent_id=agent.id,
            data={
                "from_area": current_area_id,
                "to_area": target_area_id,
                "area_name": area_name,
                "wood_used": self.REQUIRED_WOOD,
            },
        ))


class ConcocRule(BaseActionRule):
    name = "action_concoct"
    verb = "concoct"
    param_min = param_max = 2
    params = ["herb1_name", "herb2_name"]
    description = "Combine two herbs to concoct a medical remedy that restores HP. Different herb combinations yield different potency: green+green=20HP, green+red=40HP, red+red=60HP, green+blue=50HP, red+blue=70HP, blue+blue=90HP."

    HERB_IDS = {"obj_green_herb", "obj_red_herb", "obj_blue_herb"}

    # Potency table keyed by frozenset of the two base herb ids
    POTENCY = {
        frozenset(["obj_green_herb", "obj_green_herb_dup"]): 20,  # placeholder, handled below
    }

    @staticmethod
    def _potency(herb_a: str, herb_b: str) -> int:
        pair = tuple(sorted([herb_a, herb_b]))
        table = {
            ("obj_green_herb", "obj_green_herb"): 20,
            ("obj_green_herb", "obj_red_herb"): 40,
            ("obj_red_herb", "obj_red_herb"): 60,
            ("obj_blue_herb", "obj_green_herb"): 50,
            ("obj_blue_herb", "obj_red_herb"): 70,
            ("obj_blue_herb", "obj_blue_herb"): 90,
        }
        return table.get(pair, 0)

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        herb1_name, herb2_name = ctx.params[0], ctx.params[1]

        # resolve names to ids
        herb1_id = world.auxiliary["obj_name_to_id"].get(herb1_name)
        herb2_id = world.auxiliary["obj_name_to_id"].get(herb2_name)

        if herb1_id is None:
            res.add_feedback(agent.id, f"Cannot concoct: {herb1_name} is not a known object.\n")
            return
        if herb2_id is None:
            res.add_feedback(agent.id, f"Cannot concoct: {herb2_name} is not a known object.\n")
            return

        herb1_base = get_def_id(herb1_id)
        herb2_base = get_def_id(herb2_id)

        if herb1_base not in self.HERB_IDS:
            res.add_feedback(agent.id, f"Cannot concoct: {herb1_name} is not a herb.\n")
            return
        if herb2_base not in self.HERB_IDS:
            res.add_feedback(agent.id, f"Cannot concoct: {herb2_name} is not a herb.\n")
            return

        potency = self._potency(herb1_base, herb2_base)
        if potency <= 0:
            res.add_feedback(agent.id, f"Cannot concoct: invalid herb combination.\n")
            return

        # --- locate and consume herbs ---
        # We need to consume one unit of each herb.
        # If both herbs are the same type we need 2 units total.
        # Sources (priority): hand, inventory, held containers
        herbs_needed: list[str] = [herb1_base, herb2_base]
        consumed: list[tuple] = []  # list of (herb_base_id, source_label, src_tloc)

        # Build a mutable availability map
        avail_hand = dict(agent.items_in_hands)
        avail_inv = dict(agent.inventory.items) if agent.inventory.container else {}
        held_containers = [
            world.container_instances[oid]
            for oid in agent.items_in_hands.keys()
            if oid in world.container_instances
        ]
        avail_containers = [dict(ci.inventory) for ci in held_containers]

        for herb_base in herbs_needed:
            found = False
            # try hand
            if avail_hand.get(herb_base, 0) > 0:
                avail_hand[herb_base] -= 1
                consumed.append((herb_base, "hand", res.tloc("hand", agent.id)))
                found = True
            # try inventory
            if not found and avail_inv.get(herb_base, 0) > 0:
                avail_inv[herb_base] -= 1
                consumed.append((herb_base, "inventory", res.tloc("container", agent.inventory.container.id)))
                found = True
            # try held containers
            if not found:
                for i, ci in enumerate(held_containers):
                    if avail_containers[i].get(herb_base, 0) > 0:
                        avail_containers[i][herb_base] -= 1
                        consumed.append((herb_base, f"container_{ci.id}", res.tloc("container", ci.id)))
                        found = True
                        break
            if not found:
                herb_name = world.objects[herb_base].name if herb_base in world.objects else herb_base
                res.add_feedback(
                    agent.id,
                    f"Cannot concoct: not enough {herb_name}. Need one of each herb in hand, inventory, or held containers.\n"
                )
                return

        # Actually consume from real state
        for herb_base, source_label, src_tloc in consumed:
            if source_label == "hand":
                agent.items_in_hands[herb_base] = agent.items_in_hands.get(herb_base, 0) - 1
                if agent.items_in_hands[herb_base] <= 0:
                    del agent.items_in_hands[herb_base]
                res.track_consume(agent.id, herb_base, 1, src=src_tloc)
            elif source_label == "inventory":
                agent.inventory.items[herb_base] = agent.inventory.items.get(herb_base, 0) - 1
                if agent.inventory.items[herb_base] <= 0:
                    del agent.inventory.items[herb_base]
                res.track_consume(agent.id, herb_base, 1, src=src_tloc)
            else:
                # held container
                ci_id = src_tloc["id"]
                ci = world.container_instances[ci_id]
                ci.inventory[herb_base] = ci.inventory.get(herb_base, 0) - 1
                if ci.inventory[herb_base] <= 0:
                    del ci.inventory[herb_base]
                res.track_consume(agent.id, herb_base, 1, src=src_tloc)

        # --- restore HP ---
        old_hp = agent.hp
        agent.hp = min(agent.max_hp, agent.hp + potency)
        actual_heal = agent.hp - old_hp

        herb1_display = world.objects[herb1_base].name
        herb2_display = world.objects[herb2_base].name

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} combined {herb1_display} and {herb2_display} "
            f"to concoct a remedy and restored {actual_heal} HP "
            f"(HP: {old_hp} -> {agent.hp}/{agent.max_hp}).\n"
        )

        current_area_id = env.curr_agents_state["area"][agent.id]
        res.events.append(Event(
            type="herbs_concocted",
            agent_id=agent.id,
            data={
                "herb1": herb1_base,
                "herb2": herb2_base,
                "potency": potency,
                "actual_heal": actual_heal,
                "area_id": current_area_id,
            },
        ))


class ShoveRule(BaseActionRule):
    name = "action_shove"
    verb = "shove"
    param_min = param_max = 2
    params = ["furniture_name", "npc_name"]
    description = "Shove a piece of furniture (e.g. desk, shelf) in the current area at an NPC you are in combat with, stunning it for one round so it skips its next combat action. The furniture is destroyed in the process."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        furniture_name, target_npc_name = ctx.params[0], ctx.params[1]

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        # --- resolve furniture ---
        if furniture_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot shove {furniture_name}, not found in current area.\n")
            return
        furniture_id = world.auxiliary["obj_name_to_id"][furniture_name]

        # must be a furniture object
        furniture_def_id = get_def_id(furniture_id)
        if furniture_def_id not in world.objects or world.objects[furniture_def_id].category != "furniture":
            res.add_feedback(agent.id, f"Cannot shove {furniture_name}. It is not a piece of furniture.\n")
            return

        # must be present in the current area (furniture is too heavy to carry)
        if furniture_id not in current_area.objects or current_area.objects[furniture_id] <= 0:
            res.add_feedback(agent.id, f"Cannot shove {furniture_name}, not found in current area.\n")
            return

        # --- resolve target NPC ---
        if target_npc_name not in world.auxiliary["npc_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot shove furniture at {target_npc_name}, not found in current area.\n")
            return
        target_npc_id = world.auxiliary["npc_name_to_id"][target_npc_name]
        if target_npc_id not in current_area.npcs:
            res.add_feedback(agent.id, f"Cannot shove furniture at {target_npc_name}, not found in current area.\n")
            return

        # --- must be in active combat with the target ---
        active_combats = env.curr_agents_state.get("active_combats", {}).get(agent.id, {})
        if target_npc_id not in active_combats:
            res.add_feedback(
                agent.id,
                f"Cannot shove furniture at {target_npc_name}. "
                f"You must be in active combat with {target_npc_name} first.\n"
            )
            return

        target_npc_instance = world.npc_instances[target_npc_id]

        # --- consume the furniture (destroy it) ---
        current_area.objects[furniture_id] -= 1
        if current_area.objects[furniture_id] <= 0:
            del current_area.objects[furniture_id]
        res.track_consume(agent.id, furniture_id, 1, src=res.tloc("area", current_area.id))

        # --- stun the NPC: mark it to skip its next combat action ---
        stunned_npcs = env.curr_agents_state.setdefault("stunned_npcs", {})
        stunned_npcs[target_npc_id] = stunned_npcs.get(target_npc_id, 0) + 1

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} shoved the {furniture_name} at {target_npc_name}! "
            f"The {furniture_name} shatters on impact. {target_npc_name} is stunned and will skip its next action.\n"
        )

        res.events.append(Event(
            type="furniture_shoved",
            agent_id=agent.id,
            data={
                "furniture_id": furniture_id,
                "npc_id": target_npc_id,
                "area_id": current_area_id,
            },
        ))


class EavesdropRule(BaseActionRule):
    name = "action_eavesdrop"
    verb = "eavesdrop"
    param_min = param_max = 1
    params = ["area_name"]
    description = "Press your ear against the door to a neighboring area and listen for sounds on the other side. Reveals the names and number of NPCs in the adjacent area without entering it. Costs one full turn and has a small chance of alerting enemy NPCs behind the door, initiating combat."

    ALERT_PROBABILITY = 0.15  # 15% chance of alerting enemies

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        area_name = ctx.params[0]

        current_area_id = env.curr_agents_state["area"][agent.id]
        current_area = world.area_instances[current_area_id]

        # cannot eavesdrop while in active combat
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
                    f"Cannot eavesdrop while in combat with {npc_names}.\n"
                )
                return

        # find the neighboring area matching the name
        target_area_id = None
        for neighbor_id in current_area.neighbors:
            if world.area_instances[neighbor_id].name == area_name:
                target_area_id = neighbor_id
                break

        if target_area_id is None:
            res.add_feedback(
                agent.id,
                f"Cannot eavesdrop towards {area_name}. There's no path to {area_name} from here.\n"
            )
            return

        target_area = world.area_instances[target_area_id]

        # gather NPC info from the target area
        npc_names_in_target = []
        enemy_npc_ids_in_target = []
        for npc_id in target_area.npcs:
            if npc_id in world.npc_instances:
                npc_inst = world.npc_instances[npc_id]
                npc_names_in_target.append(npc_inst.name)
                if npc_inst.enemy:
                    enemy_npc_ids_in_target.append(npc_id)

        # build feedback about what the agent hears
        if npc_names_in_target:
            npc_list_str = ", ".join(npc_names_in_target)
            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} pressed "
                f"{env.person_verbalized['possessive_adjective']} ear against the door to {area_name} "
                f"and heard sounds. There seem to be {len(npc_names_in_target)} "
                f"{'creature' if len(npc_names_in_target) == 1 else 'creatures'} "
                f"on the other side: {npc_list_str}.\n"
            )
        else:
            res.add_feedback(
                agent.id,
                f"{env.person_verbalized['subject_pronoun']} pressed "
                f"{env.person_verbalized['possessive_adjective']} ear against the door to {area_name}. "
                f"It's eerily silent on the other side — no signs of life.\n"
            )

        res.events.append(Event(
            type="eavesdrop",
            agent_id=agent.id,
            data={
                "from_area": current_area_id,
                "target_area": target_area_id,
                "area_name": area_name,
                "npcs_heard": npc_names_in_target,
            },
        ))

        # chance of alerting an enemy NPC — it bursts through the door into the current area
        if enemy_npc_ids_in_target and env.rng.random() < self.ALERT_PROBABILITY:
            alerted_npc_id = env.rng.choice(enemy_npc_ids_in_target)
            alerted_npc = world.npc_instances[alerted_npc_id]

            # move the NPC from the target area into the current area
            if alerted_npc_id in target_area.npcs:
                target_area.npcs.remove(alerted_npc_id)
            if alerted_npc_id not in current_area.npcs:
                current_area.npcs.append(alerted_npc_id)

            # initiate combat
            env.curr_agents_state["active_combats"].setdefault(agent.id, {})
            env.curr_agents_state["active_combats"][agent.id][alerted_npc_id] = {
                "rhythm_index": 0,
                "area_id": current_area_id,
            }

            res.add_feedback(
                agent.id,
                f"The noise alerted {alerted_npc.name}! It burst through the door and attacked!\n"
            )

            res.events.append(Event(
                type="eavesdrop_alert",
                agent_id=agent.id,
                data={
                    "npc_id": alerted_npc_id,
                    "from_area": target_area_id,
                    "to_area": current_area_id,
                },
            ))
            res.events.append(Event(
                type="combat_started",
                agent_id=agent.id,
                data={"npc_id": alerted_npc_id, "area_id": current_area_id},
            ))


class BurnCorpsesRule(BaseActionRule):
    name = "action_burn_corpses"
    verb = "burn corpses"
    param_min = 0
    param_max = 0
    params = []
    description = (
        "Burn the corpses in the current area using a chemical and a rag from "
        "hand or inventory. Removes all corpses and resets the area's corruption "
        "level, preventing mutated enemies from spawning."
    )

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        current_area_id = env.curr_agents_state["area"][agent.id]

        area_corpses = env.curr_agents_state.get("area_corpses", {})
        corpse_list = area_corpses.get(current_area_id, [])

        if not corpse_list:
            res.add_feedback(
                agent.id,
                "There are no corpses to burn in this area.\n"
            )
            return

        # --- check for required materials: 1 chemical + 1 rag ---
        chem_id = "obj_chemical"
        rag_id = "obj_rag"

        held_containers = [
            world.container_instances[oid]
            for oid in agent.items_in_hands.keys()
            if oid in world.container_instances
        ]

        def _count(obj_id: str) -> int:
            total = int(agent.items_in_hands.get(obj_id, 0))
            if agent.inventory.container:
                total += int(agent.inventory.items.get(obj_id, 0))
            for ci in held_containers:
                total += int(ci.inventory.get(obj_id, 0))
            return total

        if _count(chem_id) < 1 or _count(rag_id) < 1:
            res.add_feedback(
                agent.id,
                "Cannot burn corpses. You need at least 1 chemical and 1 rag "
                "in your hand, inventory, or held containers.\n"
            )
            return

        # --- consume 1 chemical ---
        remaining = 1
        if remaining > 0 and agent.items_in_hands.get(chem_id, 0) > 0:
            take = min(int(agent.items_in_hands[chem_id]), remaining)
            agent.items_in_hands[chem_id] -= take
            if agent.items_in_hands[chem_id] <= 0:
                del agent.items_in_hands[chem_id]
            res.track_consume(agent.id, chem_id, take, src=res.tloc("hand", agent.id))
            remaining -= take
        if remaining > 0 and agent.inventory.container and agent.inventory.items.get(chem_id, 0) > 0:
            take = min(int(agent.inventory.items[chem_id]), remaining)
            agent.inventory.items[chem_id] -= take
            if agent.inventory.items[chem_id] <= 0:
                del agent.inventory.items[chem_id]
            res.track_consume(agent.id, chem_id, take, src=res.tloc("container", agent.inventory.container.id))
            remaining -= take
        for ci in held_containers:
            if remaining <= 0:
                break
            if ci.inventory.get(chem_id, 0) > 0:
                take = min(int(ci.inventory[chem_id]), remaining)
                ci.inventory[chem_id] -= take
                if ci.inventory[chem_id] <= 0:
                    del ci.inventory[chem_id]
                res.track_consume(agent.id, chem_id, take, src=res.tloc("container", ci.id))
                remaining -= take

        # --- consume 1 rag ---
        remaining = 1
        if remaining > 0 and agent.items_in_hands.get(rag_id, 0) > 0:
            take = min(int(agent.items_in_hands[rag_id]), remaining)
            agent.items_in_hands[rag_id] -= take
            if agent.items_in_hands[rag_id] <= 0:
                del agent.items_in_hands[rag_id]
            res.track_consume(agent.id, rag_id, take, src=res.tloc("hand", agent.id))
            remaining -= take
        if remaining > 0 and agent.inventory.container and agent.inventory.items.get(rag_id, 0) > 0:
            take = min(int(agent.inventory.items[rag_id]), remaining)
            agent.inventory.items[rag_id] -= take
            if agent.inventory.items[rag_id] <= 0:
                del agent.inventory.items[rag_id]
            res.track_consume(agent.id, rag_id, take, src=res.tloc("container", agent.inventory.container.id))
            remaining -= take
        for ci in held_containers:
            if remaining <= 0:
                break
            if ci.inventory.get(rag_id, 0) > 0:
                take = min(int(ci.inventory[rag_id]), remaining)
                ci.inventory[rag_id] -= take
                if ci.inventory[rag_id] <= 0:
                    del ci.inventory[rag_id]
                res.track_consume(agent.id, rag_id, take, src=res.tloc("container", ci.id))
                remaining -= take

        # --- clear corpses and corruption ---
        num_burned = len(corpse_list)
        area_corpses[current_area_id] = []
        env.curr_agents_state.setdefault("area_corruption", {})[current_area_id] = 0
        env.curr_agents_state.setdefault("corruption_spawn_cooldown", {})[current_area_id] = 0

        area_name = world.area_instances[current_area_id].name if current_area_id in world.area_instances else current_area_id
        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} doused the corpses with chemical and "
            f"set them ablaze. {num_burned} corpse(s) in {area_name} have been incinerated. "
            f"The corruption has been purged.\n"
        )

        res.events.append(Event(
            type="corpses_burned",
            agent_id=agent.id,
            data={
                "area_id": current_area_id,
                "num_burned": num_burned,
            },
        ))


class RefuelRule(BaseActionRule):
    name = "action_refuel"
    verb = "refuel"
    param_min = param_max = 1
    params = ["light_source_name"]
    description = "Refuel a light source (lantern or flashlight) held in hand using lantern_fuel from hand, inventory, or held containers. Restores the light source to full fuel."

    def apply(self, ctx: RuleContext, res: RuleResult) -> None:
        env, world, agent = ctx.env, ctx.world, ctx.agent
        light_name = ctx.params[0]

        if light_name not in world.auxiliary["obj_name_to_id"]:
            res.add_feedback(agent.id, f"Cannot refuel {light_name}, not found.\n")
            return
        light_id = world.auxiliary["obj_name_to_id"][light_name]
        def_id = get_def_id(light_id)
        obj_def = world.objects.get(def_id)

        if obj_def is None or not getattr(obj_def, "light_source", False):
            res.add_feedback(agent.id, f"Cannot refuel {light_name}. It is not a light source.\n")
            return

        if light_id not in agent.items_in_hands:
            res.add_feedback(agent.id, f"Cannot refuel {light_name}. It must be held in hand.\n")
            return

        fuel_id = "obj_lantern_fuel"

        # count available fuel
        held_containers = [
            world.container_instances[oid]
            for oid in agent.items_in_hands.keys()
            if oid in world.container_instances
        ]
        fuel_have = int(agent.items_in_hands.get(fuel_id, 0))
        if agent.inventory.container:
            fuel_have += int(agent.inventory.items.get(fuel_id, 0))
        for ci in held_containers:
            fuel_have += int(ci.inventory.get(fuel_id, 0))

        if fuel_have < 1:
            res.add_feedback(
                agent.id,
                f"Cannot refuel {light_name}. No lantern_fuel found in hand, inventory, or held containers.\n"
            )
            return

        # consume 1 fuel item
        remaining = 1
        if remaining > 0 and agent.items_in_hands.get(fuel_id, 0) > 0:
            take = min(int(agent.items_in_hands[fuel_id]), remaining)
            agent.items_in_hands[fuel_id] -= take
            if agent.items_in_hands[fuel_id] <= 0:
                del agent.items_in_hands[fuel_id]
            res.track_consume(agent.id, fuel_id, take, src=res.tloc("hand", agent.id))
            remaining -= take
        if remaining > 0 and agent.inventory.container and agent.inventory.items.get(fuel_id, 0) > 0:
            take = min(int(agent.inventory.items[fuel_id]), remaining)
            agent.inventory.items[fuel_id] -= take
            if agent.inventory.items[fuel_id] <= 0:
                del agent.inventory.items[fuel_id]
            res.track_consume(agent.id, fuel_id, take, src=res.tloc("container", agent.inventory.container.id))
            remaining -= take
        for ci in held_containers:
            if remaining <= 0:
                break
            if ci.inventory.get(fuel_id, 0) > 0:
                take = min(int(ci.inventory[fuel_id]), remaining)
                ci.inventory[fuel_id] -= take
                if ci.inventory[fuel_id] <= 0:
                    del ci.inventory[fuel_id]
                res.track_consume(agent.id, fuel_id, take, src=res.tloc("container", ci.id))
                remaining -= take

        # restore fuel to max
        max_fuel = int(getattr(obj_def, "max_fuel", 30))
        fuel_state = env.curr_agents_state.get("light_fuel", {}).get(agent.id, {})
        fuel_state[light_id] = max_fuel
        env.curr_agents_state.setdefault("light_fuel", {})[agent.id] = fuel_state

        res.add_feedback(
            agent.id,
            f"{env.person_verbalized['subject_pronoun']} refueled {light_name}. It burns brightly again ({max_fuel} fuel).\n"
        )
        res.events.append(Event(
            type="light_source_refueled",
            agent_id=agent.id,
            data={"obj_id": light_id, "fuel": max_fuel},
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
