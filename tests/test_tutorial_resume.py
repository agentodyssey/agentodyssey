import importlib
import json
import random
import unittest
from types import SimpleNamespace


GAMES = ("base", "remnant", "mark", "metropolis", "quarantine", "robot_kingdom", "saltglass")


class TutorialResumeTests(unittest.TestCase):
    def make_scenario(self, game):
        prefix = "games.base" if game == "base" else f"games.generated.{game}"
        module = importlib.import_module(f"{prefix}.rules.step_rules.tutorial")
        world_module = importlib.import_module(f"{prefix}.world")
        env_class = importlib.import_module(f"{prefix}.env").AgentOdysseyEnv
        bag_base = "obj_small_pouch" if game == "mark" else "obj_small_bag"
        objects = [
            {"id": bag_base, "name": bag_base[4:], "category": "container", "usage": "store", "capacity": 20},
            {"id": "obj_paper", "name": "paper", "category": "tool", "usage": "writable", "max_text_length": 200},
            {"id": "obj_pen", "name": "pen", "category": "tool", "usage": "write"},
        ]
        world = world_module.World(objects={
            obj["id"]: world_module.World._parse_object({"type": "object", "size": 1, "level": 1, **obj})
            for obj in objects
        })
        world.area_instances["area_spawn"] = module.Area(
            type="area", id="area_spawn", name="spawn", light=True, level=1,
        )
        world.place_instances["place_spawn"] = module.Place(
            type="place", id="place_spawn", name="Spawn", unlocked=True, areas=["area_spawn"],
        )
        world.auxiliary["area_to_place"] = {"area_spawn": "place_spawn"}
        rule = module.TutorialRoomStepRule()
        external_bag, _ = rule._create_instance(world, bag_base)
        external_paper, _ = rule._create_instance(world, "obj_paper")
        world.writable_instances[external_paper].text = "Main-world note"
        world.area_instances["area_spawn"].objects = {external_bag: 1, external_paper: 1}
        agent = module.Agent("test_agent", "Test Agent")
        env = SimpleNamespace(
            world=world, agents=[agent], rng=random.Random(42),
            world_definition={"custom_events": ["tutorial"], "initializations": {"spawn": {"area": "area_spawn"}}},
            curr_agents_state={"area": {agent.id: "area_spawn"}},
        )
        scenario = SimpleNamespace(
            env=env, rule=rule, module=module, env_class=env_class, world_class=world_module.World,
            agent=agent, external_bag=external_bag, external_paper=external_paper,
        )
        self.tick(scenario)
        return scenario

    def tick(self, scenario):
        result = scenario.module.RuleResult(feedback={}, info_flags={})
        scenario.rule.apply(scenario.module.RuleContext(
            env=scenario.env, world=scenario.env.world, agent=None, action="", params=[], step_index=0,
        ), result)
        return result

    def resume(self, scenario, legacy=False):
        # Round-trip the same world, agent, and progress data stored in checkpoints.
        checkpoint = json.loads(json.dumps({
            "world": scenario.env.world.to_dict(),
            "agent": scenario.agent.to_dict(),
            "state": scenario.env.curr_agents_state,
        }))
        if legacy:
            tutorial = checkpoint["world"]["auxiliary"]["tutorial_room"]
            tutorial.pop("bag_inst_id", None)
            tutorial.pop("writable_inst_id", None)
        scenario.env.world = scenario.world_class.from_dict(checkpoint["world"])
        scenario.agent = scenario.module.Agent("test_agent", "Test Agent")
        scenario.agent.from_dict(checkpoint["agent"], scenario.env.world.container_instances)
        scenario.env.agents = [scenario.agent]
        scenario.env.curr_agents_state = checkpoint["state"]
        scenario.rule = scenario.module.TutorialRoomStepRule()
        self.tick(scenario)

    def set_stage(self, scenario, step_id):
        stage = next(i for i, step in enumerate(scenario.rule.STEPS) if step["id"] == step_id)
        scenario.rule._stage[scenario.agent.id] = stage
        scenario.env.curr_agents_state["tutorial_stage"][scenario.agent.id] = stage

    def finish_and_check_world(self, scenario):
        bag_id = scenario.rule._bag_inst_id
        paper_id = scenario.rule._writable_inst_id
        final_stage = len(scenario.rule.STEPS) - 1
        scenario.rule._stage[scenario.agent.id] = final_stage
        scenario.env.curr_agents_state["tutorial_stage"][scenario.agent.id] = final_stage
        scenario.env.curr_agents_state["area"][scenario.agent.id] = "area_spawn"
        result = self.tick(scenario)
        world = scenario.env.world
        self.assertTrue(any(event.type == "tutorial_completed" for event in result.events))
        self.assertTrue(world.auxiliary["tutorial_room"]["removed"])
        self.assertNotIn(bag_id, world.container_instances)
        self.assertNotIn(paper_id, world.writable_instances)
        self.assertIn(scenario.external_bag, world.container_instances)
        self.assertEqual(world.writable_instances[scenario.external_paper].text, "Main-world note")
        self.assertEqual(world.area_instances["area_spawn"].objects, {
            scenario.external_bag: 1, scenario.external_paper: 1,
        })
        rendered = scenario.env_class.verbalize_objects(scenario.env, world.area_instances["area_spawn"].objects)
        self.assertIn(world.writable_instances[scenario.external_paper].name, rendered)
        for area in world.area_instances.values():
            for oid in area.objects:
                self.assertTrue(oid in world.objects or oid in world.container_instances or oid in world.writable_instances, oid)

    def test_resume_before_paper_is_placed_preserves_instance_ids(self):
        for game in GAMES:
            with self.subTest(game=game):
                scenario = self.make_scenario(game)
                original_bag = scenario.rule._bag_inst_id
                original_paper = scenario.rule._writable_inst_id
                room = scenario.env.world.area_instances[scenario.rule._tutorial_area_id]
                self.assertNotIn(original_paper, room.objects)
                self.resume(scenario)
                self.assertEqual(scenario.rule._bag_inst_id, original_bag)
                self.assertEqual(scenario.rule._writable_inst_id, original_paper)
                self.set_stage(scenario, "pick_writable_instance")
                self.tick(scenario)
                self.finish_and_check_world(scenario)

    def test_legacy_resume_before_paper_is_placed_uses_a_fresh_instance(self):
        for game in GAMES:
            with self.subTest(game=game):
                scenario = self.make_scenario(game)
                original_bag = scenario.rule._bag_inst_id
                original_paper = scenario.rule._writable_inst_id
                self.resume(scenario, legacy=True)
                self.assertEqual(scenario.rule._bag_inst_id, original_bag)
                self.assertNotIn(scenario.rule._writable_inst_id, (original_paper, scenario.external_paper))
                recovered_paper = scenario.rule._writable_inst_id
                instance_count = len(scenario.env.world.writable_instances)
                self.resume(scenario)
                self.assertEqual(scenario.rule._writable_inst_id, recovered_paper)
                self.assertEqual(len(scenario.env.world.writable_instances), instance_count)
                self.set_stage(scenario, "pick_writable_instance")
                self.tick(scenario)
                self.finish_and_check_world(scenario)

    def test_resume_preserves_placed_and_carried_items(self):
        for game in GAMES:
            for legacy in (False, True):
                for location in ("ground", "hands", "inventory", "held_container", "ground_container"):
                    with self.subTest(game=game, legacy=legacy, location=location):
                        scenario = self.make_scenario(game)
                        world, agent, rule = scenario.env.world, scenario.agent, scenario.rule
                        bag_id, paper_id = rule._bag_inst_id, rule._writable_inst_id
                        room = world.area_instances[rule._tutorial_area_id]
                        room.objects.pop(bag_id)
                        bag = world.container_instances[bag_id]
                        if location in ("held_container", "ground_container"):
                            if location == "held_container":
                                agent.items_in_hands[bag_id] = 1
                            else:
                                room.objects[bag_id] = 1
                            bag.inventory[paper_id] = 1
                        else:
                            agent.inventory.container = bag
                            agent.equipped_items_in_limb[bag_id] = 1
                            if location == "ground":
                                room.objects[paper_id] = 1
                            elif location == "hands":
                                agent.items_in_hands[paper_id] = 1
                            else:
                                bag.inventory[paper_id] = 1
                        world.writable_instances[paper_id].text = "hello"
                        self.set_stage(scenario, "write_writable")
                        self.resume(scenario, legacy=legacy)
                        self.assertEqual(scenario.rule._bag_inst_id, bag_id)
                        self.assertEqual(scenario.rule._writable_inst_id, paper_id)
                        self.assertEqual(scenario.env.world.writable_instances[paper_id].text, "hello")
                        self.finish_and_check_world(scenario)


if __name__ == "__main__":
    unittest.main()
