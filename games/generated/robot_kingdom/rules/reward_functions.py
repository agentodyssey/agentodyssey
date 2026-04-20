from __future__ import annotations
from games.generated.robot_kingdom.rule import RewardBreakdown, RewardFunction, RuleResult, Event
from typing import Dict, Any, TYPE_CHECKING
from tools.logger import get_logger

if TYPE_CHECKING:
    from games.generated.robot_kingdom.env import AgentOdysseyEnv


class DefaultRewardFunction(RewardFunction):
    def compute(
        self,
        env: "AgentOdysseyEnv",
        prev_state: Dict[str, Any],
        res: RuleResult,
    ) -> Dict[str, RewardBreakdown]:

        reward = {
            a.id: RewardBreakdown() for a in env.agents
        }

        tutorial_area_id = (getattr(env, "world", None) or {}).auxiliary.get("tutorial_room", {}).get("area_id") if getattr(env, "world", None) is not None else None

        # exploration, kills, crafting from curr_agents_state diffs
        for agent in env.agents:
            aid = agent.id

            # no reward during tutorial session.
            if tutorial_area_id is not None and env.curr_agents_state["area"].get(aid) == tutorial_area_id:
                continue

            # exploration
            if env.curr_agents_state["area"][aid] not in prev_state["areas_visited"][aid]:
                reward[aid].exploration += 1
                env.curr_agents_state["areas_visited"][aid].append(env.curr_agents_state["area"][aid])

            # kills
            prev_kills = len(prev_state["npcs_killed"][aid])
            curr_kills = len(env.curr_agents_state["npcs_killed"][aid])
            if curr_kills > prev_kills:
                reward[aid].kill += curr_kills - prev_kills

            # unique kills
            prev_kills = len(prev_state["unique_npcs_killed"][aid])
            curr_kills = len(env.curr_agents_state["unique_npcs_killed"][aid])
            if curr_kills > prev_kills:
                reward[aid].unique_kill += curr_kills - prev_kills

            # crafting: new object types crafted
            prev_crafted = prev_state["objects_crafted"][aid]
            curr_crafted = env.curr_agents_state["objects_crafted"][aid]
            new_types = set(curr_crafted.keys()) - set(prev_crafted.keys())
            if new_types:
                reward[aid].craft += len(new_types)
            
            # trade: new object types traded
            prev_traded = prev_state["objects_traded"][aid]
            curr_traded = env.curr_agents_state["objects_traded"][aid]
            new_trade_types = set(curr_traded.keys()) - set(prev_traded.keys())
            if new_trade_types:
                reward[aid].trade += len(new_trade_types)

        # death penalty from events
        for ev in res.events:
            if ev.type == "agent_died" and ev.agent_id in reward:
                reward[ev.agent_id].death += 1

        # quest stage completion reward (main quest)
        for ev in res.events:
            if ev.type == "quest_stage_advanced" and ev.agent_id in reward:
                reward[ev.agent_id].quest += 1

        # side quest completion reward
        for ev in res.events:
            if ev.type == "side_quest_completed" and ev.agent_id in reward:
                reward[ev.agent_id].side_quest += 1
        
        # total + xp update
        for agent in env.agents:
            aid = agent.id
            if tutorial_area_id is not None and env.curr_agents_state["area"].get(aid) == tutorial_area_id:
                continue
            levels_gained = agent.gain_xp(reward[aid].xp_total)
            if levels_gained > 0:
              res.add_feedback(agent.id, f"{env.person_verbalized['subject_pronoun']} leveled up {levels_gained} times!\n")
              res.events.append(Event("agent_leveled_up", agent.id, {"levels_gained": levels_gained}))
            if reward[aid].score_total != 0:
                get_logger("RewardLogger").info(f"Agent {aid} received reward: {reward[aid]} (Total XP: {agent.xp})")

        return reward
