from collections import defaultdict
from typing import Dict

import numpy as np
from neurips2022nmmo import Metrics

EQUIPMENT = [
    "HatLevel", "BottomLevel", "TopLevel", "HeldLevel", "AmmunitionLevel"
]
PROFESSION = [
    "MeleeLevel", "RangeLevel", "MageLevel", "FishingLevel",     
    'HerbalismLevel',
    'ProspectingLevel',
    'CarvingLevel',
    'AlchemyLevel',
]

GOALS = {
    'PlayerDefeats': 128, 
    'TimeAlive': 1024, 
    'DamageTaken': -1000, 
    'Profession': 10, 
    'MeleeLevel': 10, 
    'RangeLevel': 10, 
    'MageLevel': 10, 
    'FishingLevel': 10,
    'HerbalismLevel': 10,
    'ProspectingLevel': 10,
    'CarvingLevel': 10,
    'AlchemyLevel': 10,
    'HatLevel': 10,
    'TopLevel': 10,
    'BottomLevel': 10,
    'HeldLevel': 10,
    'AmmunitionLevel': 10,
    'MeleeAttack': 100,
    'RangeAttack': 100,
    'MageAttack': 100,
    'MeleeDefense': 100,
    'RangeDefense': 100,
    'MageDefense': 100,
    'Equipment': 10,
    'RationConsumed': 10,
    'PoulticeConsumed': 10,
    'RationLevelConsumed': 10,
    'PoulticeLevelConsumed': 10,
    'Gold': 10,
}

class RewardParser:
    def __init__(self, phase: str = "phase1"):
        assert phase in ["baseline", "team-kill", "randomized"]
        self.phase = phase
        self.best_ever_equip_level = defaultdict(
            lambda: defaultdict(lambda: 0))
        self.reset()

    def reset(self):
        self.best_ever_equip_level.clear()
        if self.phase == "randomized":
            self.goal_weights = np.random.rand(len(GOALS))
            self.team_goal_weights = np.random.rand(len(GOALS))

    def parse(
        self,
        prev_metric: Dict[int, Metrics],
        curr_metric: Dict[int, Metrics],
        obs: Dict[int, Dict[str, np.ndarray]],
        step: int,
        done: Dict[int, bool]
    ) -> Dict[int, float]:

        if self.phase == "baseline":
            return self.baseline_reward(prev_metric, curr_metric, obs, step, done)

        if self.phase == "randomized":
            return self.weighted_reward(prev_metric, curr_metric, obs, step, done)

        if self.phase == "team-kill":
            return self.team_kill_reward(prev_metric, curr_metric, obs, step, done)

        assert False

    def weighted_reward(
        self,
        prev_metric: Dict[int, Metrics],
        curr_metric: Dict[int, Metrics],
        obs: Dict[int, Dict[str, np.ndarray]],
        step: int,
        done: Dict[int, bool]
    ) -> Dict[int, float]:
        reward = {}
        team_reward = {t: 0 for t in range(8)}
        food, water = self.extract_info_from_obs(obs)

        for agent_id in curr_metric:
            curr, prev = curr_metric[agent_id], prev_metric[agent_id]
            deltas = np.array([(curr[k] - prev[k]) / GOALS[k] for k in GOALS])

            reward[agent_id] = sum(deltas * self.goal_weights)
            team_reward[agent_id // 8] = sum(deltas * self.team_goal_weights) / 8

        return {a: reward[a] + team_reward[a//8] for a in reward}

    def team_kill_reward(
        self,
        prev_metric: Dict[int, Metrics],
        curr_metric: Dict[int, Metrics],
        obs: Dict[int, Dict[str, np.ndarray]],
        step: int,
        done: Dict[int, bool]
    ) -> Dict[int, float]:
        team_rewards = {t: 0 for t in range(8)}
        for agent_id in curr_metric:
            team_rewards[agent_id // 8] += float(curr_metric[agent_id]["PlayerDefeats"] - prev_metric[agent_id]["PlayerDefeats"]) / 8
        baseline = self.baseline_reward(prev_metric, curr_metric, obs, step, done)
        return {a: baseline[a] + team_rewards[a // 8] for a in baseline}

    def baseline_reward(
        self,
        prev_metric: Dict[int, Metrics],
        curr_metric: Dict[int, Metrics],
        obs: Dict[int, Dict[str, np.ndarray]],
        step: int,
        done: Dict[int, bool]
    ) -> Dict[int, float]:
        reward = {}
        food, water = self.extract_info_from_obs(obs)
        for agent_id in curr_metric:
            curr, prev = curr_metric[agent_id], prev_metric[agent_id]
            r = 0.0
            # Alive reward
            if curr["TimeAlive"] == 1024:
                r += 10.0
            # Defeats reward
            r += (curr["PlayerDefeats"] - prev["PlayerDefeats"]) * 0.5
            # Profession reward
            for p in PROFESSION:
                r += (curr[p] - prev[p]) * 0.1 * curr[p]
            # Equipment reward
            for e in EQUIPMENT:
                delta = curr[e] - self.best_ever_equip_level[agent_id][e]
                if delta > 0:
                    r += delta * 0.1 * curr[e]
                    self.best_ever_equip_level[agent_id][e] = curr[e]
            # DamageTaken penalty
            r -= (curr["DamageTaken"] - prev["DamageTaken"]) * 0.01
            # Starvation penalty
            if agent_id in food and food[agent_id] == 0:
                r -= 0.1
            if agent_id in water and water[agent_id] == 0:
                r -= 0.1

            # Death penalty
            if agent_id in done and done[agent_id]:
                r -= 5.0

            # Team reward
            reward[agent_id] = r

        return reward

    def extract_info_from_obs(self, obs: Dict[int, Dict[str, np.ndarray]]):
        food = {i: obs[i]["self_entity"][0, 11] for i in obs}
        water = {i: obs[i]["self_entity"][0, 12] for i in obs}
        return food, water
