from collections import defaultdict
from random import random
from typing import Dict

import numpy as np
from neurips2022nmmo import Metrics

EQUIPMENT = [
    "HatLevel", "BottomLevel", "TopLevel", "HeldLevel", "AmmunitionLevel"
]
PROFESSION = [
    "FishingLevel",     
    'HerbalismLevel',
    'ProspectingLevel',
    'CarvingLevel',
    'AlchemyLevel',
]

ATTACK = [
    "MeleeLevel", 
    "RangeLevel", 
    "MageLevel", 
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
    def __init__(self, num_team_members, phase):
        self.num_team_members = num_team_members
        self.phase = phase
        self.best_ever_equip_level = defaultdict(
            lambda: defaultdict(lambda: 0))
        
        self.reward_weights = {
            "team": 0,
            "kill": 0.5,
            "dt": 0.01,
            "di": 0.01,
            "prox": 0.01
        }
        for rw in self.phase.split(","):
            n,v = rw.split(":")
            assert n in self.reward_weights, f"reward {n} not in {self.reward_weights.keys()}"
            self.reward_weights[n] = float(v)

        print(f"reward settings: ", self.reward_weights)
        self.reset()

    def reset(self):
        self.best_ever_equip_level.clear()

    def parse(
        self,
        prev_metric: Dict[int, Metrics],
        curr_metric: Dict[int, Metrics],
        obs: Dict[int, Dict[str, np.ndarray]],
        step: int,
        done: Dict[int, bool]
    ) -> Dict[int, float]:

        return self.team_reward(prev_metric, curr_metric, obs, step, done)

        assert False, "invalid --reward_setting"

    def team_reward(
        self,
        prev_metric: Dict[int, Metrics],
        curr_metric: Dict[int, Metrics],
        obs: Dict[int, Dict[str, np.ndarray]],
        step: int,
        done: Dict[int, bool]
    ) -> Dict[int, float]:
        team_rewards = {}
        baseline = self.baseline_reward(prev_metric, curr_metric, obs, step, done)

        for agent_id in curr_metric:
            if agent_id not in done:
                continue

            tid = agent_id // self.num_team_members
            team_rewards[tid] = team_rewards.get(tid, 0) + baseline[agent_id]
            proximity_rewards = {}
            
            for ally in range(self.num_team_members):
                ally_id = tid * self.num_team_members + ally
                distance = (
                    abs(obs[ally_id]["self_entity"][0, 4] - 
                        obs[agent_id]["self_entity"][0, 4]) + 
                    abs(obs[ally_id]["self_entity"][0, 5] - 
                        obs[agent_id]["self_entity"][0, 5]))
                if ally_id != agent_id and distance < 0.1:
                    proximity_rewards[agent_id] = proximity_rewards.get(agent_id, 0) + (
                        self.reward_weights["prox"] * baseline[ally_id])

        return { a: (1-self.reward_weights["team"]) * baseline[a] + 
                    self.reward_weights["team"] * team_rewards.get(a // self.num_team_members, 0) / self.num_team_members
                for a in baseline }

    def baseline_reward(
        self,
        prev_metric: Dict[int, Metrics],
        curr_metric: Dict[int, Metrics],
        obs: Dict[int, Dict[str, np.ndarray]],
        step: int,
        done: Dict[int, bool],
    ) -> Dict[int, float]:
        reward = {}
        food, water = self.extract_info_from_obs(obs)
        for agent_id in curr_metric:

            # skip over dead agents
            if agent_id not in done:
                reward[agent_id] = 0
                continue

            curr, prev = curr_metric[agent_id], prev_metric[agent_id]
            r = 0.0
            # Alive reward
            if curr["TimeAlive"] == 1024:
                r += 10.0
            # Defeats reward
            r += (curr["PlayerDefeats"] - prev["PlayerDefeats"]) * self.reward_weights["kill"]

            # Damage reward
            r += (curr["DamageInflicted"] - prev["DamageInflicted"]) * self.reward_weights["di"]

            # Profession reward
            for p in PROFESSION:
                pr = (curr[p] - prev[p]) * 0.1 * curr[p]
                r += pr

            # Combat reward
            for p in ATTACK:
                r += (curr[p] - prev[p]) * 0.1 * curr[p]

            # Equipment reward
            for e in EQUIPMENT:
                delta = curr[e] - self.best_ever_equip_level[agent_id][e]
                if delta > 0:
                    r += delta * 0.1 * curr[e]
                    self.best_ever_equip_level[agent_id][e] = curr[e]
            # DamageTaken penalty
            r -= (curr["DamageTaken"] - prev["DamageTaken"]) * self.reward_weights["dt"]

            # Starvation penalty
            if agent_id in food and food[agent_id] == 0:
                r -= 0.1
            if agent_id in water and water[agent_id] == 0:
                r -= 0.1


            # Death penalty
            if agent_id in done and done[agent_id]:
                r -= 5.0

            reward[agent_id] = r

        return reward

    def extract_info_from_obs(self, obs: Dict[int, Dict[str, np.ndarray]]):
        food = {i: obs[i]["self_entity"][0, 11] for i in obs}
        water = {i: obs[i]["self_entity"][0, 12] for i in obs}
        return food, water
