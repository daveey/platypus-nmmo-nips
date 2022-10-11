from collections import defaultdict
from typing import Dict

import numpy as np
from neurips2022nmmo import Metrics

EQUIPMENT = [
    "HatLevel", "BottomLevel", "TopLevel", "HeldLevel", "AmmunitionLevel"
]
PROFESSION = ["MeleeLevel"]


class RewardParser:
    def __init__(self, phase: str = "phase1"):
        assert phase in ["phase1", "phase2", "team", "life"]
        self.phase = phase
        self.best_ever_equip_level = defaultdict(
            lambda: defaultdict(lambda: 0))

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

        if self.phase == "life":
            return {a: float(step) / 1024 for a in obs}

        reward = {}
        team_reward = { t: 0 for t in range(8) }
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

            # phase2 only
            if self.phase == "phase2":
                # Death penalty
                if agent_id in done and done[agent_id]:
                    r -= 5.0
            reward[agent_id] = r

            if agent_id in done and not done[agent_id]:
                team_reward[agent_id // 8] += r
    

        if self.phase == "team":
            return {
                a: reward[a] / 10 + team_reward[a // 8] / 8 for a in reward
            }
        return reward

    def extract_info_from_obs(self, obs: Dict[int, Dict[str, np.ndarray]]):
        food = {i: obs[i]["self_entity"][0, 11] for i in obs}
        water = {i: obs[i]["self_entity"][0, 12] for i in obs}
        return food, water
