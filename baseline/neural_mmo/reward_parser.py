from collections import defaultdict
from typing import Dict

import numpy as np
from neurips2022nmmo import Metrics

EQUIPMENT = [
    "HatLevel", "BottomLevel", "TopLevel", "HeldLevel", "AmmunitionLevel"
]


class RewardParser:

    def __init__(self):
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
    ) -> Dict[int, float]:
        reward = {}
        death_fog_damage, food, water = self.extract_info_from_obs(obs)
        for agent_id in curr_metric:
            curr, prev = curr_metric[agent_id], prev_metric[agent_id]
            # Defeats reward
            r = (curr["PlayerDefeats"] - prev["PlayerDefeats"]) * 0.1
            # Profession reward
            r += (curr["MeleeLevel"] - prev["MeleeLevel"]) * 0.1
            # Equipment reward
            for e in EQUIPMENT:
                delta = curr[e] - self.best_ever_equip_level[agent_id][e]
                if delta > 0:
                    r += delta * 0.1
                    self.best_ever_equip_level[agent_id][e] = curr[e]
            # Death fog penalty
            if agent_id in death_fog_damage:
                r -= death_fog_damage[agent_id] * 0.5
            reward[agent_id] = r
        return reward

    def extract_info_from_obs(self, obs: Dict[int, Dict[str, np.ndarray]]):
        death_fog_damage = {i: obs[i]["death_fog_damage"][7, 7] for i in obs}
        food = {i: obs[i]["self_entity"][0, 11] for i in obs}
        water = {i: obs[i]["self_entity"][0, 12] for i in obs}
        return death_fog_damage, food, water
