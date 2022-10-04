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
        assert phase in ["phase1", "phase2", "simple"]
        self.phase = phase
        self.best_ever_equip_level = defaultdict(
            lambda: defaultdict(lambda: 0))

    def reset(self):
        self.best_ever_equip_level.clear()

    def parse(
        self,
        prev_metric: Dict[int, Dict[int, Metrics]],
        curr_metric: Dict[int, Dict[int, Metrics]],
        obs: Dict[int, Dict[int, Dict[str, np.ndarray]]],
        step: int,
        done: Dict[int, Dict[int, bool]],
    ) -> Dict[int, float]:
        # reward = {}
        # for agent_id in curr_metric:
        #     if agent_id in obs:
        #         agent_rewards = self._parse_agent(prev_metric[agent_id], curr_metric[agent_id], obs[agent_id], step, done[agent_id])
        #         reward[agent_id] = sum(agent_rewards.values()) / 8
        #     else:
        #         reward[agent_id] = 0

        return {a: float(a in obs) / 1024 for a in curr_metric}

    def _parse_agent(
        self,
        prev_metric: Dict[int, Metrics],
        curr_metric: Dict[int, Metrics],
        obs: Dict[int, Dict[str, np.ndarray]],
        step: int,
        done: int,
    ) -> Dict[int, float]:
        reward = {}
        food, water = self.extract_info_from_obs(obs)
        for body_id in curr_metric:
            curr, prev = curr_metric[body_id], prev_metric[body_id]
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
                delta = curr[e] - self.best_ever_equip_level[body_id][e]
                if delta > 0:
                    r += delta * 0.1 * curr[e]
                    self.best_ever_equip_level[body_id][e] = curr[e]
            # DamageTaken penalty
            r -= (curr["DamageTaken"] - prev["DamageTaken"]) * 0.01
            # Starvation penalty
            if body_id in food and food[body_id] == 0:
                r -= 0.1
            if body_id in water and water[body_id] == 0:
                r -= 0.1

            # phase2 only
            if self.phase == "phase2":
                # Death penalty
                if body_id in done and done[body_id]:
                    r -= 5.0
            reward[body_id] = r
        return reward

    def extract_info_from_obs(self, obs: Dict[int, Dict[str, np.ndarray]]):
        food = {i: obs[i]["self_entity"][0, 11] for i in obs}
        water = {i: obs[i]["self_entity"][0, 12] for i in obs}
        return food, water
