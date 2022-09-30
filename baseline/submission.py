from pathlib import Path
from typing import Dict
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import tree
from neurips2022nmmo import Team
from nmmo import config

from monobeast import batch, unbatch
from neural_mmo import FeatureParser, MyMeleeTeam, NMMONet, TrainEnv


class MonobeastBaseline(Team):
    def __init__(self,
                 team_id: str,
                 env_config: config.Config,
                 checkpoint_path=None):
        super().__init__(team_id, env_config)
        self._num_team_members = 8
        self._dummy_body_feature = {}
        self.feature_parser = FeatureParser()

        for key, val in self.feature_parser.spec.items():
            self._dummy_body_feature[key] = np.zeros(shape=val.shape[1:], dtype=val.dtype)

        self.model: nn.Module = NMMONet()
        if checkpoint_path is not None:
            print(f"load checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.reset()

    def reset(self):
        self.my_script = {0: MyMeleeTeam("MyMelee", self.env_config)}
        self.step = 0

    def compute_actions(
        self,
        observations: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict]:
        feature = self._obs_team_to_mb_agent(
            self.feature_parser.parse(observations, self.step))
        
        feature = tree.map_structure(
            lambda x: torch.from_numpy(x).view(1, 1, *x.shape), feature)
        output = self.model(feature, training=False)

        actions = {}
        for i in observations:
            actions[i] = {
                "move": output["move"][:,:,i].item(),
                "attack_target": output["attack_target"][:,:,i].item()
            }

        actions = TrainEnv.transform_action({0: actions}, {0: observations},
                                            self.my_script)
        return actions[0]

    def act(
        self,
        observations: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict]:
        self.step += 1
        if "stat" in observations:
            stat = observations.pop("stat")
        actions = self.compute_actions(observations)
        return actions

    def _obs_team_to_mb_agent(self, team_obs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        obs = {}
        for pid in range(self._num_team_members):
            pobs = team_obs.get(pid, self._dummy_body_feature)
            for k,v in pobs.items():
                if k not in obs:
                    obs[k] = []
                obs[k].append(v)
        for k in obs:
            obs[k] = np.stack(obs[k])

        return obs


class Submission:
    team_klass = MonobeastBaseline
    init_params = {
        "checkpoint_path":
        Path(__file__).parent / "checkpoints" / "model_2757376.pt"
    }
