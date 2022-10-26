from pathlib import Path
from typing import Dict

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
        env_config.NMAP = 1
        print(f"load checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model: nn.Module = NMMONet(num_lstm_layers=checkpoint["flags"]["lstm_layers"])
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.feature_parser = FeatureParser()
        self.reset()

    def reset(self):
        self.my_script = {0: MyMeleeTeam("MyMelee", self.env_config)}
        self.step = 0
        self.memory = {}

    def log_self(self, features):
        se = features["self_entity"][0][0][0]
        print(f"me: {100*se[11]:.2f}hp {100*se[12]:.2f}f {100*se[13]:.2f}w {se[9]:.2f}d ${se[10]:.2f} {100*se[1]:.2f}/ {100*se[2]:.2f}lvl")

    def log_items(self, feature):
        for idx, item in enumerate(feature["items"][0][0][:-1]):
            if item[2] > 0:
                equiped = "e" if (item[13] > 0) else ""
                trade = "t" if (item[3] > 0) else ""
                print(f"i: {idx} l={item[0]} q=({item[2]}) ${1000*item[12]:.2f} {equiped} {trade}")

    def log_market(self, feature):
        for idx, item in enumerate(feature["market"][0][0][0][:-1]):
            if item[2] > 0:
                print(f"m: {idx} l={item[0]} q=({item[2]}) ${1000*item[12]:.2f}")

    def compute_actions(
        self,
        observations: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict]:
        feature = self.feature_parser.parse(observations, self.step)
        feature = tree.map_structure(
            lambda x: torch.from_numpy(x).view(1, 1, *x.shape), feature)
        
        for a in feature:
            feature[a]["lstm_state"] = self.memory.get(a, self.model.initial_state())
            feature[a]["done"] = torch.zeros(1,1).bool()
        feature_batch, ids = batch(feature, list(self.feature_parser.spec.keys()) + ["lstm_state", "done"])
        output = self.model(feature_batch, training=False)
        output = unbatch(output, ids)

        # self.log_self(feature[0])
        # self.log_items(feature[0])
        # self.log_market(feature[0])

        actions = {}
        for i, out in output.items():
            actions[i] = {
                "move": out["move"].item(),
                "buy_target": out["buy_target"].item(),
                "sell_target": out["sell_target"].item(),
                "sell_price": out["sell_price"].item(),
                "use_target": out["use_target"].item(),
                "send_token": out["send_token"].item(),
                "attack_target": out["attack_target"].item(),
                "attack_style": out["attack_style"].item()
            }
            self.memory[i] = out["lstm_state"]

        # print("actions", actions[0])
        actions = TrainEnv.transform_action({0: actions}, {0: observations},
                                            self.my_script)
        return actions[0]

    @torch.no_grad()
    def act(
        self,
        observations: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, Dict]:
        self.step += 1
        if "stat" in observations:
            stat = observations.pop("stat")
        
        # for k in ["DamageInflicted", "DamageTaken"]:
        #     for a,s in stat.items():
        #         if s[k] != 0:
        #             print("xcxc", a, k, s[k])
        actions = self.compute_actions(observations)
        return actions


class Submission:
    team_klass = MonobeastBaseline
    init_params = {
        "checkpoint_path": "my-submission/model.pt"
    }
