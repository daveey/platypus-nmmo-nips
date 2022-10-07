from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.mask import MaskedPolicy


class ActionHead(nn.Module):
    name2dim = {"move": 5, "attack_target": 16}

    def __init__(self, input_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(input_dim, output_dim)
            for name, output_dim in self.name2dim.items()
        })

    def forward(self, x) -> Dict[str, torch.Tensor]:
        out = {name: self.heads[name](x) for name in self.name2dim}
        return out


class NMMONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_bodies = 8

        self.map_embedding = torch.nn.Linear(15*15*8*(16+6+2), 64)        
        self.entity_embedding = torch.nn.Linear(8*16*26, 64)        

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU())

        self.action_head = ActionHead(64)
        self.value_head = nn.Linear(64, 1)

    def forward(
        self,
        input_dict: Dict,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        T, B, *_ = input_dict["terrain"].shape
        terrain = input_dict["terrain"]
        death_fog_damage = input_dict["death_fog_damage"]
        reachable = input_dict["reachable"]
        population = input_dict["entity_population"]
        self_entity = input_dict["self_entity"]
        other_entity = input_dict["other_entity"]

        terrain = F.one_hot(terrain, num_classes=16)
        population = F.one_hot(population, num_classes=6)
        death_fog_damage = death_fog_damage.unsqueeze(dim=-1)
        reachable = reachable.unsqueeze(dim=-1)

        map = torch.cat(
            [terrain, reachable, population, death_fog_damage], dim=-1)
        map = self.map_embedding(map.view(T, B, -1).to(torch.float))
        entities = self.entity_embedding(
            torch.cat([self_entity, other_entity], 3)
            .view(T, B, -1)
        )

        obs = torch.cat([map, entities], 2)
        state = self.fc(obs)

        logits = self.action_head(state)
        value = self.value_head(state).view(T, B)

        output = {"value": value}
        for key, val in logits.items():
            if not training:
                dist = MaskedPolicy(val, input_dict[f"va_{key}"])
                action = dist.sample()
                logprob = dist.log_prob(action)
                output[key] = action
                output[f"{key}_logp"] = logprob
            else:
                output[f"{key}_logits"] = val
        return output
