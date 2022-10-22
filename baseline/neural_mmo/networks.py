from tkinter import HIDDEN
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.mask import MaskedPolicy


class ActionHead(nn.Module):
    name2dim = {
        "move": 5, 
        "attack_target": 16, 
        "attack_style": 3, 
        "use_target": 170, 
        "sell_target": 170, 
        "buy_target": 170, 
        "sell_price": 6,
        "send_token": 170,
    }

    def __init__(self, input_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            ) for name, output_dim in self.name2dim.items()
        })

    def forward(self, x) -> Dict[str, torch.Tensor]:
        out = {name: self.heads[name](x) for name in self.name2dim}
        return out


# class NMMONet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.num_bodies = 1
#         self.map_embedding = torch.nn.Linear(15*15*(16+6+2), 64)        
#         self.entity_embedding = torch.nn.Linear(16*26, 64)    
#         self.item_embedding = torch.nn.Linear(25*14, 64)    

#         self.fc = nn.Sequential(
#             nn.Linear(self.num_bodies * 4*64, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU())

#         self.action_head = ActionHead(64)
#         self.value_head = nn.Linear(64, 1)

#     def forward(
#         self,
#         input_dict: Dict,
#         training: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         T, B, *_ = input_dict["terrain"].shape
#         terrain = input_dict["terrain"]
#         death_fog_damage = input_dict["death_fog_damage"]
#         reachable = input_dict["reachable"]
#         population = input_dict["entity_population"]
#         self_entity = input_dict["self_entity"]
#         other_entity = input_dict["other_entity"]
#         items = input_dict["items"]
#         market = input_dict["market"]

#         terrain = F.one_hot(terrain, num_classes=16)
#         population = F.one_hot(population, num_classes=6)
#         death_fog_damage = death_fog_damage.unsqueeze(dim=-1)
#         reachable = reachable.unsqueeze(dim=-1)

#         map = torch.cat(
#             [terrain, reachable, population, death_fog_damage], dim=-1)
#         map = self.map_embedding(map.view(T, B * self.num_bodies, -1).to(torch.float))
#         entities = self.entity_embedding(
#             torch.cat([self_entity, other_entity], 2)
#             .view(T, B * self.num_bodies, -1)
#         )
#         items = self.item_embedding(items.view(T, B * self.num_bodies, -1))
#         market = self.item_embedding(market.view(T, B * self.num_bodies, -1))

#         obs = torch.cat([map, entities, items, market], 2).view(T, B, -1)
#         state = self.fc(obs).view(T, B, -1)

#         logits = self.action_head(state)
#         value = self.value_head(state).view(T, B)

#         output = {"value": value}
#         for key, val in logits.items():
#             if not training:
#                 dist = MaskedPolicy(val, input_dict[f"va_{key}"])
#                 action = dist.sample()
#                 logprob = dist.log_prob(action)
#                 output[key] = action
#                 output[f"{key}_logp"] = logprob
#             else:
#                 output[f"{key}_logits"] = val
#         return output

class NMMONet(nn.Module):
    def __init__(self, num_lstm_layers):
        super().__init__()
        self.num_lstm_layers = num_lstm_layers
        self.latent_size = 256

        self.local_map_cnn = nn.Sequential(
            nn.Conv2d(24, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.local_map_fc = nn.Linear(32 * 4 * 4, 64)

        self.self_entity_fc1 = nn.Linear(27, 32)
        self.self_entity_fc2 = nn.Linear(32, 32)

        self.other_entity_fc1 = nn.Linear(27, 32)
        self.other_entity_fc2 = nn.Linear(15 * 32, 32)

        self.item_fc1 = nn.Linear(14, 32)
        self.item_fc2 = nn.Linear(25*32, 32)

        # self.team_memory_net = nn.Sequential(
        #     nn.Linear(8*2*128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        # )

        self.embeddings = [
            self.local_map_fc, 
            self.self_entity_fc2,
            self.other_entity_fc2, 
            self.item_fc2, # inventory
            self.item_fc2, # market
            # self.team_memory_net
        ]
        self.embedding_size = sum([e.out_features for e in self.embeddings])

        self.fc1 = nn.Linear(self.embedding_size, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, self.latent_size)

        if self.num_lstm_layers > 0:
            self.lstm = nn.LSTM(self.latent_size, self.latent_size, num_layers=self.num_lstm_layers)
        
        self.action_head = ActionHead(self.latent_size)
        self.value_head = nn.Linear(self.latent_size, 1)

    def initial_state(self, batch_size=1):
        if self.num_lstm_layers > 0:
            return torch.zeros(batch_size, 2, self.lstm.num_layers, self.lstm.hidden_size)
        return torch.zeros(batch_size, 2, 1, 256)

    def local_map_embedding(self, input_dict):
        terrain = input_dict["terrain"]
        death_fog_damage = input_dict["death_fog_damage"]
        reachable = input_dict["reachable"]
        population = input_dict["entity_population"]

        T, B, *_ = terrain.shape

        terrain = F.one_hot(terrain, num_classes=16).permute(0, 1, 4, 2, 3)
        population = F.one_hot(population,
                               num_classes=6).permute(0, 1, 4, 2, 3)
        death_fog_damage = death_fog_damage.unsqueeze(dim=2)
        reachable = reachable.unsqueeze(dim=2)
        local_map = torch.cat(
            [terrain, reachable, population, death_fog_damage], dim=2)

        local_map = torch.flatten(local_map, 0, 1).to(torch.float32)
        local_map_emb = self.local_map_cnn(local_map)
        local_map_emb = local_map_emb.view(T, B, -1)
        local_map_emb = F.relu(self.local_map_fc(local_map_emb))

        return local_map_emb

    def entity_embedding(self, input_dict):
        self_entity = input_dict["self_entity"]
        other_entity = input_dict["other_entity"]

        T, B, *_ = self_entity.shape

        self_entity_emb = F.relu(self.self_entity_fc1(self_entity))
        self_entity_emb = self_entity_emb.view(T, B, -1)
        self_entity_emb = F.relu(self.self_entity_fc2(self_entity_emb))

        other_entity_emb = F.relu(self.other_entity_fc1(other_entity))
        other_entity_emb = other_entity_emb.view(T, B, -1)
        other_entity_emb = F.relu(self.other_entity_fc2(other_entity_emb))

        return self_entity_emb, other_entity_emb

    def item_embedding(self, item):
        T, B, *_ = item.shape

        item_emb = F.relu(self.item_fc1(item))
        item_emb = item_emb.view(T, B, -1)
        item_emb = F.relu(self.item_fc2(item_emb))
 
        return item_emb

    def forward(
        self,
        input_dict: Dict,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        T, B, *_ = input_dict["terrain"].shape
        local_map_emb = self.local_map_embedding(input_dict)
        self_entity_emb, other_entity_emb = self.entity_embedding(input_dict)
        items = self.item_embedding(input_dict["items"])
        market = self.item_embedding(input_dict["market"])

        # team_memory = input_dict["team_memory"].float().view(T, B, -1)
        # team_memory = self.team_memory_net(team_memory)

        # goal = F.relu(self.goal_fc1(input_dict["goal"].float().view(T,B, -1)))
        # goal = F.relu(self.goal_fc2(goal))

        x = torch.cat([
            local_map_emb, self_entity_emb, other_entity_emb, 
            items, market], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))


        lstm_input = x.view(T, B, -1)
        lstm_output = lstm_input
        lstm_state = input_dict["lstm_state"]
        if self.num_lstm_layers > 0:
            # [B, 2, NL, S] -> [2, NL, B, S]
            lstm_state = lstm_state.permute(1, 2, 0, 3)

            notdone = (~input_dict["done"]).float()
            notdone = torch.cat([notdone, torch.ones(1, B)])
            lstm_output_list = []
            for input, nd in zip(lstm_input.unbind(), notdone.unbind()):
                # Reset lstm state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                lstm_state = tuple(nd * s for s in lstm_state)
                output, lstm_state = self.lstm(input.unsqueeze(0), lstm_state)
                lstm_output_list.append(output)
            lstm_output = torch.flatten(torch.cat(lstm_output_list), 0, 1)
            lstm_state = torch.stack(lstm_state, 0).permute(2, 0, 1, 3)

        logits = self.action_head(lstm_output)
        value = self.value_head(lstm_output)

        output = {
            "value": value.view(T, B),
            "lstm_state": lstm_state
        }

        for key, val in logits.items():
            if not training:
                dist = MaskedPolicy(val, input_dict[f"va_{key}"])
                action = dist.sample()
                logprob = dist.log_prob(action)
                output[key] = action
                output[f"{key}_logp"] = logprob.view(T, B)
            else:
                output[f"{key}_logits"] = val.view(T, B, -1)

        return output
