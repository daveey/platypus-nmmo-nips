from pdb import set_trace as T

from nmmo import Task
from collections import namedtuple

class Tier:
    REWARD_SCALE = 15
    EASY         = 4  / REWARD_SCALE
    NORMAL       = 6  / REWARD_SCALE
    HARD         = 11 / REWARD_SCALE

def player_kills(realm, player):
    return player.history.playerKills

# def equipment(realm, player):
#     return player.loadout.defense

def exploration(realm, player):
    return player.history.exploration

# def foraging(realm, player):
#     return (player.skills.fishing.level + 
#             player.skills.herbalism.level +
#             player.skills.prospecting.level +
#             player.skills.carving.level +
#             player.skills.alchemy.level 
#         ) / 5.0

PlayerKills = [
        Task(player_kills, 1, Tier.EASY),
        Task(player_kills, 3, Tier.NORMAL),
        Task(player_kills, 6, Tier.HARD)]

# Equipment = [
#         Task(equipment, 1,  Tier.EASY),
#         Task(equipment, 10, Tier.NORMAL),
#         Task(equipment, 20, Tier.HARD)]

Exploration = [
        Task(exploration, 32,  Tier.EASY),
        Task(exploration, 64,  Tier.NORMAL),
        Task(exploration, 127, Tier.HARD)]

# Foraging = [
#         Task(foraging, 20, Tier.EASY),
#         Task(foraging, 35, Tier.NORMAL),
#         Task(foraging, 50, Tier.HARD)]

All = PlayerKills + Exploration #+ Foraging
r = 0.001
Train = [
    Task(exploration, i, r*5) for i in range(1, 128, 10)] + [
    # Task(foraging, i, r*2) for i in range(1, 50)] + [
    # Task(equipment, i, r*5) for i in range(1, 20)] + [
    Task(player_kills, i, r*10) for i in range(1, 10)]
