import argparse
from pathlib import Path
import os
import wandb

from neurips2022nmmo import CompetitionConfig, RollOut, scripted

from submission import MonobeastBaseline


def rollout(model_path, timesteps, num_trials):
    config = CompetitionConfig()
    config.RENDER = False
    config.SAVE_REPLAY = False
    my_team = MonobeastBaseline(team_id=f"my-team",
                                env_config=config,
                                checkpoint_path=model_path)
    all_teams = [scripted.CombatTeam(f"C-{i}", config) for i in range(5)]
    all_teams.extend(
        [scripted.MixtureTeam(f"M-{i}", config) for i in range(10)])
    all_teams.append(my_team)
    ro = RollOut(config, all_teams, parallel=True)
    results = ro.run(n_timestep=timesteps, n_episode=num_trials, render=False)
    # print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1
    )    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1024
    )    
    args = parser.parse_args()

    model = args.model
    if os.path.isdir(model):
            latest = max([int(m[6:-3]) for m in os.listdir(model) if m.startswith("model_")])
            model = f"{model}/model_{latest}.pt"
    
    rollout(model, args.timesteps, args.trials)
