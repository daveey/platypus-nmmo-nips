import argparse
from pathlib import Path
import os
import wandb

from neurips2022nmmo import CompetitionConfig, RollOut, scripted

from submission import MonobeastBaseline


def rollout(model_path, num_steps, num_trials):
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
    results = ro.run(n_timestep=num_steps, n_episode=num_trials, render=False)
    # print(results[0])

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
        "--steps",
        type=int,
        default=1024
    )
    # wandb settings.
    parser.add_argument("--wandb", action="store_true",
                        help="Log to wandb.")
    parser.add_argument('--group', default='default', type=str, metavar='G',
                        help='Name of the experiment group (as being used by wandb).')
    parser.add_argument('--project', default='platypus-nmmo-nips', type=str, metavar='P',
                        help='Name of the project (as being used by wandb).')
    parser.add_argument('--entity', default='platypus', type=str, metavar='P',
                        help='Which team to log to.')
    parser.add_argument("--xpid", default=None,
                        help="Experiment id (default: None).")


    flags = parser.parse_args()
    if flags.wandb:
        config = dict(vars(flags))
        wandb.init(
            id=flags.xpid, # run ID
            project=flags.project,
            config=config,
            group=flags.group,
            entity=flags.entity,
            resume="allow",
        )

    model = flags.model
    if os.path.isdir(model):
            latest = max([int(m[6:-3]) for m in os.listdir(model) if m.startswith("model_")])
            model = f"{model}/model_{latest}.pt"
    
    rollout(model, flags.steps, flags.trials)
