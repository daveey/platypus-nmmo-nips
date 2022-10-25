import argparse
from pathlib import Path

from neurips2022nmmo import CompetitionConfig, RollOut, scripted

from submission import MonobeastBaseline


def rollout(model_path, num_trials):
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
    ro.run(n_episode=num_trials, render=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1
    )    
    args = parser.parse_args()
    rollout(args.model, args.num_trials)
