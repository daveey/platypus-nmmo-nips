from data_process import ReplayParser
import argparse

parser = argparse.ArgumentParser(description="Prepare datasets of imitation learning")
parser.add_argument("--replays_dir", default="./dataset/replays")
parser.add_argument("--npy_save_dir", default="./dataset/npy")
parser.add_argument("--num_workers", default=4, type=int)


if __name__ == "__main__":
    flags = parser.parse_args()
    ReplayParser.parse_replay_files(flags.replays_dir, 
                                    flags.npy_save_dir,
                                    num_workers=flags.num_workers)
    print(f"Shuffle samples...")
    ReplayParser.simple_shuffle_samples(flags.npy_save_dir,
                                        num_workers=flags.num_workers)
    