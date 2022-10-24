import argparse
import math
import os
import random
import time

import nmmo
import numpy as np
import torch
import tree
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from neural_mmo import FeatureParser, NMMONet, networks

parser = argparse.ArgumentParser(description="Imitation learning for nmmo")
# Training settings.
parser.add_argument("--batch_size", default=2048, type=int)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--n_epoch", default=1000, type=int)
parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA.")
parser.add_argument("--checkpoint_path",
                    default=None,
                    type=str,
                    help="Load previous checkpoint to continue training")
parser.add_argument("--npy_save_dir", default="./dataset/npy")
# Optimizer settings.
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    metavar="LR",
                    help="Learning rate.")


class NMMODataset(IterableDataset):

    def __init__(self, data_dir: str, n_epoch: int) -> None:
        super().__init__()
        npy_files = os.listdir(data_dir)
        self.npy_file_paths = [
            os.path.join(data_dir, npy_file) for npy_file in npy_files
        ]
        self.n_files = len(npy_files)
        self.n_epoch = n_epoch
        print(f".npy files num: {self.n_files}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.n_files - 1
        else:
            per_worker = int(
                math.ceil(self.n_files / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_files - 1)
        feature_parser = FeatureParser()

        for epoch in range(self.n_epoch):
            for i in range(iter_start, iter_end):
                samples = np.load(self.npy_file_paths[i],
                                  allow_pickle=True)['data']
                random.shuffle(samples)
                for sample in samples:
                    obs, action = sample
                    # observation -> feature
                    x = feature_parser.parse({0: obs}, float(obs["step"])/1024)[0]
                    y = self.actions_to_tensor_labels(action)
                    yield x, y
        return

    @staticmethod
    def actions_to_tensor_labels(action):
        labels = {name: 0 for name in networks.ActionHead.name2dim.keys()}
        if nmmo.io.action.Attack in action:
            attack_style = action[nmmo.io.action.Attack][nmmo.io.action.Style]
            attack_target = action[nmmo.io.action.Attack][
                nmmo.io.action.Target]
            labels["attack_style"] = attack_style + 1
            labels["attack_target"] = attack_target + 1
        if nmmo.io.action.Move in action:
            direction = action[nmmo.io.action.Move][nmmo.io.action.Direction]
            labels["move"] = direction + 1
        if nmmo.io.action.Use in action:
            use_item = action[nmmo.io.action.Use][nmmo.io.action.Item]
            labels["use_item"] = use_item + 1
        if nmmo.io.action.Sell in action:
            sell_item = action[nmmo.io.action.Sell][nmmo.io.action.Item]
            sell_price = action[nmmo.io.action.Sell][nmmo.io.action.Price]
            labels["sell_target"] = sell_item + 1
            labels["sell_price"] = sell_price + 1
        if nmmo.io.action.Buy in action:
            buy_item = action[nmmo.io.action.Buy][nmmo.io.action.Item]
            labels["buy_item"] = buy_item + 1

        return labels


def train(flags, model, loss_fn, optimizer):
    model.train()

    writer = SummaryWriter(filename_suffix="imitation")

    action_losses = {
        "attack_style": 0,
        "attack_target": 0,
        "move": 0,
    }
    loss_coefs = {
        "attack_style": 1,
        "attack_target": 1.5,
        "move": 2,
    }

    dataset = NMMODataset(flags.npy_save_dir, n_epoch=flags.n_epoch)
    dataloader = DataLoader(dataset,
                            batch_size=flags.batch_size,
                            num_workers=flags.num_workers,
                            pin_memory=True)

    step = 0    
    start = time.time()
    for feature_batch, labels in iter(dataloader):
        feature_batch = tree.map_structure(
            lambda x: x.unsqueeze(dim=0).to(flags.device), feature_batch)
        labels = tree.map_structure(
            lambda x: x.to(flags.device), labels)
        
        # Compute prediction error
        feature_batch["lstm_state"] = model.initial_state(len(feature_batch))
        preds = model(feature_batch, training=True)
        loss = torch.zeros((), dtype=torch.float32).to(flags.device)
        target_actions = ["move", "attack_style", "attack_target"]
        for label_key in target_actions:
            label = labels[label_key]
            pred = preds[f"{label_key}_logits"]
            pred = pred.view(-1, pred.shape[-1])
            action_losses[label_key] = loss_fn(pred, label)
            loss += loss_coefs[label_key]*action_losses[label_key]

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        print(f"step: {step} took {time.time() - start}s")
        start = time.time()
        step += 1
        if step % 100 == 0:
            for label_key in target_actions:
                writer.add_scalar(f"Loss/{label_key}",
                                  action_losses[label_key].item(),
                                  global_step=step)
            writer.add_scalar("Loss/total_loss", loss.item(), global_step=step)

        # Save checkpoints
        if step % 1000 == 0:
            torch.save(model.state_dict(),
                       os.path.join(flags.ckpt_save_dir, 
                                    f"imitation_model_{step}.pth"))
            print(f"Checkpoint {step} saved!")


if __name__ == "__main__":
    # Parse flags
    flags = parser.parse_args()
    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        print("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        print("Using CPU.")
        flags.device = torch.device("cpu")
    
    flags.ckpt_save_dir = "./checkpoints"
    os.makedirs(flags.ckpt_save_dir, exist_ok=True)

    # Create model
    net = NMMONet(0).to(device=flags.device)
    if flags.checkpoint_path is not None:
        print(f"Loading checkpoint: {flags.checkpoint_path}")
        pretrained_checkpoint = torch.load(flags.checkpoint_path)
        net.load_state_dict(pretrained_checkpoint)
    else:
        print("Model training from scratch!")

    # Loss function and optimizer
    loss_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=flags.learning_rate,
    )

    # Start training
    train(flags, net, loss_ce, optimizer)
