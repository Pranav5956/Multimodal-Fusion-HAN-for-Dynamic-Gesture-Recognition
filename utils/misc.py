import torch
from torch import nn
from torch.optim import Optimizer
import numpy as np
import random
import os
import yaml
from typing import Dict, Tuple


def seed_everything(seed: str) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_params(model: nn.Module) -> int:
    return sum(map(lambda p: p.data.numel(), model.parameters()))


def use_or_create_checkpoint_dir(checkpoints_dir: str, name: str) -> str:
    while os.path.exists(os.path.join(checkpoints_dir, name)):
        name += ".0"
    checkpoint_dir = os.path.normpath(os.path.join(checkpoints_dir, name))
    
    os.makedirs(checkpoint_dir)    
    return checkpoint_dir


def get_runs_count(run_dir: str) -> int:
    runs = [int(run[-1]) for run in os.listdir(run_dir) if run.startswith("run") and run[3:].isnumeric()]
    return max(runs) if len(runs) > 0 else 0


def save_model(epoch: int, save_dir: str, model: nn.Module, optimizer: Optimizer = None, metrics: Dict = {}) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else optimizer,
        "metrics": metrics
    }

    torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))


def load_config_file(config_file: str) -> Dict:
    if not os.path.exists(config_file):
        raise FileNotFoundError("Config file not found!")
    if not config_file.endswith(".yml") and not config_file.endswith(".yaml"):
        raise TypeError("Only .yml or .yaml config files are accepted!")

    return yaml.load(open(config_file, 'r'), yaml.SafeLoader)


def save_predictions(outputs: np.ndarray, labels: np.ndarray, save_dir: str):
    np.savez_compressed(os.path.join(save_dir, "predictions"),
                        outputs=outputs, labels=labels)


def load_predictions(run_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    load = np.load(os.path.join(run_dir, "predictions.npz"))
    return load["outputs"], load["labels"]
