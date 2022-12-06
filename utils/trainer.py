import torch
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_utils import AverageMeter, accuracy as Accuracy
from typing import Callable, Tuple
from tqdm.auto import tqdm
import time
from torch.utils.tensorboard.writer import SummaryWriter

FINE = np.array([0, 2, 3, 4, 5])
COARSE = np.array([1, 6, 7, 8, 9, 10, 11, 12, 13])


def train_for_one_epoch(epoch: int, model: nn.Module, optimizer: Optimizer, criterion: Callable, dataloader: DataLoader, writer: SummaryWriter, device: torch.device) -> Tuple[float, float]:
    model.train()
    
    num_batches = len(dataloader)
    
    running_accuracy = AverageMeter("Accuracy", num_batches, fmt=":.4f")
    running_loss = AverageMeter("Loss", num_batches, fmt=":.4f")

    batch_index = 0
    with tqdm(desc=str.ljust("Train", 12), total=num_batches, unit="bt") as bar:
        for skeleton_joints, depth_features, labels in dataloader:
            skeleton_joints, depth_features, labels = skeleton_joints.to(device), depth_features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(skeleton_joints, depth_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            acc, *_ = Accuracy(outputs, labels)
            running_accuracy.update(acc.item() / 100)
            running_loss.update(loss.item())
            
            # log to tensorboard
            writer.add_scalar("accuracy_steps/train", running_accuracy.val, epoch * num_batches + batch_index)
            writer.add_scalar("loss_steps/train", running_loss.val, epoch * num_batches + batch_index)
            
            bar.set_postfix({"accuracy": f"{running_accuracy.avg:.4f}", "loss": f"{running_loss.avg:.4f}"}, refresh=True)
            bar.update()
            
            batch_index += 1
            writer.flush()
        bar.close()
            
    return running_accuracy.avg, running_loss.avg


@torch.no_grad()
def evaluate(epoch: int, model: nn.Module, criterion: Callable, dataloader: DataLoader, writer: SummaryWriter, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    
    num_batches = len(dataloader)
    
    running_accuracy = AverageMeter("Accuracy", num_batches, fmt=":.4f")
    running_loss = AverageMeter("Loss", num_batches, fmt=":.4f")
    
    batch_index = 0
    
    outputs_list = []
    labels_list = []
    
    with tqdm(desc=str.ljust("Validation", 12), total=num_batches, unit="bt") as bar:
        for batch_index, (skeleton_joints, depth_features, labels) in enumerate(dataloader):
            skeleton_joints, depth_features, labels = skeleton_joints.to(device), depth_features.to(device), labels.to(device) 
            outputs = model(skeleton_joints, depth_features)
            
            acc, *_ = Accuracy(outputs, labels)
            running_accuracy.update(acc[0].item() / 100)
            
            loss = criterion(outputs, labels)
            running_loss.update(loss.item())
            
            # log to tensorboard
            writer.add_scalar("accuracy_steps/validation",
                              running_accuracy.val, epoch * num_batches + batch_index)
            writer.add_scalar("loss_steps/validation", running_loss.val, epoch * num_batches + batch_index)
            
            # list of outputs and labels
            outputs_list.append(outputs.detach().numpy())
            labels_list.append(labels.detach().numpy())
            
            bar.set_postfix({"accuracy": f"{running_accuracy.avg:.4f}",
                            "loss": f"{running_loss.avg:.4f}"}, refresh=True)
            bar.update()
            
            batch_index += 1
            writer.flush()
            
        bar.close()
    
    outputs_list = np.concatenate(outputs_list, 0)
    labels_list = np.concatenate(labels_list, 0)
            
    return running_accuracy.avg, running_loss.avg, outputs_list, labels_list
