from argparse import Namespace
from utils.misc import load_predictions
import torch

def evaluate(args: Namespace) -> None:
    run_dir = args.run_dir
    outputs, labels = load_predictions(run_dir)
    outputs, labels = torch.from_numpy(outputs), torch.from_numpy(labels)
    
    outputs = outputs.softmax(dim=-1)
    _, outputs = outputs.max(dim=-1)
    acc = ((outputs.detach().numpy() == labels.detach().numpy()).sum() / outputs.shape[0]) * 100
    
    print(f"Accuracy: {acc:.4f}%")
    