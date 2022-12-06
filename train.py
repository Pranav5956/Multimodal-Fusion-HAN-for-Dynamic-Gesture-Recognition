import torch
from torch.optim import AdamW
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import yaml

from datasets.SHREC2017_Dataset import SHREC2017_Dataset
from models.MFHAN import MFHAN
from utils.trainer import *
from utils.scheduler import WarmupScheduler, ReduceLROnPlateau
from utils.misc import *


def train(config) -> None:
    checkpoint_dir = use_or_create_checkpoint_dir(
        config["checkpoints_dir"], config["name"])
    print(f"\nCreated run folder \"{checkpoint_dir}\"!")
    print(f"Copying config file to \"{checkpoint_dir}\" ... ", end="")
    yaml.dump(config, open(os.path.join(checkpoint_dir, "config.yml")))
    print("done!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(checkpoint_dir, comment=config, flush_secs=10)
    writer.add_custom_scalars({
        "Accuracy": {"accuracy_epochs": ["Multiline", ["accuracy_epochs/train", "accuracy_epochs/validation"]]},
        "Loss": {"loss_epochs": ["Multiline", ["loss_epochs/train", "loss_epochs/validation"]]},
    })
    
    train_dataloader = DataLoader(
        SHREC2017_Dataset(config["num_classes"], config["dataset"], split="train"),
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        pin_memory=(device == "cuda"),
        num_workers=config["num_workers"],
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        SHREC2017_Dataset(config["num_classes"], config["dataset"], split="test"),
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=(device == "cuda"),
        num_workers=config["num_workers"],
        persistent_workers=True
    )
    
    model = MFHAN(config["num_classes"], config["model"])
    model = model.to(device)
    param_count = count_params(model)
    print(f"\nCreated model with {param_count} parameters!")
    
    criterion = torch.nn.CrossEntropyLoss(**config["loss"])
    
    optimizer = AdamW(model.parameters(), **config["optimizer"])
    scheduler = ReduceLROnPlateau(optimizer, **config["scheduler"]["reduce_lr_on_plateau"], mode="min", threshold_mode="abs", eps=1e-15)
    warmup_scheduler = WarmupScheduler(
        optimizer, config["scheduler"]["warmup_epochs"])
    
    print("\nTraining started!")
    
    best_accuracy = 0
    best_loss = float('inf')
    
    # removes error for warmup scheduling
    optimizer.zero_grad()
    optimizer.step()
    
    # run epochs till lr drops n times
    epoch = 0
    while True:
        # warmup scheduler
        if epoch < config["scheduler"]["warmup_epochs"]:
            warmup_scheduler.step()
            learning_rate = float(optimizer.param_groups[0]['lr'])
            print(f"\nEpoch: {epoch + 1} | Learning Rate: {learning_rate:.1e} | Best Accuracy: {best_accuracy:.4f} | Best Loss: {best_loss:.4f} | Warmup Epochs Left: {config['scheduler']['warmup_epochs'] - epoch - 1}")
        else:
            learning_rate = float(optimizer.param_groups[0]['lr'])
            print(f"\nEpoch: {epoch + 1} | Learning Rate: {learning_rate:.1e} | Best Accuracy: {best_accuracy:.4f} | Best Loss: {best_loss:.4f} | Epochs Till LR Drop: {scheduler.patience - scheduler.num_bad_epochs}")      
        
        train_accuracy, train_loss = train_for_one_epoch(epoch, model, optimizer, criterion, train_dataloader, writer, device)
        val_accuracy, val_loss, outputs, labels = evaluate(epoch, model, criterion, val_dataloader, writer, device)
        
        # log to tensorboard
        writer.add_scalar("learning_rate", learning_rate, epoch)
        writer.add_scalar("accuracy_epochs/train", train_accuracy, epoch)
        writer.add_scalar("accuracy_epochs/validation", val_accuracy, epoch)
        writer.add_scalar("loss_epochs/train", train_loss, epoch)
        writer.add_scalar("loss_epochs/validation", val_loss, epoch)
        writer.flush()
        
        # update metrics
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = val_accuracy
            print(f"Saving \"best_model.pth\" at \"{checkpoint_dir}\" ... ", end="")
            save_model(epoch, checkpoint_dir, model, optimizer, {
                "best_loss": best_loss,
                "best_accuracy": best_accuracy
            })
            save_predictions(outputs, labels, checkpoint_dir)
            print("done!")
        
        # reduce lr on plateau
        if epoch >= config["scheduler"]["warmup_epochs"]:
            is_lr_reduced = scheduler.step(val_loss)
            if scheduler.lr_reduced_times == config["lr_drop_threshold"]:
                print(f"\nTraining terminated since learning rate was reduced for {config['lr_drop_threshold']} times!")
                break
            
            if is_lr_reduced:
                print(f"Learning rate reduced by a factor of {scheduler.factor} since validation loss did not improve for {scheduler.patience} epochs!")
        
        epoch += 1
            
    writer.close()
    print("\nTraining completed!")
