from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau as RLROP
from typing import Any, Union, List


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, last_epoch: int = -1):
        self._warmup_epochs = warmup_epochs

        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self._warmup_epochs + 1e-8) for base_lr in self.base_lrs]


class ReduceLROnPlateau(RLROP):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False) -> None:
        super().__init__(optimizer, mode, factor, patience, threshold,
                         threshold_mode, cooldown, min_lr, eps, verbose)

        self.lr_reduced_times = 0

    def step(self, metrics: Any) -> bool:
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        is_lr_reduced = False
        if self.num_bad_epochs > self.patience:
            is_lr_reduced = True
            self._reduce_lr(epoch)
            self.lr_reduced_times += 1
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        return is_lr_reduced
