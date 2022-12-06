import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1, dim: int = -1, logits: bool = True) -> None:
        super(LabelSmoothingLoss, self).__init__()

        self._confidence = 1.0 - smoothing
        self._smoothing = smoothing
        self._cls = num_classes
        self._dim = dim
        self._logits = logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert 0 <= self._smoothing < 1
        
        if self._logits:
            pred = pred.log_softmax(dim=self._dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self._smoothing / (self._cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self._confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self._dim))
