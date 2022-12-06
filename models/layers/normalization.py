import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, feature_dim: int, eps=1e-6) -> torch.Tensor:
        super(LayerNorm, self).__init__()
        self._a_2 = nn.Parameter(torch.ones(feature_dim))
        self._b_2 = nn.Parameter(torch.zeros(feature_dim))
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, time, feature_dim)
        mean = x.mean(-1, keepdim=True)

        std = x.std(-1, keepdim=True)
        return self._a_2 * (x - mean) / (std + self._eps) + self._b_2