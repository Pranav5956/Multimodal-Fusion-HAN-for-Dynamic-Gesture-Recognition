import torch.nn as nn
import torch
from .HAN import HAN
from .MFEncoder import MFEncoder

from typing import Dict


class MFHAN(nn.Module):
    def __init__(self, num_classes: int, config: Dict) -> None:
        super(MFHAN, self).__init__()
        
        self._encoder = MFEncoder(config["encoder"])
        self._HAN = HAN(num_classes, config["HAN"])
    
    def forward(self, skeleton_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        depth_feature_token = self._encoder(depth_features)
        out = self._HAN(skeleton_features, depth_feature_token)
        
        return out
