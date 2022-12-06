from torch import nn
import torch
from einops import repeat, rearrange
from .layers.tokenizer import PixelTokenizer
from .layers.attention import AttentionLayer

from typing import Dict


class MFEncoder(nn.Module):
    def __init__(self, config: Dict) -> None:
        super(MFEncoder, self).__init__()

        self._variant = config["variant"]
        self._cls_token = nn.Parameter(
            torch.randn(1, 1, 1, config["output_dim"]))

        if config["variant"] == "patch-based":
            self._tokenizer = PixelTokenizer(
                config["input_dim"], config["output_dim"], config["patch_size"])
        elif config["variant"] == "roi-based":
            # TODO: roi-based tokenizer
            raise NotImplementedError("ROI-based tokenizer is not yet implemented!")
        else:
            raise ValueError(f"No tokenizer variant called \"{config['variant']}\"!")

        self._encoder = AttentionLayer(config["attention"])

        self._out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batches, time_len = x.shape[0], x.shape[1]

        x = rearrange(x, "b t h w c -> b t c h w")
        x_tokens = self._tokenizer(x)

        cls_token = repeat(
            self._cls_token, "1 1 1 d -> b t 1 d", b=batches, t=time_len)
        x_tokens = torch.cat([cls_token, x_tokens], dim=-2)
        
        x_out = self._encoder(x_tokens)
        return self._out(x_out[:, :, 0, :]) # only class tokens
