from torch import nn
import torch
from einops import rearrange


class PixelTokenizer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int) -> None:
        """Generate 1-D patch-based token embeddings from n-D input.

        Args:
            in_channels (int, optional): input channels. Defaults to 1.
            out_channels (int, optional): output channels. Defaults to 128.
            patch_size (int, optional): patch size. Defaults to 16.
        
        Citation:
        Roy, S. K., Deria, A., Hong, D., Rasti, B., Plaza, A., & Chanussot, J. (2022). Multimodal fusion transformer for remote sensing image classification. arXiv preprint arXiv:2203.16952.
        """

        super(PixelTokenizer, self).__init__()

        self._patch_size = patch_size
        self._out_channels = out_channels

        self._conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self._wa = nn.Parameter(torch.empty(1, 1, 1))
        self._wb = nn.Parameter(torch.empty(1, 1, out_channels))

        nn.init.xavier_normal_(self._wa)
        nn.init.xavier_normal_(self._wb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape: [batch, time_len, in_channels, height, width]

        height, width = x.shape[-2], x.shape[-1]
        assert height % self._patch_size == 0, f"Height must be a multiple of patch_size {self._patch_size}, received {height}"
        assert width % self._patch_size == 0, f"Width must be a multiple of patch_size {self._patch_size}, received {width}"

        batches, time_len = x.shape[0], x.shape[1]

        x_patches = rearrange(x, "b t c (h p1) (w p2) -> (b t h w) c p1 p2",
                              p1=self._patch_size, p2=self._patch_size)
        x_conv = self._conv_block(x_patches)
        x_conv = rearrange(x_conv, "n c p1 p2 -> n c (p1 p2)",
                           c=1, p1=self._patch_size, p2=self._patch_size)
        x_conv = rearrange(x_conv, "n c p -> n p c", c=1)

        x_wa = torch.einsum('bij,bjk->bik', x_conv, self._wa)
        x_wa = rearrange(x_wa, "n p c -> n c p", c=1)
        x_wa = x_wa.softmax(dim=1)

        x_wb = torch.einsum('bij,bjk->bik', x_conv, self._wb)

        x_out = torch.einsum('bij,bjk->bik', x_wa, x_wb)
        x_out = rearrange(x_out, "(b t n) c d -> b t n c d",
                          b=batches, t=time_len, c=1, d=self._out_channels)
        x_out = x_out.squeeze(-2)

        # shape: [batch, time_len, num_patches, out_channels]
        return x_out
