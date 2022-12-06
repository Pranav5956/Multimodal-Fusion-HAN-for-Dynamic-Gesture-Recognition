import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from .normalization import LayerNorm
from .positional_encoding import PositionalEncoding

from typing import Dict


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, input_dim: int, dropout_prob: float) -> None:
        super(MultiheadAttention, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self._head_dim = head_dim
        self._num_heads = num_heads

        embed_dim = head_dim * num_heads
        self.key_map = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Dropout(dropout_prob)
        )

        self.query_map = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Dropout(dropout_prob)
        )

        self.value_map = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob))

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # [batch, time, ft_dim)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / np.sqrt(d_k)

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        size = list(q.shape)
        
        size_q = (*q.shape[0:-1], self._num_heads, self._head_dim)  # [batch, t, q_dim]
        size_kv = (*k.shape[0:-1], self._num_heads, self._head_dim) # [batch, t, kv_dim]

        query = self.query_map(q).view(size_q).transpose(-2, -3)
        key = self.key_map(k).view(size_kv).transpose(-2, -3)
        value = self.value_map(v).view(size_kv).transpose(-2, -3)

        x, _ = self._attention(query, key, value)

        size[-1] = self._num_heads * self._head_dim
        x = x.transpose(-2, -3).contiguous() \
            .view(size)  # [batch, T, h_dim * h_num ]

        return x


class AttentionLayer(nn.Module):
    def __init__(self, config: Dict, use_cross_attention=False) -> None:
        super(AttentionLayer, self).__init__()

        self._use_cross_attention = use_cross_attention

        self._positional_encoding = PositionalEncoding(config["input_dim"])

        self._attention = MultiheadAttention(
            config["num_heads"],
            config["head_dim"],
            config["input_dim"],
            config["dropout_prob"]
        )
        self._feature_map = nn.Sequential(
            nn.Linear(config["num_heads"] *
                      config["head_dim"], config["output_dim"]),
            nn.ReLU(),
            LayerNorm(config["output_dim"]),
            nn.Dropout(config["dropout_prob"]),
        )
        self._norm = LayerNorm(config["output_dim"])
        self._dropout = nn.Dropout(config["dropout_prob"])

        self._init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        x = self._positional_encoding(x)
        x_res = x
            
        if self._use_cross_attention:
            x = self._attention(x[..., :1, :], x, x)
            x_res = torch.mean(x_res, -2).unsqueeze(-2)
        else:
            x = self._attention(x, x, x)
        
        x = self._feature_map(x)

        return self._norm(x_res + self._dropout(x))

    def _init_parameters(self) -> None:
        model_list = [self._attention, self._feature_map]
        for model in model_list:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
