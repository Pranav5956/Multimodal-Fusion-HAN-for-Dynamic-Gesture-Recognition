import torch.nn as nn
import torch
from einops import rearrange, reduce

from .layers.attention import AttentionLayer

from typing import Dict


class HAN(nn.Module):
    FINGER_LIST = [
        [0, 1],
        [2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17],
        [18, 19, 20, 21]
    ]

    def __init__(self, num_classes: int, config: Dict) -> None:
        super(HAN, self).__init__()

        self._batch_norm_2d_input = nn.BatchNorm2d(config["input_dim"])
        self._input_map = nn.Sequential(
            nn.Linear(config["input_dim"], config["attention"]["input_dim"]),
            nn.ReLU(inplace=True),
        )

        self._joint_attention = AttentionLayer(config["attention"])
        self._finger_attention = AttentionLayer(
            config["attention"], use_cross_attention=True)  # cross-attention layer
        self._temporal_attention = AttentionLayer(config["attention"])
        self._fusion_attention = AttentionLayer(config["attention"])

        self._cls = nn.Sequential(
            nn.Dropout(config["dropout_prob"]),
            nn.Linear(config["attention"]["output_dim"], num_classes),
        )

    def forward(self, skeleton_features: torch.Tensor, depth_feature_token: torch.Tensor) -> torch.Tensor:
        # skeleton feature shape: [batch_size, sample_size, num_joints, feature_dim]
        # depth feature token shape: [batch_size, sample_size, num_patches, feature_dim]
        skeleton_features = rearrange(skeleton_features, "b t j c -> b c t j")
        skeleton_features = self._batch_norm_2d_input(skeleton_features)
        skeleton_features = rearrange(skeleton_features, "b c t j -> b t j c")

        #  finger attention
        finger_feature = [
            reduce(
                self._joint_attention(self._input_map(
                    skeleton_features[:, :, finger, :])),
                "b t j c -> b t c",
                "mean"
            ) for finger in HAN.FINGER_LIST
        ]

        # hand attention (feature token appended)
        hand_feature = rearrange([depth_feature_token, *finger_feature], "n b t c -> b t n c")
        hand_feature = reduce(self._finger_attention(hand_feature), "b t n c -> b t c", "mean")

        # temporal attention
        finger_temporal_feature = [
            reduce(
                self._temporal_attention(feature),
                "b t c -> b c",
                "mean"
            ) for feature in finger_feature
        ]
        hand_temporal_feature = reduce(
            self._temporal_attention(hand_feature), "b t c ->  b c", "mean"
        )
        temporal_feature = rearrange(
            [*finger_temporal_feature, hand_temporal_feature],
            "n b c -> b n c"
        )

        # fusion attention
        fusion_feature = reduce(
            self._fusion_attention(temporal_feature),
            "b n c -> b c",
            "mean"
        )

        # prediction head
        out = self._cls(fusion_feature)
        return out
