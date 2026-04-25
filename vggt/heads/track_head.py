# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor


class TrackHead(nn.Module):
    """
    【TrackHead - 追踪头】
    对输入图像中的指定查询点进行跨帧追踪。

    结构:
        1. Feature Extractor: 基于 DPTHead (feature_only=True, down_ratio=2)
           从 Aggregator token 中提取多尺度特征图
           输出特征图分辨率: H//2 x W//2
        2. Tracker: BaseTrackerPredictor, 基于 RAFT/Cotracker 风格的 correlation-based tracker
           通过多轮迭代 refine 追踪坐标

    输出:
        - coord_preds (list[Tensor]): 每轮迭代的追踪坐标预测
          每个元素 [B, S, N, 2], 像素坐标
        - vis_scores (Tensor): 可见性分数 [B, S, N]
          表示查询点在各帧是否可见
        - conf_scores (Tensor): 追踪置信度 [B, S, N]

    注意: VGGT.forward() 中只取 coord_preds 的最后一项: track = coord_preds[-1]
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        self.patch_size = patch_size

        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=2,  # Reduces spatial dimensions by factor of 2
            pos_embed=False,
        )

        # Tracker module that predicts point trajectories
        # Takes feature maps and predicts coordinates and visibility
        self.tracker = BaseTrackerPredictor(
            latent_dim=features,  # Match the output_dim of feature extractor
            predict_conf=predict_conf,
            stride=stride,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_size=hidden_size,
        )

        self.iters = iters

    def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None):
        """
        【TrackHead 前向传播】

        Args:
            aggregated_tokens_list (list): Aggregator 各层 token 列表
            images (torch.Tensor): 输入图像 (B, S, C, H, W)
            patch_start_idx (int): patch tokens 起始索引
            query_points (torch.Tensor, optional): 待追踪的查询点 [B, N, 2], 像素坐标
                若为 None, 由 tracker 自动初始化
            iters (int, optional): 迭代轮数, 默认 self.iters (默认 4)

        Returns:
            tuple:
                - coord_preds (list[Tensor]): 每轮迭代的追踪坐标 [B, S, N, 2]
                - vis_scores (Tensor): 可见性分数 [B, S, N]
                - conf_scores (Tensor): 置信度 [B, S, N]
        """
        B, S, _, H, W = images.shape

        # 【Step 1: 提取追踪特征】
        # feature_maps 形状: (B, S, C, H//2, W//2) 因为 down_ratio=2
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)

        if iters is None:
            iters = self.iters

        # 【Step 2: 迭代追踪】
        coord_preds, vis_scores, conf_scores = self.tracker(
            query_points=query_points, fmaps=feature_maps, iters=iters
        )

        return coord_preds, vis_scores, conf_scores
