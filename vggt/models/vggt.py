# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    """
    【VGGT主模型 - 重点阅读】
    Visual Geometry Grounded Transformer 的顶层封装。

    整体流程:
        1. Aggregator: 对输入图像进行交替注意力编码，得到多尺度融合特征
        2. CameraHead:  从特征中预测相机位姿 (外参+内参)
        3. DepthHead:   从特征中预测深度图
        4. PointHead:   从特征中预测点云 (世界坐标系下的3D点)
        5. TrackHead:   对指定查询点进行跨帧追踪

    关键输出 (predictions dict):
        - "pose_enc":          [B, S, 9]   相机位姿编码 (需经 pose_encoding_to_extri_intri 解码)
        - "depth":             [B, S, H, W, 1]  深度图
        - "depth_conf":        [B, S, H, W]    深度置信度
        - "world_points":      [B, S, H, W, 3] 世界坐标系下的3D点 (点云)
        - "world_points_conf": [B, S, H, W]    点云置信度
        - "track":             [B, S, N, 2]    追踪点像素坐标
        - "vis":               [B, S, N]       追踪点可见性
        - "conf":              [B, S, N]       追踪置信度

    注意: B=batch, S=sequence(帧数), H=W=518(默认), N=查询点数量
    """
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        # 【核心骨干网络】交替注意力聚合器，输出 2*embed_dim 维特征
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        # 【相机头】输入维度 2*embed_dim 因为 aggregator 输出的是 frame+global 拼接特征
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        # 【点云头】output_dim=4: 3维坐标 + 1维置信度; activation="inv_log" 用于世界坐标
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        # 【深度头】output_dim=2: 1维深度 + 1维置信度; activation="exp" 保证深度为正
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        # 【追踪头】对查询点进行跨帧追踪
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        【前向传播 - 重点阅读】
        VGGT 模型的完整推理流程。

        Args:
            images (torch.Tensor): 输入图像, 范围 [0, 1], 形状 [S, 3, H, W] 或 [B, S, 3, H, W]
                B: batch size, S: 帧数/序列长度, 3: RGB通道, H: 高, W: 宽
            query_points (torch.Tensor, optional): 追踪查询点, 像素坐标
                形状: [N, 2] 或 [B, N, 2], N 为查询点数量
                默认: None

        Returns:
            dict: 预测结果字典, 包含以下内容:
                - pose_enc (torch.Tensor): 相机位姿编码 [B, S, 9], 最后一轮迭代结果
                  【需用 pose_encoding_to_extri_intri() 解码为外参矩阵 [B,S,3,4] 和内参矩阵 [B,S,3,3]】
                - depth (torch.Tensor): 深度图 [B, S, H, W, 1]
                - depth_conf (torch.Tensor): 深度置信度 [B, S, H, W]
                - world_points (torch.Tensor): 世界坐标系3D点 [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): 点云置信度 [B, S, H, W]
                - images (torch.Tensor): 原始输入图像, 用于可视化

                如果提供了 query_points, 还包含:
                - track (torch.Tensor): 追踪点轨迹 [B, S, N, 2], 像素坐标, 最后一轮迭代结果
                - vis (torch.Tensor): 追踪点可见性分数 [B, S, N]
                - conf (torch.Tensor): 追踪点置信度 [B, S, N]
        """
        # 如果没有 batch 维度, 自动添加 (batch_size=1)
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # 【Step 1: 特征聚合】交替注意力提取多帧融合特征
        # aggregated_tokens_list: List[Tensor], 每轮迭代的特征输出
        # patch_start_idx: patch tokens 在 token 序列中的起始索引 (跳过 camera token 和 register tokens)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        # 【Step 2: 各任务头预测】
        # 注意: camera/point/depth 在 autocast(enabled=False) 中运行, 即 fp32 精度
        # track 在 amp 外运行
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                # camera_head 返回每轮迭代的 pose_enc 列表, 取最后一轮作为最终输出
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # 最后一轮位姿编码
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                # 【深度头输出】depth: [B,S,H,W,1], depth_conf: [B,S,H,W]
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                # 【点云头输出】pts3d: [B,S,H,W,3], pts3d_conf: [B,S,H,W]
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            # 【追踪头输出】track: [B,S,N,2], vis: [B,S,N], conf: [B,S,N]
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # 最后一轮追踪结果
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            # 推理时保存原始图像, 方便后续可视化
            predictions["images"] = images

        return predictions

