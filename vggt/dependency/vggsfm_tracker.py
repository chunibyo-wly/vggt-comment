# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from hydra.utils import instantiate
from omegaconf import OmegaConf

from .track_modules.track_refine import refine_track
from .track_modules.blocks import BasicEncoder, ShallowEncoder
from .track_modules.base_track_predictor import BaseTrackerPredictor


"""
1. TrackerPredictor.forward()
输入:
    images:        [B, S, 3, H, W]   输入图像 (范围 [0,1])
    query_points:  [B, N, 2]         query 点坐标 (原始图像尺度, x,y)
    fmaps:         [B, S, C, HH, WW] 预计算特征图 (可选)

输出:
    fine_pred_track:   [B, S, N, 2]  精细追踪结果 (亚像素精度)
    coarse_pred_track: [B, S, N, 2]  粗追踪结果
    pred_vis:          [B, S, N]     可见性分数 (0~1)
    pred_score:        [B, S, N]     追踪置信度 (fine 阶段为 None)

2. process_images_to_fmaps()
输入:  images [B*S, 3, H, W]
输出:  fmaps  [B*S, C, HH, WW]  (HH=H/stride/down_ratio, WW=W/stride/down_ratio)

3. BaseTrackerPredictor.forward()
输入:
    query_points: [B, N, 2]         query 点 (原始图像尺度)
    fmaps:        [B, S, C, HH, WW] 多帧特征图
    iters:        int               迭代轮数

输出:
    coord_preds: list of [B, S, N, 2]  每轮迭代的坐标预测 (通常取最后一轮)
    vis_e:       [B, S, N]              可见性分数

4. refine_track()
输入:
    images:            [B, S, 3, H, W]
    coarse_pred_track: [B, S, N, 2]     coarse 阶段输出的粗略坐标

输出:
    fine_pred_track: [B, S, N, 2]      精修后的亚像素坐标
    pred_score:      [B, S, N]          追踪分数

5. BasicEncoder (Coarse 特征编码)
输入:  x [B, 3, H, W]
输出:  x [B, output_dim, H/stride, W/stride]

6. ShallowEncoder (Fine 特征编码)
输入:  x [B, 3, H, W]
输出:  x [B, output_dim, H/stride, W/stride]  (stride=1 时分辨率不变)

7. CorrBlock
__init__: fmaps [B, S, C, H, W]
corr():   targets [B, S, N, C] → 无返回 (结果存入 self.corrs_pyramid)
sample(): coords  [B, S, N, 2] → [B, S, N, num_levels*(2r+1)^2]

8. EfficientUpdateFormer
输入:  [B, N, S, input_dim]   (B=batch, N=tracks, S=frames)
输出:  [B, N, S, output_dim]  (前 2 维是坐标增量 dx,dy，后面是特征增量)
"""

class TrackerPredictor(nn.Module):
    """
    query_points 看作 "种子点"：你在第 0 帧指定了 N 个位置，TrackerPredictor 的任务就是找出这 N 个点在第 1, 2, ..., S-1 帧分别跑到了哪里。fine_pred_track 就是这 N 个点的完整运动轨迹。

    【VGGSfM Tracker 主类 - 重点阅读】
    两阶段追踪器: Coarse (粗追踪) + Fine (精修追踪)。

    架构:
        ┌─────────────────────────────────────────────────────────────┐
        │  Stage 1: Coarse Tracking (粗粒度追踪)                       │
        │  - coarse_fnet: BasicEncoder, stride=4, 下采样 4x          │
        │  - coarse_predictor: BaseTrackerPredictor, stride=4        │
        │  - 在 1/8 分辨率特征图上做 correlation-based 追踪            │
        │  - 输出大致位置, 但精度有限 (误差约几个像素)                 │
        ├─────────────────────────────────────────────────────────────┤
        │  Stage 2: Fine Tracking (精细追踪) ← refine_track()        │
        │  - fine_fnet: ShallowEncoder, stride=1, 全分辨率           │
        │  - fine_predictor: BaseTrackerPredictor, stride=1          │
        │  - 在 coarse 结果周围提取小 patch (31x31)                  │
        │  - 在 patch 内做精细追踪, 达到亚像素精度                     │
        └─────────────────────────────────────────────────────────────┘

    为什么分两阶段:
        - 直接在全分辨率做 correlation 追踪: 显存开销巨大 (S*N*H*W)
        - Coarse 先在低分辨率定位, Fine 只在小 patch 内精修: 显存友好且精度高
    """

    def __init__(self, **extra_args):
        super(TrackerPredictor, self).__init__()

        # 【Coarse 阶段配置】
        coarse_stride = 4                   # 特征图下采样 4x
        self.coarse_down_ratio = 2          # 图像额外下采样 2x (总下采样 8x)

        # Coarse 特征提取网络: BasicEncoder (ResNet-like CNN)
        self.coarse_fnet = BasicEncoder(stride=coarse_stride)
        # Coarse 追踪器: 在 1/8 分辨率特征图上做迭代追踪
        self.coarse_predictor = BaseTrackerPredictor(stride=coarse_stride)

        # 【Fine 阶段配置】
        # Fine 特征提取网络: ShallowEncoder, stride=1 (全分辨率)
        self.fine_fnet = ShallowEncoder(stride=1)
        # Fine 追踪器: 在小 patch 内做精细追踪
        self.fine_predictor = BaseTrackerPredictor(
            stride=1,
            depth=4,                        # Transformer 层数较少 (只在 patch 内)
            corr_levels=3,
            corr_radius=3,
            latent_dim=32,                  # 特征维度较小
            hidden_size=256,
            fine=True,                      # fine=True 表示不预测 visibility
            use_spaceatt=False,             # 不使用空间注意力 (patch 太小没必要)
        )

    def forward(
        self, images, query_points, fmaps=None, coarse_iters=6, inference=True, fine_tracking=True, fine_chunk=40960
    ):
        """
        【TrackerPredictor 前向传播 - 重点阅读】

        两阶段追踪流程:
            1. Coarse: 在低分辨率特征图上迭代追踪, 得到大致位置
            2. Fine: 在 coarse 结果周围提取 patch, 做亚像素精修

        Args:
            images (Tensor): 输入图像 [B, S, 3, H, W], 范围 [0, 1]
            query_points (Tensor): 查询点坐标 [B, N, 2], 相对于左上角
            fmaps (Tensor, optional): 预计算的特征图, 若 None 则现场计算
            coarse_iters (int): Coarse 阶段迭代次数 (默认 6)
            inference (bool): 是否推理模式 (清理显存)
            fine_tracking (bool): 是否启用 Fine 阶段 (默认 True)
            fine_chunk (int): Fine 阶段分块大小 (默认 40960)

        Returns:
            tuple: (fine_pred_track, coarse_pred_track, pred_vis, pred_score)
                - fine_pred_track:   [B, S, N, 2]  精细追踪结果 (亚像素精度)
                - coarse_pred_track: [B, S, N, 2]  粗追踪结果
                - pred_vis:          [B, S, N]     可见性分数
                - pred_score:        [B, S, N]     追踪置信度 (fine_tracking=True 时为 None)
        """

        # 【Step 1: 提取特征图 (若未提供)】
        if fmaps is None:
            batch_num, frame_num, image_dim, height, width = images.shape
            # 展平为 [B*S, 3, H, W] 通过 CNN
            reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)
            fmaps = self.process_images_to_fmaps(reshaped_image)
            # 恢复为 [B, S, C, HH, WW]
            fmaps = fmaps.reshape(batch_num, frame_num, -1, fmaps.shape[-2], fmaps.shape[-1])

            if inference:
                torch.cuda.empty_cache()

        # 【Step 2: Coarse Tracking (粗追踪)】
        # 在 1/8 分辨率特征图上做迭代追踪
        # iters=6: 6 轮迭代, 每轮用 correlation + transformer 更新坐标
        coarse_pred_track_lists, pred_vis = self.coarse_predictor(
            query_points=query_points, fmaps=fmaps, iters=coarse_iters, down_ratio=self.coarse_down_ratio
        )
        # 取最后一轮迭代的输出作为 coarse 结果
        coarse_pred_track = coarse_pred_track_lists[-1]

        if inference:
            torch.cuda.empty_cache()

        # 【Step 3: Fine Tracking (精细追踪)】
        if fine_tracking:
            # 调用 refine_track: 在 coarse 结果周围提取 patch, 做亚像素精修
            # pradius=15: patch 半径 15, patch 大小 = 15*2+1 = 31x31
            fine_pred_track, pred_score = refine_track(
                images, self.fine_fnet, self.fine_predictor, coarse_pred_track, compute_score=False, chunk=fine_chunk
            )

            if inference:
                torch.cuda.empty_cache()
        else:
            # 跳过 Fine 阶段, 直接用 Coarse 结果
            fine_pred_track = coarse_pred_track
            pred_score = torch.ones_like(pred_vis)

        return fine_pred_track, coarse_pred_track, pred_vis, pred_score

    def process_images_to_fmaps(self, images):
        """
        【提取 Coarse 特征图】

        将输入图像通过 BasicEncoder 提取特征。
        若 coarse_down_ratio > 1, 先对图像下采样再提取特征。

        例: 输入图像 1024x1024, coarse_down_ratio=2
            - 先下采样到 512x512
            - BasicEncoder(stride=4) 输出 128x128 特征图
            - 总下采样 = 2 * 4 = 8x

        Args:
            images (Tensor): 输入图像 [B*S, 3, H, W]

        Returns:
            Tensor: 特征图 [B*S, C, HH, WW]
        """
        if self.coarse_down_ratio > 1:
            # 先下采样图像以节省显存
            fmaps = self.coarse_fnet(
                F.interpolate(images, scale_factor=1 / self.coarse_down_ratio, mode="bilinear", align_corners=True)
            )
        else:
            fmaps = self.coarse_fnet(images)

        return fmaps
