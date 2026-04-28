# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .blocks import EfficientUpdateFormer, CorrBlock
from .utils import sample_features4d, get_2d_embedding, get_2d_sincos_pos_embed


class BaseTrackerPredictor(nn.Module):
    """
    【Base Tracker Predictor - 核心迭代追踪器，重点阅读】
    基于 Correlation + Transformer 的迭代式点追踪器。
    每轮迭代通过 correlation 特征更新坐标估计，逐渐收敛到目标位置。

    架构流程:
        1. 初始化 query 点坐标 → 在 query 帧提取初始特征
        2. 构建 CorrBlock: 所有帧的多尺度 correlation 体积
        3. 迭代优化 (iters 轮):
            a. 用当前坐标在 CorrBlock 中采样 correlation 特征
            b. 计算相对位移 flow (coords - coords[:, 0])
            c. flow → embedding, 拼接 correlation + track features
            d. Transformer (EfficientUpdateFormer) 预测坐标增量 delta
            e. 更新坐标: coords += delta
        4. 输出: 多轮迭代的坐标预测 (取最后一轮作为最终结果)

    Modified from https://github.com/facebookresearch/co-tracker/
    """

    def __init__(
        self,
        stride=4,               # 特征图下采样倍数 (coarse=4, fine=1)
        corr_levels=5,          # correlation 金字塔层数 (多尺度)
        corr_radius=4,          # correlation 采样半径 (局部窗口大小)
        latent_dim=128,         # 追踪特征维度
        hidden_size=384,        # Transformer 隐藏层维度
        use_spaceatt=True,      # 是否使用空间注意力
        depth=6,                # Transformer 层数
        fine=False,             # fine=True 时只做追踪，不预测 visibility
    ):
        super(BaseTrackerPredictor, self).__init__()

        self.stride = stride
        self.latent_dim = latent_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.hidden_size = hidden_size
        self.fine = fine

        # 【Flow Embedding 维度】
        self.flows_emb_dim = latent_dim // 2
        # 【Transformer 输入维度】
        # correlation 特征维度 + track features 维度 (x2, 因为要做 correlation)
        # corrdim = corr_levels * (2*r+1)^2, 即多尺度局部窗口内的 correlation 值
        self.transformer_dim = self.corr_levels * (self.corr_radius * 2 + 1) ** 2 + self.latent_dim * 2

        # 【维度对齐】确保 transformer_dim 是特定倍数的整数，便于高效计算
        if self.fine:
            # TODO this is the old dummy code, will remove this when we train next model
            self.transformer_dim += 4 if self.transformer_dim % 2 == 0 else 5
        else:
            self.transformer_dim += (4 - self.transformer_dim % 4) % 4

        # 【Transformer 深度配置】
        # space_depth: 空间注意力层数 (点之间的 attention)
        # time_depth: 时间注意力层数 (帧之间的 attention)
        space_depth = depth if use_spaceatt else 0
        time_depth = depth

        # 【核心更新网络: EfficientUpdateFormer】
        # 输入: 拼接后的 correlation + flow + track features
        # 输出: 2D 坐标增量 (2) + track feature 增量 (latent_dim)
        self.updateformer = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.transformer_dim,
            hidden_size=self.hidden_size,
            output_dim=self.latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=use_spaceatt,
        )

        # 【Track Feature 归一化】
        self.norm = nn.GroupNorm(1, self.latent_dim)

        # 【Track Feature 更新层】
        # 每轮迭代用 MLP 更新 track features (残差连接)
        self.ffeat_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())

        # 【可见性预测器】
        # fine=True 时不预测 visibility (只在 patch 内追踪，patch 内默认可见)
        if not self.fine:
            self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

    def forward(self, query_points, fmaps=None, iters=4, return_feat=False, down_ratio=1):
        """
        【BaseTrackerPredictor 前向传播 - 核心迭代追踪算法，重点阅读】

        迭代式追踪流程 (类似 RAFT):
            1. 将 query 点坐标缩放到特征图分辨率
            2. 初始化所有帧的追踪坐标为 query 点位置 (假设初始不动)
            3. 提取 query 点的初始特征作为 track_feats 种子
            4. 构建 CorrBlock (多帧特征图的 correlation 体积)
            5. 迭代 iters 轮:
                a. 用当前坐标在 CorrBlock 中采样局部 correlation
                b. 计算相对位移 flow (当前坐标 - query 坐标)
                c. 将 flow, correlation, track_feats 拼成 Transformer 输入
                d. Transformer 预测坐标增量 delta 和特征增量 delta_feats
                e. 更新坐标和 track features
                f. 强制第 0 帧坐标不变 (query 帧固定)
            6. 返回每轮迭代的坐标预测 (通常取最后一轮)

        Args:
            query_points: [B, N, 2]  query 点坐标 (原始图像尺度)
            fmaps:        [B, S, C, HH, WW]  多帧特征图 (已下采样)
            iters:        迭代轮数 (coarse 默认 6, fine 默认 4)
            return_feat:  是否返回特征 (用于后续计算)
            down_ratio:   图像额外下采样比例 (coarse 阶段 = 2)

        Returns:
            coord_preds: list of [B, S, N, 2], 每轮迭代的坐标预测 (原始图像尺度)
            vis_e:       [B, S, N] 可见性分数 (fine=True 时为 None)
        """
        B, N, D = query_points.shape
        B, S, C, HH, WW = fmaps.shape

        assert D == 2

        # 【Step 1: 坐标缩放】
        # 输入 query_points 是原始图像坐标，需要缩放到特征图分辨率
        # 缩放因子 = down_ratio * stride
        # 例: 1024x1024 图像 → down_ratio=2 下采样到 512x512 → stride=4 特征图 128x128
        #     query_points 需要 / (2*4) = /8 才能对应到特征图坐标
        if down_ratio > 1:
            query_points = query_points / float(down_ratio)
        query_points = query_points / float(self.stride)

        # 【Step 2: 初始化追踪坐标】
        # 所有帧的初始坐标都设为 query 点的位置
        # 即假设追踪点在每一帧都从 query 位置开始搜索
        coords = query_points.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1)

        # 【Step 3: 提取 query 帧的特征】
        # 在 query 帧 (第 0 帧) 的 query 点位置采样特征
        # 这些特征将作为每轮迭代的 "目标模板"，用于计算 correlation
        query_track_feat = sample_features4d(fmaps[:, 0], coords[:, 0])

        # 将 query 特征复制到所有帧，作为初始 track_feats
        # track_feats: [B, S, N, C]  每帧每个追踪点都有一个特征向量
        track_feats = query_track_feat.unsqueeze(1).repeat(1, S, 1, 1)  # B, S, N, C

        # 备份初始坐标，第 0 帧(query 帧) 坐标永远不变
        coords_backup = coords.clone()

        # 【Step 4: 构建 Correlation Block】
        # CorrBlock 预先计算所有帧特征图之间的 correlation 体积
        # 支持多尺度 (num_levels) 和局部窗口采样 (radius)
        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        coord_preds = []

        # 【Step 5: 迭代优化循环】
        # 每轮迭代: correlation 采样 → Transformer → 更新坐标
        for itr in range(iters):
            # 断开梯度 (经验上性能影响不大，但节省显存)
            coords = coords.detach()

            # 【5a. 计算 Correlation 特征】
            # 用当前 track_feats 与 fmaps 计算 correlation (见 CorrBlock 实现)
            # 本质: 在当前坐标位置提取局部窗口内的特征相似度
            fcorr_fn.corr(track_feats)
            fcorrs = fcorr_fn.sample(coords)  # B, S, N, corrdim

            corrdim = fcorrs.shape[3]

            # 重排维度: [B, N, S, corrdim] → [B*N, S, corrdim]
            # 这样 Transformer 可以并行处理所有追踪点
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, corrdim)

            # 【5b. 计算相对位移 Flow】
            # flow = 当前坐标 - query 坐标 (第 0 帧)
            # 表示追踪点从 query 位置移动了多少
            flows = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)

            # 【5c. Flow Embedding】
            # 将 2D flow 映射到高维 embedding (类似 positional encoding)
            flows_emb = get_2d_embedding(flows, self.flows_emb_dim, cat_coords=False)
            # 拼接原始 flow 坐标 (给 Transformer 提供绝对位置信息)
            flows_emb = torch.cat([flows_emb, flows], dim=-1)

            # 【5d. 准备 Transformer 输入】
            # track_feats: [B, S, N, C] → [B*N, S, C]
            track_feats_ = track_feats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            # 拼接三大输入: flow embedding + correlation + track features
            transformer_input = torch.cat([flows_emb, fcorrs_, track_feats_], dim=2)

            # 【维度对齐】如果实际维度 < 预设维度，用零填充
            if transformer_input.shape[2] < self.transformer_dim:
                pad_dim = self.transformer_dim - transformer_input.shape[2]
                pad = torch.zeros_like(flows_emb[..., 0:pad_dim])
                transformer_input = torch.cat([transformer_input, pad], dim=2)

            # 【5e. 2D 位置编码】
            # 从 query 点坐标采样位置编码，加到输入上
            pos_embed = get_2d_sincos_pos_embed(self.transformer_dim, grid_size=(HH, WW)).to(query_points.device)
            sampled_pos_emb = sample_features4d(pos_embed.expand(B, -1, -1, -1), coords[:, 0])
            sampled_pos_emb = rearrange(sampled_pos_emb, "b n c -> (b n) c").unsqueeze(1)

            x = transformer_input + sampled_pos_emb

            # 重排为 Transformer 需要的维度: [B, N, S, C]
            x = rearrange(x, "(b n) s d -> b n s d", b=B)

            # 【5f. Transformer 预测增量】
            # 输出: [B, N, S, latent_dim+2]  前 2 维是坐标增量，后 latent_dim 是特征增量
            delta = self.updateformer(x)
            delta = rearrange(delta, " b n s d -> (b n) s d", b=B)
            delta_coords_ = delta[:, :, :2]
            delta_feats_ = delta[:, :, 2:]

            # 【5g. 更新 track features】
            # 用残差方式更新: new_feat = old_feat + MLP(norm(delta_feats))
            track_feats_ = track_feats_.reshape(B * N * S, self.latent_dim)
            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
            track_feats_ = self.ffeat_updater(self.norm(delta_feats_)) + track_feats_
            track_feats = track_feats_.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)  # BxSxNxC

            # 【5h. 更新坐标】
            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)

            # 【强制 query 帧坐标不变】
            # query 帧 (第 0 帧) 的坐标应该永远等于初始 query 点
            coords[:, 0] = coords_backup[:, 0]

            # 【坐标还原到原始图像尺度】
            if down_ratio > 1:
                coord_preds.append(coords * self.stride * down_ratio)
            else:
                coord_preds.append(coords * self.stride)

        # 【Step 6: 预测可见性】
        # fine 模式不预测 visibility (patch 内默认可见)
        if not self.fine:
            vis_e = self.vis_predictor(track_feats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
            vis_e = torch.sigmoid(vis_e)
        else:
            vis_e = None

        if return_feat:
            return coord_preds, vis_e, track_feats, query_track_feat
        else:
            return coord_preds, vis_e
