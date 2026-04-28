# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Modified from https://github.com/facebookresearch/co-tracker/


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
import collections
from torch import Tensor
from itertools import repeat

from .utils import bilinear_sampler

from .modules import Mlp, AttnBlock, CrossAttnBlock, ResidualBlock


class BasicEncoder(nn.Module):
    """
    【Coarse 阶段特征编码器 - 重点阅读】
    ResNet-like CNN，用于提取多尺度特征。
    在 Coarse Tracking 中使用 (stride=4)，将图像下采样到 1/8 分辨率。

    架构特点:
        - 4 个尺度层 (layer1-4)，每层包含 2 个 ResidualBlock
        - 多尺度特征融合: 将 4 个尺度的特征上采样到同一分辨率后拼接
        - 输出固定维度的特征图，供 CorrBlock 和 Tracker 使用

    Args:
        input_dim:  输入通道数 (默认 3，RGB 图像)
        output_dim: 输出特征维度 (默认 128)
        stride:     总下采样倍数 (默认 4，配合 down_ratio=2 达到 8x)
    """

    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BasicEncoder, self).__init__()

        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2

        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        # 【Stem: 7x7 卷积，stride=2，初步下采样】
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode="zeros")
        self.relu1 = nn.ReLU(inplace=True)

        # 【4 个尺度层，stride 逐渐增大】
        # layer1: stride=1 (保持分辨率)
        # layer2: stride=2 (1/2 分辨率)
        # layer3: stride=2 (1/4 分辨率)
        # layer4: stride=2 (1/8 分辨率)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        # 【多尺度特征融合头】
        # 将 4 个尺度的特征拼接后，用 1x1 卷积融合
        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4, output_dim * 2, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        # 【Stem 卷积】7x7, stride=2 → 分辨率 1/2
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # 【多尺度特征提取】
        # a: layer1 输出 (stride=1, 1/2 分辨率)
        # b: layer2 输出 (stride=2, 1/4 分辨率)
        # c: layer3 输出 (stride=2, 1/8 分辨率)
        # d: layer4 输出 (stride=2, 1/16 分辨率)
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        # 【多尺度特征上采样到统一分辨率】
        # 统一缩放到 stride 分辨率 (如 stride=4，则统一为 1/4 分辨率)
        a = _bilinear_intepolate(a, self.stride, H, W)
        b = _bilinear_intepolate(b, self.stride, H, W)
        c = _bilinear_intepolate(c, self.stride, H, W)
        d = _bilinear_intepolate(d, self.stride, H, W)

        # 【拼接 + 融合】
        # 将 4 个尺度的特征沿通道拼接，用 3x3 和 1x1 卷积融合
        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class ShallowEncoder(nn.Module):
    """
    【Fine 阶段特征编码器 - 重点阅读】
    轻量级 CNN，用于提取全分辨率特征 (stride=1)。
    在 Fine Tracking 中使用，对图像不做下采样，保留完整细节。

    与 BasicEncoder 的区别:
        - 更浅的网络 (只有 2 层)
        - stride=1，输出分辨率与输入相同
        - 输出维度更小 (默认 32)，因为只在 patch 内处理

    Args:
        input_dim:  输入通道数 (默认 3)
        output_dim: 输出特征维度 (默认 32)
        stride:     下采样倍数 (Fine 阶段 = 1)
        norm_fn:    归一化方式 ("instance" / "batch" / "group" / "none")
    """

    def __init__(self, input_dim=3, output_dim=32, stride=1, norm_fn="instance"):
        super(ShallowEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn
        self.in_planes = output_dim

        # 根据 norm_fn 选择归一化层
        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        # 【Stem: 3x3 卷积，stride=2】
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=3, stride=2, padding=1, padding_mode="zeros")
        self.relu1 = nn.ReLU(inplace=True)

        # 【2 个 ResidualBlock】
        self.layer1 = self._make_layer(output_dim, stride=2)
        self.layer2 = self._make_layer(output_dim, stride=2)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        self.in_planes = dim

        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        return layer1

    def forward(self, x):
        _, _, H, W = x.shape

        # 【Stem】3x3 卷积，stride=2
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # 【残差连接 + 上采样】
        # layer1 输出下采样了，需要上采样回原分辨率再相加
        tmp = self.layer1(x)
        x = x + F.interpolate(tmp, (x.shape[-2:]), mode="bilinear", align_corners=True)
        tmp = self.layer2(tmp)
        x = x + F.interpolate(tmp, (x.shape[-2:]), mode="bilinear", align_corners=True)
        tmp = None
        # 1x1 卷积 + 残差
        x = self.conv2(x) + x

        # 调整到目标分辨率 (stride=1 时分辨率不变)
        x = F.interpolate(x, (H // self.stride, W // self.stride), mode="bilinear", align_corners=True)

        return x


def _bilinear_intepolate(x, stride, H, W):
    return F.interpolate(x, (H // stride, W // stride), mode="bilinear", align_corners=True)


class EfficientUpdateFormer(nn.Module):
    """
    【EfficientUpdateFormer - 追踪更新 Transformer，重点阅读】
    同时建模时间 (帧间) 和空间 (点间) 关系的 Transformer，
    用于预测坐标增量和特征增量。

    核心设计:
        - 时间注意力 (Time Attention): 每帧的 token 在时间轴上做 self-attention
        - 空间注意力 (Space Attention): 用 virtual tracks 做点间的 cross-attention
          避免 O(N^2) 的全点 attention，用少量 virtual tracks (64个) 作为中介
        - 交替执行: time attn → (可选) space attn → time attn → ...

    Args:
        space_depth:         空间注意力层数
        time_depth:          时间注意力层数
        input_dim:           输入维度 (flow_emb + corr + track_feats)
        hidden_size:         Transformer 隐藏层维度
        num_heads:           注意力头数
        output_dim:          输出维度 (2 + latent_dim)
        mlp_ratio:           MLP 隐藏层比例
        add_space_attn:      是否启用空间注意力
        num_virtual_tracks:  virtual track 数量 (默认 64)
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
    ):
        super().__init__()

        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        # 【输入投影】将拼接后的特征投影到 hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        # 【输出头】从 hidden_size 映射到输出 (坐标增量 + 特征增量)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks

        # 【Virtual Tracks】
        # 少量可学习的虚拟追踪点，作为空间 attention 的中介
        # 避免所有真实追踪点之间做 O(N^2) attention
        if self.add_space_attn:
            self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        else:
            self.virual_tracks = None

        # 【时间注意力块】
        # 处理每帧在时间维度上的关系 (S 帧之间)
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(time_depth)
            ]
        )

        # 【空间注意力块】(可选)
        # 用 virtual tracks 做中介，建模点之间的关系
        if add_space_attn:
            # Virtual tracks 之间的 self-attention
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                    for _ in range(space_depth)
                ]
            )
            # 真实点 → virtual tracks 的 cross-attention
            self.space_point2virtual_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(space_depth)]
            )
            # Virtual tracks → 真实点的 cross-attention
            self.space_virtual2point_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(space_depth)]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def init_weights_vit_timm(module: nn.Module, name: str = ""):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_tensor, mask=None):
        """
        【EfficientUpdateFormer 前向传播】

        输入: [B, N, S, input_dim]  (B=batch, N=tracks, S=frames)
        输出: [B, N, S, output_dim]  (前 2 维是坐标增量，后 latent_dim 是特征增量)

        处理流程:
            1. 输入投影到 hidden_size
            2. 如果启用 space_attn，拼接 virtual tracks
            3. 交替执行 time attention 和 space attention
            4. 残差连接 + 输出头
        """
        # 【输入投影】
        tokens = self.input_transform(input_tensor)
        init_tokens = tokens

        B, _, T, _ = tokens.shape

        # 【拼接 Virtual Tracks】
        # virtual tracks 是可学习的参数，在所有帧和 batch 上共享
        if self.add_space_attn:
            virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
            tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape

        # 【交替 Time / Space Attention】
        j = 0
        for i in range(len(self.time_blocks)):
            # 【Time Attention】
            # 维度重排: [B, N, T, C] → [B*N, T, C]
            # 即在时间维度 T 上做 self-attention (帧与帧之间的关系)
            time_tokens = tokens.contiguous().view(B * N, T, -1)
            time_tokens = self.time_blocks[i](time_tokens)
            tokens = time_tokens.view(B, N, T, -1)

            # 【Space Attention】
            # 每隔一定间隔插入一次空间注意力
            # 建模点之间的关系 (避免 O(N^2)，用 virtual tracks 做中介)
            if self.add_space_attn and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0):
                # 维度重排: [B, N, T, C] → [B*T, N, C]
                # 即在空间维度 N 上做 attention
                space_tokens = tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)

                # 分离真实点和 virtual tracks
                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                # 1. 真实点 → virtual tracks (聚合信息)
                virtual_tokens = self.space_virtual2point_blocks[j](virtual_tokens, point_tokens, mask=mask)
                # 2. Virtual tracks 之间 self-attention
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                # 3. Virtual tracks → 真实点 (分发信息)
                point_tokens = self.space_point2virtual_blocks[j](point_tokens, virtual_tokens, mask=mask)

                # 拼接回一起
                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)
                j += 1

        # 【移除 Virtual Tracks】只保留真实点的输出
        if self.add_space_attn:
            tokens = tokens[:, : N - self.num_virtual_tracks]

        # 【残差连接】
        tokens = tokens + init_tokens

        # 【输出头】映射到输出维度 (坐标增量 + 特征增量)
        flow = self.flow_head(tokens)
        return flow


class CorrBlock:
    """
    【Correlation Block - 相关体积计算与采样，重点阅读】
    计算 query 点特征与所有帧特征图之间的 correlation (点积相似度)，
    构建多尺度 correlation 金字塔，支持局部窗口采样。

    核心流程:
        1. corr():   用 query track features 与所有帧特征图做矩阵乘法，得到 correlation 图
        2. sample(): 在当前坐标位置提取局部窗口内的 correlation 值

    为什么用 Correlation:
        - 直接计算特征相似度，不需要学习参数
        - 对 appearance change 鲁棒 (比直接回归坐标更稳定)
        - 多尺度金字塔处理大位移
    """

    def __init__(self, fmaps, num_levels=4, radius=4, multiple_track_feats=False, padding_mode="zeros"):
        """
        构建特征金字塔 (多尺度)。

        Args:
            fmaps:                [B, S, C, H, W] 多帧特征图
            num_levels:           correlation 金字塔层数 (默认 4 层)
            radius:               局部采样半径 (默认 4，窗口大小 = 2*r+1 = 9x9)
            multiple_track_feats: 是否将 track features 分成多组 (每层用不同组)
            padding_mode:         边界填充方式
        """
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.multiple_track_feats = multiple_track_feats

        # 【构建特征金字塔】
        # 第 0 层: 原始分辨率
        # 第 1 层: 1/2 分辨率 (avg_pool2d)
        # 第 2 层: 1/4 分辨率
        # 第 3 层: 1/8 分辨率
        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        """
        【Correlation 局部采样】

        在当前坐标周围提取局部窗口内的 correlation 值。
        多尺度金字塔每层都提取一个 (2r+1)^2 的窗口，最后拼接。

        Args:
            coords: [B, S, N, 2] 当前追踪坐标 (特征图尺度)

        Returns:
            [B, S, N, L*(2r+1)^2]  多尺度局部 correlation 特征
        """
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []

        # 遍历每层金字塔
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # [B, S, N, H, W]
            *_, H, W = corrs.shape

            # 【构建局部采样网格】
            # 以当前坐标为中心，半径 r 的方形窗口
            # 例: r=4 → 9x9 网格，共 81 个点
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(coords.device)

            # 【坐标缩放】
            # 第 i 层金字塔分辨率是原始分辨率的 1/2^i
            # 所以当前坐标也要相应缩放
            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            # 【双线性采样】
            # 在 correlation 图上用双线性插值采样局部窗口
            corrs = bilinear_sampler(corrs.reshape(B * S * N, 1, H, W), coords_lvl, padding_mode=self.padding_mode)
            corrs = corrs.view(B, S, N, -1)

            out_pyramid.append(corrs)

        # 【拼接多尺度特征】
        # 每层贡献 (2r+1)^2 维，共 num_levels 层
        out = torch.cat(out_pyramid, dim=-1).contiguous()  # [B, S, N, num_levels*(2r+1)^2]
        return out

    def corr(self, targets):
        """
        【计算 Correlation 体积】

        用 query track features 与所有帧特征图做点积，
        得到每帧每个位置的相似度图 (correlation map)。

        Args:
            targets: [B, S, N, C]  query track features (从 query 点提取的特征)

        Returns:
            无 (结果存入 self.corrs_pyramid)
        """
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            # 将 track features 分成多组，每组对应一层金字塔
            targets_split = targets.split(C // self.num_levels, dim=-1)
            B, S, N, C = targets_split[0].shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets

        # 对每层金字塔计算 correlation
        self.corrs_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            # 将特征图展平为 [B, S, C, H*W]
            fmap2s = fmaps.view(B, S, C, H * W)

            if self.multiple_track_feats:
                fmap1 = targets_split[i]

            # 【矩阵乘法计算 correlation】
            # fmap1: [B, S, N, C]  ×  fmap2s: [B, S, C, H*W]
            # 结果:   [B, S, N, H*W]  表示每个 query 点与每个空间位置的相似度
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)

            # 【归一化】除以 sqrt(C) 防止数值过大
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)

    def corr(self, targets):
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            targets_split = targets.split(C // self.num_levels, dim=-1)
            B, S, N, C = targets_split[0].shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            if self.multiple_track_feats:
                fmap1 = targets_split[i]
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)
