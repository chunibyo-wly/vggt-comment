# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    【DPTHead - 密集预测头, 重点阅读】
    DPT (Vision Transformers for Dense Prediction) 风格的解码器头。
    用于深度估计 (Depth Head) 和点云估计 (Point Head)。

    算法原理:
        1. 从 Aggregator 的多个中间层提取特征 (默认取第 4, 11, 17, 23 层)
        2. 将不同尺度的 token 特征重塑为 2D 特征图
        3. 通过不同倍率的上/下采样对齐分辨率:
           - layer 4:  4x 上采样
           - layer 11: 2x 上采样
           - layer 17: 不变
           - layer 23: 2x 下采样
        4. 使用 Feature Fusion Block 从深层到浅层逐步融合特征
        5. 最终通过卷积输出 dense prediction

    输出格式 (非 feature_only 模式):
        - preds: [B, S, output_dim-1, H, W]  预测值 (深度或3D坐标)
        - conf:  [B, S, 1, H, W]             置信度

        注意: output_conv2 输出 output_dim 通道, activate_head 将其拆分为:
              preds (前 output_dim-1 通道) 和 conf (最后1通道)

    Args:
        dim_in (int): 输入维度 (2*embed_dim = 2048)
        patch_size (int): Patch 大小 (默认 14)
        output_dim (int): 输出通道数
            - depth head: 2  (1维深度 + 1维置信度)
            - point head: 4  (3维坐标 + 1维置信度)
        activation (str): 预测值激活函数 ("exp" 用于深度, "inv_log" 用于点云)
        conf_activation (str): 置信度激活函数 (默认 "expp1")
        features (int): 中间特征维度 (默认 256)
        out_channels (List[int]): 各层投影输出通道
        intermediate_layer_idx (List[int]): 使用的 Aggregator 中间层索引
        pos_embed (bool): 是否使用位置编码
        feature_only (bool): 若 True, 只返回融合后的特征图
        down_ratio (int): 输出分辨率下采样倍数
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        【DPTHead 前向传播 - 重点阅读】
        支持分块 (chunking) 处理帧以控制显存。

        Args:
            aggregated_tokens_list (List[Tensor]): Aggregator 各层输出的 token 列表
            images (Tensor): 输入图像 [B, S, 3, H, W], 范围 [0, 1]
            patch_start_idx (int): patch tokens 在 token 序列中的起始索引
                【用于跳过 camera token 和 register tokens, 只取 patch tokens】
            frames_chunk_size (int, optional): 每批处理的帧数 (默认 8)
                若 None 或大于 S, 则一次性处理所有帧

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - feature_only=True: 特征图 [B, S, C, H, W]
                - 否则: (predictions, confidence)
                    - depth head:  ([B, S, 1, H, W], [B, S, 1, H, W])
                    - point head: ([B, S, 3, H, W], [B, S, 1, H, W])
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []
        all_conf = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        # Concatenate results along the sequence dimension
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        【DPTHead 核心实现 - 重点阅读】
        处理指定帧块的前向传播。

        流程:
            1. 从 4 个中间层取 patch tokens (跳过 special tokens)
            2. 将 [B*S, num_patches, C] 重塑为 [B*S, C, patch_h, patch_w] 的 2D 特征图
            3. 1x1 卷积投影到目标通道数
            4. 上/下采样对齐分辨率
            5. 通过 scratch_forward 融合多层特征
            6. 双线性插值到目标分辨率
            7. output_conv2 输出预测, activate_head 拆分为 pred 和 conf

        Args:
            aggregated_tokens_list (List[Tensor]): Aggregator 各层 token 列表
            images (Tensor): 输入图像 [B, S, 3, H, W]
            patch_start_idx (int): patch tokens 起始索引
            frames_start_idx (int, optional): 帧块起始索引
            frames_end_idx (int, optional): 帧块结束索引

        Returns:
            Tensor or Tuple[Tensor, Tensor]: 特征图 或 (predictions, confidence)
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # 【Step 1: 提取多层特征】
        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            # 取 patch tokens: 跳过 camera 和 register tokens
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # 如果是分块处理, 只取当前帧块
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            # [B, S, num_patches, C] -> [B*S, num_patches, C]
            x = x.reshape(B * S, -1, x.shape[-1])

            x = self.norm(x)

            # [B*S, num_patches, C] -> [B*S, C, patch_h, patch_w]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            # 1x1 卷积投影
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            # 上/下采样对齐分辨率
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # 【Step 2: 特征融合】
        out = self.scratch_forward(out)
        # 插值到目标分辨率
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        # 【Step 3: 输出预测】
        out = self.scratch.output_conv2(out)
        # activate_head 将 output_dim 通道拆分为: preds (前 output_dim-1 通道) + conf (最后1通道)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        # 恢复形状: [B*S, C, H, W] -> [B, S, C, H, W]
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        【特征融合 - 重点阅读】
        从深层到浅层逐步融合多尺度特征 (类似 U-Net 的解码器路径)。

        融合顺序 (从高分辨率到低分辨率):
            refinenet4: layer_4 (最深, 分辨率最低) 上采样到 layer_3 分辨率
            refinenet3: 融合结果 + layer_3 上采样到 layer_2 分辨率
            refinenet2: 融合结果 + layer_2 上采样到 layer_1 分辨率
            refinenet1: 融合结果 + layer_1 (最浅, 分辨率最高)

        Args:
            features (List[Tensor]): 4个尺度的特征图列表

        Returns:
            Tensor: 融合后的特征图
        """
        layer_1, layer_2, layer_3, layer_4 = features

        # 先通过 3x3 卷积统一通道数
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 从深层到浅层逐步融合并上采样
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
