# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    【Aggregator - 核心模块, 重点阅读】
    交替注意力聚合器 (Alternating Attention Aggregator)。

    算法原理:
        对输入的多帧图像, 交替进行两种注意力:
        1. Frame Attention (帧内注意力): 每帧独立做 self-attention, 学习单帧内的空间关系
        2. Global Attention (全局注意力): 所有帧的所有 token 一起做 self-attention, 学习跨帧的时序/几何关系

        默认配置: depth=24, aa_order=["frame", "global"], aa_block_size=1
        即: frame block -> global block -> frame block -> global block ... 共 24 对

    Token 结构 (每帧):
        [camera_token] + [register_tokens] + [patch_tokens]
        - camera_token:   1个, 用于预测相机参数
        - register_tokens: 4个 (默认), 类似 ViT 的 register token, 用于聚合全局信息
        - patch_tokens:   (H/patch_size)*(W/patch_size) 个, 即 37*37=1369 个 (518/14=37)

    关键实现细节:
        - camera_token 和 register_token 的形状为 (1, 2, X, C):
          第0个位置用于第一帧, 第1个位置用于其余所有帧 (通过 slice_expand_and_flatten 展开)
        - 使用 DINOv2 ViT-Large 作为 patch embed (默认)
        - 使用 2D RoPE (Rotary Position Embedding) 给 patch tokens 添加空间位置信息
        - 训练时开启 gradient checkpointing 节省显存

    输出:
        - List[Tensor]: 每轮 (frame+global) 迭代后的拼接特征, 每个元素形状 [B, S, P, 2*C]
          其中 frame_intermediate 和 global_intermediate 在最后一维拼接
        - int: patch_start_idx, patch tokens 的起始索引 (camera+register 之后)

    Args:
        img_size (int): 图像尺寸 (默认 518)
        patch_size (int): Patch 大小 (默认 14)
        embed_dim (int): Token 嵌入维度 (默认 1024)
        depth (int): Block 总数 (默认 24)
        num_heads (int): 注意力头数 (默认 16)
        mlp_ratio (float): MLP 隐藏层倍数 (默认 4.0)
        num_register_tokens (int): register token 数量 (默认 4)
        block_fn (nn.Module): 注意力 Block 类型
        qkv_bias (bool): QKV 投影是否加偏置
        proj_bias (bool): 输出投影是否加偏置
        ffn_bias (bool): MLP 是否加偏置
        patch_embed (str): Patch Embed 类型, "conv" 或 "dinov2_vitl14_reg"
        aa_order (list[str]): 交替注意力的顺序, 如 ["frame", "global"]
        aa_block_size (int): 每种注意力类型连续处理的 block 数 (默认 1)
        qk_norm (bool): 是否对 QK 做 normalization
        rope_freq (int): RoPE 基频, -1 表示禁用
        init_values (float): Layer Scale 初始化值
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        【前向传播 - 重点阅读】
        交替注意力聚合流程。

        Args:
            images (torch.Tensor): 输入图像 [B, S, 3, H, W], 范围 [0, 1]
                B: batch size, S: 帧数, 3: RGB通道, H: 高, W: 宽

        Returns:
            (list[torch.Tensor], int):
                - 每轮迭代输出的特征列表, 每个元素 [B, S, P, 2*C]
                - patch_start_idx: patch tokens 的起始索引 (用于后续 head 定位 patch tokens)
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # 【图像归一化】使用 ImageNet 的 ResNet 均值和标准差
        images = (images - self._resnet_mean) / self._resnet_std

        # 展平为 [B*S, C, H, W] 以通过 patch embed
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        # DINOv2 patch embed 返回 dict, 取其中的 patch token
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # 【特殊 token 展开】camera_token 和 register_token 形状 (1, 2, X, C)
        # 第0个位置给第一帧用, 第1个位置给其余帧用
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # 拼接: [camera] + [register] + [patch]
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        # 【2D 位置编码 (RoPE)】
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # 特殊 token (camera, register) 不使用位置编码, 将其位置设为 0
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # 更新 P (增加了 camera + register tokens)
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        # 【交替注意力主循环】
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    # 帧内注意力: token 形状保持 (B*S, P, C)
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    # 全局注意力: token 形状变为 (B, S*P, C)
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # 将 frame 和 global 中间特征在最后一维拼接: [B, S, P, 2*C]
            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        【帧内注意力 - Frame Attention】
        每帧独立做 self-attention, Token 形状保持 (B*S, P, C)。
        即: 同一帧内的所有 token (camera+register+patch) 互相注意力,
            不同帧之间没有信息交换。
        """
        # 确保 token 形状为 (B*S, P, C)
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # 默认 aa_block_size=1, 每次处理一个 block
        for _ in range(self.aa_block_size):
            if self.training:
                # 训练时使用 gradient checkpointing 节省显存
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            # 保存中间结果, 形状恢复为 (B, S, P, C) 方便后续拼接
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        【全局注意力 - Global Attention】
        所有帧的所有 token 一起做 self-attention, Token 形状变为 (B, S*P, C)。
        即: 跨帧信息融合, 所有帧的 token 互相可见, 建立多帧几何一致性。
        """
        # 将多帧展平为 (B, S*P, C) 以便全局注意力
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # 默认 aa_block_size=1, 每次处理一个 block
        for _ in range(self.aa_block_size):
            if self.training:
                # 训练时使用 gradient checkpointing 节省显存
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            # 保存中间结果, 形状恢复为 (B, S, P, C) 方便后续拼接
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    【特殊 token 展开函数 - 重点理解】
    处理形状为 (1, 2, X, C) 的特殊 token (camera_token / register_token):

    设计意图:
        - 第一帧 (query frame) 使用不同的特殊 token (第0个位置)
        - 其余帧 (support frames) 共享相同的特殊 token (第1个位置)
        这体现了"第一帧作为参考帧"的归纳偏置。

    处理步骤:
        1) 第0个位置只给第一帧用, 第1个位置给其余 S-1 帧用
        2) 展开到 batch size B
        3) 拼接为 (B, S, X, C): 每帧序列 = 1个第0位置 token + (S-1)个第1位置 token
        4) 展平为 (B*S, X, C) 供注意力模块处理

    Args:
        token_tensor: 形状 (1, 2, X, C) 的特殊 token
        B: batch size
        S: 序列长度 (帧数)

    Returns:
        torch.Tensor: 处理后的 token, 形状 (B*S, X, C)
    """

    # 取出 "第一帧专用" token => 形状 (1, 1, X, C), 展开为 (B, 1, X, C)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # 取出 "其余帧共享" token => 形状 (1, 1, X, C), 展开为 (B, S-1, X, C)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # 拼接 => (B, S, X, C)
    combined = torch.cat([query, others], dim=1)

    # 展平 batch 和 sequence => (B*S, X, C)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
