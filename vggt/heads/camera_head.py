# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose


class CameraHead(nn.Module):
    """
    【CameraHead - 相机参数预测头, 重点阅读】
    从 Aggregator 输出的 token 中预测相机参数 (外参+内参)。

    算法原理: 迭代 refine (iterative refinement)
        - 使用专门的 camera token (每帧第0个 token) 作为输入
        - 通过多轮迭代逐步优化相机位姿估计
        - 每一轮将上一轮预测结果 detach 后作为条件, 类似 DiT 的 AdaLN 调制机制
        - 最终输出 pose_enc, 需经 pose_encoding_to_extri_intri() 解码为:
          - extrinsic (外参): [B, S, 3, 4], OpenCV camera-from-world 格式
          - intrinsic (内参): [B, S, 3, 3]

    Pose Encoding 格式 (absT_quaR_FoV, 9维):
        - [0:3]:  平移向量 translation (绝对位置, 激活函数: linear)
        - [3:7]:  四元数 rotation quaternion (激活函数: linear, 后续会归一化)
        - [7:9]:  视场角 field-of-view (激活函数: relu, 保证正值)
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # 视场角激活: 确保 FOV 为正
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9  # 3 (translation) + 4 (quaternion) + 2 (fov)
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # 【Trunk】由 trunk_depth 个 Transformer Block 组成的序列
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # 归一化层
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # 【可学习的空位姿 token】用于第一轮迭代
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # 【AdaLN 调制模块】生成 shift, scale, gate 三个调制参数
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # 无仿射参数的 LayerNorm, 用于 AdaLN
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        # 输出分支: 预测 pose encoding 的残差增量
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        【前向传播】
        使用 Aggregator 最后一层输出的 token 预测相机参数。

        Args:
            aggregated_tokens_list (list): Aggregator 各轮输出的 token 列表
                【使用最后一层的输出, 即 aggregated_tokens_list[-1]】
            num_iterations (int): 迭代优化轮数 (默认 4)

        Returns:
            list: 每轮迭代的激活后 pose_enc, 每个元素形状 [B, S, 9]
                【VGGT.forward() 中只取最后一项: pose_enc_list[-1]】
        """
        # 使用最后一层的 aggregated_tokens 进行相机预测
        tokens = aggregated_tokens_list[-1]

        # 【提取 camera token】每帧的第0个 token 是 camera token
        # tokens 形状: [B, S, P, C], 取 [:, :, 0] 得 [B, S, C]
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        【迭代优化核心 - 重点阅读】
        逐步 refine 相机位姿预测。

        流程:
            1. 第一轮: 使用可学习的 empty_pose_tokens 作为初始条件
            2. 后续轮: 将上一轮预测的 pose_enc detach 后嵌入, 作为条件输入
            3. 通过 AdaLN 调制 pose_tokens
            4. 经过 trunk (Transformer blocks)
            5. pose_branch 预测残差增量 pred_pose_enc_delta
            6. 累加残差: pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
            7. 应用激活函数, 得到当前轮输出

        Args:
            pose_tokens (torch.Tensor): 归一化后的 camera token [B, S, C]
            num_iterations (int): 迭代轮数

        Returns:
            list: 每轮迭代的激活后 pose_enc [B, S, 9]
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # 第一轮使用可学习的空位姿作为初始值
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # 【关键】detach 上一轮预测, 避免 BPTT (backprop through time)
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # 生成调制参数并拆分为 shift, scale, gate
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # AdaLN 调制 + 残差连接
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            # 经过 trunk (Transformer blocks)
            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # 预测 pose_enc 的残差增量
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            # 累加残差
            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # 应用激活函数: translation(linear), quaternion(linear), fov(relu)
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    【AdaLN 调制函数】
    参考 DiT 论文: x * (1 + scale) + shift

    用于对输入特征进行条件化调制, 使网络能根据当前 pose 估计值调整注意力行为。
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
