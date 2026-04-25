# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """
    【相机位姿激活函数 - 重点阅读】
    对 CameraHead 输出的 pose_enc 的各部分应用激活函数。

    pose_enc 结构 (9维):
        - [:3]:  平移向量 translation
        - [3:7]: 四元数 quaternion (后续需归一化)
        - [7:]:  视场角 FoV / 焦距

    默认激活:
        - trans: linear (无激活)
        - quat:  linear (无激活, 后续 normalize)
        - fl:    relu (保证正值, 见 CameraHead 初始化)

    Args:
        pred_pose_enc: 位姿编码 [B, S, 9]
        trans_act: 平移分量激活类型
        quat_act:  四元数分量激活类型
        fl_act:    焦距/FOV 分量激活类型

    Returns:
        激活后的位姿编码
    """
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]  # or fov

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # or fov

    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """
    【DPTHead 输出激活 - 重点阅读】
    处理 DPTHead 的网络输出, 拆分为预测值和置信度, 并分别应用激活函数。

    输入: out (B, C, H, W)
    拆分:
        - xyz = out[:, :, :, :-1]  (前 C-1 通道): 预测值 (深度或3D坐标)
        - conf = out[:, :, :, -1]  (最后1通道): 置信度

    深度头 (output_dim=2):
        - xyz: 1维深度值
        - 激活: "exp" => torch.exp(xyz), 保证深度为正

    点云头 (output_dim=4):
        - xyz: 3维世界坐标
        - 激活: "inv_log" => inverse_log_transform, 允许正负值

    置信度激活 (conf_activation):
        - "expp1" => 1 + exp(conf), 保证置信度 > 1
        - "sigmoid" => sigmoid(conf), 范围 [0, 1]

    Args:
        out: 网络输出 (B, C, H, W)
        activation: 预测值激活类型
        conf_activation: 置信度激活类型

    Returns:
        tuple: (pts3d, conf_out)
            - pts3d (B, H, W, C-1): 激活后的预测值
            - conf_out (B, H, W): 激活后的置信度
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C expected

    # Split into xyz (first C-1 channels) and confidence (last channel)
    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


def inverse_log_transform(y):
    """
    【逆对数变换 - 点云头使用】
    公式: sign(y) * (exp(|y|) - 1)

    特性:
        - 可以输出正负值 (适合世界坐标, 中心可能在原点)
        - 小值时近似线性: 当 y≈0, 输出≈y
        - 大值时指数增长, 防止梯度消失

    对比 exp 激活:
        - exp: 只能输出正值 (适合深度值)
        - inv_log: 可正可负 (适合3D坐标)

    Args:
        y: 输入张量

    Returns:
        变换后的张量
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
