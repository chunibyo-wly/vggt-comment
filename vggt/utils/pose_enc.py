# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .rotation import quat_to_mat, mat_to_quat


def extri_intri_to_pose_encoding(
    extrinsics, intrinsics, image_size_hw=None, pose_encoding_type="absT_quaR_FoV"  # e.g., (256, 512)
):
    """
    【编码器】将相机外参和内参编码为紧凑的 pose_encoding (CameraHead 的 ground truth 用)。

    Pose Encoding 格式 "absT_quaR_FoV" (9维):
        - [:3]:  绝对平移向量 T (camera center 在世界坐标系中的位置)
        - [3:7]: 旋转四元数 quat (需归一化)
        - [7:9]: 视场角 FoV (水平和垂直, 单位: 弧度)

    坐标系:
        - OpenCV 坐标系: x-right, y-down, z-forward
        - extrinsics 格式: [R | t], 表示 camera-from-world 变换

    Args:
        extrinsics (torch.Tensor): 外参 [B, S, 3, 4], [R|t] 格式
        intrinsics (torch.Tensor): 内参 [B, S, 3, 3]
        image_size_hw (tuple): 图像尺寸 (H, W), 用于计算 FoV
        pose_encoding_type (str): 编码类型, 目前只支持 "absT_quaR_FoV"

    Returns:
        torch.Tensor: Pose encoding [B, S, 9]
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        # Note the order of h and w here
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding, image_size_hw=None, pose_encoding_type="absT_quaR_FoV", build_intrinsics=True  # e.g., (256, 512)
):
    """
    【解码器 - 重点阅读】将 CameraHead 输出的 pose_encoding 解码为相机外参和内参。

    这是使用 VGGT 相机输出的关键步骤! 模型 forward 返回的 predictions["pose_enc"]
    必须经过此函数解码才能得到可用的相机矩阵。

    解码过程 (absT_quaR_FoV):
        - T = pose_encoding[..., :3]          # 直接作为平移向量
        - quat = pose_encoding[..., 3:7]      # 四元数 -> 归一化 -> 旋转矩阵 R
        - fov_h, fov_w = pose_encoding[..., 7:9]  # 视场角 -> 焦距 fx, fy
        - 主点 (cx, cy) 假设在图像中心 (W/2, H/2)

    输出格式:
        - extrinsics [B, S, 3, 4]: [R | t] 的 camera-from-world 变换矩阵
          【注意: 这是 OpenCV 约定, 即相机坐标系 = R * world + t】
        - intrinsics [B, S, 3, 3]:
          [[fx, 0, cx],
           [0, fy, cy],
           [0,  0,  1]]

    Args:
        pose_encoding (torch.Tensor): CameraHead 输出 [B, S, 9]
        image_size_hw (tuple): 图像尺寸 (H, W), 重建内参需要
        pose_encoding_type (str): 编码类型, 默认 "absT_quaR_FoV"
        build_intrinsics (bool): 是否重建内参矩阵

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics [B, S, 3, 4]: 外参矩阵
            - intrinsics [B, S, 3, 3] 或 None: 内参矩阵
    """

    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    else:
        raise NotImplementedError

    return extrinsics, intrinsics
