# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from lightglue import ALIKED, SIFT, SuperPoint

from .vggsfm_tracker import TrackerPredictor

# Suppress verbose logging from dependencies
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# Constants
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def build_vggsfm_tracker(model_path=None):
    """
    Build and initialize the VGGSfM tracker.

    Args:
        model_path: Path to the model weights file. If None, weights are downloaded from HuggingFace.

    Returns:
        Initialized tracker model in eval mode.
    """
    tracker = TrackerPredictor()

    if model_path is None:
        default_url = "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
        tracker.load_state_dict(torch.hub.load_state_dict_from_url(default_url))
    else:
        tracker.load_state_dict(torch.load(model_path))

    tracker.eval()
    return tracker


def generate_rank_by_dino(
    images, query_frame_num, image_size=336, model_name="dinov2_vitb14_reg", device="cuda", spatial_similarity=False
):
    """
    【DINO 特征帧排序 - 重点阅读】
    用 DINOv2 特征计算帧间相似度, 选择最具代表性的 query 帧。

    算法:
        1. 用 DINOv2 提取每帧的 CLS token 特征
        2. 计算帧间余弦相似度矩阵
        3. 找出与其他帧最相似的帧作为起始帧 (most_common_frame)
        4. 从起始帧出发, 用 Farthest Point Sampling (FPS) 选出 query_frame_num 帧
           保证选出的帧在特征空间中尽可能分散 (覆盖不同视角)

    为什么用 FPS:
        - 如果选出的 query 帧太相似, 它们提取的关键点会重复追踪同一区域
        - FPS 保证 query 帧覆盖不同视角, 提高整体追踪覆盖率

    Args:
        images: 输入图像 (S, 3, H, W), 范围 [0, 1]
        query_frame_num: 要选多少帧作为 query 帧
        image_size: DINO 处理分辨率 (默认 336)
        model_name: DINO 模型名 (默认 dinov2_vitb14_reg)
        device: 计算设备
        spatial_similarity: 是否用 patch token 而非 CLS token

    Returns:
        List[int]: 选出的帧索引列表 (按 FPS 顺序)
    """
    # Resize images to the target size
    images = F.interpolate(images, (image_size, image_size), mode="bilinear", align_corners=False)

    # Load DINO model
    dino_v2_model = torch.hub.load("facebookresearch/dinov2", model_name)
    dino_v2_model.eval()
    dino_v2_model = dino_v2_model.to(device)

    # Normalize images using ResNet normalization
    resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
    resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
    images_resnet_norm = (images - resnet_mean) / resnet_std

    with torch.no_grad():
        frame_feat = dino_v2_model(images_resnet_norm, is_training=True)

    # Process features based on similarity type
    if spatial_similarity:
        frame_feat = frame_feat["x_norm_patchtokens"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

        # Compute the similarity matrix
        frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
        similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
        similarity_matrix = similarity_matrix.mean(dim=0)
    else:
        frame_feat = frame_feat["x_norm_clstoken"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)
        similarity_matrix = torch.mm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))

    distance_matrix = 100 - similarity_matrix.clone()

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(-100)
    similarity_sum = similarity_matrix.sum(dim=1)

    # Find the most common frame
    most_common_frame_index = torch.argmax(similarity_sum).item()

    # Conduct FPS sampling starting from the most common frame
    fps_idx = farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index)

    # Clean up all tensors and models to free memory
    del frame_feat, frame_feat_norm, similarity_matrix, distance_matrix
    del dino_v2_model
    torch.cuda.empty_cache()

    return fps_idx


def farthest_point_sampling(distance_matrix, num_samples, most_common_frame_index=0):
    """
    【最远点采样 FPS - 重点阅读】
    从帧集合中选出特征差异最大的帧, 保证视角多样性。

    算法步骤:
        1. 从 most_common_frame (与其他帧最相似的帧) 开始
        2. 每轮选择距离当前已选集合最远的帧
        3. 直到选出 num_samples 个帧

    等价于在特征空间均匀采样, 避免选出的 query 帧扎堆在同一视角。

    Args:
        distance_matrix: 帧间距离矩阵 (S, S)
        num_samples: 要选出的帧数
        most_common_frame_index: 起始帧索引

    Returns:
        List[int]: 选出的帧索引列表
    """
    distance_matrix = distance_matrix.clamp(min=0)
    N = distance_matrix.size(0)

    # Initialize with the most common frame
    selected_indices = [most_common_frame_index]
    check_distances = distance_matrix[selected_indices]

    while len(selected_indices) < num_samples:
        # Find the farthest point from the current set of selected points
        farthest_point = torch.argmax(check_distances)
        selected_indices.append(farthest_point.item())

        check_distances = distance_matrix[farthest_point]
        # Mark already selected points to avoid selecting them again
        check_distances[selected_indices] = 0

        # Break if all points have been selected
        if len(selected_indices) == N:
            break

    return selected_indices


def calculate_index_mappings(query_index, S, device=None):
    """
    【帧顺序重排 - 重点阅读】
    构造索引映射, 将 query_index 位置的元素换到第 0 位。

    tracker 内部假设第 0 帧是 query 帧, 所以需要将当前 query 帧
    通过索引重排移到第 0 位, 追踪完后再 swap 回来。

    例: query_index=3, S=5
        原始顺序: [0, 1, 2, 3, 4]
        重排后:   [3, 1, 2, 0, 4]

    Args:
        query_index: 要移到第 0 位的帧索引
        S: 总帧数
        device: 计算设备

    Returns:
        Tensor: 重排后的索引顺序 (S,)
    """
    new_order = torch.arange(S)
    # 将 query_index 和 0 互换位置
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order


def switch_tensor_order(tensors, order, dim=1):
    """
    Reorder tensors along a specific dimension according to the given order.

    Args:
        tensors: List of tensors to reorder
        order: Tensor of indices specifying the new order
        dim: Dimension along which to reorder

    Returns:
        List of reordered tensors
    """
    return [torch.index_select(tensor, dim, order) if tensor is not None else None for tensor in tensors]


def initialize_feature_extractors(max_query_num, det_thres=0.005, extractor_method="aliked", device="cuda"):
    """
    【初始化关键点提取器 - 重点阅读】
    根据方法字符串初始化一个或多个特征点提取器。

    支持的方法 (用 '+' 组合):
        - "aliked":  ALIKED 提取器, 速度快, 适合纹理丰富区域
        - "sp":      SuperPoint, 深度学习检测器, 泛化性好
        - "sift":    SIFT, 传统方法, 对尺度/旋转鲁棒

    组合示例:
        - "aliked+sp":      先用 ALIKED 再用 SuperPoint 补充 (默认)
        - "sp+sift+aliked": 三种都用, 提取最多特征点 (最终尝试模式用)

    输出字典结构: {"aliked": ALIKED_model, "sp": SuperPoint_model, ...}

    Args:
        max_query_num: 每帧最多提取多少关键点
        det_thres: 检测阈值 (越低关键点越多)
        extractor_method: 提取器方法字符串 (如 "aliked+sp")
        device: 计算设备

    Returns:
        Dict: 提取器名称到模型实例的字典
    """
    extractors = {}
    methods = extractor_method.lower().split("+")

    for method in methods:
        method = method.strip()
        if method == "aliked":
            aliked_extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres)
            extractors["aliked"] = aliked_extractor.to(device).eval()
        elif method == "sp":
            sp_extractor = SuperPoint(max_num_keypoints=max_query_num, detection_threshold=det_thres)
            extractors["sp"] = sp_extractor.to(device).eval()
        elif method == "sift":
            sift_extractor = SIFT(max_num_keypoints=max_query_num)
            extractors["sift"] = sift_extractor.to(device).eval()
        else:
            print(f"Warning: Unknown feature extractor '{method}', ignoring.")

    if not extractors:
        print(f"Warning: No valid extractors found in '{extractor_method}'. Using ALIKED by default.")
        aliked_extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres)
        extractors["aliked"] = aliked_extractor.to(device).eval()

    return extractors


def extract_keypoints(query_image, extractors, round_keypoints=True):
    """
    【关键点提取 - 重点阅读】
    用预初始化的提取器在 query 帧上提取关键点坐标。

    处理流程:
        1. 遍历所有提取器 (如 ALIKED, SuperPoint)
        2. 每个提取器输出关键点坐标 (1, N_i, 2)
        3. 将所有提取器的结果沿 N 维度拼接
        4. 返回合并后的关键点 (1, N_total, 2)

    坐标格式: (x, y), 其中 x 是列坐标, y 是行坐标, 与 OpenCV 一致。

    Args:
        query_image: 输入图像 (3, H, W), 范围 [0, 1]
        extractors: 提取器字典 (来自 initialize_feature_extractors)
        round_keypoints: 是否将坐标 round 到整数

    Returns:
        Tensor: 关键点坐标 (1, N, 2)
    """
    query_points = None

    with torch.no_grad():
        for extractor_name, extractor in extractors.items():
            query_points_data = extractor.extract(query_image, invalid_mask=None)
            extractor_points = query_points_data["keypoints"]
            if round_keypoints:
                extractor_points = extractor_points.round()

            if query_points is not None:
                query_points = torch.cat([query_points, extractor_points], dim=1)
            else:
                query_points = extractor_points

    return query_points


def predict_tracks_in_chunks(
    track_predictor, images_feed, query_points_list, fmaps_feed, fine_tracking, num_splits=None, fine_chunk=40960
):
    """
    【分块追踪 - 重点阅读】
    将大量 query 点拆分成多个 chunk 逐个追踪, 防止 GPU OOM。

    原理:
        tracker 的计算量与 (帧数 × 关键点数) 成正比。
        当总点数 all_points_num = S × N 超过 max_points_num 时,
        将 query_points 沿 N 维度拆成多个 chunk, 分别送入 tracker,
        最后将结果拼接。

    输入/输出维度:
        - images_feed:    (1, S, 3, H, W)
        - query_points:   list of (1, N_i, 2), sum(N_i) = N
        - fmaps_feed:     (1, S, C, H, W)
        - pred_track:     (1, S, N, 2)
        - pred_vis:       (1, S, N)
        - pred_score:     (1, S, N)

    Args:
        track_predictor: TrackerPredictor 实例
        images_feed: 图像张量 (1, S, 3, H, W)
        query_points_list: query 点分块列表, 每个元素 (1, N_i, 2)
        fmaps_feed: 特征图 (1, S, C, H, W)
        fine_tracking: 是否精细追踪
        num_splits: (兼容旧版) 分割数
        fine_chunk: 精细追踪的 chunk 大小

    Returns:
        tuple: (pred_track, pred_vis, pred_score)
    """
    # If query_points_list is not a list or tuple but a single tensor, handle it like the old version for backward compatibility
    if not isinstance(query_points_list, (list, tuple)):
        query_points = query_points_list
        if num_splits is None:
            num_splits = 1
        query_points_list = torch.chunk(query_points, num_splits, dim=1)

    # Ensure query_points_list is a list for iteration (as torch.chunk returns a tuple)
    if isinstance(query_points_list, tuple):
        query_points_list = list(query_points_list)

    fine_pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    for split_points in query_points_list:
        # Feed into track predictor for each split
        fine_pred_track, _, pred_vis, pred_score = track_predictor(
            images_feed, split_points, fmaps=fmaps_feed, fine_tracking=fine_tracking, fine_chunk=fine_chunk
        )
        fine_pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)

    # Concatenate the results from all splits
    fine_pred_track = torch.cat(fine_pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)

    if pred_score is not None:
        pred_score = torch.cat(pred_score_list, dim=2)
    else:
        pred_score = None

    return fine_pred_track, pred_vis, pred_score
