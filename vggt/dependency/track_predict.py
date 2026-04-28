# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from .vggsfm_utils import *


def predict_tracks(
    images,
    conf=None,
    points_3d=None,
    masks=None,
    max_query_pts=2048,
    query_frame_num=5,
    keypoint_extractor="aliked+sp",
    max_points_num=163840,
    fine_tracking=True,
    complete_non_vis=True,
):
    """
    【多帧追踪预测主入口 - 重点阅读】
    基于 VGGSfM tracker 对输入图像进行跨帧特征点追踪。

    核心流程:
        1. 用 DINO 特征选择最关键的 query_frame_num 个参考帧
        2. 在每帧参考帧上提取关键点 (ALIKED + SuperPoint)
        3. 用 tracker 预测这些关键点在所有帧中的轨迹
        4. (可选) 对低可见度帧补充追踪, 保证覆盖率
        5. 合并所有 query 帧的追踪结果返回

    输出格式:
        - pred_tracks:     (N, P_total, 2)    每帧每个追踪点的像素坐标
        - pred_vis_scores: (N, P_total)       每帧每个点的可见性分数
        - pred_confs:      (P_total,)         每个追踪点的置信度
        - pred_points_3d:  (P_total, 3)       每个追踪点对应的 3D 坐标
        - pred_colors:     (P_total, 3)       每个追踪点的颜色 (uint8, 0-255)

    Args:
        images (Tensor): 输入图像 [S, 3, H, W], S 为帧数
        conf (Tensor, optional): 深度/点云置信度 [S, 1, H, W], 用于筛选高质量追踪点
        points_3d (Tensor, optional): 3D 点云 [S, H, W, 3], 用于给追踪点赋予 3D 坐标
        masks (Tensor, optional): 掩码 [S, 1, H, W], 暂未支持
        max_query_pts (int): 每帧最多提取多少个关键点 (默认 2048)
        query_frame_num (int): 选多少帧作为 query 帧 (默认 5)
        keypoint_extractor (str): 特征点提取器, 默认 "aliked+sp"
        max_points_num (int): GPU 一次最多处理的点数 (默认 163840)
        fine_tracking (bool): 是否启用精细追踪 (更准但更慢)
        complete_non_vis (bool): 是否对低可见度帧补充追踪
    """

    # 获取设备信息
    device = images.device
    dtype = images.dtype

    # 【Step 1: 初始化 Tracker】构建 VGGSfM tracker 并移到 GPU
    tracker = build_vggsfm_tracker().to(device, dtype)

    # 【Step 2: 选择 Query 帧】
    # 用 DINO 特征计算帧间相似度, 选出与第一帧最相关的 query_frame_num 帧
    # 原理: 第一帧作为参考, 找与之特征最相似的其他帧, 保证 query 点能被尽量多的帧看到
    query_frame_indexes = generate_rank_by_dino(images, query_frame_num=query_frame_num, device=device)

    # 确保第一帧 (index=0) 始终在 query 帧列表最前面
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    # 【Step 3: 初始化关键点提取器】
    # 使用 ALIKED + SuperPoint 组合提取关键点
    # TODO: masks 功能尚未实现
    keypoint_extractors = initialize_feature_extractors(
        max_query_pts, extractor_method=keypoint_extractor, device=device
    )

    # 初始化结果容器
    pred_tracks = []       # 每个 query 帧的追踪轨迹列表
    pred_vis_scores = []   # 每个 query 帧的可见性分数列表
    pred_confs = []        # 每个 query 帧的置信度列表
    pred_points_3d = []    # 每个 query 帧的 3D 点列表
    pred_colors = []       # 每个 query 帧的颜色列表

    # 【Step 4: 提取 Tracker 特征图】
    # tracker 先对全部图像提取多尺度特征图, 后续追踪复用
    fmaps_for_tracker = tracker.process_images_to_fmaps(images)

    if fine_tracking:
        print("For faster inference, consider disabling fine_tracking")

    # 【Step 5: 逐 query 帧追踪】
    # 每个 query 帧独立提取关键点, 然后追踪到所有帧
    for query_index in query_frame_indexes:
        print(f"Predicting tracks for query frame {query_index}")
        pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(
            query_index,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            device,
        )

        # 收集该 query 帧的追踪结果
        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        pred_confs.append(pred_conf)
        pred_points_3d.append(pred_point_3d)
        pred_colors.append(pred_color)

    # 【Step 6: (可选) 补充低可见度帧】
    # 检查哪些帧的可见追踪点过少 (< min_vis), 对这些帧重新作为 query 进行追踪
    if complete_non_vis:
        pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            min_vis=500,          # 每帧最少需要 500 个可见追踪点
            non_vis_thresh=0.1,   # vis_score > 0.1 算可见
            device=device,
        )

    # 【Step 7: 合并所有 query 帧的结果】
    # axis=1 合并: 每个 query 帧提取不同的关键点, 沿点维度拼接
    pred_tracks = np.concatenate(pred_tracks, axis=1)
    pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
    pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
    pred_points_3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
    pred_colors = np.concatenate(pred_colors, axis=0) if pred_colors else None

    # 返回结果
    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors


def _forward_on_query(
    query_index,
    images,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num,
    fine_tracking,
    device,
):
    """
    【单 query 帧追踪处理 - 重点阅读】
    对单个参考帧提取关键点, 并追踪到所有帧。

    核心流程:
        1. 在 query 帧上提取关键点 (ALIKED + SuperPoint)
        2. 根据 conf/points_3d 筛选高质量点 (conf > 1.2)
        3. 提取关键点处的颜色作为点云颜色
        4. 重排帧顺序: 将 query 帧移到最前面 (tracker 要求)
        5. 分块追踪 (防止 GPU OOM)
        6. 恢复原始帧顺序

    输出:
        - pred_track:  (N, P, 2)   追踪轨迹, 每帧每个点的像素坐标
        - pred_vis:    (N, P)      可见性分数
        - pred_conf:   (P,)        置信度 (从输入 conf 采样)
        - pred_point_3d: (P, 3)    3D 坐标 (从输入 points_3d 采样)
        - pred_color:  (P, 3)      颜色 (uint8, 0-255)

    Args:
        query_index: query 帧索引
        images: 输入图像 [S, 3, H, W]
        conf: 置信度 [S, 1, H, W]
        points_3d: 3D 点云 [S, H, W, 3]
        fmaps_for_tracker: tracker 预提取的特征图
        keypoint_extractors: 关键点提取器
        tracker: VGGSfM tracker 实例
        max_points_num: GPU 单次最大处理点数
        fine_tracking: 是否精细追踪
        device: 计算设备
    """
    frame_num, _, height, width = images.shape

    # 【Step 1: 提取关键点】在 query 帧上用 ALIKED + SuperPoint 提取关键点
    query_image = images[query_index]
    query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
    # 随机打乱关键点顺序, 避免局部聚集导致追踪偏差
    query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

    # 【Step 2: 提取关键点颜色】
    # query_points 是 float 坐标, 先 round 到整数像素位置再取颜色
    query_points_long = query_points.squeeze(0).round().long()
    pred_color = images[query_index][:, query_points_long[:, 1], query_points_long[:, 0]]
    # 从 CHW 转为 (P, 3), 并缩放到 [0, 255] uint8
    pred_color = (pred_color.permute(1, 0).cpu().numpy() * 255).astype(np.uint8)

    # 【Step 3: (可选) 根据 conf 和 points_3d 筛选高质量点】
    # 如果有 depth conf 和 3D 点云, 只在高质量区域提取追踪点
    if (conf is not None) and (points_3d is not None):
        # 目前只支持正方形图像
        assert height == width
        assert conf.shape[-2] == conf.shape[-1]
        assert conf.shape[:3] == points_3d.shape[:3]

        # 计算 conf/points_3d 分辨率与图像分辨率的缩放比例
        scale = conf.shape[-1] / width
        # 将 query 点坐标缩放到 conf 分辨率
        query_points_scaled = (query_points.squeeze(0) * scale).round().long()
        query_points_scaled = query_points_scaled.cpu().numpy()

        # 在 conf 和 points_3d 上采样对应位置的值
        pred_conf = conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
        pred_point_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]

        # 【启发式过滤】只保留 conf > 1.2 的高质量点 (expp1 激活后 conf > 1.2 表示比较确定)
        valid_mask = pred_conf > 1.2
        if valid_mask.sum() > 512:  # 至少保留 512 个点
            query_points = query_points[:, valid_mask]
            pred_conf = pred_conf[valid_mask]
            pred_point_3d = pred_point_3d[valid_mask]
            pred_color = pred_color[valid_mask]
    else:
        pred_conf = None
        pred_point_3d = None

    # 【Step 4: 重排帧顺序】
    # tracker 内部假设第 0 帧是 query 帧, 需要把当前 query_index 移到第 0 位
    reorder_index = calculate_index_mappings(query_index, frame_num, device=device)
    images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
    # 添加 batch 维度: [1, S, 3, H, W] 和 [1, S, C, H, W]
    images_feed = images_feed[None]
    fmaps_feed = fmaps_feed[None]

    # 【Step 5: 分块处理防止 OOM】
    # all_points_num = 帧数 × 每帧关键点数, 如果太大就拆成多个 chunk
    all_points_num = images_feed.shape[1] * query_points.shape[1]
    if all_points_num > max_points_num:
        num_splits = (all_points_num + max_points_num - 1) // max_points_num
        query_points = torch.chunk(query_points, num_splits, dim=1)
    else:
        query_points = [query_points]

    # 调用 tracker 进行分块追踪
    pred_track, pred_vis, _ = predict_tracks_in_chunks(
        tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking
    )

    # 【Step 6: 恢复原始帧顺序】
    # 把 reorder_index 打乱顺序恢复回去, 使输出按原始帧顺序排列
    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)

    # 转为 numpy 并移除 batch 维度
    pred_track = pred_track.squeeze(0).float().cpu().numpy()  # (N, P, 2)
    pred_vis = pred_vis.squeeze(0).float().cpu().numpy()      # (N, P)

    return pred_track, pred_vis, pred_conf, pred_point_3d, pred_color


def _augment_non_visible_frames(
    pred_tracks: list,        # ← 运行中的追踪轨迹列表, 每个元素是 np.ndarray (N, P_i, 2)
    pred_vis_scores: list,    # ← 运行中的可见性分数列表, 每个元素是 np.ndarray (N, P_i)
    pred_confs: list,         # ← 运行中的置信度列表, 每个元素是 np.ndarray (P_i,)
    pred_points_3d: list,     # ← 运行中的 3D 点列表, 每个元素是 np.ndarray (P_i, 3)
    pred_colors: list,        # ← 运行中的颜色列表, 每个元素是 np.ndarray (P_i, 3)
    images: torch.Tensor,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num: int,
    fine_tracking: bool,
    *,
    min_vis: int = 500,           # 每帧最少可见追踪点数阈值
    non_vis_thresh: float = 0.1,  # vis_score > 0.1 算可见
    device: torch.device = None,
):
    """
    【补充低可见度帧的追踪 - 重点阅读】

    问题: 初始 query 帧选择 (如第 0, 3, 5 帧) 可能无法覆盖所有帧。
          某些帧可能只看到很少的追踪点 (如遮挡严重、视角差异大)。

    解决策略:
        1. 检查每帧的可见追踪点数量
        2. 对可见点 < min_vis 的帧, 将其作为新的 query 帧重新追踪
        3. 如果某帧一次补充后仍不够, 换更强的提取器 (sp+sift+aliked) 做最终尝试
        4. 循环直到所有帧都满足或放弃

    为什么有效:
        - 帧 A 作为 query 时提取的关键点, 在帧 B 上可能不可见
        - 但帧 B 自己作为 query 时, 会提取适合从自身视角追踪的特征点
        - 这样每个"难帧"都有机会用自己最稳定的特征点建立追踪

    Args:
        pred_tracks: 当前已收集的追踪轨迹列表
        pred_vis_scores: 当前已收集的可见性分数列表
        pred_confs: 当前已收集的置信度列表
        pred_points_3d: 当前已收集的 3D 点列表
        pred_colors: 当前已收集的颜色列表
        images: 输入图像 [S, 3, H, W]
        conf: 置信度 [S, 1, H, W]
        points_3d: 3D 点云 [S, H, W, 3]
        fmaps_for_tracker: tracker 预提取的特征图
        keypoint_extractors: 关键点提取器
        tracker: VGGSfM tracker
        max_points_num: GPU 单次最大处理点数
        fine_tracking: 是否精细追踪
        min_vis: 每帧最少可见追踪点数 (默认 500)
        non_vis_thresh: 可见性阈值 (默认 0.1)
        device: 计算设备

    Returns:
        更新后的 pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors
    """
    last_query = -1      # 记录上一轮回补充的帧, 检测是否同一帧重复失败
    final_trial = False  # 是否为最终尝试 (换更强提取器)
    cur_extractors = keypoint_extractors  # 当前使用的提取器, 最终轮会替换

    while True:
        # 【Step 1: 统计每帧可见追踪点数量】
        # 沿 axis=1 拼接所有 query 帧的结果, 得到完整可见性矩阵 (N, P_total)
        vis_array = np.concatenate(pred_vis_scores, axis=1)

        # 每帧统计 vis_score > non_vis_thresh 的追踪点数量
        sufficient_vis_count = (vis_array > non_vis_thresh).sum(axis=-1)
        # 找出可见点不足的帧索引
        non_vis_frames = np.where(sufficient_vis_count < min_vis)[0].tolist()

        # 所有帧都满足, 结束循环
        if len(non_vis_frames) == 0:
            break

        print("Processing non visible frames:", non_vis_frames)

        # 【Step 2: 决策策略】决定本轮怎么补充
        if non_vis_frames[0] == last_query:
            # 【同一帧连续两次都失败】说明当前提取器无法在该帧找到足够稳定特征
            # 换更强的提取器组合 (sp+sift+aliked), 并且一次性处理所有难帧
            final_trial = True
            cur_extractors = initialize_feature_extractors(2048, extractor_method="sp+sift+aliked", device=device)
            query_frame_list = non_vis_frames  # 一次性处理所有剩余难帧
        else:
            # 【正常情况】每次只处理第一个难帧
            query_frame_list = [non_vis_frames[0]]

        last_query = non_vis_frames[0]

        # 【Step 3: 对选中的帧进行补充追踪】
        for query_index in query_frame_list:
            new_track, new_vis, new_conf, new_point_3d, new_color = _forward_on_query(
                query_index,
                images,
                conf,
                points_3d,
                fmaps_for_tracker,
                cur_extractors,
                tracker,
                max_points_num,
                fine_tracking,
                device,
            )
            # 将新追踪结果追加到列表中
            pred_tracks.append(new_track)
            pred_vis_scores.append(new_vis)
            pred_confs.append(new_conf)
            pred_points_3d.append(new_point_3d)
            pred_colors.append(new_color)

        # 如果是最终尝试, 无论结果如何都结束 (避免无限循环)
        if final_trial:
            break

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors
