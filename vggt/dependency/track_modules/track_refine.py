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

from PIL import Image
import os
from typing import Union, Tuple


def refine_track(
    images, fine_fnet, fine_tracker, coarse_pred, compute_score=False, pradius=15, sradius=2, fine_iters=6, chunk=40960
):
    """
    【Fine Tracking 精修追踪 - 重点阅读】

    算法原理 (基于 VGGSfM, arxiv:2312.04563):
        Coarse 追踪给出了大致位置 (误差约几个像素),
        但精度不够。Fine Tracking 在 coarse 结果周围提取小 patch,
        在 patch 内重新做精细追踪, 达到亚像素精度。

    核心流程:
        1. 在 coarse 追踪结果位置提取 patch (默认 31x31)
        2. 用 fine_fnet (浅层 CNN) 提取 patch 内的高分辨率特征
        3. 将 query 点坐标转换到 patch 局部坐标系
        4. 用 fine_tracker (BaseTrackerPredictor, stride=1) 在 patch 内做精细追踪
        5. 将精细追踪结果从 patch 坐标系转回图像坐标系

    关键技巧: 整数坐标 + 小数偏移 (整数用于索引 patch, 小数用于亚像素定位)
        例: coarse_pred = (128.16, 252.78)
            - 整数部分 track_int  = (128, 252)  → 用于索引 patch 的左上角
            - 小数部分 track_frac = (0.16, 0.78) → 用于 patch 内的亚像素 query 点
            - patch 左上角 topleft = (128, 252) - pradius = (113, 237)

    Args:
        images (Tensor): 输入图像 [B, S, 3, H, W]
        fine_fnet (nn.Module): Fine 特征提取网络 (ShallowEncoder, stride=1)
        fine_tracker (nn.Module): Fine 追踪器 (BaseTrackerPredictor, stride=1)
        coarse_pred (Tensor): Coarse 追踪结果 [B, S, N, 2]
        compute_score (bool): 是否计算追踪分数 (默认 False, 当前未使用)
        pradius (int): Patch 半径 (默认 15, patch 大小 = 31x31)
        sradius (int): Score 计算时的搜索半径 (默认 2)
        fine_iters (int): Fine 阶段迭代次数 (默认 6)
        chunk (int): 分块大小, -1 表示不分块 (默认 40960)

    Returns:
        tuple: (refined_tracks, score)
            - refined_tracks (Tensor): 精修后的追踪坐标 [B, S, N, 2]
            - score (Tensor or None): 追踪分数 [B, S, N]
    """

    # coarse_pred: [B, S, N, 2]
    # B: batch, S: 帧数, N: 追踪点数, 2: (x, y) 坐标
    B, S, N, _ = coarse_pred.shape
    _, _, _, H, W = images.shape

    # 计算 patch 大小: pradius=15 → psize=31
    psize = pradius * 2 + 1

    # 假设第 0 帧是 query 帧, 提取第 0 帧的坐标作为 query_points
    query_points = coarse_pred[:, 0]

    # 【核心技巧: 用 unfold 高效提取 patches, 不用 grid_sample】
    # 不用 grid_sample 的原因: grid_sample 需要为每个 patch 单独生成 grid, 内存开销大
    # 改用 unfold: 在整个图像上做滑动窗口, 然后按索引取 patch, 更省内存
    #
    # 例: query_point = (128.16, 252.78), patch=31x31
    #   理想 patch 范围 (浮点): left_top=(113.16, 237.78), right_bottom=(143.16, 267.78)
    #   实际做法:
    #     1. 取整: left_top_int = floor(coarse_pred) - pradius = (113, 237)
    #     2. 偏移: offset = coarse_pred - floor(coarse_pred) = (0.16, 0.78)
    #     3. 用 unfold 在整个图上滑动 31x31 窗口, 然后索引取 (113, 237) 位置的 patch
    #     4. patch 内的 query 点坐标 = offset + pradius = (15.16, 15.78)

    with torch.no_grad():
        # 展平为 [B*S, 3, H, W]
        content_to_extract = images.reshape(B * S, 3, H, W)
        C_in = content_to_extract.shape[1]

        # 【Unfold 提取滑动窗口 patches】
        # unfold(dim=2, size=31, step=1): 沿 H 维度滑动 31x31 窗口, 步长 1
        # unfold(dim=3, size=31, step=1): 沿 W 维度滑动 31x31 窗口, 步长 1
        # 输出形状: [B*S, 3, H_new, W_new, 31, 31]
        # H_new = H - 31 + 1, W_new = W - 31 + 1
        # 相当于在整个图像上每个位置都预提取了一个 31x31 patch
        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1)

    # 将 coarse 坐标拆分为整数部分和小数部分
    track_int = coarse_pred.floor().int()   # 整数部分, 用于索引 patch 左上角
    track_frac = coarse_pred - track_int    # 小数部分, 用于 patch 内亚像素定位

    # 计算 patch 左上角坐标 (以整数坐标为中心, 半径 pradius)
    topleft = track_int - pradius
    # 保存原始 topleft (未 clamp), 后续坐标恢复时需要
    topleft_BSN = topleft.clone()

    # clamp 边界, 防止越界
    # ⚠️ 重要假设: H == W (正方形图像)
    topleft = topleft.clamp(0, H - psize)

    # 展平为 [B*S, N, 2]
    topleft = topleft.reshape(B * S, N, 2)

    # 构建 batch 索引: [B*S, N], 每行都是该 batch 的索引
    batch_indices = torch.arange(B * S)[:, None].expand(-1, N).to(content_to_extract.device)

    # 【按索引取 patch】
    # content_to_extract[batch_indices, :, y, x]: 对每个 (batch, N) 取对应 (y, x) 位置的 patch
    # 输出: [B*S, N, 3, 31, 31]
    extracted_patches = content_to_extract[batch_indices, :, topleft[..., 1], topleft[..., 0]]

    # 【Step 1: 提取 Fine 特征】
    # 将 patches 展平为 [B*S*N, 3, 31, 31] 通过 fine_fnet
    if chunk < 0:
        # 不分块 (点少时)
        patch_feat = fine_fnet(extracted_patches.reshape(B * S * N, C_in, psize, psize))
    else:
        # 分块 (点多时防止 OOM)
        patches = extracted_patches.reshape(B * S * N, C_in, psize, psize)
        patch_feat_list = []
        for p in torch.split(patches, chunk):
            patch_feat_list += [fine_fnet(p)]
        patch_feat = torch.cat(patch_feat_list, 0)

    C_out = patch_feat.shape[1]

    # 【Step 2: 将特征重组为 Fine Tracker 输入格式】
    # patch_feat: [B, S, N, C_out, 31, 31]
    patch_feat = patch_feat.reshape(B, S, N, C_out, psize, psize)
    # rearrange: [B*S, N, C_out, 31, 31] → [B*N, S, C_out, 31, 31]
    # 这样 fine_tracker 看到的是: batch=B*N, 帧数=S, 特征图大小=31x31
    patch_feat = rearrange(patch_feat, "b s n c p q -> (b n) s c p q")

    # 【Step 3: 准备 patch 内的 query 点坐标】
    # 将 query 点从图像坐标系转到 patch 坐标系
    # patch 坐标系: 左上角 = (0, 0), 中心 = (pradius, pradius)
    # query_point_in_patch = offset + pradius
    # 例: offset=(0.16, 0.78), pradius=15 → query=(15.16, 15.78)
    patch_query_points = track_frac[:, 0] + pradius
    patch_query_points = patch_query_points.reshape(B * N, 2).unsqueeze(1)

    # 【Step 4: Fine Tracker 精修追踪】
    # 在 patch 内做 correlation-based 追踪, 输出精细坐标 (亚像素级)
    fine_pred_track_lists, _, _, query_point_feat = fine_tracker(
        query_points=patch_query_points, fmaps=patch_feat, iters=fine_iters, return_feat=True
    )

    # fine_pred_track_lists[-1]: 最后一轮迭代的输出 [B*N, S, 1, 2]
    # 当前坐标是相对于 patch 左上角的
    fine_pred_track = fine_pred_track_lists[-1].clone()

    # 【Step 5: 将 patch 坐标转回图像坐标系】
    # 对每一轮迭代的输出都做坐标转换
    for idx in range(len(fine_pred_track_lists)):
        # rearrange: [B*N, S, 1, 2] → [B, S, N, 1, 2]
        fine_level = rearrange(fine_pred_track_lists[idx], "(b n) s u v -> b s n u v", b=B, n=N)
        # 去掉多余的维度: [B, S, N, 2]
        fine_level = fine_level.squeeze(-2)
        # 加上 patch 左上角坐标, 转回图像坐标系
        fine_level = fine_level + topleft_BSN
        fine_pred_track_lists[idx] = fine_level

    # 【Step 6: 整理最终输出】
    # refined_tracks: 最后一轮精修结果, 图像坐标系
    refined_tracks = fine_pred_track_lists[-1].clone()
    # 第 0 帧固定为 query 点 (不应改变)
    refined_tracks[:, 0] = query_points

    score = None

    if compute_score:
        score = compute_score_fn(query_point_feat, patch_feat, fine_pred_track, sradius, psize, B, N, S, C_out)

    return refined_tracks, score


def refine_track_v0(
    images, fine_fnet, fine_tracker, coarse_pred, compute_score=False, pradius=15, sradius=2, fine_iters=6
):
    """
    COPIED FROM VGGSfM

    Refines the tracking of images using a fine track predictor and a fine feature network.
    Check https://arxiv.org/abs/2312.04563 for more details.

    Args:
        images (torch.Tensor): The images to be tracked.
        fine_fnet (nn.Module): The fine feature network.
        fine_tracker (nn.Module): The fine track predictor.
        coarse_pred (torch.Tensor): The coarse predictions of tracks.
        compute_score (bool, optional): Whether to compute the score. Defaults to False.
        pradius (int, optional): The radius of a patch. Defaults to 15.
        sradius (int, optional): The search radius. Defaults to 2.

    Returns:
        torch.Tensor: The refined tracks.
        torch.Tensor, optional: The score.
    """

    # coarse_pred shape: BxSxNx2,
    # where B is the batch, S is the video/images length, and N is the number of tracks
    # now we are going to extract patches with the center at coarse_pred
    # Please note that the last dimension indicates x and y, and hence has a dim number of 2
    B, S, N, _ = coarse_pred.shape
    _, _, _, H, W = images.shape

    # Given the raidus of a patch, compute the patch size
    psize = pradius * 2 + 1

    # Note that we assume the first frame is the query frame
    # so the 2D locations of the first frame are the query points
    query_points = coarse_pred[:, 0]

    # Given 2D positions, we can use grid_sample to extract patches
    # but it takes too much memory.
    # Instead, we use the floored track xy to sample patches.

    # For example, if the query point xy is (128.16, 252.78),
    # and the patch size is (31, 31),
    # our goal is to extract the content of a rectangle
    # with left top: (113.16, 237.78)
    # and right bottom: (143.16, 267.78).
    # However, we record the floored left top: (113, 237)
    # and the offset (0.16, 0.78)
    # Then what we need is just unfolding the images like in CNN,
    # picking the content at [(113, 237), (143, 267)].
    # Such operations are highly optimized at pytorch
    # (well if you really want to use interpolation, check the function extract_glimpse() below)

    with torch.no_grad():
        content_to_extract = images.reshape(B * S, 3, H, W)
        C_in = content_to_extract.shape[1]

        # Please refer to https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # for the detailed explanation of unfold()
        # Here it runs sliding windows (psize x psize) to build patches
        # The shape changes from
        # (B*S)x C_in x H x W to (B*S)x C_in x H_new x W_new x Psize x Psize
        # where Psize is the size of patch
        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1)

    # Floor the coarse predictions to get integers and save the fractional/decimal
    track_int = coarse_pred.floor().int()
    track_frac = coarse_pred - track_int

    # Note the points represent the center of patches
    # now we get the location of the top left corner of patches
    # because the ouput of pytorch unfold are indexed by top left corner
    topleft = track_int - pradius
    topleft_BSN = topleft.clone()

    # clamp the values so that we will not go out of indexes
    # NOTE: (VERY IMPORTANT: This operation ASSUMES H=W).
    # You need to seperately clamp x and y if H!=W
    topleft = topleft.clamp(0, H - psize)

    # Reshape from BxSxNx2 -> (B*S)xNx2
    topleft = topleft.reshape(B * S, N, 2)

    # Prepare batches for indexing, shape: (B*S)xN
    batch_indices = torch.arange(B * S)[:, None].expand(-1, N).to(content_to_extract.device)

    # Extract image patches based on top left corners
    # extracted_patches: (B*S) x N x C_in x Psize x Psize
    extracted_patches = content_to_extract[batch_indices, :, topleft[..., 1], topleft[..., 0]]

    # Feed patches to fine fent for features
    patch_feat = fine_fnet(extracted_patches.reshape(B * S * N, C_in, psize, psize))

    C_out = patch_feat.shape[1]

    # Refine the coarse tracks by fine_tracker

    # reshape back to B x S x N x C_out x Psize x Psize
    patch_feat = patch_feat.reshape(B, S, N, C_out, psize, psize)
    patch_feat = rearrange(patch_feat, "b s n c p q -> (b n) s c p q")

    # Prepare for the query points for fine tracker
    # They are relative to the patch left top corner,
    # instead of the image top left corner now
    # patch_query_points: N x 1 x 2
    # only 1 here because for each patch we only have 1 query point
    patch_query_points = track_frac[:, 0] + pradius
    patch_query_points = patch_query_points.reshape(B * N, 2).unsqueeze(1)

    # Feed the PATCH query points and tracks into fine tracker
    fine_pred_track_lists, _, _, query_point_feat = fine_tracker(
        query_points=patch_query_points, fmaps=patch_feat, iters=fine_iters, return_feat=True
    )

    # relative the patch top left
    fine_pred_track = fine_pred_track_lists[-1].clone()

    # From (relative to the patch top left) to (relative to the image top left)
    for idx in range(len(fine_pred_track_lists)):
        fine_level = rearrange(fine_pred_track_lists[idx], "(b n) s u v -> b s n u v", b=B, n=N)
        fine_level = fine_level.squeeze(-2)
        fine_level = fine_level + topleft_BSN
        fine_pred_track_lists[idx] = fine_level

    # relative to the image top left
    refined_tracks = fine_pred_track_lists[-1].clone()
    refined_tracks[:, 0] = query_points

    score = None

    if compute_score:
        score = compute_score_fn(query_point_feat, patch_feat, fine_pred_track, sradius, psize, B, N, S, C_out)

    return refined_tracks, score


################################## NOTE: NOT USED ##################################


def compute_score_fn(query_point_feat, patch_feat, fine_pred_track, sradius, psize, B, N, S, C_out):
    """
    Compute the scores, i.e., the standard deviation of the 2D similarity heatmaps,
    given the query point features and reference frame feature maps
    """

    from kornia.utils.grid import create_meshgrid
    from kornia.geometry.subpix import dsnt

    # query_point_feat initial shape: B x N x C_out,
    # query_point_feat indicates the feat at the coorponsing query points
    # Therefore we don't have S dimension here
    query_point_feat = query_point_feat.reshape(B, N, C_out)
    # reshape and expand to B x (S-1) x N x C_out
    query_point_feat = query_point_feat.unsqueeze(1).expand(-1, S - 1, -1, -1)
    # and reshape to (B*(S-1)*N) x C_out
    query_point_feat = query_point_feat.reshape(B * (S - 1) * N, C_out)

    # Radius and size for computing the score
    ssize = sradius * 2 + 1

    # Reshape, you know it, so many reshaping operations
    patch_feat = rearrange(patch_feat, "(b n) s c p q -> b s n c p q", b=B, n=N)

    # Again, we unfold the patches to smaller patches
    # so that we can then focus on smaller patches
    # patch_feat_unfold shape:
    # B x S x N x C_out x (psize - 2*sradius) x (psize - 2*sradius) x ssize x ssize
    # well a bit scary, but actually not
    patch_feat_unfold = patch_feat.unfold(4, ssize, 1).unfold(5, ssize, 1)

    # Do the same stuffs above, i.e., the same as extracting patches
    fine_prediction_floor = fine_pred_track.floor().int()
    fine_level_floor_topleft = fine_prediction_floor - sradius

    # Clamp to ensure the smaller patch is valid
    fine_level_floor_topleft = fine_level_floor_topleft.clamp(0, psize - ssize)
    fine_level_floor_topleft = fine_level_floor_topleft.squeeze(2)

    # Prepare the batch indices and xy locations

    batch_indices_score = torch.arange(B)[:, None, None].expand(-1, S, N)  # BxSxN
    batch_indices_score = batch_indices_score.reshape(-1).to(patch_feat_unfold.device)  # B*S*N
    y_indices = fine_level_floor_topleft[..., 0].flatten()  # Flatten H indices
    x_indices = fine_level_floor_topleft[..., 1].flatten()  # Flatten W indices

    reference_frame_feat = patch_feat_unfold.reshape(
        B * S * N, C_out, psize - sradius * 2, psize - sradius * 2, ssize, ssize
    )

    # Note again, according to pytorch convention
    # x_indices cooresponds to [..., 1] and y_indices cooresponds to [..., 0]
    reference_frame_feat = reference_frame_feat[batch_indices_score, :, x_indices, y_indices]
    reference_frame_feat = reference_frame_feat.reshape(B, S, N, C_out, ssize, ssize)
    # pick the frames other than the first one, so we have S-1 frames here
    reference_frame_feat = reference_frame_feat[:, 1:].reshape(B * (S - 1) * N, C_out, ssize * ssize)

    # Compute similarity
    sim_matrix = torch.einsum("mc,mcr->mr", query_point_feat, reference_frame_feat)
    softmax_temp = 1.0 / C_out**0.5
    heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
    # 2D heatmaps
    heatmap = heatmap.reshape(B * (S - 1) * N, ssize, ssize)  # * x ssize x ssize

    coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
    grid_normalized = create_meshgrid(ssize, ssize, normalized_coordinates=True, device=heatmap.device).reshape(
        1, -1, 2
    )

    var = torch.sum(grid_normalized**2 * heatmap.view(-1, ssize * ssize, 1), dim=1) - coords_normalized**2
    std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # clamp needed for numerical stability

    score = std.reshape(B, S - 1, N)
    # set score as 1 for the query frame
    score = torch.cat([torch.ones_like(score[:, 0:1]), score], dim=1)

    return score


def extract_glimpse(
    tensor: torch.Tensor, size: Tuple[int, int], offsets, mode="bilinear", padding_mode="zeros", debug=False, orib=None
):
    B, C, W, H = tensor.shape

    h, w = size
    xs = torch.arange(0, w, dtype=tensor.dtype, device=tensor.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=tensor.dtype, device=tensor.device) - (h - 1) / 2.0

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2
    grid = grid[None]

    B, N, _ = offsets.shape

    offsets = offsets.reshape((B * N), 1, 1, 2)
    offsets_grid = offsets + grid

    # normalised grid  to [-1, 1]
    offsets_grid = (offsets_grid - offsets_grid.new_tensor([W / 2, H / 2])) / offsets_grid.new_tensor([W / 2, H / 2])

    # BxCxHxW -> Bx1xCxHxW
    tensor = tensor[:, None]

    # Bx1xCxHxW -> BxNxCxHxW
    tensor = tensor.expand(-1, N, -1, -1, -1)

    # BxNxCxHxW -> (B*N)xCxHxW
    tensor = tensor.reshape((B * N), C, W, H)

    sampled = torch.nn.functional.grid_sample(
        tensor, offsets_grid, mode=mode, align_corners=False, padding_mode=padding_mode
    )

    # NOTE: I am not sure it should be h, w or w, h here
    # but okay for sqaures
    sampled = sampled.reshape(B, N, C, h, w)

    return sampled
