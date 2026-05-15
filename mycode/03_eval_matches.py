#!/usr/bin/env python3
"""
观测脚本：特征匹配 + VGGT 重投影误差可视化

功能:
    1. 在 cluster 内用 SuperPoint + LightGlue 提取特征并匹配
    2. 基于 VGGT 预测的 depth + camera pose 计算重投影误差
    3. 用重投影误差判断 match 正确性（误差 < threshold 为正确）
    4. 在图片上可视化匹配对（颜色表示误差大小）
    5. 统计每个 cluster 的正确 match 比例

正确 match 定义:
    匹配点通过 VGGT depth 反投影为 3D，再重投影到另一帧，
    重投影像素误差 < threshold 视为正确。

用法（单个 cluster）:
    /home/23036584r/anaconda3/envs/vggt/bin/python mycode/03_eval_matches.py \
        --cluster_dir output_vggt_cluster/cluster_001 \
        --reproj_threshold 4.0

用法（所有 cluster）:
    /home/23036584r/anaconda3/envs/vggt/bin/python mycode/03_eval_matches.py \
        --cluster_dir output_vggt_cluster \
        --process_all \
        --reproj_threshold 4.0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
from lightglue import SuperPoint, LightGlue
from lightglue.utils import rbd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
COLOR_CORRECT = (0, 255, 0)      # 绿色: 误差 < threshold/2
COLOR_BORDER = (0, 255, 255)     # 黄色: threshold/2 <= 误差 < threshold
COLOR_WRONG = (0, 0, 255)        # 红色: 误差 >= threshold
COLOR_INVALID = (128, 128, 128)  # 灰色: 深度无效或相机后方


def load_predictions(cluster_dir):
    """加载 VGGT 预测结果 (predictions.npz)。"""
    pred_path = os.path.join(cluster_dir, "predictions.npz")
    data = np.load(pred_path)
    return {
        "depth": data["depth"],              # (S, 1, H, W)
        "extrinsic": data["extrinsic"],      # (S, 3, 4) OpenCV camera-from-world
        "intrinsic": data["intrinsic"],      # (S, 3, 3)
        "images": data["images"],            # (S, H, W, 3) [0, 1]
    }


def init_matcher(device="cuda"):
    """初始化 SuperPoint 提取器 + LightGlue 匹配器。"""
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def extract_and_match(extractor, matcher, img_i, img_j):
    """
    对两帧图片提取 SuperPoint 特征并用 LightGlue 匹配。

    Args:
        img_i, img_j: (1, 3, H, W) torch tensor, range [0, 1]

    Returns:
        mkpts_i, mkpts_j: (M, 2) numpy arrays, 匹配点像素坐标
        conf: (M,) confidence scores
    """
    with torch.no_grad():
        feats_i = extractor.extract(img_i)
        feats_j = extractor.extract(img_j)

        # LightGlue 输入需要保留 batch 维度
        matches01 = matcher({"image0": feats_i, "image1": feats_j})

        # LightGlue 输出格式: matches0[i] = j 表示 frame0 的第 i 个点匹配 frame1 的第 j 个点
        # -1 表示未匹配
        matches = matches01["matches0"][0]  # (N,)
        valid = matches > -1

        kpts_i = feats_i["keypoints"][0]   # (N, 2)
        kpts_j = feats_j["keypoints"][0]   # (M, 2)

        mkpts_i = kpts_i[valid].cpu().numpy()
        mkpts_j = kpts_j[matches[valid]].cpu().numpy()

        conf = matches01.get("matching_scores0")
        if conf is not None:
            conf = conf[0][valid].cpu().numpy()

    return mkpts_i, mkpts_j, conf


def compute_reprojection_error(
    mkpts_i, mkpts_j,
    depth_i,
    extrinsic_i, extrinsic_j,
    intrinsic_i, intrinsic_j,
):
    """
    基于 VGGT depth + pose 计算重投影误差。

    流程:
        1. 在 depth_i 上采样匹配点的深度值 Z
        2. 用 intrinsic_i 反投影为 3D 相机坐标: P_cam = Z * K^-1 * [u, v, 1]^T
        3. 用 extrinsic_i 转换到世界坐标: P_world = R_i^T * (P_cam - t_i)
        4. 用 extrinsic_j 转换到帧 j 相机坐标: P_cam_j = R_j * P_world + t_j
        5. 用 intrinsic_j 重投影: [u', v'] = K_j * P_cam_j / Z_j
        6. 计算 ||[u', v'] - [u_j, v_j]||

    Args:
        mkpts_i: (M, 2) 帧 i 匹配点坐标
        mkpts_j: (M, 2) 帧 j 匹配点坐标
        depth_i: (H, W) 帧 i 深度图
        extrinsic_i, extrinsic_j: (3, 4) 外参
        intrinsic_i, intrinsic_j: (3, 3) 内参

    Returns:
        errors: (M,) 重投影误差（像素）
        valid_mask: (M,) 是否有效
        reproj_pts: (M, 2) 重投影后的坐标
    """
    M = len(mkpts_i)
    H, W = depth_i.shape

    # 1. 采样深度
    u_i = np.clip(np.round(mkpts_i[:, 0]).astype(np.int32), 0, W - 1)
    v_i = np.clip(np.round(mkpts_i[:, 1]).astype(np.int32), 0, H - 1)
    z = depth_i[v_i, u_i]

    # 深度有效性检查
    valid_depth = (z > 0.1) & np.isfinite(z)

    # 2. 反投影到相机坐标系
    # P_cam = Z * K^-1 * [u, v, 1]^T
    K_inv_i = np.linalg.inv(intrinsic_i)
    uv1 = np.column_stack([mkpts_i, np.ones(M, dtype=np.float32)])  # (M, 3)
    pts_cam = z[:, None] * (uv1 @ K_inv_i.T)  # (M, 3)

    # 3. 转换到世界坐标系
    # P_world = R^T * (P_cam - t)
    R_i = extrinsic_i[:, :3]
    t_i = extrinsic_i[:, 3]
    pts_world = ((pts_cam - t_i) @ R_i.T)  # (M, 3)

    # 4. 转换到帧 j 相机坐标系
    # P_cam_j = R_j * P_world + t_j
    R_j = extrinsic_j[:, :3]
    t_j = extrinsic_j[:, 3]
    pts_cam_j = (pts_world @ R_j.T) + t_j  # (M, 3)

    # 相机前方检查
    valid_front = pts_cam_j[:, 2] > 0.01

    # 5. 重投影到帧 j 像素平面
    # [u', v'] = K * [X/Z, Y/Z, 1]
    z_j = pts_cam_j[:, 2]
    xy_norm = pts_cam_j[:, :2] / z_j[:, None]  # (M, 2)
    reproj_pts = (xy_norm @ intrinsic_j[:2, :2].T) + intrinsic_j[:2, 2]  # (M, 2)

    # 6. 计算误差
    errors = np.linalg.norm(reproj_pts - mkpts_j, axis=1)

    valid_mask = valid_depth & valid_front

    return errors, valid_mask, reproj_pts


def visualize_pair(
    image_i, image_j,
    mkpts_i, mkpts_j,
    errors, valid_mask, reproj_pts,
    threshold,
    output_path,
):
    """
    可视化一对图片的匹配和重投影误差。

    画布分上下两部分:
      上半: [image_i | image_j] 所有匹配连线（颜色表示正确性）
      下半: [image_i | image_j] 只展示重投影（匹配点 + 重投影点 + 误差向量）

    连线颜色:
        绿色: 误差 < threshold/2  (正确匹配)
        黄色: threshold/2 <= 误差 < threshold  (边界)
        红色: 误差 >= threshold  (错误匹配)
        灰色: 无效 (深度无效或相机后方)
    """
    Hi, Wi = image_i.shape[:2]
    Hj, Wj = image_j.shape[:2]
    H_single = max(Hi, Hj)
    W_total = Wi + Wj

    # 创建上下拼接的画布
    canvas = np.ones((H_single * 2, W_total, 3), dtype=np.uint8) * 255
    # 上半
    canvas[:Hi, :Wi] = image_i
    canvas[:Hj, Wi:] = image_j
    # 下半
    canvas[H_single:H_single + Hi, :Wi] = image_i
    canvas[H_single:H_single + Hj, Wi:] = image_j

    n_total = len(mkpts_i)
    n_valid = int(valid_mask.sum())
    n_correct = int((valid_mask & (errors < threshold)).sum())
    mean_err = errors[valid_mask].mean() if n_valid > 0 else 0.0

    # ---- 上半：匹配连线（原有逻辑）----
    for m in range(n_total):
        pt_i = tuple(mkpts_i[m].astype(int))
        pt_j = tuple((mkpts_j[m] + np.array([Wi, 0], dtype=np.float32)).astype(int))

        if not valid_mask[m]:
            color = COLOR_INVALID
            thickness = 1
        elif errors[m] < threshold / 2:
            color = COLOR_CORRECT
            thickness = 1
        elif errors[m] < threshold:
            color = COLOR_BORDER
            thickness = 1
        else:
            color = COLOR_WRONG
            thickness = 1

        cv2.line(canvas, pt_i, pt_j, color, thickness)
        cv2.circle(canvas, pt_i, 3, color, -1)
        cv2.circle(canvas, pt_j, 3, color, -1)

    # 上半统计文字
    ratio = n_correct / n_total * 100 if n_total > 0 else 0
    text1 = f"Total: {n_total} | Valid: {n_valid} | Correct: {n_correct} ({ratio:.1f}%)"
    text2 = f"MeanErr: {mean_err:.2f}px | Threshold: {threshold}px"

    cv2.rectangle(canvas, (0, 0), (W_total, 50), (255, 255, 255), -1)
    cv2.putText(canvas, text1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, text2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    legend_y = H_single - 10
    cv2.putText(canvas, "Correct", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CORRECT, 2)
    cv2.putText(canvas, "Border", (90, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BORDER, 2)
    cv2.putText(canvas, "Wrong", (170, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WRONG, 2)
    cv2.putText(canvas, "Invalid", (240, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INVALID, 2)

    # ---- 下半：重投影专门展示（只画 valid 匹配）----
    y_offset = H_single
    n_reproj_valid = 0
    n_reproj_wrong = 0

    for m in range(n_total):
        if not valid_mask[m]:
            continue

        n_reproj_valid += 1
        pt_i = tuple(mkpts_i[m].astype(int))
        pt_j = tuple((mkpts_j[m] + np.array([Wi, 0], dtype=np.float32)).astype(int))
        rp = tuple((reproj_pts[m] + np.array([Wi, 0], dtype=np.float32)).astype(int))

        if errors[m] >= threshold:
            n_reproj_wrong += 1

        # 下半部分坐标（加上 y_offset）
        pt_i_lower = (pt_i[0], pt_i[1] + y_offset)
        pt_j_lower = (pt_j[0], pt_j[1] + y_offset)
        rp_lower = (rp[0], rp[1] + y_offset)

        # image_i 侧：匹配点（绿色圆点，不连线）
        cv2.circle(canvas, pt_i_lower, 3, COLOR_CORRECT, -1)

        # image_j 侧：匹配点（绿色圆点）+ 重投影点（红色叉）+ 误差向量
        cv2.circle(canvas, pt_j_lower, 3, COLOR_CORRECT, -1)
        cv2.drawMarker(canvas, rp_lower, COLOR_WRONG, cv2.MARKER_CROSS, 6, 2)
        cv2.line(canvas, pt_j_lower, rp_lower, (255, 0, 255), 1)

    # 下半标题
    cv2.rectangle(canvas, (0, y_offset), (W_total, y_offset + 30), (255, 255, 255), -1)
    reproj_text = f"Reprojection Only (valid={n_reproj_valid}, wrong={n_reproj_wrong})"
    cv2.putText(canvas, reproj_text, (10, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 下半图例
    legend_y2 = y_offset + H_single - 10
    cv2.putText(canvas, "Match(Green)", (10, legend_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CORRECT, 2)
    cv2.putText(canvas, "Reproj(RedX)", (110, legend_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WRONG, 2)
    cv2.putText(canvas, "ErrorVec", (210, legend_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.imwrite(output_path, canvas)
    return n_total, n_valid, n_correct, mean_err


def process_cluster(cluster_dir, extractor, matcher, threshold, device, parent_output_dir=None):
    """
    处理单个 cluster：匹配 + 重投影误差 + 可视化。

    对每个 cluster，只处理 anchor (第0张) 与其余每张图片的匹配对，
    减少计算量同时保证覆盖 cluster 内所有视角。
    """
    cluster_name = os.path.basename(cluster_dir)
    print(f"\n{'='*60}")
    print(f"Processing {cluster_name}")
    print(f"{'='*60}")

    # 输出目录
    if parent_output_dir:
        output_dir = os.path.join(parent_output_dir, f"eval_{cluster_name}")
    else:
        output_dir = os.path.join(cluster_dir, "eval_matches")
    os.makedirs(output_dir, exist_ok=True)

    # 加载 VGGT 预测
    try:
        pred = load_predictions(cluster_dir)
    except Exception as e:
        print(f"  Skip: cannot load predictions: {e}")
        return None

    S = pred["images"].shape[0]
    print(f"  Images in cluster: {S}")

    # images: (S, 3, H, W) [0, 1] — 已经是 NCHW
    images_t = torch.from_numpy(pred["images"]).float().to(device)
    # depth: (S, H, W, 1) → 取 squeeze
    depth_maps = pred["depth"].squeeze(-1)  # (S, H, W)

    # 处理 anchor (第0张) 与其余每张的匹配
    anchor_idx = 0
    stats = []

    for j in range(1, S):
        print(f"  Pair {anchor_idx}-{j} ...", end=" ")

        # 提取匹配
        mkpts_i, mkpts_j, conf = extract_and_match(
            extractor, matcher,
            images_t[anchor_idx:anchor_idx+1],
            images_t[j:j+1],
        )

        if len(mkpts_i) == 0:
            print("no matches")
            continue

        # 计算重投影误差
        depth_i = depth_maps[anchor_idx]  # (H, W)
        errors, valid_mask, reproj_pts = compute_reprojection_error(
            mkpts_i, mkpts_j,
            depth_i,
            pred["extrinsic"][anchor_idx], pred["extrinsic"][j],
            pred["intrinsic"][anchor_idx], pred["intrinsic"][j],
        )

        # 可视化
        # images: (S, 3, H, W) → (H, W, 3) for cv2
        img_i = (np.transpose(pred["images"][anchor_idx], (1, 2, 0)) * 255).astype(np.uint8)
        img_j = (np.transpose(pred["images"][j], (1, 2, 0)) * 255).astype(np.uint8)

        vis_path = os.path.join(output_dir, f"match_{anchor_idx:02d}_{j:02d}.png")
        n_total, n_valid, n_correct, mean_err = visualize_pair(
            img_i, img_j,
            mkpts_i, mkpts_j,
            errors, valid_mask, reproj_pts,
            threshold, vis_path,
        )

        ratio = n_correct / n_total * 100 if n_total > 0 else 0
        print(f"matches={n_total}, valid={n_valid}, correct={n_correct} ({ratio:.1f}%), mean_err={mean_err:.2f}px")

        stats.append({
            "pair": f"{anchor_idx}-{j}",
            "total": n_total,
            "valid": n_valid,
            "correct": n_correct,
            "ratio": ratio,
            "mean_err": mean_err,
        })

    # 保存统计
    if stats:
        with open(os.path.join(output_dir, "stats.txt"), "w") as f:
            f.write(f"# Cluster: {cluster_name}\n")
            f.write(f"# Reproj threshold: {threshold}px\n\n")
            f.write(f"{'Pair':<8} {'Total':<6} {'Valid':<6} {'Correct':<8} {'Ratio%':<8} {'MeanErr':<8}\n")
            for s in stats:
                f.write(f"{s['pair']:<8} {s['total']:<6} {s['valid']:<6} "
                       f"{s['correct']:<8} {s['ratio']:<8.1f} {s['mean_err']:<8.2f}\n")

            # 汇总
            total_all = sum(s["total"] for s in stats)
            correct_all = sum(s["correct"] for s in stats)
            mean_err_all = np.mean([s["mean_err"] for s in stats])
            f.write(f"\n# Summary\n")
            f.write(f"Total matches: {total_all}\n")
            f.write(f"Total correct: {correct_all}\n")
            f.write(f"Overall ratio: {correct_all/total_all*100:.1f}%\n")
            f.write(f"Mean error: {mean_err_all:.2f}px\n")

        print(f"  Stats saved: {output_dir}/stats.txt")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Feature Matching + Reprojection Error Evaluation")
    parser.add_argument(
        "--cluster_dir", type=str, required=True,
        help="Cluster directory (e.g., output_vggt_cluster/cluster_001) or parent directory (with --process_all)",
    )
    parser.add_argument(
        "--process_all", action="store_true", default=False,
        help="Process all clusters in the parent directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Parent output directory for visualizations (default: cluster_dir/eval_matches or cluster_dir/../eval_all)",
    )
    parser.add_argument(
        "--reproj_threshold", type=float, default=4.0,
        help="Reprojection error threshold for correct match (pixels, default 4.0)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )
    args = parser.parse_args()

    # 初始化特征提取器和匹配器
    print("Initializing SuperPoint + LightGlue...")
    extractor, matcher = init_matcher(args.device)
    print("Ready.\n")

    all_stats = {}

    if args.process_all:
        # 处理所有 cluster
        parent_dir = args.cluster_dir
        cluster_dirs = sorted([
            d for d in Path(parent_dir).iterdir()
            if d.is_dir() and d.name.startswith("cluster_")
        ])

        output_dir = args.output_dir or os.path.join(parent_dir, "eval_all")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Found {len(cluster_dirs)} clusters to process")

        for cluster_dir in cluster_dirs:
            stats = process_cluster(
                str(cluster_dir),
                extractor, matcher,
                args.reproj_threshold, args.device,
                parent_output_dir=output_dir,
            )
            if stats:
                all_stats[cluster_dir.name] = stats

        # 全局汇总
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write(f"# Global Summary\n")
            f.write(f"# Reproj threshold: {args.reproj_threshold}px\n\n")
            f.write(f"{'Cluster':<15} {'Total':<8} {'Correct':<8} {'Ratio%':<8} {'MeanErr':<8}\n")

            grand_total = 0
            grand_correct = 0
            for name, stats in sorted(all_stats.items()):
                t = sum(s["total"] for s in stats)
                c = sum(s["correct"] for s in stats)
                m = np.mean([s["mean_err"] for s in stats])
                r = c / t * 100 if t > 0 else 0
                f.write(f"{name:<15} {t:<8} {c:<8} {r:<8.1f} {m:<8.2f}\n")
                grand_total += t
                grand_correct += c

            f.write(f"\n{'TOTAL':<15} {grand_total:<8} {grand_correct:<8} "
                   f"{grand_correct/grand_total*100:.1f}\n")

        print(f"\nAll done. Global summary: {output_dir}/summary.txt")

    else:
        # 处理单个 cluster
        stats = process_cluster(
            args.cluster_dir,
            extractor, matcher,
            args.reproj_threshold, args.device,
        )
        if stats:
            print(f"\nDone. Results in: {os.path.join(args.cluster_dir, 'eval_matches')}")


if __name__ == "__main__":
    main()
