#!/usr/bin/env python3
"""
VGGT Depth Map 不一致性可视化与观测

功能:
    1. 加载 VGGT 预测的 depth maps 和 camera poses
    2. 对 cluster 内每对相邻帧计算深度重投影一致性:
       - 帧 i 的每个像素反投影为 3D 点 P
       - P 通过 camera j 重投影，得到预测深度 Z_pred
       - 在帧 j 的 depth map 上采样对应位置的深度 Z_gt
       - inconsistency = |Z_pred - Z_gt| / Z_gt
    3. 可视化不一致性热力图、统计分布
    4. 输出各 cluster 的深度一致性报告

不一致性大的原因:
    - VGGT pose 估计不准（相机外参误差）
    - VGGT depth 估计不准（局部区域深度漂移）
    - 遮挡/视差导致重投影后对应关系错误
    - 动态物体

用法（单个 cluster）:
    /home/23036584r/anaconda3/envs/vggt/bin/python mycode/04_depth_inconsistency.py \
        --cluster_dir output_vggt_cluster/cluster_001 \
        --threshold 0.1

用法（所有 cluster）:
    /home/23036584r/anaconda3/envs/vggt/bin/python mycode/04_depth_inconsistency.py \
        --cluster_dir output_vggt_cluster \
        --process_all \
        --threshold 0.1
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# matplotlib 后端设置（无头环境）
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not installed, skipping heatmap visualization")
    plt = None


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CMAP_REL = "hot"
CMAP_ABS = "plasma"


def load_predictions(cluster_dir):
    """加载 VGGT 预测结果。"""
    pred_path = os.path.join(cluster_dir, "predictions.npz")
    data = np.load(pred_path)
    return {
        "depth": data["depth"],              # (S, H, W, 1)
        "extrinsic": data["extrinsic"],      # (S, 3, 4) OpenCV camera-from-world
        "intrinsic": data["intrinsic"],      # (S, 3, 3)
        "images": data["images"],            # (S, 3, H, W) [0, 1]
    }


def compute_depth_consistency(
    depth_i, depth_j,
    extrinsic_i, extrinsic_j,
    intrinsic_i, intrinsic_j,
):
    """
    计算帧 i -> 帧 j 的深度一致性。

    流程:
        1. 帧 i 的每个像素 (u, v) 用 depth_i 反投影为 3D 世界坐标 P_world
        2. P_world 用 camera j 重投影到帧 j，得到预测深度 Z_pred 和像素坐标 (u', v')
        3. 在帧 j 的 depth_j 上双线性采样 (u', v') 处的深度 Z_gt
        4. 计算 inconsistency:
             relative = |Z_pred - Z_gt| / Z_gt
             absolute = |Z_pred - Z_gt| (meters)

    Args:
        depth_i, depth_j: (H, W) 深度图
        extrinsic_i, extrinsic_j: (3, 4) 外参
        intrinsic_i, intrinsic_j: (3, 3) 内参

    Returns:
        rel_error:  (H, W) 相对误差
        abs_error:  (H, W) 绝对误差 (m)
        valid_mask: (H, W) 有效像素掩码
    """
    H, W = depth_i.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------------------------
    # Step 1: 创建像素坐标网格并反投影到世界坐标
    # -----------------------------------------------------------------------
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    # P_cam = Z * K^-1 * [u, v, 1]^T
    K_inv_i = np.linalg.inv(intrinsic_i)
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)           # (H, W, 3)
    pts_cam = depth_i[..., None] * (uv1 @ K_inv_i.T)           # (H, W, 3)

    # P_world = R_i^T * (P_cam - t_i)
    R_i = extrinsic_i[:, :3]
    t_i = extrinsic_i[:, 3]
    pts_world = (pts_cam - t_i) @ R_i.T                        # (H, W, 3)

    # -----------------------------------------------------------------------
    # Step 2: 重投影到帧 j
    # -----------------------------------------------------------------------
    R_j = extrinsic_j[:, :3]
    t_j = extrinsic_j[:, 3]
    pts_cam_j = (pts_world @ R_j.T) + t_j                      # (H, W, 3)

    Z_pred = pts_cam_j[..., 2]                                 # (H, W)
    valid_front = Z_pred > 0.01

    # 投影到像素平面
    uv_proj = pts_cam_j[..., :2] / np.maximum(Z_pred[..., None], 1e-6)  # (H, W, 2)
    uv_proj = uv_proj @ intrinsic_j[:2, :2].T + intrinsic_j[:2, 2]      # (H, W, 2)

    # -----------------------------------------------------------------------
    # Step 3: 在帧 j 的 depth map 上双线性采样
    # -----------------------------------------------------------------------
    # grid_sample 需要归一化到 [-1, 1]
    u_norm = 2.0 * uv_proj[..., 0] / (W - 1) - 1.0
    v_norm = 2.0 * uv_proj[..., 1] / (H - 1) - 1.0
    grid = np.stack([u_norm, v_norm], axis=-1)                 # (H, W, 2)

    # torch grid_sample
    depth_j_t = torch.from_numpy(depth_j).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    grid_t = torch.from_numpy(grid).unsqueeze(0).to(device)    # (1, H, W, 2)

    Z_gt_t = F.grid_sample(
        depth_j_t, grid_t,
        mode="bilinear", padding_mode="zeros", align_corners=True
    )
    Z_gt = Z_gt_t[0, 0].cpu().numpy()                          # (H, W)

    # -----------------------------------------------------------------------
    # Step 4: 计算 inconsistency
    # -----------------------------------------------------------------------
    valid = valid_front & (Z_gt > 0.1) & np.isfinite(Z_gt) & np.isfinite(Z_pred)

    rel_error = np.zeros((H, W), dtype=np.float32)
    abs_error = np.zeros((H, W), dtype=np.float32)

    rel_error[valid] = np.abs(Z_pred[valid] - Z_gt[valid]) / Z_gt[valid]
    abs_error[valid] = np.abs(Z_pred[valid] - Z_gt[valid])

    return rel_error, abs_error, valid, Z_pred, Z_gt


def visualize_consistency(
    image_j, rel_error, abs_error, valid, pair_name, output_dir, threshold=0.1
):
    """
    生成四格可视化图:
        - 左上: 参考图片 (帧 j)
        - 右上: 相对误差热力图
        - 左下: 绝对误差热力图
        - 右下: 不一致区域叠加
    """
    if plt is None:
        return None

    H, W = image_j.shape[:2]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # --- 左上: 参考图片 ---
    axes[0, 0].imshow(image_j)
    axes[0, 0].set_title(f"Reference Image (frame {pair_name.split('_')[1]})")
    axes[0, 0].axis("off")

    # --- 右上: 相对误差热力图 ---
    rel_vis = np.zeros((H, W), dtype=np.float32)
    rel_vis[valid] = rel_error[valid]
    rel_vis = np.clip(rel_vis, 0, threshold)

    im1 = axes[0, 1].imshow(rel_vis, cmap=CMAP_REL, vmin=0, vmax=threshold)
    axes[0, 1].set_title(f"Relative Inconsistency (|Z_pred-Z_gt|/Z_gt, max={threshold})")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # --- 左下: 绝对误差热力图 ---
    abs_vis = np.zeros((H, W), dtype=np.float32)
    abs_vis[valid] = abs_error[valid]
    abs_vis = np.clip(abs_vis, 0, 5.0)

    im2 = axes[1, 0].imshow(abs_vis, cmap=CMAP_ABS, vmin=0, vmax=5.0)
    axes[1, 0].set_title("Absolute Error (|Z_pred-Z_gt| in meters, max=5m)")
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # --- 右下: 不一致区域叠加 ---
    bad_mask = np.zeros((H, W), dtype=np.float32)
    if valid.sum() > 0:
        bad_mask[valid] = (rel_error[valid] > threshold / 2).astype(np.float32)

    axes[1, 1].imshow(image_j, alpha=0.6)
    axes[1, 1].imshow(bad_mask, cmap="Reds", alpha=0.4, vmin=0, vmax=1)
    axes[1, 1].set_title(f"Inconsistent Regions (rel_err > {threshold/2:.2f})")
    axes[1, 1].axis("off")

    # 统计文字
    if valid.sum() > 0:
        mean_rel = rel_error[valid].mean()
        median_rel = np.median(rel_error[valid])
        p90_rel = np.percentile(rel_error[valid], 90)
        bad_ratio = (rel_error[valid] > threshold).mean() * 100

        info_text = (
            f"valid pixels: {valid.sum()}/{H*W}\n"
            f"mean rel: {mean_rel:.3f}\n"
            f"median rel: {median_rel:.3f}\n"
            f"p90 rel: {p90_rel:.3f}\n"
            f"bad ratio: {bad_ratio:.1f}%"
        )
        fig.text(0.5, 0.02, info_text, ha="center", fontsize=11,
                 family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle(f"Depth Consistency: frame {pair_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_path = os.path.join(output_dir, f"consistency_{pair_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # 返回统计
    if valid.sum() > 0:
        return {
            "pair": pair_name,
            "valid_pixels": int(valid.sum()),
            "total_pixels": H * W,
            "mean_rel": float(rel_error[valid].mean()),
            "median_rel": float(np.median(rel_error[valid])),
            "p90_rel": float(np.percentile(rel_error[valid], 90)),
            "mean_abs": float(abs_error[valid].mean()),
            "bad_ratio": float((rel_error[valid] > threshold).mean() * 100),
        }
    return None


def plot_cluster_summary(stats_list, cluster_name, output_dir):
    """为单个 cluster 绘制误差分布统计图。"""
    if plt is None or len(stats_list) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pairs = [s["pair"] for s in stats_list]
    mean_rels = [s["mean_rel"] for s in stats_list]
    bad_ratios = [s["bad_ratio"] for s in stats_list]
    p90_rels = [s["p90_rel"] for s in stats_list]
    mean_abss = [s["mean_abs"] for s in stats_list]

    # --- mean relative error per pair ---
    axes[0, 0].bar(range(len(pairs)), mean_rels, color="steelblue")
    axes[0, 0].set_xticks(range(len(pairs)))
    axes[0, 0].set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
    axes[0, 0].set_ylabel("Mean Relative Error")
    axes[0, 0].set_title("Mean Relative Error per Pair")
    axes[0, 0].axhline(0.1, color="r", linestyle="--", label="threshold=0.1")
    axes[0, 0].legend()

    # --- bad ratio per pair ---
    axes[0, 1].bar(range(len(pairs)), bad_ratios, color="coral")
    axes[0, 1].set_xticks(range(len(pairs)))
    axes[0, 1].set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
    axes[0, 1].set_ylabel("Bad Ratio (%)")
    axes[0, 1].set_title(f"Bad Pixel Ratio (rel_err > threshold)")

    # --- p90 relative error ---
    axes[1, 0].bar(range(len(pairs)), p90_rels, color="seagreen")
    axes[1, 0].set_xticks(range(len(pairs)))
    axes[1, 0].set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("P90 Relative Error")
    axes[1, 0].set_title("90th Percentile Relative Error")
    axes[1, 0].axhline(0.1, color="r", linestyle="--")

    # --- mean absolute error ---
    axes[1, 1].bar(range(len(pairs)), mean_abss, color="mediumpurple")
    axes[1, 1].set_xticks(range(len(pairs)))
    axes[1, 1].set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
    axes[1, 1].set_ylabel("Mean Absolute Error (m)")
    axes[1, 1].set_title("Mean Absolute Error per Pair")

    plt.suptitle(f"Cluster Summary: {cluster_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "summary_plot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Summary plot: {out_path}")


def process_cluster(cluster_dir, threshold, parent_output_dir=None):
    """处理单个 cluster。"""
    cluster_name = os.path.basename(cluster_dir)
    print(f"\n{'='*60}")
    print(f"Processing {cluster_name}")
    print(f"{'='*60}")

    if parent_output_dir:
        output_dir = os.path.join(parent_output_dir, f"depth_eval_{cluster_name}")
    else:
        output_dir = os.path.join(cluster_dir, "depth_inconsistency")
    os.makedirs(output_dir, exist_ok=True)

    # 加载预测
    try:
        pred = load_predictions(cluster_dir)
    except Exception as e:
        print(f"  Skip: cannot load predictions: {e}")
        return None

    S = pred["images"].shape[0]
    H, W = pred["images"].shape[2], pred["images"].shape[3]
    print(f"  Images: {S}, Resolution: {H}x{W}")

    depth_maps = pred["depth"].squeeze(-1)  # (S, H, W)
    stats_list = []

    # 处理相邻帧对
    for i in range(S - 1):
        j = i + 1
        pair_name = f"{i:02d}_{j:02d}"
        print(f"  Pair {pair_name}...", end=" ", flush=True)

        rel_err, abs_err, valid, Z_pred, Z_gt = compute_depth_consistency(
            depth_maps[i], depth_maps[j],
            pred["extrinsic"][i], pred["extrinsic"][j],
            pred["intrinsic"][i], pred["intrinsic"][j],
        )

        # 可视化
        img_j = np.transpose(pred["images"][j], (1, 2, 0))  # (H, W, 3)
        stat = visualize_consistency(
            img_j, rel_err, abs_err, valid, pair_name, output_dir, threshold
        )

        if stat:
            print(f"mean_rel={stat['mean_rel']:.3f}, bad={stat['bad_ratio']:.1f}%")
            stats_list.append(stat)
        else:
            print("no valid pixels")

    # 保存统计
    if stats_list:
        with open(os.path.join(output_dir, "stats.txt"), "w") as f:
            f.write(f"# Depth Consistency Evaluation\n")
            f.write(f"# cluster: {cluster_name}\n")
            f.write(f"# threshold: {threshold}\n")
            f.write(f"# images: {S}, resolution: {H}x{W}\n\n")
            f.write(f"{'Pair':<8} {'Valid':<8} {'MeanRel':<10} {'MedianRel':<10} "
                   f"{'P90Rel':<10} {'MeanAbs':<10} {'Bad%':<8}\n")
            for s in stats_list:
                f.write(f"{s['pair']:<8} {s['valid_pixels']:<8} {s['mean_rel']:<10.3f} "
                       f"{s['median_rel']:<10.3f} {s['p90_rel']:<10.3f} "
                       f"{s['mean_abs']:<10.3f} {s['bad_ratio']:<8.1f}\n")

            # 汇总
            f.write(f"\n# Summary\n")
            f.write(f"mean_rel_all: {np.mean([s['mean_rel'] for s in stats_list]):.3f}\n")
            f.write(f"median_rel_all: {np.median([s['median_rel'] for s in stats_list]):.3f}\n")
            f.write(f"max_bad_ratio: {max(s['bad_ratio'] for s in stats_list):.1f}%\n")

        plot_cluster_summary(stats_list, cluster_name, output_dir)
        print(f"  Stats saved: {output_dir}/stats.txt")

    return stats_list


def main():
    parser = argparse.ArgumentParser(description="VGGT Depth Map Inconsistency Visualization")
    parser.add_argument(
        "--cluster_dir", type=str, required=True,
        help="Cluster directory or parent directory (with --process_all)",
    )
    parser.add_argument(
        "--process_all", action="store_true", default=False,
        help="Process all clusters in parent directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Parent output directory for visualizations",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Relative inconsistency threshold (default 0.1 = 10%)",
    )
    args = parser.parse_args()

    if args.process_all:
        parent_dir = args.cluster_dir
        cluster_dirs = sorted([
            d for d in Path(parent_dir).iterdir()
            if d.is_dir() and d.name.startswith("cluster_")
        ])

        output_dir = args.output_dir or os.path.join(parent_dir, "depth_eval_all")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Found {len(cluster_dirs)} clusters to process")

        all_stats = {}
        for cluster_dir in cluster_dirs:
            stats = process_cluster(str(cluster_dir), args.threshold, output_dir)
            if stats:
                all_stats[cluster_dir.name] = stats

        # 全局汇总
        with open(os.path.join(output_dir, "global_summary.txt"), "w") as f:
            f.write(f"# Global Depth Consistency Summary\n")
            f.write(f"# threshold: {args.threshold}\n\n")
            f.write(f"{'Cluster':<15} {'Pairs':<6} {'MeanRel':<10} {'MedianRel':<10} "
                   f"{'MaxBad%':<10} {'MeanAbs':<10}\n")

            for name, stats in sorted(all_stats.items()):
                mean_rel = np.mean([s["mean_rel"] for s in stats])
                median_rel = np.median([s["median_rel"] for s in stats])
                max_bad = max(s["bad_ratio"] for s in stats)
                mean_abs = np.mean([s["mean_abs"] for s in stats])
                f.write(f"{name:<15} {len(stats):<6} {mean_rel:<10.3f} {median_rel:<10.3f} "
                       f"{max_bad:<10.1f} {mean_abs:<10.3f}\n")

        print(f"\nAll done. Global summary: {output_dir}/global_summary.txt")

    else:
        process_cluster(args.cluster_dir, args.threshold)
        print(f"\nDone. Results in: {os.path.join(args.cluster_dir, 'depth_inconsistency')}")


if __name__ == "__main__":
    main()
