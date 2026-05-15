#!/usr/bin/env python3
"""
Cluster 间 Pose 一致性观测

利用共享 image 的 relative pose 换算，评估不同 cluster 重建结果的一致性。

原理:
    Image k 同时出现在 cluster A 和 cluster B 中:
        - cluster A 中 image k 的 pose: T_A^k = [R_A | t_A] (camera-from-world)
        - cluster B 中 image k 的 pose: T_B^k = [R_B | t_B] (camera-from-world)

    两个 cluster 坐标系之间的相对变换:
        T_AB = inv(T_A^k) * T_B^k
             = [R_A^T * R_B | R_A^T * (t_B - t_A)]

    如果所有共享 images 给出一致的 T_AB，说明两个 cluster 的重建一致。
    如果不一致，说明 VGGT pose 估计存在误差。

用法:
    /home/23036584r/anaconda3/envs/vggt/bin/python mycode/05_cluster_pose_consistency.py \
        --cluster_a output_vggt_cluster/cluster_001 \
        --cluster_b output_vggt_cluster/cluster_002
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_cluster_poses(cluster_dir):
    """
    加载 cluster 的 VGGT 预测结果。

    Returns:
        poses: dict, {filename: {"extrinsic": (3,4), "intrinsic": (3,3), "index": int}}
    """
    pred_path = os.path.join(cluster_dir, "predictions.npz")
    data = np.load(pred_path)

    image_dir = os.path.join(cluster_dir, "images")
    image_names = sorted([
        p.name for p in Path(image_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])

    extrinsics = data["extrinsic"]   # (S, 3, 4)
    intrinsics = data["intrinsic"]   # (S, 3, 3)

    poses = {}
    for i, name in enumerate(image_names):
        poses[name] = {
            "extrinsic": extrinsics[i],
            "intrinsic": intrinsics[i],
            "index": i,
        }
    return poses


def compute_relative_transform(T_A, T_B):
    """
    计算 T_AB = inv(T_A) * T_B。

    T_A, T_B: (3, 4) OpenCV camera-from-world [R | t]

    Returns:
        T_AB: (3, 4) 相对变换 [R_AB | t_AB]
        rot_angle: 旋转角 (度)
        trans_mag: 平移量
    """
    R_A = T_A[:, :3]
    t_A = T_A[:, 3]
    R_B = T_B[:, :3]
    t_B = T_B[:, 3]

    # inv(T_A) = [R_A^T | -R_A^T * t_A]
    # T_AB = inv(T_A) * T_B = [R_A^T * R_B | R_A^T * (t_B - t_A)]
    R_AB = R_A.T @ R_B
    t_AB = R_A.T @ (t_B - t_A)

    # 旋转角
    rot = Rotation.from_matrix(R_AB)
    rot_angle = np.linalg.norm(rot.as_rotvec()) * 180 / np.pi

    # 平移量
    trans_mag = np.linalg.norm(t_AB)

    T_AB = np.hstack([R_AB, t_AB[:, None]])
    return T_AB, rot_angle, trans_mag


def align_scale(results):
    """
    对旋转一致的共享图片进行尺度统一。

    输入:
        results: list of dict, 每个元素包含 'T_AB' (3,4)

    输出:
        T_AB_unified: (3, 4) 统一后的相对变换 [R_AB | t_AB_unified]
    """
    # ---- Step 1: 统一旋转 ----
    rotations = Rotation.from_matrix(np.stack([r["T_AB"][:, :3] for r in results]))
    R_AB_unified = rotations.mean().as_matrix()

    # ---- Step 2: 统一平移（尺度对齐）----
    t_ABs = np.stack([r["T_AB"][:, 3] for r in results])  # (N, 3)

    # 计算平均方向
    norms = np.linalg.norm(t_ABs, axis=1, keepdims=True) + 1e-8
    directions = t_ABs / norms
    d = np.mean(directions, axis=0)
    d = d / (np.linalg.norm(d) + 1e-8)

    # 标量投影
    projections = t_ABs @ d  # (N,)
    s = np.median(projections)

    t_AB_unified = s * d

    T_AB_unified = np.hstack([R_AB_unified, t_AB_unified[:, None]])
    return T_AB_unified


def evaluate_with_aligned_pose(poses_A, poses_B, shared, T_AB_unified):
    """
    用统一后的 T_AB 将 cluster B 的 pose 转换到 cluster A 坐标系，重新评估一致性。

    流程:
        T_B_in_A = T_B * inv(T_AB_unified)
        然后比较 T_A 和 T_B_in_A 的差异

    Returns:
        aligned_rot_angles: (N,) 对齐后的旋转误差 (度)
        aligned_trans_mags: (N,) 对齐后的平移误差
    """
    R_AB = T_AB_unified[:, :3]
    t_AB = T_AB_unified[:, 3]

    # inv(T_AB) = [R_AB^T | -R_AB^T * t_AB]
    R_AB_inv = R_AB.T
    t_AB_inv = -R_AB_inv @ t_AB

    aligned_rot_angles = []
    aligned_trans_mags = []

    for name in shared:
        T_A = poses_A[name]["extrinsic"]
        T_B = poses_B[name]["extrinsic"]

        R_A = T_A[:, :3]
        t_A = T_A[:, 3]
        R_B = T_B[:, :3]
        t_B = T_B[:, 3]

        # T_B_in_A = T_B * inv(T_AB_unified)
        R_B_in_A = R_B @ R_AB_inv
        t_B_in_A = t_B + R_B @ t_AB_inv

        # 旋转误差: angle(R_A^T * R_B_in_A)
        R_diff = R_A.T @ R_B_in_A
        rot = Rotation.from_matrix(R_diff)
        rot_angle = np.linalg.norm(rot.as_rotvec()) * 180 / np.pi

        # 平移误差: ||t_A - t_B_in_A||
        trans_mag = np.linalg.norm(t_A - t_B_in_A)

        aligned_rot_angles.append(rot_angle)
        aligned_trans_mags.append(trans_mag)

    return np.array(aligned_rot_angles), np.array(aligned_trans_mags)


def visualize_consistency(shared_names, rot_angles, trans_mags, cluster_a_name, cluster_b_name, output_dir,
                          aligned_rot=None, aligned_trans=None):
    """可视化两个 cluster 的 pose 一致性。如果提供了 aligned 数据，额外展示对齐后的结果。"""
    if plt is None:
        return

    has_aligned = aligned_rot is not None and aligned_trans is not None

    if has_aligned:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    x = np.arange(len(shared_names))

    # --- 原始旋转角 ---
    axes[0, 0].bar(x, rot_angles, color="steelblue")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"{i}" for i in range(len(shared_names))], fontsize=7)
    axes[0, 0].set_ylabel("Rotation Angle (°)")
    axes[0, 0].set_title("Raw Relative Rotation Angle")
    axes[0, 0].axhline(np.mean(rot_angles), color="r", linestyle="--", label=f"mean={np.mean(rot_angles):.2f}°")
    axes[0, 0].legend()

    # --- 原始平移量 ---
    axes[0, 1].bar(x, trans_mags, color="coral")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"{i}" for i in range(len(shared_names))], fontsize=7)
    axes[0, 1].set_ylabel("Translation Magnitude")
    axes[0, 1].set_title("Raw Relative Translation Magnitude")
    axes[0, 1].axhline(np.mean(trans_mags), color="r", linestyle="--", label=f"mean={np.mean(trans_mags):.4f}")
    axes[0, 1].legend()

    # --- 旋转角分布直方图 ---
    axes[1, 0].hist(rot_angles, bins=20, color="steelblue", edgecolor="black")
    axes[1, 0].set_xlabel("Rotation Angle (°)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Raw Rotation Angle Distribution")
    axes[1, 0].axvline(np.mean(rot_angles), color="r", linestyle="--", label=f"mean={np.mean(rot_angles):.2f}°")
    axes[1, 0].legend()

    # --- 平移量分布直方图 ---
    axes[1, 1].hist(trans_mags, bins=20, color="coral", edgecolor="black")
    axes[1, 1].set_xlabel("Translation Magnitude")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Raw Translation Magnitude Distribution")
    axes[1, 1].axvline(np.mean(trans_mags), color="r", linestyle="--", label=f"mean={np.mean(trans_mags):.4f}")
    axes[1, 1].legend()

    # --- 对齐后的结果（如果有） ---
    if has_aligned:
        # Aligned 旋转角
        axes[2, 0].bar(x, aligned_rot, color="seagreen")
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels([f"{i}" for i in range(len(shared_names))], fontsize=7)
        axes[2, 0].set_ylabel("Rotation Error (°)")
        axes[2, 0].set_title("Aligned Rotation Error (T_B in A coords)")
        axes[2, 0].axhline(np.mean(aligned_rot), color="r", linestyle="--", label=f"mean={np.mean(aligned_rot):.2f}°")
        axes[2, 0].legend()

        # Aligned 平移误差
        axes[2, 1].bar(x, aligned_trans, color="mediumpurple")
        axes[2, 1].set_xticks(x)
        axes[2, 1].set_xticklabels([f"{i}" for i in range(len(shared_names))], fontsize=7)
        axes[2, 1].set_ylabel("Translation Error")
        axes[2, 1].set_title("Aligned Translation Error (T_B in A coords)")
        axes[2, 1].axhline(np.mean(aligned_trans), color="r", linestyle="--", label=f"mean={np.mean(aligned_trans):.4f}")
        axes[2, 1].legend()

    plt.suptitle(f"Pose Consistency: {cluster_a_name} vs {cluster_b_name}\n({len(shared_names)} shared images)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "pose_consistency.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Visualization: {out_path}")


def process_pair(cluster_a_dir, cluster_b_dir, output_dir):
    """处理一对 cluster。"""
    cluster_a_name = os.path.basename(cluster_a_dir)
    cluster_b_name = os.path.basename(cluster_b_dir)

    print(f"\n{'='*60}")
    print(f"Pose Consistency: {cluster_a_name} vs {cluster_b_name}")
    print(f"{'='*60}")

    # 加载 poses
    poses_A = load_cluster_poses(cluster_a_dir)
    poses_B = load_cluster_poses(cluster_b_dir)

    # 找共享 images
    shared = [name for name in poses_A if name in poses_B]
    print(f"  Shared images: {len(shared)}")

    if len(shared) == 0:
        print("  No shared images found.")
        return None

    # 计算每对共享 image 的 relative transform
    results = []
    rot_angles = []
    trans_mags = []

    for name in shared:
        T_A = poses_A[name]["extrinsic"]
        T_B = poses_B[name]["extrinsic"]

        T_AB, rot_angle, trans_mag = compute_relative_transform(T_A, T_B)

        results.append({
            "name": name,
            "T_AB": T_AB,
            "rot_angle": rot_angle,
            "trans_mag": trans_mag,
        })
        rot_angles.append(rot_angle)
        trans_mags.append(trans_mag)

    rot_angles = np.array(rot_angles)
    trans_mags = np.array(trans_mags)

    # 统计
    print(f"\n  Rotation Angle (°):")
    print(f"    mean={rot_angles.mean():.3f}, std={rot_angles.std():.3f}, "
          f"min={rot_angles.min():.3f}, max={rot_angles.max():.3f}")

    print(f"\n  Translation Magnitude:")
    print(f"    mean={trans_mags.mean():.4f}, std={trans_mags.std():.4f}, "
          f"min={trans_mags.min():.4f}, max={trans_mags.max():.4f}")

    # 一致性判断
    rot_threshold = 5.0    # 5度
    trans_threshold = 0.5  # 0.5米 (相对坐标系)

    consistent = (rot_angles < rot_threshold) & (trans_mags < trans_threshold)
    print(f"\n  Raw Consistency (rot<{rot_threshold}° & trans<{trans_threshold}):")
    print(f"    {consistent.sum()}/{len(shared)} ({consistent.sum()/len(shared)*100:.1f}%)")

    # ---- 尺度统一 (Scale Alignment) ----
    aligned_rot = None
    aligned_trans = None
    scale_aligned = False

    if rot_angles.max() < rot_threshold:
        print(f"\n  Rotation consistent (max={rot_angles.max():.3f}° < {rot_threshold}°). Performing scale alignment...")

        T_AB_unified = align_scale(results)
        aligned_rot, aligned_trans = evaluate_with_aligned_pose(poses_A, poses_B, shared, T_AB_unified)

        scale_aligned = True
        print(f"\n  Aligned Rotation Error (°):")
        print(f"    mean={aligned_rot.mean():.3f}, std={aligned_rot.std():.3f}, "
              f"min={aligned_rot.min():.3f}, max={aligned_rot.max():.3f}")
        print(f"\n  Aligned Translation Error:")
        print(f"    mean={aligned_trans.mean():.4f}, std={aligned_trans.std():.4f}, "
              f"min={aligned_trans.min():.4f}, max={aligned_trans.max():.4f}")

        aligned_consistent = (aligned_rot < rot_threshold) & (aligned_trans < trans_threshold)
        print(f"\n  Aligned Consistency (rot<{rot_threshold}° & trans<{trans_threshold}):")
        print(f"    {aligned_consistent.sum()}/{len(shared)} ({aligned_consistent.sum()/len(shared)*100:.1f}%)")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "pose_consistency.txt"), "w") as f:
        f.write(f"# Pose Consistency: {cluster_a_name} vs {cluster_b_name}\n")
        f.write(f"# Shared images: {len(shared)}\n\n")

        # Raw results
        f.write("# Raw Results (T_AB = inv(T_A) * T_B per shared image)\n")
        f.write(f"{'Idx':<5} {'Name':<40} {'Rot(°)':<10} {'Trans':<10} {'Consistent':<12}\n")
        f.write("-" * 80 + "\n")
        for i, r in enumerate(results):
            cons = "YES" if consistent[i] else "NO"
            f.write(f"{i:<5} {r['name']:<40} {r['rot_angle']:<10.3f} {r['trans_mag']:<10.4f} {cons:<12}\n")

        f.write("\n# Raw Summary\n")
        f.write(f"Rotation: mean={rot_angles.mean():.3f}°, std={rot_angles.std():.3f}°\n")
        f.write(f"Translation: mean={trans_mags.mean():.4f}, std={trans_mags.std():.4f}\n")
        f.write(f"Consistent: {consistent.sum()}/{len(shared)} ({consistent.sum()/len(shared)*100:.1f}%)\n")

        # Aligned results
        if scale_aligned:
            f.write("\n" + "=" * 60 + "\n")
            f.write("# Scale-Aligned Results (T_B transformed into A coords via unified T_AB)\n")
            f.write(f"{'Idx':<5} {'Name':<40} {'RotErr(°)':<12} {'TransErr':<12} {'Consistent':<12}\n")
            f.write("-" * 80 + "\n")
            for i, name in enumerate(shared):
                cons = "YES" if aligned_consistent[i] else "NO"
                f.write(f"{i:<5} {name:<40} {aligned_rot[i]:<12.3f} {aligned_trans[i]:<12.4f} {cons:<12}\n")

            f.write("\n# Aligned Summary\n")
            f.write(f"Rotation Error: mean={aligned_rot.mean():.3f}°, std={aligned_rot.std():.3f}°\n")
            f.write(f"Translation Error: mean={aligned_trans.mean():.4f}, std={aligned_trans.std():.4f}\n")
            f.write(f"Consistent: {aligned_consistent.sum()}/{len(shared)} ({aligned_consistent.sum()/len(shared)*100:.1f}%)\n")

    print(f"\n  Report: {output_dir}/pose_consistency.txt")

    # 可视化
    visualize_consistency(shared, rot_angles, trans_mags, cluster_a_name, cluster_b_name, output_dir,
                          aligned_rot=aligned_rot, aligned_trans=aligned_trans)

    return results


def scan_all_pairs(parent_dir, output_dir):
    """扫描 parent_dir 下所有 cluster 对，找出有共享 images 的组合。"""
    cluster_dirs = sorted([
        d for d in Path(parent_dir).iterdir()
        if d.is_dir() and d.name.startswith("cluster_")
    ])

    print(f"Found {len(cluster_dirs)} clusters")

    all_poses = {}
    for d in cluster_dirs:
        all_poses[d.name] = load_cluster_poses(str(d))

    # 找所有有共享 images 的 pair
    pairs = []
    for i in range(len(cluster_dirs)):
        for j in range(i + 1, len(cluster_dirs)):
            name_a = cluster_dirs[i].name
            name_b = cluster_dirs[j].name
            shared = set(all_poses[name_a].keys()) & set(all_poses[name_b].keys())
            if len(shared) > 0:
                pairs.append((name_a, name_b, len(shared)))

    print(f"\nPairs with shared images: {len(pairs)}")
    for a, b, n in pairs:
        print(f"  {a} <-> {b}: {n} shared")

    # 处理每对
    all_results = {}
    for a, b, n in pairs:
        pair_output = os.path.join(output_dir, f"{a}_vs_{b}")
        results = process_pair(
            os.path.join(parent_dir, a),
            os.path.join(parent_dir, b),
            pair_output,
        )
        if results:
            all_results[f"{a}_vs_{b}"] = results

    # 全局汇总
    with open(os.path.join(output_dir, "global_pose_summary.txt"), "w") as f:
        f.write("# Global Pose Consistency Summary\n\n")
        f.write(f"{'Pair':<30} {'Shared':<8} {'MeanRot°':<12} {'StdRot°':<12} {'MeanTrans':<12} {'Consistent%':<12}\n")
        for pair_name, results in sorted(all_results.items()):
            rot_angles = np.array([r["rot_angle"] for r in results])
            trans_mags = np.array([r["trans_mag"] for r in results])
            consistent = (rot_angles < 5.0) & (trans_mags < 0.5)
            f.write(f"{pair_name:<30} {len(results):<8} {rot_angles.mean():<12.3f} "
                   f"{rot_angles.std():<12.3f} {trans_mags.mean():<12.4f} "
                   f"{consistent.sum()/len(results)*100:<12.1f}\n")

    print(f"\nGlobal summary: {output_dir}/global_pose_summary.txt")


def main():
    parser = argparse.ArgumentParser(description="Cluster Pose Consistency Evaluation")
    parser.add_argument("--cluster_a", type=str, default=None, help="First cluster directory")
    parser.add_argument("--cluster_b", type=str, default=None, help="Second cluster directory")
    parser.add_argument("--parent_dir", type=str, default=None, help="Parent directory containing all clusters")
    parser.add_argument("--output_dir", type=str, default="./output_pose_consistency", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.parent_dir:
        # 扫描所有 cluster 对
        scan_all_pairs(args.parent_dir, args.output_dir)
    elif args.cluster_a and args.cluster_b:
        # 处理单个 pair
        pair_name = f"{os.path.basename(args.cluster_a)}_vs_{os.path.basename(args.cluster_b)}"
        output_dir = os.path.join(args.output_dir, pair_name)
        process_pair(args.cluster_a, args.cluster_b, output_dir)
    else:
        print("Error: specify either --parent_dir or both --cluster_a and --cluster_b")
        return

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
