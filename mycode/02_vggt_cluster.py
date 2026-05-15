#!/usr/bin/env python3
"""
分治式 VGGT 重建：基于 DINO 特征聚类 + FPS anchor 选择

参考: vggt/dependency/vggsfm_utils.py 中的 generate_rank_by_dino + farthest_point_sampling

算法流程:
    1. 加载所有图像，提取 DINOv2 CLS token 特征
    2. 固定随机种子，随机选择第一张 init image 作为起始 anchor
    3. 迭代构建 clusters:
       a. 从未覆盖图片中，找与当前 anchor 特征最相似的 20 张
       b. anchor + 20 邻居组成一个 cluster (共 21 张)
       c. 用 VGGT 对该 cluster 进行重建 (depth + camera + point cloud)
       d. 标记 cluster 内图片为已覆盖
       e. 从未覆盖图片中，选与已覆盖集合特征距离最远的作为下一个 anchor (FPS)
    4. 重复直到所有图片被覆盖

用法:
    /home/23036584r/anaconda3/envs/vggt/bin/python mycode/02_vggt_cluster.py \
        --image_dir /home/23036584r/workspace/data/LVBA/CBD01 \
        --output_dir ./output_vggt_cluster \
        --images_per_cluster 21 \
        --seed 42
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 将 vggt 添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# ---------------------------------------------------------------------------
# DINO 相关常量
# ---------------------------------------------------------------------------
_RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_RESNET_STD = torch.tensor([0.229, 0.224, 0.225])


def load_image_paths(image_dir):
    """加载目录下的所有图像路径（按文件名排序）。"""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    paths = sorted([str(p) for p in Path(image_dir).iterdir() if p.suffix.lower() in exts])
    if len(paths) == 0:
        raise ValueError(f"No images found in {image_dir}")
    return paths


def extract_dino_features(image_paths, model_name="dinov2_vitb14_reg", batch_size=32, device="cuda"):
    """
    用 DINOv2 提取所有图片的 CLS token 特征。

    Args:
        image_paths: 图像路径列表
        model_name: DINOv2 模型名
        batch_size: 批大小
        device: 计算设备

    Returns:
        features: (N, C) numpy 数组，每帧的 CLS token 特征
    """
    dinov2_input_size = 336

    print(f"Loading DINOv2 model: {model_name} ...")
    dino_model = torch.hub.load("facebookresearch/dinov2", model_name)
    dino_model.eval().to(device)

    mean = _RESNET_MEAN.view(1, 3, 1, 1).to(device)
    std = _RESNET_STD.view(1, 3, 1, 1).to(device)

    all_features = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting DINO features"):
        batch_paths = image_paths[i:i + batch_size]

        # 加载并预处理图片
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = img.resize((dinov2_input_size, dinov2_input_size), Image.BILINEAR)
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append(tensor)

        batch = torch.stack(tensors, dim=0).to(device)
        batch = (batch - mean) / std

        with torch.no_grad():
            output = dino_model(batch, is_training=True)
            feat = output["x_norm_clstoken"]  # (B, C)
            all_features.append(feat.cpu())

    features = torch.cat(all_features, dim=0).numpy()  # (N, C)
    del dino_model
    torch.cuda.empty_cache()

    return features


def build_clusters(features, images_per_cluster=21, seed=42, num_clusters=None,
                     min_coverage=1):
    """
    基于 DINO 特征迭代构建 clusters（允许 cluster 间重复）。

    策略（参考 colmap demo 的多 query 帧逻辑）:
        1. 随机选第一个 anchor (init image)
        2. 从全部图片中（允许重复），找与当前 anchor 特征最相似的 k-1 张作为邻居
        3. anchor + 邻居组成 cluster
        4. cluster 内所有图片的访问计数 +1
        5. 选下一个 anchor:
           - 如果存在 visit_count < min_coverage 的图片：
             从这些未充分覆盖的图片中，选与已选 anchors 距离最远的 (FPS)
           - 否则（全部已充分覆盖）：
             从全部图片中，选与上一个 cluster 成员平均距离最远的 (全局 FPS)
        6. 重复直到 num_clusters 个 cluster 生成完毕，或所有图片覆盖 min_coverage 次

    与旧版的区别:
        - 旧版: 每个图片只能属于一个 cluster（互斥）
        - 新版: 每个图片可以属于多个 cluster（重叠），类似 colmap 多 query 帧

    Args:
        features: (N, C) DINO 特征
        images_per_cluster: 每个 cluster 包含的图片数 (anchor + neighbors)
        seed: 随机种子
        num_clusters: 生成多少个 cluster (None 则自动计算)
        min_coverage: 每张图片最少被覆盖多少次 (默认 1)

    Returns:
        clusters: list of dict
        visit_count: (N,) 每张图片被包含的 cluster 数量
    """
    N = features.shape[0]

    # ===================================================================
    # 基于 DINO 特征相似度选邻居 + FPS 选 anchor
    # ===================================================================
    rng = np.random.RandomState(seed)

    # 归一化特征并计算余弦距离矩阵
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sim_matrix = features_norm @ features_norm.T  # (N, N), 余弦相似度
    dist_matrix = 1 - sim_matrix  # 余弦距离 [0, 2], 越大越不相似

    # 默认 cluster 数: 考虑重叠后的估算
    if num_clusters is None:
        # 假设 30% 重叠，每个图片平均出现在 1.3 个 cluster 中
        num_clusters = int(np.ceil(N / (images_per_cluster * 0.7)))
        print(f"Auto num_clusters: {num_clusters}")

    visit_count = np.zeros(N, dtype=np.int32)
    clusters = []
    anchor_list = []

    # 随机选第一个 anchor
    anchor = rng.randint(0, N)
    anchor_list.append(anchor)
    print(f"Init anchor (random): {anchor}")

    for _ in range(num_clusters):
        # 从全部图片中（允许重复），找与 anchor 最近的 k-1 个邻居
        # 加入访问计数惩罚：优先选择访问次数少的图片，保证覆盖均匀
        k = min(images_per_cluster - 1, N - 1)
        dists = dist_matrix[anchor].copy()

        # 加权：visit_count 高的图片被惩罚，降低被选中概率
        # 惩罚系数 0.05 意味着每被访问一次，距离增加 0.05（约 2.5% 的 max 距离）
        penalty = 0.05
        weighted_dists = dists + penalty * visit_count
        weighted_dists[anchor] = np.inf  # 排除 anchor 自己

        # 取加权距离最小的 k 个
        nearest_idx = np.argsort(weighted_dists)[:k]
        neighbors = nearest_idx.tolist()

        members = [anchor] + neighbors
        clusters.append({"anchor": anchor, "neighbors": neighbors, "members": members})

        # 更新访问计数
        for idx in members:
            visit_count[idx] += 1

        # 统计当前 cluster 与之前 cluster 的重叠
        if len(clusters) > 1:
            prev_members = set(clusters[-2]["members"])
            overlap = len(prev_members & set(members))
            print(f"  Cluster {len(clusters)}: anchor={anchor}, members={len(members)}, "
                  f"overlap with prev={overlap}, min_visit={visit_count.min()}, max_visit={visit_count.max()}")
        else:
            print(f"  Cluster {len(clusters)}: anchor={anchor}, members={len(members)}, "
                  f"min_visit={visit_count.min()}, max_visit={visit_count.max()}")

        # 选下一个 anchor
        # 先检查是否所有图片都已充分覆盖
        under_covered = np.where(visit_count < min_coverage)[0]

        if len(under_covered) > 0:
            # 优先覆盖未充分覆盖的图片：从未充分覆盖中选与已选 anchors 最远的 (FPS)
            min_dists = np.min(dist_matrix[under_covered][:, anchor_list], axis=1)
            anchor = under_covered[np.argmax(min_dists)]
            print(f"    -> Next anchor (under-covered FPS): {anchor}")
        else:
            # 全部已充分覆盖：全局 FPS，选与上一个 cluster 平均距离最远的
            prev_members = clusters[-1]["members"]
            avg_dists = np.mean(dist_matrix[:, prev_members], axis=1)
            # 排除上一个 anchor 自己
            avg_dists[anchor] = -1
            anchor = int(np.argmax(avg_dists))
            print(f"    -> Next anchor (global FPS): {anchor}")

        anchor_list.append(anchor)

    return clusters, visit_count


def run_vggt(image_paths, model, device, dtype):
    """
    对一组图片运行 VGGT 推理。

    Args:
        image_paths: 图片路径列表
        model: VGGT 模型实例
        device: 计算设备
        dtype: 推理精度 (bfloat16 / float16)

    Returns:
        predictions: dict, 包含 depth, camera, point cloud 等
    """
    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

        # 解码相机参数
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # 转为 numpy 并移除 batch 维度
        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        predictions["pose_enc_list"] = None

        # 生成 depth-based 点云
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(
            depth_map, predictions["extrinsic"], predictions["intrinsic"]
        )
        predictions["world_points_from_depth"] = world_points

    torch.cuda.empty_cache()
    return predictions


def save_pointcloud_ply(vertices, colors, filepath):
    """
    保存点云为 PLY 格式 (纯点云，不含相机)。

    Args:
        vertices: (N, 3) 或 (S, H, W, 3)
        colors: (N, 3) 或 (S, H, W, 3)
        filepath: 输出路径
    """
    import trimesh

    # 展平
    if vertices.ndim == 4:
        vertices = vertices.reshape(-1, 3)
    if colors.ndim == 4:
        if colors.shape[1] == 3:  # NCHW -> NHWC
            colors = np.transpose(colors, (0, 2, 3, 1))
        colors = colors.reshape(-1, 3)
    if colors.dtype != np.uint8:
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    # 过滤无效点
    valid = np.isfinite(vertices).all(axis=1)
    vertices = vertices[valid]
    colors = colors[valid]

    if len(vertices) == 0:
        print(f"  Warning: no valid points to save to {filepath}")
        return

    pc = trimesh.PointCloud(vertices=vertices, colors=colors)
    pc.export(file_obj=filepath)
    print(f"  Point cloud saved: {filepath} ({len(vertices)} points)")


def main():
    parser = argparse.ArgumentParser(description="VGGT Cluster Reconstruction (DINO + FPS)")
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output_vggt_cluster",
        help="Output directory for reconstruction results",
    )
    parser.add_argument(
        "--images_per_cluster", type=int, default=21,
        help="Number of images per cluster (anchor + neighbors), default 21",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for anchor initialization",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )
    parser.add_argument(
        "--dino_model", type=str, default="dinov2_vitb14_reg",
        help="DINOv2 model name",
    )
    parser.add_argument(
        "--dino_batch_size", type=int, default=32,
        help="Batch size for DINO feature extraction",
    )
    parser.add_argument(
        "--num_clusters", type=int, default=None,
        help="Number of clusters to generate (None = auto estimate)",
    )
    parser.add_argument(
        "--min_coverage", type=int, default=1,
        help="Minimum times each image should appear in clusters (default 1)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 设置随机种子
    # -----------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed: {args.seed}")

    # -----------------------------------------------------------------------
    # Step 1: 加载图片路径
    # -----------------------------------------------------------------------
    image_paths = load_image_paths(args.image_dir)
    N = len(image_paths)
    print(f"Total images: {N}")

    # -----------------------------------------------------------------------
    # Step 2: 提取 DINO 特征
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Extracting DINOv2 features")
    print("=" * 60)
    features = extract_dino_features(
        image_paths, model_name=args.dino_model, batch_size=args.dino_batch_size, device=args.device
    )
    print(f"Features shape: {features.shape}")

    # 保存特征（供后续复用）
    np.save(os.path.join(args.output_dir, "dino_features.npy"), features)
    with open(os.path.join(args.output_dir, "image_paths.txt"), "w") as f:
        for i, p in enumerate(image_paths):
            f.write(f"{i}\t{p}\n")

    # -----------------------------------------------------------------------
    # Step 3: 构建 clusters
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Step 2: Building clusters (images_per_cluster={args.images_per_cluster})")
    print("=" * 60)
    clusters, visit_count = build_clusters(
        features,
        images_per_cluster=args.images_per_cluster,
        seed=args.seed,
        num_clusters=args.num_clusters,
        min_coverage=args.min_coverage,
    )
    print(f"\nTotal clusters: {len(clusters)}")
    print(f"Visit count stats: min={visit_count.min()}, max={visit_count.max()}, "
          f"mean={visit_count.mean():.1f}")

    # 保存 cluster 分配信息
    with open(os.path.join(args.output_dir, "clusters.txt"), "w") as f:
        f.write(f"# Cluster assignment (with overlap)\n")
        f.write(f"# seed: {args.seed}\n")
        f.write(f"# images_per_cluster: {args.images_per_cluster}\n")
        f.write(f"# num_clusters: {len(clusters)}\n")
        f.write(f"# min_coverage: {args.min_coverage}\n")
        f.write(f"# total_images: {N}\n")
        f.write(f"# visit_count: min={visit_count.min()}, max={visit_count.max()}, mean={visit_count.mean():.1f}\n\n")

        for cid, c in enumerate(clusters, 1):
            f.write(f"cluster {cid}:\n")
            f.write(f"  anchor: {c['anchor']}\n")
            f.write(f"  members ({len(c['members'])}): {c['members']}\n")
            for rank, idx in enumerate(c["members"]):
                marker = " [ANCHOR]" if idx == c["anchor"] else ""
                f.write(f"    {rank}: {idx} -> {image_paths[idx]}{marker}\n")
            f.write("\n")

        f.write("# Visit count per image\n")
        for i in range(N):
            f.write(f"  {i}: visit_count={visit_count[i]} -> {image_paths[i]}\n")

    # -----------------------------------------------------------------------
    # Step 4: 加载 VGGT 模型
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Loading VGGT model")
    print("=" * 60)
    model = VGGT()
    model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    print("Downloading VGGT weights...")
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
    model.eval().to(args.device)
    print("VGGT model loaded")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using dtype: {dtype}")

    # -----------------------------------------------------------------------
    # Step 5: 对每个 cluster 运行 VGGT 重建
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Reconstructing clusters with VGGT")
    print("=" * 60)

    for cid, cluster in enumerate(clusters, 1):
        members = cluster["members"]
        anchor = cluster["anchor"]

        cluster_dir = os.path.join(args.output_dir, f"cluster_{cid:03d}")
        os.makedirs(cluster_dir, exist_ok=True)

        # 复制图片到 cluster 目录
        cluster_img_dir = os.path.join(cluster_dir, "images")
        os.makedirs(cluster_img_dir, exist_ok=True)

        cluster_paths = []
        for rank, idx in enumerate(members):
            src = image_paths[idx]
            dst = os.path.join(cluster_img_dir, f"{rank:04d}_{os.path.basename(src)}")
            shutil.copy2(src, dst)
            cluster_paths.append(src)  # 使用原始路径（load_and_preprocess_images 需要）

        print(f"\nCluster {cid}/{len(clusters)}: {len(members)} images, anchor={anchor}")
        print(f"  Output: {cluster_dir}")

        try:
            # VGGT 推理
            predictions = run_vggt(cluster_paths, model, args.device, dtype)

            # 保存预测结果
            np.savez(
                os.path.join(cluster_dir, "predictions.npz"),
                **{k: v for k, v in predictions.items() if v is not None}
            )

            # 导出点云 PLY（不含相机）
            if "world_points_from_depth" in predictions:
                pts = predictions["world_points_from_depth"]
                imgs = predictions["images"]
                save_pointcloud_ply(
                    pts, imgs, os.path.join(cluster_dir, "pointcloud.ply")
                )

            print(f"  Cluster {cid} done.")

        except Exception as e:
            print(f"  ERROR on cluster {cid}: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print(f"Total images: {N}")
    print(f"Total clusters: {len(clusters)}")
    print(f"Images per cluster: ~{args.images_per_cluster}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
