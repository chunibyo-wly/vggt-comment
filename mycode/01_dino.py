#!/usr/bin/env python3
"""
DINOv2 特征聚类 + Farthest Point Sampling (FPS) 帧选择

参考: vggt/dependency/vggsfm_utils.py 中的 generate_rank_by_dino
功能:
    1. 加载输入目录下的所有图像
    2. 用 DINOv2 提取每帧的 CLS token 特征
    3. 计算帧间余弦相似度矩阵
    4. 用 Farthest Point Sampling (FPS) 选出最具代表性的帧
    5. 输出选择的帧索引、相似度矩阵可视化

用法:
    /home/23036584r/anaconda3/envs/vggt/bin/python mycode/01_dino.py \
        --image_dir /home/23036584r/workspace/data/LVBA/CBD01 \
        --output_dir ./output_dino \
        --query_frame_num 10
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_RESNET_STD = torch.tensor([0.229, 0.224, 0.225])


def load_images(image_dir, image_size=336, device="cuda"):
    """
    加载目录下的所有图像，并预处理为 DINOv2 输入格式。

    Args:
        image_dir: 图像目录路径
        image_size: DINOv2 输入分辨率 (默认 336)
        device: 计算设备

    Returns:
        images_tensor: (S, 3, H, W), 范围 [0, 1]
        image_paths: 图像路径列表
    """
    # 支持的图像格式
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    image_paths = sorted(
        [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]
    )

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Found {len(image_paths)} images")

    tensors = []
    for p in tqdm(image_paths, desc="Loading images"):
        img = Image.open(p).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0  # H, W, 3
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # 3, H, W
        tensors.append(tensor)

    images_tensor = torch.stack(tensors, dim=0).to(device)  # S, 3, H, W
    return images_tensor, [str(p) for p in image_paths]


def farthest_point_sampling(distance_matrix, num_samples, most_common_frame_index=0):
    """
    最远点采样 (FPS)

    从帧集合中选出特征差异最大的帧，保证视角多样性。

    Args:
        distance_matrix: 帧间距离矩阵 (S, S)
        num_samples: 要选出的帧数
        most_common_frame_index: 起始帧索引

    Returns:
        List[int]: 选出的帧索引列表
    """
    distance_matrix = distance_matrix.clamp(min=0)
    N = distance_matrix.size(0)

    selected_indices = [most_common_frame_index]
    check_distances = distance_matrix[selected_indices]

    while len(selected_indices) < num_samples:
        farthest_point = torch.argmax(check_distances)
        selected_indices.append(farthest_point.item())
        check_distances = distance_matrix[farthest_point]
        check_distances[selected_indices] = 0

        if len(selected_indices) == N:
            break

    return selected_indices


def dino_fps_select(
    images_tensor,
    query_frame_num,
    model_name="dinov2_vitb14_reg",
    device="cuda",
    spatial_similarity=False,
):
    """
    DINOv2 特征提取 + FPS 帧选择

    Args:
        images_tensor: (S, 3, H, W), 范围 [0, 1]
        query_frame_num: 要选出的帧数
        model_name: DINOv2 模型名
        device: 计算设备
        spatial_similarity: 是否用 patch token

    Returns:
        selected_indices: List[int], 选出的帧索引
        similarity_matrix: (S, S), 帧间余弦相似度
        features: (S, C), 帧特征向量
    """
    S = images_tensor.shape[0]

    # 用 DINOv2 官方输入尺寸 (patch_size=14, 推荐 336 或 518)
    # dinov2_vitb14_reg 的 patch_size=14, 336 是 14 的整数倍
    dinov2_input_size = 336
    images_resized = F.interpolate(
        images_tensor, (dinov2_input_size, dinov2_input_size), mode="bilinear", align_corners=False
    )

    # ResNet 归一化 (DINOv2 训练时用的)
    mean = _RESNET_MEAN.view(1, 3, 1, 1).to(device)
    std = _RESNET_STD.view(1, 3, 1, 1).to(device)
    images_norm = (images_resized - mean) / std

    print(f"Loading DINOv2 model: {model_name} ...")
    dino_model = torch.hub.load("facebookresearch/dinov2", model_name)
    dino_model.eval()
    dino_model = dino_model.to(device)

    print("Extracting features ...")
    with torch.no_grad():
        output = dino_model(images_norm, is_training=True)

    # 选择特征类型
    if spatial_similarity:
        # 使用 patch token，计算空间相似度
        feat = output["x_norm_patchtokens"]  # S, num_patches, C
        feat_norm = F.normalize(feat, p=2, dim=-1)
        feat_norm = feat_norm.permute(1, 0, 2)  # num_patches, S, C
        sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(-1, -2))
        sim_matrix = sim_matrix.mean(dim=0)  # S, S
        features = feat.mean(dim=1)  # S, C
    else:
        # 使用 CLS token (默认)
        features = output["x_norm_clstoken"]  # S, C
        feat_norm = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.mm(feat_norm, feat_norm.transpose(-1, -2))  # S, S

    # 距离矩阵 = 100 - 相似度 (FPS 用距离选最远)
    distance_matrix = 100 - sim_matrix.clone()

    # 找 "最常见" 帧：与其他帧相似度总和最大的帧
    sim_no_diag = sim_matrix.clone()
    sim_no_diag.fill_diagonal_(-100)
    similarity_sum = sim_no_diag.sum(dim=1)
    most_common_idx = torch.argmax(similarity_sum).item()

    # FPS 采样
    print(f"Running FPS sampling (num_samples={query_frame_num}) ...")
    selected_indices = farthest_point_sampling(
        distance_matrix, query_frame_num, most_common_idx
    )

    # 清理显存
    del output, feat_norm, sim_matrix, distance_matrix
    del dino_model
    torch.cuda.empty_cache()

    return selected_indices, sim_no_diag.cpu().numpy(), features.cpu().numpy()


def visualize_similarity(similarity_matrix, image_paths, selected_indices, output_dir):
    """
    可视化帧间相似度矩阵，并标注 FPS 选择的帧。

    Args:
        similarity_matrix: (S, S)
        image_paths: 图像路径列表
        selected_indices: 选出的帧索引
        output_dir: 输出目录
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    S = len(image_paths)
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(similarity_matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Frame Index")
    ax.set_title("Frame Similarity Matrix (DINOv2 CLS Token)")

    # 标注 FPS 选择的帧
    for idx in selected_indices:
        ax.axvline(x=idx, color="red", linestyle="--", alpha=0.3)
        ax.axhline(y=idx, color="red", linestyle="--", alpha=0.3)

    # 在右侧标注选中的帧序号
    for rank, idx in enumerate(selected_indices):
        ax.text(S + 1, idx, f"  #{rank}", color="red", fontsize=8, va="center")

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "similarity_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Similarity matrix saved to: {out_path}")


def copy_selected_frames(image_paths, selected_indices, output_dir):
    """
    将 FPS 选出的帧复制到输出目录。

    Args:
        image_paths: 原始图像路径列表
        selected_indices: 选出的帧索引
        output_dir: 输出目录
    """
    selected_dir = os.path.join(output_dir, "selected_frames")
    os.makedirs(selected_dir, exist_ok=True)

    for rank, idx in enumerate(selected_indices):
        src = image_paths[idx]
        dst = os.path.join(selected_dir, f"{rank:03d}_frame{idx:04d}_{os.path.basename(src)}")
        import shutil

        shutil.copy2(src, dst)

    print(f"Selected frames copied to: {selected_dir}")


def main():
    parser = argparse.ArgumentParser(description="DINOv2 + FPS Frame Selection")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dino",
        help="Output directory for results",
    )
    parser.add_argument(
        "--query_frame_num",
        type=int,
        default=10,
        help="Number of frames to select via FPS",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dinov2_vitb14_reg",
        help="DINOv2 model name (dinov2_vitb14_reg, dinov2_vitl14_reg, etc.)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )
    parser.add_argument(
        "--spatial_similarity",
        action="store_true",
        default=False,
        help="Use spatial patch tokens instead of CLS token",
    )
    parser.add_argument(
        "--no_copy",
        action="store_true",
        default=False,
        help="Do not copy selected frames to output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load images
    # -----------------------------------------------------------------------
    images_tensor, image_paths = load_images(args.image_dir, device=args.device)
    print(f"Loaded {len(image_paths)} images, shape: {images_tensor.shape}")

    # -----------------------------------------------------------------------
    # Step 2: DINO feature extraction + FPS
    # -----------------------------------------------------------------------
    selected_indices, similarity_matrix, features = dino_fps_select(
        images_tensor,
        query_frame_num=args.query_frame_num,
        model_name=args.model_name,
        device=args.device,
        spatial_similarity=args.spatial_similarity,
    )

    print(f"\nSelected {len(selected_indices)} frames (FPS order):")
    for rank, idx in enumerate(selected_indices):
        print(f"  [{rank}] frame {idx}: {image_paths[idx]}")

    # -----------------------------------------------------------------------
    # Step 3: Save results
    # -----------------------------------------------------------------------
    # 保存帧索引
    indices_path = os.path.join(args.output_dir, "selected_indices.txt")
    with open(indices_path, "w") as f:
        f.write(f"# DINOv2 + FPS selected frames\n")
        f.write(f"# image_dir: {args.image_dir}\n")
        f.write(f"# query_frame_num: {args.query_frame_num}\n")
        f.write(f"# model_name: {args.model_name}\n")
        f.write(f"# total_frames: {len(image_paths)}\n")
        f.write("# rank, frame_index, filename\n")
        for rank, idx in enumerate(selected_indices):
            f.write(f"{rank}, {idx}, {image_paths[idx]}\n")
    print(f"\nSelected indices saved to: {indices_path}")

    # 保存相似度矩阵
    np.save(os.path.join(args.output_dir, "similarity_matrix.npy"), similarity_matrix)
    print(f"Similarity matrix saved to: {args.output_dir}/similarity_matrix.npy")

    # 保存特征向量
    np.save(os.path.join(args.output_dir, "features.npy"), features)
    print(f"Features saved to: {args.output_dir}/features.npy")

    # 可视化相似度矩阵
    visualize_similarity(
        similarity_matrix, image_paths, selected_indices, args.output_dir
    )

    # 复制选出的帧
    if not args.no_copy:
        copy_selected_frames(image_paths, selected_indices, args.output_dir)

    print(f"\nDone. All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
