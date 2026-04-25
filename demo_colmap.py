# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    """
    【VGGT 推理核心 - 重点阅读】
    运行 VGGT 模型, 提取相机参数和深度图。

    输出流程:
        1. Aggregator 提取特征
        2. CameraHead 输出 pose_enc -> pose_encoding_to_extri_intri 解码为:
           - extrinsic (S, 3, 4): 外参矩阵, OpenCV camera-from-world
           - intrinsic (S, 3, 3): 内参矩阵
        3. DepthHead 输出:
           - depth_map (S, 1, H, W): 深度图
           - depth_conf (S, 1, H, W): 深度置信度

    Args:
        model: VGGT 模型实例
        images (Tensor): 输入图像 [B, 3, H, W], 范围 [0, 1]
        dtype: 推理精度 (bfloat16 或 float16)
        resolution (int): VGGT 固定处理分辨率 (默认 518)

    Returns:
        tuple: (extrinsic, intrinsic, depth_map, depth_conf) 均为 numpy 数组
    """
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # VGGT 固定使用 518x518 分辨率
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # 添加 batch 维度: [1, B, 3, H, W]
            # 【Step 1: 特征聚合】
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # 【Step 2: 相机预测】
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # 【关键解码】pose_enc -> 外参 + 内参, OpenCV camera-from-world 约定
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # 【Step 3: 深度预测】
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # 移除 batch 维度并转为 numpy
    extrinsic = extrinsic.squeeze(0).cpu().numpy()    # (S, 3, 4)
    intrinsic = intrinsic.squeeze(0).cpu().numpy()    # (S, 3, 3)
    depth_map = depth_map.squeeze(0).cpu().numpy()    # (S, 1, H, W)
    depth_conf = depth_conf.squeeze(0).cpu().numpy()  # (S, 1, H, W)
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    """
    【COLMAP 导出完整流程 - 重点阅读】

    整体流程:
        1. 加载并预处理图像 (1024分辨率)
        2. VGGT 推理: 提取相机参数 + 深度图 (518分辨率)
        3. 深度图反投影为 3D 点云 (unproject_depth_map_to_point_map)
        4. (可选 BA) 使用追踪点进行 bundle adjustment 优化
        5. 保存为 COLMAP 格式 (cameras.bin, images.bin, points3D.bin)

    输出目录结构:
        SCENE_DIR/
        ├── images/
        └── sparse/
            ├── cameras.bin
            ├── images.bin
            └── points3D.bin

    Args:
        args: 命令行参数
    """
    # 打印配置
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # 加载图像 (1024分辨率用于后续处理, 但 VGGT 内部固定用 518)
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # 【Step 1: VGGT 推理】得到相机参数和深度图 (518分辨率)
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)

    # 【Step 2: 深度图反投影为 3D 点云】
    # extrinsic (S,3,4) + intrinsic (S,3,3) + depth_map (S,1,H,W) => points_3d (S,H,W,3)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.use_ba:
        # 【带 Bundle Adjustment 的模式】
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            # 【追踪预测】使用 VGGSfM tracker (比 VGGT tracker 更高效)
            # 输出:
            #   - pred_tracks: 追踪点坐标 (用于 BA)
            #   - pred_vis_scores: 可见性分数
            #   - points_3d: 优化后的 3D 点
            #   - points_rgb: 点云颜色
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # 将内参从 518 分辨率缩放到 1024 分辨率
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # 【Step 3: 构建 COLMAP reconstruction 对象】
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # 【Step 4: Bundle Adjustment 优化】
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        # 【纯前馈模式 (无 BA)】
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # 最多保留 10万 个 3D 点
        shared_camera = False  # 前馈模式不支持共享相机
        camera_type = "PINHOLE"  # 前馈模式只支持 PINHOLE 相机

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        # 获取点云颜色 (从输入图像插值)
        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # 像素坐标网格: (S, H, W, 3), 包含 (x, y, frame_idx)
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        # 按置信度过滤点云
        conf_mask = depth_conf >= conf_thres_value
        # 随机采样, 限制最多 10万 点
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
