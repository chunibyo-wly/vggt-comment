# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import trimesh
import gradio as gr
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation
import copy
import cv2
import os
import requests


def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
) -> trimesh.Scene:
    """
    【可视化输出转换 - 重点阅读】
    将 VGGT 的预测结果转换为 3D GLB 场景 (用于 Gradio 展示)。

    输入 predictions 字典必须包含:
        - world_points 或 world_points_from_depth: 3D 点坐标 (S, H, W, 3)
        - world_points_conf 或 depth_conf: 置信度 (S, H, W)
        - images: 输入图像 (S, H, W, 3)
        - extrinsic: 相机外参矩阵 (S, 3, 4)

    两种点云来源:
        1. "Predicted Pointmap": 使用 point_head 直接输出的 world_points
        2. "Depth-based": 使用 depth + camera 反投影的 world_points_from_depth
           【通常更准确, 推荐优先使用】

    处理流程:
        1. 根据 prediction_mode 选择点云来源
        2. (可选) 天空分割过滤
        3. 按置信度阈值过滤低质量点
        4. (可选) 过滤黑/白背景
        5. 构建 trimesh 点云和相机可视化
        6. 返回 trimesh.Scene (可导出为 GLB)

    Args:
        predictions (dict): VGGT 预测结果字典
        conf_thres (float): 置信度过滤百分比 (默认 50.0, 即过滤最低的 50%)
        filter_by_frames (str): 帧过滤 (默认 "all")
        mask_black_bg (bool): 是否过滤黑色背景
        mask_white_bg (bool): 是否过滤白色背景
        show_cam (bool): 是否显示相机可视化
        mask_sky (bool): 是否进行天空分割过滤
        target_dir (str): 中间文件输出目录
        prediction_mode (str): 点云来源选择 ("Predicted Pointmap" 或 "Depth-based")

    Returns:
        trimesh.Scene: 包含点云和相机的 3D 场景
    """
    # 输入校验: predictions 必须是字典
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # 默认置信度阈值: 10.0 (百分比形式, 过滤掉最低的 10%)
    if conf_thres is None:
        conf_thres = 10.0

    print("Building GLB scene")

    # 解析 filter_by_frames 参数, 提取帧索引 (格式如 "0:frame_name")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    # 【Step 1: 选择点云来源】
    # prediction_mode 决定使用 point_head 直接输出还是 depth+camera 反投影
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            # 使用 point_head 直接输出的 world_points (S, H, W, 3)
            pred_world_points = predictions["world_points"]
            # 置信度: world_points_conf, 若不存在则默认全 1
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            # 回退到 depth-based 点云
            print("Warning: world_points not found in predictions, falling back to depth-based points")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap and Camera Branch")
        # 【推荐】使用 depth + camera 反投影的点云, 通常更准确
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # 从 predictions 提取输入图像和相机外参
    images = predictions["images"]              # (S, H, W, 3) 或 (S, 3, H, W)
    camera_matrices = predictions["extrinsic"]  # (S, 3, 4) OpenCV camera-from-world

    # 【Step 2: (可选) 天空分割过滤】
    # 使用 ONNX 天空分割模型, 将天空区域的置信度置 0
    if mask_sky:
        if target_dir is not None:
            import onnxruntime

            skyseg_session = None
            target_dir_images = target_dir + "/images"
            image_list = sorted(os.listdir(target_dir_images))
            sky_mask_list = []

            # 获取与置信度图匹配的 H, W (S 帧, H 高, W 宽)
            S, H, W = (
                pred_world_points_conf.shape
                if hasattr(pred_world_points_conf, "shape")
                else (len(images), images.shape[1], images.shape[2])
            )

            # 下载天空分割模型 (若不存在)
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
                )

            # 逐帧生成/加载天空掩码
            for i, image_name in enumerate(image_list):
                image_filepath = os.path.join(target_dir_images, image_name)
                mask_filepath = os.path.join(target_dir, "sky_masks", image_name)

                if os.path.exists(mask_filepath):
                    # 复用已保存的掩码
                    sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                else:
                    # 用 ONNX 模型推理生成掩码
                    if skyseg_session is None:
                        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
                    sky_mask = segment_sky(image_filepath, skyseg_session, mask_filepath)

                # 若尺寸不匹配, resize 到 HxW
                if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                    sky_mask = cv2.resize(sky_mask, (W, H))

                sky_mask_list.append(sky_mask)

            # 将 sky mask list 转为 numpy 数组, 形状 SxHxW
            sky_mask_array = np.array(sky_mask_list)

            # 二值化: >0.1 视为非天空区域 (255 保留, 0 过滤)
            sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
            # 将天空区域置信度置 0
            pred_world_points_conf = pred_world_points_conf * sky_mask_binary

    # 【Step 3: (可选) 只保留指定帧】
    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]         # (1, H, W, 3)
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]  # (1, H, W)
        images = images[selected_frame_idx][None]                                  # (1, H, W, 3)
        camera_matrices = camera_matrices[selected_frame_idx][None]                # (1, 3, 4)

    # 【Step 4: 展平点云和颜色】
    # 将 (S, H, W, 3) 展平为 (S*H*W, 3), 每个像素对应一个 3D 顶点
    vertices_3d = pred_world_points.reshape(-1, 3)

    # 处理图像格式: NCHW (S,3,H,W) -> NHWC (S,H,W,3)
    if images.ndim == 4 and images.shape[1] == 3:
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    # 展平颜色并转为 uint8 (0-255)
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    # 【Step 5: 按置信度过滤点云】
    # 展平置信度: (S, H, W) -> (S*H*W,)
    conf = pred_world_points_conf.reshape(-1)

    # 将百分比阈值转换为实际置信度值
    # 例: conf_thres=50 表示过滤掉置信度最低的 50% 的点
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)

    # 构建掩码: 保留置信度 >= 阈值 且 > 1e-5 的点
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    # (可选) 过滤黑色背景: RGB 之和 < 16 视为黑色
    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    # (可选) 过滤白色背景: RGB 均 > 240 视为白色
    if mask_white_bg:
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    # 应用掩码过滤
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    # 【Step 6: 计算场景尺度】
    # 用于相机可视化的大小缩放
    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        # 若点云为空, 使用默认值避免崩溃
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # 用 5%-95% 分位点计算包围盒对角线长度作为场景尺度
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    # 为每帧相机分配不同颜色 (彩虹色映射)
    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # 【Step 7: 构建 trimesh 3D 场景】
    scene_3d = trimesh.Scene()

    # 将点云添加到场景
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    # 【Step 8: (可选) 添加相机可视化】
    # 将 (S, 3, 4) 外参补齐为 (S, 4, 4) 齐次矩阵
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1  # 齐次坐标最后一行

    if show_cam:
        for i in range(num_cameras):
            # 外参是世界到相机的变换, 取逆得到相机到世界
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)

            # 为当前相机分配颜色
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            # 将相机视锥体添加到场景
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    # 【Step 9: 场景对齐】
    # 以第一帧相机为参考, 对齐整个场景到合适的观察角度
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    print("GLB Scene built")
    return scene_3d


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray) -> trimesh.Scene:
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    # Apply transformation
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
