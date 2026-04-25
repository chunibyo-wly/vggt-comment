# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) is a feed-forward neural network that predicts camera parameters, depth maps, point maps, and 3D point tracks from one or more input images. It is a research codebase from VGG Oxford and Meta AI.

## Setup

Prerequisites: PyTorch 2.3.1 + CUDA. Install PyTorch first, then the project.

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

For demo dependencies (Gradio, Viser, COLMAP export):

```bash
pip install -r requirements_demo.txt
```

## Running Demos

All demos assume images are in `<scene_dir>/images/`.

- **Gradio web UI**: `python demo_gradio.py`
- **Viser 3D viewer**: `python demo_viser.py --image_folder path/to/images/folder`
- **Export COLMAP**: `python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba` (add `--use_ba` for bundle adjustment)

## Training

Training code lives in `training/` and uses Hydra for configuration and PyTorch DDP.

1. Install VGGT as a package: `pip install -e .`
2. Edit `training/config/default.yaml` to set dataset paths (`CO3D_DIR`, `CO3D_ANNOTATION_DIR`) and `resume_checkpoint_path`.
3. Run with DDP:

```bash
cd training
torchrun --nproc_per_node=4 launch.py
```

Key training config files:
- `training/config/default.yaml` — main config (model, loss, optimizer, data, distributed)
- `training/config/default_dataset.yaml` — dataset and augmentation defaults

The default config fine-tunes the pretrained model with the camera and depth heads active, the aggregator frozen, and gradient accumulation of 2 steps.

## Architecture

### Model Structure

`vggt/models/vggt.py` is the top-level `VGGT` model. It wraps:

- **`Aggregator`** (`vggt/models/aggregator.py`) — the transformer backbone.
- **`CameraHead`** (`vggt/heads/camera_head.py`) — predicts camera pose encoding.
- **`DPTHead`** (`vggt/heads/dpt_head.py`) — shared DPT-style decoder used for both depth and point map heads.
- **`TrackHead`** (`vggt/heads/track_head.py`) — predicts 3D point tracks.

### Aggregator: Alternating Attention

The `Aggregator` is the core of the model. It processes input frames with alternating **frame-wise self-attention** and **global cross-frame attention** over 24 blocks (default). Key details:

- Patch embedding uses a DINOv2 ViT-Large by default (`patch_embed="dinov2_vitl14_reg"` in `aggregator.py`).
- Two special token types are prepended to patch tokens: **camera tokens** (shape `1, 2, 1, C`) and **register tokens** (`1, 2, num_register_tokens, C`). The first position is used for the first frame; the second for all remaining frames. See `slice_expand_and_flatten()` in `aggregator.py`.
- **Rotary Position Embedding (RoPE)** for 2D spatial positions is applied to patch tokens but not to special tokens.
- During training, gradient checkpointing is enabled on the frame/global blocks to reduce memory.
- The aggregator returns a list of intermediate outputs (concatenated frame + global features) and `patch_start_idx` (index where patch tokens begin after special tokens).

### Heads

- **CameraHead** uses iterative refinement over camera tokens. It outputs pose encodings that are converted to extrinsic/intrinsic matrices via `vggt/utils/pose_enc.py:pose_encoding_to_extri_intri()`.
- **DPTHead** follows the DPT (Vision Transformers for Dense Prediction) architecture. It fuses multi-scale transformer features and upsamples to dense predictions. Used for both depth (`output_dim=2`) and point maps (`output_dim=4`).
- **TrackHead** tracks query points across frames.

### Utilities

Important utility modules:

- `vggt/utils/load_fn.py` — image loading and preprocessing (`load_and_preprocess_images`, `load_and_preprocess_images_square`).
- `vggt/utils/pose_enc.py` — pose encoding/decoding between model outputs and OpenCV camera-from-world extrinsics + intrinsics.
- `vggt/utils/geometry.py` — `unproject_depth_map_to_point_map()`, `closed_form_inverse_se3()`, and other 3D geometry helpers.
- `visual_util.py` — converts predictions to GLB meshes for visualization.

## Conventions

- Images are expected in range `[0, 1]` with shape `[B, S, 3, H, W]` or `[S, 3, H, W]`.
- Camera poses follow the **OpenCV camera-from-world convention**.
- The default image size is `518x518` (patch size 14).
- `bfloat16` autocast is used on Ampere+ GPUs; fallback to `float16` on older GPUs.
- Model weights are loaded from Hugging Face (`facebook/VGGT-1B`). The checkpoint auto-downloads on first use.