# predict_tracks 详解

> 文件位置: `vggt/dependency/track_predict.py`  
> 依赖工具: `vggt/dependency/vggsfm_utils.py`

---

## 一、解决了什么问题

### 核心问题：VGGT 前馈输出缺乏跨帧 2D-3D 对应关系

VGGT 的前馈输出（每帧独立预测深度图 + 相机参数）存在根本性缺陷：

- 第 1 帧的 `(100, 200)` 像素和第 2 帧的 `(100, 200)` 像素**不一定对应同一个 3D landmark**
- 直接把所有深度图反投影的点云堆在一起，会产生**大量重复点**
- 例如 10 帧 × 26 万点/帧 = **260 万个点**，其中同一物理位置被描述多次

### predict_tracks 的作用

建立**显式的跨帧 2D 对应关系**，为 Bundle Adjustment (BA) 提供约束：

| 输出 | 形状 | 含义 |
|------|------|------|
| `pred_tracks` | `(S, P, 2)` | 每个 3D 点在各帧的像素坐标 |
| `pred_vis_scores` | `(S, P)` | 该点在该帧是否可见 |
| `pred_confs` | `(P,)` | 追踪点置信度 |
| `pred_points_3d` | `(P, 3)` | 每个 track 对应的 3D 坐标 |
| `pred_colors` | `(P, 3)` | 每个 track 的颜色 |

有了这些对应关系，就可以把多帧对同一 3D 点的观测关联起来，去重后进行 BA 优化。

### 应用场景

仅在 `demo_colmap.py --use_ba` 中使用：

```
VGGT 前馈输出 (depth + camera)
    ↓
predict_tracks()  ← 建立 2D-3D 对应
    ↓
batch_np_matrix_to_pycolmap()  ← 构建 COLMAP reconstruction
    ↓
pycolmap.bundle_adjustment()  ← 全局优化
    ↓
cameras.bin + images.bin + points3D.bin
```

---

## 二、使用了什么方法

### 2.1 整体方法概述

基于 **VGGSfM Tracker**（而非 VGGT 自带的 TrackHead）进行跨帧特征点追踪：

- **关键点检测**: ALIKED + SuperPoint（可选 +SIFT）
- **特征匹配与追踪**: VGGSfM Tracker（correlation-based，类似 RAFT）
- **帧选择策略**: DINOv2 特征 + Farthest Point Sampling (FPS)
- **补充策略**: 对低可见度帧重新作为 query 进行追踪

### 2.2 为什么选择 VGGSfM Tracker 而不是 VGGT TrackHead

代码注释说明 (`demo_colmap.py:196-200`)：

```python
# Using VGGSfM tracker instead of VGGT tracker for efficiency
# VGGT tracker requires multiple backbone runs to query different frames
# (this is a problem caused by the training process)
# Will be fixed in VGGT v2
```

VGGT TrackHead 的问题：
- 每换一个 query 帧，都需要重新跑一遍 backbone
- VGGSfM Tracker 预先提取所有帧的特征图，换 query 帧只需重新追踪，效率高很多

---

## 三、内部做了哪些操作

### 3.1 主流程 `predict_tracks()`

```
输入: images [S, 3, H, W], conf [S, 1, H, W], points_3d [S, H, W, 3]
    │
    ├─ Step 1: 初始化 VGGSfM Tracker
    │
    ├─ Step 2: DINO + FPS 选择 query 帧 (默认 5 帧)
    │
    ├─ Step 3: 初始化关键点提取器 (ALIKED + SuperPoint)
    │
    ├─ Step 4: Tracker 预提取所有帧的特征图
    │
    ├─ Step 5: 逐 query 帧追踪 → _forward_on_query()
    │   │
    │   └─ 每个 query 帧独立提取关键点，追踪到所有帧
    │
    ├─ Step 6: (可选) 补充低可见度帧 → _augment_non_visible_frames()
    │   │
    │   └─ 对可见点 < 500 的帧，重新作为 query 追踪
    │
    └─ Step 7: 合并所有 query 帧的结果，沿点维度拼接
```

### 3.2 Query 帧选择 `generate_rank_by_dino()`

**目标**: 选出最具代表性的帧作为 query，保证关键点能被尽量多的帧看到。

**算法**:

1. **DINOv2 特征提取**: 用 `dinov2_vitb14_reg` 提取每帧的 CLS token
2. **相似度矩阵**: 计算帧间余弦相似度 `(S, S)`
3. **找最"常见"帧**: `similarity_sum = similarity_matrix.sum(dim=1)`，取最大值的帧作为起始点。该帧与其他帧最相似，说明视角居中、重叠区域大
4. **Farthest Point Sampling (FPS)**: 从起始帧出发，每次选距离已选集合最远的帧，保证视角多样性

**为什么用 FPS**：
- 避免选出的 query 帧扎堆在同一视角
- 不同视角提取的关键点覆盖不同区域，提高整体追踪覆盖率

### 3.3 单帧追踪处理 `_forward_on_query()`

```
输入: query_index, images, conf, points_3d, fmaps_for_tracker
    │
    ├─ Step 1: 在 query 帧上用 ALIKED + SuperPoint 提取关键点
    │           (最多 max_query_pts=2048 个)
    │
    ├─ Step 2: 提取关键点处的颜色 → pred_color
    │
    ├─ Step 3: (可选) 用 conf 和 points_3d 筛选高质量点
    │           │
    │           ├─ 将 query 点坐标缩放到 conf 分辨率
    │           ├─ 采样 conf 和 points_3d 对应位置的值
    │           └─ 启发式过滤: 只保留 conf > 1.2 的点 (至少保留 512 个)
    │
    ├─ Step 4: 帧顺序重排 (swap query_index ↔ 0)
    │           Tracker 内部假设第 0 帧是 query 帧
    │
    ├─ Step 5: 分块追踪 (防止 GPU OOM)
    │           │
    │           ├─ all_points_num = S × N
    │           ├─ 若 > max_points_num (163840)，拆分为多个 chunk
    │           └─ 分别送入 tracker，结果拼接
    │
    └─ Step 6: 恢复原始帧顺序，转为 numpy
```

**输出**:
- `pred_track`: `(N, P, 2)` — 每帧每个追踪点的像素坐标
- `pred_vis`: `(N, P)` — 可见性分数
- `pred_conf`: `(P,)` — 置信度
- `pred_point_3d`: `(P, 3)` — 3D 坐标
- `pred_color`: `(P, 3)` — 颜色

### 3.4 补充低可见度帧 `_augment_non_visible_frames()`

**问题**: 初始 query 帧选择（如第 0, 3, 5 帧）可能无法覆盖所有帧。某些帧因遮挡严重或视角差异大，看到的追踪点很少。

**解决策略**:

```
循环:
    ├─ 统计每帧可见追踪点数量 (vis_score > 0.1)
    ├─ 找出可见点 < 500 的帧
    │
    ├─ 正常情况: 每次只处理第一个难帧
    │   └─ 将其作为新的 query 帧重新追踪
    │   └─ 原理: 帧 B 自己做 query 时，会提取适合从自身视角追踪的特征点
    │
    └─ 同一帧连续两次失败:
        └─ 换更强的提取器组合 (sp + sift + aliked)
        └─ 一次性处理所有剩余难帧（最终尝试）
```

**为什么有效**：
- 帧 A 作为 query 时提取的关键点，在帧 B 上可能不可见
- 但帧 B 自己作为 query 时，会提取适合从自身视角追踪的特征点
- 这样每个"难帧"都有机会用自己最稳定的特征点建立追踪

### 3.5 关键点提取器 `initialize_feature_extractors()`

支持的方法（用 `+` 组合）：

| 方法 | 特点 | 适用场景 |
|------|------|---------|
| **ALIKED** | 速度快，适合纹理丰富区域 | 默认首选 |
| **SuperPoint (sp)** | 深度学习检测器，泛化性好 | 补充检测 |
| **SIFT** | 传统方法，对尺度/旋转鲁棒 | 最终尝试 |

组合策略：
- `"aliked+sp"`（默认）：先用 ALIKED，再用 SuperPoint 补充
- `"sp+sift+aliked"`（最终尝试）：三种都用，提取最多特征点

### 3.6 分块追踪 `predict_tracks_in_chunks()`

**目的**: 防止 GPU OOM

**原理**: Tracker 计算量 ∝ (帧数 × 关键点数)。当 `S × N > max_points_num` (默认 163840) 时，将 `query_points` 沿 N 维度拆分：

```python
# 例: S=10, N=2048, all_points_num=20480 (< 163840) → 不分块
# 例: S=50, N=4096, all_points_num=204800 (> 163840) → 拆成 2 个 chunk
num_splits = (all_points_num + max_points_num - 1) // max_points_num
```

---

## 四、关键设计要点总结

| 设计 | 目的 |
|------|------|
| DINO + FPS 选帧 | 保证 query 帧视角多样性，提高追踪覆盖率 |
| 多提取器组合 | ALIKED 快 + SP 准 + SIFT 鲁棒，互补检测 |
| conf > 1.2 过滤 | 只在 VGGT 高置信度区域提取关键点，减少噪声 |
| 帧顺序重排 | 复用 tracker（固定假设第 0 帧为 query） |
| 分块处理 | 控制 GPU 内存，支持大量点追踪 |
| 补充追踪 | 难帧用自己的关键点重新追踪，提高覆盖率 |
| 最终尝试换提取器 | 用最强组合 (sp+sift+aliked) 处理顽固难帧 |

---

## 五、与 VGGT TrackHead 的对比

| | `predict_tracks` (VGGSfM) | VGGT TrackHead |
|--|---------------------------|----------------|
| **使用场景** | `demo_colmap.py --use_ba` | `VGGT.forward(query_points=...)` |
| **效率** | 高（预提取特征，换 query 不重跑 backbone） | 低（每次换 query 需重跑 backbone） |
| **关键点来源** | ALIKED / SuperPoint / SIFT 检测 | 用户指定或均匀采样 |
| **追踪方式** | Correlation-based (RAFT 风格) | Transformer-based attention |
| **是否需要 3D 点** | 可选（用于筛选和初始化） | 不需要 |
| **输出** | tracks + vis + conf + 3D + color | tracks + vis + conf |
