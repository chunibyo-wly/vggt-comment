# points_3d 与 extract_keypoints、Tracker 的关系

> 一句话总结: `extract_keypoints` 只负责在 2D 图像上找点，`points_3d` 负责给这些 2D 点赋予 3D 坐标并做质量筛选，而 tracker 只追踪 2D 坐标，完全不知道 3D 的存在。

三者是**串行协作**关系，而非 tracker 内部使用了 3D 信息。

---

## 一、三者的职责边界

| 组件 | 输入 | 输出 | 职责 |
|------|------|------|------|
| `extract_keypoints` | 单帧 RGB 图像 | 2D 关键点坐标 `(1, N, 2)` | 在图像纹理丰富处检测特征点（ALIKED / SuperPoint / SIFT） |
| `points_3d` + `conf` | 2D 关键点坐标 + 深度点云 + 置信度 | 筛选后的 2D 点 + 对应 3D 坐标 + 置信度 | **筛选**：去掉低置信度区域的关键点；**关联**：给每个 2D 点赋予 3D 坐标 |
| `tracker` (VGGSfM) | 2D 关键点坐标 + 多帧图像/特征图 | 每帧 2D 追踪轨迹 `(S, N, 2)` + 可见性 `(S, N)` | 基于 correlation 的纯 2D 追踪，建立跨帧像素对应关系 |

---

## 二、完整流程（代码层面的时序）

整个流程发生在 `_forward_on_query()` 中（`track_predict.py:151`）。

### Step 1: `extract_keypoints` 提出候选 2D 点

```python
query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
# query_points: (1, N, 2) — N 个 2D 候选点，坐标为浮点数
```

这一步完全基于图像局部纹理（角点、边缘等），**不感知任何 3D 信息**。例如可能提出 2048 个候选点，遍布整幅图像。

### Step 2: `points_3d` + `conf` 做筛选和关联

```python
if (conf is not None) and (points_3d is not None):
    scale = conf.shape[-1] / width
    # 将图像坐标缩放到 points_3d/conf 的低分辨率
    query_points_scaled = (query_points.squeeze(0) * scale).round().long()

    # 【采样】在 conf 和 points_3d 的对应位置取值
    pred_conf = conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
    pred_point_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]

    # 【过滤】只保留 conf > 1.2 的高质量点
    valid_mask = pred_conf > 1.2
    query_points = query_points[:, valid_mask]      # 2D 坐标筛选
    pred_point_3d = pred_point_3d[valid_mask]       # 保留对应 3D 坐标
    pred_conf = pred_conf[valid_mask]               # 保留置信度
```

这里的关键操作：

1. **坐标映射**：`query_points` 是图像分辨率（如 518x518），而 `points_3d` 和 `conf` 通常分辨率更低（如模型输出的 1/14 patch 分辨率），所以要先 `scale` 映射。
2. **一一对应**：`pred_point_3d[i]` 就是 `query_points[0, i]` 这个像素位置反投影得到的 3D 坐标。
3. **启发式过滤**：`conf > 1.2` 是因为 confidence head 使用 `expp1` 激活，大于 1.2 表示模型对该点的深度估计比较有把握。只在高置信度区域提取追踪点，可以显著减少 tracker 的噪声输入。

### Step 3: tracker 建立纯 2D 追踪

```python
pred_track, pred_vis, _ = predict_tracks_in_chunks(
    tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking
)
# pred_track: (1, S, N, 2) — 每个筛选后的 2D 点在各帧的新坐标
```

tracker 接收的输入只有：
- `query_points`：筛选后的 2D 坐标（已和 3D 解耦，只剩纯像素坐标）
- `images_feed` / `fmaps_feed`：多帧图像/特征图

tracker 内部做的是 correlation-based 迭代追踪（见 `base_track_predictor.py`），**完全在 2D 像素空间操作**，不涉及任何 3D 几何。

---

## 三、为什么这样设计？

### 3.1 Tracker 不做 3D，因为 3D 坐标本身不需要"追踪"

一个 3D landmark 的坐标是**固定的**（在世界坐标系中）。需要追踪的只是它在不同帧图像上的**投影位置**。所以：

- 3D 坐标只需在 query 帧上采样一次 → `pred_point_3d`
- 2D 坐标需要在所有帧上追踪 → `pred_track`

### 3.2 `points_3d` 的筛选作用至关重要

VGGT 前馈输出的深度图/点云在**低纹理区域、边缘、天空**等位置的置信度很低。如果让 tracker 在这些区域提取关键点：

- 关键点可能落在错误的几何位置上
- 追踪时容易跟丢或漂移
- 最终 BA 优化时引入离群点（outliers）

用 `conf > 1.2` 过滤后，tracker 只在**几何可靠区域**工作，大幅提升追踪质量。

### 3.3 如果没有 `points_3d`，tracker 还能工作吗？

可以。`predict_tracks()` 中 `conf` 和 `points_3d` 都是 `Optional`：

```python
def predict_tracks(images, conf=None, points_3d=None, ...):
```

如果不提供，`Step 2` 的筛选直接跳过，`extract_keypoints` 提出的所有点都会进入 tracker。这在以下场景有用：
- 你不关心 3D 坐标，只需要 2D tracks
- 没有 depth/point map 可用（如纯 2D 视频）

---

## 四、最终输出的关联关系

`_forward_on_query` 返回 5 个值，它们之间的索引对应关系是：

```
第 i 个 track:
    pred_track[:, i, :]  →  该点在各帧的 2D 像素坐标 (S, 2)
    pred_vis[:, i]       →  该点在各帧是否可见 (S,)
    pred_conf[i]         →  该点在 query 帧的置信度 (来自 conf 采样)
    pred_point_3d[i]     →  该点的 3D 世界坐标 (3,)  ← 来自 points_3d 采样
    pred_color[i]        →  该点的 RGB 颜色 (3,)
```

`pred_point_3d[i]` 和 `pred_track[:, i, :]` 通过**索引 `i`** 隐式关联：
- `pred_point_3d[i]` 是该 3D 点的固定坐标
- `pred_track[:, i, :]` 是该 3D 点在各帧图像上的投影轨迹

在 `demo_colmap.py` 中，这些对应关系被直接送入 COLMAP 做 Bundle Adjustment，将多帧 2D 观测关联到同一个 3D point 上。
