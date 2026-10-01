"""
SfM 稀疏点云 → 初始高斯，供 3D Gaussian Splatting 使用。

模块职责：
- sample_point_colors: IO + 多视角颜色采样（投影 → 可见性 → 像素取值 → 平均）。
- initialize_gaussians: 纯数值构造高斯参数（离群点剔除、位置微扰、
  kNN 估尺度、opacity/SH/rotation 初始化），不依赖图像 IO。

颜色约定：
  对外统一使用 uint8 [0, 255]，与 poses.SfMResult.rgb 天然对齐。
  - sample_point_colors 返回 uint8；
  - initialize_gaussians 接受 uint8（推荐）或 float32 [0,1]，内部转 [0,1]
    算 SH DC，输出 "colors" 字段统一回写为 uint8。
"""

import logging
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 官方 3DGS SH 常数：DC 系数 = (RGB - 0.5) / SH_C0；eval_sh 求值时补回 +0.5
SH_C0 = 0.28209479177387814

# SH degree 3 → (3+1)^2 = 16 个基函数
SH_NUM_BASES = 16


# ---------- 颜色采样（IO 侧） ----------
def sample_point_colors(
    sparse_points: np.ndarray,
    poses: list,
    frame_loader,
    intrinsics,
    *,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """从多视角采样每个 3D 点的平均 RGB。

    只负责 IO 与采样：投影 → 可见性判断 → 像素取值 → 跨帧平均。
    不做高斯参数构造，也不剔除离群点（交由 initialize_gaussians 处理）。

    参数：
        sparse_points: (N, 3) float32 稀疏点云（未剔除离群点）。
        poses: CameraPose 列表，长度等于帧数；None 表示该帧位姿缺失。
        frame_loader: 可索引对象。LazyFrames 命中内存缓存；
                      list[str] 走 cv2.imread；list[ndarray] 直接使用；
                      ndarray dtype=uint8 按 [0,255]，dtype=float 视为 [0,1]。
        intrinsics: 含 .K 属性的对象。
        progress_cb: 可选进度回调，签名 (done, total)。

    返回：
        colors: (N, 3) uint8 [0,255]，每点观测到的平均 RGB；
                未被任何帧观测到的点填可见点的中位数颜色。
        counts: (N,) int32，每点被多少帧观测到。
    """
    n_points = int(sparse_points.shape[0])
    # 累加用 float32 [0,1]，最终再转 uint8，避免逐帧量化误差累积
    colors_acc = np.zeros((n_points, 3), dtype=np.float32)
    counts = np.zeros(n_points, dtype=np.int32)
    if n_points == 0:
        return np.zeros((0, 3), dtype=np.uint8), counts

    valid_indices = [j for j, p in enumerate(poses) if p is not None]
    n_valid = len(valid_indices)
    if n_valid == 0:
        return np.zeros((n_points, 3), dtype=np.uint8), counts

    # 预构建每帧的 (3, 4) 投影矩阵 P = K @ [R | t]
    proj_matrices = []
    for j in valid_indices:
        pose = poses[j]
        R = np.asarray(pose.R).reshape(3, 3)
        t = np.asarray(pose.t).reshape(3, 1)
        proj_matrices.append(intrinsics.K @ np.hstack([R, t]))

    # 齐次坐标 (4, N)
    points_h = np.vstack([sparse_points.T, np.ones(n_points, dtype=np.float32)])

    for fi, P in enumerate(proj_matrices):
        raw = frame_loader[valid_indices[fi]]
        if isinstance(raw, str):
            img_bgr = cv2.imread(raw)
            if img_bgr is None:
                logger.warning("读取图像失败，跳过该帧：%s", raw)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            arr = np.asarray(raw)
            if arr.dtype == np.uint8:
                img_rgb = arr.astype(np.float32) / 255.0
            else:
                img_rgb = arr.astype(np.float32)

        h, w = img_rgb.shape[:2]
        proj = P @ points_h
        z = proj[2]
        u = proj[0] / z
        v = proj[1] / z

        # 只保留相机前方且落在图像范围内的点
        valid = (z > 0.01) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if np.any(valid):
            u_int = np.clip(np.round(u[valid]).astype(np.int32), 0, w - 1)
            v_int = np.clip(np.round(v[valid]).astype(np.int32), 0, h - 1)
            point_idx = np.where(valid)[0]
            colors_acc[point_idx] += img_rgb[v_int, u_int]
            counts[point_idx] += 1

        if progress_cb is not None:
            progress_cb(fi + 1, n_valid)
        elif (fi + 1) % 20 == 0:
            logger.info("颜色采样进度：%d/%d 帧", fi + 1, n_valid)

    # 跨帧平均；未观测的点填可见点的中位数颜色
    visible = counts > 0
    safe_counts = np.maximum(counts, 1).astype(np.float32)
    colors_acc /= safe_counts[:, np.newaxis]
    if np.any(~visible):
        if np.any(visible):
            median_color = np.median(colors_acc[visible], axis=0)
        else:
            median_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        colors_acc[~visible] = median_color

    colors = np.clip(np.round(colors_acc * 255.0), 0, 255).astype(np.uint8)
    logger.info("至少被一帧观测到的点数：%d/%d", int(visible.sum()), n_points)
    return colors, counts


# ---------- 高斯参数构造（纯数值侧） ----------
def initialize_gaussians(
    sparse_points: np.ndarray,
    colors: np.ndarray,
    counts: np.ndarray,
    *,
    noise_std: float = 0.01,
) -> dict:
    """从稀疏点 + 采样颜色构造高斯参数（纯数值，不读图）。

    步骤：
    1. 颜色归一化（uint8 → float32 [0,1]，用于 SH DC）；
    2. 自适应离群点剔除（阈值 = clip(场景跨度 × 5, 50, 500)）；
    3. 位置微扰；
    4. kNN 局部密度估尺度；
    5. opacity / SH / rotation 初始化。

    参数：
        sparse_points: (N, 3) float32，与 colors/counts 行对齐。
        colors: (N, 3) uint8 [0,255]（推荐）或 float32 [0,1]；float 且最大值
                大于 1.5 时视为误传 [0,255]，自动 /255 并告警。
        counts: (N,) int 每点观测数（仅用于日志）。
        noise_std: 位置微扰标准差。

    返回：
        dict（positions/scales/opacities/sh_coeffs/rotations/colors/counts/
        scale_domain），与 Gaussian3D.initialize_from_dict 兼容。
        scales 存线性尺度（initialize_from_dict 内部再取 log 得到 log σ）。
        colors 字段统一为 uint8 [0,255]。
    """
    n_in = int(sparse_points.shape[0])
    if n_in == 0:
        return _empty_result()

    # 颜色归一到 float32 [0,1]，容忍 uint8 与 float 两种输入
    colors_arr = np.asarray(colors)
    if colors_arr.dtype == np.uint8:
        colors_f = colors_arr.astype(np.float32) / 255.0
    else:
        colors_f = colors_arr.astype(np.float32)
        if colors_f.size > 0 and float(colors_f.max()) > 1.5:
            logger.warning(
                "initialize_gaussians: colors 收到 float 值域 [%.2f, %.2f]，"
                "疑似 [0,255]，已自动 /255 归一化到 [0,1]。",
                float(colors_f.min()), float(colors_f.max()),
            )
            colors_f = colors_f / 255.0

    # 离群点剔除：按 keep_mask 同步裁剪点、颜色、计数
    keep_mask, scene_span, threshold = _outlier_keep_mask(sparse_points)
    n_kept = int(keep_mask.sum())
    if n_kept < n_in:
        logger.info(
            "剔除离群点 %d 个（场景跨度 %.1f，阈值 %.1f）",
            n_in - n_kept, scene_span, threshold,
        )
    sparse_points = sparse_points[keep_mask]
    colors_f = colors_f[keep_mask]
    counts = counts[keep_mask]
    n = n_kept
    if n == 0:
        logger.warning("所有 SfM 点均被判为离群点，无法初始化高斯。")
        return _empty_result()

    # 位置微扰，避免大量点完全共面
    positions = sparse_points + np.random.randn(n, 3).astype(np.float32) * noise_std

    # kNN 局部密度估尺度
    scales = _estimate_scales_knn(positions)

    # opacity / SH / rotation
    opacities = np.full(n, 0.5, dtype=np.float32)
    sh_coeffs = np.zeros((n, SH_NUM_BASES, 3), dtype=np.float32)
    # 官方 SH 约定：DC 系数 = (RGB - 0.5) / C0（eval_sh 求值补回 +0.5）
    sh_coeffs[:, 0, :] = (colors_f - 0.5) / SH_C0
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)

    # 对外 colors 统一回写 uint8，与 SfMResult.rgb 语义一致
    colors_out = np.clip(np.round(colors_f * 255.0), 0, 255).astype(np.uint8)

    result = {
        "positions": positions.astype(np.float32),
        "scales": scales.astype(np.float32),
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
        "rotations": rotations,
        "colors": colors_out,
        "counts": counts,
        "scale_domain": np.array("linear"),
    }
    logger.info(
        "已初始化 %d 个高斯，颜色范围 [%d, %d]（uint8）",
        n, int(colors_out.min()), int(colors_out.max()),
    )
    return result


# ---------- 内部辅助 ----------
def _outlier_keep_mask(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """按点云跨度计算自适应阈值，返回 (keep_mask, scene_span, threshold)。

    阈值 = clip(场景跨度 × 5, 50, 500)，以点云中位数为中心，
    距离超阈值的点视为离群点。空输入或退化点云返回全 True。
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0, 0.0
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    scene_span = float(np.linalg.norm(bbox_max - bbox_min))
    if scene_span < 1e-6:
        return np.ones(n, dtype=bool), scene_span, 0.0
    median = np.median(points, axis=0)
    dists = np.linalg.norm(points - median, axis=1)
    threshold = float(np.clip(scene_span * 5.0, 50.0, 500.0))
    return dists < threshold, scene_span, threshold


def _estimate_scales_knn(positions: np.ndarray) -> np.ndarray:
    """用 kNN 平均邻距估每点尺度，返回 (N, 1) float32。

    邻距被裁剪到 [0.01, 1.0]，避免初始 σ 过小或过大。
    scipy 不可用时退化为常数 0.1。
    """
    n = positions.shape[0]
    if n <= 1:
        return np.full((n, 1), 0.1, dtype=np.float32)

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        logger.warning("scipy 不可用，尺度退化为常数 0.1")
        return np.full((n, 1), 0.1, dtype=np.float32)

    tree = cKDTree(positions)
    k = min(10, n)
    dists, _ = tree.query(positions, k=k)
    if dists.ndim < 2 or dists.shape[1] <= 1:
        return np.full((n, 1), 0.1, dtype=np.float32)
    avg_dist = np.mean(dists[:, 1:], axis=1)
    return np.clip(avg_dist, 0.01, 1.0).reshape(-1, 1).astype(np.float32)


def _empty_result() -> dict:
    """空输入 / 全部为离群点时的返回，字段与正常路径保持一致。"""
    return {
        "positions": np.empty((0, 3), dtype=np.float32),
        "scales": np.empty((0, 1), dtype=np.float32),
        "opacities": np.empty((0,), dtype=np.float32),
        "sh_coeffs": np.empty((0, SH_NUM_BASES, 3), dtype=np.float32),
        "rotations": np.empty((0, 4), dtype=np.float32),
        "colors": np.empty((0, 3), dtype=np.uint8),
        "counts": np.empty((0,), dtype=np.int32),
        "scale_domain": np.array("linear"),
    }