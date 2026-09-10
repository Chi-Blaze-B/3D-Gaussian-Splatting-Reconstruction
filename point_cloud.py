"""
Point cloud → Gaussian initialization for 3D Gaussian Splatting.

职责拆分为两个函数：
- sample_point_colors: IO + 多视角颜色采样（投影 → 可见性 → 像素取值 → 平均）
- initialize_gaussians: 纯数值构造高斯参数（离群点剔除、位置微扰、
  kNN 估尺度、opacity/SH/rotation 初始化）

拆分后颜色采样可复用调用方已有的帧内存缓存（LazyFrames），
而高斯构造不依赖图像 IO，可独立单测。
"""

from typing import Callable, Optional, Tuple

import cv2
import numpy as np

# 官方 3DGS SH 常数：DC 系数 = (RGB - 0.5) / SH_C0；eval_sh 求值时补回 +0.5
SH_C0 = 0.28209479177387814

# SH degree 3 → (3+1)^2 = 16 个基函数
SH_NUM_BASES = 16


def migrate_legacy_scales(params: dict) -> dict:
    """兼容旧 gaussian_params.npz 缓存的尺度语义。

    旧缓存把 LOG 尺度存在键 "scales"，initialize_from_dict 再取一次 log
    → 双重取 log，全部高斯 σ 塌缩为 1e-6。新缓存存线性尺度并带
    scale_domain='linear' 标记。本函数把无标记的旧缓存（log 值）转回线性。
    """
    params = dict(params)
    dom = params.get("scale_domain")
    is_linear = dom is not None and np.asarray(dom).item() == "linear"
    if not is_linear and "scales" in params:
        params["scales"] = np.exp(params["scales"].astype(np.float64)).astype(np.float32)
    params["scale_domain"] = np.array("linear")
    return params


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

    只做 IO 与采样：投影 → 可见性判断 → 像素取值 → 跨帧平均。
    不涉及高斯参数构造，也不做离群点剔除（由 initialize_gaussians 负责）。

    参数：
        sparse_points: (N, 3) float32 稀疏点云（原始，未剔除离群点）。
        poses: CameraPose 列表，长度 = 帧数，None 表示该帧位姿缺失。
        frame_loader: 任意可索引对象。LazyFrames 实例命中内存缓存；
                      list[str] 走 cv2.imread；list[ndarray] 直接使用。
                      ndarray 元素 dtype=uint8 按 [0,255] 归一化，
                      dtype=float 时按已是 [0,1] 处理。
        intrinsics: 含 .K 属性的对象。
        progress_cb: 可选进度回调，签名 (done, total)。

    返回：
        colors: (N, 3) float32，每点被观测到的平均 RGB；
                未被任何帧观测到的点填可见点的中位数颜色。
        counts: (N,) int32，每点被多少帧观测到。
    """
    N = int(sparse_points.shape[0])
    colors = np.zeros((N, 3), dtype=np.float32)
    counts = np.zeros(N, dtype=np.int32)
    if N == 0:
        return colors, counts

    valid_indices = [j for j, p in enumerate(poses) if p is not None]
    n_valid = len(valid_indices)
    if n_valid == 0:
        return colors, counts

    # 预构建每帧的 (3,4) 投影矩阵 P = K @ [R | t]
    proj_matrices = []
    for j in valid_indices:
        pose = poses[j]
        R = np.asarray(pose.R).reshape(3, 3)
        t = np.asarray(pose.t).reshape(3, 1)
        proj_matrices.append(intrinsics.K @ np.hstack([R, t]))

    Xh = np.vstack([sparse_points.T, np.ones(N, dtype=np.float32)])  # (4, N)

    for fi, P in enumerate(proj_matrices):
        raw = frame_loader[valid_indices[fi]]
        if isinstance(raw, str):
            img_bgr = cv2.imread(raw)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            arr = np.asarray(raw)
            if arr.dtype == np.uint8:
                img_rgb = arr.astype(np.float32) / 255.0
            else:
                img_rgb = arr.astype(np.float32)

        h, w = img_rgb.shape[:2]
        proj = P @ Xh
        z = proj[2]
        u = proj[0] / z
        v = proj[1] / z

        valid = (z > 0.01) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if np.any(valid):
            u_int = np.clip(np.round(u[valid]).astype(np.int32), 0, w - 1)
            v_int = np.clip(np.round(v[valid]).astype(np.int32), 0, h - 1)
            point_idx = np.where(valid)[0]
            colors[point_idx] += img_rgb[v_int, u_int]
            counts[point_idx] += 1

        if progress_cb is not None:
            progress_cb(fi + 1, n_valid)
        elif (fi + 1) % 20 == 0:
            print(f"  [POINT CLOUD] Processed {fi+1}/{n_valid} frames", flush=True)

    # 跨帧平均；未观测点填中位数颜色
    visible = counts > 0
    safe_counts = np.maximum(counts, 1).astype(np.float32)
    colors /= safe_counts[:, np.newaxis]
    if np.any(~visible):
        if np.any(visible):
            median_color = np.median(colors[visible], axis=0)
        else:
            median_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        colors[~visible] = median_color
    print(f"  Points seen by ≥1 frame: {int(visible.sum())}/{N}")
    return colors, counts


# ---------- 高斯参数构造（纯数值侧） ----------
def initialize_gaussians(
    sparse_points: np.ndarray,
    colors: np.ndarray,
    counts: np.ndarray,
    *,
    noise_std: float = 0.01,
) -> dict:
    """从稀疏点 + 采样颜色构造高斯参数。纯数值，不读图。

    步骤：
    1. 自适应离群点剔除（阈值 = clip(场景跨度 × 5, 50, 500)）；
    2. 位置微扰；
    3. kNN 局部密度估尺度；
    4. opacity/SH/rotation 初始化。

    参数：
        sparse_points: (N, 3) float32，与 colors/counts 行对齐。
        colors: (N, 3) float32 每点平均 RGB。
        counts: (N,) int 每点观测数（仅用于日志）。
        noise_std: 位置微扰标准差。

    返回：
        dict，键为 positions/scales/opacities/sh_coeffs/rotations/
        colors/counts/scale_domain，与 Gaussian3D.initialize_from_dict 兼容。
    """
    N_in = int(sparse_points.shape[0])
    if N_in == 0:
        return _empty_result()

    # --- 自适应离群点剔除：按 keep_mask 同步裁剪点、颜色、计数 ---
    keep_mask, scene_span, threshold = _outlier_keep_mask(sparse_points)
    n_kept = int(keep_mask.sum())
    if n_kept < N_in:
        print(f"  [OUTLIER] Removed {N_in - n_kept} points "
              f"(scene span: {scene_span:.1f}, threshold: {threshold:.1f})")
    sparse_points = sparse_points[keep_mask]
    colors = colors[keep_mask]
    counts = counts[keep_mask]
    N = n_kept
    if N == 0:
        print("  [WARN] All SfM points were outliers — cannot initialize Gaussians.")
        return _empty_result()

    # --- 位置微扰 ---
    positions = sparse_points + np.random.randn(N, 3).astype(np.float32) * noise_std

    # --- kNN 局部密度估尺度 ---
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        k = min(10, N)
        if k <= 1:
            scales = np.full((N, 1), 0.1, dtype=np.float32)
        else:
            dists, _ = tree.query(positions, k=k)
            if dists.shape[1] <= 1:
                scales = np.full((N, 1), 0.1, dtype=np.float32)
            else:
                avg_dist = np.mean(dists[:, 1:], axis=1)
                scales = np.clip(avg_dist, 0.01, 1.0).reshape(-1, 1)
    except ImportError:
        print("  [WARN] scipy not available — using constant scale 0.1")
        scales = np.full((N, 1), 0.1, dtype=np.float32)

    # --- opacity / SH / rotation ---
    opacities = np.full(N, 0.5, dtype=np.float32)
    sh_coeffs = np.zeros((N, SH_NUM_BASES, 3), dtype=np.float32)
    # 官方 SH 约定：DC 系数 = (RGB - 0.5) / C0（eval_sh 求值补回 +0.5）
    sh_coeffs[:, 0, :] = (colors - 0.5) / SH_C0
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1)).astype(np.float32)

    result = {
        "positions": positions.astype(np.float32),
        # 存线性尺度（initialize_from_dict 内部再取 log 得到 log σ）
        "scales": scales.astype(np.float32),
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
        "rotations": rotations,
        "colors": colors.astype(np.float32),
        "counts": counts,
        "scale_domain": np.array("linear"),
    }
    print(f"Initialized {N} Gaussians, color range: "
          f"[{colors.min():.2f}, {colors.max():.2f}]")
    return result


# ---------- 内部辅助 ----------
def _outlier_keep_mask(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """按点云跨度计算自适应阈值，返回 (keep_mask, scene_span, threshold)。

    阈值 = clip(场景跨度 × 5, 50, 500)，以点云中位数为中心，
    距离超过阈值的点视为离群点。空输入或退化点云（跨度 < 1e-6）
    返回全 True。scene_span 与 threshold 供调用方打印日志。
    """
    N = len(points)
    if N == 0:
        return np.zeros(0, dtype=bool), 0.0, 0.0
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    scene_span = float(np.linalg.norm(bbox_max - bbox_min))
    if scene_span < 1e-6:
        return np.ones(N, dtype=bool), scene_span, 0.0
    median = np.median(points, axis=0)
    dists = np.linalg.norm(points - median, axis=1)
    threshold = float(np.clip(scene_span * 5.0, 50.0, 500.0))
    return dists < threshold, scene_span, threshold


def _empty_result() -> dict:
    """空输入 / 全部离群点时的返回，字段与正常路径一致。"""
    return {
        "positions": np.empty((0, 3), dtype=np.float32),
        "scales": np.empty((0, 1), dtype=np.float32),
        "opacities": np.empty((0,), dtype=np.float32),
        "sh_coeffs": np.empty((0, SH_NUM_BASES, 3), dtype=np.float32),
        "rotations": np.empty((0, 4), dtype=np.float32),
        "colors": np.empty((0, 3), dtype=np.float32),
        "counts": np.empty((0,), dtype=np.int32),
        "scale_domain": np.array("linear"),
    }