"""
Point cloud initialization for 3D Gaussian Splatting (optimized, SH degree 3).

Takes sparse 3D points and camera poses from SfM, densifies via
multi-view color averaging, and initializes per-point Gaussians with
appropriate covariance, opacity, and spherical-harmonic colors (degree 3).

Updated: Adaptive outlier threshold based on point cloud scale.
"""

import cv2
import numpy as np


def initialize_gaussians(
    sparse_points: np.ndarray,
    poses: list,
    frame_paths: list[str],
    intrinsics,
    *,
    num_samples_per_point: int = 5,
    noise_std: float = 0.01,
) -> dict:
    """Initialize 3D Gaussian parameters from sparse SfM points.

    For each 3D point, sample colors from multiple views that see it,
    compute a mean color, and assign initial Gaussian properties.

    Parameters
    ----------
    sparse_points : (N, 3) array of 3D coordinates.
    poses : list of CameraPose objects.
    frame_paths : sorted list of frame image paths.
    intrinsics : object with ``K`` attribute — (3,3) intrinsic matrix.
    num_samples_per_point : how many views to sample color from.
    noise_std : std-dev for initial position jitter.

    Returns
    -------
    dict with keys: positions, scales, opacities, sh_coeffs, rotations
        sh_coeffs shape is (N, 16, 3) for SH degree 3.
    """
    N = int(sparse_points.shape[0])
    SH_NUM_BASES = 16  # degree 3 => (3+1)^2 = 16

    if N == 0:
        return {
            "positions": np.empty((0, 3), dtype=np.float32),
            "scales": np.empty((0, 1), dtype=np.float32),
            "opacities": np.empty((0,), dtype=np.float32),
            "sh_coeffs": np.empty((0, SH_NUM_BASES, 3), dtype=np.float32),
            "rotations": np.empty((0, 4), dtype=np.float32),
            "colors": np.empty((0, 3), dtype=np.float32),
            "counts": np.empty((0,), dtype=int),
        }

    if not frame_paths or not poses:
        return {k: np.empty((0,)) for k in [
            "positions", "scales", "opacities", "sh_coeffs", "rotations",
        ]}

    # --- Adaptive outlier removal ---
    def _remove_outliers(points: np.ndarray) -> tuple[np.ndarray, int]:
        """Remove points that are extreme outliers relative to the point cloud scale.

        Uses a threshold proportional to the point cloud span to adapt to
        different scene scales (e.g., close-up vs. large outdoor).
        """
        Np = len(points)
        if Np == 0:
            return points, 0

        # Compute the bounding box diagonal as a measure of scene scale
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        scene_span = np.linalg.norm(bbox_max - bbox_min)

        if scene_span < 1e-6:
            return points, 0

        # Median center
        median = np.median(points, axis=0)
        dists = np.linalg.norm(points - median, axis=1)

        # Adaptive threshold: 5x the scene span, with a floor of 50 and ceiling of 500
        threshold = np.clip(scene_span * 5.0, 50.0, 500.0)

        normal_mask = dists < threshold
        n_removed = int(Np - int(normal_mask.sum()))

        if n_removed > 0:
            print(f"  [OUTLIER] Removed {n_removed} points "
                  f"(scene span: {scene_span:.1f}, threshold: {threshold:.1f})")
            return points[normal_mask], n_removed

        return points, 0

    sparse_points, n_removed = _remove_outliers(sparse_points)
    N = int(sparse_points.shape[0])

    if N == 0:
        print("  [WARN] All SfM points were outliers — cannot initialize Gaussians.")
        return {
            "positions": np.empty((0, 3), dtype=np.float32),
            "scales": np.empty((0, 1), dtype=np.float32),
            "opacities": np.empty((0,), dtype=np.float32),
            "sh_coeffs": np.empty((0, SH_NUM_BASES, 3), dtype=np.float32),
            "rotations": np.empty((0, 4), dtype=np.float32),
            "colors": np.empty((0, 3), dtype=np.float32),
            "counts": np.empty((0,), dtype=int),
        }

    # --- Build projection matrices once ---
    valid_indices = [j for j, p in enumerate(poses) if p is not None]
    proj_matrices = []
    for j in valid_indices:
        pose = poses[j]
        # ----- 修复开始 -----
        # 强制将 R 和 t 转换为正确的二维形状，避免拼接时维度不一致
        R = np.asarray(pose.R).reshape(3, 3)
        t = np.asarray(pose.t).reshape(3, 1)
        P = intrinsics.K @ np.hstack([R, t])  # (3, 4)
        # ----- 修复结束 -----
        proj_matrices.append(P)
    n_valid_poses = len(proj_matrices)

    # --- Multi-view color sampling (vectorized per frame) ---
    positions = sparse_points + np.random.randn(N, 3).astype(np.float32) * noise_std
    colors = np.zeros((N, 3), dtype=np.float32)
    counts = np.zeros(N, dtype=int)

    Xh = np.vstack([sparse_points.T, np.ones(N, dtype=np.float32)])  # (4, N)

    for fi, P in enumerate(proj_matrices):
        img_bgr = cv2.imread(frame_paths[valid_indices[fi]])
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        del img_bgr

        h, w = img_rgb.shape[:2]

        proj = P @ Xh  # (3, N)
        z = proj[2]
        u = proj[0] / z
        v = proj[1] / z

        valid = (z > 0.01) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(valid):
            del img_rgb
            continue

        u_int = np.clip(np.round(u[valid]).astype(np.int32), 0, w - 1)
        v_int = np.clip(np.round(v[valid]).astype(np.int32), 0, h - 1)
        point_idx = np.where(valid)[0]

        colors[point_idx] += img_rgb[v_int, u_int]
        counts[point_idx] += 1

        del img_rgb

        if (fi + 1) % 20 == 0:
            print(f"  [POINT CLOUD] Processed {fi+1}/{n_valid_poses} frames", flush=True)

    # Average colors across views
    safe_counts = np.maximum(counts, 1, dtype=np.float32)
    colors /= safe_counts[:, np.newaxis]

    visible_mask = counts > 0
    print(f"  Points seen by ≥1 frame: {visible_mask.sum()}/{N}")

    if np.any(~visible_mask):
        median_color = np.median(colors[visible_mask], axis=0) if visible_mask.sum() > 0 else np.array([0.5, 0.5, 0.5])
        colors[~visible_mask] = median_color

    # --- Estimate scale from local point density ---
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

    # --- Initialize parameters ---
    opacities = np.full(N, 0.5, dtype=np.float32)
    sh_coeffs = np.zeros((N, SH_NUM_BASES, 3), dtype=np.float32)
    sh_coeffs[:, 0, :] = colors
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1)).astype(np.float32)
    log_scales = np.log(scales.astype(np.float32))

    result = {
        "positions": positions.astype(np.float32),
        "scales": log_scales,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
        "rotations": rotations,
        "colors": colors,
        "counts": counts,
    }

    print(f"Initialized {N} Gaussians, mean color range: [{colors.min():.2f}, {colors.max():.2f}]")
    return result