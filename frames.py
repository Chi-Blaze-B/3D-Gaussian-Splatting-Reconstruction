"""
从视频中提取帧，支持三种策略：

- 均匀采样（默认）：等间隔抽取目标数量的帧。对缓慢移动的手机视频最稳定，
  可复现，无失败模式。
- 单阶段智能采样：用清晰度和帧间变化做门控，剔除几乎无特征（白墙、模糊）
  或几乎无变化（相机静止）的帧。
- 两阶段智能采样：在单阶段门控基础上，用粗位姿的视差做加权，让视角覆盖好、
  基线足够的帧占更大比例。

核心设计：门控与加权分离
    门控是硬剔除，剔除不适合参与姿态估计的帧；加权是软分配，对通过门控的帧
    按视差等信号分配帧数。若门控过严（通过帧数低于 min_frames），回退到均匀
    采样，避免返回过少帧导致 SfM 无法工作。

光流在本项目中的作用是门控，而非加权。手机视频中 EIS 数字防抖和自动曝光
会污染光流幅度，使其不适合作为"信息量"的度量；但光流可以可靠地识别"帧间
几乎无变化"的帧（相机静止、场景无运动），用于剔除。

确定性：所有采样与分配均不使用随机数，同一视频每次运行得到相同帧集合。
"""

import os
import tempfile
import shutil
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------- 配置 ----------
DEFAULT_OPTICAL_FLOW_METHOD = "farneback"  # 或 "lk"
LK_WINDOW_SIZE = (15, 15)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

# 每次打分的最大采样帧数，控制打分阶段耗时上界
MAX_SCORE_SAMPLES = 100

# 门控分位数：剔除分数最低的对应比例
SHARPNESS_QUANTILE = 0.15   # 清晰度最低的 15%
FLOW_QUANTILE = 0.15        # 帧间变化最低的 15%

# 门控绝对下限（防止整个视频都很平淡时分位数失去意义）
SHARPNESS_ABS_MIN = 20.0    # 拉普拉斯方差下限
FLOW_ABS_MIN = 0.15         # 每帧平均光流幅度下限（像素）

# 两阶段粗提取帧数
COARSE_FRAMES = 40


# ---------- 公共入口 ----------
def extract_frames(
    video_path: str,
    output_dir: str,
    *,
    fps: float = 15.0,
    scale: float = 0.5,
    min_frames: int = 30,
    max_frames: int = 200,
    smart_sampling: bool = False,
    two_stage: bool = False,
    poses_output_dir: Optional[str] = None,
    optical_flow_method: str = DEFAULT_OPTICAL_FLOW_METHOD,
    feature_type: str = "orb",
) -> List[str]:
    """从视频提取帧。

    参数：
        video_path: 输入视频路径。
        output_dir: 输出 PNG 帧的目录。
        fps: 目标采样率（仅用于计算目标帧数）。
        scale: 缩放比例，0 < scale <= 1。
        min_frames, max_frames: 最终帧数上下界。
        smart_sampling: 是否启用单阶段智能采样（门控）。
        two_stage: 是否启用两阶段智能采样（粗位姿 + 视差加权）。
        poses_output_dir: 粗位姿中间产物输出目录（两阶段用，None 时使用临时目录）。
        optical_flow_method: 'farneback' 或 'lk'。
        feature_type: 'orb' 或 'sift'，用于两阶段粗位姿估计。

    返回：
        保存的帧图像绝对路径列表。
    """
    if two_stage:
        return _two_stage_extract(
            video_path, output_dir, fps, scale,
            min_frames, max_frames, poses_output_dir,
            optical_flow_method, feature_type,
        )
    if smart_sampling:
        return _smart_extract(
            video_path, output_dir, fps, scale,
            min_frames, max_frames, optical_flow_method,
        )
    return _uniform_extract(video_path, output_dir, fps, scale, min_frames, max_frames)


# ---------- 均匀采样 ----------
def _uniform_extract(
    video_path: str,
    output_dir: str,
    fps: float,
    scale: float,
    min_frames: int,
    max_frames: int,
) -> List[str]:
    """均匀采样：在时间轴上等间隔抽取目标数量的帧。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = _compute_resized_size(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            scale,
        )
        num_frames = _target_frame_count(total, orig_fps, fps, min_frames, max_frames)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        os.makedirs(output_dir, exist_ok=True)
        paths, _ = _extract_indices(cap, indices, output_dir, w, h)
        return paths
    finally:
        cap.release()


# ---------- 单阶段智能采样 ----------
def _smart_extract(
    video_path: str,
    output_dir: str,
    fps: float,
    scale: float,
    min_frames: int,
    max_frames: int,
    flow_method: str,
) -> List[str]:
    """单阶段智能采样：清晰度 + 光流门控，通过门控的帧均匀分配。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = _compute_resized_size(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            scale,
        )
        return _smart_extract_from_cap(
            cap, output_dir, total, orig_fps, fps,
            min_frames, max_frames, w, h, flow_method,
        )
    finally:
        cap.release()


def _smart_extract_from_cap(
    cap: cv2.VideoCapture,
    output_dir: str,
    total: int,
    orig_fps: float,
    fps: float,
    min_frames: int,
    max_frames: int,
    w: int,
    h: int,
    flow_method: str,
) -> List[str]:
    """在已打开的 cap 上执行单阶段智能采样。

    步骤：
    1. 在约 MAX_SCORE_SAMPLES 个采样帧上计算清晰度和光流。
    2. 插值扩展到每一帧。
    3. 门控：剔除清晰度或光流低于阈值的帧。
    4. 通过门控的帧按均匀权重，确定性选择最终帧。
    5. 若通过帧数低于 min_frames，回退到均匀采样。
    """
    sample_indices = _make_sample_indices(total)
    sample_step = _sample_step(total)
    sharpness, flow = _compute_gating_scores(
        cap, sample_indices, sample_step, flow_method,
    )
    sharp_full = _interp_to_full(sample_indices, sharpness, total)
    flow_full = _interp_to_full(sample_indices, flow, total)

    num_frames = _target_frame_count(total, orig_fps, fps, min_frames, max_frames)
    valid = _gating_mask(sharp_full, flow_full)
    n_valid = int(valid.sum())

    if n_valid < min_frames:
        # 门控过严：视频整体质量差，回退到均匀采样保证帧数
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        target = min(num_frames, n_valid)
        weights = valid.astype(np.float64)
        indices = _deterministic_select(weights, target)

    os.makedirs(output_dir, exist_ok=True)
    paths, _ = _extract_indices(cap, indices, output_dir, w, h)
    return paths


# ---------- 两阶段智能采样 ----------
def _two_stage_extract(
    video_path: str,
    output_dir: str,
    fps: float,
    scale: float,
    min_frames: int,
    max_frames: int,
    poses_output_dir: Optional[str],
    flow_method: str,
    feature_type: str,
) -> List[str]:
    """两阶段智能采样。

    阶段 1：均匀抽取 COARSE_FRAMES 帧，估计粗略相机位姿。
    阶段 2：在全部帧上计算清晰度、光流、视差；清晰度 + 光流门控，
            视差加权，确定性选择最终帧。若粗位姿估计失败，回退到单阶段。
    """
    from poses import estimate_poses  # 延迟导入，避免无两阶段需求时的依赖

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = _compute_resized_size(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        scale,
    )

    coarse_dir = tempfile.mkdtemp(prefix="coarse_")
    pose_dir = poses_output_dir or tempfile.mkdtemp(prefix="poses_coarse_")
    os.makedirs(pose_dir, exist_ok=True)

    try:
        # ---------- 阶段 1：粗提取 + 粗位姿 ----------
        coarse_raw = np.linspace(0, total - 1, min(COARSE_FRAMES, total), dtype=int)
        coarse_paths, coarse_indices = _extract_indices(cap, coarse_raw, coarse_dir, w, h)
        if len(coarse_paths) < 2:
            print("  [警告] 粗提取帧数不足，回退到单阶段智能采样。")
            return _smart_extract_from_cap(
                cap, output_dir, total, orig_fps, fps,
                min_frames, max_frames, w, h, flow_method,
            )

        try:
            _, coarse_poses, _ = estimate_poses(
                coarse_paths, min_inliers=10, feature_type=feature_type,
            )
        except Exception as e:
            print(f"  [警告] 粗位姿估计失败: {e}。回退到单阶段智能采样。")
            return _smart_extract_from_cap(
                cap, output_dir, total, orig_fps, fps,
                min_frames, max_frames, w, h, flow_method,
            )

        # ---------- 阶段 2：打分 + 门控 + 视差加权 ----------
        sample_indices = _make_sample_indices(total)
        sample_step = _sample_step(total)
        sharpness, flow = _compute_gating_scores(
            cap, sample_indices, sample_step, flow_method,
        )
        sharp_full = _interp_to_full(sample_indices, sharpness, total)
        flow_full = _interp_to_full(sample_indices, flow, total)
        parallax_full = _compute_parallax_scores(total, coarse_indices, coarse_poses)

        num_frames = _target_frame_count(total, orig_fps, fps, min_frames, max_frames)
        valid = _gating_mask(sharp_full, flow_full)
        n_valid = int(valid.sum())

        if n_valid < min_frames:
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
        else:
            weights = parallax_full * valid
            if weights.sum() <= 1e-9:
                # 视差全为零（位姿退化）：退化为均匀分配
                weights = valid.astype(np.float64)
            target = min(num_frames, n_valid)
            indices = _deterministic_select(weights, target)

        os.makedirs(output_dir, exist_ok=True)
        paths, _ = _extract_indices(cap, indices, output_dir, w, h)
        return paths
    finally:
        cap.release()
        shutil.rmtree(coarse_dir, ignore_errors=True)
        if poses_output_dir is None:
            shutil.rmtree(pose_dir, ignore_errors=True)


# ---------- 打分：清晰度 + 光流 ----------
def _compute_gating_scores(
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    sample_step: int,
    flow_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """一次遍历同时计算每个采样帧的清晰度和光流分数。

    清晰度：拉普拉斯方差。反映图像是否模糊以及是否几乎无内容（白墙）。
    光流：与前一采样帧的平均位移幅度，再除以采样间隔得到"每帧光流"。
          反映帧间是否有真实变化，用于剔除相机静止或重复帧。

    返回：
        sharpness: 长度等于 indices 的清晰度数组
        flow: 长度等于 indices 的每帧光流数组。第一帧没有前驱，用第二帧的值填充。
    """
    n = len(indices)
    sharpness = np.zeros(n, dtype=np.float64)
    flow = np.zeros(n, dtype=np.float64)
    if n == 0:
        return sharpness, flow

    method = flow_method.lower()
    prev_gray = None
    prev_idx = None

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            prev_gray = None
            prev_idx = None
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 清晰度：拉普拉斯方差
        sharpness[i] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 光流：与前一采样帧比较
        if prev_gray is not None:
            if method == "lk":
                mag = _lk_flow_magnitude(prev_gray, gray)
            else:
                f = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, **FARNEBACK_PARAMS,
                )
                mag = float(np.mean(np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)))
            # 归一化到"每帧"量级，避免采样间隔放大光流
            gap = max(1, int(idx) - int(prev_idx)) if prev_idx is not None else 1
            flow[i] = mag / gap
        prev_gray = gray
        prev_idx = int(idx)

    # 第一帧无前驱：用第二帧填充，避免固定门控误杀
    if n >= 2 and flow[0] == 0.0 and flow[1] > 0.0:
        flow[0] = flow[1]
    return sharpness, flow


def _gating_mask(sharp_full: np.ndarray, flow_full: np.ndarray) -> np.ndarray:
    """根据清晰度和光流计算门控掩码。

    阈值 = max(绝对下限, 分位数)。绝对下限防止视频整体平淡时分位数失去意义；
    分位数让阈值自适应视频自身的分数分布。
    """
    sharp_thresh = max(SHARPNESS_ABS_MIN, float(np.quantile(sharp_full, SHARPNESS_QUANTILE)))
    flow_thresh = max(FLOW_ABS_MIN, float(np.quantile(flow_full, FLOW_QUANTILE)))
    return (sharp_full >= sharp_thresh) & (flow_full >= flow_thresh)


def _lk_flow_magnitude(prev: np.ndarray, curr: np.ndarray) -> float:
    """Lucas-Kanade 稀疏光流：在网格点上计算平均位移幅度。"""
    h, w = prev.shape
    step = 16
    y_coords = np.arange(0, h, step, dtype=np.float32)
    x_coords = np.arange(0, w, step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev, curr, pts, None,
        winSize=LK_WINDOW_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA,
    )
    if status is None:
        return 0.0
    valid = status.ravel() == 1
    if not valid.any():
        return 0.0
    disp = (next_pts[valid] - pts[valid]).reshape(-1, 2)
    return float(np.mean(np.sqrt(np.sum(disp ** 2, axis=1))))


# ---------- 视差打分 ----------
def _compute_parallax_scores(
    total: int,
    coarse_indices: np.ndarray,
    coarse_poses: List,
) -> np.ndarray:
    """基于粗位姿的相邻基线计算每帧视差分数。

    思路：相邻有效粗帧之间的基线长度（平移）加上旋转角，作为该时间区间内
    的视差代理。基线越大，三角化越稳定，这些帧越值得保留。用 np.interp 把
    粗帧位置的分数插值到全部帧。
    """
    valid_pairs = [(i, p) for i, p in enumerate(coarse_poses) if p is not None]
    if len(valid_pairs) < 2:
        return np.zeros(total, dtype=np.float64)

    coarse_scores = np.zeros(len(coarse_indices), dtype=np.float64)
    for k in range(1, len(valid_pairs)):
        i1, p1 = valid_pairs[k - 1]
        i2, p2 = valid_pairs[k]
        t1 = np.asarray(p1.t).flatten()
        t2 = np.asarray(p2.t).flatten()
        baseline = float(np.linalg.norm(t2 - t1))
        R_rel = np.asarray(p2.R) @ np.asarray(p1.R).T
        angle = _rotation_angle(R_rel)
        coarse_scores[i2] = baseline + 0.1 * np.radians(angle)

    parallax = np.interp(np.arange(total), coarse_indices, coarse_scores)
    parallax = _gaussian_smooth(parallax, sigma=2.0)
    if parallax.max() > 1e-9:
        parallax = parallax / parallax.max()
    return parallax


def _rotation_angle(R: np.ndarray) -> float:
    """从旋转矩阵提取旋转角（弧度）。"""
    rv, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(rv))


# ---------- 采样与分配辅助 ----------
def _make_sample_indices(total: int) -> np.ndarray:
    """构造打分用的采样索引，最多 MAX_SCORE_SAMPLES 个，并保证覆盖最后一帧。"""
    step = _sample_step(total)
    indices = np.arange(0, total, step, dtype=int)
    if indices.size == 0:
        indices = np.array([0], dtype=int)
    if indices[-1] != total - 1:
        indices = np.append(indices, total - 1)
    return indices


def _sample_step(total: int) -> int:
    """采样步长：使采样数不超过 MAX_SCORE_SAMPLES。"""
    return max(1, total // MAX_SCORE_SAMPLES)


def _interp_to_full(sample_indices: np.ndarray, sample_scores: np.ndarray, total: int) -> np.ndarray:
    """把采样分数线性插值扩展到全部帧。"""
    sample_indices = np.asarray(sample_indices)
    sample_scores = np.asarray(sample_scores, dtype=np.float64)
    if sample_indices.size == 0:
        return np.zeros(total, dtype=np.float64)
    if sample_indices.size == 1:
        return np.full(total, float(sample_scores[0]), dtype=np.float64)
    return np.interp(np.arange(total), sample_indices, sample_scores)


def _gaussian_smooth(scores: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """对 1D 分数做高斯平滑，避免孤立峰值主导分配。"""
    if len(scores) < 3 or sigma <= 0:
        return scores
    size = int(4 * sigma + 1) | 1  # 保证奇数
    kernel = cv2.getGaussianKernel(size, sigma).reshape(-1)
    pad = size // 2
    padded = np.pad(scores, pad, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(scores)]


def _deterministic_select(weights: np.ndarray, num_frames: int) -> np.ndarray:
    """按权重确定性选择 num_frames 个下标。

    使用分层采样：把累积分布 [0,1] 均分为 num_frames 段，每段取中点反查对应
    下标。不使用随机数，同一输入每次得到相同结果，便于复现。
    """
    n = len(weights)
    num_frames = min(num_frames, n)
    if num_frames >= n:
        return np.arange(n)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = w.sum()
    if total <= 0.0:
        return np.linspace(0, n - 1, num_frames, dtype=int)
    cum = np.cumsum(w) / total
    targets = (np.arange(num_frames) + 0.5) / num_frames
    indices = np.searchsorted(cum, targets)
    indices = np.clip(indices, 0, n - 1)
    return np.unique(indices)


def _target_frame_count(
    total: int,
    orig_fps: float,
    target_fps: float,
    min_frames: int,
    max_frames: int,
) -> int:
    """计算目标帧数：按目标采样率换算后夹到 [min_frames, max_frames]，再夹到 total。"""
    raw = int(total * target_fps / orig_fps)
    n = min(max(raw, min_frames), max_frames)
    return max(1, min(n, total))


def _compute_resized_size(orig_w: int, orig_h: int, scale: float) -> Tuple[int, int]:
    """计算缩放后的尺寸，宽高取偶数（避免后续处理中奇数尺寸带来的边界问题）。"""
    w = int(orig_w * scale)
    h = int(orig_h * scale)
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1
    return max(w, 2), max(h, 2)


def _extract_indices(
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    output_dir: str,
    w: int,
    h: int,
) -> Tuple[List[str], np.ndarray]:
    """按给定索引抽取帧并保存为 PNG。

    返回：
        paths: 成功保存的帧路径列表（绝对路径）。
        used_indices: 与 paths 一一对应的原始帧索引数组。若某帧读取失败会被跳过，
                      因此 used_indices 的长度可能小于 indices。
    """
    paths: List[str] = []
    used: List[int] = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        fname = f"frame_{i:04d}.png"
        fpath = os.path.join(output_dir, fname)
        if not cv2.imwrite(fpath, resized):
            raise OSError(f"无法写入帧图像: {fpath}")
        paths.append(os.path.abspath(fpath))
        used.append(int(idx))
    return paths, np.asarray(used, dtype=int)