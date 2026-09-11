"""
从视频中提取帧，支持三种策略：

- 均匀采样（默认）：等间隔抽取目标数量的帧。
- 单阶段智能采样：用清晰度和帧间变化做门控。
- 两阶段智能采样：单阶段门控 + 粗位姿视差加权。

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

# 打分时的缩放后短边长度，保证不同视频之间分数尺度可比
SCORE_MAX_SIDE = 480

# 门控分位数：剔除分数最低的对应比例
SHARPNESS_QUANTILE = 0.05
FLOW_QUANTILE = 0.05

# 门控相对下限：阈值不低于中位数的对应比例。
# 用相对下限代替固定绝对下限，避免固定常数在不同尺度/纹理的视频上失效
# （例如 4K 手机视频拉普拉斯方差中位数可能只有 3~5，固定阈值 20 会误杀 97% 帧）。
SHARPNESS_MEDIAN_RATIO = 0.08
FLOW_MEDIAN_RATIO = 0.08

# 时间覆盖兜底：每个时间分箱至少保留 min_per_bin 个通过帧
TIME_COVERAGE_BINS = 20
TIME_COVERAGE_MIN_PER_BIN = 12

# 最终选择的时间分层段数。各段均分名额，避免低权重段被完全跳过。
SELECT_BINS = 20

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
        poses_output_dir: 预留参数（当前未使用，仅为 API 兼容）。
        optical_flow_method: 'farneback' 或 'lk'。
        feature_type: 'orb' 或 'sift'，用于两阶段粗位姿估计。

    返回：
        保存的帧图像绝对路径列表。
    """
    if two_stage:
        return _two_stage_extract(
            video_path, output_dir, fps, scale,
            min_frames, max_frames,
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
    """单阶段智能采样：清晰度 + 光流门控，通过门控的帧按时间分层选择。"""
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
    4. 时间覆盖兜底：每个时间分箱至少保留少量帧。
    5. 时间分层选择：每段均分名额，段内按权重确定性选择。
    6. 若通过帧数低于 min_frames，回退到均匀采样。
    """
    sample_indices = _make_sample_indices(total)
    sharpness, flow = _compute_gating_scores(
        cap, sample_indices, flow_method,
    )
    sharp_full = _interp_to_full(sample_indices, sharpness, total)
    flow_full = _interp_to_full(sample_indices, flow, total)

    num_frames = _target_frame_count(total, orig_fps, fps, min_frames, max_frames)
    valid = _gating_mask(sharp_full, flow_full)
    valid = _enforce_time_coverage(valid, sharp_full, flow_full, total)
    n_valid = int(valid.sum())

    if n_valid < min_frames:
        # 门控过严：视频整体质量差，回退到均匀采样保证帧数
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        target = min(num_frames, n_valid)
        weights = valid.astype(np.float64)
        # 单阶段没有视差，权重全为 1，但依然走时间分层，保证时间均匀
        indices = _stratified_select(weights, valid, target, n_bins=SELECT_BINS)

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
    flow_method: str,
    feature_type: str,
) -> List[str]:
    """两阶段智能采样。

    阶段 1：均匀抽取 COARSE_FRAMES 帧，估计粗略相机位姿。
    阶段 2：在全部帧上计算清晰度、光流、视差；清晰度 + 光流门控，
            视差加权，时间分层选择最终帧。若粗位姿估计失败，回退到单阶段。
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

        # ---------- 阶段 2：打分 + 门控 + 视差加权 + 分层选择 ----------
        sample_indices = _make_sample_indices(total)
        sharpness, flow = _compute_gating_scores(
            cap, sample_indices, flow_method,
        )
        sharp_full = _interp_to_full(sample_indices, sharpness, total)
        flow_full = _interp_to_full(sample_indices, flow, total)
        parallax_full = _compute_parallax_scores(total, coarse_indices, coarse_poses)

        num_frames = _target_frame_count(total, orig_fps, fps, min_frames, max_frames)
        valid = _gating_mask(sharp_full, flow_full)
        valid = _enforce_time_coverage(valid, sharp_full, flow_full, total)
        n_valid = int(valid.sum())

        if n_valid < min_frames:
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
        else:
            # 视差加权：白墙等低视差帧保留基础权重，占比低但不会被完全跳过。
            # 时间分层选择再保证每段都有名额。
            weights = (0.3 + 0.7 * parallax_full) * valid
            if weights.sum() <= 1e-9:
                weights = valid.astype(np.float64)
            target = min(num_frames, n_valid)
            indices = _stratified_select(weights, valid, target, n_bins=SELECT_BINS)

        os.makedirs(output_dir, exist_ok=True)
        paths, _ = _extract_indices(cap, indices, output_dir, w, h)
        return paths
    finally:
        cap.release()
        shutil.rmtree(coarse_dir, ignore_errors=True)


# ---------- 时间分层选择 ----------
def _stratified_select(
    weights: np.ndarray,
    valid: np.ndarray,
    num_frames: int,
    n_bins: int = SELECT_BINS,
) -> np.ndarray:
    """按时间分层选择帧。

    把时间轴均分为 n_bins 段，名额尽量平均分配到每段（每段约 num_frames/n_bins）。
    段内按 weights 确定性选择。这样可以保证即使某段权重很低（白墙、静止），
    也能分到一定名额，不会被全局按权重选择时完全跳过。

    参数：
        weights: 长度等于总帧数的权重数组。非有效帧权重会被强制置 0。
        valid: 长度等于总帧数的布尔掩码，标出可选的帧。
        num_frames: 目标选择帧数。
        n_bins: 时间分层段数。若超过 num_frames 会被夹到 num_frames，
                避免出现某些段无名额。

    返回：
        排序后的选中帧索引数组。
    """
    n = len(weights)
    if n == 0 or num_frames <= 0:
        return np.array([], dtype=int)
    num_frames = min(num_frames, int(valid.sum()))
    if num_frames <= 0:
        return np.array([], dtype=int)

    # 段数不能超过目标帧数，否则每段名额不足，会有大量空段
    n_bins = max(1, min(n_bins, num_frames))

    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    per_bin = num_frames // n_bins
    extra = num_frames % n_bins

    selected: List[int] = []
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        k = per_bin + (1 if b < extra else 0)
        k = min(k, hi - lo)
        if k <= 0:
            continue

        seg_valid = valid[lo:hi]
        if not seg_valid.any():
            # 该段没有有效帧，跳过（真的没有可用素材）
            continue

        seg_w = weights[lo:hi].copy()
        seg_w[~seg_valid] = 0.0

        if seg_w.sum() <= 1e-9:
            # 段内权重全 0 但存在有效帧：段内均匀取 k 个
            valid_local = np.where(seg_valid)[0]
            take = min(k, len(valid_local))
            idx_local = np.linspace(0, len(valid_local) - 1, take).astype(int)
            seg_indices = valid_local[idx_local]
        else:
            seg_indices = _deterministic_select(seg_w, k)

        selected.extend((seg_indices + lo).tolist())

    return np.asarray(sorted(set(selected)), dtype=int)


# ---------- 打分：清晰度 + 光流 ----------
def _compute_gating_scores(
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    flow_method: str,
    score_max_side: int = SCORE_MAX_SIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """一次遍历同时计算每个采样帧的清晰度和光流分数。

    清晰度：拉普拉斯方差。反映图像是否模糊以及是否几乎无内容（白墙）。
    光流：与前一采样帧的平均位移幅度，再除以采样间隔得到"每帧光流"。
          反映帧间是否有真实变化，用于剔除相机静止或重复帧。

    注意：打分前先把帧缩放到短边 score_max_side（默认 480）。原始分辨率
    差异巨大（如 4K vs 720p）时，拉普拉斯方差与光流幅度会被分辨率主导，
    导致阈值不可比；统一缩放后分数尺度稳定，相对阈值才有意义。

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

        # 缩放到统一短边，保证不同视频之间的分数尺度可比
        h0, w0 = bgr.shape[:2]
        s = score_max_side / max(h0, w0)
        if s < 1.0:
            bgr = cv2.resize(
                bgr,
                (max(2, int(w0 * s)), max(2, int(h0 * s))),
                interpolation=cv2.INTER_AREA,
            )
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

    阈值 = max(分位数, 中位数 × 比例)。

    原来的实现用固定绝对下限（如 sharpness>=20、flow>=0.15），在低纹理
    或低分辨率视频上会把绝大多数帧误杀，导致通过门控的帧高度集中。
    改用中位数比例后，阈值随视频自身的分数尺度自适应，鲁棒性显著提升。
    """
    sharp_q = float(np.quantile(sharp_full, SHARPNESS_QUANTILE))
    flow_q = float(np.quantile(flow_full, FLOW_QUANTILE))
    sharp_med = float(np.median(sharp_full))
    flow_med = float(np.median(flow_full))

    sharp_thresh = max(sharp_q, SHARPNESS_MEDIAN_RATIO * sharp_med)
    flow_thresh = max(flow_q, FLOW_MEDIAN_RATIO * flow_med)
    return (sharp_full >= sharp_thresh) & (flow_full >= flow_thresh)


def _enforce_time_coverage(
    valid: np.ndarray,
    sharp_full: np.ndarray,
    flow_full: np.ndarray,
    total: int,
    n_bins: int = TIME_COVERAGE_BINS,
    min_per_bin: int = TIME_COVERAGE_MIN_PER_BIN,
) -> np.ndarray:
    """时间覆盖兜底：每个时间分箱至少保留 min_per_bin 个通过帧。

    即使门控阈值合理，若视频某一段整体平淡（相机静止、纹理少），那一段可能
    没有帧通过门控，最终选帧仍会挤在其他段，造成视角/时间分布不均。这里对
    分箱内通过帧不足的段，按 清晰度 × 光流 的乘积补足若干帧。

    返回新的布尔掩码，不修改入参。
    """
    valid = valid.copy()
    if total <= 0:
        return valid
    edges = np.linspace(0, total, n_bins + 1, dtype=int)
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        if int(valid[lo:hi].sum()) >= min_per_bin:
            continue
        seg_score = sharp_full[lo:hi] * flow_full[lo:hi]
        k = min(min_per_bin, hi - lo)
        top = np.argsort(seg_score)[-k:] + lo
        valid[top] = True
    return valid


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
    """按权重确定性选择 num_frames 个下标（段内使用）。

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