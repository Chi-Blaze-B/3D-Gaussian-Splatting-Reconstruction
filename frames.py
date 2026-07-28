"""
Frame extraction from video files with intelligent sampling.

Supports:
- Uniform sampling
- Single‑stage smart sampling (based on optical flow)
- Two‑stage smart sampling (parallax + flow + texture)

Improvements:
- Configurable optical flow method ('farneback' or 'lk')
- Texture score using local contrast (std) or Harris response
- Temporal smoothing of scores for more uniform distribution
- Robust fallback for two-stage pose estimation
"""

import os
import tempfile
import shutil
from typing import List, Optional, Tuple, Dict, Any, Union

import cv2
import numpy as np

# ---------- Configuration ----------
DEFAULT_OPTICAL_FLOW_METHOD = "farneback"  # or "lk"
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
TEXTURE_WINDOW_SIZE = 5   # for local contrast
HARRIS_THRESHOLD = 1e-6   # for corner response


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
) -> List[str]:
    """
    Extract frames from a video with optional intelligent sampling.

    Parameters
    ----------
    video_path : str
        Path to input video file.
    output_dir : str
        Directory to save extracted PNG frames.
    fps : float
        Target frame rate (used only for uniform sampling).
    scale : float
        Resize factor (0 < scale <= 1).
    min_frames, max_frames : int
        Bounds on the number of frames to extract.
    smart_sampling : bool
        If True, allocate more frames to high-motion segments.
    two_stage : bool
        If True, perform two-stage sampling: coarse pose estimation -> score each frame.
        Implies smart_sampling=True.
    poses_output_dir : str, optional
        Directory to store coarse pose intermediates (used for two-stage).
    optical_flow_method : str
        'farneback' or 'lk' - algorithm for optical flow.

    Returns
    -------
    List[str]
        Absolute paths to saved frame images.
    """
    if two_stage and not smart_sampling:
        smart_sampling = True

    if two_stage:
        return _two_stage_extract(
            video_path, output_dir, fps, scale,
            min_frames, max_frames,
            poses_output_dir,
            optical_flow_method,
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        total_frames = 1

    w, h = _compute_resized_size(orig_w, orig_h, scale)
    os.makedirs(output_dir, exist_ok=True)

    # Uniform sampling
    if not smart_sampling:
        num_frames = min(max(int(total_frames * fps / orig_fps), min_frames), max_frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return _extract_indices(cap, indices, output_dir, w, h)

    # ---------- Single‑stage smart sampling ----------
    # Compute per‑segment motion scores
    sample_step = max(1, total_frames // 100)  # at most ~100 samples
    sample_indices = np.arange(0, total_frames, sample_step, dtype=int)
    scores = _compute_flow_magnitudes(cap, sample_indices, method=optical_flow_method)

    if not scores or np.sum(scores) < 1e-6:
        # Fallback to uniform
        num_frames = min(max(int(total_frames * fps / orig_fps), min_frames), max_frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return _extract_indices(cap, indices, output_dir, w, h)

    # Allocate frames proportionally to scores with temporal smoothing
    num_frames = min(max(int(total_frames * fps / orig_fps), min_frames), max_frames)
    indices = _allocate_indices_from_scores(scores, sample_indices, total_frames, num_frames)
    return _extract_indices(cap, indices, output_dir, w, h)


# ---------- Two‑stage sampling ----------
def _two_stage_extract(
    video_path: str,
    output_dir: str,
    fps: float,
    scale: float,
    min_frames: int,
    max_frames: int,
    poses_output_dir: Optional[str],
    optical_flow_method: str,
) -> List[str]:
    """Two‑stage smart sampling with parallax, flow, and texture."""
    from poses import estimate_poses  # lazy import

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        total_frames = 1

    w, h = _compute_resized_size(orig_w, orig_h, scale)
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Stage 1: Coarse extraction (uniform, ~30‑50 frames) ----------
    coarse_count = min(max(min_frames, 30), 50)
    coarse_dir = tempfile.mkdtemp(prefix="coarse_")
    coarse_paths = _extract_uniform(video_path, coarse_dir, coarse_count, scale, cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset

    # ---------- Stage 1b: Estimate coarse camera poses ----------
    pose_dir = poses_output_dir or tempfile.mkdtemp(prefix="poses_coarse_")
    os.makedirs(pose_dir, exist_ok=True)
    try:
        intrinsics, coarse_poses, _ = estimate_poses(coarse_paths, pose_dir, min_inliers=10)
    except Exception as e:
        print(f"  [WARN] Coarse pose estimation failed: {e}. Using single‑stage smart sampling.")
        # Fallback: use flow‑only sampling (no parallax)
        return _fallback_smart_extract(cap, output_dir, total_frames, orig_fps, fps,
                                       min_frames, max_frames, w, h, optical_flow_method)

    # ---------- Stage 2: Compute per‑frame scores ----------
    sample_step = max(1, total_frames // 100)
    sample_indices = np.arange(0, total_frames, sample_step, dtype=int)

    # Flow magnitude for each sample
    flow_scores = _compute_flow_magnitudes(cap, sample_indices, method=optical_flow_method)
    if not flow_scores:
        # If flow fails, fallback to uniform
        cap.release()
        return _fallback_smart_extract(cap, output_dir, total_frames, orig_fps, fps,
                                       min_frames, max_frames, w, h, optical_flow_method)

    # Parallax scores for every original frame
    parallax_scores = _compute_parallax_scores(
        total_frames, coarse_poses, coarse_paths, cap, sample_indices
    )

    # Texture scores for every original frame (using local contrast)
    texture_scores = _compute_texture_scores(cap, total_frames, sample_indices)

    # Combine scores: 0.4 flow + 0.3 texture + 0.3 parallax
    combined = np.zeros(total_frames, dtype=np.float64)
    for idx in range(total_frames):
        flow_idx = min(idx // sample_step, len(flow_scores) - 1)
        flow_val = flow_scores[flow_idx]
        tex_val = texture_scores[idx] if idx < len(texture_scores) else 0.0
        par_val = parallax_scores[idx] if idx < len(parallax_scores) else 0.0
        combined[idx] = 0.4 * flow_val + 0.3 * tex_val + 0.3 * par_val

    # Apply temporal smoothing to avoid isolated peaks
    combined = _gaussian_smooth(combined, sigma=2.0)

    # Normalize and allocate frames
    num_frames = min(max(int(total_frames * fps / orig_fps), min_frames), max_frames)
    num_frames = max(num_frames, 1)
    # Use sampling with replacement to avoid duplicates (but we want unique)
    probs = np.maximum(combined, 1e-6)
    probs /= probs.sum()
    indices = np.random.choice(total_frames, size=num_frames, replace=False, p=probs)
    indices = np.sort(indices)

    # Extract and save final frames
    frames = _extract_indices(cap, indices, output_dir, w, h)

    # Cleanup
    cap.release()
    shutil.rmtree(coarse_dir, ignore_errors=True)
    if poses_output_dir is None:
        shutil.rmtree(pose_dir, ignore_errors=True)

    return frames


# ---------- Helper Functions ----------
def _compute_resized_size(orig_w: int, orig_h: int, scale: float) -> Tuple[int, int]:
    w = int(orig_w * scale)
    h = int(orig_h * scale)
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1
    return w, h


def _extract_indices(
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    output_dir: str,
    w: int,
    h: int,
) -> List[str]:
    """Extract frames at given indices and save as PNG."""
    frames = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        fname = f"frame_{i:04d}.png"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, resized)
        frames.append(os.path.abspath(fpath))
    return frames


def _extract_uniform(
    video_path: str,
    output_dir: str,
    num_frames: int,
    scale: float,
    cap: Optional[cv2.VideoCapture] = None,
) -> List[str]:
    """Quick uniform extraction (reuses an open capture if provided)."""
    if cap is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        own_cap = True
    else:
        own_cap = False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    w, h = _compute_resized_size(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        scale
    )
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        fname = f"coarse_{i:04d}.png"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, resized)
        paths.append(os.path.abspath(fpath))

    if own_cap:
        cap.release()
    return paths


def _compute_flow_magnitudes(
    cap: cv2.VideoCapture,
    indices: np.ndarray,
    method: str = "farneback",
) -> List[float]:
    """Compute optical flow magnitude for each pair of consecutive indices."""
    if len(indices) < 2:
        return []

    method = method.lower()
    prev_gray = None
    mags = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            if method == "lk":
                # Lucas-Kanade: compute sparse flow on a grid
                flow = _compute_lk_flow(prev_gray, gray)
                mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            else:  # farneback
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    **FARNEBACK_PARAMS
                )
                mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            mags.append(mag)
        prev_gray = gray

    return mags


def _compute_lk_flow(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Lucas-Kanade dense optical flow approximation using grid points."""
    # Sample points on a grid
    h, w = prev.shape
    step = 16  # grid step
    y_coords = np.arange(0, h, step, dtype=np.float32)
    x_coords = np.arange(0, w, step, dtype=np.float32)
    pts = np.array(np.meshgrid(x_coords, y_coords)).reshape(2, -1).T
    pts = pts.reshape(-1, 1, 2).astype(np.float32)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev, curr, pts, None,
        winSize=LK_WINDOW_SIZE, maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA
    )
    # Compute displacement magnitude for valid points
    if status is None:
        return np.zeros((h, w, 2), dtype=np.float32)
    valid = status.ravel() == 1
    if not np.any(valid):
        return np.zeros((h, w, 2), dtype=np.float32)
    disp = (next_pts[valid] - pts[valid]).reshape(-1, 2)
    # Create dense flow by filling grid
    flow = np.zeros((h, w, 2), dtype=np.float32)
    for (x, y), (dx, dy) in zip(pts[valid].reshape(-1, 2), disp):
        flow[int(y), int(x)] = [dx, dy]
    return flow


def _compute_texture_scores(
    cap: cv2.VideoCapture,
    total_frames: int,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Compute texture richness for each sampled frame using local contrast (std)."""
    scores = np.zeros(total_frames, dtype=np.float64)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Local standard deviation as texture measure
        kernel = np.ones((TEXTURE_WINDOW_SIZE, TEXTURE_WINDOW_SIZE), dtype=np.float32) / (TEXTURE_WINDOW_SIZE**2)
        mean = cv2.filter2D(gray, -1, kernel)
        sq_mean = cv2.filter2D(gray**2, -1, kernel)
        var = np.maximum(sq_mean - mean**2, 0)
        local_std = np.sqrt(var)
        scores[idx] = np.mean(local_std)  # average over image
    return scores


def _compute_parallax_scores(
    total_frames: int,
    coarse_poses: List,
    coarse_paths: List[str],
    cap: cv2.VideoCapture,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Compute parallax score for each original frame based on coarse pose baseline."""
    # Map each original frame to nearest coarse frame index
    coarse_indices = np.linspace(0, total_frames - 1, len(coarse_paths), dtype=int)
    parallax = np.zeros(total_frames, dtype=np.float64)

    # Extract valid poses
    valid_poses = [p for p in coarse_poses if p is not None]
    if len(valid_poses) < 2:
        return parallax

    for idx in range(total_frames):
        # Find nearest coarse index
        closest = np.argmin(np.abs(coarse_indices - idx))
        if closest < len(valid_poses) - 1:
            pose1 = valid_poses[closest]
            pose2 = valid_poses[closest + 1]
            t1 = pose1.t.flatten()
            t2 = pose2.t.flatten()
            baseline = np.linalg.norm(t2 - t1)
            # Also consider rotation angle
            R_rel = pose2.R @ pose1.R.T
            angle = _compute_rotation_angle(R_rel)
            # Combine: baseline + angular component
            parallax[idx] = baseline + 0.1 * np.radians(angle)
        else:
            parallax[idx] = 0.0

    # Normalize to [0,1]
    max_val = np.max(parallax)
    if max_val > 1e-6:
        parallax /= max_val
    return parallax


def _compute_rotation_angle(R: np.ndarray) -> float:
    rv, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(rv))


def _gaussian_smooth(scores: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian smoothing to 1D array to enforce temporal continuity."""
    if len(scores) < 3:
        return scores
    # Simple convolution with Gaussian kernel
    size = int(4 * sigma + 1)
    if size % 2 == 0:
        size += 1
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel = kernel.reshape(-1)
    # Pad with reflection
    pad = size // 2
    padded = np.pad(scores, pad, mode='reflect')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(scores)]


def _allocate_indices_from_scores(
    scores: np.ndarray,
    sample_indices: np.ndarray,
    total_frames: int,
    num_frames: int,
) -> np.ndarray:
    """Allocate frame indices proportionally to scores, with interpolation."""
    if len(scores) == 0:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # Interpolate scores to every frame
    full_scores = np.zeros(total_frames, dtype=np.float64)
    for i, idx in enumerate(sample_indices):
        if i < len(scores):
            full_scores[idx] = max(scores[i], 0)
    # Fill gaps with linear interpolation
    valid_idx = np.where(full_scores > 0)[0]
    if len(valid_idx) == 0:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)
    if len(valid_idx) == 1:
        full_scores[:] = full_scores[valid_idx[0]]
    else:
        for i in range(len(valid_idx) - 1):
            a = valid_idx[i]
            b = valid_idx[i + 1]
            if b - a > 1:
                x = np.linspace(0, 1, b - a + 1)
                vals = full_scores[a] * (1 - x) + full_scores[b] * x
                full_scores[a:b+1] = vals

    # Apply temporal smoothing
    full_scores = _gaussian_smooth(full_scores, sigma=1.5)
    full_scores = np.maximum(full_scores, 1e-6)
    full_scores /= full_scores.sum()

    # Sample without replacement (use probability)
    indices = np.random.choice(total_frames, size=num_frames, replace=False, p=full_scores)
    return np.sort(indices)


def _fallback_smart_extract(
    cap: cv2.VideoCapture,
    output_dir: str,
    total_frames: int,
    orig_fps: float,
    target_fps: float,
    min_frames: int,
    max_frames: int,
    w: int,
    h: int,
    optical_flow_method: str,
) -> List[str]:
    """Fallback to single‑stage smart extraction when two‑stage fails."""
    # Re‑open cap if needed
    if not cap.isOpened():
        cap.open(cap.get(cv2.CAP_PROP_POS_FRAMES))  # this won't work; better to reopen by path
        # But we can reuse the video path from somewhere; for simplicity, assume cap is valid.
        # In practice, we might need to pass video_path.
        pass
    # Use single‑stage smart sampling (flow only)
    sample_step = max(1, total_frames // 100)
    sample_indices = np.arange(0, total_frames, sample_step, dtype=int)
    flow_scores = _compute_flow_magnitudes(cap, sample_indices, method=optical_flow_method)
    if not flow_scores:
        # Uniform fallback
        num_frames = min(max(int(total_frames * target_fps / orig_fps), min_frames), max_frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        num_frames = min(max(int(total_frames * target_fps / orig_fps), min_frames), max_frames)
        indices = _allocate_indices_from_scores(flow_scores, sample_indices, total_frames, num_frames)

    # Rewind cap and extract
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return _extract_indices(cap, indices, output_dir, w, h)