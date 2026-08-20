"""
Robust camera pose estimation using ORB (or SIFT) features + incremental SfM.

Implements:
  - Feature extraction (ORB/SIFT)
  - Essential matrix + PnP relocalization
  - Keyframe management with co-visibility
  - Local and global Bundle Adjustment (with point optimization)
  - Map point pruning and filtering

No external COLMAP dependency — purely OpenCV + SciPy.
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set, Any
from collections import defaultdict

import numpy as np
import cv2
from scipy.optimize import least_squares

# ---------- Constants ----------
MIN_INLIERS = 25
KEYFRAME_ANGLE_DEG = 5.0
KEYFRAME_TRANS_RATIO = 0.05
COVIS_RATIO_THRESH = 0.25
BA_MAX_ITER = 15                # was 30, reduced for speed and stability
GLOBAL_BA_ITER = 25             # was 50
MIN_BA_WINDOW = 5
PRUNE_INTERVAL = 200            # fixed
MIN_OBSERVATIONS = 2 
MAX_REPROJ_ERROR = 4.0
SMALL_TRANSLATION = 1e-4
MATCH_DIST = 90
DESC_UPDATE_THRESH = 35
MIN_FEATURES = 80
MIN_TRI_ANGLE_DEG = 2.0
INIT_MIN_TRANSLATION = 0.01
KEYFRAME_CULLING_WINDOW = 10
LOCAL_MAP_RADIUS = 2
PNP_WINDOW = 12           # PnP 精修使用的最近关键帧数（原 LOCAL_MAP_RADIUS=2 匹配不足，全部关键帧太慢）
MAX_POINTS_IN_BA = 300          # was 150, increase for better point quality
EPS_MIN = 1e-8
EPS_MAX = 0.1
BA_MAX_OBS = 2000               # new: cap observations per BA call
BA_F_SCALE_MULTIPLIER = 3.0     # new: make loss more robust

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Pose] %(message)s")


# ---------- Data Structures ----------
@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float32)


@dataclass(frozen=True)
class CameraPose:
    R: np.ndarray
    t: np.ndarray

    @property
    def RT(self) -> np.ndarray:
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = self.R
        RT[:3, 3] = self.t.flatten()
        return RT

    def scaled(self, factor: float) -> "CameraPose":
        """返回平移缩放后的新 CameraPose（frozen dataclass 不可原地改）。"""
        return CameraPose(self.R, (self.t * factor).astype(np.float32))


# ---------- Main Estimator ----------
def estimate_poses(
    frame_paths: List[str],
    *,
    min_inliers: int = MIN_INLIERS,
    feature_type: str = "orb",
    focal_guess: Optional[float] = None,
    aspect_ratio: float = 1.0,
) -> Tuple[CameraIntrinsics, List[CameraPose], np.ndarray]:
    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")

    logger.info("Loading images and extracting features...")
    img_shape, kp_list, desc_list = _extract_features(frame_paths, feature_type)

    h, w = img_shape
    cx, cy = w / 2.0, h / 2.0
    focal0 = focal_guess if focal_guess is not None else max(w, h) * 1.2
    fy0 = focal0 * aspect_ratio
    image_size = max(w, h)

    reproj_thresh = max(0.5, min(3.0, image_size * 0.0015))
    ransac_thresh = reproj_thresh * 0.8
    triang_thresh = reproj_thresh
    logger.info(f"Thresholds: reproj={reproj_thresh:.2f}, ransac={ransac_thresh:.2f}")

    map_points: List[Dict] = []
    frame_poses: List[Optional[CameraPose]] = [None] * len(frame_paths)
    feat_map = [[-1] * len(kp) for kp in kp_list]
    frame_to_points: Dict[int, Set[int]] = defaultdict(set)

    frame_poses[0] = CameraPose(np.eye(3), np.zeros((3, 1)))
    keyframes = [0]
    last_pose = frame_poses[0]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    initialized = False
    init_candidates = []
    ba_counter = 0

    for i in range(1, len(frame_paths)):
        logger.info(f"Processing frame {i}/{len(frame_paths)-1}")

        if len(kp_list[i]) < MIN_FEATURES // 2:
            logger.warning(f"Frame {i} has too few features, copying previous pose")
            frame_poses[i] = last_pose
            continue

        matches = _match_features(desc_list[i-1], desc_list[i], bf)
        if len(matches) < min_inliers:
            frame_poses[i] = last_pose
            continue

        pts_prev = np.array([kp_list[i-1][m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_curr = np.array([kp_list[i][m.trainIdx].pt for m in matches], dtype=np.float32)

        E, mask = cv2.findEssentialMat(pts_prev, pts_curr, focal=focal0, pp=(cx, cy),
                                       method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
        if E is None or mask.sum() < min_inliers:
            frame_poses[i] = last_pose
            continue

        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_prev, pts_curr,
                                                     focal=focal0, pp=(cx, cy), mask=mask)
        inlier_count = int(mask_pose.sum())
        if inlier_count < min_inliers:
            frame_poses[i] = last_pose
            continue

        trans_norm = np.linalg.norm(t_rel)
        is_pure_rotation = trans_norm < SMALL_TRANSLATION

        R_curr = R_rel @ last_pose.R
        t_curr = R_rel @ last_pose.t + t_rel

        if not initialized:
            if not is_pure_rotation and trans_norm > INIT_MIN_TRANSLATION:
                norm_t = np.linalg.norm(t_curr)
                if norm_t > 0:
                    t_curr = t_curr / norm_t
                    new_pose = CameraPose(R_curr, t_curr)
                    frame_poses[i] = new_pose
                    last_pose = new_pose
                    initialized = True
                    _, new_pose = _triangulate_new_points(i, matches, mask_pose.ravel().astype(bool),
                                                           kp_list, pts_prev, pts_curr,
                                                           frame_poses[i-1], new_pose,
                                                           focal0, fy0, cx, cy,
                                                           map_points, feat_map, desc_list, frame_to_points,
                                                           triang_thresh)
                    frame_poses[i] = new_pose
                    last_pose = new_pose
                    keyframes.append(i)
                    logger.info(f"Initialized with frames 0 and {i}")
                    continue
                else:
                    init_candidates.append((i, matches, mask_pose, pts_prev, pts_curr))
                    frame_poses[i] = last_pose
                    continue
            else:
                if len(init_candidates) >= 2:
                    for j in range(len(init_candidates)-1, -1, -1):
                        idx_cand, _, _, _, _ = init_candidates[j]
                        if idx_cand > 0 and idx_cand < i:
                            matches_cand = _match_features(desc_list[idx_cand], desc_list[i], bf)
                            if len(matches_cand) >= min_inliers:
                                pts_cand = np.array([kp_list[idx_cand][m.queryIdx].pt for m in matches_cand], dtype=np.float32)
                                pts_cur2 = np.array([kp_list[i][m.trainIdx].pt for m in matches_cand], dtype=np.float32)
                                E2, mask2 = cv2.findEssentialMat(pts_cand, pts_cur2, focal=focal0, pp=(cx, cy),
                                                                  method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
                                if E2 is not None and mask2.sum() >= min_inliers:
                                    _, R_rel2, t_rel2, mask_pose2 = cv2.recoverPose(E2, pts_cand, pts_cur2,
                                                                                    focal=focal0, pp=(cx, cy), mask=mask2)
                                    if mask_pose2.sum() >= min_inliers and np.linalg.norm(t_rel2) > INIT_MIN_TRANSLATION:
                                        R_curr2 = R_rel2 @ frame_poses[idx_cand].R
                                        t_curr2 = R_rel2 @ frame_poses[idx_cand].t + t_rel2
                                        norm_t2 = np.linalg.norm(t_curr2)
                                        if norm_t2 > 0:
                                            t_curr2 = t_curr2 / norm_t2
                                            new_pose = CameraPose(R_curr2, t_curr2)
                                            frame_poses[i] = new_pose
                                            last_pose = new_pose
                                            initialized = True
                                            _, new_pose = _triangulate_new_points(i, matches_cand, mask_pose2.ravel().astype(bool),
                                                                                  kp_list, pts_cand, pts_cur2,
                                                                                  frame_poses[idx_cand], new_pose,
                                                                                  focal0, fy0, cx, cy,
                                                                                  map_points, feat_map, desc_list, frame_to_points,
                                                                                  triang_thresh)
                                            frame_poses[i] = new_pose
                                            last_pose = new_pose
                                            keyframes.append(i)
                                            logger.info(f"Initialized with frames {idx_cand} and {i}")
                                            break
                    if initialized:
                        continue
                frame_poses[i] = last_pose
                continue

        new_pose = CameraPose(R_curr, t_curr)
        inlier_mask = mask_pose.ravel().astype(bool)

        _, new_pose = _triangulate_new_points(i, matches, inlier_mask,
                                              kp_list, pts_prev, pts_curr,
                                              frame_poses[i-1], new_pose,
                                              focal0, fy0, cx, cy,
                                              map_points, feat_map, desc_list, frame_to_points,
                                              triang_thresh)

        # ----- Local map tracking (PnP refinement) -----
        # PnP 用已有 3D 点计算相机位姿，天然提供正确的尺度锚点（解决 t_rel 单位范数
        # 导致的累积尺度漂移）。原条件 LOCAL_MAP_RADIUS=2 匹配点常不足，PnP 很少触发。
        # 放宽到最近 PNP_WINDOW 个关键帧（而非全部——全部关键帧匹配太慢），
        # 兼顾 PnP 触发率与速度。
        if len(keyframes) > 1:
            pts3d_local, pts2d_local = [], []
            local_kfs = keyframes[-PNP_WINDOW:]
            for kf in local_kfs:
                if kf == i:
                    continue
                matches_kf = _match_features(desc_list[kf], desc_list[i], bf)
                for m in matches_kf:
                    pt_idx = feat_map[kf][m.queryIdx]
                    if pt_idx >= 0:
                        pts3d_local.append(map_points[pt_idx]['xyz'])
                        pts2d_local.append(kp_list[i][m.trainIdx].pt)
            if len(pts3d_local) >= 8:
                pts3d_local = np.array(pts3d_local, dtype=np.float32)
                pts2d_local = np.array(pts2d_local, dtype=np.float32)
                _, rvec_pnp, tvec_pnp, inliers_pnp = cv2.solvePnPRansac(
                    pts3d_local, pts2d_local,
                    np.array([[focal0, 0, cx], [0, fy0, cy], [0, 0, 1]], dtype=np.float32),
                    np.zeros(4),
                    iterationsCount=50, reprojectionError=reproj_thresh, confidence=0.95
                )
                if inliers_pnp is not None and len(inliers_pnp) >= 8:
                    R_pnp, _ = cv2.Rodrigues(rvec_pnp)
                    t_pnp = tvec_pnp.reshape(3, 1)
                    depth_median = np.median([np.linalg.norm(p) for p in pts3d_local])
                    if np.linalg.norm(t_pnp) < 10.0 * depth_median:
                        new_pose = CameraPose(R_pnp, t_pnp)

        # ----- Keyframe decision -----
        angle = _compute_rotation_angle(R_rel)
        trans_world = np.linalg.norm(t_curr - last_pose.t) / (np.median([np.linalg.norm(p['xyz']) for p in map_points]) + 1e-6)
        covis_ratio = _compute_covisibility_ratio(i, keyframes, matches, feat_map, frame_to_points)
        is_keyframe = (angle > KEYFRAME_ANGLE_DEG or
                       trans_world > KEYFRAME_TRANS_RATIO or
                       covis_ratio < COVIS_RATIO_THRESH)

        if is_keyframe and not is_pure_rotation and len(map_points) > 20:
            if len(keyframes) > KEYFRAME_CULLING_WINDOW:
                recent = keyframes[-KEYFRAME_CULLING_WINDOW:-1]
                for kf in recent:
                    if i in frame_to_points and kf in frame_to_points:
                        covis = len(frame_to_points[i] & frame_to_points[kf]) / max(1, min(len(frame_to_points[i]), len(frame_to_points[kf])))
                        if covis > 0.8:
                            keyframes.remove(kf)
                            break
            keyframes.append(i)

            ba_counter += 1
            if len(keyframes) >= MIN_BA_WINDOW and ba_counter % 2 == 0:
                window = keyframes[-MIN_BA_WINDOW:]
                focal0, fy0 = _bundle_adjustment(
                    window, map_points, feat_map, frame_poses,
                    kp_list, focal0, fy0, cx, cy,
                    optimize_points=True,
                    reproj_thresh=reproj_thresh,
                    image_size=image_size,
                    max_iter=BA_MAX_ITER,
                    is_global=False
                )

        last_pose = new_pose
        frame_poses[i] = new_pose

        if i % PRUNE_INTERVAL == 0 and len(map_points) > 100:
            _prune_map_points(map_points, feat_map, frame_to_points, reproj_thresh,
                              frame_poses, focal0, fy0, cx, cy)

    if not initialized:
        raise RuntimeError(
            "Could not initialize SfM: no pair of frames with sufficient translation found."
        )

    if len(keyframes) >= 3 and len(map_points) > 50:
        logger.info("Running global BA on all keyframes...")
        focal0, fy0 = _bundle_adjustment(
            keyframes, map_points, feat_map, frame_poses,
            kp_list, focal0, fy0, cx, cy,
            optimize_points=True,
            reproj_thresh=reproj_thresh,
            image_size=image_size,
            max_iter=GLOBAL_BA_ITER,
            is_global=True
        )

    _prune_map_points(map_points, feat_map, frame_to_points, reproj_thresh,
                      frame_poses, focal0, fy0, cx, cy)
    all_xyz, mask = _filter_point_cloud(map_points, frame_poses, focal0, fy0, cx, cy, reproj_thresh)
    all_xyz = all_xyz[mask] if np.any(mask) else all_xyz

    last_valid = frame_poses[0]
    for i, p in enumerate(frame_poses):
        if p is None:
            frame_poses[i] = last_valid
        else:
            last_valid = p

    intrinsics = CameraIntrinsics(fx=focal0, fy=fy0, cx=cx, cy=cy, k1=0.0)
    logger.info(f"Final: fx={focal0:.2f}, fy={fy0:.2f}, points={len(all_xyz)}")
    return intrinsics, frame_poses, all_xyz


# ---------- Feature Extraction ----------
def _extract_features(paths: List[str], feature_type: str = "orb"):
    # 修复(2026-08): 不再把全部原图常驻内存（200 帧 1080p 约 1.2GB）——调用方只用首帧 shape。
    kps, descs = [], []
    if feature_type == "sift":
        try:
            detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.03, edgeThreshold=10, sigma=1.6)
        except cv2.error as e:
            logger.warning(f"SIFT not available ({e}), falling back to ORB.")
            detector = cv2.ORB_create(nfeatures=12000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31)
    else:
        detector = cv2.ORB_create(nfeatures=12000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31)

    shape0 = None
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        if shape0 is None:
            shape0 = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        kps.append(kp)
        descs.append(desc if desc is not None else np.zeros((0, 32), dtype=np.uint8))
        del img, gray
    return shape0, kps, descs


# ---------- Match Features ----------
def _match_features(desc1, desc2, bf, ratio=0.75):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    if len(desc2) < 2:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc1, desc2)
        return [m for m in matches if m.distance < MATCH_DIST]
    raw = bf.knnMatch(desc1, desc2, k=2)
    raw = [pair for pair in raw if len(pair) == 2]
    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance and m.distance < MATCH_DIST:
            good.append(m)
    return good


# ---------- Helpers ----------
def _compute_rotation_angle(R):
    rv, _ = cv2.Rodrigues(R)
    return np.linalg.norm(rv) * 180.0 / np.pi


def _compute_covisibility_ratio(curr_idx, keyframes, matches, feat_map, frame_to_points):
    if not matches or len(keyframes) < 2:
        return 1.0
    last_kf = keyframes[-1]
    if last_kf not in frame_to_points:
        return 0.0
    last_points = frame_to_points[last_kf]
    count = sum(1 for m in matches if feat_map[curr_idx-1][m.queryIdx] in last_points)
    return count / len(matches) if matches else 1.0


def _add_observation(pt_dict, frame_idx, kp_idx, uv, frame_to_points):
    key = (frame_idx, kp_idx)
    if key not in pt_dict.get('obs_set', set()):
        pt_dict.setdefault('obs_set', set()).add(key)
        pt_dict.setdefault('obs', []).append((frame_idx, kp_idx, float(uv[0]), float(uv[1])))
        pt_dict['obs_count'] = pt_dict.get('obs_count', 0) + 1
        pt_idx = pt_dict.get('idx', -1)
        if pt_idx >= 0:
            frame_to_points[frame_idx].add(pt_idx)


def _update_map_point_descriptor(pt_dict, new_desc):
    old = pt_dict.get('desc')
    if old is None:
        pt_dict['desc'] = new_desc.copy()
        pt_dict['desc_age'] = 0
        return
    if old.dtype != np.uint8:
        old = old.astype(np.uint8)
    if new_desc.dtype != np.uint8:
        new_desc = new_desc.astype(np.uint8)
    dist = cv2.norm(old, new_desc, cv2.NORM_HAMMING)
    age = pt_dict.get('desc_age', 0)
    if pt_dict.get('obs_count', 0) < 3 or age > 5 or dist >= DESC_UPDATE_THRESH * 1.2:
        pt_dict['desc'] = new_desc.copy()
        pt_dict['desc_age'] = 0
    else:
        pt_dict['desc_age'] = age + 1


def _triangulate_new_points(curr_idx, matches, inlier_mask,
                            kp_list, pts_prev, pts_curr,
                            pose_prev, pose_curr,
                            focal, fy, cx, cy,
                            map_points, feat_map, desc_list, frame_to_points,
                            triang_thresh):
    K = np.array([[focal, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    P_prev = K @ pose_prev.RT[:3]
    P_curr = K @ pose_curr.RT[:3]
    new_indices = []  # 记录本次新增的点（用于尺度归一化）
    cam_prev = -pose_prev.R.T @ pose_prev.t
    cam_curr = -pose_curr.R.T @ pose_curr.t

    for idx in np.where(inlier_mask)[0]:
        m = matches[idx]
        pt_prev = pts_prev[idx].reshape(2, 1)
        pt_curr = pts_curr[idx].reshape(2, 1)
        prev_feat = m.queryIdx
        curr_feat = m.trainIdx

        if feat_map[curr_idx][curr_feat] >= 0:
            continue

        existing = feat_map[curr_idx - 1][prev_feat]
        if existing >= 0:
            _add_observation(map_points[existing], curr_idx, curr_feat, pt_curr.flatten(), frame_to_points)
            feat_map[curr_idx][curr_feat] = existing
            _update_map_point_descriptor(map_points[existing], desc_list[curr_idx][curr_feat])
            continue

        pts4d = cv2.triangulatePoints(P_prev, P_curr, pt_prev, pt_curr)
        pt3d = pts4d[:3] / (float(pts4d[3][0]) + 1e-12)
        pt3d = pt3d.flatten()

        if not _is_valid_point(pt3d, pose_prev, pose_curr, cam_prev, cam_curr,
                               triang_thresh, P_prev, P_curr, pt_prev, pt_curr):
            continue

        pt_idx = len(map_points)
        pt_dict = {
            'idx': pt_idx,
            'xyz': pt3d.astype(np.float32),
            'desc': desc_list[curr_idx-1][prev_feat].copy(),
            'obs': [],
            'obs_set': set(),
            'obs_count': 0,
            'desc_age': 0
        }
        _add_observation(pt_dict, curr_idx-1, prev_feat, pt_prev.flatten(), frame_to_points)
        _add_observation(pt_dict, curr_idx, curr_feat, pt_curr.flatten(), frame_to_points)
        map_points.append(pt_dict)
        feat_map[curr_idx-1][prev_feat] = pt_idx
        feat_map[curr_idx][curr_feat] = pt_idx
        new_indices.append(pt_idx)

    # ===== 修复: 尺度归一化（锚定到已有地图点的中位深度） =====
    # 问题: 每次本质矩阵恢复的 t_rel 是单位范数（up-to-scale），链式叠加
    #   t_curr = R_rel @ last_pose.t + t_rel 导致尺度随帧数累积漂移
    #   （实测 20帧可见性 74% → 200帧 6%）。
    # 修复: 新增点的中位深度应接近已有地图点的中位深度。用该比例同时缩放
    #   新增点的坐标和当前相机平移 pose_curr.t，使全序列尺度一致。
    # 注意: 深度用"相机前方 z 深度"（R_curr[2] @ (p - cam)）而非"到相机中心距离"，
    #   后者在相机靠近点时病态（ref_median 骤降 → scale 爆炸，实测点云跨度 393万）。
    #   scale 因子也 clamp 到 [0.05, 20]，防止极端值破坏点云。
    if new_indices and map_points:
        old_count = len(map_points) - len(new_indices)
        if old_count >= 10:
            old_depths = []
            cam_curr_flat = cam_curr.flatten()
            for p in map_points[:old_count]:
                if p['xyz'].size == 3:
                    # 相机前方 z 深度（正深度才有效）
                    d = float(pose_curr.R[2] @ (p['xyz'] - cam_curr_flat))
                    if d > 0:
                        old_depths.append(d)
            if len(old_depths) >= 5:
                ref_median = np.median(old_depths)
                new_depths = []
                for pi in new_indices:
                    d = float(pose_curr.R[2] @ (map_points[pi]['xyz'] - cam_curr_flat))
                    if d > 0:
                        new_depths.append(d)
                if new_depths:
                    new_median = np.median(new_depths)
                    if new_median > 1e-8 and ref_median > 1e-8:
                        scale = ref_median / new_median
                        # clamp 尺度因子：严格限制防止逐帧累积放大。
                        # 之前 [0.05,20] 允许连续帧各乘 ~20 → 点坐标指数爆炸
                        # （实测个别点范数 >1e6）。限制到 [0.5,2] 只做温和微调。
                        scale = np.clip(scale, 0.5, 2.0)
                        # 只缩放显著偏离的尺度（防止数值噪声抖动）
                        if scale > 1.5 or scale < 0.5:
                            for pi in new_indices:
                                map_points[pi]['xyz'] = (map_points[pi]['xyz'] * scale).astype(np.float32)
                            # 同步缩放当前相机平移，保持相机-点几何一致
                            pose_curr = pose_curr.scaled(scale)
    return new_indices, pose_curr


def _is_valid_point(pt3d, pose_prev, pose_curr, cam_prev, cam_curr,
                    reproj_th, P_prev, P_curr, pt_prev, pt_curr):
    if pt3d.shape != (3,):
        pt3d = pt3d.flatten()
    if not np.isfinite(pt3d).all():
        return False

    depth_prev = float(pose_prev.R[2] @ (pt3d - cam_prev.flatten()))
    depth_curr = float(pose_curr.R[2] @ (pt3d - cam_curr.flatten()))
    if depth_prev <= 0 or depth_curr <= 0:
        return False

    v1 = pt3d - cam_prev.flatten()
    v2 = pt3d - cam_curr.flatten()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return False
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    if cos_angle > np.cos(np.radians(MIN_TRI_ANGLE_DEG)):
        return False

    proj_prev = P_prev @ np.append(pt3d, 1.0)
    proj_curr = P_curr @ np.append(pt3d, 1.0)
    proj_prev = proj_prev[:2] / (proj_prev[2] + 1e-12)
    proj_curr = proj_curr[:2] / (proj_curr[2] + 1e-12)
    if np.linalg.norm(proj_prev - pt_prev.flatten()) > reproj_th or \
       np.linalg.norm(proj_curr - pt_curr.flatten()) > reproj_th:
        return False
    return True


# ---------- Robust Bundle Adjustment ----------
def _bundle_adjustment(
    keyframe_ids, map_points, feat_map, frame_poses,
    kp_list, focal, fy, cx, cy,
    optimize_points=True,
    reproj_thresh=1.0,
    image_size=1920,
    max_iter=BA_MAX_ITER,
    is_global=False,
):
    valid_kfs = [f for f in keyframe_ids if f < len(frame_poses) and frame_poses[f] is not None]
    if len(valid_kfs) < 2:
        return focal, fy
    keyframe_ids = valid_kfs

    obs = []
    for f_idx in keyframe_ids:
        for kp_idx, pt_idx in enumerate(feat_map[f_idx]):
            if pt_idx >= 0:
                u, v = kp_list[f_idx][kp_idx].pt
                obs.append((f_idx, pt_idx, u, v))

    if len(obs) < 10:
        return focal, fy

    # ===== 限制观测数量 =====
    # 修复(2026-08): 先用局部 RNG 并记录原始数量——旧版全局 np.random.seed(42) 污染
    #   frames/point_cloud 的随机性；且日志在 obs 被覆盖后才 len(obs)，恒打印 "2000/2000"。
    n_obs_orig = len(obs)
    rng = np.random.default_rng(42)
    if n_obs_orig > BA_MAX_OBS:
        idx = rng.choice(n_obs_orig, BA_MAX_OBS, replace=False)
        obs = [obs[i] for i in idx]
        logger.info(f"BA sampled {BA_MAX_OBS} observations out of {n_obs_orig}")

    logger.info(f"BA started: {len(keyframe_ids)} keyframes, {len(obs)} obs")

    if is_global and len(obs) > 3000:
        idx = rng.choice(len(obs), 3000, replace=False)
        obs = [obs[i] for i in idx]

    obs_count = {f: 0 for f in keyframe_ids}
    for f, _, _, _ in obs:
        obs_count[f] += 1
    fixed_kf = max(obs_count, key=obs_count.get)
    other_kfs = [f for f in keyframe_ids if f != fixed_kf]

    eps = _compute_adaptive_eps(map_points)
    param = [float(focal), float(fy)]
    for idx in other_kfs:
        rv, _ = cv2.Rodrigues(frame_poses[idx].R)
        param.extend(rv.flatten())
        param.extend(frame_poses[idx].t.flatten())

    point_ids = []
    if optimize_points:
        point_ids = sorted({pt for _, pt, _, _ in obs})
        if len(point_ids) > MAX_POINTS_IN_BA:
            cnt = {pid: 0 for pid in point_ids}
            for _, pid, _, _ in obs:
                cnt[pid] += 1
            point_ids = sorted(cnt.keys(), key=lambda x: cnt[x], reverse=True)[:MAX_POINTS_IN_BA]
        for pid in point_ids:
            param.extend(map_points[pid]['xyz'])

    # ===== 修复: BA 平移/点坐标边界基于场景尺度，防止相机尺度漂移 =====
    # 旧实现: bound_scale = 100*image_size (~384000)，远超场景尺度（点云跨度~67），
    #   允许 BA 把相机推到离点云极远处（实测相机轨迹跨度 2579 vs 点云 67），
    #   导致投影后点云仅 ~7% 落入画面。
    # 新实现: 用 map_points 的中位深度/跨度作为场景尺度基准，平移边界设为
    #   场景跨度的数倍，相机被约束在点云附近，保持尺度一致。
    scene_scale = 1.0
    if map_points:
        xs = np.array([p['xyz'] for p in map_points if p['xyz'].size == 3])
        if len(xs) > 0:
            scene_scale = float(np.linalg.norm(xs.max(axis=0) - xs.min(axis=0))) + 1e-6
    # 相机平移边界：基于场景尺度与当前相机范围（确保初始猜测在界内）
    # 平移可能因尺度漂移略超场景尺度，取场景尺度 5 倍与相机最大平移 2 倍的较大者
    cam_t_max = 1.0
    if other_kfs:
        cam_ts = [np.linalg.norm(frame_poses[k].t.flatten()) for k in other_kfs
                  if frame_poses[k] is not None]
        if cam_ts:
            cam_t_max = max(cam_ts)
    t_bound = max(scene_scale * 5.0, cam_t_max * 2.0, 10.0)
    # 旋转向量边界：cv2.Rodrigues 的旋转向量范数可远超 π（冗余表示，接近 360°
    # 旋转时范数可达 ~4π 甚至更大），设任何有限边界都会导致 "Initial guess is
    # outside of provided bounds"。旋转向量本身有界性由 Rodrigues 保证（同一旋转
    # 有多个表示），因此对旋转不设边界（np.inf），只约束相机平移防尺度漂移。
    r_bound = np.inf
    # 修复(2026-08): bounds 必须与交错参数排布一致（每关键帧 3 旋转 + 3 平移，见 param 构造）。
    # 旧版先全部旋转再全部平移 → 前半关键帧平移拿到 ±inf（防尺度漂移失效）、
    #   后半关键帧旋转被 ±t_bound 误约束（旋转向量范数可达 ~4π，越界即 BA 失败）。
    lower_pose, upper_pose = [], []
    for _kf in other_kfs:
        lower_pose += [-r_bound] * 3 + [-t_bound] * 3
        upper_pose += [r_bound] * 3 + [t_bound] * 3
    bounds_lower = [1.0, 1.0] + lower_pose
    bounds_upper = [10.0 * image_size, 10.0 * image_size] + upper_pose
    if optimize_points and point_ids:
        # 点坐标边界：不设硬限制（用 np.inf），避免 BA 因点坐标越界失败。
        # 相机平移边界已约束尺度漂移（t_bound 基于场景尺度），
        # 点坐标在 BA 中相对相机优化，无需单独限制（旧实现 ±1e6 同效）。
        bounds_lower += [-np.inf] * len(point_ids) * 3
        bounds_upper += [np.inf] * len(point_ids) * 3

    def residuals(params):
        f = params[0]
        fy_local = params[1]
        n_poses = len(other_kfs)
        poses = {fixed_kf: frame_poses[fixed_kf]}
        for i, idx in enumerate(other_kfs):
            start = 2 + i * 6
            rv = params[start:start+3]
            t = params[start+3:start+6]
            R, _ = cv2.Rodrigues(rv)
            poses[idx] = CameraPose(R, t.reshape(3, 1))

        pts = {pid: map_points[pid]['xyz'] for _, pid, _, _ in obs}
        if optimize_points and point_ids:
            pts_start = 2 + n_poses * 6
            for j, pid in enumerate(point_ids):
                pts[pid] = params[pts_start + j*3 : pts_start + j*3 + 3]

        res = []
        for f_idx, pt_idx, u_obs, v_obs in obs:
            pt3d = pts[pt_idx]
            pose = poses[f_idx]
            pt_cam = pose.R @ pt3d.reshape(3, 1) + pose.t
            depth = float(pt_cam[2, 0])

            # ===== 修复: 深度惩罚随深度单调递增，防止点被推到相机后方 =====
            # 旧实现: depth<=1e-8 时惩罚固定 1000（soft_l1 下约束弱），
            # 且 depth<eps 用 -log(depth/eps)，导致深度越接近 0 惩罚反而骤降，
            # 点可以被推到负深度（45% 点在相机后方的根因）。
            # 新实现: 对 depth<=1e-6 施加随深度减小的强对数障碍，单调递增惩罚，
            # 保证 BA 不会把点推到相机后方。
            if depth <= 1e-6:
                # 深度为负或近零 -> 强惩罚（随深度减小单调递增）
                barrier = -np.log(max(depth / 1e-6, 1e-10))
                res.append(barrier)
                res.append(barrier)
                continue

            if depth < eps:
                barrier = -np.log(max(depth / eps, 1e-10))
                res.append(barrier)
                res.append(barrier)
                continue

            x = float(pt_cam[0, 0]) / depth
            y = float(pt_cam[1, 0]) / depth
            u_pred = f * x + cx
            v_pred = fy_local * y + cy
            res.append(u_pred - u_obs)
            res.append(v_pred - v_obs)

        return np.array(res, dtype=np.float64)

    try:
        # ===== 使用更宽容的 f_scale =====
        f_scale = reproj_thresh * BA_F_SCALE_MULTIPLIER
        # ===== 诊断: 检查初始猜测是否越界（定位越界参数） =====
        param_np = np.array(param, dtype=np.float64)
        lb = np.array(bounds_lower, dtype=np.float64)
        ub = np.array(bounds_upper, dtype=np.float64)
        if len(param_np) == len(lb):
            viol = (param_np < lb) | (param_np > ub)
            if viol.any():
                idxs = np.where(viol)[0]
                msg = ", ".join(
                    f"p[{i}]={param_np[i]:.2f} (bnd [{lb[i]:.2f},{ub[i]:.2f}])"
                    for i in idxs[:5]
                )
                logger.info(f"[BA] initial guess violates bounds: {msg}")
        result = least_squares(
            residuals, param_np,
            bounds=(bounds_lower, bounds_upper),
            method='trf', loss='soft_l1', f_scale=f_scale,
            max_nfev=max_iter, verbose=0,
            ftol=1e-4, xtol=1e-4, gtol=1e-4   # relaxed tolerance
        )
        if result.success:
            focal_new = max(float(result.x[0]), 1.0)
            fy_new = max(float(result.x[1]), 1.0)
            n_poses = len(other_kfs)
            for i, idx in enumerate(other_kfs):
                start = 2 + i * 6
                rv = result.x[start:start+3]
                t = result.x[start+3:start+6]
                R, _ = cv2.Rodrigues(rv)
                frame_poses[idx] = CameraPose(R, t.reshape(3, 1))
            if optimize_points and point_ids:
                pts_start = 2 + n_poses * 6
                for j, pid in enumerate(point_ids):
                    map_points[pid]['xyz'] = result.x[pts_start + j*3 : pts_start + j*3 + 3]
            logger.info(f"BA finished: focal {focal:.2f}->{focal_new:.2f}, fy {fy:.2f}->{fy_new:.2f}")
            return focal_new, fy_new
        else:
            logger.warning("BA failed (least_squares did not converge)")
            return focal, fy
    except Exception as e:
        logger.warning(f"BA exception: {e}")
        return focal, fy


def _compute_adaptive_eps(map_points, default_eps=0.01):
    depths = []
    for p in map_points:
        if p['xyz'].size == 3 and np.isfinite(p['xyz']).all():
            depths.append(np.linalg.norm(p['xyz']))
    if depths:
        median = np.median(depths)
        eps = median * 0.001
        return np.clip(eps, EPS_MIN, EPS_MAX)
    return default_eps


def _prune_map_points(map_points, feat_map, frame_to_points, reproj_thresh,
                      frame_poses, focal, fy, cx, cy):
    if not map_points:
        return
    to_remove = []
    for idx, pt in enumerate(map_points):
        pt['idx'] = idx
        if pt.get('obs_count', 0) < MIN_OBSERVATIONS:
            to_remove.append(idx)
            continue
        xyz = pt['xyz']
        if not np.isfinite(xyz).all():
            to_remove.append(idx)
            continue
        total_err = 0.0
        count = 0
        neg_depth = 0
        total_obs = 0
        for f_idx, kp_idx, u_obs, v_obs in pt['obs']:
            pose = frame_poses[f_idx]
            if pose is None:
                continue
            pt_cam = pose.R @ xyz.reshape(3, 1) + pose.t
            depth = float(pt_cam[2, 0])
            total_obs += 1
            if depth <= 0:
                neg_depth += 1
                continue
            x = float(pt_cam[0, 0]) / depth
            y = float(pt_cam[1, 0]) / depth
            u_pred = focal * x + cx
            v_pred = fy * y + cy
            err = np.sqrt((u_pred - u_obs)**2 + (v_pred - v_obs)**2)
            total_err += err
            count += 1
        # ===== 修复: 负深度为主的点直接修剪（相机后方的点无效） =====
        if total_obs > 0 and neg_depth / total_obs >= 0.5:
            to_remove.append(idx)
            continue
        if count > 0 and total_err / count > MAX_REPROJ_ERROR * reproj_thresh:
            to_remove.append(idx)

    if not to_remove:
        return

    remove_set = set(to_remove)
    idx_map = {}
    new_idx = 0
    for old_idx in range(len(map_points)):
        if old_idx in remove_set:
            idx_map[old_idx] = -1
        else:
            idx_map[old_idx] = new_idx
            new_idx += 1
    for f_idx in range(len(feat_map)):
        for kp_idx in range(len(feat_map[f_idx])):
            old = feat_map[f_idx][kp_idx]
            if old >= 0:
                feat_map[f_idx][kp_idx] = idx_map.get(old, -1)

    for f in list(frame_to_points.keys()):
        frame_to_points[f] = set()
    for old_idx in reversed(to_remove):
        del map_points[old_idx]
    for new_idx, pt in enumerate(map_points):
        pt['idx'] = new_idx
        for f_idx, _, _, _ in pt['obs']:
            frame_to_points[f_idx].add(new_idx)

    logger.debug(f"Pruned {len(to_remove)} points, remaining {len(map_points)}")


def _filter_point_cloud(map_points, frame_poses, focal, fy, cx, cy, reproj_thresh):
    if not map_points:
        return np.array([]), np.array([])
    all_xyz = np.array([p['xyz'] for p in map_points])
    if all_xyz.size == 0:
        return all_xyz, np.array([])
    errors = []
    for pt in map_points:
        xyz = pt['xyz']
        obs = pt.get('obs', [])
        if len(obs) < MIN_OBSERVATIONS:
            errors.append(float('inf'))
            continue
        total_err = 0.0
        count = 0
        neg_depth = 0
        total_obs = 0
        for f_idx, _, u_obs, v_obs in obs:
            pose = frame_poses[f_idx]
            if pose is None:
                continue
            pt_cam = pose.R @ xyz.reshape(3, 1) + pose.t
            depth = float(pt_cam[2, 0])
            total_obs += 1
            if depth <= 0:
                neg_depth += 1
                continue
            x = float(pt_cam[0, 0]) / depth
            y = float(pt_cam[1, 0]) / depth
            u_pred = focal * x + cx
            v_pred = fy * y + cy
            err = np.sqrt((u_pred - u_obs)**2 + (v_pred - v_obs)**2)
            total_err += err
            count += 1
        # ===== 修复: 负深度为主的点直接过滤（相机后方的点无效） =====
        # 旧实现: 负深度观测被跳过不计入误差，点保留 —— 导致大量点在相机后方
        if total_obs > 0 and neg_depth / total_obs >= 0.5:
            errors.append(float('inf'))
            continue
        if count >= MIN_OBSERVATIONS:
            errors.append(total_err / count)
        else:
            errors.append(float('inf'))
    errors = np.array(errors)
    finite = np.isfinite(errors)
    if not np.any(finite):
        mask = np.zeros(len(map_points), dtype=bool)
    else:
        median = np.median(errors[finite])
        threshold = max(median * 2.5, reproj_thresh * 1.5)
        mask = finite & (errors < threshold)
    return all_xyz, mask