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
KEYFRAME_ANGLE_DEG = 10.0
KEYFRAME_TRANS_RATIO = 0.08
COVIS_RATIO_THRESH = 0.3
BA_MAX_ITER = 80
GLOBAL_BA_ITER = 150
MIN_BA_WINDOW = 5
PRUNE_INTERVAL = 50
MIN_OBSERVATIONS = 3
MAX_REPROJ_ERROR = 3.0
SMALL_TRANSLATION = 1e-4
MATCH_DIST = 65                  # 降低此值（如 50）可增加匹配数量，但可能增加误匹配
DESC_UPDATE_THRESH = 35
MIN_FEATURES = 80
MIN_TRI_ANGLE_DEG = 1.5
INIT_MIN_TRANSLATION = 0.01
KEYFRAME_CULLING_WINDOW = 10
LOCAL_MAP_RADIUS = 3
MAX_POINTS_IN_BA = 500          # limit for global BA
EPS_MIN = 1e-8
EPS_MAX = 0.1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Pose] %(message)s")


# ---------- Data Structures ----------
@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0   # not used, kept for compatibility

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float32)


@dataclass(frozen=True)
class CameraPose:
    R: np.ndarray   # (3,3)
    t: np.ndarray   # (3,1)

    @property
    def RT(self) -> np.ndarray:
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = self.R
        RT[:3, 3] = self.t.flatten()
        return RT


# ---------- Main Estimator ----------
def estimate_poses(
    frame_paths: List[str],
    output_dir: str,
    *,
    min_inliers: int = MIN_INLIERS,
    feature_type: str = "orb",      # "orb" or "sift"
    focal_guess: Optional[float] = None,
    aspect_ratio: float = 1.0,
) -> Tuple[CameraIntrinsics, List[CameraPose], np.ndarray]:
    """
    Estimate camera poses from a sequence of frames.

    Returns:
        intrinsics: CameraIntrinsics
        poses: list of CameraPose, same length as frame_paths (None for unregistered frames filled with last valid)
        sparse_points: (N,3) array of reconstructed 3D points
    """
    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")

    logger.info("Loading images and extracting features...")
    images, kp_list, desc_list = _extract_features(frame_paths, feature_type)

    h, w = images[0].shape[:2]
    cx, cy = w / 2.0, h / 2.0
    focal0 = focal_guess if focal_guess is not None else max(w, h) * 1.2
    fy0 = focal0 * aspect_ratio
    image_size = max(w, h)

    # Adaptive thresholds
    reproj_thresh = max(0.5, min(2.0, image_size * 0.001))
    ransac_thresh = reproj_thresh * 0.8
    triang_thresh = reproj_thresh
    logger.info(f"Thresholds: reproj={reproj_thresh:.2f}, ransac={ransac_thresh:.2f}")

    # Data structures
    map_points: List[Dict] = []          # each dict: xyz, desc, obs list, obs_count, desc_age
    frame_poses: List[Optional[CameraPose]] = [None] * len(images)
    feat_map = [[-1] * len(kp) for kp in kp_list]   # frame -> keypoint -> map point index
    frame_to_points: Dict[int, Set[int]] = defaultdict(set)

    # First frame: identity
    frame_poses[0] = CameraPose(np.eye(3), np.zeros((3, 1)))
    keyframes = [0]
    last_pose = frame_poses[0]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ----- Process frames -----
    initialized = False
    # We'll attempt to initialize with the first non‑pure‑rotation pair.
    init_candidates = []   # store (prev_idx, curr_idx, matches, pts_prev, pts_curr, mask)

    for i in range(1, len(images)):
        logger.debug(f"Frame {i}/{len(images)-1}")

        if len(kp_list[i]) < MIN_FEATURES // 2:
            logger.warning(f"Frame {i} has too few features, copying previous pose")
            frame_poses[i] = last_pose
            continue

        # Match with previous frame
        matches = _match_features(desc_list[i-1], desc_list[i], bf)
        if len(matches) < min_inliers:
            frame_poses[i] = last_pose
            continue

        pts_prev = np.array([kp_list[i-1][m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_curr = np.array([kp_list[i][m.trainIdx].pt for m in matches], dtype=np.float32)

        # Essential matrix
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
        t_curr = R_rel @ last_pose.t + t_rel   # scale will be normalized upon init

        # ----- Initialization -----
        if not initialized:
            if not is_pure_rotation and trans_norm > INIT_MIN_TRANSLATION:
                # Found a good pair: initialize
                norm_t = np.linalg.norm(t_curr)
                if norm_t > 0:
                    t_curr = t_curr / norm_t
                    new_pose = CameraPose(R_curr, t_curr)
                    frame_poses[i] = new_pose
                    last_pose = new_pose
                    initialized = True
                    # Triangulate points from this pair
                    _triangulate_new_points(i, matches, mask_pose.ravel().astype(bool),
                                            kp_list, pts_prev, pts_curr,
                                            frame_poses[i-1], new_pose,
                                            focal0, fy0, cx, cy,
                                            map_points, feat_map, desc_list, frame_to_points,
                                            triang_thresh)
                    keyframes.append(i)
                    logger.info(f"Initialized with frames 0 and {i}")
                    continue
                else:
                    # Pure rotation or too small translation – store potential candidate
                    init_candidates.append((i, matches, mask_pose, pts_prev, pts_curr))
                    frame_poses[i] = last_pose
                    continue
            else:
                # Still looking for good pair; if we have enough candidates, try pairing among them
                if len(init_candidates) >= 2:
                    # Try to initialize using the last two candidates
                    for j in range(len(init_candidates)-1, -1, -1):
                        idx_cand, _, _, _, _ = init_candidates[j]
                        if idx_cand > 0 and idx_cand < i:
                            # Try essential matrix between candidate frame and current
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
                                            _triangulate_new_points(i, matches_cand, mask_pose2.ravel().astype(bool),
                                                                    kp_list, pts_cand, pts_cur2,
                                                                    frame_poses[idx_cand], new_pose,
                                                                    focal0, fy0, cx, cy,
                                                                    map_points, feat_map, desc_list, frame_to_points,
                                                                    triang_thresh)
                                            keyframes.append(i)
                                            logger.info(f"Initialized with frames {idx_cand} and {i}")
                                            break
                    if initialized:
                        continue
                # If still not initialized, copy pose and continue
                frame_poses[i] = last_pose
                continue

        # ----- Normal processing (initialized) -----
        # Estimate new pose from relative motion
        new_pose = CameraPose(R_curr, t_curr)
        inlier_mask = mask_pose.ravel().astype(bool)

        # Triangulate new points
        _triangulate_new_points(i, matches, inlier_mask,
                                kp_list, pts_prev, pts_curr,
                                frame_poses[i-1], new_pose,
                                focal0, fy0, cx, cy,
                                map_points, feat_map, desc_list, frame_to_points,
                                triang_thresh)

        # ----- Local map tracking (PnP refinement) -----
        if len(keyframes) > 1:
            local_kfs = keyframes[-LOCAL_MAP_RADIUS:]
            pts3d_local, pts2d_local = [], []
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
                    # Check scale
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
            # Keyframe culling
            if len(keyframes) > KEYFRAME_CULLING_WINDOW:
                recent = keyframes[-KEYFRAME_CULLING_WINDOW:-1]
                for kf in recent:
                    if i in frame_to_points and kf in frame_to_points:
                        covis = len(frame_to_points[i] & frame_to_points[kf]) / max(1, min(len(frame_to_points[i]), len(frame_to_points[kf])))
                        if covis > 0.8:
                            keyframes.remove(kf)
                            break
            keyframes.append(i)

            # Run local BA
            if len(keyframes) >= MIN_BA_WINDOW:
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
                # Update scene depth scale
                depths = [np.linalg.norm(p['xyz']) for p in map_points if np.isfinite(np.linalg.norm(p['xyz']))]
                if depths:
                    scale = np.median(depths)
                    # Scale all poses and points to keep numeric stability
                    # (we don't explicitly rescale here; BA handles it)

        # Update last pose
        last_pose = new_pose
        frame_poses[i] = new_pose

        # Periodic pruning
        if i % PRUNE_INTERVAL == 0 and len(map_points) > 100:
            _prune_map_points(map_points, feat_map, frame_to_points, reproj_thresh,
                              frame_poses, focal0, fy0, cx, cy)

    # ----- Handle uninitialized case -----
    if not initialized:
        raise RuntimeError(
            "Could not initialize SfM: no pair of frames with sufficient translation found. "
            "Ensure the video contains translational motion (not pure rotation)."
        )

    # ----- Final global BA -----
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

    # Final pruning and filtering
    _prune_map_points(map_points, feat_map, frame_to_points, reproj_thresh,
                      frame_poses, focal0, fy0, cx, cy)
    all_xyz, mask = _filter_point_cloud(map_points, frame_poses, focal0, fy0, cx, cy, reproj_thresh)
    all_xyz = all_xyz[mask] if np.any(mask) else all_xyz

    # Fill missing poses
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
    images, kps, descs = [], [], []
    # 增加特征数量以产生更多点云（默认5000已足够，可调至8000）
    if feature_type == "sift":
        detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.04, edgeThreshold=10)
    else:
        detector = cv2.ORB_create(nfeatures=8000, scaleFactor=1.2, nlevels=8)

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        images.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        kps.append(kp)
        descs.append(desc if desc is not None else np.zeros((0, 32), dtype=np.uint8))
    return images, kps, descs


def _match_features(desc1, desc2, bf, ratio=0.75):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    if len(desc2) < 2:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc1, desc2)
        return [m for m in matches if m.distance < MATCH_DIST]
    raw = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance and m.distance < MATCH_DIST:
            good.append(m)
    return good


# ---------- Geometry Helpers ----------
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


# ---------- Triangulation ----------
def _triangulate_new_points(curr_idx, matches, inlier_mask,
                            kp_list, pts_prev, pts_curr,
                            pose_prev, pose_curr,
                            focal, fy, cx, cy,
                            map_points, feat_map, desc_list, frame_to_points,
                            triang_thresh):
    K = np.array([[focal, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    P_prev = K @ pose_prev.RT[:3]
    P_curr = K @ pose_curr.RT[:3]
    cam_prev = -pose_prev.R.T @ pose_prev.t
    cam_curr = -pose_curr.R.T @ pose_curr.t

    for idx in np.where(inlier_mask)[0]:
        m = matches[idx]
        pt_prev = pts_prev[idx].reshape(2, 1)
        pt_curr = pts_curr[idx].reshape(2, 1)
        prev_feat = m.queryIdx
        curr_feat = m.trainIdx

        # Skip if already linked
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

    # Parallax
    v1 = pt3d - cam_prev.flatten()
    v2 = pt3d - cam_curr.flatten()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return False
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    if cos_angle > np.cos(np.radians(MIN_TRI_ANGLE_DEG)):
        return False

    # Reprojection
    proj_prev = P_prev @ np.append(pt3d, 1.0)
    proj_curr = P_curr @ np.append(pt3d, 1.0)
    proj_prev = proj_prev[:2] / (proj_prev[2] + 1e-12)
    proj_curr = proj_curr[:2] / (proj_curr[2] + 1e-12)
    if np.linalg.norm(proj_prev - pt_prev.flatten()) > reproj_th or \
       np.linalg.norm(proj_curr - pt_curr.flatten()) > reproj_th:
        return False
    return True


# ---------- Bundle Adjustment ----------
def _bundle_adjustment(
    keyframe_ids, map_points, feat_map, frame_poses,
    kp_list, focal, fy, cx, cy,
    optimize_points=True,
    reproj_thresh=1.0,
    image_size=1920,
    max_iter=BA_MAX_ITER,
    is_global=False,
):
    """
    Bundle adjustment for a set of keyframes.
    Returns updated focal, fy.
    """
    # ---------- FIX: Filter out invalid keyframes (pose is None) ----------
    valid_kfs = [f for f in keyframe_ids if f < len(frame_poses) and frame_poses[f] is not None]
    if len(valid_kfs) < 2:
        return focal, fy
    keyframe_ids = valid_kfs
    # ---------- End of fix ----------

    if len(keyframe_ids) < 2:
        return focal, fy

    # Collect observations
    obs = []
    for f_idx in keyframe_ids:
        for kp_idx, pt_idx in enumerate(feat_map[f_idx]):
            if pt_idx >= 0:
                u, v = kp_list[f_idx][kp_idx].pt
                obs.append((f_idx, pt_idx, u, v))

    if len(obs) < 10:
        return focal, fy

    # For global BA, limit points to avoid explosion
    if is_global and len(obs) > 5000:
        # Subsample observations (keep all frames, random points)
        np.random.seed(42)
        obs = np.array(obs, dtype=object)
        indices = np.random.choice(len(obs), 5000, replace=False)
        obs = obs[indices].tolist()

    # Determine fixed frame (most observations)
    obs_count = {f: 0 for f in keyframe_ids}
    for f, _, _, _ in obs:
        obs_count[f] += 1
    fixed_kf = max(obs_count, key=obs_count.get)
    other_kfs = [f for f in keyframe_ids if f != fixed_kf]

    # Build parameter vector: [focal, fy] + poses (6 per keyframe) + points (3 per point)
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
            # Keep points with most observations
            cnt = {pid: 0 for pid in point_ids}
            for _, pid, _, _ in obs:
                cnt[pid] += 1
            point_ids = sorted(cnt.keys(), key=lambda x: cnt[x], reverse=True)[:MAX_POINTS_IN_BA]
        for pid in point_ids:
            param.extend(map_points[pid]['xyz'])

    # Bounds
    bound_scale = 100.0 * image_size
    bounds_lower = [1.0, 1.0] + [-bound_scale] * len(other_kfs) * 6
    bounds_upper = [10.0 * image_size, 10.0 * image_size] + [bound_scale] * len(other_kfs) * 6
    if optimize_points and point_ids:
        bounds_lower += [-1e6] * len(point_ids) * 3
        bounds_upper += [1e6] * len(point_ids) * 3

    def residuals(params):
        f = params[0]
        fy_local = params[1]
        n_poses = len(other_kfs)
        # Reconstruct poses
        poses = {fixed_kf: frame_poses[fixed_kf]}
        for i, idx in enumerate(other_kfs):
            start = 2 + i * 6
            rv = params[start:start+3]
            t = params[start+3:start+6]
            R, _ = cv2.Rodrigues(rv)
            poses[idx] = CameraPose(R, t.reshape(3, 1))

        pts = {}
        if optimize_points and point_ids:
            pts_start = 2 + n_poses * 6
            for j, pid in enumerate(point_ids):
                pts[pid] = params[pts_start + j*3 : pts_start + j*3 + 3]
        else:
            for pid in set(pt for _, pt, _, _ in obs):
                pts[pid] = map_points[pid]['xyz']

        res = []
        for f_idx, pt_idx, u_obs, v_obs in obs:
            pt3d = pts[pt_idx]
            pose = poses[f_idx]
            pt_cam = pose.R @ pt3d.reshape(3, 1) + pose.t
            depth = float(pt_cam[2, 0])
            # Barrier for depth
            if depth < eps:
                barrier = -np.log(max(depth/eps, 1e-10))
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
        result = least_squares(
            residuals, np.array(param, dtype=np.float64),
            bounds=(bounds_lower, bounds_upper),
            method='trf', loss='soft_l1', f_scale=reproj_thresh,
            max_nfev=max_iter, verbose=0
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
            logger.info(f"BA: focal {focal:.2f}->{focal_new:.2f}, fy {fy:.2f}->{fy_new:.2f}")
            return focal_new, fy_new
        else:
            logger.warning("BA failed")
            return focal, fy
    except Exception as e:
        logger.warning(f"BA exception: {e}")
        return focal, fy


def _compute_adaptive_eps(map_points, default_eps=0.01):
    depths = [np.linalg.norm(p['xyz']) for p in map_points if np.isfinite(np.linalg.norm(p['xyz'])) and p['xyz'].size == 3]
    if depths:
        median = np.median(depths)
        eps = median * 0.001
        return np.clip(eps, EPS_MIN, EPS_MAX)
    return default_eps


# ---------- Pruning ----------
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
        for f_idx, kp_idx, u_obs, v_obs in pt['obs']:
            pose = frame_poses[f_idx]
            if pose is None:
                continue
            pt_cam = pose.R @ xyz.reshape(3, 1) + pose.t
            depth = float(pt_cam[2, 0])
            if depth <= 0:
                continue
            x = float(pt_cam[0, 0]) / depth
            y = float(pt_cam[1, 0]) / depth
            u_pred = focal * x + cx
            v_pred = fy * y + cy
            err = np.sqrt((u_pred - u_obs)**2 + (v_pred - v_obs)**2)
            total_err += err
            count += 1
        if count > 0 and total_err / count > MAX_REPROJ_ERROR * reproj_thresh:
            to_remove.append(idx)

    if not to_remove:
        return

    remove_set = set(to_remove)
    # Update feat_map
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

    # Rebuild frame_to_points
    for f in list(frame_to_points.keys()):
        frame_to_points[f] = set()
    # Remove points
    for old_idx in reversed(to_remove):
        del map_points[old_idx]
    # Update indices and frame_to_points
    for new_idx, pt in enumerate(map_points):
        pt['idx'] = new_idx
        for f_idx, _, _, _ in pt['obs']:
            frame_to_points[f_idx].add(new_idx)

    logger.debug(f"Pruned {len(to_remove)} points, remaining {len(map_points)}")


# ---------- Filtering ----------
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
        for f_idx, _, u_obs, v_obs in obs:
            pose = frame_poses[f_idx]
            if pose is None:
                continue
            pt_cam = pose.R @ xyz.reshape(3, 1) + pose.t
            depth = float(pt_cam[2, 0])
            if depth <= 0:
                continue
            x = float(pt_cam[0, 0]) / depth
            y = float(pt_cam[1, 0]) / depth
            u_pred = focal * x + cx
            v_pred = fy * y + cy
            err = np.sqrt((u_pred - u_obs)**2 + (v_pred - v_obs)**2)
            total_err += err
            count += 1
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