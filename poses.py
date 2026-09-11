"""
基于 ORB（或 SIFT）特征 + 增量式 SfM 的鲁棒相机位姿估计。

实现：
  - 特征提取（ORB / SIFT）
  - 本质矩阵 + PnP 重定位
  - 带共视关系的关键帧管理
  - 局部 / 全局光束法平差（BA，含地图点优化，外层套 GMM-EM 软内点加权）
  - 地图点修剪与过滤

不依赖 COLMAP，纯 OpenCV + SciPy。
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.special import expit

# ---------- 常量 ----------
MIN_INLIERS = 25                     # 有效内点下限
KEYFRAME_ANGLE_DEG = 5.0             # 关键帧旋转阈值（度）
KEYFRAME_TRANS_RATIO = 0.05          # 关键帧平移阈值（相对场景尺度）
COVIS_RATIO_THRESH = 0.25            # 共视比例阈值
BA_MAX_ITER = 15                     # 局部 BA 最大迭代
GLOBAL_BA_ITER = 25                  # 全局 BA 最大迭代
MIN_BA_WINDOW = 5                    # 触发局部 BA 所需的关键帧数
PRUNE_INTERVAL = 200                 # 地图点修剪间隔（帧）
MIN_OBSERVATIONS = 2                 # 地图点最少观测数
MAX_REPROJ_ERROR = 4.0               # 重投影误差上限（像素）
SMALL_TRANSLATION = 1e-4             # 判定“纯旋转”的平移阈值
MATCH_DIST = 90                      # ORB（Hamming）匹配距离上限
DESC_UPDATE_THRESH = 35              # ORB 描述子更新距离阈值
# SIFT 描述子是 float32，OpenCV 归一化到约 512 的范数，必须用 L2 度量；
# Hamming 只适用于 uint8 二进制描述子（ORB），对 float 使用会得到随机匹配
# 或在新版 OpenCV 直接断言崩溃。
SIFT_MATCH_DIST = 400.0              # SIFT（L2）匹配距离上限
SIFT_DESC_UPDATE_THRESH = 150.0      # SIFT 描述子更新距离阈值
MIN_FEATURES = 80                    # 单帧最少特征数
MIN_TRI_ANGLE_DEG = 2.0              # 三角化最小视差角
INIT_MIN_TRANSLATION = 0.01          # 判定“可用于初始化”的最小平移
KEYFRAME_CULLING_WINDOW = 10         # 关键帧剔除窗口
PNP_WINDOW = 12                      # PnP 使用最近多少关键帧
MAX_POINTS_IN_BA = 300               # BA 单次最多参与优化的点数
EPS_MIN = 1e-8                       # BA 深度障碍下限
EPS_MAX = 0.1                        # BA 深度障碍上限
BA_MAX_OBS = 2000                    # BA 观测上限
BA_F_SCALE_MULTIPLIER = 3.0          # soft_l1 核的 f_scale 倍数（BA_USE_EM=False 时使用）
SCALE_CLAMP = (0.5, 2.0)             # 尺度归一化裁剪区间
SCALE_DEADBAND = 0.05                # 尺度修正死区（避免抖动）

# ---------- EM-BA 相关 ----------
EM_ITERS = 3                         # 外层 EM 迭代次数
EM_PI_IN_INIT = 0.85                 # 内点比例初值
EM_PI_IN_MAX = 0.95                  # π_in 静态上限，自适应放宽
EM_SIGMA_CAP_RATIO = 1.5             # σ 上限 = reproj_thresh × ratio（自适应）
EM_SIGMA_OUT_RATIO = 8.0             # σ_out = reproj_thresh × ratio（自适应）
EM_GAMMA_FLOOR = 1e-4                # γ 下限，防止权重完全归零
FOCAL_MAX_STEP_RATIO = 0.05          # BA 单步焦距最大变化比例
FOCAL_GLOBAL_DRIFT_MAX = 0.15        # 焦距相对初始值的累计漂移上限
BA_MAX_INIT_RMS_RATIO = 3.0          # 初值中位 RMS 超过 reproj_thresh × ratio 则跳过
BA_MIN_OBS = 200                     # BA 最少观测数，低于此值跳过
BA_OVERFIT_MIN_RMS = 0.02            # BA 后中位 RMS 低于此值视为可疑（过拟合）
BA_OVERFIT_MIN_BEFORE = 0.15         # 且 BA 前中位 RMS 高于此值时才判过拟合
BA_MAX_RMS_INCREASE = 1.2            # BA 后中位 RMS 超过此倍数则视为变差并回滚
BA_USE_EM = True                     # 对比开关：False 走 soft_l1 基线

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Pose] %(message)s")


# ---------- 数据结构 ----------
@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

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

    @property
    def center(self) -> np.ndarray:
        """相机光心的世界坐标：C = -R^T @ t。"""
        return (-self.R.T @ self.t).flatten()


# ---------- 主入口 ----------
def estimate_poses(
    frame_paths: List[str],
    *,
    min_inliers: int = MIN_INLIERS,
    feature_type: str = "orb",
    focal_guess: Optional[float] = None,
    aspect_ratio: float = 1.0,
) -> Tuple[CameraIntrinsics, List[CameraPose], np.ndarray]:
    if not frame_paths:
        raise ValueError("frame_paths 不能为空")

    logger.info("加载图像并提取特征…")
    img_shape, kp_list, desc_list = _extract_features(frame_paths, feature_type)

    h, w = img_shape
    cx, cy = w / 2.0, h / 2.0
    focal0 = focal_guess if focal_guess is not None else max(w, h) * 1.2
    focal_init = float(focal0)       # 初始焦距，用于全局漂移保护（不随 BA 更新）
    fy0 = focal0 * aspect_ratio
    image_size = max(w, h)

    reproj_thresh = max(0.5, min(3.0, image_size * 0.0015))
    ransac_thresh = reproj_thresh * 0.8
    triang_thresh = reproj_thresh

    # 自适应 EM 参数：按 reproj_thresh 缩放，跨分辨率鲁棒
    sigma_cap = reproj_thresh * EM_SIGMA_CAP_RATIO
    sigma_out = reproj_thresh * EM_SIGMA_OUT_RATIO

    logger.info(f"阈值：reproj={reproj_thresh:.2f}, ransac={ransac_thresh:.2f}, "
                f"σ_cap={sigma_cap:.2f}, σ_out={sigma_out:.2f}")

    map_points: List[Dict] = []
    frame_poses: List[Optional[CameraPose]] = [None] * len(frame_paths)
    feat_map = [[-1] * len(kp) for kp in kp_list]
    frame_to_points: Dict[int, Set[int]] = defaultdict(set)

    frame_poses[0] = CameraPose(np.eye(3), np.zeros((3, 1)))
    keyframes = [0]
    last_pose = frame_poses[0]

    # 描述子度量由 dtype 决定（ORB→Hamming，SIFT→L2），预先算一次
    norm_type, match_dist, _ = _descriptor_metric(desc_list)
    bf = cv2.BFMatcher(norm_type, crossCheck=False)

    initialized = False
    init_candidates: List[int] = []   # 待配对的候选帧（纯旋转 / 平移不足）
    ba_counter = 0

    for i in range(1, len(frame_paths)):
        logger.info(f"处理帧 {i}/{len(frame_paths)-1}")

        if len(kp_list[i]) < MIN_FEATURES // 2:
            logger.warning(f"帧 {i} 特征过少，沿用上一帧位姿")
            frame_poses[i] = last_pose
            continue

        matches = _match_features(desc_list[i-1], desc_list[i], bf, match_dist=match_dist)
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
        if int(mask_pose.sum()) < min_inliers:
            frame_poses[i] = last_pose
            continue

        trans_norm = float(np.linalg.norm(t_rel))
        is_pure_rotation = trans_norm < SMALL_TRANSLATION

        R_curr = R_rel @ last_pose.R
        t_curr = R_rel @ last_pose.t + t_rel

        # ===== 初始化 =====
        if not initialized:
            can_init_now = (not is_pure_rotation) and (trans_norm > INIT_MIN_TRANSLATION)

            if can_init_now:
                norm_t = float(np.linalg.norm(t_curr))
                if norm_t > 1e-12:
                    # 以相邻帧（i-1 + i）直接初始化，平移归一到单位长度
                    t_curr = t_curr / norm_t
                    new_pose = CameraPose(R_curr, t_curr)
                    frame_poses[i] = new_pose
                    last_pose = new_pose
                    initialized = True
                    _, new_pose = _triangulate_new_points(
                        i, matches, mask_pose.ravel().astype(bool),
                        kp_list, pts_prev, pts_curr,
                        frame_poses[i-1], new_pose,
                        focal0, fy0, cx, cy,
                        map_points, feat_map, desc_list, frame_to_points,
                        triang_thresh)
                    frame_poses[i] = new_pose
                    last_pose = new_pose
                    keyframes.append(i)
                    logger.info(f"初始化成功（帧 {i-1} + {i}）")
                    continue

            # 纯旋转 / 平移不足：登记为候选，等后续出现足够基线的帧再配对初始化
            init_candidates.append(i)

            if len(init_candidates) >= 2:
                for j in range(len(init_candidates) - 1, -1, -1):
                    idx_cand = init_candidates[j]
                    if idx_cand <= 0 or idx_cand >= i:
                        continue

                    matches_cand = _match_features(
                        desc_list[idx_cand], desc_list[i], bf, match_dist=match_dist)
                    if len(matches_cand) < min_inliers:
                        continue

                    pts_cand = np.array([kp_list[idx_cand][m.queryIdx].pt for m in matches_cand],
                                        dtype=np.float32)
                    pts_cur2 = np.array([kp_list[i][m.trainIdx].pt for m in matches_cand],
                                        dtype=np.float32)

                    E2, mask2 = cv2.findEssentialMat(
                        pts_cand, pts_cur2, focal=focal0, pp=(cx, cy),
                        method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
                    if E2 is None or mask2.sum() < min_inliers:
                        continue

                    _, R_rel2, t_rel2, mask_pose2 = cv2.recoverPose(
                        E2, pts_cand, pts_cur2, focal=focal0, pp=(cx, cy), mask=mask2)
                    if mask_pose2.sum() < min_inliers or np.linalg.norm(t_rel2) <= INIT_MIN_TRANSLATION:
                        continue

                    R_curr2 = R_rel2 @ frame_poses[idx_cand].R
                    t_curr2 = R_rel2 @ frame_poses[idx_cand].t + t_rel2
                    norm_t2 = float(np.linalg.norm(t_curr2))
                    if norm_t2 <= 1e-12:
                        continue
                    t_curr2 = t_curr2 / norm_t2

                    new_pose = CameraPose(R_curr2, t_curr2)
                    frame_poses[i] = new_pose
                    last_pose = new_pose
                    initialized = True
                    _, new_pose = _triangulate_new_points(
                        i, matches_cand, mask_pose2.ravel().astype(bool),
                        kp_list, pts_cand, pts_cur2,
                        frame_poses[idx_cand], new_pose,
                        focal0, fy0, cx, cy,
                        map_points, feat_map, desc_list, frame_to_points,
                        triang_thresh)
                    frame_poses[i] = new_pose
                    last_pose = new_pose
                    keyframes.append(i)
                    logger.info(f"初始化成功（帧 {idx_cand} + {i}）")
                    break

            if initialized:
                continue

            frame_poses[i] = last_pose
            continue

        # ===== 已初始化，正常增量推进 =====
        new_pose = CameraPose(R_curr, t_curr)
        inlier_mask = mask_pose.ravel().astype(bool)

        _, new_pose = _triangulate_new_points(
            i, matches, inlier_mask,
            kp_list, pts_prev, pts_curr,
            frame_poses[i-1], new_pose,
            focal0, fy0, cx, cy,
            map_points, feat_map, desc_list, frame_to_points,
            triang_thresh)

        # ----- 局部地图 PnP 重定位 -----
        # PnP 用已知 3D 地图点恢复位姿，天然提供正确的场景尺度，
        # 是抑制纯 E 矩阵链式累积尺度漂移的关键。
        # 只用最近 PNP_WINDOW 个关键帧（全量匹配太慢，且旧关键帧增益有限）。
        if len(keyframes) > 1:
            pts3d_local, pts2d_local = [], []
            for kf in keyframes[-PNP_WINDOW:]:
                matches_kf = _match_features(desc_list[kf], desc_list[i], bf,
                                             match_dist=match_dist)
                for m in matches_kf:
                    pt_idx = feat_map[kf][m.queryIdx]
                    if pt_idx >= 0:
                        pts3d_local.append(map_points[pt_idx]['xyz'])
                        pts2d_local.append(kp_list[i][m.trainIdx].pt)
            if len(pts3d_local) >= 8:
                pts3d_local = np.array(pts3d_local, dtype=np.float32)
                pts2d_local = np.array(pts2d_local, dtype=np.float32)
                K_pnp = np.array([[focal0, 0, cx],
                                  [0, fy0, cy],
                                  [0, 0, 1]], dtype=np.float32)
                _, rvec_pnp, tvec_pnp, inliers_pnp = cv2.solvePnPRansac(
                    pts3d_local, pts2d_local, K_pnp, np.zeros(4),
                    iterationsCount=50, reprojectionError=reproj_thresh, confidence=0.95
                )
                if inliers_pnp is not None and len(inliers_pnp) >= 8:
                    R_pnp, _ = cv2.Rodrigues(rvec_pnp)
                    t_pnp = tvec_pnp.reshape(3, 1)
                    depth_median = float(np.median([np.linalg.norm(p) for p in pts3d_local]))
                    # 拒绝明显远离场景的 PnP 解（防尺度漂移把相机推到无穷远）
                    if np.linalg.norm(t_pnp) < 10.0 * depth_median:
                        new_pose = CameraPose(R_pnp, t_pnp)

        # ----- 关键帧判定 -----
        angle = _compute_rotation_angle(R_rel)
        pt_norms = [np.linalg.norm(p['xyz']) for p in map_points]
        scene_ref = np.median(pt_norms) + 1e-6 if pt_norms else 1e-6
        trans_world = np.linalg.norm(t_curr - last_pose.t) / scene_ref
        covis_ratio = _compute_covisibility_ratio(i, keyframes, matches, feat_map, frame_to_points)

        is_keyframe = (angle > KEYFRAME_ANGLE_DEG or
                       trans_world > KEYFRAME_TRANS_RATIO or
                       covis_ratio < COVIS_RATIO_THRESH)

        if is_keyframe and not is_pure_rotation and len(map_points) > 20:
            # 冗余关键帧剔除：窗口内与当前帧共视过高则替换
            if len(keyframes) > KEYFRAME_CULLING_WINDOW:
                recent = keyframes[-KEYFRAME_CULLING_WINDOW:-1]
                for kf in recent:
                    if i in frame_to_points and kf in frame_to_points:
                        covis = len(frame_to_points[i] & frame_to_points[kf]) / max(
                            1, min(len(frame_to_points[i]), len(frame_to_points[kf])))
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
                    is_global=False,
                    focal_init=focal_init,
                )

        last_pose = new_pose
        frame_poses[i] = new_pose

        if i % PRUNE_INTERVAL == 0 and len(map_points) > 100:
            _prune_map_points(map_points, feat_map, frame_to_points, reproj_thresh,
                              frame_poses, focal0, fy0, cx, cy)

    if not initialized:
        raise RuntimeError(
            "无法初始化 SfM：未找到具有足够平移的帧对。"
        )

    if len(keyframes) >= 3 and len(map_points) > 50:
        logger.info("对所有关键帧执行全局 BA…")
        focal0, fy0 = _bundle_adjustment(
            keyframes, map_points, feat_map, frame_poses,
            kp_list, focal0, fy0, cx, cy,
            optimize_points=True,
            reproj_thresh=reproj_thresh,
            image_size=image_size,
            max_iter=GLOBAL_BA_ITER,
            is_global=True,
            focal_init=focal_init,
        )

    _prune_map_points(map_points, feat_map, frame_to_points, reproj_thresh,
                      frame_poses, focal0, fy0, cx, cy)
    all_xyz, mask = _filter_point_cloud(map_points, frame_poses, focal0, fy0, cx, cy, reproj_thresh)
    all_xyz = all_xyz[mask] if np.any(mask) else all_xyz

    # 未成功估计位姿的帧，用上一个有效位姿补齐
    last_valid = frame_poses[0]
    for i, p in enumerate(frame_poses):
        if p is None:
            frame_poses[i] = last_valid
        else:
            last_valid = p

    intrinsics = CameraIntrinsics(fx=focal0, fy=fy0, cx=cx, cy=cy)
    logger.info(f"结束：fx={focal0:.2f}, fy={fy0:.2f}, 点数={len(all_xyz)}")
    return intrinsics, frame_poses, all_xyz


# ---------- 特征提取 ----------
def _extract_features(paths: List[str], feature_type: str = "orb"):
    """逐帧提取特征。不把整幅原图常驻内存（200 帧 1080p 约 1.2GB）。"""
    kps, descs = [], []
    if feature_type == "sift":
        try:
            detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.03,
                                       edgeThreshold=10, sigma=1.6)
        except cv2.error as e:
            logger.warning(f"SIFT 不可用（{e}），回退到 ORB。")
            detector = cv2.ORB_create(nfeatures=12000, scaleFactor=1.2, nlevels=8,
                                      edgeThreshold=31, patchSize=31)
    else:
        detector = cv2.ORB_create(nfeatures=12000, scaleFactor=1.2, nlevels=8,
                                  edgeThreshold=31, patchSize=31)

    shape0 = None
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{p}")
        if shape0 is None:
            shape0 = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        kps.append(kp)
        descs.append(desc if desc is not None else np.zeros((0, 32), dtype=np.uint8))
        del img, gray
    return shape0, kps, descs


# ---------- 描述子度量 ----------
def _descriptor_metric(desc_list):
    """按实际描述子 dtype 返回 (cv2 范数类型, 匹配距离上限, 描述子更新阈值)。

    ORB 描述子为 uint8 二进制 → Hamming；
    SIFT 描述子为 float32 → L2。
    对 float 描述子用 Hamming 会得到随机匹配或直接断言崩溃。
    空描述子列表按 ORB/Hamming 处理（空数组不会触发匹配）。
    """
    for d in desc_list:
        if d is not None and len(d) > 0:
            if d.dtype != np.uint8:
                return cv2.NORM_L2, SIFT_MATCH_DIST, SIFT_DESC_UPDATE_THRESH
            return cv2.NORM_HAMMING, MATCH_DIST, DESC_UPDATE_THRESH
    return cv2.NORM_HAMMING, MATCH_DIST, DESC_UPDATE_THRESH


# ---------- 特征匹配 ----------
def _match_features(desc1, desc2, bf, ratio=0.75, match_dist=None):
    """Lowe 比值测试 + 距离上限过滤。

    match_dist 可由调用方预先算好并传入；未提供时按 desc1 的 dtype 现算。
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    if match_dist is None:
        _, match_dist, _ = _descriptor_metric([desc1])

    norm_type, _, _ = _descriptor_metric([desc1])
    if len(desc2) < 2:
        # 太少描述子无法做 kNN，退回 crossCheck 暴力匹配
        matches = cv2.BFMatcher(norm_type, crossCheck=True).match(desc1, desc2)
        return [m for m in matches if m.distance < match_dist]

    raw = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and m.distance < match_dist:
            good.append(m)
    return good


# ---------- 辅助函数 ----------
def _compute_rotation_angle(R):
    rv, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(rv) * 180.0 / np.pi)


def _compute_covisibility_ratio(curr_idx, keyframes, matches, feat_map, frame_to_points):
    """当前帧匹配与最近关键帧地图点的共视比例。"""
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
        pt_dict.setdefault('obs', []).append(
            (frame_idx, kp_idx, float(uv[0]), float(uv[1])))
        pt_dict['obs_count'] = pt_dict.get('obs_count', 0) + 1
        pt_idx = pt_dict.get('idx', -1)
        if pt_idx >= 0:
            frame_to_points[frame_idx].add(pt_idx)


def _update_map_point_descriptor(pt_dict, new_desc):
    """按观测数和描述子年龄决定是否替换地图点描述子。"""
    old = pt_dict.get('desc')
    if old is None:
        pt_dict['desc'] = new_desc.copy()
        pt_dict['desc_age'] = 0
        return
    norm_type, _, update_thresh = _descriptor_metric([new_desc])
    dist = cv2.norm(old, new_desc, norm_type)
    age = pt_dict.get('desc_age', 0)
    # 观测少 / 描述子过旧 / 差异过大 → 替换；否则累积年龄
    if pt_dict.get('obs_count', 0) < 3 or age > 5 or dist >= update_thresh * 1.2:
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
    """对当前帧的内点做三角化，新增或复用地图点，并做尺度归一化。"""
    K = np.array([[focal, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    P_prev = K @ pose_prev.RT[:3]
    P_curr = K @ pose_curr.RT[:3]
    new_indices = []
    cam_prev = pose_prev.center.reshape(3, 1)
    cam_curr = pose_curr.center.reshape(3, 1)

    for idx in np.where(inlier_mask)[0]:
        m = matches[idx]
        pt_prev = pts_prev[idx].reshape(2, 1)
        pt_curr = pts_curr[idx].reshape(2, 1)
        prev_feat = m.queryIdx
        curr_feat = m.trainIdx

        if feat_map[curr_idx][curr_feat] >= 0:
            continue

        # 上一帧已有对应地图点 → 只添加观测，不重复三角化
        existing = feat_map[curr_idx - 1][prev_feat]
        if existing >= 0:
            _add_observation(map_points[existing], curr_idx, curr_feat,
                             pt_curr.flatten(), frame_to_points)
            feat_map[curr_idx][curr_feat] = existing
            _update_map_point_descriptor(map_points[existing], desc_list[curr_idx][curr_feat])
            continue

        pts4d = cv2.triangulatePoints(P_prev, P_curr, pt_prev, pt_curr)
        pt3d = (pts4d[:3] / (float(pts4d[3][0]) + 1e-12)).flatten()

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

    # ===== 尺度归一化 =====
    # 每次本质矩阵恢复的 t_rel 是单位范数（up-to-scale），链式叠加
    #   t_curr = R_rel @ last_pose.t + t_rel
    # 会导致尺度随帧数累积漂移。这里把“本次新增点的相机 z 向深度中位数”
    # 对齐到“已有地图点的相机 z 向深度中位数”，并同步缩放当前相机位移。
    #
    # 注意两点（此前的两个 bug）：
    # 1) 缩放必须相对上一帧相机中心，而不是世界原点；否则越到后段，
    #    相对位移被放大得越离谱。
    # 2) 判断“是否需要修正”必须在裁剪之前，否则 <0.5 的负向修正永远被
    #    裁剪成 0.5 且 <0.5 分支永远不成立，等于禁用了负向修正。
    if new_indices and map_points:
        old_count = len(map_points) - len(new_indices)
        if old_count >= 10:
            cam_curr_flat = cam_curr.flatten()
            old_depths = []
            for p in map_points[:old_count]:
                if p['xyz'].size == 3:
                    d = float(pose_curr.R[2] @ (p['xyz'] - cam_curr_flat))
                    if d > 0:
                        old_depths.append(d)
            if len(old_depths) >= 5:
                ref_median = float(np.median(old_depths))
                new_depths = []
                for pi in new_indices:
                    d = float(pose_curr.R[2] @ (map_points[pi]['xyz'] - cam_curr_flat))
                    if d > 0:
                        new_depths.append(d)
                if new_depths:
                    new_median = float(np.median(new_depths))
                    if new_median > 1e-8 and ref_median > 1e-8:
                        raw_scale = ref_median / new_median
                        # 先判断死区，再裁剪到安全区间
                        if abs(raw_scale - 1.0) > SCALE_DEADBAND:
                            scale = float(np.clip(raw_scale, *SCALE_CLAMP))
                            # 相对上一帧相机中心缩放：新点和当前相机位移一起缩，
                            # 上一帧保持不动。
                            origin = pose_prev.center
                            for pi in new_indices:
                                xyz = map_points[pi]['xyz']
                                map_points[pi]['xyz'] = (
                                    origin + scale * (xyz - origin)
                                ).astype(np.float32)
                            cam_curr_new = origin + scale * (cam_curr_flat - origin)
                            # 由相机中心反解 t：t = -R @ C
                            t_new = (-pose_curr.R @ cam_curr_new.reshape(3, 1))
                            pose_curr = CameraPose(
                                pose_curr.R, t_new.astype(np.float32))

    return new_indices, pose_curr


def _is_valid_point(pt3d, pose_prev, pose_curr, cam_prev, cam_curr,
                    reproj_th, P_prev, P_curr, pt_prev, pt_curr):
    """三角化点的合法性检查：有限性、正深度、视差角、重投影误差。"""
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
    cos_angle = float(np.dot(v1, v2) / (n1 * n2))
    if cos_angle > np.cos(np.radians(MIN_TRI_ANGLE_DEG)):
        return False

    proj_prev = P_prev @ np.append(pt3d, 1.0)
    proj_curr = P_curr @ np.append(pt3d, 1.0)
    proj_prev = proj_prev[:2] / (proj_prev[2] + 1e-12)
    proj_curr = proj_curr[:2] / (proj_curr[2] + 1e-12)
    if (np.linalg.norm(proj_prev - pt_prev.flatten()) > reproj_th or
            np.linalg.norm(proj_curr - pt_curr.flatten()) > reproj_th):
        return False
    return True


# ---------- EM-BA 辅助函数 ----------
def _em_e_step(res, obs_depths, sigma, pi_in, sigma_out, eps):
    """EM 的 E 步：计算每个观测属于内点的后验概率 γ。

    模型：2 维残差 r 来自混合高斯
        p(r) = π_in · N(0, σ²I) + (1-π_in) · N(0, σ_out²I)
    γ = p(z_in | r)，即后验内点概率。

    关键公式（2 维高斯）：
        log p(r|in) - log p(r|out)
          = 2·log(σ_out/σ) + r²·(1/(2σ_out²) - 1/(2σ²))
    注意 r² 项的符号为负——残差越大，越倾向外点。
    之前版本这里符号写反，导致 γ 随残差增大而升高，最终塌缩到 1。

    深度不足（depth ≤ eps）的观测强制 γ=1：其残差是深度障碍项，
    不是真实重投影误差，不应参与外点判定。
    """
    n_obs = len(obs_depths)
    r2 = np.empty(n_obs, dtype=np.float64)
    for i in range(n_obs):
        r2[i] = res[2 * i] ** 2 + res[2 * i + 1] ** 2

    log_ratio = (np.log(pi_in / max(1.0 - pi_in, 1e-12))
                 + 2.0 * np.log(sigma_out / sigma)
                 + r2 * (1.0 / (2.0 * sigma_out ** 2) - 1.0 / (2.0 * sigma ** 2)))
    # 用 expit 代替手写 sigmoid：log_ratio 很负时 exp 会溢出，
    # expit 按符号分段计算，永不溢出。
    gamma = expit(log_ratio)

    invalid = obs_depths <= eps
    gamma[invalid] = 1.0
    r2[invalid] = 0.0

    return gamma, r2


def _em_m_step(r2, gamma, sigma_cap):
    """EM 的 M 步：加权重估 σ 和 π_in。

    σ 只从高置信内点（γ > 0.9）估计。若用全部 γ 加权，大残差观测
    即使 γ 只有 0.5 也会贡献一半权重，把 σ 拉大；σ 变大后 γ 又降不下来，
    形成“塌缩”正反馈。用高置信子集估计可打破该循环。

    π_in 上限根据当前 γ 分布动态调整：如果高置信内点占比很高，
    说明当前场景匹配质量好，允许 π_in 接近 1。

    r² 是 2 维残差平方和，E[r²] = 2σ²。
    """
    # 自适应 π_in 上限：高置信内点占比高时放宽上限
    high_conf_ratio = float((gamma > 0.9).mean())
    if high_conf_ratio > 0.8:
        pi_in_max = min(EM_PI_IN_MAX + 0.04, 0.99)
    elif high_conf_ratio > 0.6:
        pi_in_max = EM_PI_IN_MAX
    else:
        pi_in_max = max(EM_PI_IN_MAX - 0.05, 0.7)
    pi_in_new = float(np.clip(np.mean(gamma), 0.05, pi_in_max))

    high_conf = gamma > 0.9
    if high_conf.sum() >= 5:
        sigma2 = float(np.sum(r2[high_conf]) / (2.0 * high_conf.sum()))
    else:
        denom = 2.0 * (np.sum(gamma) + 1e-12)
        sigma2 = float(np.sum(gamma * r2) / denom)

    sigma_new = float(np.clip(np.sqrt(max(sigma2, 1e-6)), 0.1, sigma_cap))
    return sigma_new, pi_in_new


def _compute_adaptive_eps(map_points, ref_pose, default_eps=0.01):
    """按参考相机下的 z 向深度中位数确定深度障碍阈值。

    先前版本用 ||p||（到世界原点的距离）近似，相机远离原点后失真严重。
    """
    if ref_pose is None:
        return default_eps
    cam_center = ref_pose.center
    R = ref_pose.R
    depths = []
    for p in map_points:
        xyz = p['xyz']
        if xyz.size == 3 and np.isfinite(xyz).all():
            d = float(R[2] @ (xyz - cam_center))
            if d > 0:
                depths.append(d)
    if depths:
        median = float(np.median(depths))
        return float(np.clip(median * 0.001, EPS_MIN, EPS_MAX))
    return default_eps


# ---------- 鲁棒光束法平差（EM 加权） ----------
def _bundle_adjustment(
    keyframe_ids, map_points, feat_map, frame_poses,
    kp_list, focal, fy, cx, cy,
    optimize_points=True,
    reproj_thresh=1.0,
    image_size=1920,
    max_iter=BA_MAX_ITER,
    is_global=False,
    focal_init=None,
):
    """局部 / 全局 BA，外层套 GMM-EM 软内点加权。

    参数化：[focal, (rv_kf, t_kf) * N_other_kf, (xyz_pt) * N_pts]
    焦距 fy 锁定为 focal * fy_ratio，避免与场景尺度耦合导致病态。

    观测加权：每个观测先按混合高斯模型估计内点后验概率 γ，再用
    sqrt(γ) 加权跑最小二乘（loss='linear'，鲁棒性完全由 γ 提供）。
    深度不足的观测强制 γ=1，其残差（深度障碍项）保留完整权重。

    防护机制：
    - BA 最小观测门槛：观测数 < BA_MIN_OBS 时跳过
    - BA 初值门槛：中位 RMS > reproj_thresh × BA_MAX_INIT_RMS_RATIO 时跳过
    - 焦距全局漂移保护：焦距相对 focal_init 最多漂移 ±FOCAL_GLOBAL_DRIFT_MAX
    - BA 后过拟合/变差检测：中位 RMS 异常低或异常升高时回滚

    BA_USE_EM = False 时退化为单次 soft_l1 基线，用于对比。
    """
    if focal_init is None:
        focal_init = focal

    # 自适应 EM 参数
    sigma_cap = reproj_thresh * EM_SIGMA_CAP_RATIO
    sigma_out = reproj_thresh * EM_SIGMA_OUT_RATIO

    # ---------- 1. 数据准备 ----------
    valid_kfs = [f for f in keyframe_ids
                 if f < len(frame_poses) and frame_poses[f] is not None]
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

    # BA 最小观测门槛：观测太少时强行 BA 容易过拟合重投影误差
    if len(obs) < BA_MIN_OBS:
        logger.info(f"[BA] 观测数 {len(obs)} < {BA_MIN_OBS}，跳过本次 BA")
        return focal, fy

    n_obs_orig = len(obs)
    rng = np.random.default_rng(42)
    if n_obs_orig > BA_MAX_OBS:
        idx = rng.choice(n_obs_orig, BA_MAX_OBS, replace=False)
        obs = [obs[i] for i in idx]
        logger.info(f"[BA] 抽样 {BA_MAX_OBS} / {n_obs_orig} 个观测")

    # 固定观测最多的关键帧作为基准
    obs_count = {f: 0 for f in keyframe_ids}
    for f, _, _, _ in obs:
        obs_count[f] += 1
    fixed_kf = max(obs_count, key=obs_count.get)
    other_kfs = [f for f in keyframe_ids if f != fixed_kf]

    # ---------- 2. 参数与边界 ----------
    fy_ratio = (fy / focal) if focal > 0 else 1.0
    eps = _compute_adaptive_eps(map_points, frame_poses[fixed_kf])

    param = [float(focal)]
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
            point_ids = sorted(cnt.keys(), key=lambda x: cnt[x],
                               reverse=True)[:MAX_POINTS_IN_BA]
        for pid in point_ids:
            param.extend(map_points[pid]['xyz'])

    scene_scale = 1.0
    if map_points:
        xs = np.array([p['xyz'] for p in map_points if p['xyz'].size == 3])
        if len(xs) > 0:
            scene_scale = float(np.linalg.norm(xs.max(axis=0) - xs.min(axis=0))) + 1e-6
    cam_ts = [np.linalg.norm(frame_poses[k].t.flatten()) for k in other_kfs
              if frame_poses[k] is not None]
    cam_t_max = max(cam_ts) if cam_ts else 1.0
    t_bound = max(scene_scale * 5.0, cam_t_max * 2.0, 10.0)
    r_bound = np.inf   # 旋转向量范数可近 4π，任何有限边界都可能触发初值越界

    # 焦距边界：单步 ±5% 与相对初始值 ±15% 取交集
    focal_lo = max(focal * (1.0 - FOCAL_MAX_STEP_RATIO),
                   focal_init * (1.0 - FOCAL_GLOBAL_DRIFT_MAX))
    focal_hi = min(focal * (1.0 + FOCAL_MAX_STEP_RATIO),
                   focal_init * (1.0 + FOCAL_GLOBAL_DRIFT_MAX))

    lower_pose, upper_pose = [], []
    for _kf in other_kfs:
        lower_pose += [-r_bound] * 3 + [-t_bound] * 3
        upper_pose += [r_bound] * 3 + [t_bound] * 3
    bounds_lower = [focal_lo] + lower_pose
    bounds_upper = [focal_hi] + upper_pose
    if optimize_points and point_ids:
        bounds_lower += [-np.inf] * len(point_ids) * 3
        bounds_upper += [np.inf] * len(point_ids) * 3

    # ---------- 3. 残差闭包 ----------
    n_obs = len(obs)

    def _compute_residuals(params):
        """返回 (res, obs_depths)。

        res 长度 2 * n_obs（每个观测 2 个残差）；
        obs_depths 长度 n_obs，供 EM 的 E 步判断哪些观测深度不足。
        """
        f = params[0]
        fy_local = f * fy_ratio
        n_poses = len(other_kfs)

        poses = {fixed_kf: frame_poses[fixed_kf]}
        for i_kf, idx in enumerate(other_kfs):
            start = 1 + i_kf * 6
            rv = params[start:start + 3]
            t = params[start + 3:start + 6]
            R, _ = cv2.Rodrigues(rv)
            poses[idx] = CameraPose(R, t.reshape(3, 1))

        pts = {}
        for _, pid, _, _ in obs:
            pts[pid] = map_points[pid]['xyz']
        if optimize_points and point_ids:
            pts_start = 1 + n_poses * 6
            for j, pid in enumerate(point_ids):
                pts[pid] = params[pts_start + j * 3: pts_start + j * 3 + 3]

        res = []
        obs_depths = np.zeros(n_obs, dtype=np.float64)
        for obs_i, (f_idx, pt_idx, u_obs, v_obs) in enumerate(obs):
            pt3d = pts[pt_idx]
            pose = poses[f_idx]
            pt_cam = pose.R @ pt3d.reshape(3, 1) + pose.t
            depth = float(pt_cam[2, 0])
            obs_depths[obs_i] = depth

            if depth <= eps:
                # 深度障碍：对深度过浅或为负的观测施加对数惩罚。
                # 这些观测在 E 步会被强制 γ=1，不参与外点判定。
                if depth <= 1e-6:
                    barrier = -np.log(max(depth / 1e-6, 1e-10))
                else:
                    barrier = -np.log(max(depth / eps, 1e-10))
                res.append(barrier)
                res.append(barrier)
                continue

            x = float(pt_cam[0, 0]) / depth
            y = float(pt_cam[1, 0]) / depth
            res.append(f * x + cx - u_obs)
            res.append(fy_local * y + cy - v_obs)

        return np.array(res, dtype=np.float64), obs_depths

    # ---------- 4. 优化 ----------
    param_np = np.array(param, dtype=np.float64)
    res0, depths0 = _compute_residuals(param_np)
    r2_0 = np.sum(res0.reshape(n_obs, 2) ** 2, axis=1)
    valid_mask = depths0 > eps
    # 初始 RMS 用中位数估计：残差通常重尾，少数野点会把平均值抬得很高，
    # 中位数更能反映"典型观测"的拟合质量。
    rms_before = (float(np.sqrt(np.median(r2_0[valid_mask]) / 2))
                  if valid_mask.sum() > 0 else 0.0)

    # BA 初值门槛：初值太差时 BA 只会越优化越糟，甚至把焦距拉偏
    init_rms_limit = reproj_thresh * BA_MAX_INIT_RMS_RATIO
    if rms_before > init_rms_limit:
        logger.warning(
            f"[BA] 初值过差（中位 RMS={rms_before:.2f}px > 限 {init_rms_limit:.2f}px），"
            f"跳过本次 BA 以免破坏位姿和焦距"
        )
        return focal, fy

    # 保存 BA 前状态，用于 BA 后过拟合/变差检测的回滚
    pre_ba_poses = {idx: frame_poses[idx] for idx in other_kfs}
    pre_ba_points_xyz = {pid: map_points[pid]['xyz'].copy() for pid in point_ids}
    pre_ba_focal = float(param_np[0])
    pre_ba_fy = float(param_np[0]) * fy_ratio

    result = None
    gamma = np.ones(n_obs, dtype=np.float64)

    if not BA_USE_EM:
        # ===== 对比基线：单次 soft_l1 =====
        logger.info(f"[BA] EM 已禁用，走 soft_l1 基线："
                    f"{len(keyframe_ids)} 关键帧，{len(obs)} 观测，"
                    f"初始中位 RMS={rms_before:.3f}px")
        try:
            result = least_squares(
                lambda p: _compute_residuals(p)[0], param_np,
                bounds=(bounds_lower, bounds_upper),
                method='trf', loss='soft_l1',
                f_scale=reproj_thresh * BA_F_SCALE_MULTIPLIER,
                max_nfev=max_iter, verbose=0,
                ftol=1e-4, xtol=1e-4, gtol=1e-4
            )
            param_np = result.x
        except Exception as e:
            logger.warning(f"[BA] 基线优化异常：{e}")
            return focal, fy
    else:
        # ===== EM 加权 =====
        if valid_mask.sum() > 0:
            sigma = max(float(np.sqrt(np.median(r2_0[valid_mask]) / 2.0)), 0.5)
        else:
            sigma = reproj_thresh
        pi_in = EM_PI_IN_INIT

        logger.info(
            f"[BA-EM] 开始：{len(keyframe_ids)} 关键帧，{len(obs)} 观测，"
            f"深度不足={int((~valid_mask).sum())}，"
            f"σ₀={sigma:.3f}px, π_in₀={pi_in:.3f}, "
            f"σ_cap={sigma_cap:.2f}px, σ_out={sigma_out:.2f}px, "
            f"初始中位 RMS={rms_before:.3f}px"
        )

        for em_iter in range(EM_ITERS):
            # ----- E 步：用当前 σ/π_in 算每个观测的内点后验概率 -----
            gamma, r2 = _em_e_step(res0, depths0, sigma, pi_in,
                                   sigma_out, eps)
            gamma = np.maximum(gamma, EM_GAMMA_FLOOR)

            valid_now = depths0 > eps
            if valid_now.sum() > 0:
                gamma_valid = gamma[valid_now]
                inlier_high = float((gamma_valid > 0.5).mean())
                inlier_mean = float(gamma_valid.mean())
                rms_now = float(np.sqrt(np.median(r2[valid_now]) / 2.0))
            else:
                inlier_high = inlier_mean = rms_now = 0.0

            logger.info(
                f"[BA-EM] EM {em_iter + 1}/{EM_ITERS} E步："
                f"π_in={pi_in:.3f}, σ={sigma:.3f}, "
                f"γ>0.5 占比={inlier_high:.3f}, γ 均值={inlier_mean:.3f}, "
                f"中位残差={rms_now:.3f}px"
            )

            # ----- M 步：加权最小二乘 -----
            sqrt_gamma = np.sqrt(gamma)

            def _weighted_residuals(params, _sg=sqrt_gamma):
                res_plain, _ = _compute_residuals(params)
                return res_plain * np.repeat(_sg, 2)

            # 第一轮给完整预算，后续轮起点已接近最优，减半
            iters_this = max_iter if em_iter == 0 else max(3, max_iter // 2)

            try:
                result = least_squares(
                    _weighted_residuals, param_np,
                    bounds=(bounds_lower, bounds_upper),
                    method='trf', loss='linear',
                    max_nfev=iters_this, verbose=0,
                    ftol=1e-4, xtol=1e-4, gtol=1e-4
                )
                param_np = result.x
            except Exception as e:
                logger.warning(f"[BA-EM] M 步异常：{e}")
                break

            # ----- 更新 σ 和 π_in（用新残差 + 当前 γ） -----
            res0, depths0 = _compute_residuals(param_np)
            r2 = np.sum(res0.reshape(n_obs, 2) ** 2, axis=1)
            valid_now = depths0 > eps
            if valid_now.sum() > 0:
                sigma, pi_in = _em_m_step(r2[valid_now], gamma[valid_now],
                                          sigma_cap)
                # 分开报告两种 RMS：
                # - inlier_rms：只统计 γ>0.5 的观测，反映"内点拟合质量"
                # - weighted_rms：所有观测按 γ 加权的 RMS，反映"整体优化目标"
                # 未加权平均 RMS 会被少数野点严重拉高，不适合用作健康指标。
                in_mask = gamma[valid_now] > 0.5
                if in_mask.sum() > 0:
                    inlier_rms = float(np.sqrt(
                        np.mean(r2[valid_now][in_mask]) / 2))
                else:
                    inlier_rms = 0.0
                weighted_rms = float(np.sqrt(
                    np.mean(gamma[valid_now] * r2[valid_now]) / 2))
            else:
                inlier_rms = weighted_rms = 0.0

            logger.info(
                f"[BA-EM] EM {em_iter + 1}/{EM_ITERS} M步："
                f"cost={result.cost:.1f}, "
                f"新 σ={sigma:.3f}, 新 π_in={pi_in:.3f}, "
                f"内点 RMS={inlier_rms:.3f}px, 加权 RMS={weighted_rms:.3f}px"
            )

    # ---------- 5. 结果提取 ----------
    if result is None:
        logger.warning("[BA] 无有效优化结果，返回原参数")
        return focal, fy

    focal_new = max(float(param_np[0]), 1.0)
    fy_new = focal_new * fy_ratio

    # BA 后统计：先用当前参数临时计算，用于过拟合/变差检测
    res_final, depths_final = _compute_residuals(param_np)
    r2_final = np.sum(res_final.reshape(n_obs, 2) ** 2, axis=1)
    valid_final = depths_final > eps
    if valid_final.sum() > 0:
        rms_after = float(np.sqrt(np.median(r2_final[valid_final]) / 2.0))
    else:
        rms_after = 0.0

    # 过拟合检测：BA 后中位 RMS 异常低（真实匹配噪声不可能低于 0.02px），
    # 或 BA 后 RMS 反而升高（BA 破坏了结果），都回滚。
    overfit = (rms_after < BA_OVERFIT_MIN_RMS and rms_before > BA_OVERFIT_MIN_BEFORE)
    worse = (rms_after > rms_before * BA_MAX_RMS_INCREASE)
    if overfit or worse:
        reason = "疑似过拟合" if overfit else "结果变差"
        logger.warning(
            f"[BA] {reason}（中位 RMS {rms_before:.3f}→{rms_after:.3f}px），"
            f"回滚到 BA 前状态"
        )
        for idx in other_kfs:
            frame_poses[idx] = pre_ba_poses[idx]
        for pid in point_ids:
            map_points[pid]['xyz'] = pre_ba_points_xyz[pid]
        return pre_ba_focal, pre_ba_fy

    # 应用 BA 结果到位姿和点
    n_poses = len(other_kfs)
    for i_kf, idx in enumerate(other_kfs):
        start = 1 + i_kf * 6
        rv = param_np[start:start + 3]
        t = param_np[start + 3:start + 6]
        R, _ = cv2.Rodrigues(rv)
        frame_poses[idx] = CameraPose(R, t.reshape(3, 1))

    if optimize_points and point_ids:
        pts_start = 1 + n_poses * 6
        for j, pid in enumerate(point_ids):
            map_points[pid]['xyz'] = param_np[pts_start + j * 3: pts_start + j * 3 + 3]

    if valid_final.sum() > 0:
        inlier_ratio = float((gamma[valid_final] > 0.5).mean())
    else:
        inlier_ratio = 0.0

    logger.info(
        f"[BA-{'EM' if BA_USE_EM else 'soft_l1'}] 结束："
        f"focal {focal:.2f}→{focal_new:.2f}, "
        f"中位 RMS {rms_before:.3f}→{rms_after:.3f}px, "
        f"最终内点率={inlier_ratio:.3f}"
    )

    return focal_new, fy_new


# ---------- 点误差统计（prune 与 filter 共用） ----------
def _compute_point_errors(map_points, frame_poses, focal, fy, cx, cy):
    """遍历所有地图点的观测，统计重投影误差。

    返回三个与 map_points 等长的数组：
        mean_err: 有效观测的平均重投影误差；xyz 非法或无有效观测时为 inf
        valid_count: 有效观测数（位姿存在且 depth > 0）
        neg_ratio: 负深度观测占已评估观测的比例；xyz 非法时记 1.0
    """
    N = len(map_points)
    mean_err = np.full(N, np.inf, dtype=np.float64)
    valid_count = np.zeros(N, dtype=np.int32)
    neg_ratio = np.zeros(N, dtype=np.float64)
    if N == 0:
        return mean_err, valid_count, neg_ratio

    for i, pt in enumerate(map_points):
        xyz = pt['xyz']
        if xyz.size != 3 or not np.isfinite(xyz).all():
            neg_ratio[i] = 1.0
            continue

        obs = pt.get('obs', [])
        if not obs:
            continue

        xyz_col = xyz.reshape(3, 1)
        total_err = 0.0
        count = 0
        neg_depth = 0
        total_obs = 0
        for f_idx, _, u_obs, v_obs in obs:
            pose = frame_poses[f_idx]
            if pose is None:
                continue
            total_obs += 1
            pt_cam = pose.R @ xyz_col + pose.t
            depth = float(pt_cam[2, 0])
            if depth <= 0:
                neg_depth += 1
                continue
            x = float(pt_cam[0, 0]) / depth
            y = float(pt_cam[1, 0]) / depth
            total_err += np.sqrt((focal * x + cx - u_obs) ** 2 +
                                 (fy * y + cy - v_obs) ** 2)
            count += 1

        valid_count[i] = count
        if total_obs > 0:
            neg_ratio[i] = neg_depth / total_obs
        if count > 0:
            mean_err[i] = total_err / count

    return mean_err, valid_count, neg_ratio


# ---------- 修剪 / 过滤 ----------
def _prune_map_points(map_points, feat_map, frame_to_points, reproj_thresh,
                      frame_poses, focal, fy, cx, cy):
    """剔除观测不足、负深度为主或重投影误差过大的地图点，并重建索引。"""
    if not map_points:
        return

    for idx, pt in enumerate(map_points):
        pt['idx'] = idx

    mean_err, valid_count, neg_ratio = _compute_point_errors(
        map_points, frame_poses, focal, fy, cx, cy)

    err_thresh = MAX_REPROJ_ERROR * reproj_thresh
    to_remove = []
    for idx, pt in enumerate(map_points):
        if pt.get('obs_count', 0) < MIN_OBSERVATIONS:
            to_remove.append(idx)
            continue
        if neg_ratio[idx] >= 0.5:
            to_remove.append(idx)
            continue
        if valid_count[idx] > 0 and mean_err[idx] > err_thresh:
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

    # feat_map 重映射
    for f_idx in range(len(feat_map)):
        row = feat_map[f_idx]
        for kp_idx in range(len(row)):
            old = row[kp_idx]
            if old >= 0:
                row[kp_idx] = idx_map.get(old, -1)

    # frame_to_points 重建
    for f in list(frame_to_points.keys()):
        frame_to_points[f] = set()
    for old_idx in reversed(to_remove):
        del map_points[old_idx]
    for new_idx, pt in enumerate(map_points):
        pt['idx'] = new_idx
        for f_idx, _, _, _ in pt['obs']:
            frame_to_points[f_idx].add(new_idx)

    logger.debug(f"修剪 {len(to_remove)} 个点，剩余 {len(map_points)}")


def _filter_point_cloud(map_points, frame_poses, focal, fy, cx, cy, reproj_thresh):
    """输出点云：剔除观测不足 / 负深度为主 / 误差过大的点。"""
    if not map_points:
        return np.array([]), np.array([])
    all_xyz = np.array([p['xyz'] for p in map_points])
    if all_xyz.size == 0:
        return all_xyz, np.array([])

    mean_err, valid_count, neg_ratio = _compute_point_errors(
        map_points, frame_poses, focal, fy, cx, cy)

    errors = np.where(
        (valid_count >= MIN_OBSERVATIONS) & (neg_ratio < 0.5),
        mean_err,
        np.inf,
    )

    finite = np.isfinite(errors)
    if not np.any(finite):
        mask = np.zeros(len(map_points), dtype=bool)
    else:
        median = np.median(errors[finite])
        threshold = max(median * 2.5, reproj_thresh * 1.5)
        mask = finite & (errors < threshold)
    return all_xyz, mask