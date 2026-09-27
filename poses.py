"""
基于 ORB/SIFT 特征 + 增量式 SfM 的相机位姿估计。

职责：
  - 特征提取与匹配（ORB→Hamming，SIFT→L2）
  - 本质矩阵初始化 + 局部地图 PnP 重定位
  - 关键帧管理与冗余剔除
  - 局部/全局 BA（含地图点优化，外层套 GMM-EM 软内点加权）
  - 地图点修剪、过滤与输出

依赖：OpenCV + SciPy + NumPy。
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.special import expit

# =========================================================================
# 阈值与常量
# =========================================================================
# 帧级判定
MIN_INLIERS = 25                     # 内点下限，低于视为匹配失败
MIN_FEATURES = 80                    # 单帧最少特征数
KEYFRAME_ANGLE_DEG = 5.0             # 关键帧旋转阈值（相对最后一个关键帧）
KEYFRAME_TRANS_RATIO = 0.05          # 关键帧平移阈值（相对场景尺度）
COVIS_RATIO_THRESH = 0.25            # 共视比例阈值
KEYFRAME_CULLING_WINDOW = 10         # 冗余剔除回看窗口
PNP_WINDOW = 12                      # PnP 使用的最近关键帧数
SMALL_TRANSLATION = 1e-4             # 纯旋转判定阈值
INIT_MIN_TRANSLATION = 0.01          # 可用于初始化的最小平移
MAX_INIT_CANDIDATES = 30             # 初始化候选队列上限
MAX_CANDIDATE_TRIES = 5              # 每次尝试的最新候选数
MIN_COVIS_MATCHES = 10               # 共视估计所需最少绑定匹配数

# 特征匹配
MATCH_DIST = 90
DESC_UPDATE_THRESH = 35
SIFT_MATCH_DIST = 400.0
SIFT_DESC_UPDATE_THRESH = 150.0

# 三角化
MIN_TRI_ANGLE_DEG = 2.0              # 最小视差角

# 地图点维护
PRUNE_INTERVAL = 200                 # 修剪间隔（帧）
MIN_OBSERVATIONS = 2                 # 最少观测数
MAX_REPROJ_ERROR = 4.0               # 重投影误差上限（倍 reproj_thresh）

# BA 规模
BA_MAX_ITER = 15
GLOBAL_BA_ITER = 25
MIN_BA_WINDOW = 5
MAX_POINTS_IN_BA = 300
BA_MAX_OBS = 2000
BA_MIN_OBS = 200

# BA 防护
BA_MAX_INIT_RMS_RATIO = 3.0          # 初值中位 RMS 超过阈值倍数则跳过
BA_OVERFIT_MIN_RMS = 0.02            # BA 后 RMS 低于此值疑似过拟合（局部）
BA_OVERFIT_MIN_RMS_GLOBAL = 0.005    # 全局 BA 放宽
BA_OVERFIT_MIN_BEFORE = 0.15         # BA 前 RMS 需高于此值才判过拟合
BA_MAX_RMS_INCREASE = 1.2            # BA 后 RMS 超过此倍数视为变差
FOCAL_MAX_STEP_RATIO = 0.05          # 单步焦距最大变化比例
FOCAL_GLOBAL_DRIFT_MAX = 0.15        # 焦距相对初值的累计漂移上限

# EM-BA
EM_ITERS = 3
EM_PI_IN_INIT = 0.85
EM_PI_IN_MAX = 0.95
EM_PI_IN_HARD_CAP = 0.99
EM_SIGMA_CAP_RATIO = 1.5
EM_SIGMA_OUT_RATIO = 8.0
EM_GAMMA_FLOOR = 1e-4

# 深度障碍
EPS_MIN = 1e-8
EPS_MAX = 0.1

# 尺度归一化
SCALE_CLAMP = (0.5, 2.0)
SCALE_DEADBAND = 0.05

logger = logging.getLogger(__name__)


# =========================================================================
# 数据结构
# =========================================================================
@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参。"""
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
    """相机外参：X_cam = R · X_world + t。"""
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
        """相机光心的世界坐标 C = -R^T · t。"""
        return (-self.R.T @ self.t).flatten()


# =========================================================================
# 主入口
# =========================================================================
def estimate_poses(
    frame_paths: List[str],
    *,
    min_inliers: int = MIN_INLIERS,
    feature_type: str = "orb",
    focal_guess: Optional[float] = None,
    aspect_ratio: float = 1.0,
) -> Tuple[CameraIntrinsics, List[CameraPose], np.ndarray]:
    """从有序图像序列估计相机位姿与稀疏点云。

    返回 (intrinsics, poses, xyz)。焦距语义：
        focal_init — 冻结初值，供 BA 漂移保护用。
        focal0     — 当前焦距，循环内随 BA 更新。
    像素阈值（reproj_thresh 等）在循环外一次性算好并冻结，保持像素绝对
    容差语义，不随 focal0 变动。
    """
    if not frame_paths:
        raise ValueError("frame_paths 不能为空")
    if len(frame_paths) < 2:
        raise ValueError("至少需要 2 帧才能初始化 SfM")

    logger.info("加载图像并提取特征…")
    img_shape, kp_list, desc_list = _extract_features(frame_paths, feature_type)

    h, w = img_shape
    cx, cy = w / 2.0, h / 2.0
    focal0 = focal_guess if focal_guess is not None else max(w, h) * 1.2
    focal_init = float(focal0)
    fy0 = focal0 * aspect_ratio
    image_size = max(w, h)

    # 阈值按图像尺寸自适应，跨分辨率可用
    reproj_thresh = max(0.5, min(3.0, image_size * 0.0015))
    ransac_thresh = reproj_thresh * 0.8
    triang_thresh = reproj_thresh

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

    # 描述子度量由 dtype 决定（ORB→Hamming，SIFT→L2）
    norm_type, match_dist, _ = _descriptor_metric(desc_list)
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    bf_cross = cv2.BFMatcher(norm_type, crossCheck=True)

    initialized = False
    init_candidates: List[int] = []   # 纯旋转/平移不足的候选帧
    ba_counter = 0

    for i in range(1, len(frame_paths)):
        logger.info(f"处理帧 {i}/{len(frame_paths)-1}")

        # 每轮从 frame_poses 读起，避免缓存与 BA 修正后的位姿分叉
        prev_pose = frame_poses[i - 1]

        # 特征过少的帧：无法可靠估计，沿用上一帧位姿
        if len(kp_list[i]) < MIN_FEATURES // 2:
            logger.warning(f"帧 {i} 特征过少，沿用上一帧位姿")
            frame_poses[i] = prev_pose
            continue

        matches = _match_features(desc_list[i - 1], desc_list[i], bf,
                                  match_dist=match_dist, norm_type=norm_type,
                                  bf_cross=bf_cross)
        if len(matches) < min_inliers:
            frame_poses[i] = prev_pose
            continue

        pts_prev = np.array([kp_list[i - 1][m.queryIdx].pt for m in matches],
                            dtype=np.float32)
        pts_curr = np.array([kp_list[i][m.trainIdx].pt for m in matches],
                            dtype=np.float32)

        E, mask = cv2.findEssentialMat(pts_prev, pts_curr, focal=focal0,
                                       pp=(cx, cy), method=cv2.RANSAC,
                                       prob=0.999, threshold=ransac_thresh)
        if E is None or mask.sum() < min_inliers:
            frame_poses[i] = prev_pose
            continue

        _, R_rel, t_rel, mask_pose = cv2.recoverPose(
            E, pts_prev, pts_curr, focal=focal0, pp=(cx, cy), mask=mask)
        if int(mask_pose.sum()) < min_inliers:
            frame_poses[i] = prev_pose
            continue

        trans_norm = float(np.linalg.norm(t_rel))
        is_pure_rotation = trans_norm < SMALL_TRANSLATION

        R_curr = R_rel @ prev_pose.R
        t_curr = R_rel @ prev_pose.t + t_rel

        # ============ 初始化阶段 ============
        if not initialized:
            can_init_now = (not is_pure_rotation) and (trans_norm > INIT_MIN_TRANSLATION)

            if can_init_now:
                norm_t = float(np.linalg.norm(t_curr))
                if norm_t > 1e-12:
                    # 相邻帧直接初始化，平移归一到单位长度
                    t_curr = t_curr / norm_t
                    new_pose = CameraPose(R_curr, t_curr)
                    frame_poses[i] = new_pose
                    initialized = True
                    _, new_pose = _triangulate_new_points(
                        i, i - 1, matches, mask_pose.ravel().astype(bool),
                        kp_list, pts_prev, pts_curr,
                        prev_pose, new_pose,
                        focal0, fy0, cx, cy,
                        map_points, feat_map, desc_list, frame_to_points,
                        triang_thresh)
                    frame_poses[i] = new_pose
                    keyframes.append(i)
                    init_candidates.clear()
                    logger.info(f"初始化成功（帧 {i-1} + {i}）")
                    continue

            # 纯旋转/平移不足：登记为候选，等后续出现足够基线的帧再配对。
            # 队列有上限，否则长时间无法初始化会退化成 O(n²) 匹配
            init_candidates.append(i)
            if len(init_candidates) > MAX_INIT_CANDIDATES:
                init_candidates.pop(0)

            if len(init_candidates) >= 2:
                # 只尝试最新的 MAX_CANDIDATE_TRIES 个：旧候选基线虽大但成功
                # 概率低，全量尝试在 ORB 12000 特征时开销过大
                for idx_cand in reversed(init_candidates[-MAX_CANDIDATE_TRIES:]):
                    if idx_cand <= 0 or idx_cand >= i:
                        continue

                    matches_cand = _match_features(
                        desc_list[idx_cand], desc_list[i], bf,
                        match_dist=match_dist, norm_type=norm_type,
                        bf_cross=bf_cross)
                    if len(matches_cand) < min_inliers:
                        continue

                    pts_cand = np.array(
                        [kp_list[idx_cand][m.queryIdx].pt for m in matches_cand],
                        dtype=np.float32)
                    pts_cur2 = np.array(
                        [kp_list[i][m.trainIdx].pt for m in matches_cand],
                        dtype=np.float32)

                    E2, mask2 = cv2.findEssentialMat(
                        pts_cand, pts_cur2, focal=focal0, pp=(cx, cy),
                        method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
                    if E2 is None or mask2.sum() < min_inliers:
                        continue

                    _, R_rel2, t_rel2, mask_pose2 = cv2.recoverPose(
                        E2, pts_cand, pts_cur2, focal=focal0, pp=(cx, cy),
                        mask=mask2)
                    if (mask_pose2.sum() < min_inliers
                            or np.linalg.norm(t_rel2) <= INIT_MIN_TRANSLATION):
                        continue

                    R_curr2 = R_rel2 @ frame_poses[idx_cand].R
                    t_curr2 = R_rel2 @ frame_poses[idx_cand].t + t_rel2
                    norm_t2 = float(np.linalg.norm(t_curr2))
                    if norm_t2 <= 1e-12:
                        continue
                    t_curr2 = t_curr2 / norm_t2

                    new_pose = CameraPose(R_curr2, t_curr2)
                    frame_poses[i] = new_pose
                    initialized = True
                    # prev_idx 必须与三角化参考帧一致：显式传 idx_cand，
                    # 避免观测错记帧、描述子取错、feat_map 错位
                    _, new_pose = _triangulate_new_points(
                        i, idx_cand, matches_cand,
                        mask_pose2.ravel().astype(bool),
                        kp_list, pts_cand, pts_cur2,
                        frame_poses[idx_cand], new_pose,
                        focal0, fy0, cx, cy,
                        map_points, feat_map, desc_list, frame_to_points,
                        triang_thresh)
                    frame_poses[i] = new_pose
                    keyframes.append(i)
                    init_candidates.clear()
                    logger.info(f"初始化成功（帧 {idx_cand} + {i}）")
                    break

            if initialized:
                continue

            # 候选帧保留 R（小基线时 R 准，t 不可靠）
            frame_poses[i] = CameraPose(R_curr, t_curr)
            continue

        # ============ 已初始化：正常增量推进 ============
        new_pose = CameraPose(R_curr, t_curr)
        inlier_mask = mask_pose.ravel().astype(bool)

        _, new_pose = _triangulate_new_points(
            i, i - 1, matches, inlier_mask,
            kp_list, pts_prev, pts_curr,
            prev_pose, new_pose,
            focal0, fy0, cx, cy,
            map_points, feat_map, desc_list, frame_to_points,
            triang_thresh)

        # ----- 局部地图 PnP 重定位 -----
        # PnP 用已知 3D 点恢复位姿，提供正确尺度，抑制纯 E 矩阵链式累积的
        # 尺度漂移。只用最近 PNP_WINDOW 个关键帧；匹配只喂「已绑定到地图点」
        # 的描述子，规模缩减一个量级
        if len(keyframes) > 1:
            pts3d_local, pts2d_local = [], []
            for kf in keyframes[-PNP_WINDOW:]:
                kf_row = feat_map[kf]
                if not kf_row:
                    continue
                kf_bound_kp = [k for k, p in enumerate(kf_row) if p >= 0]
                if len(kf_bound_kp) < 4:
                    continue
                mapped_idx = np.array(kf_bound_kp, dtype=np.int64)
                mapped_desc = desc_list[kf][mapped_idx]
                matches_kf = _match_features(
                    mapped_desc, desc_list[i], bf,
                    match_dist=match_dist, norm_type=norm_type,
                    bf_cross=bf_cross)
                for m in matches_kf:
                    orig_kf_feat = int(mapped_idx[m.queryIdx])
                    pt_idx = kf_row[orig_kf_feat]
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
                    iterationsCount=50, reprojectionError=reproj_thresh,
                    confidence=0.95)
                if inliers_pnp is not None and len(inliers_pnp) >= 8:
                    R_pnp, _ = cv2.Rodrigues(rvec_pnp)
                    t_pnp = tvec_pnp.reshape(3, 1)
                    # 用局部点云质心 + 中位半径刻画场景尺度：不用 ||t_pnp||
                    # 与世界原点比较，相机远离原点后会失真
                    centroid = pts3d_local.mean(axis=0)
                    scene_scale = float(np.median(np.linalg.norm(
                        pts3d_local - centroid, axis=1))) + 1e-6
                    cam_center = (-R_pnp.T @ t_pnp).flatten()
                    cam_offset = float(np.linalg.norm(cam_center - centroid))
                    if cam_offset < 10.0 * scene_scale:
                        new_pose = CameraPose(R_pnp, t_pnp)

        # BA 之前必须写回 frame_poses：否则本帧会被 valid_kfs 过滤掉，
        # 观测在 BA 中丢失，位姿要等下一轮才第一次被修正
        frame_poses[i] = new_pose

        # ----- 关键帧判定 -----
        # 旋转阈值量相对最后一个关键帧的累积旋转：用相邻帧会漏掉匀速旋转
        # （每帧 <5° 但累积 >5°）并误判抖动帧
        if keyframes:
            R_delta = new_pose.R @ frame_poses[keyframes[-1]].R.T
            angle = _compute_rotation_angle(R_delta)
        else:
            angle = _compute_rotation_angle(R_rel)

        if len(map_points) >= 5:
            recent_pts = np.array([p['xyz'] for p in map_points[-200:]],
                                  dtype=np.float32)
            c_recent = recent_pts.mean(axis=0)
            scene_ref = float(np.median(np.linalg.norm(
                recent_pts - c_recent, axis=1))) + 1e-6
        else:
            scene_ref = 1e-6
        # 平移用相机光心之差：旋转大时 t 与实际位移偏离明显
        trans_world = (np.linalg.norm(new_pose.center - prev_pose.center)
                       / scene_ref)
        covis_ratio = _compute_covisibility_ratio(
            i, keyframes, matches, feat_map, frame_to_points)

        is_keyframe = (angle > KEYFRAME_ANGLE_DEG or
                       trans_world > KEYFRAME_TRANS_RATIO or
                       covis_ratio < COVIS_RATIO_THRESH)

        if is_keyframe and not is_pure_rotation and len(map_points) > 20:
            # 冗余剔除：窗口内与当前帧共视过高则替换
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
                # BA 可能修正了本帧位姿，读回以保持一致
                new_pose = frame_poses[i]

        frame_poses[i] = new_pose

        if i % PRUNE_INTERVAL == 0 and len(map_points) > 100:
            _prune_map_points(map_points, feat_map, frame_to_points,
                              reproj_thresh, frame_poses, focal0, fy0, cx, cy)

    if not initialized:
        raise RuntimeError("无法初始化 SfM：未找到具有足够平移的帧对。")

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
    all_xyz, mask = _filter_point_cloud(
        map_points, frame_poses, focal0, fy0, cx, cy, reproj_thresh)

    # 过滤后为空就直接返回空点云，不回退到未过滤点云（那是最差的一批）
    if all_xyz.size == 0:
        pass
    elif np.any(mask):
        all_xyz = all_xyz[mask]
    else:
        logger.warning(f"[filter] 所有 {len(all_xyz)} 个地图点均被过滤，返回空点云")
        all_xyz = all_xyz[:0]

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


# =========================================================================
# 特征提取
# =========================================================================
def _extract_features(paths: List[str], feature_type: str = "orb"):
    """逐帧提取特征。

    不把原图常驻内存（200 帧 1080p 约 1.2GB），每帧读完立即释放。
    无特征帧填与特征类型匹配的空描述子（ORB: (0,32) uint8，SIFT:
    (0,128) float32），保证下游早退逻辑不产生歧义。
    """
    kps, descs = [], []
    if feature_type == "sift":
        try:
            detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.03,
                                       edgeThreshold=10, sigma=1.6)
        except cv2.error as e:
            logger.warning(f"SIFT 不可用（{e}），回退到 ORB。")
            feature_type = "orb"
            detector = cv2.ORB_create(nfeatures=12000, scaleFactor=1.2,
                                      nlevels=8, edgeThreshold=31, patchSize=31)
    else:
        detector = cv2.ORB_create(nfeatures=12000, scaleFactor=1.2,
                                  nlevels=8, edgeThreshold=31, patchSize=31)

    empty_desc = (np.zeros((0, 128), dtype=np.float32) if feature_type == "sift"
                  else np.zeros((0, 32), dtype=np.uint8))

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
        descs.append(desc if desc is not None else empty_desc)
        del img, gray
    return shape0, kps, descs


# =========================================================================
# 描述子度量
# =========================================================================
def _descriptor_metric(desc_list):
    """按描述子 dtype 返回 (cv2 范数类型, 匹配距离上限, 描述子更新阈值)。

    ORB 为 uint8 二进制 → Hamming；SIFT 为 float32（OpenCV 归一化约 512
    范数）→ L2。对 float 描述子用 Hamming 会得到随机匹配或崩溃。
    空描述子列表按 ORB/Hamming 处理。
    """
    for d in desc_list:
        if d is not None and len(d) > 0:
            if d.dtype != np.uint8:
                return cv2.NORM_L2, SIFT_MATCH_DIST, SIFT_DESC_UPDATE_THRESH
            return cv2.NORM_HAMMING, MATCH_DIST, DESC_UPDATE_THRESH
    return cv2.NORM_HAMMING, MATCH_DIST, DESC_UPDATE_THRESH


# =========================================================================
# 特征匹配
# =========================================================================
def _match_features(desc1, desc2, bf, ratio=0.75,
                    match_dist=None, norm_type=None, bf_cross=None):
    """Lowe 比值测试 + 距离上限过滤。

    bf_cross 仅在 desc2 < 2 无法做 kNN 时使用。
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    if match_dist is None or norm_type is None:
        nt, md, _ = _descriptor_metric([desc1])
        if match_dist is None:
            match_dist = md
        if norm_type is None:
            norm_type = nt

    if len(desc2) < 2:
        # 描述子太少无法做 kNN，退回 crossCheck 暴力匹配
        if bf_cross is None:
            bf_cross = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = bf_cross.match(desc1, desc2)
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


# =========================================================================
# 辅助函数
# =========================================================================
def _compute_rotation_angle(R):
    """返回旋转矩阵对应的旋转向量模长（度）。"""
    rv, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(rv) * 180.0 / np.pi)


def _compute_covisibility_ratio(curr_idx, keyframes, matches,
                                feat_map, frame_to_points):
    """当前帧匹配与最近关键帧地图点的共视比例。

    分母用「已绑定到某地图点的当前帧匹配数」而非全部匹配数：三角化失败
    率高时用全部匹配数会系统性低估共视率，导致关键帧膨胀。
    绑定匹配过少时返回 0，触发新关键帧。需在三角化之后调用。
    """
    if not matches or len(keyframes) < 2:
        return 1.0
    last_kf = keyframes[-1]
    if last_kf not in frame_to_points or not frame_to_points[last_kf]:
        return 0.0
    if curr_idx >= len(feat_map):
        return 0.0
    curr_feat_row = feat_map[curr_idx]
    last_points = frame_to_points[last_kf]

    total_bound = 0
    count = 0
    for m in matches:
        t_idx = m.trainIdx
        if t_idx < 0 or t_idx >= len(curr_feat_row):
            continue
        bound = curr_feat_row[t_idx]
        if bound < 0:
            continue
        total_bound += 1
        if bound in last_points:
            count += 1

    if total_bound < MIN_COVIS_MATCHES:
        return 0.0
    return count / total_bound


def _add_observation(pt_dict, frame_idx, kp_idx, uv, frame_to_points):
    """向地图点追加一条观测，按 (frame_idx, kp_idx) 去重。"""
    key = (frame_idx, kp_idx)
    if key in pt_dict['obs_set']:
        return
    pt_dict['obs_set'].add(key)
    pt_dict['obs'].append((frame_idx, kp_idx, float(uv[0]), float(uv[1])))
    pt_dict['obs_count'] += 1
    pt_idx = pt_dict['idx']
    if pt_idx >= 0:
        frame_to_points[frame_idx].add(pt_idx)


def _update_map_point_descriptor(pt_dict, new_desc):
    """观测少/描述子过旧/差异过大时替换描述子，否则累积年龄。"""
    old = pt_dict.get('desc')
    if old is None:
        pt_dict['desc'] = new_desc.copy()
        pt_dict['desc_age'] = 0
        return
    norm_type, _, update_thresh = _descriptor_metric([new_desc])
    dist = cv2.norm(old, new_desc, norm_type)
    age = pt_dict.get('desc_age', 0)
    if pt_dict.get('obs_count', 0) < 3 or age > 5 or dist >= update_thresh * 1.2:
        pt_dict['desc'] = new_desc.copy()
        pt_dict['desc_age'] = 0
    else:
        pt_dict['desc_age'] = age + 1


# =========================================================================
# 三角化
# =========================================================================
def _triangulate_new_points(curr_idx, prev_idx, matches, inlier_mask,
                            kp_list, pts_prev, pts_curr,
                            pose_prev, pose_curr,
                            focal, fy, cx, cy,
                            map_points, feat_map, desc_list, frame_to_points,
                            triang_thresh):
    """对当前帧内点做三角化，新增或复用地图点，并做尺度归一化。

    prev_idx 必须由调用方显式传入（正常增量路径为 i-1，候选配对路径为
    idx_cand），硬编码 curr_idx-1 会让候选路径的观测错记帧、feat_map 错位。
    返回 (新增点索引列表, 可能被尺度归一化调整后的 pose_curr)。

    批处理版本：cv2.triangulatePoints 与深度/视差/重投影合法性检查全部
    向量化，避免逐点 Python 循环。
    """
    if prev_idx >= curr_idx:
        raise RuntimeError(
            f"_triangulate_new_points: prev_idx={prev_idx} 必须 < "
            f"curr_idx={curr_idx}"
        )

    K = np.array([[focal, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    P_prev = K @ np.asarray(pose_prev.RT[:3], dtype=np.float64)
    P_curr = K @ np.asarray(pose_curr.RT[:3], dtype=np.float64)
    new_indices = []

    inlier_ids = np.where(inlier_mask)[0]
    if len(inlier_ids) == 0:
        return new_indices, pose_curr

    feat_map_curr_row = feat_map[curr_idx]
    feat_map_prev_row = feat_map[prev_idx]
    R_curr = pose_curr.R
    t_curr = pose_curr.t
    err_thresh = MAX_REPROJ_ERROR * triang_thresh
    err_thresh_sq = err_thresh * err_thresh

    # ---- 分类：复用已有地图点 vs 新三角化 ----
    reuse_items: List[Tuple[int, int]] = []
    new_items: List[int] = []
    for idx in inlier_ids:
        m = matches[idx]
        curr_feat = m.trainIdx
        if feat_map_curr_row[curr_feat] >= 0:
            continue
        prev_feat = m.queryIdx
        existing = feat_map_prev_row[prev_feat]
        if existing >= 0:
            xyz_existing = map_points[existing]['xyz']
            pt_cam = R_curr @ xyz_existing.reshape(3, 1) + t_curr
            depth_check = float(pt_cam[2, 0])
            if depth_check <= 1e-6:
                continue
            du = (focal * float(pt_cam[0, 0]) / depth_check + cx
                  - pts_curr[idx][0])
            dv = (fy * float(pt_cam[1, 0]) / depth_check + cy
                  - pts_curr[idx][1])
            if du * du + dv * dv > err_thresh_sq:
                continue
            reuse_items.append((int(idx), int(existing)))
        else:
            new_items.append(int(idx))

    # ---- 复用已有地图点：仅追加观测 ----
    for idx, existing in reuse_items:
        m = matches[idx]
        _add_observation(map_points[existing], curr_idx, m.trainIdx,
                         pts_curr[idx], frame_to_points)
        feat_map_curr_row[m.trainIdx] = existing
        _update_map_point_descriptor(map_points[existing],
                                     desc_list[curr_idx][m.trainIdx])

    # ---- 批量三角化 + 批量合法性检查 ----
    if new_items:
        new_arr = np.asarray(new_items, dtype=np.int64)
        pts_prev_b = pts_prev[new_arr].astype(np.float64)   # (N,2)
        pts_curr_b = pts_curr[new_arr].astype(np.float64)   # (N,2)

        pts4d = cv2.triangulatePoints(
            P_prev, P_curr,
            pts_prev_b.T.reshape(2, -1).astype(np.float64),
            pts_curr_b.T.reshape(2, -1).astype(np.float64),
        )
        w = pts4d[3].astype(np.float64)
        pts3d = (pts4d[:3].astype(np.float64) / (w + 1e-12)).T   # (N,3)

        finite = np.isfinite(pts3d).all(axis=1)

        cam_prev_c = pose_prev.center.astype(np.float64)
        cam_curr_c = pose_curr.center.astype(np.float64)
        R_prev_2 = pose_prev.R[2].astype(np.float64)
        R_curr_2 = pose_curr.R[2].astype(np.float64)
        c_prev = float(R_prev_2 @ cam_prev_c)
        c_curr = float(R_curr_2 @ cam_curr_c)

        d_prev = pts3d @ R_prev_2 - c_prev
        d_curr = pts3d @ R_curr_2 - c_curr
        valid = finite & (d_prev > 0) & (d_curr > 0)

        # 视差角
        v1 = pts3d - cam_prev_c
        v2 = pts3d - cam_curr_c
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        n1_safe = np.where(n1 > 1e-8, n1, 1.0)
        n2_safe = np.where(n2 > 1e-8, n2, 1.0)
        cos_a = np.sum(v1 * v2, axis=1) / (n1_safe * n2_safe)
        cos_thresh = np.cos(np.radians(MIN_TRI_ANGLE_DEG))
        valid &= (n1 > 1e-8) & (n2 > 1e-8) & (cos_a <= cos_thresh)

        # 重投影误差
        pts_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])
        proj_prev = pts_h @ P_prev.T
        proj_curr = pts_h @ P_curr.T
        proj_prev = proj_prev[:, :2] / (proj_prev[:, 2:3] + 1e-12)
        proj_curr = proj_curr[:, :2] / (proj_curr[:, 2:3] + 1e-12)
        err_prev = np.linalg.norm(proj_prev - pts_prev_b, axis=1)
        err_curr = np.linalg.norm(proj_curr - pts_curr_b, axis=1)
        valid &= (err_prev <= triang_thresh) & (err_curr <= triang_thresh)

        # ---- 落库 ----
        for k in np.where(valid)[0]:
            idx = int(new_arr[k])
            m = matches[idx]
            prev_feat = m.queryIdx
            curr_feat = m.trainIdx
            pt_idx = len(map_points)
            pt_dict = {
                'idx': pt_idx,
                'xyz': pts3d[k].astype(np.float32),
                'desc': desc_list[prev_idx][prev_feat].copy(),
                'obs': [],
                'obs_set': set(),
                'obs_count': 0,
                'desc_age': 0,
            }
            _add_observation(pt_dict, prev_idx, prev_feat,
                             pts_prev[idx], frame_to_points)
            _add_observation(pt_dict, curr_idx, curr_feat,
                             pts_curr[idx], frame_to_points)
            map_points.append(pt_dict)
            feat_map[prev_idx][prev_feat] = pt_idx
            feat_map_curr_row[curr_feat] = pt_idx
            new_indices.append(pt_idx)

    # ----- 尺度归一化（向量化）-----
    # E 矩阵恢复的 t_rel 是单位范数，链式叠加会导致尺度随帧累积漂移。
    # 把本次新增点的相机 z 深度中位数对齐到已有地图点的中位数，并同步
    # 缩放当前相机位移。
    # 关键：缩放相对上一帧相机中心（不是世界原点）；且是否修正必须在
    # 裁剪之前判断，否则 <0.5 的负向修正永远被裁成 0.5，等于禁用它。
    if new_indices and map_points:
        old_count = len(map_points) - len(new_indices)
        if old_count >= 10:
            cam_curr_flat = pose_curr.center.astype(np.float64)
            R_curr_2 = pose_curr.R[2].astype(np.float64)
            c_curr = float(R_curr_2 @ cam_curr_flat)

            old_xyz_list = [p['xyz'] for p in map_points[:old_count]
                            if p['xyz'].size == 3]
            if len(old_xyz_list) >= 5:
                old_xyz = np.asarray(old_xyz_list, dtype=np.float64)
                old_depths = old_xyz @ R_curr_2 - c_curr
                old_depths = old_depths[old_depths > 0]
            else:
                old_depths = np.empty(0, dtype=np.float64)

            if len(old_depths) >= 5:
                ref_median = float(np.median(old_depths))
                new_xyz = np.asarray(
                    [map_points[pi]['xyz'] for pi in new_indices],
                    dtype=np.float64)
                new_depths = new_xyz @ R_curr_2 - c_curr
                new_depths = new_depths[new_depths > 0]

                if len(new_depths) > 0:
                    new_median = float(np.median(new_depths))
                    if new_median > 1e-8 and ref_median > 1e-8:
                        raw_scale = ref_median / new_median
                        if abs(raw_scale - 1.0) > SCALE_DEADBAND:
                            scale = float(np.clip(raw_scale, *SCALE_CLAMP))
                            origin = pose_prev.center.astype(np.float64)
                            scaled = origin + scale * (new_xyz - origin)
                            for k, pi in enumerate(new_indices):
                                map_points[pi]['xyz'] = scaled[k].astype(np.float32)
                            cam_curr_new = origin + scale * (cam_curr_flat - origin)
                            # 由相机中心反解 t：t = -R @ C
                            t_new = -pose_curr.R @ cam_curr_new.reshape(3, 1)
                            pose_curr = CameraPose(
                                pose_curr.R, t_new.astype(np.float32))

    return new_indices, pose_curr


def _is_valid_point(pt3d, pose_prev, pose_curr,
                    reproj_th, P_prev, P_curr, pt_prev, pt_curr):
    """三角化点的合法性检查：有限性、正深度、视差角、重投影误差。

    保留标量版；_triangulate_new_points 内部已经改用批处理路径。
    """
    if pt3d.shape != (3,):
        pt3d = pt3d.flatten()
    if not np.isfinite(pt3d).all():
        return False

    cam_prev = pose_prev.center
    cam_curr = pose_curr.center
    depth_prev = float(pose_prev.R[2] @ (pt3d - cam_prev))
    depth_curr = float(pose_curr.R[2] @ (pt3d - cam_curr))
    if depth_prev <= 0 or depth_curr <= 0:
        return False

    # 视差角太小 → 三角化数值不稳定
    v1 = pt3d - cam_prev
    v2 = pt3d - cam_curr
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


# =========================================================================
# EM-BA 辅助函数
# =========================================================================
def _em_e_step(res, obs_depths, sigma, pi_in, sigma_out, eps):
    """EM 的 E 步：计算每个观测属于内点的后验概率 γ。

    模型：r ~ π_in·N(0,σ²I) + (1-π_in)·N(0,σ_out²I)。
    γ 由 log-odds 经 sigmoid 得到，r² 项系数为负（残差越大越倾向外点）。
    深度不足的观测强制 γ=1：其残差是深度障碍项，不参与外点判定。
    返回 (gamma, r2)；r2 中深度不足的观测置 0。
    """
    n_obs = len(obs_depths)
    # res 由 _compute_residuals 产出，每个观测固定 2 个残差；用 raise 而非
    # assert，避免 -O 下不变量检查被剥离后静默错位残差对
    if len(res) != 2 * n_obs:
        raise RuntimeError(
            f"_em_e_step: res 长度 {len(res)} 与观测数 {n_obs} 不匹配（应为 2×）"
        )

    r2 = (res.reshape(n_obs, 2) ** 2).sum(axis=1)

    log_ratio = (np.log(pi_in / max(1.0 - pi_in, 1e-12))
                 + 2.0 * np.log(sigma_out / sigma)
                 + r2 * (1.0 / (2.0 * sigma_out ** 2)
                         - 1.0 / (2.0 * sigma ** 2)))
    # expit 按符号分段，log_ratio 很负时也不会溢出
    gamma = expit(log_ratio)

    invalid = obs_depths <= eps
    gamma[invalid] = 1.0
    r2[invalid] = 0.0

    return gamma, r2


def _em_m_step(r2, gamma, sigma_cap):
    """EM 的 M 步：加权重估 σ 和 π_in。

    σ 只从高置信内点（γ>0.9）估计：用全部 γ 加权时大残差观测也会贡献权重
    把 σ 拉大，σ 变大又让 γ 降不下来，形成塌缩正反馈。
    π_in 上限按当前 γ 分布动态调整。r² 是 2 维残差平方和，E[r²]=2σ²。
    """
    high_conf_ratio = float((gamma > 0.9).mean())
    if high_conf_ratio > 0.8:
        pi_in_max = min(EM_PI_IN_MAX + 0.04, EM_PI_IN_HARD_CAP)
    elif high_conf_ratio > 0.6:
        pi_in_max = EM_PI_IN_MAX
    else:
        pi_in_max = max(EM_PI_IN_MAX - 0.05, 0.7)
    pi_in_new = float(np.clip(np.mean(gamma), 0.05, pi_in_max))

    high_conf = gamma > 0.9
    if high_conf.sum() >= 5:
        sigma2 = float(np.sum(r2[high_conf]) / (2.0 * high_conf.sum()))
    else:
        # 高置信样本太少，退回 γ 加权
        denom = 2.0 * (np.sum(gamma) + 1e-12)
        sigma2 = float(np.sum(gamma * r2) / denom)

    sigma_new = float(np.clip(np.sqrt(max(sigma2, 1e-6)), 0.1, sigma_cap))
    return sigma_new, pi_in_new


def _compute_adaptive_eps(map_points, ref_pose, default_eps=0.01):
    """按参考相机下 z 向深度中位数确定深度障碍阈值。

    用 ||p||（到世界原点距离）近似会在相机远离原点后失真。
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


# =========================================================================
# 鲁棒光束法平差（EM 加权）
# =========================================================================
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
    """局部/全局 BA，外层套 GMM-EM 软内点加权。

    参数化：[focal, (rv_kf, t_kf) * N_other_kf, (xyz_pt) * N_pts]。
    fy 锁定为 focal * fy_ratio，避免与场景尺度耦合导致病态。
    每个观测先估计内点后验概率 γ，再用 sqrt(γ) 加权跑最小二乘。
    深度不足的观测强制 γ=1，残差（深度障碍项）保持完整权重。

    防护：最小观测门槛、全深度不足跳过、初值过差跳过、焦距边界交叉跳过、
    焦距全局漂移保护、BA 后过拟合/变差检测回滚。

    残差函数已完全向量化：所有观测的投影、深度障碍、加权一次算完。
    """
    if focal_init is None:
        focal_init = focal

    sigma_cap = reproj_thresh * EM_SIGMA_CAP_RATIO
    sigma_out = reproj_thresh * EM_SIGMA_OUT_RATIO
    # 全局 BA 是最终精修，允许 RMS 更低；局部 BA 用较严阈值避免误判
    overfit_rms_thresh = (BA_OVERFIT_MIN_RMS_GLOBAL if is_global
                          else BA_OVERFIT_MIN_RMS)

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

    if len(obs) < BA_MIN_OBS:
        logger.info(f"[BA] 观测数 {len(obs)} < {BA_MIN_OBS}，跳过本次 BA")
        return focal, fy

    n_obs_orig = len(obs)
    rng = np.random.default_rng(42)
    if n_obs_orig > BA_MAX_OBS:
        idx = rng.choice(n_obs_orig, BA_MAX_OBS, replace=False)
        obs = [obs[i] for i in idx]
        logger.info(f"[BA] 抽样 {BA_MAX_OBS} / {n_obs_orig} 个观测")

    # 固定观测最多的关键帧作为基准（gauge fix）
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
    else:
        # 点参数冻结：观测限制在已有点上，未知点的观测丢弃
        known_point_ids = set(range(len(map_points)))
        obs = [o for o in obs if o[1] in known_point_ids]

    # 只保留可优化点的观测：否则静态点带误差会持续贡献正 cost，优化器只能
    # 移动位姿和点「绕着」误差走
    if optimize_points:
        point_id_set = set(point_ids)
        obs = [o for o in obs if o[1] in point_id_set]
    n_obs = len(obs)
    if n_obs < 10:
        return focal, fy

    # 平移上界：与场景尺度和相机位移相关，防止 BA 把相机推到无穷远
    scene_scale = 1.0
    if map_points:
        xs = np.array([p['xyz'] for p in map_points if p['xyz'].size == 3])
        if len(xs) > 0:
            scene_scale = float(np.linalg.norm(
                xs.max(axis=0) - xs.min(axis=0))) + 1e-6
    cam_ts = [np.linalg.norm(frame_poses[k].t.flatten()) for k in other_kfs
              if frame_poses[k] is not None]
    cam_t_max = max(cam_ts) if cam_ts else 1.0
    t_bound = max(scene_scale * 5.0, cam_t_max * 2.0, 10.0)
    # 旋转向量范数可近 4π，任何有限边界都可能触发初值越界
    r_bound = np.inf

    # 焦距边界取「单步 ±ratio」与「相对初值 ±drift」的交集；两者理论上可能
    # 交叉，导致 least_squares 因 lower > upper 抛异常，必须提前拦截
    focal_lo = max(focal * (1.0 - FOCAL_MAX_STEP_RATIO),
                   focal_init * (1.0 - FOCAL_GLOBAL_DRIFT_MAX))
    focal_hi = min(focal * (1.0 + FOCAL_MAX_STEP_RATIO),
                   focal_init * (1.0 + FOCAL_GLOBAL_DRIFT_MAX))
    if focal_lo >= focal_hi:
        logger.warning(
            f"[BA] 焦距边界交叉（lo={focal_lo:.2f} ≥ hi={focal_hi:.2f}），"
            f"跳过本次 BA 以免破坏位姿和焦距"
        )
        return focal, fy

    lower_pose, upper_pose = [], []
    for _kf in other_kfs:
        lower_pose += [-r_bound] * 3 + [-t_bound] * 3
        upper_pose += [r_bound] * 3 + [t_bound] * 3
    bounds_lower = [focal_lo] + lower_pose
    bounds_upper = [focal_hi] + upper_pose
    if optimize_points and point_ids:
        bounds_lower += [-np.inf] * len(point_ids) * 3
        bounds_upper += [np.inf] * len(point_ids) * 3

    # ---------- 3. 残差闭包（向量化） ----------
    point_ids_local = list(point_ids)
    n_poses = len(other_kfs)

    kf_to_pos = {kf: k for k, kf in enumerate(keyframe_ids)}
    n_kfs = len(keyframe_ids)
    fixed_kf_pos = kf_to_pos[fixed_kf]

    # 点参数位置索引：optimize_points 时用 point_ids 的顺序（与 param 追加
    # 顺序一致）；否则用 obs 中出现过的点集
    if optimize_points and point_ids_local:
        all_pt_ids = list(point_ids_local)
    else:
        all_pt_ids = sorted({o[1] for o in obs})
    ptid_to_pos = {pid: j for j, pid in enumerate(all_pt_ids)}

    obs_fpos = np.fromiter((kf_to_pos[f] for f, _, _, _ in obs),
                           dtype=np.int64, count=n_obs)
    obs_ppos = np.fromiter((ptid_to_pos[pid] for _, pid, _, _ in obs),
                           dtype=np.int64, count=n_obs)
    obs_u = np.fromiter((u for _, _, u, _ in obs),
                        dtype=np.float64, count=n_obs)
    obs_v = np.fromiter((v for _, _, _, v in obs),
                        dtype=np.float64, count=n_obs)

    # 点参数冻结时预先构造静态坐标数组
    static_pts_arr = None
    if not (optimize_points and point_ids_local):
        static_pts_arr = np.zeros((len(all_pt_ids), 3), dtype=np.float64)
        for pid, j in ptid_to_pos.items():
            static_pts_arr[j] = map_points[pid]['xyz']

    def _compute_residuals(params):
        """返回 (res_flat, obs_depths)，res_flat 长度恒为 2*n_obs。"""
        f = params[0]
        fy_local = f * fy_ratio

        # ---- 位姿 ----
        R_arr = np.empty((n_kfs, 3, 3), dtype=np.float64)
        t_arr = np.empty((n_kfs, 3), dtype=np.float64)
        R_arr[fixed_kf_pos] = frame_poses[fixed_kf].R
        t_arr[fixed_kf_pos] = frame_poses[fixed_kf].t.flatten()
        for i_kf, idx in enumerate(other_kfs):
            start = 1 + i_kf * 6
            R, _ = cv2.Rodrigues(params[start:start + 3])
            R_arr[kf_to_pos[idx]] = R
            t_arr[kf_to_pos[idx]] = params[start + 3:start + 6]

        # ---- 点 ----
        if optimize_points and point_ids_local:
            pts_start = 1 + n_poses * 6
            n_pts = len(point_ids_local)
            pts_arr = params[pts_start:pts_start + n_pts * 3].reshape(-1, 3)
        else:
            pts_arr = static_pts_arr

        # ---- 批量投影 ----
        xyz_o = pts_arr[obs_ppos]                       # (M,3)
        R_o = R_arr[obs_fpos]                           # (M,3,3)
        t_o = t_arr[obs_fpos]                           # (M,3)
        pt_cam = np.einsum('nij,nj->ni', R_o, xyz_o) + t_o
        depth = pt_cam[:, 2]

        # ---- 残差 ----
        res = np.empty((n_obs, 2), dtype=np.float64)
        depth_safe = np.where(depth > eps, depth, 1.0)
        res[:, 0] = f * (pt_cam[:, 0] / depth_safe) + cx - obs_u
        res[:, 1] = fy_local * (pt_cam[:, 1] / depth_safe) + cy - obs_v

        # ---- 深度障碍 ----
        shallow = depth <= eps
        if np.any(shallow):
            d_sh = depth[shallow]
            very_sh = d_sh <= 1e-6
            denom = np.where(very_sh, 1e-6, eps)
            barrier = -np.log(np.maximum(d_sh / denom, 1e-10))
            res[shallow, 0] = barrier
            res[shallow, 1] = barrier

        return res.flatten(), depth

    # ---------- 4. 初值检查 ----------
    param_np = np.array(param, dtype=np.float64)
    res0, depths0 = _compute_residuals(param_np)
    r2_0 = (res0.reshape(n_obs, 2) ** 2).sum(axis=1)
    valid_mask = depths0 > eps
    if valid_mask.sum() == 0:
        logger.warning("[BA] 所有观测深度不足，跳过本次 BA")
        return focal, fy

    # 中位 RMS：残差重尾，中位数比均值更能反映典型拟合质量
    rms_before = float(np.sqrt(np.median(r2_0[valid_mask]) / 2))

    init_rms_limit = reproj_thresh * BA_MAX_INIT_RMS_RATIO
    if rms_before > init_rms_limit:
        logger.warning(
            f"[BA] 初值过差（中位 RMS={rms_before:.2f}px > 限 "
            f"{init_rms_limit:.2f}px），跳过本次 BA 以免破坏位姿和焦距"
        )
        return focal, fy

    # 保存 BA 前状态，供过拟合/变差检测回滚
    pre_ba_poses = {idx: frame_poses[idx] for idx in other_kfs}
    pre_ba_points_xyz = {pid: map_points[pid]['xyz'].copy()
                         for pid in point_ids_local}
    pre_ba_focal = float(param_np[0])
    pre_ba_fy = float(param_np[0]) * fy_ratio

    # ---------- 5. EM 迭代 ----------
    sigma = max(float(np.sqrt(np.median(r2_0[valid_mask]) / 2.0)), 0.5)
    pi_in = EM_PI_IN_INIT

    logger.info(
        f"[BA] 开始：{len(keyframe_ids)} 关键帧，{len(obs)} 观测，"
        f"深度不足={int((~valid_mask).sum())}，"
        f"sigma0={sigma:.3f}px, pi_in0={pi_in:.3f}, "
        f"sigma_cap={sigma_cap:.2f}px, sigma_out={sigma_out:.2f}px, "
        f"初始中位 RMS={rms_before:.3f}px"
    )

    result = None
    gamma = np.ones(n_obs, dtype=np.float64)

    for em_iter in range(EM_ITERS):
        # ----- E 步 -----
        gamma, r2 = _em_e_step(res0, depths0, sigma, pi_in, sigma_out, eps)
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
            f"[BA] EM {em_iter + 1}/{EM_ITERS} E步："
            f"pi_in={pi_in:.3f}, sigma={sigma:.3f}, "
            f"gamma>0.5 占比={inlier_high:.3f}, "
            f"gamma 均值={inlier_mean:.3f}, "
            f"中位残差={rms_now:.3f}px"
        )

        # ----- M 步：加权最小二乘 -----
        sqrt_gamma = np.sqrt(gamma)
        sqrt_gamma_rep = np.repeat(sqrt_gamma, 2)

        def _weighted_residuals(params, _sg_rep=sqrt_gamma_rep):
            res_plain, _ = _compute_residuals(params)
            return res_plain * _sg_rep

        # 第一轮给完整预算，后续轮起点已接近最优，减半
        iters_this = max_iter if em_iter == 0 else max(3, max_iter // 2)

        try:
            result = least_squares(
                _weighted_residuals, param_np,
                bounds=(bounds_lower, bounds_upper),
                method='trf', loss='linear',
                max_nfev=iters_this, verbose=0,
                ftol=1e-4, xtol=1e-4, gtol=1e-4,
            )
            param_np = result.x
        except Exception as e:
            logger.warning(f"[BA] M 步异常：{e}")
            break

        # ----- 更新 σ 和 π_in -----
        res0, depths0 = _compute_residuals(param_np)
        r2 = (res0.reshape(n_obs, 2) ** 2).sum(axis=1)
        valid_now = depths0 > eps
        if valid_now.sum() > 0:
            sigma, pi_in = _em_m_step(r2[valid_now], gamma[valid_now],
                                      sigma_cap)
            # inlier_rms 只统计 γ>0.5 的观测，反映内点拟合质量；
            # weighted_rms 按 γ 加权，反映整体优化目标
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
            f"[BA] EM {em_iter + 1}/{EM_ITERS} M步："
            f"cost={result.cost:.1f}, "
            f"新 sigma={sigma:.3f}, 新 pi_in={pi_in:.3f}, "
            f"内点 RMS={inlier_rms:.3f}px, 加权 RMS={weighted_rms:.3f}px"
        )

    # ---------- 6. 结果提取 ----------
    if result is None:
        logger.warning("[BA] 无有效优化结果，返回原参数")
        return focal, fy

    focal_new = max(float(param_np[0]), 1.0)
    fy_new = focal_new * fy_ratio

    res_final, depths_final = _compute_residuals(param_np)
    r2_final = (res_final.reshape(n_obs, 2) ** 2).sum(axis=1)
    valid_final = depths_final > eps
    if valid_final.sum() > 0:
        rms_after = float(np.sqrt(np.median(r2_final[valid_final]) / 2.0))
    else:
        rms_after = 0.0

    # 过拟合：RMS 异常低（真实噪声不会低于阈值下限）；变差：RMS 反升。
    # 两者都回滚
    overfit = (rms_after < overfit_rms_thresh
               and rms_before > BA_OVERFIT_MIN_BEFORE)
    worse = (rms_after > rms_before * BA_MAX_RMS_INCREASE)
    if overfit or worse:
        reason = "疑似过拟合" if overfit else "结果变差"
        logger.warning(
            f"[BA] {reason}（中位 RMS {rms_before:.3f}→{rms_after:.3f}px），"
            f"回滚到 BA 前状态"
        )
        for idx in other_kfs:
            frame_poses[idx] = pre_ba_poses[idx]
        for pid in point_ids_local:
            map_points[pid]['xyz'] = pre_ba_points_xyz[pid]
        return pre_ba_focal, pre_ba_fy

    # 应用 BA 结果
    for i_kf, idx in enumerate(other_kfs):
        start = 1 + i_kf * 6
        rv = param_np[start:start + 3]
        t = param_np[start + 3:start + 6]
        R, _ = cv2.Rodrigues(rv)
        frame_poses[idx] = CameraPose(R, t.reshape(3, 1))

    if optimize_points and point_ids_local:
        pts_start = 1 + n_poses * 6
        for j, pid in enumerate(point_ids_local):
            map_points[pid]['xyz'] = param_np[pts_start + j * 3:
                                              pts_start + j * 3 + 3]

    # 重算 γ 匹配最终 σ/π_in：循环内的 γ 来自上一轮 E 步，与最终残差不同步
    if valid_final.sum() > 0:
        gamma, _ = _em_e_step(res_final, depths_final, sigma, pi_in,
                              sigma_out, eps)
        inlier_ratio = float((gamma[valid_final] > 0.5).mean())
    else:
        inlier_ratio = 0.0

    logger.info(
        f"[BA] 结束：focal {focal:.2f}→{focal_new:.2f}, "
        f"中位 RMS {rms_before:.3f}→{rms_after:.3f}px, "
        f"最终内点率={inlier_ratio:.3f}"
    )

    return focal_new, fy_new


# =========================================================================
# 点误差统计（prune 与 filter 共用）
# =========================================================================
def _compute_point_errors(map_points, frame_poses, focal, fy, cx, cy):
    """遍历所有地图点的观测，统计重投影误差。

    返回三个与 map_points 等长数组：
        mean_err:    有效观测的平均重投影误差；xyz 非法或无有效观测时为 inf
        valid_count: 有效观测数（位姿存在且 depth > 0）
        neg_ratio:   负深度观测占比；xyz 非法时记 1.0

    向量化版本：先收集所有观测，批量投影，再用 bincount 聚合到每个点上。
    """
    N = len(map_points)
    mean_err = np.full(N, np.inf, dtype=np.float64)
    valid_count = np.zeros(N, dtype=np.int32)
    neg_ratio = np.zeros(N, dtype=np.float64)
    if N == 0:
        return mean_err, valid_count, neg_ratio

    xyz_all = np.full((N, 3), np.nan, dtype=np.float64)
    pt_ids_l, f_ids_l, us_l, vs_l = [], [], [], []
    for i, pt in enumerate(map_points):
        xyz = pt['xyz']
        if xyz.size != 3:
            continue
        xyz_all[i] = xyz
        for f_idx, _, u_obs, v_obs in pt.get('obs', []):
            pt_ids_l.append(i)
            f_ids_l.append(f_idx)
            us_l.append(u_obs)
            vs_l.append(v_obs)

    invalid_pts = ~np.isfinite(xyz_all).all(axis=1)
    neg_ratio[invalid_pts] = 1.0

    if not pt_ids_l:
        return mean_err, valid_count, neg_ratio

    pt_ids = np.asarray(pt_ids_l, dtype=np.int64)
    f_ids = np.asarray(f_ids_l, dtype=np.int64)
    us = np.asarray(us_l, dtype=np.float64)
    vs = np.asarray(vs_l, dtype=np.float64)

    n_frames = len(frame_poses)
    R_all = np.zeros((n_frames, 3, 3), dtype=np.float64)
    t_all = np.zeros((n_frames, 3), dtype=np.float64)
    pose_valid = np.zeros(n_frames, dtype=bool)
    for f_idx, pose in enumerate(frame_poses):
        if pose is not None:
            R_all[f_idx] = pose.R
            t_all[f_idx] = pose.t.flatten()
            pose_valid[f_idx] = True

    keep = (~invalid_pts[pt_ids]) & pose_valid[f_ids]
    if not np.any(keep):
        return mean_err, valid_count, neg_ratio
    pt_ids = pt_ids[keep]
    f_ids = f_ids[keep]
    us = us[keep]
    vs = vs[keep]

    xyz_o = xyz_all[pt_ids]                 # (M,3)
    R_o = R_all[f_ids]                      # (M,3,3)
    t_o = t_all[f_ids]                      # (M,3)
    pt_cam = np.einsum('nij,nj->ni', R_o, xyz_o) + t_o

    depth = pt_cam[:, 2]
    valid_depth = depth > 0
    depth_safe = np.where(valid_depth, depth, 1.0)
    proj_u = focal * (pt_cam[:, 0] / depth_safe) + cx
    proj_v = fy    * (pt_cam[:, 1] / depth_safe) + cy
    err = np.sqrt((proj_u - us) ** 2 + (proj_v - vs) ** 2)
    err[~valid_depth] = 0.0

    total_obs_per_pt = np.bincount(pt_ids, minlength=N)
    vd_ids = pt_ids[valid_depth]
    neg_ids = pt_ids[~valid_depth]
    valid_count[:] = np.bincount(vd_ids, minlength=N)
    err_sum = np.bincount(vd_ids, weights=err[valid_depth], minlength=N)
    neg_depth_count = np.bincount(neg_ids, minlength=N)

    has_valid = valid_count > 0
    mean_err[has_valid] = err_sum[has_valid] / valid_count[has_valid]
    has_obs = total_obs_per_pt > 0
    neg_ratio[has_obs] = neg_depth_count[has_obs] / total_obs_per_pt[has_obs]
    neg_ratio[invalid_pts] = 1.0

    return mean_err, valid_count, neg_ratio


# =========================================================================
# 修剪 / 过滤
# =========================================================================
def _prune_map_points(map_points, feat_map, frame_to_points, reproj_thresh,
                      frame_poses, focal, fy, cx, cy):
    """剔除观测不足、负深度为主或重投影误差过大的地图点，并重建索引。"""
    if not map_points:
        return

    n_old = len(map_points)
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

    # 用向量化的索引映射替换逐点字典重建
    keep_mask = np.ones(n_old, dtype=bool)
    keep_mask[np.asarray(to_remove, dtype=np.int64)] = False
    idx_map_arr = np.full(n_old, -1, dtype=np.int64)
    idx_map_arr[keep_mask] = np.arange(int(keep_mask.sum()), dtype=np.int64)

    # feat_map 重映射（向量化）：越界索引显式告警，避免静默错位掩盖
    # feat_map 与 map_points 不同步的 bug
    for f_idx in range(len(feat_map)):
        row = feat_map[f_idx]
        if not row:
            continue
        arr = np.asarray(row, dtype=np.int64)
        if arr.size == 0:
            continue
        oob = arr >= n_old
        if np.any(oob):
            for kp_idx in np.where(oob)[0]:
                logger.warning(
                    f"[prune] feat_map[{f_idx}][{int(kp_idx)}]={int(arr[kp_idx])} "
                    f"越界（n_old={n_old}），重置为 -1；"
                    f"这通常意味着 feat_map 与 map_points 不同步"
                )
            arr = np.where(oob, -1, arr)
        valid = arr >= 0
        arr_new = np.full(arr.shape, -1, dtype=np.int64)
        if np.any(valid):
            arr_new[valid] = idx_map_arr[arr[valid]]
        feat_map[f_idx] = arr_new.tolist()

    # frame_to_points 重建
    for f in list(frame_to_points.keys()):
        frame_to_points[f] = set()
    for old_idx in reversed(to_remove):
        del map_points[old_idx]
    for new_idx, pt in enumerate(map_points):
        pt['idx'] = new_idx
        for f_idx, _, _, _ in pt['obs']:
            frame_to_points[f_idx].add(new_idx)

    logger.info("[prune] 修剪 %d 个点，剩余 %d", len(to_remove), len(map_points))


def _filter_point_cloud(map_points, frame_poses, focal, fy, cx, cy,
                        reproj_thresh):
    """输出点云：剔除观测不足/负深度为主/误差过大的点。

    阈值 = max(中位误差 × 2.5, reproj_thresh × 1.5)。2.5× 中位是经验值；
    绝对下限防止低噪声场景把所有点都判为外点。
    """
    if not map_points:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)

    all_xyz = np.array([p['xyz'] for p in map_points])
    if all_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)

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