"""
基于 ORB（或 SIFT）特征 + 增量式 SfM 的鲁棒相机位姿估计。

模块职责：
  - 特征提取（ORB / SIFT，按 dtype 自动选择 Hamming / L2 度量）
  - 本质矩阵初始化 + 局部地图 PnP 重定位
  - 带共视关系与冗余剔除的关键帧管理
  - 局部 / 全局光束法平差（BA，含地图点优化；外层套 GMM-EM 软内点加权）
  - 地图点修剪、过滤与输出

依赖：仅 OpenCV + SciPy + NumPy，不依赖 COLMAP。
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
# —— 帧级判定 ——
MIN_INLIERS = 25                     # 有效内点下限，低于此值的帧视为匹配失败
MIN_FEATURES = 80                    # 单帧最少特征数
KEYFRAME_ANGLE_DEG = 5.0             # 关键帧旋转阈值（相对最后一个关键帧，度）
KEYFRAME_TRANS_RATIO = 0.05          # 关键帧平移阈值（相对场景尺度）
COVIS_RATIO_THRESH = 0.25            # 共视比例阈值，低于此值触发新关键帧
KEYFRAME_CULLING_WINDOW = 10         # 冗余关键帧剔除的回看窗口
PNP_WINDOW = 12                      # PnP 使用最近多少个关键帧
SMALL_TRANSLATION = 1e-4             # 判定「纯旋转」的平移阈值
INIT_MIN_TRANSLATION = 0.01          # 判定「可用于初始化」的最小平移
MAX_INIT_CANDIDATES = 30             # 初始化候选帧队列上限，防止 O(n²) 配对
MAX_CANDIDATE_TRIES = 5              # 每次尝试的最新候选数
MIN_COVIS_MATCHES = 10               # 共视估计所需最少绑定匹配数

# —— 特征匹配 ——
MATCH_DIST = 90                      # ORB 匹配距离上限
DESC_UPDATE_THRESH = 35              # ORB 描述子更新距离阈值
SIFT_MATCH_DIST = 400.0              # SIFT 匹配距离上限
SIFT_DESC_UPDATE_THRESH = 150.0      # SIFT 描述子更新距离阈值

# —— 三角化 ——
MIN_TRI_ANGLE_DEG = 2.0              # 三角化最小视差角（度）

# —— 地图点维护 ——
PRUNE_INTERVAL = 200                 # 地图点修剪间隔（帧）
MIN_OBSERVATIONS = 2                 # 地图点最少观测数
MAX_REPROJ_ERROR = 4.0               # 重投影误差上限（倍 reproj_thresh）

# —— BA 迭代与规模 ——
BA_MAX_ITER = 15                     # 局部 BA 最大迭代
GLOBAL_BA_ITER = 25                  # 全局 BA 最大迭代
MIN_BA_WINDOW = 5                    # 触发局部 BA 所需的关键帧数
MAX_POINTS_IN_BA = 300               # BA 单次最多参与优化的点数
BA_MAX_OBS = 2000                    # BA 观测上限（超过则随机抽样）
BA_MIN_OBS = 200                     # BA 最少观测数，低于此值跳过

# —— BA 防护 ——
BA_MAX_INIT_RMS_RATIO = 3.0          # 初值中位 RMS 超过 reproj_thresh × ratio 则跳过
BA_OVERFIT_MIN_RMS = 0.02            # BA 后中位 RMS 低于此值视为可疑（局部 BA）
BA_OVERFIT_MIN_RMS_GLOBAL = 0.005    # 全局 BA 阶段放宽阈值（最终精修）
BA_OVERFIT_MIN_BEFORE = 0.15         # 且 BA 前中位 RMS 高于此值时才判过拟合
BA_MAX_RMS_INCREASE = 1.2            # BA 后中位 RMS 超过此倍数则视为变差并回滚
FOCAL_MAX_STEP_RATIO = 0.05          # BA 单步焦距最大变化比例
FOCAL_GLOBAL_DRIFT_MAX = 0.15        # 焦距相对初始值的累计漂移上限

# —— EM-BA ——
EM_ITERS = 3                         # 外层 EM 迭代次数
EM_PI_IN_INIT = 0.85                 # 内点比例初值
EM_PI_IN_MAX = 0.95                  # π_in 动态上限的基准值
EM_PI_IN_HARD_CAP = 0.99             # π_in 绝对上限，任何情况下不突破
EM_SIGMA_CAP_RATIO = 1.5             # σ 上限 = reproj_thresh × ratio
EM_SIGMA_OUT_RATIO = 8.0             # σ_out = reproj_thresh × ratio
EM_GAMMA_FLOOR = 1e-4                # γ 下限，防止权重完全归零

# —— 深度障碍 ——
EPS_MIN = 1e-8                       # BA 深度障碍下限
EPS_MAX = 0.1                        # BA 深度障碍上限

# —— 尺度归一化 ——
SCALE_CLAMP = (0.5, 2.0)             # 尺度修正裁剪区间
SCALE_DEADBAND = 0.05                # 尺度修正死区（避免抖动）

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Pose] %(message)s")


# =========================================================================
# 数据结构
# =========================================================================
@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参（fx, fy, cx, cy）。"""
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
    """相机外参：世界→相机的 R、t（X_cam = R · X_world + t）。"""
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
        """相机光心的世界坐标：C = -R^T · t。"""
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

    参数：
        frame_paths: 帧路径列表（按拍摄时间排序）
        min_inliers: 单帧有效内点下限
        feature_type: "orb" 或 "sift"（SIFT 不可用时自动回退 ORB）
        focal_guess: 焦距初值（像素），默认 max(w, h) * 1.2
        aspect_ratio: fy / fx 比例

    返回：
        (intrinsics, poses, xyz) — 内参、每帧位姿、稀疏点云

    焦距语义：
        focal_init  — 初始焦距（冻结），只用于 BA 内部的漂移保护。
        focal0      — 「当前焦距」，循环内会被 BA 更新，用于 E 矩阵 / PnP /
                      BA 输出的内参。像素空间阈值（reproj_thresh /
                      ransac_thresh）在循环外一次性算好后冻结，保持
                      「像素绝对容差」语义，不随 focal0 变动。
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
    focal_init = float(focal0)       # 冻结的初始焦距，供 BA 漂移保护
    fy0 = focal0 * aspect_ratio
    image_size = max(w, h)

    # 阈值按图像尺寸自适应，跨分辨率可用。冻结，不随 focal0 变动。
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

    # 描述子度量由 dtype 决定（ORB→Hamming，SIFT→L2）。
    # bf 供 knnMatch 使用（crossCheck=False），bf_cross 供 desc < 2 的
    # 退化场景使用（crossCheck=True）。
    norm_type, match_dist, _ = _descriptor_metric(desc_list)
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    bf_cross = cv2.BFMatcher(norm_type, crossCheck=True)

    initialized = False
    init_candidates: List[int] = []   # 待配对的候选帧（纯旋转 / 平移不足）
    ba_counter = 0

    for i in range(1, len(frame_paths)):
        logger.info(f"处理帧 {i}/{len(frame_paths)-1}")

        # 每轮从 frame_poses 读起，避免任何缓存与 BA 修正后的位姿分叉。
        # frame_poses[i-1] 一定有效：初值在第 0 帧设置，之后每轮末尾都会
        # 写回 frame_poses[i]。
        prev_pose = frame_poses[i - 1]

        # 特征过少的帧：无法可靠估计，沿用上一帧位姿。
        # 本帧不会登记进 feat_map，共视率会偏低，可能额外触发关键帧——
        # 这是可接受的权衡。
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

            # 纯旋转 / 平移不足：登记为候选，等后续出现足够基线的帧再配对。
            # 队列长度有上限，否则长时间无法初始化会退化成 O(n²) 匹配。
            init_candidates.append(i)
            if len(init_candidates) > MAX_INIT_CANDIDATES:
                init_candidates.pop(0)

            if len(init_candidates) >= 2:
                # 只尝试最新的 MAX_CANDIDATE_TRIES 个候选。全量尝试在
                # ORB 12000 特征时每帧都要跑最多 30 次 _match_features。
                # 旧候选基线更大但成功概率更低。
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
                    # prev_idx 必须与三角化的参考帧一致：显式传 idx_cand，
                    # 不用 i-1——否则观测错记帧、描述子取错、feat_map 错位。
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

            # 候选帧保留 R（小基线时 R 准，t 不可靠）。若退化为 I，
            # 后续配对三角化会用错位姿 → 明显误差。
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
        # PnP 用已知 3D 地图点恢复位姿，天然提供正确的场景尺度，抑制
        # 纯 E 矩阵链式累积的尺度漂移。只用最近 PNP_WINDOW 个关键帧。
        if len(keyframes) > 1:
            pts3d_local, pts2d_local = [], []
            for kf in keyframes[-PNP_WINDOW:]:
                matches_kf = _match_features(
                    desc_list[kf], desc_list[i], bf,
                    match_dist=match_dist, norm_type=norm_type,
                    bf_cross=bf_cross)
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
                    iterationsCount=50, reprojectionError=reproj_thresh,
                    confidence=0.95)
                if inliers_pnp is not None and len(inliers_pnp) >= 8:
                    R_pnp, _ = cv2.Rodrigues(rvec_pnp)
                    t_pnp = tvec_pnp.reshape(3, 1)
                    # 用局部点云质心 + 中位半径刻画场景尺度。不用 ||t_pnp||
                    # 与世界原点比较：相机远离原点后会失真。
                    centroid = pts3d_local.mean(axis=0)
                    scene_scale = float(np.median(np.linalg.norm(
                        pts3d_local - centroid, axis=1))) + 1e-6
                    cam_center = (-R_pnp.T @ t_pnp).flatten()
                    cam_offset = float(np.linalg.norm(cam_center - centroid))
                    if cam_offset < 10.0 * scene_scale:
                        new_pose = CameraPose(R_pnp, t_pnp)

        # 关键：BA 之前把位姿写回 frame_poses。否则本帧会被
        # _bundle_adjustment 的 valid_kfs 过滤掉，其观测在 BA 中被丢弃，
        # 位姿要等下一轮 BA 才第一次被修正。
        frame_poses[i] = new_pose

        # ----- 关键帧判定 -----
        # 旋转阈值量「相对最后一个关键帧」的累积旋转，而非相邻帧。
        # 用相邻帧会漏掉匀速旋转（每帧 <5° 但累积 >5°）并误判抖动帧。
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
        # 平移用相机光心之差：t 在旋转大时与实际位移偏离明显。
        trans_world = (np.linalg.norm(new_pose.center - prev_pose.center)
                       / scene_ref)
        covis_ratio = _compute_covisibility_ratio(
            i, keyframes, matches, feat_map, frame_to_points)

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
                # BA 可能修正了本帧位姿，读回以保持与 frame_poses 一致
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

    # 过滤后若为空，直接返回空点云——不回退到未过滤点云（那是最差的一批）。
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

    不把整幅原图常驻内存（200 帧 1080p 约 1.2GB），每帧读完立即释放。
    返回 (图像尺寸, 关键点列表, 描述子列表)。

    无特征帧填「与该特征类型匹配的空描述子」：
      - ORB  → (0, 32) uint8
      - SIFT → (0, 128) float32
    形状/类型一致让下游 _descriptor_metric 与 _match_features 的早退逻辑
    不产生歧义。
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
    """按实际描述子 dtype 返回 (cv2 范数类型, 匹配距离上限, 描述子更新阈值)。

    ORB 描述子为 uint8 二进制 → Hamming；
    SIFT 描述子为 float32（OpenCV 归一化到约 512 范数）→ L2。
    对 float 描述子用 Hamming 会得到随机匹配或直接断言崩溃。
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

    参数：
        desc1, desc2: 两帧描述子
        bf:          预建的 BFMatcher（crossCheck=False）
        ratio:       Lowe 比值阈值
        match_dist:  距离上限；None 时按 desc1 的 dtype 现算
        norm_type:   范数类型；None 时按 desc1 的 dtype 现算
        bf_cross:    预建的 BFMatcher（crossCheck=True），仅当 desc2 < 2
                     时使用；None 时惰性创建
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

    对每条匹配，查当前帧特征已绑定的地图点是否被 last_kf 观测过。
    应在 _triangulate_new_points 之后调用——那时 feat_map[curr_idx] 才
    含有本帧三角化得到的绑定。

    分母用「已绑定到某个地图点的当前帧匹配数」而非全部匹配数：三角化
    失败率高时用全部匹配数会系统性低估共视率，导致关键帧膨胀。
    绑定匹配过少（< MIN_COVIS_MATCHES）时返回 0，触发新关键帧。
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
    """按观测数和描述子年龄决定是否替换地图点描述子。

    观测少 / 描述子过旧 / 差异过大 → 替换；否则累积年龄。
    """
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
    """对当前帧的内点做三角化，新增或复用地图点，并做尺度归一化。

    参数：
        prev_idx:  三角化参考帧索引（必须 < curr_idx）
        pose_prev: prev_idx 的位姿
        pose_curr: curr_idx 的位姿（可能在尺度归一化时被调整）

    返回 (新增点索引列表, 可能被尺度归一化调整后的 pose_curr)。

    prev_idx 必须由调用方显式传入：正常增量路径为 i-1，候选配对路径为
    idx_cand。硬编码 curr_idx-1 会让候选路径的观测错记帧、描述子取错、
    feat_map 索引错位。
    """
    if prev_idx >= curr_idx:
        raise RuntimeError(
            f"_triangulate_new_points: prev_idx={prev_idx} 必须 < "
            f"curr_idx={curr_idx}"
        )

    K = np.array([[focal, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    P_prev = K @ pose_prev.RT[:3]
    P_curr = K @ pose_curr.RT[:3]
    new_indices = []

    for idx in np.where(inlier_mask)[0]:
        m = matches[idx]
        pt_prev = pts_prev[idx].reshape(2, 1)
        pt_curr = pts_curr[idx].reshape(2, 1)
        prev_feat = m.queryIdx
        curr_feat = m.trainIdx

        if feat_map[curr_idx][curr_feat] >= 0:
            continue

        # prev_idx 已有对应地图点 → 只添加观测，不重复三角化。
        # 但先校验重投影误差：漂移点可能已偏离当前观测，加进去会污染 BA。
        existing = feat_map[prev_idx][prev_feat]
        if existing >= 0:
            xyz_existing = map_points[existing]['xyz']
            pt_cam = pose_curr.R @ xyz_existing.reshape(3, 1) + pose_curr.t
            depth_check = float(pt_cam[2, 0])
            if depth_check <= 1e-6:
                continue
            uv_proj = np.array([
                focal * float(pt_cam[0, 0]) / depth_check + cx,
                fy * float(pt_cam[1, 0]) / depth_check + cy,
            ])
            if (np.linalg.norm(uv_proj - pt_curr.flatten())
                    > MAX_REPROJ_ERROR * triang_thresh):
                continue
            _add_observation(map_points[existing], curr_idx, curr_feat,
                             pt_curr.flatten(), frame_to_points)
            feat_map[curr_idx][curr_feat] = existing
            _update_map_point_descriptor(
                map_points[existing], desc_list[curr_idx][curr_feat])
            continue

        pts4d = cv2.triangulatePoints(P_prev, P_curr, pt_prev, pt_curr)
        pt3d = (pts4d[:3] / (float(pts4d[3][0]) + 1e-12)).flatten()

        if not _is_valid_point(pt3d, pose_prev, pose_curr,
                               triang_thresh, P_prev, P_curr, pt_prev, pt_curr):
            continue

        pt_idx = len(map_points)
        pt_dict = {
            'idx': pt_idx,
            'xyz': pt3d.astype(np.float32),
            'desc': desc_list[prev_idx][prev_feat].copy(),
            'obs': [],
            'obs_set': set(),
            'obs_count': 0,
            'desc_age': 0,
        }
        _add_observation(pt_dict, prev_idx, prev_feat,
                         pt_prev.flatten(), frame_to_points)
        _add_observation(pt_dict, curr_idx, curr_feat,
                         pt_curr.flatten(), frame_to_points)
        map_points.append(pt_dict)
        feat_map[prev_idx][prev_feat] = pt_idx
        feat_map[curr_idx][curr_feat] = pt_idx
        new_indices.append(pt_idx)

    # ----- 尺度归一化 -----
    # 每次本质矩阵恢复的 t_rel 是单位范数（up-to-scale），链式叠加
    #   t_curr = R_rel @ prev.t + t_rel
    # 会导致尺度随帧数累积漂移。这里把「本次新增点的相机 z 向深度中位数」
    # 对齐到「已有地图点的相机 z 向深度中位数」，并同步缩放当前相机位移。
    #
    # 两个关键点：
    # 1) 缩放相对上一帧相机中心，不是世界原点；否则越到后段越离谱。
    # 2) 「是否需要修正」必须在裁剪之前判断，否则 <0.5 的负向修正永远被
    #    裁剪成 0.5，等于禁用负向修正。
    if new_indices and map_points:
        old_count = len(map_points) - len(new_indices)
        if old_count >= 10:
            cam_curr_flat = pose_curr.center
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
                        if abs(raw_scale - 1.0) > SCALE_DEADBAND:
                            scale = float(np.clip(raw_scale, *SCALE_CLAMP))
                            origin = pose_prev.center
                            for pi in new_indices:
                                xyz = map_points[pi]['xyz']
                                map_points[pi]['xyz'] = (
                                    origin + scale * (xyz - origin)
                                ).astype(np.float32)
                            cam_curr_new = origin + scale * (cam_curr_flat - origin)
                            # 由相机中心反解 t：t = -R @ C
                            t_new = -pose_curr.R @ cam_curr_new.reshape(3, 1)
                            pose_curr = CameraPose(
                                pose_curr.R, t_new.astype(np.float32))

    return new_indices, pose_curr


def _is_valid_point(pt3d, pose_prev, pose_curr,
                    reproj_th, P_prev, P_curr, pt_prev, pt_curr):
    """三角化点的合法性检查：有限性、正深度、视差角、重投影误差。"""
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

    模型：每个观测的 2 维残差 r 来自混合高斯
        p(r) = π_in · N(0, σ²I) + (1 - π_in) · N(0, σ_out²I)
    γ = p(z_in | r)，用 log-odds 经 sigmoid 得到：
        log p(r|in) - log p(r|out)
          = log(π_in / (1-π_in)) + 2·log(σ_out/σ)
            + r² · (1/(2σ_out²) - 1/(2σ²))

    r² 项系数为负——残差越大越倾向外点。符号写反会让 γ 随残差升高而
    升高，最终塌缩到 1。

    深度不足（depth ≤ eps）的观测强制 γ=1：其残差是深度障碍项，不是
    真实重投影误差，不参与外点判定。

    返回 (gamma, r2)；r2 中深度不足的观测置 0。
    """
    n_obs = len(obs_depths)
    # res 由 _compute_residuals 产出，每个观测固定 2 个残差。
    # 用 raise 而非 assert：-O 下 assert 被剥离，而这个不变量一旦破坏
    # 会静默错位残差对（不报错但结果全错）。
    if len(res) != 2 * n_obs:
        raise RuntimeError(
            f"_em_e_step: res 长度 {len(res)} 与观测数 {n_obs} 不匹配（应为 2×）"
        )

    r2 = np.empty(n_obs, dtype=np.float64)
    for i in range(n_obs):
        r2[i] = res[2 * i] ** 2 + res[2 * i + 1] ** 2

    log_ratio = (np.log(pi_in / max(1.0 - pi_in, 1e-12))
                 + 2.0 * np.log(sigma_out / sigma)
                 + r2 * (1.0 / (2.0 * sigma_out ** 2)
                         - 1.0 / (2.0 * sigma ** 2)))
    # expit 按符号分段计算，log_ratio 很负时也不会溢出
    gamma = expit(log_ratio)

    invalid = obs_depths <= eps
    gamma[invalid] = 1.0
    r2[invalid] = 0.0

    return gamma, r2


def _em_m_step(r2, gamma, sigma_cap):
    """EM 的 M 步：加权重估 σ 和 π_in。

    σ 只从高置信内点（γ > 0.9）估计。若用全部 γ 加权，大残差观测即使
    γ 只有 0.5 也会贡献一半权重，把 σ 拉大；σ 变大后 γ 又降不下来，
    形成「塌缩」正反馈。用高置信子集可打破该循环。

    π_in 上限根据当前 γ 分布动态调整：高置信内点占比高说明匹配质量好，
    允许 π_in 更接近 1。r² 是 2 维残差平方和，E[r²] = 2σ²。
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
        # 高置信样本太少，退回到 γ 加权
        denom = 2.0 * (np.sum(gamma) + 1e-12)
        sigma2 = float(np.sum(gamma * r2) / denom)

    sigma_new = float(np.clip(np.sqrt(max(sigma2, 1e-6)), 0.1, sigma_cap))
    return sigma_new, pi_in_new


def _compute_adaptive_eps(map_points, ref_pose, default_eps=0.01):
    """按参考相机下的 z 向深度中位数确定深度障碍阈值。

    用 ||p||（到世界原点的距离）近似会在相机远离原点后失真。
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
    """局部 / 全局 BA，外层套 GMM-EM 软内点加权。

    参数化：[focal, (rv_kf, t_kf) * N_other_kf, (xyz_pt) * N_pts]
    焦距 fy 锁定为 focal * fy_ratio，避免与场景尺度耦合导致病态。

    每个观测先按混合高斯模型估计内点后验概率 γ，再用 sqrt(γ) 加权跑
    最小二乘（loss='linear'，鲁棒性完全由 γ 提供）。深度不足的观测强制
    γ=1，其残差（深度障碍项）保留完整权重。

    防护机制：
      - BA 最小观测门槛：观测数 < BA_MIN_OBS 时跳过
      - 全深度不足：直接跳过
      - BA 初值门槛：中位 RMS > reproj_thresh × BA_MAX_INIT_RMS_RATIO 时跳过
      - 焦距边界交叉：lo ≥ hi 时跳过
      - 焦距全局漂移保护：相对 focal_init 最多 ±FOCAL_GLOBAL_DRIFT_MAX
      - BA 后过拟合 / 变差检测：中位 RMS 异常低或异常升高时回滚
    """
    if focal_init is None:
        focal_init = focal

    sigma_cap = reproj_thresh * EM_SIGMA_CAP_RATIO
    sigma_out = reproj_thresh * EM_SIGMA_OUT_RATIO
    # 全局 BA 是最终精修，允许 RMS 降到更低；局部 BA 用较严阈值避免误判
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

    # 只保留可优化点的观测。参与残差的点必须可优化，否则带误差的静态
    # 点会持续贡献正 cost，优化器只能移动位姿和点「绕着」误差走。
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
    r_bound = np.inf   # 旋转向量范数可近 4π，任何有限边界都可能触发初值越界

    # 焦距边界：单步 ±FOCAL_MAX_STEP_RATIO 与相对初值 ±FOCAL_GLOBAL_DRIFT_MAX
    # 取交集。两者理论上可能交叉（例如 focal 已接近漂移上限时），
    # 交叉会导致 least_squares 因 lower > upper 抛异常，必须提前拦截。
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

    # ---------- 3. 残差闭包 ----------
    point_ids_local = list(point_ids)
    n_poses = len(other_kfs)

    def _compute_residuals(params):
        """返回 (res, obs_depths)。

        res 长度恒为 2 * n_obs；obs_depths 长度 n_obs，供 E 步判断深度。
        """
        f = params[0]
        fy_local = f * fy_ratio

        poses = {fixed_kf: frame_poses[fixed_kf]}
        for i_kf, idx in enumerate(other_kfs):
            start = 1 + i_kf * 6
            rv = params[start:start + 3]
            t = params[start + 3:start + 6]
            R, _ = cv2.Rodrigues(rv)
            poses[idx] = CameraPose(R, t.reshape(3, 1))

        pts = {}
        if point_ids_local:
            pts_start = 1 + n_poses * 6
            for j, pid in enumerate(point_ids_local):
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
                # 深度障碍：对深度过浅或为负的观测施加对数惩罚。这些观测
                # 在 E 步会被强制 γ=1，不参与外点判定。
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

    # ---------- 4. 初值检查 ----------
    param_np = np.array(param, dtype=np.float64)
    res0, depths0 = _compute_residuals(param_np)
    r2_0 = np.sum(res0.reshape(n_obs, 2) ** 2, axis=1)
    valid_mask = depths0 > eps
    if valid_mask.sum() == 0:
        logger.warning("[BA] 所有观测深度不足，跳过本次 BA")
        return focal, fy

    # 中位 RMS：残差重尾，中位数比平均值更能反映「典型观测」的拟合质量
    rms_before = float(np.sqrt(np.median(r2_0[valid_mask]) / 2))

    init_rms_limit = reproj_thresh * BA_MAX_INIT_RMS_RATIO
    if rms_before > init_rms_limit:
        logger.warning(
            f"[BA] 初值过差（中位 RMS={rms_before:.2f}px > 限 "
            f"{init_rms_limit:.2f}px），跳过本次 BA 以免破坏位姿和焦距"
        )
        return focal, fy

    # 保存 BA 前状态，用于过拟合 / 变差检测的回滚
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
        # ----- E 步：用当前 σ/π_in 算每个观测的内点后验概率 -----
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
                ftol=1e-4, xtol=1e-4, gtol=1e-4,
            )
            param_np = result.x
        except Exception as e:
            logger.warning(f"[BA] M 步异常：{e}")
            break

        # ----- 更新 σ 和 π_in（用新残差 + 当前 γ） -----
        res0, depths0 = _compute_residuals(param_np)
        r2 = np.sum(res0.reshape(n_obs, 2) ** 2, axis=1)
        valid_now = depths0 > eps
        if valid_now.sum() > 0:
            sigma, pi_in = _em_m_step(r2[valid_now], gamma[valid_now],
                                      sigma_cap)
            # inlier_rms：只统计 γ>0.5 的观测，反映内点拟合质量
            # weighted_rms：所有观测按 γ 加权，反映整体优化目标
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

    # BA 后统计（用于过拟合 / 变差检测）
    res_final, depths_final = _compute_residuals(param_np)
    r2_final = np.sum(res_final.reshape(n_obs, 2) ** 2, axis=1)
    valid_final = depths_final > eps
    if valid_final.sum() > 0:
        rms_after = float(np.sqrt(np.median(r2_final[valid_final]) / 2.0))
    else:
        rms_after = 0.0

    # 过拟合检测：BA 后中位 RMS 异常低（真实匹配噪声不可能低于阈值下限），
    # 或 BA 后 RMS 反而升高（BA 破坏了结果），都回滚。
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

    # 重算 γ 以匹配最终的 σ/π_in，让 inlier_ratio 反映最终参数。
    # 循环内的 γ 来自上一轮 E 步，与最终残差不同步。
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

    返回三个与 map_points 等长的数组：
        mean_err:    有效观测的平均重投影误差；xyz 非法或无有效观测时为 inf
        valid_count: 有效观测数（位姿存在且 depth > 0）
        neg_ratio:   负深度观测占已评估观测的比例；xyz 非法时记 1.0
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

    remove_set = set(to_remove)
    idx_map = {}
    new_idx = 0
    for old_idx in range(n_old):
        if old_idx in remove_set:
            idx_map[old_idx] = -1
        else:
            idx_map[old_idx] = new_idx
            new_idx += 1

    # feat_map 重映射：显式处理越界索引，避免静默错位掩盖同步问题。
    # 用 warning 而非 debug——feat_map 与 map_points 不同步是需要立刻
    # 发现的 bug。
    for f_idx in range(len(feat_map)):
        row = feat_map[f_idx]
        for kp_idx in range(len(row)):
            old = row[kp_idx]
            if old < 0:
                continue
            if old >= n_old:
                logger.warning(
                    f"[prune] feat_map[{f_idx}][{kp_idx}]={old} 越界"
                    f"（n_old={n_old}），重置为 -1；"
                    f"这通常意味着 feat_map 与 map_points 不同步"
                )
                row[kp_idx] = -1
                continue
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


def _filter_point_cloud(map_points, frame_poses, focal, fy, cx, cy,
                        reproj_thresh):
    """输出点云：剔除观测不足 / 负深度为主 / 误差过大的点。

    阈值 = max(中位误差 × 2.5, reproj_thresh × 1.5)。
    经验上 >2.5× 中位的点大概率是外点；同时给一个绝对下限，防止在极低
    噪声场景（中位误差极小时）把所有点都判为外点。
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