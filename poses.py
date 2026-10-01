"""
增量式 SfM 流水线

帧状态机：UNPROCESSED → TRACKED / KEYFRAME / LOST / RELOCATED
流程：顺序匹配 → 三角化/PnP → 局部 BA → 回环 → 全局 BA → PGO。
依赖 OpenCV + SciPy + NumPy；输出 SfMResult。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set, Iterable, Any
from collections import defaultdict

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.sparse import lil_matrix

logger = logging.getLogger(__name__)


# =========================================================================
# 常量与配置
# =========================================================================
EPS = 1e-8


class FrameStatus:
    UNPROCESSED = "unprocessed"
    TRACKED = "tracked"          # 帧间几何跟踪成功
    KEYFRAME = "keyframe"        # 被选为关键帧
    LOST = "lost"                # 跟踪失败
    RELOCATED = "relocated"      # 重定位恢复
    INVALID = "invalid"          # 特征太少，直接丢弃


# 特征
MIN_FEATURES = 60
ORB_FEATURES = 12000
SIFT_FEATURES = 12000
MIN_INLIERS = 25

MATCH_RATIO = 0.75
MATCH_DIST_ORB = 90
MATCH_DIST_SIFT = 400.0
MATCH_DIST_UPDATE_ORB = 35
MATCH_DIST_UPDATE_SIFT = 150.0

# 关键帧判定
KF_ANGLE_DEG = 5.0
KF_TRANS_RATIO = 0.05
KF_COVIS_RATIO = 0.25
KF_CULL_WINDOW = 10
KF_CULL_COVIS = 0.80

# 局部 PnP
PNP_WINDOW = 12
PNP_MIN_INLIERS = 12
PNP_REPROJ = 4.0

# 三角化
MIN_TRI_ANGLE_DEG = 1.0
MAX_REPROJ_ERROR = 4.0
MIN_OBSERVATIONS = 2
MULTIVIEW_MIN_OBS = 3
MULTIVIEW_TRIGGER_INTERVAL = 60
MULTIVIEW_RANSAC_ITERS = 50
MULTIVIEW_RANSAC_THRESH = 3.0

# 初始化
BASELINE_MIN_POS_RATIO = 0.5
BASELINE_MIN_ANGLE_DEG = 1.0

# BA
BA_LOCAL_ITERS = 20
BA_GLOBAL_ITERS = 50
BA_WINDOW = 8
BA_MAX_POINTS = 5000
BA_MAX_OBS = 10000
BA_MIN_OBS = 200
BA_LOCAL_TRIGGER_EVERY = 2  # 每新增 N 个关键帧触发一次局部 BA

# BA 防护
BA_MAX_INIT_RMS_RATIO = 3.0
BA_OVERFIT_MIN_RMS_LOCAL = 0.02
BA_OVERFIT_MIN_RMS_GLOBAL = 0.005
BA_OVERFIT_MIN_BEFORE = 0.15
BA_MAX_RMS_INCREASE = 1.2
FOCAL_MAX_STEP_RATIO = 0.05
FOCAL_GLOBAL_DRIFT_MAX = 0.15

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
SCALE_CLAMP = (0.25, 4.0)
SCALE_DEADBAND = 0.05

# 点云过滤
FILTER_MEDIAN_FACTOR = 3.0
FILTER_REPROJ_FACTOR = 2.0

# 回环检测
LOOP_VOCAB_SIZE = 400
LOOP_MIN_SCORE = 0.20
LOOP_MIN_INLIERS = 25
LOOP_SKIP_RECENT = 15
LOOP_MIN_GAP = 8
LOOP_MAX_CANDIDATES = 5

# 重定位
RELOC_MIN_MATCHES = 40
RELOC_MIN_INLIERS = 20
RELOC_SEARCH_KEYFRAMES = -1  # -1 表示搜索全部关键帧

# PGO
PGO_ITERS = 60
PGO_HUBER_DELTA = 1.0

# 内存
MAX_FRAMES_IN_MEMORY = 3000  # 超过后释放非关键帧描述子

_warned_keys: Set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    logger.warning(message)


# =========================================================================
# 数据结构
# =========================================================================
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Optional[np.ndarray] = None

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float32)

    @property
    def K64(self) -> np.ndarray:
        return self.K.astype(np.float64)


@dataclass
class CameraPose:
    """X_cam = R · X_world + t。"""
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
        return (-self.R.T @ self.t).flatten()

    def copy(self) -> "CameraPose":
        return CameraPose(self.R.copy(), self.t.copy())


@dataclass
class Observation:
    """单条观测。color 为 RGB uint8，供 3DGS 多视图颜色聚合。"""
    frame_idx: int
    kp_idx: int
    u: float
    v: float
    color: Optional[np.ndarray] = None


@dataclass
class MapPoint:
    idx: int
    xyz: np.ndarray
    desc: np.ndarray
    obs: List[Observation] = field(default_factory=list)
    obs_set: Set[Tuple[int, int]] = field(default_factory=set)
    desc_age: int = 0

    @property
    def obs_count(self) -> int:
        return len(self.obs)

    def add_observation(self, obs: Observation) -> bool:
        key = (obs.frame_idx, obs.kp_idx)
        if key in self.obs_set:
            return False
        self.obs_set.add(key)
        self.obs.append(obs)
        return True


@dataclass
class Frame:
    """单帧完整状态。"""
    idx: int
    path: str
    kps: List[cv2.KeyPoint] = field(default_factory=list)
    desc: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None  # (N, 3) uint8 RGB
    pose: Optional[CameraPose] = None
    status: str = FrameStatus.UNPROCESSED
    feat_map: Optional[np.ndarray] = None   # len(kps) int64；-1 表示未绑定
    bow_hist: Optional[np.ndarray] = None   # 回环检索直方图
    track_quality: float = 0.0              # 0~1，跟踪内点率
    is_keyframe: bool = False

    def __post_init__(self):
        if self.feat_map is None:
            self.feat_map = np.full(len(self.kps), -1, dtype=np.int64)


@dataclass
class SfMResult:
    intrinsics: CameraIntrinsics
    poses: List[CameraPose]
    xyz: np.ndarray
    rgb: np.ndarray
    obs_per_point: List[List[Tuple[int, int, float, float]]]
    keyframes: List[int]
    feature_type: str
    loop_closures: List[Tuple[int, int]] = field(default_factory=list)
    frame_status: Optional[List[str]] = None

    def __iter__(self):
        yield self.intrinsics
        yield self.poses
        yield self.xyz

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return (self.intrinsics, self.poses, self.xyz)[idx]


# =========================================================================
# 内参 / 颜色 / 畸变辅助
# =========================================================================
def _build_K(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _sample_colors(bgr: np.ndarray, kps) -> np.ndarray:
    if not kps:
        return np.zeros((0, 3), dtype=np.uint8)
    h, w = bgr.shape[:2]
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
    return bgr[ys, xs][:, ::-1].copy()


def _undistort_kps(kps, K: np.ndarray, dist):
    if dist is None or not kps:
        return kps
    pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
    try:
        undist = cv2.undistortPoints(
            pts, K.astype(np.float64),
            np.asarray(dist, dtype=np.float64).reshape(-1, 1),
            P=K.astype(np.float64),
        ).reshape(-1, 2)
    except cv2.error as e:
        _warn_once("undistort_fail", f"undistortPoints 失败（{e}）")
        return kps
    return [cv2.KeyPoint(float(x), float(y), kp.size, kp.angle,
                         kp.response, kp.octave, kp.class_id)
            for kp, (x, y) in zip(kps, undist)]


# =========================================================================
# 特征提取
# =========================================================================
def _make_detector(feature_type: str):
    if feature_type == "sift":
        try:
            det = cv2.SIFT_create(nfeatures=SIFT_FEATURES,
                                  contrastThreshold=0.03,
                                  edgeThreshold=10, sigma=1.6)
            return det, "sift"
        except cv2.error as e:
            logger.warning(f"SIFT 不可用（{e}），回退到 ORB。")
            feature_type = "orb"
    det = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2,
                         nlevels=8, edgeThreshold=31, patchSize=31)
    return det, feature_type


def extract_features(paths: List[str], feature_type: str = "orb",
                     dist_coeffs: Optional[np.ndarray] = None,
                     camera_matrix: Optional[np.ndarray] = None,
                     ) -> Tuple[List[Frame], Tuple[int, int], str]:
    detector, feature_type = _make_detector(feature_type)
    empty_desc = (np.zeros((0, 128), dtype=np.float32) if feature_type == "sift"
                  else np.zeros((0, 32), dtype=np.uint8))

    frames: List[Frame] = []
    img_shape = None
    K_und = None
    if camera_matrix is not None and dist_coeffs is not None:
        K_und = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{p}")
        if img_shape is None:
            img_shape = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        kp = kp if kp is not None else []
        if K_und is not None and dist_coeffs is not None:
            kp = _undistort_kps(kp, K_und, dist_coeffs)
        colors = _sample_colors(img, kp)
        if desc is None:
            desc = empty_desc
        frames.append(Frame(idx=i, path=p, kps=kp, desc=desc, colors=colors))
        del img, gray
    return frames, img_shape, feature_type


# =========================================================================
# 描述子度量与匹配
# =========================================================================
_MATCHER_CACHE: Dict[int, cv2.BFMatcher] = {}


def _get_bf(norm_type):
    bf = _MATCHER_CACHE.get(norm_type)
    if bf is None:
        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        _MATCHER_CACHE[norm_type] = bf
    return bf


def get_metric(desc: Optional[np.ndarray]):
    """返回 (norm_type, max_dist, update_dist)。"""
    if desc is None or len(desc) == 0:
        return cv2.NORM_HAMMING, MATCH_DIST_ORB, MATCH_DIST_UPDATE_ORB
    if desc.dtype != np.uint8:
        return cv2.NORM_L2, MATCH_DIST_SIFT, MATCH_DIST_UPDATE_SIFT
    return cv2.NORM_HAMMING, MATCH_DIST_ORB, MATCH_DIST_UPDATE_ORB


def match_descriptors(desc1, desc2, norm_type, max_dist,
                      ratio: float = MATCH_RATIO) -> List[cv2.DMatch]:
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    if len(desc2) < 2:
        bf_cross = cv2.BFMatcher(norm_type, crossCheck=True)
        return [m for m in bf_cross.match(desc1, desc2)
                if m.distance < max_dist]
    bf = _get_bf(norm_type)
    raw = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and m.distance < max_dist:
            good.append(m)
    return good


# =========================================================================
# 本质矩阵 / 基线检查
# =========================================================================
def _find_essential(pts1, pts2, K, threshold):
    K32 = K.astype(np.float32)
    try:
        return cv2.findEssentialMat(pts1, pts2, cameraMatrix=K32,
                                    method=cv2.RANSAC, prob=0.999,
                                    threshold=threshold)
    except (cv2.error, TypeError):
        pass
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    if abs(fx - fy) / max(fx, fy) > 1e-4:
        _warn_once("aspect_ratio_ignored",
                   "findEssentialMat 不支持 cameraMatrix 且 fx!=fy，按 fx=fy 近似。")
    return cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy),
                                method=cv2.RANSAC, prob=0.999,
                                threshold=threshold)


def _recover_pose(E, pts1, pts2, K, mask):
    K32 = K.astype(np.float32)
    try:
        return cv2.recoverPose(E, pts1, pts2, cameraMatrix=K32, mask=mask)
    except (cv2.error, TypeError):
        pass
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    if abs(fx - fy) / max(fx, fy) > 1e-4:
        _warn_once("aspect_ratio_ignored_recover",
                   "recoverPose 不支持 cameraMatrix 且 fx!=fy。")
    return cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy), mask=mask)


def _check_baseline_sufficient(pts_prev, pts_curr, R_rel, t_rel, K,
                               inlier_mask,
                               min_pos_ratio=BASELINE_MIN_POS_RATIO,
                               min_angle_deg=BASELINE_MIN_ANGLE_DEG) -> bool:
    """三角化后检查正深度比例与中位视差角。"""
    inlier_ids = np.where(inlier_mask)[0]
    if len(inlier_ids) < 8:
        return False
    t_norm = float(np.linalg.norm(t_rel))
    if t_norm < 1e-12:
        return False
    t_unit = (t_rel / t_norm).reshape(3)

    RT1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    RT2 = np.hstack([np.asarray(R_rel), t_unit.reshape(3, 1)])
    P1 = np.asarray(K) @ RT1
    P2 = np.asarray(K) @ RT2

    pp = pts_prev[inlier_ids].T.astype(np.float64)
    pc = pts_curr[inlier_ids].T.astype(np.float64)
    pts4d = cv2.triangulatePoints(P1, P2, pp, pc)
    w = pts4d[3]
    valid_w = np.abs(w) > 1e-12
    if int(valid_w.sum()) < 5:
        return False
    pts3d = (pts4d[:3] / np.where(valid_w, w, 1.0)).T

    cam_prev = np.zeros(3)
    cam_curr = (-np.asarray(R_rel).T @ t_unit).reshape(3)
    d1 = pts3d[:, 2]
    d2 = (pts3d - cam_curr) @ np.asarray(R_rel)[2]
    pos = valid_w & (d1 > 0) & (d2 > 0)
    if int(pos.sum()) < 5 or float(pos.mean()) < min_pos_ratio:
        return False

    v1 = pts3d[pos] - cam_prev
    v2 = pts3d[pos] - cam_curr
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    good = (n1 > 1e-12) & (n2 > 1e-12)
    if int(good.sum()) < 5:
        return False
    cos_a = np.sum(v1[good] * v2[good], axis=1) / (n1[good] * n2[good])
    angles = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    return float(np.median(angles)) >= min_angle_deg


# =========================================================================
# 回环检测（BoW + 余弦相似度）
# =========================================================================
class LoopDetector:
    """极简词袋：随机采样若干描述子作视觉词，最近邻量化后做余弦相似度。"""

    def __init__(self, vocab_size: int = LOOP_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.words: Optional[np.ndarray] = None
        self.hists: Dict[int, np.ndarray] = {}
        self._trained = False
        self._norm_type = cv2.NORM_HAMMING

    def train(self, keyframe_descs: List[np.ndarray], norm_type: int):
        pool = [d for d in keyframe_descs if d is not None and len(d) > 0]
        if not pool:
            return
        all_desc = np.vstack(pool)
        if len(all_desc) < self.vocab_size:
            return
        rng = np.random.default_rng(0)
        sel = rng.choice(len(all_desc), size=self.vocab_size, replace=False)
        self.words = all_desc[sel].copy()
        self._norm_type = norm_type
        self._trained = True
        logger.info(f"[loop] 词袋训练完成，词数={self.vocab_size}")

    def _quantize(self, desc: np.ndarray) -> np.ndarray:
        if self.words is None or len(desc) == 0:
            return np.zeros(self.vocab_size, dtype=np.float32)
        if self._norm_type == cv2.NORM_HAMMING:
            # 用 popcount 查表做汉明最近邻
            words = self.words.astype(np.uint8)
            d = desc.astype(np.uint8)
            xor = np.bitwise_xor(d[:, None, :], words[None, :, :])
            lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None],
                                axis=1).sum(axis=1).astype(np.uint8)
            dist = lut[xor].sum(axis=2)
        else:
            d = desc.astype(np.float32)
            w = self.words.astype(np.float32)
            dist = np.linalg.norm(d[:, None, :] - w[None, :, :], axis=2)
        assign = np.argmin(dist, axis=1)
        hist = np.bincount(assign, minlength=self.vocab_size).astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist

    def add(self, idx: int, desc: np.ndarray):
        if not self._trained:
            return
        self.hists[idx] = self._quantize(desc)

    def query(self, idx: int, desc: np.ndarray,
              keyframes: List[int],
              skip_recent: int = LOOP_SKIP_RECENT,
              min_score: float = LOOP_MIN_SCORE,
              max_candidates: int = LOOP_MAX_CANDIDATES,
              ) -> List[Tuple[int, float]]:
        if not self._trained:
            return []
        q = self._quantize(desc)
        if q.sum() <= 0:
            return []
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return []
        scores = []
        for kf in keyframes:
            if abs(kf - idx) < skip_recent:
                continue
            h = self.hists.get(kf)
            if h is None:
                continue
            hn = np.linalg.norm(h)
            if hn < 1e-12:
                continue
            s = float(np.dot(q, h) / (qn * hn))
            if s >= min_score:
                scores.append((kf, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:max_candidates]


# =========================================================================
# 位姿图优化（PGO）
# =========================================================================
def _se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) 对数映射，返回 6 维 [rho, phi]。"""
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)
    rvec, _ = cv2.Rodrigues(R)
    phi = rvec.flatten()
    theta = float(np.linalg.norm(phi))
    if theta < 1e-10:
        return np.concatenate([t, phi])
    phi_hat = cv2.Rodrigues(phi)[0]
    A = np.sin(theta) / theta
    B = (1 - np.cos(theta)) / theta
    C = (1 - A) / (theta * theta)
    V_inv = (np.eye(3) - 0.5 * phi_hat + C * (phi_hat @ phi_hat))
    rho = V_inv @ t
    return np.concatenate([rho, phi])


def _pose_to_T(pose: CameraPose) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose.R
    T[:3, 3] = pose.t.flatten()
    return T


def _T_to_pose(T: np.ndarray) -> CameraPose:
    return CameraPose(T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy())


def pose_graph_optimize(
    keyframes: List[int],
    poses: Dict[int, CameraPose],
    edges: List[Tuple[int, int, CameraPose]],
    iters: int = PGO_ITERS,
    huber_delta: float = PGO_HUBER_DELTA,
) -> Dict[int, CameraPose]:
    """位姿图优化。edges 中 (i, j, T_ij) 为 i→j 的相对位姿，固定首个关键帧。"""
    if not keyframes or len(edges) < 2:
        return poses

    kf_to_pos = {kf: i for i, kf in enumerate(keyframes)}
    n = len(keyframes)
    if n < 2:
        return poses

    # 参数：[rvec, t] × n，节点 0 固定
    param = np.zeros(6 * n, dtype=np.float64)
    for kf, i in kf_to_pos.items():
        rv, _ = cv2.Rodrigues(poses[kf].R)
        param[i * 6:i * 6 + 3] = rv.flatten()
        param[i * 6 + 3:i * 6 + 6] = poses[kf].t.flatten()

    edge_i = np.array([kf_to_pos[e[0]] for e in edges], dtype=np.int64)
    edge_j = np.array([kf_to_pos[e[1]] for e in edges], dtype=np.int64)
    T_meas = np.stack([_pose_to_T(e[2]) for e in edges])

    fixed_pos = 0

    def _params_to_T(params):
        T = np.empty((n, 4, 4), dtype=np.float64)
        for i in range(n):
            rv = params[i * 6:i * 6 + 3]
            R, _ = cv2.Rodrigues(rv)
            T[i, :3, :3] = R
            T[i, :3, 3] = params[i * 6 + 3:i * 6 + 6]
            T[i, 3, :] = [0, 0, 0, 1]
        return T

    def _residuals(params):
        params_fixed = params.copy()
        params_fixed[fixed_pos * 6:fixed_pos * 6 + 6] = param[
            fixed_pos * 6:fixed_pos * 6 + 6]
        T = _params_to_T(params_fixed)
        T_i = T[edge_i]
        T_j = T[edge_j]
        T_i_inv = np.linalg.inv(T_i)
        T_pred = np.einsum('nij,njk->nik', T_j, T_i_inv)
        T_err = np.einsum('nij,njk->nik', np.linalg.inv(T_meas), T_pred)
        res = np.empty((len(edges), 6), dtype=np.float64)
        for k in range(len(edges)):
            res[k] = _se3_log(T_err[k])
        return res.flatten()

    try:
        result = least_squares(
            _residuals, param, method='trf',
            loss='huber', f_scale=huber_delta,
            max_nfev=iters, verbose=0,
        )
        param = result.x
    except Exception as e:
        logger.warning(f"[PGO] 优化失败：{e}，保留原位姿。")
        return poses

    params_fixed = param.copy()
    params_fixed[fixed_pos * 6:fixed_pos * 6 + 6] = param[
        fixed_pos * 6:fixed_pos * 6 + 6]
    T = _params_to_T(params_fixed)

    new_poses = dict(poses)
    for kf, i in kf_to_pos.items():
        new_poses[kf] = _T_to_pose(T[i])
    logger.info(f"[PGO] 完成：{n} 个节点，{len(edges)} 条边。")
    return new_poses


# =========================================================================
# 多视图三角化（DLT + 正深度 + 视差角检查）
# =========================================================================
def _triangulate_dlt(obs_uvs: np.ndarray, proj_mats: List[np.ndarray]):
    """给定 (N,2) 观测与 N 个 (3,4) 投影矩阵，DLT 求 3D 点。"""
    A = []
    for (u, v), P in zip(obs_uvs, proj_mats):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A, dtype=np.float64)
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-12:
        return None
    X = X_h[:3] / X_h[3]
    if not np.isfinite(X).all():
        return None
    return X


def _retriangulate_multiview(map_points: List[MapPoint],
                             frames: Dict[int, Frame],
                             K: np.ndarray,
                             min_obs: int = MULTIVIEW_MIN_OBS,
                             ransac_iters: int = MULTIVIEW_RANSAC_ITERS,
                             ransac_thresh: float = MULTIVIEW_RANSAC_THRESH,
                             ) -> int:
    """对观测数 ≥ min_obs 的点做 RANSAC + DLT 重三角化。"""
    if not map_points:
        return 0
    K64 = np.asarray(K, dtype=np.float64)
    updated = 0

    for pt in map_points:
        obs = pt.obs
        if len(obs) < min_obs:
            continue

        proj_mats = []
        uvs = []
        pose_list = []
        for o in obs:
            f = frames.get(o.frame_idx)
            if f is None or f.pose is None:
                continue
            P = K64 @ np.asarray(f.pose.RT[:3], dtype=np.float64)
            proj_mats.append(P)
            uvs.append((o.u, o.v))
            pose_list.append(f.pose)
        if len(uvs) < min_obs:
            continue

        uvs = np.asarray(uvs, dtype=np.float64)

        n = len(uvs)
        best_inliers = -1
        best_X = None
        rng = np.random.default_rng(pt.idx)
        n_iters = min(ransac_iters, max(10, n * 3))
        for _ in range(n_iters):
            if n <= min_obs:
                subset = np.arange(n)
            else:
                subset = rng.choice(n, size=min_obs, replace=False)
            X = _triangulate_dlt(uvs[subset], [proj_mats[k] for k in subset])
            if X is None:
                continue
            inl = 0
            for k in range(n):
                P = proj_mats[k]
                proj = P @ np.append(X, 1.0)
                if abs(proj[2]) < 1e-12:
                    continue
                u, v = proj[0] / proj[2], proj[1] / proj[2]
                err = np.hypot(u - uvs[k, 0], v - uvs[k, 1])
                pt_cam = pose_list[k].R @ X + pose_list[k].t.flatten()
                if err < ransac_thresh and pt_cam[2] > 1e-6:
                    inl += 1
            if inl > best_inliers:
                best_inliers = inl
                best_X = X

        if best_X is None or best_inliers < max(min_obs, int(0.6 * n)):
            continue

        # 视差角检查
        centers = np.array([p.center for p in pose_list])
        v1 = best_X - centers
        norms = np.linalg.norm(v1, axis=1)
        if (norms < 1e-9).any():
            continue
        v1n = v1 / norms[:, None]
        max_angle = 0.0
        for a in range(len(v1n)):
            for b in range(a + 1, len(v1n)):
                c = float(np.clip(np.dot(v1n[a], v1n[b]), -1.0, 1.0))
                max_angle = max(max_angle, np.degrees(np.arccos(c)))
        if max_angle < MIN_TRI_ANGLE_DEG:
            continue

        old = pt.xyz
        if np.linalg.norm(best_X - old.astype(np.float64)) < 10.0 * max(1e-6, np.linalg.norm(old)):
            pt.xyz = best_X.astype(np.float32)
            updated += 1

    logger.info(f"[multiview] DLT 重三角化更新 {updated} / {len(map_points)} 个点")
    return updated


# =========================================================================
# EM-BA
# =========================================================================
def _em_e_step(res, obs_depths, sigma, pi_in, sigma_out, eps):
    """E 步：按残差与深度算软内点权重 gamma。"""
    n_obs = len(obs_depths)
    if len(res) != 2 * n_obs:
        raise RuntimeError("_em_e_step: res 长度与观测数不匹配")
    r2 = (res.reshape(n_obs, 2) ** 2).sum(axis=1)
    log_ratio = (np.log(pi_in / max(1.0 - pi_in, 1e-12))
                 + 2.0 * np.log(sigma_out / sigma)
                 + r2 * (1.0 / (2.0 * sigma_out ** 2)
                         - 1.0 / (2.0 * sigma ** 2)))
    gamma = expit(log_ratio)
    invalid = obs_depths <= eps
    gamma[invalid] = 1.0
    r2[invalid] = 0.0
    return gamma, r2


def _em_m_step(r2, gamma, sigma_cap):
    """M 步：更新 sigma 与 pi_in。"""
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
        denom = 2.0 * (np.sum(gamma) + 1e-12)
        sigma2 = float(np.sum(gamma * r2) / denom)

    sigma_new = float(np.clip(np.sqrt(max(sigma2, 1e-6)), 0.1, sigma_cap))
    return sigma_new, pi_in_new


def _compute_adaptive_eps(map_points, ref_pose, default_eps=0.01):
    """按参考相机深度中位数推算深度障碍。"""
    if ref_pose is None:
        return default_eps
    cam_center = ref_pose.center
    R = ref_pose.R
    depths = []
    for p in map_points:
        xyz = p.xyz
        if xyz.size == 3 and np.isfinite(xyz).all():
            d = float(R[2] @ (xyz - cam_center))
            if d > 0:
                depths.append(d)
    if depths:
        return float(np.clip(float(np.median(depths)) * 0.001,
                             EPS_MIN, EPS_MAX))
    return default_eps


# =========================================================================
# BA
# =========================================================================
def _build_ba_sparsity(n_obs, n_poses, n_points, optimize_points,
                       obs_fpos, obs_ppos, other_kfs, kf_to_pos):
    n_params = 1 + 6 * n_poses + (3 * n_points if optimize_points else 0)
    sparsity = lil_matrix((2 * n_obs, n_params), dtype=np.int8)
    sparsity[:, 0] = 1

    kf_pos_to_col = {kf_to_pos[idx]: 1 + i * 6
                     for i, idx in enumerate(other_kfs)}
    pts_col_start = 1 + n_poses * 6
    for i_obs in range(n_obs):
        r0, r1 = 2 * i_obs, 2 * i_obs + 1
        c0 = kf_pos_to_col.get(int(obs_fpos[i_obs]))
        if c0 is not None:
            for k in range(6):
                sparsity[r0, c0 + k] = 1
                sparsity[r1, c0 + k] = 1
        if optimize_points:
            pc = pts_col_start + 3 * int(obs_ppos[i_obs])
            for k in range(3):
                sparsity[r0, pc + k] = 1
                sparsity[r1, pc + k] = 1
    return sparsity.tocsr()


def bundle_adjustment(
    keyframe_ids: List[int],
    map_points: List[MapPoint],
    frames: Dict[int, Frame],
    focal: float, fy: float, cx: float, cy: float,
    optimize_points: bool = True,
    reproj_thresh: float = 1.0,
    image_size: int = 1920,
    max_iter: int = BA_LOCAL_ITERS,
    is_global: bool = False,
    focal_init: Optional[float] = None,
) -> Tuple[float, float]:
    """BA：EM 软内点加权 + SciPy 稀疏 LM/TRF。失败时回滚并返回原值。"""
    if focal_init is None:
        focal_init = focal
    sigma_cap = reproj_thresh * EM_SIGMA_CAP_RATIO
    sigma_out = reproj_thresh * EM_SIGMA_OUT_RATIO
    overfit_rms = (BA_OVERFIT_MIN_RMS_GLOBAL if is_global
                   else BA_OVERFIT_MIN_RMS_LOCAL)

    valid_kfs = [f for f in keyframe_ids if f in frames
                 and frames[f].pose is not None]
    if len(valid_kfs) < 2:
        return focal, fy
    keyframe_ids = valid_kfs

    obs = []
    for f_idx in keyframe_ids:
        frame = frames[f_idx]
        fm = frame.feat_map
        for kp_idx, pt_idx in enumerate(fm):
            if pt_idx >= 0 and pt_idx < len(map_points):
                u, v = frame.kps[kp_idx].pt
                obs.append((f_idx, int(pt_idx), float(u), float(v)))

    if len(obs) < BA_MIN_OBS:
        logger.info(f"[BA] 观测数 {len(obs)} < {BA_MIN_OBS}，跳过。")
        return focal, fy

    n_obs_orig = len(obs)
    rng = np.random.default_rng(42)
    if n_obs_orig > BA_MAX_OBS:
        idx = rng.choice(n_obs_orig, BA_MAX_OBS, replace=False)
        obs = [obs[i] for i in idx]
        logger.info(f"[BA] 抽样 {BA_MAX_OBS} / {n_obs_orig} 个观测。")

    obs_count = {f: 0 for f in keyframe_ids}
    for f, _, _, _ in obs:
        obs_count[f] += 1
    fixed_kf = max(obs_count, key=obs_count.get)
    other_kfs = [f for f in keyframe_ids if f != fixed_kf]

    fy_ratio = (fy / focal) if focal > 0 else 1.0
    eps = _compute_adaptive_eps(map_points, frames[fixed_kf].pose)

    param = [float(focal)]
    for idx in other_kfs:
        rv, _ = cv2.Rodrigues(frames[idx].pose.R)
        param.extend(rv.flatten())
        param.extend(frames[idx].pose.t.flatten())

    point_ids = []
    if optimize_points:
        point_ids = sorted({pt for _, pt, _, _ in obs})
        if len(point_ids) > BA_MAX_POINTS:
            cnt = {pid: 0 for pid in point_ids}
            for _, pid, _, _ in obs:
                cnt[pid] += 1
            point_ids = sorted(cnt.keys(), key=lambda x: cnt[x],
                               reverse=True)[:BA_MAX_POINTS]
        for pid in point_ids:
            param.extend(map_points[pid].xyz)
    else:
        known = set(range(len(map_points)))
        obs = [o for o in obs if o[1] in known]

    if optimize_points:
        pset = set(point_ids)
        obs = [o for o in obs if o[1] in pset]

    n_obs = len(obs)
    if n_obs < 10:
        return focal, fy

    # 边界
    scene_scale = 1.0
    if map_points:
        xs = np.array([p.xyz for p in map_points if p.xyz.size == 3])
        if len(xs):
            scene_scale = float(np.linalg.norm(xs.max(0) - xs.min(0))) + 1e-6
    cam_ts = [np.linalg.norm(frames[k].pose.t.flatten())
              for k in other_kfs]
    t_bound = max(scene_scale * 5.0, max(cam_ts, default=1.0) * 2.0, 10.0)

    focal_lo = max(focal * (1.0 - FOCAL_MAX_STEP_RATIO),
                   focal_init * (1.0 - FOCAL_GLOBAL_DRIFT_MAX))
    focal_hi = min(focal * (1.0 + FOCAL_MAX_STEP_RATIO),
                   focal_init * (1.0 + FOCAL_GLOBAL_DRIFT_MAX))
    if focal_lo >= focal_hi:
        logger.warning("[BA] 焦距边界交叉，跳过。")
        return focal, fy

    bounds_lower = [focal_lo]
    bounds_upper = [focal_hi]
    for _ in other_kfs:
        bounds_lower += [-np.inf] * 3 + [-t_bound] * 3
        bounds_upper += [np.inf] * 3 + [t_bound] * 3
    if optimize_points and point_ids:
        bounds_lower += [-np.inf] * len(point_ids) * 3
        bounds_upper += [np.inf] * len(point_ids) * 3

    n_poses = len(other_kfs)
    kf_to_pos = {kf: i for i, kf in enumerate(keyframe_ids)}
    n_kfs = len(keyframe_ids)
    fixed_pos = kf_to_pos[fixed_kf]

    all_pt_ids = (point_ids if (optimize_points and point_ids)
                  else sorted({o[1] for o in obs}))
    ptid_to_pos = {pid: j for j, pid in enumerate(all_pt_ids)}

    obs_fpos = np.fromiter((kf_to_pos[f] for f, _, _, _ in obs),
                           dtype=np.int64, count=n_obs)
    obs_ppos = np.fromiter((ptid_to_pos[pid] for _, pid, _, _ in obs),
                           dtype=np.int64, count=n_obs)
    obs_u = np.fromiter((u for _, _, u, _ in obs), dtype=np.float64,
                        count=n_obs)
    obs_v = np.fromiter((v for _, _, _, v in obs), dtype=np.float64,
                        count=n_obs)

    static_pts = None
    if not (optimize_points and point_ids):
        static_pts = np.zeros((len(all_pt_ids), 3), dtype=np.float64)
        for pid, j in ptid_to_pos.items():
            static_pts[j] = map_points[pid].xyz

    def _compute_residuals(params):
        f = params[0]
        fy_l = f * fy_ratio

        R_arr = np.empty((n_kfs, 3, 3), dtype=np.float64)
        t_arr = np.empty((n_kfs, 3), dtype=np.float64)
        R_arr[fixed_pos] = frames[fixed_kf].pose.R
        t_arr[fixed_pos] = frames[fixed_kf].pose.t.flatten()
        for i_kf, idx in enumerate(other_kfs):
            s = 1 + i_kf * 6
            R, _ = cv2.Rodrigues(params[s:s + 3])
            R_arr[kf_to_pos[idx]] = R
            t_arr[kf_to_pos[idx]] = params[s + 3:s + 6]

        if optimize_points and point_ids:
            ps = 1 + n_poses * 6
            n_pts = len(point_ids)
            pts = params[ps:ps + n_pts * 3].reshape(-1, 3)
        else:
            pts = static_pts

        xyz_o = pts[obs_ppos]
        R_o = R_arr[obs_fpos]
        t_o = t_arr[obs_fpos]
        pt_cam = np.einsum('nij,nj->ni', R_o, xyz_o) + t_o
        depth = pt_cam[:, 2]
        depth_safe = np.where(depth > eps, depth, 1.0)

        res = np.empty((n_obs, 2), dtype=np.float64)
        res[:, 0] = f * (pt_cam[:, 0] / depth_safe) + cx - obs_u
        res[:, 1] = fy_l * (pt_cam[:, 1] / depth_safe) + cy - obs_v

        # 深度障碍：让过浅的点产生强惩罚
        shallow = depth <= eps
        if np.any(shallow):
            d_sh = depth[shallow]
            very = d_sh <= 1e-6
            denom = np.where(very, 1e-6, eps)
            barrier = -np.log(np.maximum(d_sh / denom, 1e-10))
            res[shallow, 0] = barrier
            res[shallow, 1] = barrier
        return res.flatten(), depth

    param_np = np.asarray(param, dtype=np.float64)
    res0, depths0 = _compute_residuals(param_np)
    r2_0 = (res0.reshape(n_obs, 2) ** 2).sum(axis=1)
    valid0 = depths0 > eps
    if valid0.sum() == 0:
        logger.warning("[BA] 无有效深度观测，跳过。")
        return focal, fy

    rms_before = float(np.sqrt(np.median(r2_0[valid0]) / 2))
    if rms_before > reproj_thresh * BA_MAX_INIT_RMS_RATIO:
        logger.warning(f"[BA] 初值过差（中位 RMS={rms_before:.2f}px），跳过。")
        return focal, fy

    pre_poses = {k: frames[k].pose.copy() for k in other_kfs}
    pre_pts = {pid: map_points[pid].xyz.copy() for pid in point_ids}
    pre_focal = float(param_np[0])
    pre_fy = pre_focal * fy_ratio

    sigma = max(float(np.sqrt(np.median(r2_0[valid0]) / 2.0)), 0.5)
    pi_in = EM_PI_IN_INIT

    sparsity = _build_ba_sparsity(
        n_obs, n_poses,
        len(point_ids) if (optimize_points and point_ids) else 0,
        bool(optimize_points and point_ids),
        obs_fpos, obs_ppos, other_kfs, kf_to_pos,
    )

    logger.info(
        f"[BA] 开始：{len(keyframe_ids)} 关键帧，{n_obs} 观测，"
        f"σ0={sigma:.3f}px，π_in0={pi_in:.3f}，"
        f"初始中位 RMS={rms_before:.3f}px。"
    )

    result = None
    gamma = np.ones(n_obs, dtype=np.float64)

    for em_iter in range(EM_ITERS):
        gamma, r2 = _em_e_step(res0, depths0, sigma, pi_in, sigma_out, eps)
        gamma = np.maximum(gamma, EM_GAMMA_FLOOR)
        sqrt_g = np.sqrt(gamma)
        sqrt_g_rep = np.repeat(sqrt_g, 2)

        def _weighted(params, _sg=sqrt_g_rep):
            r, _ = _compute_residuals(params)
            return r * _sg

        iters_this = max_iter if em_iter == 0 else max(3, max_iter // 2)
        try:
            result = least_squares(
                _weighted, param_np,
                bounds=(bounds_lower, bounds_upper),
                method='trf', loss='linear',
                max_nfev=iters_this, verbose=0,
                ftol=1e-4, xtol=1e-4, gtol=1e-4,
                jac_sparsity=sparsity,
            )
            param_np = result.x
        except Exception as e:
            logger.warning(f"[BA] M 步异常：{e}")
            break

        res0, depths0 = _compute_residuals(param_np)
        r2 = (res0.reshape(n_obs, 2) ** 2).sum(axis=1)
        valid_now = depths0 > eps
        if valid_now.sum() > 0:
            sigma, pi_in = _em_m_step(r2[valid_now], gamma[valid_now],
                                      sigma_cap)

    if result is None:
        logger.warning("[BA] 无有效优化结果。")
        return focal, fy

    focal_new = max(float(param_np[0]), 1.0)
    fy_new = focal_new * fy_ratio

    res_f, dep_f = _compute_residuals(param_np)
    r2_f = (res_f.reshape(n_obs, 2) ** 2).sum(axis=1)
    valid_f = dep_f > eps
    rms_after = (float(np.sqrt(np.median(r2_f[valid_f]) / 2))
                 if valid_f.sum() > 0 else 0.0)

    # 过拟合 / 变差检测
    overfit = (rms_after < overfit_rms and rms_before > BA_OVERFIT_MIN_BEFORE)
    worse = rms_after > rms_before * BA_MAX_RMS_INCREASE
    if overfit or worse:
        reason = "疑似过拟合" if overfit else "结果变差"
        logger.warning(f"[BA] {reason}（RMS {rms_before:.3f}→{rms_after:.3f}），回滚。")
        for k in other_kfs:
            frames[k].pose = pre_poses[k]
        for pid in point_ids:
            map_points[pid].xyz = pre_pts[pid]
        return pre_focal, pre_fy

    for i_kf, idx in enumerate(other_kfs):
        s = 1 + i_kf * 6
        rv = param_np[s:s + 3]
        t = param_np[s + 3:s + 6]
        R, _ = cv2.Rodrigues(rv)
        frames[idx].pose = CameraPose(R, t.reshape(3, 1))

    if optimize_points and point_ids:
        ps = 1 + n_poses * 6
        for j, pid in enumerate(point_ids):
            map_points[pid].xyz = param_np[ps + j * 3:ps + j * 3 + 3]

    logger.info(
        f"[BA] 结束：focal {focal:.2f}→{focal_new:.2f}，"
        f"RMS {rms_before:.3f}→{rms_after:.3f}px。"
    )
    return focal_new, fy_new


# =========================================================================
# 地图点管理
# =========================================================================
def _compute_point_errors(map_points, poses_by_frame, focal, fy, cx, cy):
    """向量化计算每个点的平均重投影误差、有效观测数、负深度比例。"""
    N = len(map_points)
    mean_err = np.full(N, np.inf, dtype=np.float64)
    valid_count = np.zeros(N, dtype=np.int32)
    neg_ratio = np.zeros(N, dtype=np.float64)
    if N == 0:
        return mean_err, valid_count, neg_ratio

    xyz_all = np.full((N, 3), np.nan, dtype=np.float64)
    pt_ids, f_ids, us, vs = [], [], [], []
    for i, pt in enumerate(map_points):
        xyz = pt.xyz
        if xyz.size != 3:
            continue
        xyz_all[i] = xyz
        for o in pt.obs:
            pt_ids.append(i)
            f_ids.append(o.frame_idx)
            us.append(o.u)
            vs.append(o.v)

    invalid_pts = ~np.isfinite(xyz_all).all(axis=1)
    neg_ratio[invalid_pts] = 1.0
    if not pt_ids:
        return mean_err, valid_count, neg_ratio

    pt_ids = np.asarray(pt_ids, dtype=np.int64)
    f_ids = np.asarray(f_ids, dtype=np.int64)
    us = np.asarray(us, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)

    max_f = int(f_ids.max()) + 1 if len(f_ids) else 0
    R_all = np.zeros((max_f, 3, 3))
    t_all = np.zeros((max_f, 3))
    pose_ok = np.zeros(max_f, dtype=bool)
    for f, pose in poses_by_frame.items():
        if f < max_f and pose is not None:
            R_all[f] = pose.R
            t_all[f] = pose.t.flatten()
            pose_ok[f] = True

    keep = (~invalid_pts[pt_ids]) & pose_ok[f_ids]
    if not np.any(keep):
        return mean_err, valid_count, neg_ratio
    pt_ids = pt_ids[keep]
    f_ids = f_ids[keep]
    us = us[keep]
    vs = vs[keep]

    xyz_o = xyz_all[pt_ids]
    R_o = R_all[f_ids]
    t_o = t_all[f_ids]
    pt_cam = np.einsum('nij,nj->ni', R_o, xyz_o) + t_o
    depth = pt_cam[:, 2]
    valid_depth = depth > 0
    depth_safe = np.where(valid_depth, depth, 1.0)
    pu = focal * (pt_cam[:, 0] / depth_safe) + cx
    pv = fy * (pt_cam[:, 1] / depth_safe) + cy
    err = np.sqrt((pu - us) ** 2 + (pv - vs) ** 2)
    err[~valid_depth] = 0.0

    total = np.bincount(pt_ids, minlength=N)
    vd = pt_ids[valid_depth]
    neg = pt_ids[~valid_depth]
    valid_count[:] = np.bincount(vd, minlength=N)
    err_sum = np.bincount(vd, weights=err[valid_depth], minlength=N)
    neg_count = np.bincount(neg, minlength=N)

    has_valid = valid_count > 0
    mean_err[has_valid] = err_sum[has_valid] / valid_count[has_valid]
    has_obs = total > 0
    neg_ratio[has_obs] = neg_count[has_obs] / total[has_obs]
    neg_ratio[invalid_pts] = 1.0
    return mean_err, valid_count, neg_ratio


def prune_map_points(map_points: List[MapPoint], frames: Dict[int, Frame],
                     reproj_thresh, focal, fy, cx, cy):
    """剔除观测不足、深度多为负、重投影误差过大的点，同步更新 feat_map。"""
    if not map_points:
        return
    n_old = len(map_points)
    poses_by_frame = {f: fr.pose for f, fr in frames.items()}
    mean_err, valid_count, neg_ratio = _compute_point_errors(
        map_points, poses_by_frame, focal, fy, cx, cy)
    err_thresh = MAX_REPROJ_ERROR * reproj_thresh

    to_remove = []
    for i, pt in enumerate(map_points):
        if pt.obs_count < MIN_OBSERVATIONS:
            to_remove.append(i)
        elif neg_ratio[i] >= 0.5:
            to_remove.append(i)
        elif valid_count[i] > 0 and mean_err[i] > err_thresh:
            to_remove.append(i)

    if not to_remove:
        return

    keep = np.ones(n_old, dtype=bool)
    keep[np.asarray(to_remove, dtype=np.int64)] = False
    idx_map = np.full(n_old, -1, dtype=np.int64)
    idx_map[keep] = np.arange(int(keep.sum()))

    for frame in frames.values():
        fm = frame.feat_map
        if fm is None or fm.size == 0:
            continue
        oob = fm >= n_old
        if np.any(oob):
            fm = np.where(oob, -1, fm)
        valid = fm >= 0
        new_fm = np.full(fm.shape, -1, dtype=np.int64)
        if np.any(valid):
            new_fm[valid] = idx_map[fm[valid]]
        frame.feat_map = new_fm

    new_points = [p for i, p in enumerate(map_points) if keep[i]]
    for new_idx, pt in enumerate(new_points):
        pt.idx = new_idx
    map_points[:] = new_points

    logger.info(f"[prune] 删除 {len(to_remove)} 个点，剩余 {len(map_points)}。")


def filter_point_cloud(map_points: List[MapPoint],
                       poses_by_frame: Dict[int, CameraPose],
                       focal, fy, cx, cy, reproj_thresh):
    """按中位误差倍数 + 阈值下限过滤点云，返回 (xyz, keep_mask)。"""
    if not map_points:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)
    xyz = np.array([p.xyz for p in map_points], dtype=np.float32)
    mean_err, valid_count, neg_ratio = _compute_point_errors(
        map_points, poses_by_frame, focal, fy, cx, cy)
    errors = np.where((valid_count >= MIN_OBSERVATIONS) & (neg_ratio < 0.5),
                      mean_err, np.inf)
    finite = np.isfinite(errors)
    if not np.any(finite):
        return xyz, np.zeros(len(map_points), dtype=bool)
    median = float(np.median(errors[finite]))
    thresh = max(median * FILTER_MEDIAN_FACTOR,
                 reproj_thresh * FILTER_REPROJ_FACTOR)
    mask = finite & (errors < thresh)
    return xyz, mask


# =========================================================================
# 主流水线
# =========================================================================
def estimate_poses(
    frame_paths: List[str],
    *,
    min_inliers: int = MIN_INLIERS,
    feature_type: str = "orb",
    focal_guess: Optional[float] = None,
    aspect_ratio: float = 1.0,
    dist_coeffs: Optional[np.ndarray] = None,
    camera_matrix: Optional[np.ndarray] = None,
    enable_loop: bool = True,
    enable_pgo: bool = True,
) -> SfMResult:
    """从有序图像序列估计相机位姿与稀疏点云。"""
    if not frame_paths:
        raise ValueError("frame_paths 不能为空")
    if len(frame_paths) < 2:
        raise ValueError("至少需要 2 帧")

    t_start = time.time()
    logger.info("=== 提取特征 ===")
    frames_list, img_shape, feature_type = extract_features(
        frame_paths, feature_type, dist_coeffs, camera_matrix)
    frames = {f.idx: f for f in frames_list}

    h, w = img_shape
    if camera_matrix is not None:
        K_in = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        fx = float(K_in[0, 0]); fy = float(K_in[1, 1])
        cx = float(K_in[0, 2]); cy = float(K_in[1, 2])
    else:
        cx, cy = w / 2.0, h / 2.0
        fx = focal_guess if focal_guess is not None else max(w, h) * 1.2
        fy = fx * aspect_ratio

    focal0 = float(fx)
    fy0 = float(fy)
    focal_init = focal0
    image_size = max(w, h)

    reproj_thresh = max(0.5, min(3.0, image_size * 0.0015))
    ransac_thresh = reproj_thresh * 0.8
    triang_thresh = reproj_thresh

    logger.info(f"阈值：reproj={reproj_thresh:.2f}, ransac={ransac_thresh:.2f}")

    norm_type, match_dist, update_dist = get_metric(
        next((f.desc for f in frames_list if f.desc is not None and len(f.desc)),
             None))

    map_points: List[MapPoint] = []
    keyframes: List[int] = []
    loop_detector = LoopDetector()
    pose_edges: List[Tuple[int, int, CameraPose]] = []  # (i, j, T_ij)
    loop_closures: List[Tuple[int, int]] = []

    # 首帧固定为单位位姿
    frames[0].pose = CameraPose(np.eye(3, dtype=np.float32),
                                np.zeros((3, 1), dtype=np.float32))
    frames[0].status = FrameStatus.KEYFRAME
    frames[0].is_keyframe = True
    keyframes.append(0)

    initialized = False
    last_good_frame = 0
    kf_since_ba = 0
    kf_since_multiview = 0

    # =====================================================================
    # 逐帧处理
    # =====================================================================
    for i in range(1, len(frame_paths)):
        t_frame = time.time()
        frame = frames[i]
        prev = frames[i - 1]

        if len(frame.kps) < MIN_FEATURES // 2:
            logger.warning(f"[帧 {i}] 特征过少（{len(frame.kps)}），标记 INVALID。")
            frame.status = FrameStatus.INVALID
            frame.pose = prev.pose
            continue

        # ---- 与上一帧匹配 ----
        matches = match_descriptors(prev.desc, frame.desc, norm_type,
                                    match_dist)
        track_quality = 0.0
        has_pose = False

        if len(matches) >= min_inliers:
            pts_prev = np.array([prev.kps[m.queryIdx].pt for m in matches],
                                dtype=np.float32)
            pts_curr = np.array([frame.kps[m.trainIdx].pt for m in matches],
                                dtype=np.float32)
            K_init = _build_K(focal0, fy0, cx, cy)
            E, mask = _find_essential(pts_prev, pts_curr, K_init,
                                      ransac_thresh)
            if E is not None and int(mask.sum()) >= min_inliers:
                _, R_rel, t_rel, mask_pose = _recover_pose(
                    E, pts_prev, pts_curr, K_init, mask)
                pose_inlier = mask_pose.ravel().astype(bool)
                if int(pose_inlier.sum()) >= min_inliers:
                    has_baseline = _check_baseline_sufficient(
                        pts_prev, pts_curr, R_rel, t_rel, K_init, pose_inlier)
                    if has_baseline or not initialized:
                        if has_baseline:
                            R_curr = R_rel @ prev.pose.R
                            t_curr = R_rel @ prev.pose.t + t_rel

                            if not initialized:
                                norm_t = float(np.linalg.norm(t_curr))
                                if norm_t > 1e-12:
                                    t_curr = t_curr / norm_t
                                    frame.pose = CameraPose(R_curr, t_curr)
                                    initialized = True
                                    frame.status = FrameStatus.KEYFRAME
                                    frame.is_keyframe = True
                                    _triangulate_from_pair(
                                        i, i - 1, matches, pose_inlier,
                                        frames, map_points,
                                        focal0, fy0, cx, cy,
                                        triang_thresh)
                                    keyframes.append(i)
                                    frame.track_quality = float(
                                        pose_inlier.sum() / len(matches))
                                    logger.info(f"初始化成功（帧 {i-1} + {i}）")
                                    continue
                            else:
                                frame.pose = CameraPose(R_curr, t_curr)
                                frame.status = FrameStatus.TRACKED
                                track_quality = float(
                                    pose_inlier.sum() / len(matches))
                                has_pose = True

        if not initialized:
            frame.status = FrameStatus.LOST
            frame.pose = prev.pose
            continue

        # ---- 三角化新点 ----
        if has_pose:
            _triangulate_from_pair(
                i, i - 1, matches,
                mask_pose.ravel().astype(bool) if len(matches) else np.zeros(0, bool),
                frames, map_points,
                focal0, fy0, cx, cy, triang_thresh)
            frame.track_quality = track_quality

        # ---- PnP 重定位（局部地图） ----
        if len(keyframes) > 1:
            pose_pnp = _try_pnp_relocalize(frame, keyframes, map_points,
                                           frames, focal0, fy0, cx, cy,
                                           reproj_thresh)
            if pose_pnp is not None:
                frame.pose = pose_pnp
                if frame.status != FrameStatus.KEYFRAME:
                    frame.status = FrameStatus.TRACKED
                has_pose = True

        # ---- 全局重定位 ----
        if (not has_pose) or frame.track_quality < 0.15:
            pose_reloc = _try_global_relocalization(
                frame, keyframes, map_points, frames, loop_detector,
                focal0, fy0, cx, cy, reproj_thresh)
            if pose_reloc is not None:
                frame.pose = pose_reloc
                frame.status = FrameStatus.RELOCATED
                logger.info(f"[帧 {i}] 全局重定位成功。")
                has_pose = True

        if not has_pose and frame.pose is None:
            frame.status = FrameStatus.LOST
            frame.pose = frames[last_good_frame].pose
            continue

        last_good_frame = i

        # ---- 关键帧判定 ----
        last_kf = keyframes[-1]
        last_kf_pose = frames[last_kf].pose
        R_delta = frame.pose.R @ last_kf_pose.R.T
        angle = _rotation_angle_deg(R_delta)
        delta_c = frame.pose.center - last_kf_pose.center

        if len(map_points) >= 5:
            recent = np.array([p.xyz for p in map_points[-200:]],
                              dtype=np.float32)
            c_r = recent.mean(axis=0)
            scene_ref = float(np.median(np.linalg.norm(
                recent - c_r, axis=1))) + 1e-6
        else:
            scene_ref = 1e-6

        trans_ratio = float(np.linalg.norm(delta_c) / scene_ref)
        covis_ratio = _compute_covis_ratio(i, keyframes, matches, frames)

        is_kf = (angle > KF_ANGLE_DEG or trans_ratio > KF_TRANS_RATIO
                 or covis_ratio < KF_COVIS_RATIO)

        if is_kf and len(map_points) > 20:
            _cull_keyframes(keyframes, frames, i)
            keyframes.append(i)
            frame.is_keyframe = True
            frame.status = FrameStatus.KEYFRAME
            kf_since_ba += 1
            kf_since_multiview += 1

            prev_kf = keyframes[-2] if len(keyframes) >= 2 else None
            if prev_kf is not None:
                T_rel = _relative_pose(frames[prev_kf].pose, frame.pose)
                pose_edges.append((prev_kf, i, T_rel))

            # 回环检测
            if enable_loop and len(keyframes) >= 5:
                if not loop_detector._trained and len(keyframes) >= 5:
                    descs = [frames[k].desc for k in keyframes]
                    loop_detector.train(descs, norm_type)
                    for k in keyframes:
                        loop_detector.add(k, frames[k].desc)
                else:
                    loop_detector.add(i, frame.desc)
                loops = _detect_and_verify_loops(
                    i, keyframes, frames, loop_detector, map_points,
                    focal0, fy0, cx, cy, reproj_thresh)
                for cand_kf, T_rel in loops:
                    pose_edges.append((cand_kf, i, T_rel))
                    loop_closures.append((cand_kf, i))
                    logger.info(f"[回环] 帧 {cand_kf} ↔ {i}")

            # 局部 BA
            if (kf_since_ba >= BA_LOCAL_TRIGGER_EVERY
                    and len(keyframes) >= BA_WINDOW):
                window = keyframes[-BA_WINDOW:]
                focal0, fy0 = bundle_adjustment(
                    window, map_points, frames,
                    focal0, fy0, cx, cy,
                    optimize_points=True,
                    reproj_thresh=reproj_thresh,
                    image_size=image_size,
                    max_iter=BA_LOCAL_ITERS,
                    is_global=False,
                    focal_init=focal_init)
                kf_since_ba = 0

        # ---- 周期性多视图三角化 ----
        if (kf_since_multiview >= MULTIVIEW_TRIGGER_INTERVAL
                and len(map_points) > 50):
            K_now = _build_K(focal0, fy0, cx, cy)
            _retriangulate_multiview(map_points, frames, K_now,
                                     min_obs=MULTIVIEW_MIN_OBS)
            kf_since_multiview = 0

        # ---- 周期性 prune ----
        if i % 200 == 0 and len(map_points) > 100:
            prune_map_points(map_points, frames, reproj_thresh,
                             focal0, fy0, cx, cy)

        dt = time.time() - t_frame
        logger.info(
            f"[帧 {i}/{len(frame_paths)-1}] 状态={frame.status} "
            f"跟踪质量={frame.track_quality:.2f} 点数={len(map_points)} "
            f"关键帧数={len(keyframes)} 耗时={dt:.2f}s"
        )

    if not initialized:
        raise RuntimeError("无法初始化 SfM：未找到具有足够平移的帧对。")

    # =====================================================================
    # 全局 BA + PGO
    # =====================================================================
    K_final = _build_K(focal0, fy0, cx, cy)
    _retriangulate_multiview(map_points, frames, K_final,
                             min_obs=MULTIVIEW_MIN_OBS)

    if len(keyframes) >= 3 and len(map_points) > 50:
        logger.info("=== 全局 BA ===")
        focal0, fy0 = bundle_adjustment(
            keyframes, map_points, frames,
            focal0, fy0, cx, cy,
            optimize_points=True,
            reproj_thresh=reproj_thresh,
            image_size=image_size,
            max_iter=BA_GLOBAL_ITERS,
            is_global=True,
            focal_init=focal_init)

    if enable_pgo and len(pose_edges) >= 3:
        logger.info("=== 位姿图优化 ===")
        poses_dict = {k: frames[k].pose for k in keyframes
                      if frames[k].pose is not None}
        new_poses = pose_graph_optimize(keyframes, poses_dict, pose_edges,
                                        iters=PGO_ITERS)
        for k, p in new_poses.items():
            frames[k].pose = p

    # PGO 后位姿变化，再重三角化一次
    _retriangulate_multiview(map_points, frames, K_final,
                             min_obs=MULTIVIEW_MIN_OBS)

    prune_map_points(map_points, frames, reproj_thresh,
                     focal0, fy0, cx, cy)

    # =====================================================================
    # 输出
    # =====================================================================
    all_poses: List[CameraPose] = []
    last_valid = frames[0].pose
    for i in range(len(frame_paths)):
        p = frames[i].pose
        if p is None:
            p = last_valid
        else:
            last_valid = p
        all_poses.append(p)

    poses_by_frame = {i: all_poses[i] for i in range(len(all_poses))}
    xyz, mask = filter_point_cloud(map_points, poses_by_frame,
                                   focal0, fy0, cx, cy, reproj_thresh)

    if xyz.size == 0 or not np.any(mask):
        xyz_out = np.zeros((0, 3), dtype=np.float32)
        rgb_out = np.zeros((0, 3), dtype=np.uint8)
        obs_out: List[List[Tuple[int, int, float, float]]] = []
    else:
        kept = [map_points[i] for i in np.where(mask)[0]]
        xyz_out = np.array([p.xyz for p in kept], dtype=np.float32)
        rgb_out = _aggregate_colors(kept)
        obs_out = _aggregate_observations(kept)

    intrinsics = CameraIntrinsics(
        fx=focal0, fy=fy0, cx=cx, cy=cy,
        dist=(np.asarray(dist_coeffs).ravel()
              if dist_coeffs is not None else None))

    logger.info(
        f"=== 结束：fx={focal0:.2f}, fy={fy0:.2f}, "
        f"点数={len(xyz_out)}, 关键帧={len(keyframes)}, "
        f"回环={len(loop_closures)}, "
        f"总耗时={time.time() - t_start:.1f}s ==="
    )

    return SfMResult(
        intrinsics=intrinsics,
        poses=all_poses,
        xyz=xyz_out,
        rgb=rgb_out,
        obs_per_point=obs_out,
        keyframes=list(keyframes),
        feature_type=feature_type,
        loop_closures=list(loop_closures),
        frame_status=[frames[i].status for i in range(len(frame_paths))],
    )


# =========================================================================
# 辅助：三角化 / 关键帧 / 回环 / 重定位 / 颜色
# =========================================================================
def _triangulate_from_pair(curr_idx, prev_idx, matches, inlier_mask,
                           frames, map_points,
                           focal, fy, cx, cy, triang_thresh):
    """两视图三角化 + 尺度归一化，观测中记录颜色。

    优先给已有点补观测（重投影误差合格时），其余匹配新三角化。
    """
    if prev_idx >= curr_idx:
        raise RuntimeError(f"prev_idx={prev_idx} 必须 < curr_idx={curr_idx}")
    prev = frames[prev_idx]
    curr = frames[curr_idx]
    if prev.pose is None or curr.pose is None:
        return

    K = _build_K(focal, fy, cx, cy)
    P_prev = K @ np.asarray(prev.pose.RT[:3], dtype=np.float64)
    P_curr = K @ np.asarray(curr.pose.RT[:3], dtype=np.float64)

    if len(inlier_mask) == 0 or len(matches) == 0:
        return
    if len(inlier_mask) != len(matches):
        inlier_mask = np.ones(len(matches), dtype=bool)

    R_curr = curr.pose.R
    t_curr = curr.pose.t
    err_thresh = MAX_REPROJ_ERROR * triang_thresh
    err_sq = err_thresh * err_thresh

    prev_feat_map = prev.feat_map
    curr_feat_map = curr.feat_map
    prev_colors = prev.colors
    curr_colors = curr.colors

    reuse_pairs: List[Tuple[int, int]] = []  # (match_idx, existing_pt_idx)
    new_match_ids: List[int] = []

    inlier_ids = np.where(inlier_mask)[0]
    for k in inlier_ids:
        m = matches[k]
        curr_feat = m.trainIdx
        if curr_feat_map[curr_feat] >= 0:
            continue
        prev_feat = m.queryIdx
        existing = prev_feat_map[prev_feat]
        if existing >= 0 and existing < len(map_points):
            xyz_e = map_points[existing].xyz
            pt_cam = R_curr @ xyz_e.reshape(3, 1) + t_curr
            d = float(pt_cam[2, 0])
            if d <= 1e-6:
                continue
            u = focal * float(pt_cam[0, 0]) / d + cx
            v = fy * float(pt_cam[1, 0]) / d + cy
            if (u - curr.kps[curr_feat].pt[0]) ** 2 + \
               (v - curr.kps[curr_feat].pt[1]) ** 2 > err_sq:
                continue
            reuse_pairs.append((int(k), int(existing)))
        else:
            new_match_ids.append(int(k))

    # 复用：给已有点加观测
    for k, existing in reuse_pairs:
        m = matches[k]
        pt = map_points[existing]
        c = (curr_colors[m.trainIdx]
             if curr_colors is not None and m.trainIdx < len(curr_colors)
             else None)
        pt.add_observation(Observation(
            frame_idx=curr_idx, kp_idx=int(m.trainIdx),
            u=float(curr.kps[m.trainIdx].pt[0]),
            v=float(curr.kps[m.trainIdx].pt[1]),
            color=(c.copy() if c is not None else None)))
        curr_feat_map[m.trainIdx] = existing
        _update_desc(pt, curr.desc[m.trainIdx])

    if not new_match_ids:
        return

    pts_prev = np.array([prev.kps[matches[k].queryIdx].pt
                         for k in new_match_ids], dtype=np.float64)
    pts_curr = np.array([curr.kps[matches[k].trainIdx].pt
                         for k in new_match_ids], dtype=np.float64)

    pts4d = cv2.triangulatePoints(P_prev, P_curr,
                                  pts_prev.T, pts_curr.T)
    w = pts4d[3]
    pts3d = (pts4d[:3] / (w + 1e-12)).T
    finite = np.isfinite(pts3d).all(axis=1)

    cam_prev_c = prev.pose.center
    cam_curr_c = curr.pose.center
    d_prev = pts3d @ prev.pose.R[2] - float(prev.pose.R[2] @ cam_prev_c)
    d_curr = pts3d @ curr.pose.R[2] - float(curr.pose.R[2] @ cam_curr_c)
    valid = finite & (d_prev > 0) & (d_curr > 0)

    v1 = pts3d - cam_prev_c
    v2 = pts3d - cam_curr_c
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos_a = np.sum(v1 * v2, axis=1) / (np.where(n1 > 1e-8, n1, 1.0) *
                                       np.where(n2 > 1e-8, n2, 1.0))
    cos_th = np.cos(np.radians(MIN_TRI_ANGLE_DEG))
    valid &= (n1 > 1e-8) & (n2 > 1e-8) & (cos_a <= cos_th)

    pts_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])
    pj_p = pts_h @ P_prev.T
    pj_c = pts_h @ P_curr.T
    pj_p = pj_p[:, :2] / (pj_p[:, 2:3] + 1e-12)
    pj_c = pj_c[:, :2] / (pj_c[:, 2:3] + 1e-12)
    err_p = np.linalg.norm(pj_p - pts_prev, axis=1)
    err_c = np.linalg.norm(pj_c - pts_curr, axis=1)
    valid &= (err_p <= triang_thresh) & (err_c <= triang_thresh)

    added_ids: List[int] = []
    for k in np.where(valid)[0]:
        mid = new_match_ids[k]
        m = matches[mid]
        prev_feat = m.queryIdx
        curr_feat = m.trainIdx
        pt_idx = len(map_points)

        c_prev = (prev_colors[prev_feat]
                  if prev_colors is not None and prev_feat < len(prev_colors)
                  else None)
        pt = MapPoint(
            idx=pt_idx,
            xyz=pts3d[k].astype(np.float32),
            desc=prev.desc[prev_feat].copy(),
        )
        pt.add_observation(Observation(
            frame_idx=prev_idx, kp_idx=int(prev_feat),
            u=float(prev.kps[prev_feat].pt[0]),
            v=float(prev.kps[prev_feat].pt[1]),
            color=(c_prev.copy() if c_prev is not None else None)))
        c_curr = (curr_colors[curr_feat]
                  if curr_colors is not None and curr_feat < len(curr_colors)
                  else None)
        pt.add_observation(Observation(
            frame_idx=curr_idx, kp_idx=int(curr_feat),
            u=float(curr.kps[curr_feat].pt[0]),
            v=float(curr.kps[curr_feat].pt[1]),
            color=(c_curr.copy() if c_curr is not None else None)))
        map_points.append(pt)
        prev_feat_map[prev_feat] = pt_idx
        curr_feat_map[curr_feat] = pt_idx
        added_ids.append(pt_idx)

    # 尺度归一化：以当前相机深度中位数为参考
    if added_ids and len(map_points) - len(added_ids) >= 10:
        cam_curr_flat = curr.pose.center
        R_curr_2 = curr.pose.R[2]
        c_curr = float(R_curr_2 @ cam_curr_flat)
        old_xyz = np.asarray(
            [p.xyz for p in map_points[:len(map_points) - len(added_ids)]
             if p.xyz.size == 3], dtype=np.float64)
        if len(old_xyz) >= 5:
            old_d = old_xyz @ R_curr_2 - c_curr
            old_d = old_d[old_d > 0]
            if len(old_d) >= 5:
                ref_med = float(np.median(old_d))
                new_xyz = np.asarray([map_points[pi].xyz for pi in added_ids],
                                     dtype=np.float64)
                new_d = new_xyz @ R_curr_2 - c_curr
                new_d = new_d[new_d > 0]
                if len(new_d) > 0:
                    new_med = float(np.median(new_d))
                    if new_med > 1e-8 and ref_med > 1e-8:
                        raw = ref_med / new_med
                        if abs(raw - 1.0) > SCALE_DEADBAND:
                            s = float(np.clip(raw, *SCALE_CLAMP))
                            origin = prev.pose.center
                            scaled = origin + s * (new_xyz - origin)
                            for k, pi in enumerate(added_ids):
                                map_points[pi].xyz = scaled[k].astype(np.float32)
                            cam_new = origin + s * (cam_curr_flat - origin)
                            t_new = -curr.pose.R @ cam_new.reshape(3, 1)
                            curr.pose = CameraPose(curr.pose.R,
                                                   t_new.astype(np.float32))


def _update_desc(pt: MapPoint, new_desc):
    """按观测数 / 描述子年龄 / 距离决定是否更新地图点描述子。"""
    old = pt.desc
    if old is None:
        pt.desc = new_desc.copy()
        pt.desc_age = 0
        return
    _, _, up_dist = get_metric(new_desc)
    if new_desc.dtype != old.dtype:
        pt.desc = new_desc.copy()
        pt.desc_age = 0
        return
    if old.dtype == np.uint8:
        d = int(cv2.norm(old, new_desc, cv2.NORM_HAMMING))
    else:
        d = float(cv2.norm(old, new_desc, cv2.NORM_L2))
    if pt.obs_count < 3 or pt.desc_age > 5 or d >= up_dist * 1.2:
        pt.desc = new_desc.copy()
        pt.desc_age = 0
    else:
        pt.desc_age += 1


def _rotation_angle_deg(R):
    rv, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(rv) * 180.0 / np.pi)


def _relative_pose(pose_i: CameraPose, pose_j: CameraPose) -> CameraPose:
    """T_ij：i→j 的相对位姿（j 系下表示）。"""
    R = pose_j.R @ pose_i.R.T
    t = pose_j.t - R @ pose_i.t
    return CameraPose(R, t)


def _compute_covis_ratio(curr_idx, keyframes, matches, frames):
    """当前帧与上一关键帧的共视点比例。"""
    if not matches or len(keyframes) < 2:
        return 1.0
    last_kf = keyframes[-1]
    if last_kf == curr_idx:
        return 1.0
    kf_fm = frames[last_kf].feat_map
    if kf_fm is None:
        return 0.0
    kf_pts = set(int(x) for x in kf_fm if x >= 0)
    if not kf_pts:
        return 0.0
    curr_fm = frames[curr_idx].feat_map
    if curr_fm is None:
        return 0.0

    total = 0
    count = 0
    for m in matches:
        t = m.trainIdx
        if t < 0 or t >= len(curr_fm):
            continue
        b = curr_fm[t]
        if b < 0:
            continue
        total += 1
        if int(b) in kf_pts:
            count += 1
    if total < 10:
        return 0.0
    return count / total


def _cull_keyframes(keyframes: List[int], frames, curr_idx):
    """共视度过高的旧关键帧从关键帧列表中移除（每调用最多移除一个）。"""
    if len(keyframes) <= KF_CULL_WINDOW:
        return
    recent = keyframes[-KF_CULL_WINDOW:-1]
    curr_fm = frames[curr_idx].feat_map
    if curr_fm is None:
        return
    curr_pts = set(int(x) for x in curr_fm if x >= 0)
    if not curr_pts:
        return
    for kf in recent:
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        kf_pts = set(int(x) for x in kf_fm if x >= 0)
        if not kf_pts:
            continue
        covis = len(curr_pts & kf_pts) / max(1, min(len(curr_pts),
                                                    len(kf_pts)))
        if covis > KF_CULL_COVIS:
            keyframes.remove(kf)
            frames[kf].is_keyframe = False
            break


def _try_pnp_relocalize(frame, keyframes, map_points, frames,
                        focal, fy, cx, cy, reproj_thresh):
    """用最近关键帧的局部地图做 PnP 位姿估计。"""
    if len(keyframes) <= 1:
        return None
    pts3d, pts2d = [], []
    for kf in keyframes[-PNP_WINDOW:]:
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        bound = np.where(kf_fm >= 0)[0]
        if len(bound) < 4:
            continue
        bound_desc = frames[kf].desc[bound]
        norm_type, md, _ = get_metric(bound_desc)
        matches = match_descriptors(bound_desc, frame.desc, norm_type, md)
        for m in matches:
            orig_feat = int(bound[m.queryIdx])
            pt_idx = int(kf_fm[orig_feat])
            if 0 <= pt_idx < len(map_points):
                pts3d.append(map_points[pt_idx].xyz)
                pts2d.append(frame.kps[m.trainIdx].pt)
    if len(pts3d) < 8:
        return None
    pts3d = np.asarray(pts3d, dtype=np.float32)
    pts2d = np.asarray(pts2d, dtype=np.float32)
    K = _build_K(focal, fy, cx, cy).astype(np.float32)
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            pts3d, pts2d, K, np.zeros(4),
            iterationsCount=80, reprojectionError=reproj_thresh,
            confidence=0.99)
    except cv2.error:
        return None
    if not ok or inl is None or len(inl) < PNP_MIN_INLIERS:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    # 相机离场景过远则拒绝
    centroid = pts3d.mean(axis=0)
    scene_scale = float(np.median(np.linalg.norm(pts3d - centroid, axis=1))) + 1e-6
    cam_c = (-R.T @ t).flatten()
    if np.linalg.norm(cam_c - centroid) > 10.0 * scene_scale:
        return None
    return CameraPose(R, t)


def _try_global_relocalization(frame, keyframes, map_points, frames,
                               loop_detector, focal, fy, cx, cy,
                               reproj_thresh):
    """BoW 全局检索候选关键帧 → 2D-3D 匹配 → PnP 恢复位姿。"""
    if not loop_detector._trained or len(keyframes) < 5:
        return None
    if frame.desc is None or len(frame.desc) == 0:
        return None
    cands = loop_detector.query(frame.idx, frame.desc, keyframes,
                                skip_recent=0, min_score=LOOP_MIN_SCORE,
                                max_candidates=LOOP_MAX_CANDIDATES)
    if not cands:
        return None
    pts3d, pts2d = [], []
    for kf, _score in cands:
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        bound = np.where(kf_fm >= 0)[0]
        if len(bound) < 4:
            continue
        bd = frames[kf].desc[bound]
        nt, md, _ = get_metric(bd)
        matches = match_descriptors(bd, frame.desc, nt, md)
        for m in matches:
            pt_idx = int(kf_fm[int(bound[m.queryIdx])])
            if 0 <= pt_idx < len(map_points):
                pts3d.append(map_points[pt_idx].xyz)
                pts2d.append(frame.kps[m.trainIdx].pt)
    if len(pts3d) < RELOC_MIN_MATCHES:
        return None
    pts3d = np.asarray(pts3d, dtype=np.float32)
    pts2d = np.asarray(pts2d, dtype=np.float32)
    K = _build_K(focal, fy, cx, cy).astype(np.float32)
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            pts3d, pts2d, K, np.zeros(4),
            iterationsCount=100, reprojectionError=reproj_thresh,
            confidence=0.99)
    except cv2.error:
        return None
    if not ok or inl is None or len(inl) < RELOC_MIN_INLIERS:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return CameraPose(R, tvec.reshape(3, 1))


def _detect_and_verify_loops(curr_idx, keyframes, frames, loop_detector,
                             map_points, focal, fy, cx, cy,
                             reproj_thresh):
    """BoW 检索 + PnP RANSAC 几何验证，返回 [(kf, T_rel)]。"""
    frame = frames[curr_idx]
    if frame.desc is None or len(frame.desc) == 0:
        return []
    cands = loop_detector.query(curr_idx, frame.desc, keyframes,
                                skip_recent=LOOP_SKIP_RECENT,
                                min_score=LOOP_MIN_SCORE,
                                max_candidates=LOOP_MAX_CANDIDATES)
    results = []
    for kf, score in cands:
        if abs(kf - curr_idx) < LOOP_MIN_GAP:
            continue
        kf_fm = frames[kf].feat_map
        if kf_fm is None:
            continue
        bound = np.where(kf_fm >= 0)[0]
        if len(bound) < 4:
            continue
        bd = frames[kf].desc[bound]
        nt, md, _ = get_metric(bd)
        matches = match_descriptors(bd, frame.desc, nt, md)
        if len(matches) < LOOP_MIN_INLIERS:
            continue
        pts3d, pts2d = [], []
        for m in matches:
            pt_idx = int(kf_fm[int(bound[m.queryIdx])])
            if 0 <= pt_idx < len(map_points):
                pts3d.append(map_points[pt_idx].xyz)
                pts2d.append(frame.kps[m.trainIdx].pt)
        if len(pts3d) < LOOP_MIN_INLIERS:
            continue
        pts3d = np.asarray(pts3d, dtype=np.float32)
        pts2d = np.asarray(pts2d, dtype=np.float32)
        K = _build_K(focal, fy, cx, cy).astype(np.float32)
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                pts3d, pts2d, K, np.zeros(4),
                iterationsCount=80, reprojectionError=reproj_thresh,
                confidence=0.99)
        except cv2.error:
            continue
        if not ok or inl is None or len(inl) < LOOP_MIN_INLIERS:
            continue
        R, _ = cv2.Rodrigues(rvec)
        pose_loop = CameraPose(R, tvec.reshape(3, 1))
        # 与当前位姿的一致性检查（避免误回环）
        if frame.pose is not None:
            dc = np.linalg.norm(pose_loop.center - frame.pose.center)
            recent = np.array([p.xyz for p in map_points[-200:]],
                              dtype=np.float32) if map_points else None
            if recent is not None and len(recent) > 5:
                cr = recent.mean(axis=0)
                scale = float(np.median(np.linalg.norm(
                    recent - cr, axis=1))) + 1e-6
                if dc > 5.0 * scale:
                    continue
        T_rel = _relative_pose(frames[kf].pose, pose_loop)
        results.append((kf, T_rel))
    return results


def _aggregate_colors(points: List[MapPoint]) -> np.ndarray:
    """多视图颜色聚合：逐通道取中位数，无颜色时填灰。"""
    out = np.full((len(points), 3), 128, dtype=np.uint8)
    for j, pt in enumerate(points):
        colors = [o.color for o in pt.obs if o.color is not None]
        if not colors:
            continue
        arr = np.stack(colors, axis=0).astype(np.float32)
        out[j] = np.clip(np.median(arr, axis=0), 0, 255).astype(np.uint8)
    return out


def _aggregate_observations(points: List[MapPoint]):
    out = []
    for pt in points:
        out.append([(o.frame_idx, o.kp_idx, o.u, o.v) for o in pt.obs])
    return out