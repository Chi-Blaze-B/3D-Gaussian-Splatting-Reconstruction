"""
3D Gaussian Splatting — core representation, rasterizer, and training.
Pure PyTorch implementation (no CUDA extension required).
Supports SH up to degree 3.
"""

import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Union

# ---------- Hyperparameters ----------
LR_POSITIONS = 1.6e-4
LR_LOG_SCALES = 5.0e-3
LR_OPACITIES = 5.0e-2
LR_ROTATIONS = 1.0e-3
LR_SH = 2.5e-3
LR_FOCAL = 1.0e-5
LR_K1 = 1.0e-5

DENSIFY_EVERY = 100
PRUNE_EVERY = 1000
GRAD_THRESH_BASE = 0.0002
SCALE_THRESH = 0.01
MIN_OPACITY = 0.005
MAX_GAUSSIANS = 1_000_000
SH_WARMUP_STEPS = 1000
SSIM_WARMUP_STEPS = 500
SSIM_WEIGHT_MAX = 0.2
GRAD_CLIP_NORM = 10.0
LOSS_THRESHOLD = 1.0
CHECKPOINT_INTERVAL_STEPS = 500
LR_DECAY_STEPS = 1000
LR_DECAY_GAMMA = 0.998
USE_LR_SCHEDULE = True
# 光栅化器分块尺寸：把逐像素合成按深度有序高斯切块，显存上界 = RASTER_CHUNK × (2·max_radius+1)²
RASTER_CHUNK = 512


# ---------- Frame loader ----------
def _load_frame_from_path(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_frame_raw(path: str) -> np.ndarray:
    """读帧为 uint8 RGB（内存为 float32 的 1/4），供预加载缓存使用。"""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class LazyFrames:
    """帧容器：内存缓存 + 按需转 float32。

    背景：训练每 epoch 顺序遍历全部帧，读盘 + 解码是主要开销（实测单帧
    PNG 解码约 100ms，且 1000 轮 × 200 帧 = 20 万次磁盘读取，伤硬盘）。

    方案：
    - **预加载到内存**：构造时把全部帧解码为 uint8 RGB（200 帧约 5GB，
      仅为 float32 的 1/4）。训练期间零磁盘读取。
    - **按需转 float32**：访问时 `uint8.astype(np.float32)/255.0`（约 46ms/帧，
      远快于读盘解码），避免全量 float32（200 帧约 20GB）的内存压力。
    - `preload=False` 时回退为惰性加载（不预载，首次访问才读盘），
      供内存不足场景使用。
    """
    def __init__(self, sources: List[Union[str, np.ndarray]], preload: bool = True,
                 cache_size: int = 0):
        self._sources = sources
        self._preload_enabled = preload
        # 转 float32 结果缓存（LRU，容量 = cache_size 帧；0 表示不缓存转换结果）
        self._cache_size = max(0, cache_size)
        self._cache = OrderedDict()  # 路径 -> float32 ndarray
        self._raw: Optional[List[Optional[np.ndarray]]] = None  # 预加载的 uint8 数组
        self._hits = 0
        self._misses = 0
        if preload:
            # 用显式 list[np.ndarray | None] 构造，避免 Pyright 推断为 list[None]
            raw_list: List[Optional[np.ndarray]] = [None] * len(sources)
            for i, src in enumerate(sources):
                raw_list[i] = _load_frame_raw(src) if isinstance(src, str) else src
            self._raw = raw_list
    def __len__(self): return len(self._sources)
    def __iter__(self):
        for idx in range(len(self._sources)):
            yield self.__getitem__(idx)
    def __getitem__(self, idx):
        n = len(self._sources)
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(n))]
        if idx < 0: idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Frame index {idx} out of range [0, {n-1}]")
        src = self._sources[idx]
        if not isinstance(src, str):
            return src
        # 转换结果缓存（LRU）
        if self._cache_size > 0:
            cached = self._cache.get(src)
            if cached is not None:
                self._hits += 1
                self._cache.move_to_end(src)
                return cached
            self._misses += 1
            img = self._convert(src, idx)
            self._cache[src] = img
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return img
        return self._convert(src, idx)
    def _convert(self, src: str, idx: int) -> np.ndarray:
        """从预加载 raw（或读盘）取帧并转 float32。"""
        raw_frame = self._raw[idx] if self._raw is not None else None
        if raw_frame is not None:
            return raw_frame.astype(np.float32) / 255.0
        return _load_frame_from_path(src)
    def preload(self) -> None:
        """手动触发预加载（幂等）。"""
        if self._raw is None:
            raw_list: List[Optional[np.ndarray]] = [None] * len(self._sources)
            self._raw = raw_list
        for i, src in enumerate(self._sources):
            if isinstance(src, str) and self._raw[i] is None:
                self._raw[i] = _load_frame_raw(src)
            elif not isinstance(src, str):
                self._raw[i] = src
    def clear_cache(self) -> None:
        self._cache.clear()
    def cache_stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}
    @property
    def preloaded(self) -> bool:
        return self._raw is not None


# ---------- Quaternion utilities ----------
def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1, p=2)
    w, x, y, z = q.unbind(dim=-1)
    R = torch.zeros((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    R[..., 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return R


def build_covariance(log_scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    s = torch.exp(log_scales)
    R = quat_to_rot(rotations)
    M = R * s.unsqueeze(-1)
    cov = M @ M.transpose(1, 2)
    return cov


# ---------- Spherical Harmonics evaluation (up to degree 3) ----------
def eval_sh(deg: int, sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics for given directions.
    sh_coeffs: [N, (deg+1)^2, 3]  (or [N, C, 3] with C >= (deg+1)^2)
    dirs: [N, 3]  (unit direction vectors)
    Returns: [N, 3] color
    """
    N = sh_coeffs.shape[0]
    device = sh_coeffs.device
    dtype = sh_coeffs.dtype

    # Normalize directions
    dirs = F.normalize(dirs, dim=-1)
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    # Precompute SH basis functions for each degree
    # Degree 0
    sh0 = torch.ones(N, 1, device=device, dtype=dtype) * 0.28209479177387814
    # Degree 1
    sh1 = torch.stack([
        0.4886025119029199 * y,               # y
        0.4886025119029199 * z,               # z
        0.4886025119029199 * x                # x
    ], dim=1)  # [N, 3]
    # Degree 2
    sh2 = torch.stack([
        1.0925484305920792 * x * y,           # xy
        1.0925484305920792 * y * z,           # yz
        0.9461746957575601 * z * z - 0.31539156525252005,  # 2z^2 - x^2 - y^2
        1.0925484305920792 * x * z,           # xz
        0.5462742152960396 * (x * x - y * y)  # x^2 - y^2
    ], dim=1)  # [N, 5]
    # Degree 3
    sh3 = torch.stack([
        0.5900435899266435 * y * (3 * x * x - y * y),          # y(3x^2-y^2)
        2.890611442640554 * x * y * z,                         # xyz
        0.4570457994644658 * y * (5 * z * z - 1),              # y(5z^2-1)
        0.3731763325901154 * z * (5 * z * z - 3),              # z(5z^2-3)
        0.4570457994644658 * x * (5 * z * z - 1),              # x(5z^2-1)
        1.445305721320277 * z * (x * x - y * y),               # z(x^2-y^2)
        0.5900435899266435 * x * (x * x - 3 * y * y)           # x(x^2-3y^2)
    ], dim=1)  # [N, 7]

    # Concatenate all basis functions up to degree
    basis_list = [sh0, sh1, sh2, sh3]
    selected = basis_list[:deg+1]
    basis = torch.cat(selected, dim=1)  # [N, (deg+1)^2]

    # Apply coefficients: color = sum_{l,m} coeffs[l,m] * basis[l,m]
    # coeffs shape: [N, C, 3], basis shape: [N, C]
    color = torch.einsum('nc, ncd -> nd', basis, sh_coeffs[:, :basis.shape[1], :])
    # 官方 SH 约定（2026-08）：DC 系数已存 (RGB-0.5)/C0，求值后补回 +0.5 → deg0 时 color == RGB。
    # clamp_min(0) 与官方光栅化器一致，避免负颜色进入 alpha 合成造成病态梯度。
    return torch.clamp(color + 0.5, min=0.0)


# ---------- Gaussian3D ----------
@dataclass
class Gaussian3D:
    positions: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3))
    log_scales: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3))
    opacities_raw: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    rotations: torch.Tensor = field(default_factory=lambda: torch.empty(0, 4))
    sh_coeffs: torch.Tensor = field(default_factory=lambda: torch.empty(0, 16, 3))

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities_raw)

    @property
    def cov3d(self) -> torch.Tensor:
        return build_covariance(self.log_scales, self.rotations)

    def parameters(self) -> List[torch.Tensor]:
        return [self.positions, self.log_scales, self.opacities_raw,
                self.rotations, self.sh_coeffs]

    def export_ply_dict(self) -> Dict[str, np.ndarray]:
        """导出官方 3DGS PLY 所需的原始参数（不做激活转换）。

        官方约定：scale 存 log σ、opacity 存 logit、sh_coeffs 通道 0 已是 (RGB-0.5)/C0、
        rot 存 (w,x,y,z)。exporter 直接消费本方法的返回值。
        """
        return {
            "positions": self.positions.detach().cpu().numpy(),
            "scales": self.log_scales.detach().cpu().numpy(),        # 原始 log σ
            "opacities": self.opacities_raw.detach().cpu().numpy(),  # 原始 logit
            "rotations": self.rotations.detach().cpu().numpy(),      # (w,x,y,z)
            "sh_coeffs": self.sh_coeffs.detach().cpu().numpy(),      # [N,16,3]
        }

    def initialize_from_dict(self, data: Dict[str, np.ndarray], device: str = "cpu") -> "Gaussian3D":
        device = torch.device(device)
        self.positions = torch.from_numpy(data["positions"]).float().to(device).clone()
        scales_np = data["scales"]
        if scales_np.ndim == 1 or scales_np.shape[1] == 1:
            scales_3 = np.repeat(scales_np, 3, axis=1) if scales_np.ndim == 2 else np.stack([scales_np]*3, axis=1)
            scales_3 = scales_3 * (1.0 + 0.01 * np.random.randn(*scales_3.shape))
        else:
            scales_3 = scales_np
        self.log_scales = torch.from_numpy(np.log(np.maximum(scales_3, 1e-6))).float().to(device).clone()
        ops_np = np.clip(data.get("opacities", np.zeros(self.positions.shape[0])), 1e-6, 1.0 - 1e-6)
        self.opacities_raw = torch.logit(torch.from_numpy(ops_np)).float().to(device).clone()
        self.rotations = torch.from_numpy(data["rotations"]).float().to(device).clone()
        sh_in = torch.from_numpy(data["sh_coeffs"]).float().to(device)
        if sh_in.shape[1] < 16:
            pad = torch.zeros(sh_in.shape[0], 16 - sh_in.shape[1], 3, device=device)
            sh_in = torch.cat([sh_in, pad], dim=1)
        self.sh_coeffs = sh_in.clone()
        for t in [self.positions, self.log_scales, self.opacities_raw,
                  self.rotations, self.sh_coeffs]:
            if isinstance(t, torch.Tensor):
                t.requires_grad_(True)
        return self


def densify_initial_gaussians(gaussians: Gaussian3D, expansion_factor: int = 8, noise_scale: float = 0.02):
    n = gaussians.num_gaussians
    if n == 0:
        return
    device = gaussians.positions.device
    pos = gaussians.positions.detach().cpu().numpy()
    log_scales = gaussians.log_scales.detach().cpu().numpy()
    opa = gaussians.opacities_raw.detach().cpu().numpy()
    rot = gaussians.rotations.detach().cpu().numpy()
    sh = gaussians.sh_coeffs.detach().cpu().numpy()
    new_pos, new_log_scales, new_opa, new_rot, new_sh = [], [], [], [], []
    for i in range(n):
        for _ in range(expansion_factor):
            noise = np.random.normal(0, noise_scale, 3).astype(np.float32)
            new_pos.append(pos[i] + noise)
            new_log_scales.append(log_scales[i] + np.log(0.8))
            new_opa.append(opa[i] + np.random.normal(0, 0.1))
            new_rot.append(rot[i] + np.random.normal(0, 0.01, 4))
            new_sh.append(sh[i] + np.random.normal(0, 0.01, (sh.shape[1], 3)))
    gaussians.positions = torch.from_numpy(np.array(new_pos)).float().to(device)
    gaussians.log_scales = torch.from_numpy(np.array(new_log_scales)).float().to(device)
    gaussians.opacities_raw = torch.from_numpy(np.array(new_opa)).float().to(device)
    gaussians.rotations = torch.from_numpy(np.array(new_rot)).float().to(device)
    gaussians.sh_coeffs = torch.from_numpy(np.array(new_sh)).float().to(device)
    for param in [gaussians.positions, gaussians.log_scales, gaussians.opacities_raw,
                  gaussians.rotations, gaussians.sh_coeffs]:
        param.requires_grad_(True)


# ---------- Differentiable Rasterizer (Pure PyTorch, supports SH up to 3) ----------
class DifferentiableRasterizer(nn.Module):
    def __init__(self, image_width: int, image_height: int, max_radius: int = 32):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.max_radius = max_radius

    def forward(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree=3, k1=None):
        return self._render_batch(positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree, k1)

    # ---------- 向量化光栅化（方向 A：排序式逐像素 splat） ----------
    def _render_batch(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree, k1=None):
        """主实现：排序式逐像素 splat，详见 _render_batch_vectorized 的说明。"""
        return self._render_batch_vectorized(positions, cov3d, opacities, sh_coeffs,
                                             view_matrix, K, background, sh_degree, k1)

    # ---------- 向量化光栅化（方向 A：排序式逐像素 splat） ----------
    def _render_batch_vectorized(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree, k1=None):
        """
        与 legacy/_render_batch_batched 数值语义等价，但把最后那段逐高斯
        Python 内层循环（每颗高斯 ~7 个 kernel launch）替换为纯向量化算子：
        展平覆盖像素对 → stable sort → 分段透射率（log1p + cumsum + exp）→
        scatter_add 归约。

        正确性依据：
        - 高斯已按深度全局排序（near→far），像素组内 stable sort 保持深度序；
        - over-blend 等价于逐像素：T_before(p)=Π(1-α_j, j<i)，color(p)=Σα_i·T·c_i，
          alpha(p)=1-Π(1-α_i)；用 log 空间 cumsum 实现分段累计透射率；
        - 不同像素的合成互相独立，可并行 scatter_add。
        """
        N = positions.shape[0]
        H, W = self.image_height, self.image_width
        if N == 0:
            connected_zero = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
            zero = connected_zero.view(1).expand(H * W * 3).view(H, W, 3).contiguous()
            return zero, connected_zero.view(1).expand(H * W).view(H, W).contiguous()

        # Camera space transformation
        R_cam = view_matrix[:3, :3]
        t_cam = view_matrix[:3, 3]
        cam_positions = positions @ R_cam.T + t_cam
        cam_cov = R_cam @ cov3d @ R_cam.T
        # 修复(2026-08): SH 求值方向必须用世界系（官方约定：高斯中心 - 相机中心）。
        # 旧版用相机系方向使 SH 基底随相机旋转，视角相关效果不可移植。
        center_world = -R_cam.T @ t_cam

        # Projection
        fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
        z = cam_positions[:, 2].clamp(min=0.01)
        x_c = cam_positions[:, 0]; y_c = cam_positions[:, 1]
        u = fx * (x_c / z) + cx
        v = fy * (y_c / z) + cy

        # Compute 2D covariance
        B = torch.zeros(N, 2, 3, dtype=cov3d.dtype, device=cov3d.device)
        B[:, 0, 0] = fx / z; B[:, 0, 2] = -fx * x_c / (z * z)
        B[:, 1, 1] = fy / z; B[:, 1, 2] = -fy * y_c / (z * z)
        cov2d = (B @ cam_cov) @ B.transpose(1, 2)

        # ---- 一阶径向畸变（k1，2026-08 实现） ----
        # 门控必须按 `k1 is not None`（即 enable_k1），绝不能按值 k1==0 跳过：
        # 初始 k1=0 时其梯度 du·r² 非零但"按值跳过"会让 k1 永不产生梯度，特性永久失效。
        # k1=None（默认）→ 整块跳过，与历史输出逐位一致。
        if k1 is not None:
            du = u - cx
            dv = v - cy
            xn = du / fx
            yn = dv / fy
            r2 = xn * xn + yn * yn
            D = 1.0 + k1 * r2
            u = cx + du * D
            v = cy + dv * D
            # 畸变雅可比 d(u_d,v_d)/d(u_c,v_c)，用于把针孔 2D 协方差变换到畸变图像空间
            J00 = D + 2.0 * k1 * du * du / (fx * fx)
            J01 = 2.0 * k1 * du * dv / (fy * fy)
            J10 = 2.0 * k1 * du * dv / (fx * fx)
            J11 = D + 2.0 * k1 * dv * dv / (fy * fy)
            J = torch.stack([torch.stack([J00, J01], dim=-1),
                             torch.stack([J10, J11], dim=-1)], dim=-2)  # [N,2,2]
            cov2d = J @ cov2d @ J.transpose(1, 2)

        # Compute radius (3 sigma)
        a = cov2d[:, 0, 0]; c = cov2d[:, 1, 1]; b = cov2d[:, 0, 1]
        det = a * c - b * b
        trace = a + c
        disc = torch.clamp(trace ** 2 - 4 * det, min=1e-8)
        half = 0.5 * (trace + torch.sqrt(disc))
        sigma = torch.sqrt(half + 1e-6)
        radius = (sigma * 3.0).ceil().int().clamp(max=self.max_radius)

        valid = (z > 0.01) & (radius > 0) & (radius < 1000)
        if not valid.any():
            connected_zero = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
            zero = connected_zero.view(1).expand(H * W * 3).view(H, W, 3).contiguous()
            return zero, connected_zero.view(1).expand(H * W).view(H, W).contiguous()

        u_v, v_v, r_v = u[valid], v[valid], radius[valid]
        cov2d_v = cov2d[valid]
        op_v = opacities[valid]
        N_valid = int(valid.sum())

        # Depth ordering (front-to-back)
        depth_sorted = cam_positions[valid][:, 2]
        order = torch.argsort(depth_sorted)  # near to far

        u_s = u_v[order]; v_s = v_v[order]; r_s = r_v[order]
        cov2d_s = cov2d_v[order]; opa_s = op_v[order]

        # Compute colors via SH
        dirs = F.normalize(positions[valid][order] - center_world, dim=-1)
        colors = eval_sh(sh_degree, sh_coeffs[valid][order], dirs)

        # Ensure gradient connection even when all Gaussians land off-screen
        _graph_link = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
        out_color = _graph_link.view(1,1,1).expand(H, W, 3).contiguous()
        out_alpha = _graph_link.view(1,1).expand(H, W).contiguous()

        # ---- 逐像素向量化合成（分块 + 跨块 carry，替代逐高斯 Python 循环） ----
        # 显存上界：原实现整表物化 [batch_n, max_h, max_w]，max_h/max_w 取全部高斯里
        #   最大那颗的包围盒（radius 已 clamp 到 max_radius → 最坏 65×65），一颗大
        #   高斯让全体陪跑，38665 高斯时约 4-5GB。改为按深度有序高斯切块，每块只
        #   用块内 max_h/max_w（大高斯只撑大自己的块），峰值 = RASTER_CHUNK×65×65。
        # 跨块透射率：块 k 的 exclusive 前缀是"相对块首"的透射率，真正的 T_before
        #   还要乘以前序所有块对该像素的累计 log(1-α)。逐像素 carry（acc_log）在
        #   块 k 自己的 index_add_ 之前快照，与整表 cumsum 在精确算术下等价。
        mu_u = u_s; mu_v = v_s; rad = r_s
        A = cov2d_s[:, 0, 0]; B_ = cov2d_s[:, 0, 1]; C = cov2d_s[:, 1, 1]
        opa = opa_s; col = colors

        y_min = (mu_v - rad).clamp(min=0).int(); y_max = (mu_v + rad + 1).clamp(max=H).int()
        x_min = (mu_u - rad).clamp(min=0).int(); x_max = (mu_u + rad + 1).clamp(max=W).int()

        valid_b = (y_min < y_max) & (x_min < x_max)
        if not valid_b.any():
            return out_color + background.view(1,1,3) * (1.0 - out_alpha.unsqueeze(-1)), out_alpha

        y_min_b = y_min[valid_b]; y_max_b = y_max[valid_b]; x_min_b = x_min[valid_b]; x_max_b = x_max[valid_b]
        mu_u_b = mu_u[valid_b]; mu_v_b = mu_v[valid_b]
        A_b = A[valid_b]; B_b = B_[valid_b]; C_b = C[valid_b]
        opa_b = opa[valid_b]; col_b = col[valid_b]
        batch_n = int(valid_b.sum())

        det_inv = 1.0 / (A_b * C_b - B_b * B_b + 1e-6)
        inv_A = det_inv * C_b; inv_B = -det_inv * B_b; inv_C = det_inv * A_b

        HpW = H * W
        device = colors.device
        acc_color = torch.zeros(HpW, 3, dtype=torch.float32, device=device)
        acc_log = torch.zeros(HpW, dtype=torch.float32, device=device)

        for start in range(0, batch_n, RASTER_CHUNK):
            end = min(start + RASTER_CHUNK, batch_n)
            n_chunk = end - start

            y_lo = y_min_b[start:end]; y_hi = y_max_b[start:end]
            x_lo = x_min_b[start:end]; x_hi = x_max_b[start:end]
            mu_u_c = mu_u_b[start:end]; mu_v_c = mu_v_b[start:end]
            iA = inv_A[start:end]; iB = inv_B[start:end]; iC = inv_C[start:end]
            opa_c = opa_b[start:end]; col_c = col_b[start:end]

            h_sizes = (y_hi - y_lo).int(); w_sizes = (x_hi - x_lo).int()
            max_h = int(h_sizes.max()); max_w = int(w_sizes.max())
            if max_h == 0 or max_w == 0:
                continue

            gy = torch.arange(max_h, device=device, dtype=torch.float32).view(1, -1, 1)
            gx = torch.arange(max_w, device=device, dtype=torch.float32).view(1, 1, -1)
            gy_g = gy + y_lo.view(-1, 1, 1); gx_g = gx + x_lo.view(-1, 1, 1)
            dy = gy_g - mu_v_c.view(-1, 1, 1); dx = gx_g - mu_u_c.view(-1, 1, 1)

            y_valid = (gy_g >= y_lo.view(-1, 1, 1)) & (gy_g < y_hi.view(-1, 1, 1))
            x_valid = (gx_g >= x_lo.view(-1, 1, 1)) & (gx_g < x_hi.view(-1, 1, 1))
            valid_mask = y_valid & x_valid

            exponent = -(iA.view(-1,1,1) * dx**2 + 2*iB.view(-1,1,1)*dx*dy + iC.view(-1,1,1)*dy**2)*0.5
            exponent = exponent.clamp(max=0)
            alpha = exponent.exp() * opa_c.view(-1,1,1)
            alpha = alpha.masked_fill(~valid_mask, 0.0)

            # 展平覆盖像素对：只保留 valid_mask 为真的格点
            mask = valid_mask  # [n_chunk, max_h, max_w]
            flat_alpha = alpha[mask]
            if flat_alpha.shape[0] == 0:
                continue

            # 重建每个覆盖格的全局像素坐标（y, x）
            # 用 torch.arange(n_chunk) 常数（块内全是 valid_b 幸存者），避免 int() 同步
            gauss_idx_3d = torch.arange(n_chunk, device=device).view(-1, 1, 1).expand_as(mask)
            y_3d = gy_g.expand_as(mask)
            x_3d = gx_g.expand_as(mask)
            gauss_ids = gauss_idx_3d[mask].long()
            y_coord = y_3d[mask].long()
            x_coord = x_3d[mask].long()
            pix = y_coord * W + x_coord  # 展平像素索引 [n_pairs]

            flat_color = col_c[gauss_ids]  # [n_pairs, 3]

            # 按像素 stable sort（保持深度序：gauss_ids 已按深度排序）
            pix_sorted, sort_idx = torch.sort(pix, stable=True)
            a_sorted = flat_alpha[sort_idx]
            c_sorted = flat_color[sort_idx]

            # 段内透射率（相对块首）：T_before(p) = exp(Σ_{j<i} log(1-α_j))，按像素分组
            log_ta = torch.log1p(-a_sorted)
            log_cum = torch.cumsum(log_ta, dim=0)
            # 段内 exclusive 前缀：log_T_before[i] = log_cum[i-1]，段首为 0
            log_cum_shift = torch.cat([torch.zeros(1, dtype=log_cum.dtype, device=log_cum.device), log_cum[:-1]])
            new_group = pix_sorted[1:] != pix_sorted[:-1]
            group_starts = torch.cat([torch.tensor([True], device=pix_sorted.device), new_group])
            # 组首位置用 cummax 前向传播（不能用 cumsum-1，那给出的是组 id 而非组首索引）
            arange = torch.arange(group_starts.shape[0], device=pix_sorted.device)
            group_start_pos = torch.where(group_starts, arange, torch.zeros_like(arange))
            group_start_pos = torch.cummax(group_start_pos, dim=0).values
            seg_offset = log_cum_shift[group_start_pos]
            log_T_before_chunk = log_cum_shift - seg_offset

            # 跨块 carry：块内 exclusive 前缀 + 前序所有块对该像素的累计 log(1-α)
            # 高级索引产生副本，acc_log 快照独立于后续 in-place index_add_，不会重复计入
            carry = acc_log[pix_sorted]
            log_T_before = carry + log_T_before_chunk

            T_before = torch.exp(log_T_before.clamp(min=-50.0))
            weight = a_sorted * T_before  # α_i * T_before(p)

            # 归约：跨块累加进同一个 [H*W] 缓冲
            acc_color = acc_color.index_add_(0, pix_sorted, weight.unsqueeze(-1) * c_sorted)
            acc_log = acc_log.index_add_(0, pix_sorted, log_ta)

        out_color = out_color.view(HpW, 3) + acc_color
        out_alpha = out_alpha.view(HpW) + (1.0 - torch.exp(acc_log.clamp(min=-50.0)))
        out_color = out_color.view(H, W, 3)
        out_alpha = out_alpha.view(H, W)

        out_color = out_color + background.view(1,1,3) * (1.0 - out_alpha.unsqueeze(-1))
        return out_color, out_alpha

    # ---------- 方向 B 实现（保留：批量同步 + 逐高斯串行合成） ----------
    def _render_batch_batched(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree, k1=None):
        """
        与 legacy 数值语义一致；把内层循环每颗高斯的 4 次 GPU→CPU 同步
        改为每 batch 一次 .tolist() 批量拉取。像素级 over-blend 仍逐高斯串行。
        """
        N = positions.shape[0]
        H, W = self.image_height, self.image_width
        if N == 0:
            # Must maintain gradient graph connection for loss.backward()
            connected_zero = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
            zero = connected_zero.view(1).expand(H * W * 3).view(H, W, 3).contiguous()
            return zero, connected_zero.view(1).expand(H * W).view(H, W).contiguous()

        # Camera space transformation
        R_cam = view_matrix[:3, :3]
        t_cam = view_matrix[:3, 3]
        cam_positions = positions @ R_cam.T + t_cam
        cam_cov = R_cam @ cov3d @ R_cam.T
        # 修复(2026-08): SH 求值方向必须用世界系（官方约定：高斯中心 - 相机中心）。
        # 旧版用相机系方向使 SH 基底随相机旋转，视角相关效果不可移植。
        center_world = -R_cam.T @ t_cam

        # Projection
        fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
        z = cam_positions[:, 2].clamp(min=0.01)
        x_c = cam_positions[:, 0]; y_c = cam_positions[:, 1]
        u = fx * (x_c / z) + cx
        v = fy * (y_c / z) + cy

        # Compute 2D covariance
        B = torch.zeros(N, 2, 3, dtype=cov3d.dtype, device=cov3d.device)
        B[:, 0, 0] = fx / z; B[:, 0, 2] = -fx * x_c / (z * z)
        B[:, 1, 1] = fy / z; B[:, 1, 2] = -fy * y_c / (z * z)
        cov2d = (B @ cam_cov) @ B.transpose(1, 2)

        # ---- 一阶径向畸变（k1，2026-08 实现） ----
        # 门控必须按 `k1 is not None`（即 enable_k1），绝不能按值 k1==0 跳过：
        # 初始 k1=0 时其梯度 du·r² 非零但"按值跳过"会让 k1 永不产生梯度，特性永久失效。
        # k1=None（默认）→ 整块跳过，与历史输出逐位一致。
        if k1 is not None:
            du = u - cx
            dv = v - cy
            xn = du / fx
            yn = dv / fy
            r2 = xn * xn + yn * yn
            D = 1.0 + k1 * r2
            u = cx + du * D
            v = cy + dv * D
            # 畸变雅可比 d(u_d,v_d)/d(u_c,v_c)，用于把针孔 2D 协方差变换到畸变图像空间
            J00 = D + 2.0 * k1 * du * du / (fx * fx)
            J01 = 2.0 * k1 * du * dv / (fy * fy)
            J10 = 2.0 * k1 * du * dv / (fx * fx)
            J11 = D + 2.0 * k1 * dv * dv / (fy * fy)
            J = torch.stack([torch.stack([J00, J01], dim=-1),
                             torch.stack([J10, J11], dim=-1)], dim=-2)  # [N,2,2]
            cov2d = J @ cov2d @ J.transpose(1, 2)

        # Compute radius (3 sigma)
        a = cov2d[:, 0, 0]; c = cov2d[:, 1, 1]; b = cov2d[:, 0, 1]
        det = a * c - b * b
        trace = a + c
        disc = torch.clamp(trace ** 2 - 4 * det, min=1e-8)
        half = 0.5 * (trace + torch.sqrt(disc))
        sigma = torch.sqrt(half + 1e-6)
        radius = (sigma * 3.0).ceil().int().clamp(max=self.max_radius)

        valid = (z > 0.01) & (radius > 0) & (radius < 1000)
        if not valid.any():
            # Maintain gradient graph connection
            connected_zero = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
            zero = connected_zero.view(1).expand(H * W * 3).view(H, W, 3).contiguous()
            return zero, connected_zero.view(1).expand(H * W).view(H, W).contiguous()

        u_v, v_v, r_v = u[valid], v[valid], radius[valid]
        cov2d_v = cov2d[valid]
        op_v = opacities[valid]
        N_valid = int(valid.sum())

        # Depth ordering (back-to-front for blending)
        depth_sorted = cam_positions[valid][:, 2]
        order = torch.argsort(depth_sorted, descending=True)  # far to near? Typically splatting uses far-to-near or near-to-far? We'll use near-to-far and blend with over operator.
        # For correct alpha blending, we should render back-to-front (far to near) with over operator.
        # But our loop currently renders front-to-back using (1-alpha) multiplication. Let's keep as is but ensure order is front-to-back.
        # Actually we want front-to-back because we use out_alpha to compute contribution.
        order = torch.argsort(depth_sorted)  # near to far

        u_s = u_v[order]; v_s = v_v[order]; r_s = r_v[order]
        cov2d_s = cov2d_v[order]; opa_s = op_v[order]

        # Compute colors via SH
        dirs = F.normalize(positions[valid][order] - center_world, dim=-1)
        colors = eval_sh(sh_degree, sh_coeffs[valid][order], dirs)

        # Ensure gradient connection even when all Gaussians land off-screen
        _graph_link = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
        out_color = _graph_link.view(1,1,1).expand(H, W, 3).contiguous()
        out_alpha = _graph_link.view(1,1).expand(H, W).contiguous()

        MAX_BATCH = 64
        for start in range(0, N_valid, MAX_BATCH):
            end = min(start + MAX_BATCH, N_valid)
            mu_u = u_s[start:end]; mu_v = v_s[start:end]; rad = r_s[start:end]
            A = cov2d_s[start:end, 0, 0]; B_ = cov2d_s[start:end, 0, 1]; C = cov2d_s[start:end, 1, 1]
            opa = opa_s[start:end]; col = colors[start:end]

            y_min = (mu_v - rad).clamp(min=0).int(); y_max = (mu_v + rad + 1).clamp(max=H).int()
            x_min = (mu_u - rad).clamp(min=0).int(); x_max = (mu_u + rad + 1).clamp(max=W).int()

            valid_b = (y_min < y_max) & (x_min < x_max)
            if not valid_b.any(): continue

            y_min_b = y_min[valid_b]; y_max_b = y_max[valid_b]; x_min_b = x_min[valid_b]; x_max_b = x_max[valid_b]
            mu_u_b = mu_u[valid_b]; mu_v_b = mu_v[valid_b]
            A_b = A[valid_b]; B_b = B_[valid_b]; C_b = C[valid_b]
            opa_b = opa[valid_b]; col_b = col[valid_b]
            batch_n = int(valid_b.sum())

            det_inv = 1.0 / (A_b * C_b - B_b * B_b + 1e-6)
            inv_A = det_inv * C_b; inv_B = -det_inv * B_b; inv_C = det_inv * A_b

            h_sizes = (y_max_b - y_min_b).int(); w_sizes = (x_max_b - x_min_b).int()
            max_h = int(h_sizes.max()); max_w = int(w_sizes.max())
            if max_h == 0 or max_w == 0: continue

            gy = torch.arange(max_h, device=colors.device, dtype=torch.float32).view(1, -1, 1)
            gx = torch.arange(max_w, device=colors.device, dtype=torch.float32).view(1, 1, -1)
            gy_g = gy + y_min_b.view(-1, 1, 1); gx_g = gx + x_min_b.view(-1, 1, 1)
            dy = gy_g - mu_v_b.view(-1, 1, 1); dx = gx_g - mu_u_b.view(-1, 1, 1)

            y_valid = (gy_g >= y_min_b.view(-1, 1, 1)) & (gy_g < y_max_b.view(-1, 1, 1))
            x_valid = (gx_g >= x_min_b.view(-1, 1, 1)) & (gx_g < x_max_b.view(-1, 1, 1))
            valid_mask = y_valid & x_valid

            exponent = -(inv_A.view(-1,1,1) * dx**2 + 2*inv_B.view(-1,1,1)*dx*dy + inv_C.view(-1,1,1)*dy**2)*0.5
            exponent = exponent.clamp(max=0)
            alpha = exponent.exp() * opa_b.view(-1,1,1)
            alpha = alpha.masked_fill(~valid_mask, 0.0)

            for i in range(batch_n):
                y_lo = int(y_min_b[i]); y_hi = int(y_max_b[i])
                x_lo = int(x_min_b[i]); x_hi = int(x_max_b[i])
                h_i = min(y_hi - y_lo, alpha.shape[1]); w_i = min(x_hi - x_lo, alpha.shape[2])
                if h_i <= 0 or w_i <= 0: continue
                a_i = alpha[i, :h_i, :w_i]; c_i = col_b[i]
                old_alpha = out_alpha[y_lo:y_hi, x_lo:x_hi]
                out_color[y_lo:y_hi, x_lo:x_hi] += (a_i.unsqueeze(-1) * (1.0 - old_alpha.unsqueeze(-1)) * c_i)
                out_alpha[y_lo:y_hi, x_lo:x_hi] += a_i * (1.0 - old_alpha)

        out_color = out_color + background.view(1,1,3) * (1.0 - out_alpha.unsqueeze(-1))
        return out_color, out_alpha


# ---------- Loss functions ----------
def compute_ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
                      kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = img1.shape[-1]
    if kernel is None:
        kernel = torch.ones((channels, 1, window_size, window_size), dtype=img1.dtype, device=img1.device) / (window_size ** 2)
    x = img1.permute(2,0,1).unsqueeze(0)
    y = img2.permute(2,0,1).unsqueeze(0)
    mu_x = F.conv2d(x, kernel, padding=window_size//2, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=window_size//2, groups=channels)
    mu_xx, mu_yy = mu_x**2, mu_y**2
    mu_xy = mu_x * mu_y
    sigma_xx = F.conv2d(x**2, kernel, padding=window_size//2, groups=channels) - mu_xx
    sigma_yy = F.conv2d(y**2, kernel, padding=window_size//2, groups=channels) - mu_yy
    sigma_xy = F.conv2d(x*y, kernel, padding=window_size//2, groups=channels) - mu_xy
    ssim = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_xx + mu_yy + C1)*(sigma_xx + sigma_yy + C2))
    return 1.0 - ssim.mean()


class LossDivergenceError(Exception):
    pass


# ---------- Trainer ----------
class Trainer:
    def __init__(self, gaussians: Gaussian3D, rasterizer: Optional[DifferentiableRasterizer],
                 K: np.ndarray, image_width: int, image_height: int, device: str = "cpu",
                 sh_degree: int = 3,
                 random_background: bool = True, train_focal: bool = True,
                 max_gaussians: int = MAX_GAUSSIANS, sh_warmup_steps: int = SH_WARMUP_STEPS,
                 ssim_warmup_steps: int = SSIM_WARMUP_STEPS, ssim_weight_max: float = SSIM_WEIGHT_MAX,
                 enable_k1: bool = False, use_amp: bool = False,
                 use_lr_schedule: bool = USE_LR_SCHEDULE,
                 lr_decay_steps: int = LR_DECAY_STEPS, lr_decay_gamma: float = LR_DECAY_GAMMA,
                 grad_thresh_base: float = GRAD_THRESH_BASE, scale_thresh: float = SCALE_THRESH,
                 min_opacity: float = MIN_OPACITY, densify_every: int = DENSIFY_EVERY,
                 prune_every: int = PRUNE_EVERY):
        self.device = device
        self.image_height = image_height
        self.image_width = image_width

        # Force use of PyTorch rasterizer (no CUDA dependency)
        self.use_cuda_rasterizer = False
        print("  [INFO] Using PyTorch rasterizer (supports SH up to 3).")

        self.gaussians = gaussians
        self.K = torch.from_numpy(K.astype(np.float32)).to(device)
        self.view_matrix = torch.eye(4, dtype=torch.float32, device=device)
        self.random_background = random_background
        self.train_focal = train_focal
        self.enable_k1 = enable_k1
        # 混合精度开关：仅 CUDA 生效（AMP 在 CPU 上无意义），需 fp16 硬件
        self.use_amp = use_amp and torch.cuda.is_available() and str(device).startswith("cuda")
        if self.use_amp:
            try:
                _scaler_cls = getattr(torch.amp, "GradScaler")  # torch>=2.3
                self._scaler = _scaler_cls("cuda", enabled=True)
            except (AttributeError, TypeError):  # torch 2.0-2.2 回退到旧 API
                self._scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self._scaler = None
        self.sh_degree = min(sh_degree, 3)   # max 3
        self.sh_warmup_steps = sh_warmup_steps
        self.ssim_warmup_steps = ssim_warmup_steps
        self.ssim_weight_max = ssim_weight_max

        if self.train_focal:
            self.fx = nn.Parameter(torch.tensor(K[0,0], dtype=torch.float32, device=device))
            self.fy = nn.Parameter(torch.tensor(K[1,1], dtype=torch.float32, device=device))
        else:
            self.fx = float(K[0,0])
            self.fy = float(K[1,1])

        self.cx = float(K[0,2])
        self.cy = float(K[1,2])

        self.k1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device)) if enable_k1 else None

        self.lr_positions = LR_POSITIONS; self.lr_log_scales = LR_LOG_SCALES
        self.lr_opacities = LR_OPACITIES; self.lr_rotations = LR_ROTATIONS
        self.lr_sh = LR_SH; self.lr_focal = LR_FOCAL; self.lr_k1 = LR_K1
        self.use_lr_schedule = use_lr_schedule
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_gamma = lr_decay_gamma
        self.current_step = 0
        # 修复(2026-08): 最近处理过的帧下标（帧级断点续训用）。train_epoch 每处理一帧更新。
        self.last_frame_index = -1
        # ===== 修复: 添加 best_loss 成员变量 =====
        self.best_loss = float("inf")
        self.background = torch.rand(3, dtype=torch.float32, device=device)
        channels = 3
        self._ssim_kernel = torch.ones((channels, 1, 11, 11), dtype=torch.float32, device=device) / 121.0
        self.adaptive_density = AdaptiveDensityController(self, densify_every, prune_every, max_gaussians,
                                                          grad_thresh_base, scale_thresh, min_opacity)
        # Use provided rasterizer or create default
        if rasterizer is None:
            self.rasterizer = DifferentiableRasterizer(image_width, image_height)
        else:
            self.rasterizer = rasterizer
        self._update_tanfov()
        # ===== 修复(2026-08-14): 初始 8× 稠密化必须先于 _setup_optimizers =====
        # 原实现优化器先包裹旧张量、再 densify_initial_gaussians 替换高斯张量 → 优化器
        # 持有陈旧引用，前 densify_every 步位置参数不被更新（动量积累在死张量上）。
        if self.gaussians.num_gaussians < 2000:
            densify_initial_gaussians(self.gaussians, expansion_factor=8, noise_scale=0.02)
            print(f"  [INIT] Densified to {self.gaussians.num_gaussians} Gaussians")
        self._setup_optimizers()

    def _setup_optimizers(self):
        self.optimizers = {}
        self.optimizers["positions"] = torch.optim.Adam([self.gaussians.positions], lr=self.lr_positions)
        self.optimizers["log_scales"] = torch.optim.Adam([self.gaussians.log_scales], lr=self.lr_log_scales)
        self.optimizers["opacities"] = torch.optim.Adam([self.gaussians.opacities_raw], lr=self.lr_opacities)
        self.optimizers["rotations"] = torch.optim.Adam([self.gaussians.rotations], lr=self.lr_rotations)
        self.optimizers["sh"] = torch.optim.Adam([self.gaussians.sh_coeffs], lr=self.lr_sh)
        focal_params = []
        if self.train_focal and isinstance(self.fx, nn.Parameter):
            focal_params = [self.fx, self.fy]
        if focal_params:
            self.optimizers["focal"] = torch.optim.Adam(focal_params, lr=self.lr_focal)
        else:
            self.optimizers["focal"] = None
        if self.enable_k1 and self.k1 is not None and isinstance(self.k1, nn.Parameter):
            self.optimizers["k1"] = torch.optim.Adam([self.k1], lr=self.lr_k1)
        else:
            self.optimizers["k1"] = None
        self.optimizers = {k: v for k, v in self.optimizers.items() if v is not None}

    # ---------- 密度控制优化器状态原地保留（官方 3DGS 语义，2026-08-14） ----------
    # 原实现：densify/prune 后 _setup_optimizers() 重建全新 Adam → 全部高斯动量清零，
    #   训练震荡、收敛变慢。官方只在"新增高斯"处补零动量、存量高斯动量按行保留。
    # 关键坑（HANDOFF §五）：torch.cat([活参, 新行]) 产生非叶子张量 → .grad 为 None
    #   → Adam 静默跳过 → 训练悄悄冻结。必须从 .detach() 的片段构造真叶子张量。
    _GAUSS_ATTR = {
        "positions": "positions", "log_scales": "log_scales",
        "opacities": "opacities_raw", "rotations": "rotations", "sh": "sh_coeffs",
    }

    def _cat_tensors_to_optimizer(self, new_tensors: Dict[str, torch.Tensor]) -> None:
        """把新高斯追加到各参数尾部：存量动量按行保留，新增行动量补零。

        new_tensors: {组名: 新行张量}，只接受 positions/log_scales/opacities/rotations/sh。
        """
        g = self.gaussians
        for name, opt in self.optimizers.items():
            if name not in new_tensors or name not in self._GAUSS_ATTR:
                continue
            t = new_tensors[name].detach()
            p = opt.param_groups[0]["params"][0]
            # 必须从 .detach() 片段构造叶子：cat 活参会产生非叶子，autograd 不再填 .grad
            cat_p = torch.cat([p.detach(), t], dim=0).requires_grad_(True)
            setattr(g, self._GAUSS_ATTR[name], cat_p)
            opt.param_groups[0]["params"][0] = cat_p
            stored = opt.state.get(p)
            if stored is not None and "exp_avg" in stored:
                stored["exp_avg"] = torch.cat([stored["exp_avg"], torch.zeros_like(t)], dim=0)
                stored["exp_avg_sq"] = torch.cat([stored["exp_avg_sq"], torch.zeros_like(t)], dim=0)
                del opt.state[p]
                opt.state[cat_p] = stored

    def _prune_optimizer(self, mask: torch.Tensor) -> None:
        """按 bool mask 裁剪高斯参数：幸存者 Adam 动量按 mask 保留。"""
        g = self.gaussians
        for name, opt in self.optimizers.items():
            if name not in self._GAUSS_ATTR:
                continue
            p = opt.param_groups[0]["params"][0]
            new_p = p[mask].detach().clone().requires_grad_(True)
            setattr(g, self._GAUSS_ATTR[name], new_p)
            opt.param_groups[0]["params"][0] = new_p
            stored = opt.state.get(p)
            if stored is not None and "exp_avg" in stored:
                stored["exp_avg"] = stored["exp_avg"][mask]
                stored["exp_avg_sq"] = stored["exp_avg_sq"][mask]
                del opt.state[p]
                opt.state[new_p] = stored

    def _update_lr(self):
        if not self.use_lr_schedule:
            return
        decay = self.lr_decay_gamma ** (self.current_step // self.lr_decay_steps)
        for name, opt in self.optimizers.items():
            base = getattr(self, f"lr_{name}", 1e-4)
            new_lr = base * decay
            for pg in opt.param_groups:
                pg["lr"] = max(new_lr, 1e-7)

    def _update_tanfov(self):
        if self.train_focal:
            fx = self.fx.item() if isinstance(self.fx, torch.Tensor) else self.fx
            fy = self.fy.item() if isinstance(self.fy, torch.Tensor) else self.fy
        else:
            fx = self.fx if isinstance(self.fx, float) else float(self.fx)
            fy = self.fy if isinstance(self.fy, float) else float(self.fy)
        self.tanfovx = self.image_width / (2.0 * fx)
        self.tanfovy = self.image_height / (2.0 * fy)

    def _build_projection_matrix(self, znear=0.01, zfar=100.0):
        if self.train_focal:
            fx_val = self.fx.item() if isinstance(self.fx, torch.Tensor) else self.fx
            fy_val = self.fy.item() if isinstance(self.fy, torch.Tensor) else self.fy
        else:
            fx_val = self.fx if isinstance(self.fx, float) else float(self.fx)
            fy_val = self.fy if isinstance(self.fy, float) else float(self.fy)
        cx, cy = self.cx, self.cy
        h, w = self.image_height, self.image_width
        P = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        P[0,0] = 2*fx_val/w; P[1,1] = 2*fy_val/h
        P[0,2] = 2*(cx/w)-1; P[1,2] = 2*(cy/h)-1
        P[2,2] = (zfar+znear)/(znear-zfar)
        P[2,3] = (2*zfar*znear)/(znear-zfar)
        P[3,2] = -1
        return P.T

    def _get_camera_center(self, view_matrix):
        R = view_matrix[:3,:3]; t = view_matrix[:3,3]
        return -R.T @ t

    def effective_sh_degree(self) -> int:
        if self.sh_warmup_steps <= 0:
            return self.sh_degree
        phase = self.current_step // max(1, self.sh_warmup_steps)
        return min(phase, self.sh_degree)

    def current_ssim_weight(self) -> float:
        if self.ssim_warmup_steps <= 0:
            return self.ssim_weight_max
        progress = min(1.0, self.current_step / self.ssim_warmup_steps)
        return self.ssim_weight_max * progress

    def step(self, target_image: Union[np.ndarray, torch.Tensor], camera_pose: Optional[np.ndarray] = None) -> float:
        if camera_pose is not None:
            self.view_matrix = torch.from_numpy(camera_pose.astype(np.float32)).to(self.device)

        if self.random_background:
            # 修复(2026-08): 真黑白背景（README 宣称"随机黑白背景"）。
            # 旧版每通道独立 0/1 → 8 种颜色而非黑白。
            self.background = torch.randint(0, 2, (1,), device=self.device, dtype=torch.float32).expand(3)
        else:
            if not isinstance(self.background, torch.Tensor):
                self.background = torch.tensor(self.background, dtype=torch.float32, device=self.device)
            else:
                self.background = self.background.to(device=self.device, dtype=torch.float32)

        if isinstance(target_image, torch.Tensor):
            target = target_image.float().to(self.device)
        else:
            target = torch.from_numpy(target_image).float().to(self.device)
        if target.dim() == 3 and target.shape[0] == 3:
            target = target.permute(1, 2, 0)

        means3D = self.gaussians.positions
        rotations = self.gaussians.rotations
        opacities = self.gaussians.opacities
        sh_coeffs = self.gaussians.sh_coeffs

        eff_deg = self.effective_sh_degree()

        # Use PyTorch rasterizer
        viewmat = self.view_matrix
        K = torch.zeros(3, 3, dtype=torch.float32, device=self.device)
        # 修复(2026-08): 旧版用 .item() 取 float 会切断计算图 → fx.grad 恒 None → 焦距从不更新。
        # 直接赋 nn.Parameter 进 K 产生 CopySlices 图边，梯度沿投影回传（train_focal 生效）。
        K[0,0] = self.fx if isinstance(self.fx, nn.Parameter) else float(self.fx)
        K[1,1] = self.fy if isinstance(self.fy, nn.Parameter) else float(self.fy)
        K[0,2] = self.cx
        K[1,2] = self.cy
        K[2,2] = 1.0

        params_to_clip = [
            self.gaussians.positions, self.gaussians.log_scales,
            self.gaussians.opacities_raw, self.gaussians.rotations,
            self.gaussians.sh_coeffs
        ]
        if self.train_focal and isinstance(self.fx, nn.Parameter):
            params_to_clip.append(self.fx); params_to_clip.append(self.fy)
        if self.enable_k1 and self.k1 is not None and isinstance(self.k1, nn.Parameter):
            params_to_clip.append(self.k1)

        amp_on = self.use_amp and self._scaler is not None
        if amp_on:
            assert self._scaler is not None  # amp_on 已保证
            # AMP：cov3d 组合 matmul 走 fp16（Tensor Core 可命中）；光栅化器内部
            #   保持 fp32（其 cumsum/scatter 不吃 Tensor Core，硬上 fp16 伤数值），
            #   cov3d.float() 在边界回铸（autocast(enabled=False) 不会回铸输入）；
            #   SSIM 卷积 fp16 计算、输出仍 fp32。
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                cov3d = self.gaussians.cov3d
                with torch.autocast(device_type="cuda", enabled=False):
                    rendered, _ = self.rasterizer(
                        means3D, cov3d.float(), opacities, sh_coeffs, viewmat, K,
                        self.background, sh_degree=eff_deg, k1=(self.k1 if self.enable_k1 else None)
                    )
                if rendered.dim() == 3 and rendered.shape[0] == 3:
                    rendered = rendered.permute(1, 2, 0)
                # ---------- 损失 ----------
                l1_loss = F.l1_loss(rendered, target)
                ssim_loss = compute_ssim_loss(rendered, target, kernel=self._ssim_kernel)
                w_ssim = self.current_ssim_weight()
                loss = (1.0 - w_ssim) * l1_loss + w_ssim * ssim_loss

            for opt in self.optimizers.values():
                opt.zero_grad()
            self._scaler.scale(loss).backward()
            # 逐优化器守卫：无梯度优化器（focal/k1 未启用、光栅化器 early-return 帧下
            #   log_scales/rotations/sh）跳过 scaler，否则 step 抛
            #   "No inf checks were recorded for this optimizer"
            has_grad = [
                any(p.grad is not None for grp in opt.param_groups for p in grp["params"])
                for opt in self.optimizers.values()
            ]
            for (opt, hg) in zip(self.optimizers.values(), has_grad):
                if hg:
                    self._scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)
            for (opt, hg) in zip(self.optimizers.values(), has_grad):
                if hg:
                    self._scaler.step(opt)
            self._scaler.update()
        else:
            cov3d = self.gaussians.cov3d
            rendered, _ = self.rasterizer(
                means3D, cov3d, opacities, sh_coeffs, viewmat, K, self.background, sh_degree=eff_deg,
                k1=(self.k1 if self.enable_k1 else None)
            )
            if rendered.dim() == 3 and rendered.shape[0] == 3:
                rendered = rendered.permute(1, 2, 0)

            # ---------- 损失 ----------
            l1_loss = F.l1_loss(rendered, target)
            ssim_loss = compute_ssim_loss(rendered, target, kernel=self._ssim_kernel)
            w_ssim = self.current_ssim_weight()
            loss = (1.0 - w_ssim) * l1_loss + w_ssim * ssim_loss

            for opt in self.optimizers.values():
                opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)
            for opt in self.optimizers.values():
                opt.step()

        self.current_step += 1
        self._update_lr()

        if self.adaptive_density is not None:
            self.adaptive_density.step()

        return loss.item()

    # ===== 修复: train_epoch 增加 start_frame 参数，支持从中断帧恢复 =====
    def train_epoch(self, frames_iter, camera_poses: List[Optional[np.ndarray]],
                    stop_event: Optional[threading.Event] = None,
                    progress_callback: Optional[Callable] = None,
                    loss_threshold: float = LOSS_THRESHOLD,
                    checkpoint_path: Optional[str] = None,
                    start_frame: int = 0) -> float:
        total_loss = 0.0
        processed_count = 0
        n = len(frames_iter) if hasattr(frames_iter, '__len__') else 0

        for i, frame in enumerate(frames_iter):
            # 跳过已完成的帧（用于断点续训）
            if i < start_frame:
                continue

            if stop_event and stop_event.is_set():
                raise KeyboardInterrupt("Stopped by user")
            if isinstance(frame, str):
                frame = _load_frame_from_path(frame)
            pose = camera_poses[i] if i < len(camera_poses) else None
            if pose is None:
                if progress_callback:
                    progress_callback(i + 1, n if n > 0 else i + 1, 0.0)
                continue
            loss = self.step(frame, pose)
            self.last_frame_index = i  # 帧级断点续训：记录最近处理过的帧下标
            total_loss += loss
            processed_count += 1

            if checkpoint_path and self.current_step % CHECKPOINT_INTERVAL_STEPS == 0:
                self.save_training_state(checkpoint_path)
            # 2026-08-14: 移除逐帧 torch.cuda.empty_cache() —— 分块光栅化后显存有界，
            #   逐帧释放只会迫使缓存分配器反复分配、拖慢训练（原为对抗网格爆炸的 OOM 添加）
            if loss_threshold and loss > loss_threshold:
                if checkpoint_path:
                    self.save_training_state(checkpoint_path)
                raise LossDivergenceError(f"Loss {loss:.4f} > threshold {loss_threshold} at frame {i+1}")
            if progress_callback:
                progress_callback(i + 1, n if n > 0 else i + 1, loss)

        avg_loss = total_loss / max(processed_count, 1)
        if checkpoint_path:
            self.save_training_state(checkpoint_path)
        return avg_loss

    # 2026-08: get_parameters() 已移除 —— 它返回激活后的值（exp(sigmoid)），会破坏官方 PLY 约定。
    # 改用 Gaussian3D.export_ply_dict() 返回原始参数。

    # ===== 修复: save_training_state 增加 best_loss 保存 =====
    def save_training_state(self, path: str) -> None:
        if self.train_focal:
            fx_val = self.fx.item() if isinstance(self.fx, torch.Tensor) else self.fx
            fy_val = self.fy.item() if isinstance(self.fy, torch.Tensor) else self.fy
        else:
            fx_val = self.fx if isinstance(self.fx, float) else float(self.fx)
            fy_val = self.fy if isinstance(self.fy, float) else float(self.fy)

        state = {
            "gaussian_params": {
                "positions": self.gaussians.positions.detach().cpu(),
                "log_scales": self.gaussians.log_scales.detach().cpu(),
                "opacities_raw": self.gaussians.opacities_raw.detach().cpu(),
                "rotations": self.gaussians.rotations.detach().cpu(),
                "sh_coeffs": self.gaussians.sh_coeffs.detach().cpu(),
            },
            "optimizer_states": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "step_count": self.current_step,
            "last_frame_index": self.last_frame_index,  # 2026-08: 帧级断点续训
            "best_loss": self.best_loss,  # ===== 新增 =====
            "background": self.background.detach().cpu(),
            "adaptive_density": {
                "step_count": self.adaptive_density._step_count,
                "cadence": self.adaptive_density._cadence,
                "opacity_accum": self.adaptive_density._opacity_accum.detach().cpu() if self.adaptive_density._opacity_accum is not None else None,
                "grad_accum": self.adaptive_density._grad_accum.detach().cpu() if self.adaptive_density._grad_accum is not None else None,
                "max_gaussians": self.adaptive_density.max_gaussians,
            },
            "sh_degree": self.sh_degree,
            "train_focal": self.train_focal,
            "enable_k1": self.enable_k1,
            "fx": fx_val,
            "fy": fy_val,
            "k1": self.k1.detach().cpu() if self.k1 is not None else 0.0,
            "sh_warmup_steps": self.sh_warmup_steps,
            "ssim_warmup_steps": self.ssim_warmup_steps,
            "ssim_weight_max": self.ssim_weight_max,
            "use_lr_schedule": self.use_lr_schedule,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_decay_gamma": self.lr_decay_gamma,
            "grad_thresh_base": self.adaptive_density.grad_thresh_base,
            "scale_thresh": self.adaptive_density.scale_thresh,
            "min_opacity": self.adaptive_density.min_opacity,
            "densify_every": self.adaptive_density.densify_every,
            "prune_every": self.adaptive_density.prune_every,
        }
        torch.save(state, path)

    # ===== 修复: load_training_state 支持高斯基数与当前初始化不同的续训 =====
    # 原实现用 .data.copy_() 把检查点参数写入当前高斯张量，但二者形状必须严格一致。
    # 密度自适应（duplicate 使总数 N→N+N_dup、prune 使总数减少）会让训练过程中的
    # 高斯基数与"本次从 SfM 点云初始化"的数量几乎必然不同 → copy_ 抛 shape 不匹配
    # → 上层捕获后从头开始，损失所有训练进度。
    # 正确做法：直接用检查点张量重建高斯参数（替换而非 copy），并重建优化器，
    # 让优化器状态（moment/var，形状 N）对新数量自动对齐（Adam 会按零初始化新槽）。
    def load_training_state(self, path: str, device: str = "cpu") -> None:
        state = torch.load(path, map_location=device, weights_only=False)
        device = torch.device(device)
        g = self.gaussians
        params = state["gaussian_params"]
        # 用检查点张量直接替换（而非 copy_），支持高斯基数变化
        g.positions = params["positions"].to(device).requires_grad_(True)
        g.log_scales = params["log_scales"].to(device).requires_grad_(True)
        g.opacities_raw = params["opacities_raw"].to(device).requires_grad_(True)
        g.rotations = params["rotations"].to(device).requires_grad_(True)
        g.sh_coeffs = params["sh_coeffs"].to(device).requires_grad_(True)

        self.current_step = state["step_count"]
        self.last_frame_index = state.get("last_frame_index", -1)  # 旧检查点无此键 → -1
        # ===== 2026-08 兼容提示 =====
        # sh_coeffs 约定已改为官方 (RGB-0.5)/C0（eval_sh 求值补 +0.5）。旧检查点的
        # sh_coeffs 存的是原始 RGB，在新约定下渲染会偏色 —— 预发布可接受，旧检查点需重训。
        self.best_loss = state.get("best_loss", float("inf"))  # ===== 新增（兼容旧检查点） =====
        self.sh_degree = state["sh_degree"]
        self.train_focal = state["train_focal"]
        self.enable_k1 = state["enable_k1"]

        fx_val = state["fx"]
        fy_val = state["fy"]
        if isinstance(fx_val, torch.Tensor):
            fx_val = fx_val.item()
        if isinstance(fy_val, torch.Tensor):
            fy_val = fy_val.item()
        fx_val = float(fx_val)
        fy_val = float(fy_val)

        if self.train_focal:
            self.fx = nn.Parameter(torch.tensor(fx_val, dtype=torch.float32, device=device))
            self.fy = nn.Parameter(torch.tensor(fy_val, dtype=torch.float32, device=device))
        else:
            self.fx = fx_val
            self.fy = fy_val

        if self.enable_k1:
            k1_data = state["k1"]
            if isinstance(k1_data, torch.Tensor):
                k1_val = k1_data.item()
            else:
                k1_val = float(k1_data)
            self.k1 = nn.Parameter(torch.tensor(k1_val, dtype=torch.float32, device=device))
        else:
            self.k1 = None

        self.sh_warmup_steps = state["sh_warmup_steps"]
        self.ssim_warmup_steps = state["ssim_warmup_steps"]
        self.ssim_weight_max = state["ssim_weight_max"]
        self.use_lr_schedule = state["use_lr_schedule"]
        self.lr_decay_steps = state["lr_decay_steps"]
        self.lr_decay_gamma = state["lr_decay_gamma"]

        self.background = state["background"].to(device).float()

        self._setup_optimizers()
        for name, opt in self.optimizers.items():
            if name in state["optimizer_states"]:
                opt.load_state_dict(state["optimizer_states"][name])

        ad = self.adaptive_density
        ad._step_count = state["adaptive_density"]["step_count"]
        ad._cadence = state["adaptive_density"].get("cadence", 0)  # 旧检查点无此键 → 0
        ad._opacity_accum = state["adaptive_density"]["opacity_accum"].to(device) if state["adaptive_density"]["opacity_accum"] is not None else None
        ad._grad_accum = state["adaptive_density"]["grad_accum"].to(device) if state["adaptive_density"]["grad_accum"] is not None else None
        ad.max_gaussians = state["adaptive_density"]["max_gaussians"]
        ad.grad_thresh_base = state.get("grad_thresh_base", GRAD_THRESH_BASE)
        ad.scale_thresh = state.get("scale_thresh", SCALE_THRESH)
        ad.min_opacity = state.get("min_opacity", MIN_OPACITY)
        ad.densify_every = state.get("densify_every", DENSIFY_EVERY)
        ad.prune_every = state.get("prune_every", PRUNE_EVERY)

        self._update_tanfov()


# ---------- Adaptive Density Controller ----------
class AdaptiveDensityController:
    def __init__(self, trainer: Trainer, densify_every: int = DENSIFY_EVERY,
                 prune_every: int = PRUNE_EVERY, max_gaussians: int = MAX_GAUSSIANS,
                 grad_thresh_base: float = GRAD_THRESH_BASE, scale_thresh: float = SCALE_THRESH,
                 min_opacity: float = MIN_OPACITY):
        self.trainer = trainer
        self.densify_every = densify_every
        self.prune_every = prune_every
        self.max_gaussians = max_gaussians
        self.grad_thresh_base = grad_thresh_base
        self.scale_thresh = scale_thresh
        self.min_opacity = min_opacity
        self._step_count = 0
        self._opacity_accum = None
        self._grad_accum = None
        # 修复(2026-08): 单调 cadence 计数器。旧版 densify 会重置 _step_count，
        #   prune_every(1000) 永远达不到 → 常规低透明度修剪从未执行。
        #   _step_count 仍是"窗口内步数"（用于累积器平均），_cadence 单调用于 densify/prune 节奏。
        self._cadence = 0

    def step(self) -> None:
        self._step_count += 1
        self._cadence += 1
        g = self.trainer.gaussians
        n = g.num_gaussians
        if n == 0:
            return

        if self._opacity_accum is None:
            self._opacity_accum = torch.zeros(n, device=g.positions.device)
            self._grad_accum = torch.zeros(n, device=g.positions.device)
        elif self._opacity_accum.shape[0] != n:
            if n > self._opacity_accum.shape[0]:
                pad = torch.zeros(n - self._opacity_accum.shape[0], device=g.positions.device)
                self._opacity_accum = torch.cat([self._opacity_accum, pad])
                self._grad_accum = torch.cat([self._grad_accum, pad])
            else:
                self._opacity_accum = self._opacity_accum[:n]
                self._grad_accum = self._grad_accum[:n]

        with torch.no_grad():
            self._opacity_accum += g.opacities.detach()
            grad = g.positions.grad
            if grad is not None:
                self._grad_accum += grad.norm(dim=-1).detach()

        if self.should_densify():
            stats = self.densify()
            print(f"  [DENSIFY] split={stats['split']}, duplicate={stats['duplicate']}")
        if self.should_prune():
            n_pruned = self.prune()
            if n_pruned > 0:
                print(f"  [PRUNE] removed {n_pruned} Gaussians")

    def should_densify(self) -> bool:
        return self._cadence > 0 and self._cadence % self.densify_every == 0

    def should_prune(self) -> bool:
        return self._cadence > 0 and self._cadence % self.prune_every == 0

    def reset_accumulators(self) -> None:
        # 只重置窗口计数器与累积器，不碰 _cadence（否则 prune 节奏再次被 densify 打断）
        self._step_count = 0
        self._opacity_accum = None
        self._grad_accum = None

    def densify(self) -> Dict[str, int]:
        g = self.trainer.gaussians
        n = g.num_gaussians
        stats = {"split": 0, "duplicate": 0}
        if n == 0 or self._step_count == 0:
            return stats

        avg_grad = self._grad_accum / max(1, self._step_count)
        avg_opacity = self._opacity_accum / max(1, self._step_count)
        # ===== 修复: grad_thresh 自适应匹配实际梯度量级 =====
        # 旧实现: grad_thresh = max(grad_thresh_base=0.0002, median*0.5)
        #   —— 硬底 0.0002 是官方 grad² 语义下的阈值，但本项目 loss 用 mean reduction
        #      （除以 H*W 像素），梯度被稀释到 ~1e-7，导致 0/所有高斯超过阈值，
        #      密度控制永远不触发。
        # 新实现: 完全自适应，用梯度分布的相对分位数决定分裂/复制阈值，
        #   不依赖绝对量级，无论 loss reduction 或场景尺度如何都能触发密度控制。
        nz_grad = avg_grad[avg_grad > 0]
        if nz_grad.numel() == 0:
            return stats
        # ===== 修复: grad_thresh 完全自适应，移除绝对硬底 =====
        # 旧实现: grad_thresh = max(quantile, eps*1e3)。eps*1e3≈1.19e-4 是"防数值
        #   误差"的绝对硬底，但本项目 loss 用 mean reduction，位置梯度量级 ~1e-7，
        #   硬底比所有梯度（最大 ~5e-6）还大 → n_gt=0，densify 永不触发。
        # 新实现: 用 p60 分位本身做阈值，仅在梯度为 0 的极端情况下用极小底防除零。
        grad_thresh = torch.quantile(nz_grad, 0.6)
        # 极小兜底：仅当 p60 为 0 时（全部梯度相同）才给一个相对 p95 的底
        if grad_thresh <= 0:
            grad_thresh = torch.quantile(nz_grad, 0.95)
        if grad_thresh <= 0:
            return stats
        max_log_scale = torch.max(g.log_scales, dim=1).values
        split_mask = (avg_grad > grad_thresh) & (max_log_scale > self.scale_thresh) & (avg_opacity > 0.01)
        duplicate_mask = (avg_grad > grad_thresh) & ~split_mask & (avg_opacity > 0.01)
        split_idx = torch.where(split_mask)[0]
        dup_idx = torch.where(duplicate_mask)[0]
        n_split = split_idx.numel()
        n_dup = dup_idx.numel()

        # 无分裂/复制 → 不触碰参数与优化器（累积器继续累积到下一个 densify 窗口）
        if n_split == 0 and n_dup == 0:
            return stats

        device = g.positions.device
        dtype = g.positions.dtype

        # 先捕获分裂/复制候选的原始张量（_prune_optimizer 改参后旧索引失效）
        if n_split > 0:
            base_pos = g.positions[split_idx].detach()
            base_log_scales = g.log_scales[split_idx].detach()
            base_opa = g.opacities_raw[split_idx].detach()
            base_rot = g.rotations[split_idx].detach()
            base_sh = g.sh_coeffs[split_idx].detach()
        if n_dup > 0:
            dup_pos = g.positions[dup_idx].detach()
            dup_log_scales = g.log_scales[dup_idx].detach()
            dup_opa = g.opacities_raw[dup_idx].detach()
            dup_rot = g.rotations[dup_idx].detach()
            dup_sh = g.sh_coeffs[dup_idx].detach()

        # 1. 删除 split 原体：keep ∪ dup 连同动量保留（dup 原体要保留动量）
        if n_split > 0:
            self.trainer._prune_optimizer(~split_mask)

        # 2. split 孩子：每颗 split 高斯 2 个孩子（尺度 0.8/0.6，原 numpy 逻辑逐行翻译）
        if n_split > 0:
            pos_parts: List[torch.Tensor] = []
            ls_parts: List[torch.Tensor] = []
            opa_parts: List[torch.Tensor] = []
            rot_parts: List[torch.Tensor] = []
            sh_parts: List[torch.Tensor] = []
            for scale_factor in [0.8, 0.6]:
                jitter = torch.randn(n_split, 3, device=device, dtype=dtype) * 0.001
                pos_parts.append(base_pos + jitter * (1.0 if scale_factor == 0.8 else -0.5))
                ls_parts.append(base_log_scales + np.log(scale_factor))
                opa_parts.append(base_opa + torch.randn(n_split, device=device, dtype=dtype) * 0.1)
                rot_parts.append(base_rot + torch.randn(n_split, 4, device=device, dtype=dtype) * 0.01)
                sh_parts.append(base_sh + torch.randn(n_split, base_sh.shape[1], base_sh.shape[2],
                                                      device=device, dtype=dtype) * 0.01)
            self.trainer._cat_tensors_to_optimizer({
                "positions": torch.cat(pos_parts, dim=0),
                "log_scales": torch.cat(ls_parts, dim=0),
                "opacities": torch.cat(opa_parts, dim=0),
                "rotations": torch.cat(rot_parts, dim=0),
                "sh": torch.cat(sh_parts, dim=0),
            })
            stats["split"] = n_split * 2

        # 3. dup clone：追加带微扰副本（dup 原体已在步骤 1 幸存，动量保留）
        if n_dup > 0:
            self.trainer._cat_tensors_to_optimizer({
                "positions": dup_pos + torch.randn(n_dup, 3, device=device, dtype=dtype) * 0.001,
                "log_scales": dup_log_scales,
                "opacities": dup_opa + torch.randn(n_dup, device=device, dtype=dtype) * 0.1,
                "rotations": dup_rot + torch.randn(n_dup, 4, device=device, dtype=dtype) * 0.01,
                "sh": dup_sh + torch.randn(n_dup, dup_sh.shape[1], dup_sh.shape[2],
                                           device=device, dtype=dtype) * 0.01,
            })
            stats["duplicate"] = n_dup

        self.reset_accumulators()

        n_current = g.num_gaussians
        if n_current > self.max_gaussians:
            n_remove = min(n_current - self.max_gaussians, int(n_current * 0.15))
            if n_remove > 0:
                self.prune(target_remove=n_remove)
        return stats

    def prune(self, target_remove: Optional[int] = None) -> int:
        g = self.trainer.gaussians
        n = g.num_gaussians
        if n == 0:
            return 0

        if self._opacity_accum is not None:
            avg_opacity = self._opacity_accum / max(1, self._step_count)
        else:
            avg_opacity = g.opacities

        if target_remove is not None and target_remove < n:
            vals, indices = torch.topk(avg_opacity, k=target_remove, largest=False)
            prune_mask = torch.zeros(n, dtype=torch.bool, device=g.positions.device)
            prune_mask[indices] = True
        else:
            prune_mask = avg_opacity < self.min_opacity
            if self._grad_accum is not None and n > 0:
                avg_grad = self._grad_accum / max(1, self._step_count)
                # ===== 修复: 保护阈值自适应（与 densify 的 grad_thresh 一致） =====
                # 旧实现: avg_grad > 0.005（官方 grad² 量级硬阈值），对 loss mean
                #   reduction 的梯度（~1e-7）永远为 False，保护机制失效。
                # 新实现: 用梯度中位数，保护"梯度高于中位数"的低透明度高斯。
                nz_grad = avg_grad[avg_grad > 0]
                if nz_grad.numel() > 0:
                    protect_thresh = torch.quantile(nz_grad, 0.5)
                    protect_mask = avg_grad > protect_thresh
                    prune_mask = prune_mask & ~protect_mask

        n_pruned = int(prune_mask.sum())
        if n_pruned == 0:
            return 0

        keep_mask = ~prune_mask
        # 原地裁剪 + 动量按 mask 保留（不再重建优化器，不再清空动量）
        self.trainer._prune_optimizer(keep_mask)
        self.reset_accumulators()
        return n_pruned