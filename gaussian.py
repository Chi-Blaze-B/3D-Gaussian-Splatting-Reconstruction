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
MAX_SPAN = 33
# 光栅化器分块尺寸：按深度有序高斯切块，单块显存上界 = RASTER_CHUNK × MAX_SPAN²
RASTER_CHUNK = 512


# ---------- Frame loader ----------
def _load_frame_from_path(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_frame_raw(path: str) -> np.ndarray:
    """读取为 uint8 RGB，内存占用为 float32 的 1/4，供预加载缓存使用。"""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class LazyFrames:
    """帧容器：内存缓存 uint8 + 按需转 float32。

    训练每 epoch 顺序遍历全部帧，读盘 + 解码是主要开销。本容器在构造时把
    全部帧解码为 uint8 RGB（内存约为 float32 的 1/4），访问时按需转 float32，
    避免全量 float32 的内存压力。preload=False 时回退为惰性加载。
    """

    def __init__(self, sources: List[Union[str, np.ndarray]], preload: bool = True,
                 cache_size: int = 0):
        self._sources = sources
        self._cache_size = max(0, cache_size)
        self._cache: OrderedDict = OrderedDict()  # 路径 -> float32 ndarray
        self._raw: Optional[List[Optional[np.ndarray]]] = None
        self._hits = 0
        self._misses = 0
        if preload:
            raw_list: List[Optional[np.ndarray]] = [None] * len(sources)
            for i, src in enumerate(sources):
                raw_list[i] = _load_frame_raw(src) if isinstance(src, str) else src
            self._raw = raw_list

    def __len__(self):
        return len(self._sources)

    def __iter__(self):
        for idx in range(len(self._sources)):
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        n = len(self._sources)
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(n))]
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Frame index {idx} out of range [0, {n-1}]")
        src = self._sources[idx]
        if not isinstance(src, str):
            return src
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
        raw_frame = self._raw[idx] if self._raw is not None else None
        if raw_frame is not None:
            return raw_frame.astype(np.float32) / 255.0
        return _load_frame_from_path(src)

    def preload(self) -> None:
        """手动触发预加载（幂等）。"""
        if self._raw is None:
            self._raw = [None] * len(self._sources)
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
    return M @ M.transpose(1, 2)


# ---------- Spherical Harmonics evaluation (up to degree 3) ----------
def eval_sh(deg: int, sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """求值球谐，返回 [N, 3] 颜色。

    sh_coeffs: [N, (deg+1)^2, 3]，DC 通道已存 (RGB-0.5)/C0；
    dirs: [N, 3] 单位方向（世界系：高斯中心 - 相机中心）。
    求值后补回 +0.5，deg0 时 color == RGB；clamp_min(0) 与官方光栅化器一致。
    """
    N = sh_coeffs.shape[0]
    device = sh_coeffs.device
    dtype = sh_coeffs.dtype

    dirs = F.normalize(dirs, dim=-1)
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    sh0 = torch.ones(N, 1, device=device, dtype=dtype) * 0.28209479177387814
    sh1 = torch.stack([
        0.4886025119029199 * y,
        0.4886025119029199 * z,
        0.4886025119029199 * x
    ], dim=1)
    sh2 = torch.stack([
        1.0925484305920792 * x * y,
        1.0925484305920792 * y * z,
        0.9461746957575601 * z * z - 0.31539156525252005,
        1.0925484305920792 * x * z,
        0.5462742152960396 * (x * x - y * y)
    ], dim=1)
    sh3 = torch.stack([
        0.5900435899266435 * y * (3 * x * x - y * y),
        2.890611442640554 * x * y * z,
        0.4570457994644658 * y * (5 * z * z - 1),
        0.3731763325901154 * z * (5 * z * z - 3),
        0.4570457994644658 * x * (5 * z * z - 1),
        1.445305721320277 * z * (x * x - y * y),
        0.5900435899266435 * x * (x * x - 3 * y * y)
    ], dim=1)

    basis_list = [sh0, sh1, sh2, sh3]
    basis = torch.cat(basis_list[:deg + 1], dim=1)

    color = torch.einsum('nc, ncd -> nd', basis, sh_coeffs[:, :basis.shape[1], :])
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

    def export_ply_dict(self) -> Dict[str, np.ndarray]:
        """导出官方 3DGS PLY 所需的原始参数（不做激活转换）。

        scale 存 log σ、opacity 存 logit、sh_coeffs 通道 0 已是 (RGB-0.5)/C0、
        rot 存 (w,x,y,z)。
        """
        return {
            "positions": self.positions.detach().cpu().numpy(),
            "scales": self.log_scales.detach().cpu().numpy(),
            "opacities": self.opacities_raw.detach().cpu().numpy(),
            "rotations": self.rotations.detach().cpu().numpy(),
            "sh_coeffs": self.sh_coeffs.detach().cpu().numpy(),
        }

    def initialize_from_dict(self, data: Dict[str, np.ndarray], device: str = "cpu") -> "Gaussian3D":
        device = torch.device(device)
        self.positions = torch.from_numpy(data["positions"]).float().to(device).clone()
        scales_np = data["scales"]
        if scales_np.ndim == 1 or scales_np.shape[1] == 1:
            scales_3 = np.repeat(scales_np, 3, axis=1) if scales_np.ndim == 2 else np.stack([scales_np] * 3, axis=1)
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
            t.requires_grad_(True)
        return self


def densify_initial_gaussians(gaussians: Gaussian3D, expansion_factor: int = 8, noise_scale: float = 0.02):
    """对初始稀疏高斯做 8× 稠密化：每颗高斯复制 expansion_factor 份并加微扰。"""
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
            new_pos.append(pos[i] + np.random.normal(0, noise_scale, 3).astype(np.float32))
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


# ---------- Differentiable Rasterizer (Pure PyTorch) ----------
class DifferentiableRasterizer(nn.Module):
    """排序式逐像素 splat 光栅化器。

    流程：
    1. 世界系 → 相机系（位置、协方差）。
    2. 投影到像素系，构建 2D 协方差与包围半径。
    3. 按深度 near→far 排序。
    4. 按深度有序高斯分块，块内展平覆盖像素对，按像素分组做
       stable 深度序的 over-blend；跨块用 log 空间 carry 保持透射率连续。
    """

    def __init__(self, image_width: int, image_height: int, max_radius: int = 16):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.max_radius = max_radius
        self._arange_cache: Optional[tuple] = None

    def _get_aranges(self, device, dtype):
        cache = self._arange_cache
        if cache is None or cache[0].device != device or cache[0].dtype != dtype:
            cache = (torch.arange(MAX_SPAN, device=device, dtype=dtype),
                     torch.arange(MAX_SPAN, device=device, dtype=dtype))
            self._arange_cache = cache
        return cache

    def forward(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K,
                background, sh_degree=3):
        N = positions.shape[0]
        H, W = self.image_height, self.image_width

        if N == 0:
            connected_zero = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
            zero = connected_zero.view(1).expand(H * W * 3).view(H, W, 3).contiguous()
            return zero, connected_zero.view(1).expand(H * W).view(H, W).contiguous()

        # ---- 相机系变换 ----
        R_cam = view_matrix[:3, :3]
        t_cam = view_matrix[:3, 3]
        cam_positions = positions @ R_cam.T + t_cam
        cam_cov = R_cam @ cov3d @ R_cam.T
        # SH 方向约定：世界系下 高斯中心 - 相机中心
        center_world = -R_cam.T @ t_cam

        # ---- 投影 ----
        fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
        z = cam_positions[:, 2].clamp(min=0.01)
        x_c = cam_positions[:, 0]; y_c = cam_positions[:, 1]
        u = fx * (x_c / z) + cx
        v = fy * (y_c / z) + cy

        # ---- 2D 协方差 ----
        B = torch.zeros(N, 2, 3, dtype=cov3d.dtype, device=cov3d.device)
        B[:, 0, 0] = fx / z; B[:, 0, 2] = -fx * x_c / (z * z)
        B[:, 1, 1] = fy / z; B[:, 1, 2] = -fy * y_c / (z * z)
        cov2d = (B @ cam_cov) @ B.transpose(1, 2)

        # ---- 半径（3σ，clamp 到 16 与 MAX_SPAN 匹配） ----
        a = cov2d[:, 0, 0]; c = cov2d[:, 1, 1]; b = cov2d[:, 0, 1]
        det = a * c - b * b
        trace = a + c
        disc = torch.clamp(trace ** 2 - 4 * det, min=1e-8)
        half = 0.5 * (trace + torch.sqrt(disc))
        sigma = torch.sqrt(half + 1e-6)
        radius = (sigma * 3.0).ceil().int().clamp(max=16)

        valid = (z > 0.01) & (radius > 0) & (radius < 1000)
        N_valid = int(valid.sum())
        if N_valid == 0:
            connected_zero = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
            zero = connected_zero.view(1).expand(H * W * 3).view(H, W, 3).contiguous()
            return zero, connected_zero.view(1).expand(H * W).view(H, W).contiguous()

        u_v, v_v, r_v = u[valid], v[valid], radius[valid]
        cov2d_v = cov2d[valid]
        op_v = opacities[valid]

        # ---- 深度排序（near → far） ----
        depth_sorted = cam_positions[valid][:, 2]
        order = torch.argsort(depth_sorted)
        u_s = u_v[order]; v_s = v_v[order]; r_s = r_v[order]
        cov2d_s = cov2d_v[order]; opa_s = op_v[order]

        dirs = F.normalize(positions[valid][order] - center_world, dim=-1)
        colors = eval_sh(sh_degree, sh_coeffs[valid][order], dirs)

        _graph_link = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
        out_color = _graph_link.view(1, 1, 1).expand(H, W, 3).contiguous()
        out_alpha = _graph_link.view(1, 1).expand(H, W).contiguous()

        mu_u = u_s; mu_v = v_s; rad = r_s
        A = cov2d_s[:, 0, 0]; B_ = cov2d_s[:, 0, 1]; C = cov2d_s[:, 1, 1]
        opa = opa_s; col = colors

        y_min = (mu_v - rad).clamp(min=0).int(); y_max = (mu_v + rad + 1).clamp(max=H).int()
        x_min = (mu_u - rad).clamp(min=0).int(); x_max = (mu_u + rad + 1).clamp(max=W).int()

        valid_b = (y_min < y_max) & (x_min < x_max)
        batch_n = int(valid_b.sum())
        if batch_n == 0:
            return out_color + background.view(1, 1, 3) * (1.0 - out_alpha.unsqueeze(-1)), out_alpha

        y_min_b = y_min[valid_b]; y_max_b = y_max[valid_b]
        x_min_b = x_min[valid_b]; x_max_b = x_max[valid_b]
        mu_u_b = mu_u[valid_b]; mu_v_b = mu_v[valid_b]
        A_b = A[valid_b]; B_b = B_[valid_b]; C_b = C[valid_b]
        opa_b = opa[valid_b]; col_b = col[valid_b]

        det_inv = 1.0 / (A_b * C_b - B_b * B_b + 1e-6)
        inv_A = det_inv * C_b; inv_B = -det_inv * B_b; inv_C = det_inv * A_b

        HpW = H * W
        device = colors.device
        # 单一 [HpW, 4] 缓冲：前 3 列 color，第 4 列 log(1-α) 累加，一次 index_add_ 完成
        acc = torch.zeros(HpW, 4, dtype=torch.float32, device=device)

        arange_h, arange_w = self._get_aranges(device, torch.float32)

        # 预计算所有 chunk 的 max_h/max_w，一次 GPU→CPU 同步替代 2×n_chunks 次
        n_chunks = (batch_n + RASTER_CHUNK - 1) // RASTER_CHUNK
        h_pad = torch.empty(n_chunks, device=device, dtype=torch.int32)
        w_pad = torch.empty(n_chunks, device=device, dtype=torch.int32)
        sizes_h = y_max_b - y_min_b
        sizes_w = x_max_b - x_min_b
        for k, start in enumerate(range(0, batch_n, RASTER_CHUNK)):
            end = min(start + RASTER_CHUNK, batch_n)
            h_pad[k] = sizes_h[start:end].max()
            w_pad[k] = sizes_w[start:end].max()
        h_list = h_pad.clamp(max=MAX_SPAN).tolist()
        w_list = w_pad.clamp(max=MAX_SPAN).tolist()

        for k, start in enumerate(range(0, batch_n, RASTER_CHUNK)):
            end = min(start + RASTER_CHUNK, batch_n)
            n_chunk = end - start
            max_h = h_list[k]
            max_w = w_list[k]
            if max_h == 0 or max_w == 0:
                continue

            y_lo = y_min_b[start:end]; y_hi = y_max_b[start:end]
            x_lo = x_min_b[start:end]; x_hi = x_max_b[start:end]
            mu_u_c = mu_u_b[start:end]; mu_v_c = mu_v_b[start:end]
            iA = inv_A[start:end]; iB = inv_B[start:end]; iC = inv_C[start:end]
            opa_c = opa_b[start:end]; col_c = col_b[start:end]

            gy = arange_h[:max_h].view(1, -1, 1)
            gx = arange_w[:max_w].view(1, 1, -1)
            gy_g = gy + y_lo.view(-1, 1, 1)
            gx_g = gx + x_lo.view(-1, 1, 1)
            dy = gy_g - mu_v_c.view(-1, 1, 1)
            dx = gx_g - mu_u_c.view(-1, 1, 1)

            y_valid = (gy_g >= y_lo.view(-1, 1, 1)) & (gy_g < y_hi.view(-1, 1, 1))
            x_valid = (gx_g >= x_lo.view(-1, 1, 1)) & (gx_g < x_hi.view(-1, 1, 1))
            valid_mask = y_valid & x_valid

            exponent = -(iA.view(-1, 1, 1) * dx ** 2
                         + 2 * iB.view(-1, 1, 1) * dx * dy
                         + iC.view(-1, 1, 1) * dy ** 2) * 0.5
            exponent = exponent.clamp(max=0)
            alpha = exponent.exp() * opa_c.view(-1, 1, 1)
            alpha = alpha.masked_fill(~valid_mask, 0.0)

            # 一次 nonzero 得到局部 (g, y, x)，替代三次 expand + 三次布尔索引
            g_idx, y_idx, x_idx = torch.nonzero(valid_mask, as_tuple=True)
            if g_idx.shape[0] == 0:
                continue
            flat_alpha = alpha[g_idx, y_idx, x_idx]
            gauss_ids = g_idx
            y_coord = y_lo[g_idx] + y_idx
            x_coord = x_lo[g_idx] + x_idx
            pix = y_coord * W + x_coord
            flat_color = col_c[gauss_ids]

            # 复合键排序：pix 主键 + gauss_ids（深度序）次键 → 非 stable sort 即保持深度序
            pix_key = pix * (n_chunk + 1) + gauss_ids
            pix_key_sorted, sort_idx = torch.sort(pix_key)
            pix_sorted = pix_key_sorted // (n_chunk + 1)

            a_sorted = flat_alpha[sort_idx]
            c_sorted = flat_color[sort_idx]

            # alpha 理论 ≤ 1，浮点误差可能达 1.0；clamp 防 log1p(-1) 产生 -inf 污染 cumsum
            a_safe = a_sorted.clamp(max=1.0 - 1e-7)
            log_ta = torch.log1p(-a_safe)
            log_cum = torch.cumsum(log_ta, dim=0)
            log_cum_shift = torch.cat([
                torch.zeros(1, dtype=log_cum.dtype, device=log_cum.device),
                log_cum[:-1]
            ])

            # 分段透射率：把 cumsum 前缀平移到每段段首
            new_group = pix_sorted[1:] != pix_sorted[:-1]
            group_starts = torch.cat([torch.tensor([True], device=pix_sorted.device), new_group])
            arange = torch.arange(group_starts.shape[0], device=pix_sorted.device)
            group_start_pos = torch.where(group_starts, arange, torch.zeros_like(arange))
            group_start_pos = torch.cummax(group_start_pos, dim=0).values
            seg_offset = log_cum_shift[group_start_pos]
            log_T_before_chunk = log_cum_shift - seg_offset

            # 跨块 carry：块内 exclusive 前缀 + 前序块对该像素的累计 log(1-α)
            carry = acc[pix_sorted, 3]
            log_T_before = carry + log_T_before_chunk
            T_before = torch.exp(log_T_before.clamp(min=-50.0))
            weight = a_sorted * T_before

            # 一次 index_add_ 同时散射 color 与 log_ta
            payload = torch.cat([
                weight.unsqueeze(-1) * c_sorted,
                log_ta.unsqueeze(-1),
            ], dim=-1)
            acc.index_add_(0, pix_sorted, payload)

        out_color = out_color.view(HpW, 3) + acc[:, :3]
        out_alpha = out_alpha.view(HpW) + (1.0 - torch.exp(acc[:, 3].clamp(min=-50.0)))
        out_color = out_color.view(H, W, 3)
        out_alpha = out_alpha.view(H, W)

        out_color = out_color + background.view(1, 1, 3) * (1.0 - out_alpha.unsqueeze(-1))
        return out_color, out_alpha


# ---------- Loss functions ----------
def compute_ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
                      kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = img1.shape[-1]
    if kernel is None:
        kernel = torch.ones((channels, 1, window_size, window_size),
                            dtype=img1.dtype, device=img1.device) / (window_size ** 2)
    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)
    mu_x = F.conv2d(x, kernel, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2, groups=channels)
    mu_xx, mu_yy = mu_x ** 2, mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, kernel, padding=window_size // 2, groups=channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, kernel, padding=window_size // 2, groups=channels) - mu_yy
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2, groups=channels) - mu_xy
    ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
           ((mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2))
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
                 use_amp: bool = False,
                 use_lr_schedule: bool = USE_LR_SCHEDULE,
                 lr_decay_steps: int = LR_DECAY_STEPS, lr_decay_gamma: float = LR_DECAY_GAMMA,
                 grad_thresh_base: float = GRAD_THRESH_BASE, scale_thresh: float = SCALE_THRESH,
                 min_opacity: float = MIN_OPACITY, densify_every: int = DENSIFY_EVERY,
                 prune_every: int = PRUNE_EVERY):
        self.device = device
        self.image_height = image_height
        self.image_width = image_width

        self.use_cuda_rasterizer = False
        print("[INFO] Using PyTorch rasterizer (supports SH up to 3).")

        self.gaussians = gaussians
        self.K = torch.from_numpy(K.astype(np.float32)).to(device)
        self.view_matrix = torch.eye(4, dtype=torch.float32, device=device)
        self.random_background = random_background
        self.train_focal = train_focal
        # AMP 仅 CUDA 生效
        self.use_amp = use_amp and torch.cuda.is_available() and str(device).startswith("cuda")
        if self.use_amp:
            try:
                _scaler_cls = getattr(torch.amp, "GradScaler")
                self._scaler = _scaler_cls("cuda", enabled=True)
            except (AttributeError, TypeError):
                self._scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self._scaler = None
        self.sh_degree = min(sh_degree, 3)
        self.sh_warmup_steps = sh_warmup_steps
        self.ssim_warmup_steps = ssim_warmup_steps
        self.ssim_weight_max = ssim_weight_max

        if self.train_focal:
            self.fx = nn.Parameter(torch.tensor(K[0, 0], dtype=torch.float32, device=device))
            self.fy = nn.Parameter(torch.tensor(K[1, 1], dtype=torch.float32, device=device))
        else:
            self.fx = float(K[0, 0])
            self.fy = float(K[1, 1])

        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

        self.lr_positions = LR_POSITIONS; self.lr_log_scales = LR_LOG_SCALES
        self.lr_opacities = LR_OPACITIES; self.lr_rotations = LR_ROTATIONS
        self.lr_sh = LR_SH; self.lr_focal = LR_FOCAL
        self.use_lr_schedule = use_lr_schedule
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_gamma = lr_decay_gamma
        self.current_step = 0
        self.last_frame_index = -1
        self.best_loss = float("inf")
        self.background = torch.rand(3, dtype=torch.float32, device=device)
        channels = 3
        self._ssim_kernel = torch.ones((channels, 1, 11, 11),
                                       dtype=torch.float32, device=device) / 121.0
        self.adaptive_density = AdaptiveDensityController(
            self, densify_every, prune_every, max_gaussians,
            grad_thresh_base, scale_thresh, min_opacity)
        if rasterizer is None:
            self.rasterizer = DifferentiableRasterizer(image_width, image_height)
        else:
            self.rasterizer = rasterizer
        self._update_tanfov()
        # 初始 8× 稠密化必须先于 _setup_optimizers：优化器包裹的是最终张量
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
        focal_params = [self.fx, self.fy] if self.train_focal and isinstance(self.fx, nn.Parameter) else []
        if focal_params:
            self.optimizers["focal"] = torch.optim.Adam(focal_params, lr=self.lr_focal)

    # ---------- 密度控制：Adam 动量原地保留 ----------
    # densify/prune 后不重建优化器，只在尾部补零动量 / 按 mask 裁剪动量。
    # 关键：cat 活参会产生非叶子张量 → .grad 为 None → Adam 静默跳过。
    # 必须从 .detach() 的片段构造真叶子。
    _GAUSS_ATTR = {
        "positions": "positions", "log_scales": "log_scales",
        "opacities": "opacities_raw", "rotations": "rotations", "sh": "sh_coeffs",
    }

    def _cat_tensors_to_optimizer(self, new_tensors: Dict[str, torch.Tensor]) -> None:
        """尾部追加新高斯：存量动量按行保留，新增行补零。"""
        g = self.gaussians
        for name, opt in self.optimizers.items():
            if name not in new_tensors or name not in self._GAUSS_ATTR:
                continue
            t = new_tensors[name].detach()
            p = opt.param_groups[0]["params"][0]
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
        """按 bool mask 裁剪参数：幸存者 Adam 动量按 mask 保留。"""
        g = self.gaussians
        for name, opt in self.optimizers.items():
            if name not in self._GAUSS_ATTR:
                continue
            p = opt.param_groups[0]["params"][0]
            # p[mask] 已是新存储，detach 后即叶子；无需 clone
            new_p = p[mask].detach().requires_grad_(True)
            setattr(g, self._GAUSS_ATTR[name], new_p)
            opt.param_groups[0]["params"][0] = new_p
            stored = opt.state.get(p)
            if stored is not None and "exp_avg" in stored:
                stored["exp_avg"] = stored["exp_avg"][mask]
                stored["exp_avg_sq"] = stored["exp_avg_sq"][mask]
                del opt.state[p]
                opt.state[new_p] = stored

    def _update_lr(self):
        if not self.use_lr_schedule or self.lr_decay_steps <= 0:
            return
        # lr 只在 lr_decay_steps 整数倍处变化，非整步直接跳过
        if self.current_step % self.lr_decay_steps != 0:
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

    def step(self, target_image: Union[np.ndarray, torch.Tensor],
             camera_pose: Optional[np.ndarray] = None) -> float:
        if camera_pose is not None:
            self.view_matrix = torch.from_numpy(camera_pose.astype(np.float32)).to(self.device)

        if self.random_background:
            # 随机黑白背景：三通道同值
            self.background = torch.randint(0, 2, (1,), device=self.device,
                                            dtype=torch.float32).expand(3)
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
        opacities = self.gaussians.opacities
        sh_coeffs = self.gaussians.sh_coeffs

        eff_deg = self.effective_sh_degree()

        viewmat = self.view_matrix
        K = torch.zeros(3, 3, dtype=torch.float32, device=self.device)
        # 焦距以 nn.Parameter 直接写入 K，保留计算图（.item() 会切断梯度）
        K[0, 0] = self.fx if isinstance(self.fx, nn.Parameter) else float(self.fx)
        K[1, 1] = self.fy if isinstance(self.fy, nn.Parameter) else float(self.fy)
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K[2, 2] = 1.0

        params_to_clip = [
            self.gaussians.positions, self.gaussians.log_scales,
            self.gaussians.opacities_raw, self.gaussians.rotations,
            self.gaussians.sh_coeffs
        ]
        if self.train_focal and isinstance(self.fx, nn.Parameter):
            params_to_clip.append(self.fx); params_to_clip.append(self.fy)

        amp_on = self.use_amp and self._scaler is not None
        if amp_on:
            assert self._scaler is not None
            # AMP 仅覆盖 cov3d 构建与损失侧；光栅化器内部保持 fp32
            # （cumsum/scatter 不吃 Tensor Core，硬上 fp16 伤数值）。
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                cov3d = self.gaussians.cov3d
                with torch.autocast(device_type="cuda", enabled=False):
                    rendered, _ = self.rasterizer(
                        means3D, cov3d.float(), opacities, sh_coeffs, viewmat, K,
                        self.background, sh_degree=eff_deg,
                    )
                if rendered.dim() == 3 and rendered.shape[0] == 3:
                    rendered = rendered.permute(1, 2, 0)
                l1_loss = F.l1_loss(rendered, target)
                ssim_loss = compute_ssim_loss(rendered, target, kernel=self._ssim_kernel)
                w_ssim = self.current_ssim_weight()
                loss = (1.0 - w_ssim) * l1_loss + w_ssim * ssim_loss

            for opt in self.optimizers.values():
                opt.zero_grad()
            self._scaler.scale(loss).backward()
            # 逐优化器守卫：无梯度的优化器跳过 scaler.step，否则抛
            # "No inf checks were recorded for this optimizer"
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
                means3D, cov3d, opacities, sh_coeffs, viewmat, K, self.background,
                sh_degree=eff_deg
            )
            if rendered.dim() == 3 and rendered.shape[0] == 3:
                rendered = rendered.permute(1, 2, 0)

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
            self.last_frame_index = i
            total_loss += loss
            processed_count += 1

            if checkpoint_path and self.current_step % CHECKPOINT_INTERVAL_STEPS == 0:
                self.save_training_state(checkpoint_path)
            if loss_threshold and loss > loss_threshold:
                if checkpoint_path:
                    self.save_training_state(checkpoint_path)
                raise LossDivergenceError(
                    f"Loss {loss:.4f} > threshold {loss_threshold} at frame {i+1}")
            if progress_callback:
                progress_callback(i + 1, n if n > 0 else i + 1, loss)

        avg_loss = total_loss / max(processed_count, 1)
        if checkpoint_path:
            self.save_training_state(checkpoint_path)
        return avg_loss

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
            "last_frame_index": self.last_frame_index,
            "best_loss": self.best_loss,
            "background": self.background.detach().cpu(),
            "adaptive_density": {
                "step_count": self.adaptive_density._step_count,
                "cadence": self.adaptive_density._cadence,
                "opacity_accum": self.adaptive_density._opacity_accum.detach().cpu()
                if self.adaptive_density._opacity_accum is not None else None,
                "grad_accum": self.adaptive_density._grad_accum.detach().cpu()
                if self.adaptive_density._grad_accum is not None else None,
                "max_gaussians": self.adaptive_density.max_gaussians,
            },
            "sh_degree": self.sh_degree,
            "train_focal": self.train_focal,
            "fx": fx_val,
            "fy": fy_val,
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

    def load_training_state(self, path: str, device: str = "cpu") -> None:
        """从检查点恢复。

        直接以检查点张量重建高斯参数（而非 copy_），以支持高斯基数变化：
        密度自适应会让训练过程中的 N 与初始 N 不同，copy_ 会因形状不匹配失败。
        """
        state = torch.load(path, map_location=device, weights_only=False)
        device = torch.device(device)
        g = self.gaussians
        params = state["gaussian_params"]
        g.positions = params["positions"].to(device).requires_grad_(True)
        g.log_scales = params["log_scales"].to(device).requires_grad_(True)
        g.opacities_raw = params["opacities_raw"].to(device).requires_grad_(True)
        g.rotations = params["rotations"].to(device).requires_grad_(True)
        g.sh_coeffs = params["sh_coeffs"].to(device).requires_grad_(True)

        self.current_step = state["step_count"]
        self.last_frame_index = state["last_frame_index"]
        self.best_loss = state["best_loss"]
        self.sh_degree = state["sh_degree"]
        self.train_focal = state["train_focal"]

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
        ad._cadence = state["adaptive_density"]["cadence"]
        ad._opacity_accum = state["adaptive_density"]["opacity_accum"].to(device) \
            if state["adaptive_density"]["opacity_accum"] is not None else None
        ad._grad_accum = state["adaptive_density"]["grad_accum"].to(device) \
            if state["adaptive_density"]["grad_accum"] is not None else None
        ad.max_gaussians = state["adaptive_density"]["max_gaussians"]
        ad.grad_thresh_base = state["grad_thresh_base"]
        ad.scale_thresh = state["scale_thresh"]
        ad.min_opacity = state["min_opacity"]
        ad.densify_every = state["densify_every"]
        ad.prune_every = state["prune_every"]

        self._update_tanfov()


# ---------- Adaptive Density Controller ----------
class AdaptiveDensityController:
    """密度自适应控制器。

    - 每 densify_every 步：按梯度分位数分裂/复制高斯（grad_thresh 完全自适应，
      不依赖绝对量级，兼容不同 loss reduction 与场景尺度）。
    - 每 prune_every 步：移除低透明度高斯，同时保护高梯度高斯。
    - _step_count 是窗口内步数（用于累积器平均），_cadence 单调计数用于触发节奏，
      避免 densify 重置 _step_count 后 prune 永不触发。
    """

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
            print(f"[稠密化] 分裂了{stats['split']}个高斯, 复制{stats['duplicate']}了个高斯")
            # 低显存用户依赖此清理把缓存归还驱动，避免溢出到共享显存
            torch.cuda.empty_cache()
        if self.should_prune():
            n_pruned = self.prune()
            if n_pruned > 0:
                print(f"[修剪] 移除了 {n_pruned} 个高斯")
            torch.cuda.empty_cache()

    def should_densify(self) -> bool:
        return self._cadence > 0 and self._cadence % self.densify_every == 0

    def should_prune(self) -> bool:
        return self._cadence > 0 and self._cadence % self.prune_every == 0

    def reset_accumulators(self) -> None:
        # 只重置窗口计数与累积器，不碰 _cadence
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

        # grad_thresh 完全自适应：用非零梯度的 p60 分位做阈值，
        # 不依赖绝对量级，兼容 loss mean reduction 与任意场景尺度。
        nz_grad = avg_grad[avg_grad > 0]
        if nz_grad.numel() == 0:
            return stats
        grad_thresh = torch.quantile(nz_grad, 0.6)
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

        # 1. 删除 split 原体：keep ∪ dup 连同动量保留
        if n_split > 0:
            self.trainer._prune_optimizer(~split_mask)

        # 2. split 孩子：每颗 split 高斯 2 个孩子（尺度 0.8 / 0.6）
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
                # 保护阈值自适应：用非零梯度中位数保护"高梯度"低透明度高斯
                nz_grad = avg_grad[avg_grad > 0]
                if nz_grad.numel() > 0:
                    protect_thresh = torch.quantile(nz_grad, 0.5)
                    protect_mask = avg_grad > protect_thresh
                    prune_mask = prune_mask & ~protect_mask

        n_pruned = int(prune_mask.sum())
        if n_pruned == 0:
            return 0

        keep_mask = ~prune_mask
        self.trainer._prune_optimizer(keep_mask)
        self.reset_accumulators()
        return n_pruned