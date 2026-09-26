"""
3D 高斯泼溅：核心模块、光栅化器与训练器。纯 PyTorch 实现，无 CUDA 扩展。
球谐函数阶数最高 3 阶。
"""

import os
import gc
import logging
import math
import sys
import threading
from dataclasses import dataclass, field, replace
from typing import Optional, List, Dict, Union, Callable
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------- 训练超参数 ----------
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
SH_WARMUP_STEPS = 1000
SSIM_WARMUP_STEPS = 500
SSIM_WEIGHT_MAX = 0.2
GRAD_CLIP_NORM = 10.0
LOSS_THRESHOLD = 1.0
CHECKPOINT_INTERVAL_STEPS = 500
LR_DECAY_STEPS = 1000
LR_DECAY_GAMMA = 0.998
USE_LR_SCHEDULE = True


# ---------- 硬件探测 ----------
def _detect_gpu_total_memory_gb(device_index: int = 0) -> float:
    """探测 GPU 总显存（GB）。无 CUDA 或失败返回 0。"""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.get_device_properties(device_index).total_memory) / (1024 ** 3)
    except Exception as e:
        logger.warning("探测 GPU 显存失败：%s", e)
        return 0.0


def _detect_system_memory_gb() -> float:
    """探测系统内存（GB）。psutil → sysconf → Windows ctypes。"""
    try:
        import psutil  # type: ignore
        return float(psutil.virtual_memory().total) / (1024 ** 3)
    except ImportError:
        pass
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return float(pages * page_size) / (1024 ** 3)
        except (ValueError, OSError, AttributeError):
            pass

    if sys.platform == "win32":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return float(stat.ullTotalPhys) / (1024 ** 3)
        except Exception:
            pass

    return 0.0


# ---------- 渲染配置 ----------
@dataclass(frozen=True)
class RenderConfig:
    """光栅化器 + 密度上限的硬件相关配置。"""
    raster_chunk: int
    radius_max: int
    max_gaussians: int
    source: str
    hardware_gb: float

    @property
    def max_span(self) -> int:
        return 2 * self.radius_max + 1


def _tune_for_gpu(vram_gb: float) -> RenderConfig:
    if vram_gb < 4:
        return RenderConfig(64, 3, 100_000, "cuda", vram_gb)
    if vram_gb < 6:
        return RenderConfig(96, 5, 200_000, "cuda", vram_gb)
    if vram_gb < 8:
        return RenderConfig(128, 6, 300_000, "cuda", vram_gb)
    if vram_gb < 12:
        return RenderConfig(192, 10, 500_000, "cuda", vram_gb)
    if vram_gb < 16:
        return RenderConfig(256, 12, 700_000, "cuda", vram_gb)
    if vram_gb < 24:
        return RenderConfig(384, 14, 1_000_000, "cuda", vram_gb)
    return RenderConfig(512, 16, 1_500_000, "cuda", vram_gb)


def _tune_for_cpu(ram_gb: float) -> RenderConfig:
    """CPU 分档：max_gaussians 保持保守，避免 RAM 打爆。"""
    if ram_gb < 8:
        return RenderConfig(32, 6, 50_000, "cpu", ram_gb)
    if ram_gb < 16:
        return RenderConfig(48, 7, 100_000, "cpu", ram_gb)
    if ram_gb < 32:
        return RenderConfig(64, 8, 200_000, "cpu", ram_gb)
    if ram_gb < 64:
        return RenderConfig(96, 9, 300_000, "cpu", ram_gb)
    if ram_gb < 128:
        return RenderConfig(128, 10, 400_000, "cpu", ram_gb)
    return RenderConfig(192, 12, 600_000, "cpu", ram_gb)


def auto_tune_config(device: Optional[str] = None,
                     vram_gb: Optional[float] = None,
                     ram_gb: Optional[float] = None,
                     device_index: int = 0) -> RenderConfig:
    """按实际训练设备推导渲染配置。

    device 决定分档依据：cuda → 显存；cpu → 系统内存；None/auto → 自动判断。
    """
    if device is None or device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    if use_cuda:
        if vram_gb is None:
            vram_gb = _detect_gpu_total_memory_gb(device_index)
        if vram_gb > 0:
            cfg = _tune_for_gpu(vram_gb)
            logger.info(
                "[AutoTune] 设备=%s，GPU 显存 %.1fGB → raster_chunk=%d radius_max=%d "
                "max_span=%d max_gaussians=%d",
                device or "cuda", cfg.hardware_gb, cfg.raster_chunk, cfg.radius_max,
                cfg.max_span, cfg.max_gaussians,
            )
            return cfg
        logger.warning("[AutoTune] 指定 CUDA 但探测不到显存，回落到系统内存分档。")

    if ram_gb is None:
        ram_gb = _detect_system_memory_gb()
    if ram_gb > 0:
        cfg = _tune_for_cpu(ram_gb)
        logger.info(
            "[AutoTune] 设备=%s，系统内存 %.1fGB → raster_chunk=%d radius_max=%d "
            "max_span=%d max_gaussians=%d",
            device or "cpu", cfg.hardware_gb, cfg.raster_chunk, cfg.radius_max,
            cfg.max_span, cfg.max_gaussians,
        )
        return cfg

    cfg = RenderConfig(32, 6, 50_000, "fallback", 0.0)
    logger.warning(
        "[AutoTune] 无法探测系统内存，使用保守默认 → "
        "raster_chunk=%d radius_max=%d max_span=%d max_gaussians=%d",
        cfg.raster_chunk, cfg.radius_max, cfg.max_span, cfg.max_gaussians,
    )
    return cfg


# ---------- 帧加载 ----------
def _load_frame_from_path(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_frame_raw(path: str) -> np.ndarray:
    """读取为 uint8 RGB（内存为 float32 的 1/4）。"""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class LazyFrames:
    """帧容器：预载 uint8，按需转 float32。"""

    def __init__(self, sources: List[Union[str, np.ndarray]], preload: bool = True,
                 cache_size: int = 0):
        self._sources = sources
        self._cache_size = max(0, cache_size)
        self._cache: OrderedDict = OrderedDict()
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


# ---------- 四元数工具 ----------
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
    # 协方差 = R @ diag(s) @ R^T，等价于把尺度乘到 R 的每一列。
    M = R * s.unsqueeze(1)
    return M @ M.transpose(1, 2)


# ---------- 球谐求值 ----------
def eval_sh(deg: int, sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """求值球谐，返回 [N, 3] 颜色。"""
    dirs = F.normalize(dirs, dim=-1)
    if deg == 0:
        return torch.clamp(sh_coeffs[:, 0] * 0.28209479177387814 + 0.5, min=0.0)

    N = sh_coeffs.shape[0]
    device = sh_coeffs.device
    dtype = sh_coeffs.dtype
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

    basis = torch.cat([sh0, sh1, sh2, sh3][:deg + 1], dim=1)
    color = torch.einsum('nc, ncd -> nd', basis, sh_coeffs[:, :basis.shape[1], :])
    return torch.clamp(color + 0.5, min=0.0)


# ---------- 高斯表示 ----------
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
    """对初始稀疏高斯做复制加噪（全向量化）。"""
    n = gaussians.num_gaussians
    if n == 0:
        return
    device = gaussians.positions.device
    dtype = gaussians.positions.dtype
    S = gaussians.sh_coeffs.shape[1]

    idx = torch.arange(n, device=device).repeat_interleave(expansion_factor)
    m = n * expansion_factor

    new_pos = gaussians.positions.detach()[idx] + \
        torch.randn(m, 3, device=device, dtype=dtype) * noise_scale
    new_log_scales = gaussians.log_scales.detach()[idx] + math.log(0.8)
    new_opa = gaussians.opacities_raw.detach()[idx] + \
        torch.randn(m, device=device, dtype=dtype) * 0.1
    new_rot = gaussians.rotations.detach()[idx] + \
        torch.randn(m, 4, device=device, dtype=dtype) * 0.01
    new_sh = gaussians.sh_coeffs.detach()[idx] + \
        torch.randn(m, S, 3, device=device, dtype=dtype) * 0.01

    gaussians.positions = new_pos.requires_grad_(True)
    gaussians.log_scales = new_log_scales.requires_grad_(True)
    gaussians.opacities_raw = new_opa.requires_grad_(True)
    gaussians.rotations = new_rot.requires_grad_(True)
    gaussians.sh_coeffs = new_sh.requires_grad_(True)


# ---------- 可微光栅化器（纯 PyTorch） ----------
class DifferentiableRasterizer(nn.Module):
    def __init__(self, image_width: int, image_height: int,
                 raster_chunk: int, radius_max: int):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.raster_chunk = int(raster_chunk)
        self.radius_max = int(radius_max)
        self.max_span = 2 * self.radius_max + 1
        self._arange_cache: Optional[tuple] = None

    @classmethod
    def from_config(cls, image_width: int, image_height: int,
                    config: RenderConfig) -> "DifferentiableRasterizer":
        return cls(image_width, image_height,
                   raster_chunk=config.raster_chunk,
                   radius_max=config.radius_max)

    def _get_aranges(self, device, dtype):
        cache = self._arange_cache
        if cache is None or cache[0].device != device or cache[0].dtype != dtype:
            cache = (torch.arange(self.max_span, device=device, dtype=dtype),
                     torch.arange(self.max_span, device=device, dtype=dtype))
            self._arange_cache = cache
        return cache

    def _zero_output(self, positions, opacities, H, W):
        """N=0 或无有效高斯时的零输出，保持图连接。"""
        link = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
        zero = link.view(1, 1, 1).expand(H, W, 3)
        alpha = link.view(1, 1).expand(H, W)
        return zero, alpha

    def forward(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K,
                background, sh_degree=3):
        N = positions.shape[0]
        H, W = self.image_height, self.image_width
        max_span = self.max_span
        chunk = self.raster_chunk
        radius_max = self.radius_max

        if N == 0:
            return self._zero_output(positions, opacities, H, W)

        R_cam = view_matrix[:3, :3]
        t_cam = view_matrix[:3, 3]
        cam_positions = positions @ R_cam.T + t_cam
        cam_cov = R_cam @ cov3d @ R_cam.T
        center_world = -R_cam.T @ t_cam

        fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
        z = cam_positions[:, 2].clamp(min=0.01)
        x_c = cam_positions[:, 0]; y_c = cam_positions[:, 1]
        u = fx * (x_c / z) + cx
        v = fy * (y_c / z) + cy

        # 2D 协方差 —— 保留梯度
        B = torch.zeros(N, 2, 3, dtype=cov3d.dtype, device=cov3d.device)
        B[:, 0, 0] = fx / z
        B[:, 0, 2] = -fx * x_c / (z * z)
        B[:, 1, 1] = fy / z
        B[:, 1, 2] = -fy * y_c / (z * z)
        cov2d = (B @ cam_cov) @ B.transpose(1, 2)

        # tile 半径与有效性：纯索引，不需要梯度
        with torch.no_grad():
            a = cov2d[:, 0, 0].detach()
            c = cov2d[:, 1, 1].detach()
            b = cov2d[:, 0, 1].detach()
            det = a * c - b * b
            trace = a + c
            disc = torch.clamp(trace ** 2 - 4 * det, min=1e-8)
            sigma = torch.sqrt(0.5 * (trace + torch.sqrt(disc)) + 1e-6)
            radius = (sigma * 3.0).ceil().int().clamp(max=radius_max)
            valid = (z.detach() > 0.01) & (radius > 0) & (radius < 1000)
            n_valid = int(valid.sum())

        if n_valid == 0:
            return self._zero_output(positions, opacities, H, W)

        u_v = u[valid]; v_v = v[valid]; r_v = radius[valid]
        cov2d_v = cov2d[valid]
        op_v = opacities[valid]
        depth_v = cam_positions[valid][:, 2]

        # 深度排序：索引不需要梯度
        with torch.no_grad():
            order = torch.argsort(depth_v)

        u_s = u_v[order]; v_s = v_v[order]; r_s = r_v[order]
        cov2d_s = cov2d_v[order]; opa_s = op_v[order]

        dirs = F.normalize(positions[valid][order] - center_world, dim=-1)
        colors = eval_sh(sh_degree, sh_coeffs[valid][order], dirs)

        # 零标量，仅用于保持图连接
        link = positions.sum() * 0.0

        mu_u = u_s; mu_v = v_s; rad = r_s
        A = cov2d_s[:, 0, 0]; B_ = cov2d_s[:, 0, 1]; C = cov2d_s[:, 1, 1]
        opa = opa_s; col = colors

        # tile 边界：整数，不需要梯度
        with torch.no_grad():
            y_min = (mu_v - rad).clamp(min=0).int()
            y_max = (mu_v + rad + 1).clamp(max=H).int()
            x_min = (mu_u - rad).clamp(min=0).int()
            x_max = (mu_u + rad + 1).clamp(max=W).int()
            valid_b = (y_min < y_max) & (x_min < x_max)
            batch_n = int(valid_b.sum())

        if batch_n == 0:
            # 空 tile：零图 + 背景
            zero_alpha = link.view(1, 1).expand(H, W)
            zero_color = link.view(1, 1, 1).expand(H, W, 3)
            return (zero_color + background.view(1, 1, 3) * (1.0 - zero_alpha.unsqueeze(-1)),
                    zero_alpha)

        y_min_b = y_min[valid_b]; y_max_b = y_max[valid_b]
        x_min_b = x_min[valid_b]; x_max_b = x_max[valid_b]
        mu_u_b = mu_u[valid_b]; mu_v_b = mu_v[valid_b]
        A_b = A[valid_b]; B_b = B_[valid_b]; C_b = C[valid_b]
        opa_b = opa[valid_b]; col_b = col[valid_b]

        det_inv = 1.0 / (A_b * C_b - B_b * B_b + 1e-6)
        inv_A = det_inv * C_b; inv_B = -det_inv * B_b; inv_C = det_inv * A_b

        HpW = H * W
        device = colors.device
        acc = torch.zeros(HpW, 4, dtype=torch.float32, device=device)

        arange_h, arange_w = self._get_aranges(device, torch.float32)

        n_chunks = (batch_n + chunk - 1) // chunk

        # 一次 D2H 取全部 chunk 的 tile 边界，避免逐 chunk host 同步
        with torch.no_grad():
            sizes_h_np = (y_max_b - y_min_b).clamp(max=max_span).cpu().numpy()
            sizes_w_np = (x_max_b - x_min_b).clamp(max=max_span).cpu().numpy()
        h_list = [int(sizes_h_np[k * chunk:min((k + 1) * chunk, batch_n)].max())
                  for k in range(n_chunks)]
        w_list = [int(sizes_w_np[k * chunk:min((k + 1) * chunk, batch_n)].max())
                  for k in range(n_chunks)]

        for k in range(n_chunks):
            start = k * chunk
            end = min(start + chunk, batch_n)
            n_chunk = end - start
            max_h = h_list[k]
            max_w = w_list[k]
            if max_h == 0 or max_w == 0:
                continue

            y_lo_c = y_min_b[start:end]; y_hi_c = y_max_b[start:end]
            x_lo_c = x_min_b[start:end]; x_hi_c = x_max_b[start:end]
            mu_u_c = mu_u_b[start:end]; mu_v_c = mu_v_b[start:end]
            iA_c = inv_A[start:end]; iB_c = inv_B[start:end]; iC_c = inv_C[start:end]
            opa_c = opa_b[start:end]; col_c = col_b[start:end]

            # ---------- 阶段 1：索引、排序、分组（no_grad） ----------
            with torch.no_grad():
                gy = arange_h[:max_h].view(1, -1, 1)
                gx = arange_w[:max_w].view(1, 1, -1)
                gy_g = gy + y_lo_c.view(-1, 1, 1)
                gx_g = gx + x_lo_c.view(-1, 1, 1)

                y_valid = (gy_g >= y_lo_c.view(-1, 1, 1)) & (gy_g < y_hi_c.view(-1, 1, 1))
                x_valid = (gx_g >= x_lo_c.view(-1, 1, 1)) & (gx_g < x_hi_c.view(-1, 1, 1))
                valid_mask = y_valid & x_valid

                g_idx, y_idx, x_idx = torch.nonzero(valid_mask, as_tuple=True)
                if g_idx.shape[0] == 0:
                    continue

                y_abs = y_lo_c[g_idx] + y_idx
                x_abs = x_lo_c[g_idx] + x_idx
                y_coord = y_abs.float()
                x_coord = x_abs.float()
                pix = y_abs * W + x_abs

                pix_key = pix * (n_chunk + 1) + g_idx
                pix_key_sorted, sort_idx = torch.sort(pix_key)
                pix_sorted = pix_key_sorted // (n_chunk + 1)

                gauss_sorted = g_idx[sort_idx]
                y_coord_s = y_coord[sort_idx]
                x_coord_s = x_coord[sort_idx]

                # 每个像素的分组起始位置
                new_group = pix_sorted[1:] != pix_sorted[:-1]
                group_starts = torch.cat([
                    torch.tensor([True], device=pix_sorted.device), new_group])
                arange = torch.arange(group_starts.shape[0], device=pix_sorted.device)
                group_start_pos = torch.where(group_starts, arange,
                                              torch.zeros_like(arange))
                group_start_pos = torch.cummax(group_start_pos, dim=0).values

            # ---------- 阶段 2：alpha 与混合（保留梯度） ----------
            mu_u_sel = mu_u_c[gauss_sorted]
            mu_v_sel = mu_v_c[gauss_sorted]
            iA_sel = iA_c[gauss_sorted]
            iB_sel = iB_c[gauss_sorted]
            iC_sel = iC_c[gauss_sorted]
            opa_sel = opa_c[gauss_sorted]
            col_sel = col_c[gauss_sorted]

            dx = x_coord_s - mu_u_sel
            dy = y_coord_s - mu_v_sel
            exponent = -(iA_sel * dx ** 2
                         + 2 * iB_sel * dx * dy
                         + iC_sel * dy ** 2) * 0.5
            exponent = exponent.clamp(max=0)
            a_sorted = exponent.exp() * opa_sel
            c_sorted = col_sel

            a_safe = a_sorted.clamp(max=1.0 - 1e-7)
            log_ta = torch.log1p(-a_safe)
            log_cum = torch.cumsum(log_ta, dim=0)
            log_cum_shift = torch.cat([
                torch.zeros(1, dtype=log_cum.dtype, device=log_cum.device),
                log_cum[:-1]])

            seg_offset = log_cum_shift[group_start_pos]
            log_T_before_chunk = log_cum_shift - seg_offset

            carry = acc[pix_sorted, 3]
            log_T_before = carry + log_T_before_chunk
            T_before = torch.exp(log_T_before.clamp(min=-50.0))
            weight = a_sorted * T_before

            payload = torch.cat([
                weight.unsqueeze(-1) * c_sorted,
                log_ta.unsqueeze(-1),
            ], dim=-1)
            acc.index_add_(0, pix_sorted, payload)

        # 由 acc 直接合成，省去 H×W 零基线物化
        out_color = (acc[:, :3] + link).reshape(H, W, 3)
        out_alpha = (1.0 - torch.exp(acc[:, 3].clamp(min=-50.0)) + link).reshape(H, W)

        out_color = out_color + background.view(1, 1, 3) * (1.0 - out_alpha.unsqueeze(-1))
        return out_color, out_alpha


# ---------- 损失函数 ----------
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


# ---------- 训练器 ----------
class Trainer:
    def __init__(self, gaussians: Gaussian3D, K: np.ndarray,
                 image_width: int, image_height: int, device: str = "cpu",
                 rasterizer: Optional[DifferentiableRasterizer] = None,
                 sh_degree: int = 3,
                 random_background: bool = True, train_focal: bool = True,
                 render_config: Optional[RenderConfig] = None,
                 max_gaussians: Optional[int] = None,
                 sh_warmup_steps: int = SH_WARMUP_STEPS,
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

        logger.info("使用 PyTorch 光栅化器（SH 最高 3 阶）。")

        cfg = render_config if render_config is not None else auto_tune_config(device=device)
        if max_gaussians is not None and max_gaussians != cfg.max_gaussians:
            cfg = replace(cfg, max_gaussians=int(max_gaussians))
        self.render_config = cfg

        self.gaussians = gaussians
        self.K = torch.from_numpy(K.astype(np.float32)).to(device)
        self.view_matrix = torch.eye(4, dtype=torch.float32, device=device)
        self.random_background = random_background
        self.train_focal = train_focal
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
        self._ssim_kernel = torch.ones((3, 1, 11, 11),
                                       dtype=torch.float32, device=device) / 121.0
        self.adaptive_density = AdaptiveDensityController(
            self, densify_every, prune_every, cfg.max_gaussians,
            grad_thresh_base, scale_thresh, min_opacity)

        if rasterizer is None:
            self.rasterizer = DifferentiableRasterizer.from_config(
                image_width, image_height, cfg)
        else:
            self.rasterizer = rasterizer

        self._update_tanfov()
        if self.gaussians.num_gaussians < 2000:
            densify_initial_gaussians(self.gaussians, expansion_factor=8, noise_scale=0.02)
            logger.info("[INIT] 已稠密化到 %d 个高斯", self.gaussians.num_gaussians)
        self._setup_optimizers()
        self._pre_train_cache_cleared = False

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

    _GAUSS_ATTR = {
        "positions": "positions", "log_scales": "log_scales",
        "opacities": "opacities_raw", "rotations": "rotations", "sh": "sh_coeffs",
    }

    def _cat_tensors_to_optimizer(self, new_tensors: Dict[str, torch.Tensor]) -> None:
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
        g = self.gaussians
        for name, opt in self.optimizers.items():
            if name not in self._GAUSS_ATTR:
                continue
            p = opt.param_groups[0]["params"][0]
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
        if self.train_focal and isinstance(self.fx, nn.Parameter):
            # 先用常量建 K，再 index_put_ 原地嵌入参数——保留 autograd 连接
            K = torch.eye(3, dtype=torch.float32, device=self.device)
            K[0, 2] = self.cx
            K[1, 2] = self.cy
            K[0, 0] = self.fx
            K[1, 1] = self.fy
        else:
            K = torch.zeros(3, 3, dtype=torch.float32, device=self.device)
            K[0, 0] = float(self.fx)
            K[1, 1] = float(self.fy)
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.autocast(device_type="cuda", enabled=False):
                    # cov3d 与光栅化器都在 fp32 下算，避免协方差精度损失
                    cov3d = self.gaussians.cov3d
                    rendered, _ = self.rasterizer(
                        means3D, cov3d, opacities, sh_coeffs, viewmat, K,
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
            rendered, _ = self.rasterizer.forward(
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
                    loss_threshold: float = LOSS_THRESHOLD,
                    checkpoint_path: Optional[str] = None,
                    start_frame: int = 0,
                    progress_callback: Optional[Callable[[int, int, float], None]] = None) -> float:
        total_loss = 0.0
        processed_count = 0

        try:
            total_frames = len(frames_iter)  # type: ignore[arg-type]
        except TypeError:
            total_frames = len(camera_poses)

        try:
            if not self._pre_train_cache_cleared:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._pre_train_cache_cleared = True

            for i, frame in enumerate(frames_iter):
                if i < start_frame:
                    continue

                if stop_event and stop_event.is_set():
                    raise KeyboardInterrupt("Stopped by user")
                if isinstance(frame, str):
                    frame = _load_frame_from_path(frame)
                pose = camera_poses[i] if i < len(camera_poses) else None
                if pose is None:
                    continue
                loss = self.step(frame, pose)
                self.last_frame_index = i
                total_loss += loss
                processed_count += 1

                if progress_callback is not None:
                    progress_callback(i + 1, total_frames, loss)

                if checkpoint_path and self.current_step % CHECKPOINT_INTERVAL_STEPS == 0:
                    self.save_training_state(checkpoint_path)
                if loss_threshold and loss > loss_threshold:
                    if checkpoint_path:
                        self.save_training_state(checkpoint_path)
                    raise LossDivergenceError(
                        f"Loss {loss:.4f} > threshold {loss_threshold} at frame {i+1}")

            avg_loss = total_loss / max(processed_count, 1)
            if checkpoint_path:
                self.save_training_state(checkpoint_path)
            return avg_loss
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            "render_config": {
                "raster_chunk": self.render_config.raster_chunk,
                "radius_max": self.render_config.radius_max,
                "max_gaussians": self.render_config.max_gaussians,
                "source": self.render_config.source,
                "hardware_gb": self.render_config.hardware_gb,
            },
        }
        torch.save(state, path)

    def _release_training_state(self) -> None:
        g = self.gaussians
        dev = g.positions.device
        g.positions = torch.empty(0, 3, device=dev)
        g.log_scales = torch.empty(0, 3, device=dev)
        g.opacities_raw = torch.empty(0, device=dev)
        g.rotations = torch.empty(0, 4, device=dev)
        g.sh_coeffs = torch.empty(0, 16, 3, device=dev)
        if hasattr(self, "optimizers"):
            self.optimizers.clear()
        ad = self.adaptive_density
        ad._opacity_accum = None
        ad._grad_accum = None
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def load_training_state(self, path: str, device: str = "cpu") -> None:
        device = torch.device(device)

        self._release_training_state()
        state = torch.load(path, map_location="cpu", weights_only=False)

        g = self.gaussians

        gp = state.pop("gaussian_params")
        for attr in ("positions", "log_scales", "opacities_raw", "rotations", "sh_coeffs"):
            t_cpu = gp.pop(attr)
            setattr(g, attr, t_cpu.to(device).requires_grad_(True))
            del t_cpu
        del gp

        self.current_step = state["step_count"]
        self.last_frame_index = state["last_frame_index"]
        self.best_loss = state["best_loss"]
        self.sh_degree = state["sh_degree"]
        self.train_focal = state["train_focal"]

        fx_val = state["fx"]; fy_val = state["fy"]
        if isinstance(fx_val, torch.Tensor):
            fx_val = fx_val.item()
        if isinstance(fy_val, torch.Tensor):
            fy_val = fy_val.item()
        fx_val = float(fx_val); fy_val = float(fy_val)

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
        opt_states = state.pop("optimizer_states", {})
        for name, opt in self.optimizers.items():
            if name not in opt_states:
                continue
            sd_cpu = opt_states.pop(name)
            opt.load_state_dict(sd_cpu)
            del sd_cpu
        del opt_states

        ad = self.adaptive_density
        ad_state = state.pop("adaptive_density")
        ad._step_count = ad_state["step_count"]
        ad._cadence = ad_state["cadence"]
        oa = ad_state["opacity_accum"]
        ga = ad_state["grad_accum"]
        ad._opacity_accum = oa.to(device) if oa is not None else None
        ad._grad_accum = ga.to(device) if ga is not None else None
        ad.max_gaussians = ad_state["max_gaussians"]
        ad.grad_thresh_base = state["grad_thresh_base"]
        ad.scale_thresh = state["scale_thresh"]  # setter 自动重算 log_scale_thresh
        ad.min_opacity = state["min_opacity"]
        ad.densify_every = state["densify_every"]
        ad.prune_every = state["prune_every"]
        del ad_state

        rcfg = state.pop("render_config", None)
        if rcfg is not None and isinstance(self.rasterizer, DifferentiableRasterizer):
            new_cfg = RenderConfig(
                raster_chunk=int(rcfg["raster_chunk"]),
                radius_max=int(rcfg["radius_max"]),
                max_gaussians=int(rcfg["max_gaussians"]),
                source=rcfg.get("source", "?"),
                hardware_gb=float(rcfg.get("hardware_gb", 0.0)),
            )
            self.render_config = new_cfg
            self.rasterizer.raster_chunk = new_cfg.raster_chunk
            self.rasterizer.radius_max = new_cfg.radius_max
            self.rasterizer.max_span = new_cfg.max_span
            self.rasterizer._arange_cache = None
            logger.info(
                "[load] 恢复渲染配置：raster_chunk=%d radius_max=%d max_span=%d "
                "max_gaussians=%d (来源=%s, %.1fGB)",
                new_cfg.raster_chunk, new_cfg.radius_max, new_cfg.max_span,
                new_cfg.max_gaussians, new_cfg.source, new_cfg.hardware_gb,
            )

        del state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self._update_tanfov()
        self._pre_train_cache_cleared = False


# ---------- 密度自适应控制器 ----------
class AdaptiveDensityController:
    """梯度驱动的密度自适应。

    每 densify_every 步按梯度分位数分裂/复制高斯；
    每 prune_every 步移除低透明度高斯，并保护高梯度高斯。
    """

    def __init__(self, trainer: Trainer, densify_every: int = DENSIFY_EVERY,
                 prune_every: int = PRUNE_EVERY, max_gaussians: int = 300_000,
                 grad_thresh_base: float = GRAD_THRESH_BASE, scale_thresh: float = SCALE_THRESH,
                 min_opacity: float = MIN_OPACITY):
        self.trainer = trainer
        self.densify_every = densify_every
        self.prune_every = prune_every
        self.max_gaussians = max_gaussians
        self.grad_thresh_base = grad_thresh_base
        self._scale_thresh = scale_thresh
        # log 域等价的尺度阈值：log(σ) > log(scale_thresh) 等价于 σ > scale_thresh
        self._log_scale_thresh = math.log(scale_thresh)
        self.min_opacity = min_opacity
        self._step_count = 0
        self._opacity_accum = None
        self._grad_accum = None
        self._cadence = 0

    @property
    def scale_thresh(self) -> float:
        return self._scale_thresh

    @scale_thresh.setter
    def scale_thresh(self, value: float) -> None:
        self._scale_thresh = value
        # 必须同步重算 log 阈值，否则 split_mask 会静默失效
        self._log_scale_thresh = math.log(value)

    @property
    def log_scale_thresh(self) -> float:
        return self._log_scale_thresh

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
            logger.info("[密度自适应][稠密化] 分裂 %d 个高斯, 复制 %d 个高斯",
                        stats["split"], stats["duplicate"])
        if self.should_prune():
            n_pruned = self.prune()
            if n_pruned > 0:
                logger.info("[密度自适应][修剪] 移除 %d 个高斯", n_pruned)

    def should_densify(self) -> bool:
        return self._cadence > 0 and self._cadence % self.densify_every == 0

    def should_prune(self) -> bool:
        return self._cadence > 0 and self._cadence % self.prune_every == 0

    def reset_accumulators(self) -> None:
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

        nz_grad = avg_grad[avg_grad > 0]
        if nz_grad.numel() == 0:
            return stats
        grad_thresh = torch.quantile(nz_grad, 0.6)
        if grad_thresh <= 0:
            grad_thresh = torch.quantile(nz_grad, 0.95)
        if grad_thresh <= 0:
            return stats

        max_log_scale = torch.max(g.log_scales, dim=1).values
        split_mask = (avg_grad > grad_thresh) & (max_log_scale > self.log_scale_thresh) & (avg_opacity > 0.01)
        duplicate_mask = (avg_grad > grad_thresh) & ~split_mask & (avg_opacity > 0.01)
        split_idx = torch.where(split_mask)[0]
        dup_idx = torch.where(duplicate_mask)[0]
        n_split = split_idx.numel()
        n_dup = dup_idx.numel()

        if n_split == 0 and n_dup == 0:
            return stats

        device = g.positions.device
        dtype = g.positions.dtype

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

        if n_split > 0:
            self.trainer._prune_optimizer(~split_mask)

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
            self.prune(enforce_cap=True)
        return stats

    def prune(self, target_remove: Optional[int] = None, enforce_cap: bool = False) -> int:
        g = self.trainer.gaussians
        n = g.num_gaussians
        if n == 0:
            return 0

        if self._opacity_accum is not None:
            avg_opacity = self._opacity_accum / max(1, self._step_count)
        else:
            avg_opacity = g.opacities

        # enforce_cap：把"硬裁剪到 max_gaussians"独立成一条分支
        if enforce_cap:
            if n <= self.max_gaussians:
                return 0
            target_remove = n - self.max_gaussians

        if target_remove is not None and target_remove < n:
            vals, indices = torch.topk(avg_opacity, k=target_remove, largest=False)
            prune_mask = torch.zeros(n, dtype=torch.bool, device=g.positions.device)
            prune_mask[indices] = True
        else:
            prune_mask = avg_opacity < self.min_opacity
            if self._grad_accum is not None and n > 0:
                avg_grad = self._grad_accum / max(1, self._step_count)
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
        # 硬裁剪时衰减幸存高斯的 Adam 动量，降低分布突变导致的发散风险
        if enforce_cap and n_pruned > 0:
            for opt in self.trainer.optimizers.values():
                for p, state in opt.state.items():
                    if "exp_avg" in state:
                        state["exp_avg"] *= 0.5
                        state["exp_avg_sq"] *= 0.5
        self.reset_accumulators()
        return n_pruned