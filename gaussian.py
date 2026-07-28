"""
3D Gaussian Splatting — core representation, rasterizer, and training.

This module implements:
  - Gaussian3D: learnable per‑point Gaussian parameters.
  - DifferentiableRasterizer: fallback PyTorch rasterizer (SH0 only).
  - Trainer: main training loop with L1 + SSIM loss, CUDA rasterizer integration,
    density control, learning rate scheduling, and checkpointing.
  - AdaptiveDensityController: split/duplicate/prune with budget control.

All hyperparameters are centralized at the top for easy tuning.
Checkpoints are strict: loading expects exactly the same structure as saved.
"""

import os
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Optional CUDA rasterizer ----------
try:
    from diff_gaussian_rasterization import GaussianRasterizer as CUDARasterizer, RasterizationSettings
    HAS_CUDA_RASTERIZER = True
except ImportError:
    HAS_CUDA_RASTERIZER = False
    CUDARasterizer = None
    RasterizationSettings = None

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
MAX_GAUSSIANS = 300_000
SH_WARMUP_STEPS = 1000
SSIM_WARMUP_STEPS = 500
SSIM_WEIGHT_MAX = 0.2
GRAD_CLIP_NORM = 10.0
LOSS_THRESHOLD = 1.0
CHECKPOINT_INTERVAL_STEPS = 500
LR_DECAY_STEPS = 1000
LR_DECAY_GAMMA = 0.998
USE_LR_SCHEDULE = True


# ---------- Frame loader ----------
def _load_frame_from_path(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class LazyFrames:
    """Lazy loader for a list of frame paths or numpy arrays."""
    def __init__(self, sources: List[Union[str, np.ndarray]]):
        self._sources = sources

    def __len__(self) -> int:
        return len(self._sources)

    def __iter__(self):
        for src in self._sources:
            yield self._load(src)

    def __getitem__(self, idx):
        n = len(self._sources)
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(n))]
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Frame index {idx} out of range [0, {n-1}]")
        return self._load(self._sources[idx])

    @staticmethod
    def _load(src):
        if isinstance(src, str):
            return _load_frame_from_path(src)
        return src


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


def build_covariance(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    R = quat_to_rot(rotations)
    s = scales.exp()
    M = R * s.unsqueeze(1)
    cov = M @ M.transpose(1, 2)
    return cov


# ---------- SH evaluation (used only for custom rasterizer fallback, SH0 only) ----------
def eval_sh_0(sh_coeffs: torch.Tensor) -> torch.Tensor:
    """Return the 0th order SH coefficient (diffuse color), shape (N, 3)."""
    return sh_coeffs[:, 0, :] + 0.5   # assuming coefficients are stored as 0-centered


# ---------- Gaussian3D ----------
@dataclass
class Gaussian3D:
    positions: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3))
    log_scales: torch.Tensor = field(default_factory=lambda: torch.empty(0, 1))
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

    def initialize_from_dict(self, data: Dict[str, np.ndarray], device: str = "cpu") -> "Gaussian3D":
        device = torch.device(device)
        self.positions = torch.from_numpy(data["positions"]).float().to(device).clone()
        self.log_scales = torch.from_numpy(data["scales"]).float().to(device).clone()
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


# ---------- Differentiable Rasterizer (fallback, SH0 only) ----------
class DifferentiableRasterizer(nn.Module):
    """Pure PyTorch rasterizer (SH0 only). Used when CUDA rasterizer is unavailable."""
    def __init__(self, image_width: int, image_height: int, max_radius: int = 32):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.max_radius = max_radius

    def forward(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background):
        return self._render_batch(positions, cov3d, opacities, sh_coeffs, view_matrix, K, background)

    def _render_batch(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background):
        N = positions.shape[0]
        H, W = self.image_height, self.image_width
        if N == 0:
            zero = torch.zeros((H, W, 3), dtype=positions.dtype, device=positions.device)
            return zero, zero[..., 0]

        R_cam = view_matrix[:3, :3]
        t_cam = view_matrix[:3, 3]
        cam_positions = positions @ R_cam.T + t_cam
        cam_cov = R_cam @ cov3d @ R_cam.T

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        z = cam_positions[:, 2].clamp(min=0.01)
        x_c = cam_positions[:, 0]
        y_c = cam_positions[:, 1]
        u = fx * (x_c / z) + cx
        v = fy * (y_c / z) + cy

        B = torch.zeros(N, 2, 3, dtype=cov3d.dtype, device=cov3d.device)
        B[:, 0, 0] = fx / z
        B[:, 0, 2] = -fx * x_c / (z * z)
        B[:, 1, 1] = fy / z
        B[:, 1, 2] = -fy * y_c / (z * z)

        cov2d = (B @ cam_cov) @ B.transpose(1, 2)
        a = cov2d[:, 0, 0]
        c = cov2d[:, 1, 1]
        b = cov2d[:, 0, 1]
        det = a * c - b * b
        trace = a + c
        half = 0.5 * (trace + torch.sqrt(trace.clamp(min=0) ** 2 - 4 * det.clamp(min=0)))
        sigma = torch.sqrt(half + 1e-6)
        radius = (sigma * 3.0).ceil().int().clamp(max=self.max_radius)

        valid = (z > 0.01) & (radius > 0) & (radius < 1000)
        if not valid.any():
            zero = torch.zeros((H, W, 3), dtype=positions.dtype, device=positions.device)
            return zero, zero[..., 0]

        u_v, v_v, r_v = u[valid], v[valid], radius[valid]
        cov2d_v = cov2d[valid]
        op_v = opacities[valid]
        N_valid = int(valid.sum())

        # Depth sorting
        depth_sorted = cam_positions[valid][:, 2]
        order = torch.argsort(depth_sorted, descending=True)
        u_s = u_v[order]
        v_s = v_v[order]
        r_s = r_v[order]
        cov2d_s = cov2d_v[order]
        opa_s = op_v[order]

        # SH0 colors
        colors = eval_sh_0(sh_coeffs)[valid][order]

        out_color = torch.zeros((H, W, 3), dtype=colors.dtype, device=colors.device)
        out_alpha = torch.zeros((H, W), dtype=colors.dtype, device=colors.device)

        # Tile rendering with batching
        MAX_BATCH = 64
        for start in range(0, N_valid, MAX_BATCH):
            end = min(start + MAX_BATCH, N_valid)
            mu_u = u_s[start:end]
            mu_v = v_s[start:end]
            rad = r_s[start:end]
            A = cov2d_s[start:end, 0, 0]
            B_ = cov2d_s[start:end, 0, 1]
            C = cov2d_s[start:end, 1, 1]
            opa = opa_s[start:end]
            col = colors[start:end]

            y_min = (mu_v - rad).clamp(min=0).int()
            y_max = (mu_v + rad + 1).clamp(max=H).int()
            x_min = (mu_u - rad).clamp(min=0).int()
            x_max = (mu_u + rad + 1).clamp(max=W).int()

            valid_b = (y_min < y_max) & (x_min < x_max)
            if not valid_b.any():
                continue

            y_min_b = y_min[valid_b]; y_max_b = y_max[valid_b]
            x_min_b = x_min[valid_b]; x_max_b = x_max[valid_b]
            mu_u_b = mu_u[valid_b]; mu_v_b = mu_v[valid_b]
            A_b = A[valid_b]; B_b = B_[valid_b]; C_b = C[valid_b]
            opa_b = opa[valid_b]; col_b = col[valid_b]
            batch_n = int(valid_b.sum())

            det_inv = 1.0 / (A_b * C_b - B_b * B_b + 1e-6)
            inv_A = det_inv * C_b
            inv_B = -det_inv * B_b
            inv_C = det_inv * A_b

            h_sizes = (y_max_b - y_min_b).int()
            w_sizes = (x_max_b - x_min_b).int()
            max_h = int(h_sizes.max())
            max_w = int(w_sizes.max())
            if max_h == 0 or max_w == 0:
                continue

            gy = torch.arange(max_h, device=colors.device, dtype=torch.float32).view(1, -1, 1)
            gx = torch.arange(max_w, device=colors.device, dtype=torch.float32).view(1, 1, -1)
            gy_g = gy + y_min_b.view(-1, 1, 1)
            gx_g = gx + x_min_b.view(-1, 1, 1)
            dy = gy_g - mu_v_b.view(-1, 1, 1)
            dx = gx_g - mu_u_b.view(-1, 1, 1)

            y_valid = (gy_g >= y_min_b.view(-1, 1, 1)) & (gy_g < y_max_b.view(-1, 1, 1))
            x_valid = (gx_g >= x_min_b.view(-1, 1, 1)) & (gx_g < x_max_b.view(-1, 1, 1))
            valid_mask = y_valid & x_valid

            exponent = -(inv_A.view(-1, 1, 1) * dx ** 2 +
                         2 * inv_B.view(-1, 1, 1) * dx * dy +
                         inv_C.view(-1, 1, 1) * dy ** 2) * 0.5
            exponent = exponent.clamp(max=0)
            alpha = exponent.exp() * opa_b.view(-1, 1, 1)
            alpha = alpha.masked_fill(~valid_mask, 0.0)

            for i in range(batch_n):
                y_lo = int(y_min_b[i]); y_hi = int(y_max_b[i])
                x_lo = int(x_min_b[i]); x_hi = int(x_max_b[i])
                h_i = min(y_hi - y_lo, alpha.shape[1])
                w_i = min(x_hi - x_lo, alpha.shape[2])
                if h_i <= 0 or w_i <= 0:
                    continue
                a_i = alpha[i, :h_i, :w_i]
                c_i = col_b[i]
                old_alpha = out_alpha[y_lo:y_hi, x_lo:x_hi]
                out_color[y_lo:y_hi, x_lo:x_hi] += (
                    a_i.unsqueeze(-1) * (1.0 - old_alpha.unsqueeze(-1)) * c_i
                )
                out_alpha[y_lo:y_hi, x_lo:x_hi] += a_i * (1.0 - old_alpha)

        out_color = out_color + background.view(1, 1, 3) * (1.0 - out_alpha.unsqueeze(-1))
        return out_color, out_alpha


# ---------- Loss functions ----------
def compute_weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss with edge‑aware weighting."""
    diff = (pred - target).abs()
    grad_x = torch.abs(target[:, 1:, :] - target[:, :-1, :])
    grad_y = torch.abs(target[1:, :, :] - target[:-1, :, :])
    gx_avg = grad_x.mean(dim=-1, keepdim=True)
    gy_avg = grad_y.mean(dim=-1, keepdim=True)
    h, w = pred.shape[:2]
    weight_map = torch.ones((h, w, 1), dtype=pred.dtype, device=pred.device)
    cy, cx = gy_avg.shape[0], gx_avg.shape[1]
    if cx > 0:
        weight_map[:, :cx, 0] += gx_avg[:, :, 0]
        weight_map[:, 1:cx+1, 0] += gx_avg[:, :, 0]
    if cy > 0:
        weight_map[:cy, :, 0] += gy_avg[:, :, 0]
        weight_map[1:cy+1, :, 0] += gy_avg[:, :, 0]
    max_w = weight_map.max().clamp(min=1e-6)
    weight = (1.0 + 0.2 * (weight_map / max_w)).clamp(max=1.2)
    return (diff * weight).mean()


def compute_ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural Similarity Index loss."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = img1.shape[-1]
    kernel = torch.ones((channels, 1, window_size, window_size), dtype=img1.dtype,
                        device=img1.device) / (window_size ** 2)
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
    def __init__(
        self,
        gaussians: Gaussian3D,
        rasterizer: Optional[DifferentiableRasterizer],
        K: np.ndarray,
        image_width: int,
        image_height: int,
        device: str = "cpu",
        use_cuda_rasterizer: bool = True,
        sh_degree: int = 3,
        random_background: bool = True,
        train_focal: bool = True,
        max_gaussians: int = MAX_GAUSSIANS,
        sh_warmup_steps: int = SH_WARMUP_STEPS,
        ssim_warmup_steps: int = SSIM_WARMUP_STEPS,
        ssim_weight_max: float = SSIM_WEIGHT_MAX,
        enable_k1: bool = False,
        use_lr_schedule: bool = USE_LR_SCHEDULE,
        lr_decay_steps: int = LR_DECAY_STEPS,
        lr_decay_gamma: float = LR_DECAY_GAMMA,
        grad_thresh_base: float = GRAD_THRESH_BASE,
        scale_thresh: float = SCALE_THRESH,
        min_opacity: float = MIN_OPACITY,
        densify_every: int = DENSIFY_EVERY,
        prune_every: int = PRUNE_EVERY,
    ):
        self.gaussians = gaussians
        self.K = torch.from_numpy(K.astype(np.float32)).to(device)
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        self.view_matrix = torch.eye(4, dtype=torch.float32, device=device)

        self.random_background = random_background
        self.train_focal = train_focal
        self.enable_k1 = enable_k1

        self.use_cuda_rasterizer = (use_cuda_rasterizer and HAS_CUDA_RASTERIZER and device == "cuda")
        if not self.use_cuda_rasterizer and sh_degree > 0:
            print(f"  [WARN] CUDA rasterizer not available, SH>0 will be ignored. Install diff-gaussian-rasterization.")
            sh_degree = 0
        self.sh_degree = min(sh_degree, 3)
        self.sh_warmup_steps = sh_warmup_steps
        self.ssim_warmup_steps = ssim_warmup_steps
        self.ssim_weight_max = ssim_weight_max

        # Focal length
        if self.train_focal:
            self.fx = nn.Parameter(torch.tensor(K[0, 0], dtype=torch.float32, device=device))
            self.fy = nn.Parameter(torch.tensor(K[1, 1], dtype=torch.float32, device=device))
            self.cx = K[0, 2]
            self.cy = K[1, 2]
        else:
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]

        # Radial distortion
        self.k1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device)) if enable_k1 else None

        # Learning rates
        self.lr_positions = LR_POSITIONS
        self.lr_log_scales = LR_LOG_SCALES
        self.lr_opacities = LR_OPACITIES
        self.lr_rotations = LR_ROTATIONS
        self.lr_sh = LR_SH
        self.lr_focal = LR_FOCAL
        self.lr_k1 = LR_K1
        self.use_lr_schedule = use_lr_schedule
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_gamma = lr_decay_gamma

        self._setup_optimizers()

        self.current_step = 0
        self.background = torch.rand(3, dtype=torch.float32, device=device)

        # Density controller
        self.adaptive_density = AdaptiveDensityController(
            self,
            densify_every=densify_every,
            prune_every=prune_every,
            max_gaussians=max_gaussians,
            grad_thresh_base=grad_thresh_base,
            scale_thresh=scale_thresh,
            min_opacity=min_opacity,
        )

        self.rasterizer = rasterizer
        self._update_tanfov()

    def _setup_optimizers(self):
        """Initialize optimizers with current learning rates."""
        self.optimizers = {
            "positions": torch.optim.Adam([self.gaussians.positions], lr=self.lr_positions),
            "log_scales": torch.optim.Adam([self.gaussians.log_scales], lr=self.lr_log_scales),
            "opacities": torch.optim.Adam([self.gaussians.opacities_raw], lr=self.lr_opacities),
            "rotations": torch.optim.Adam([self.gaussians.rotations], lr=self.lr_rotations),
            "sh": torch.optim.Adam([self.gaussians.sh_coeffs], lr=self.lr_sh),
        }
        if self.train_focal:
            self.optimizers["focal"] = torch.optim.Adam([self.fx, self.fy], lr=self.lr_focal)
        if self.enable_k1:
            self.optimizers["k1"] = torch.optim.Adam([self.k1], lr=self.lr_k1)

    def _update_lr(self):
        """Apply exponential decay if enabled."""
        if not self.use_lr_schedule:
            return
        decay = self.lr_decay_gamma ** (self.current_step // self.lr_decay_steps)
        for name, opt in self.optimizers.items():
            base = getattr(self, f"lr_{name}", 1e-4)
            new_lr = base * decay
            for pg in opt.param_groups:
                pg["lr"] = max(new_lr, 1e-7)

    def _update_tanfov(self):
        fx = self.fx.item() if isinstance(self.fx, torch.Tensor) else self.fx
        fy = self.fy.item() if isinstance(self.fy, torch.Tensor) else self.fy
        self.tanfovx = self.image_width / (2.0 * fx)
        self.tanfovy = self.image_height / (2.0 * fy)

    def _build_projection_matrix(self, znear=0.01, zfar=100.0):
        fx = self.fx if not self.train_focal else self.fx.detach().clone()
        fy = self.fy if not self.train_focal else self.fy.detach().clone()
        fx = fx.item() if isinstance(fx, torch.Tensor) else fx
        fy = fy.item() if isinstance(fy, torch.Tensor) else fy
        cx, cy = self.cx, self.cy
        h, w = self.image_height, self.image_width
        P = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        P[0, 0] = 2 * fx / w
        P[1, 1] = 2 * fy / h
        P[0, 2] = 2 * (cx / w) - 1
        P[1, 2] = 2 * (cy / h) - 1
        P[2, 2] = (zfar + znear) / (znear - zfar)
        P[2, 3] = (2 * zfar * znear) / (znear - zfar)
        P[3, 2] = -1
        return P.T

    def _get_camera_center(self, view_matrix):
        R = view_matrix[:3, :3]
        t = view_matrix[:3, 3]
        return -R.T @ t

    def effective_sh_degree(self) -> int:
        """Progressive SH warmup."""
        if self.sh_warmup_steps <= 0:
            return self.sh_degree
        phase = self.current_step // max(1, self.sh_warmup_steps)
        return min(phase, self.sh_degree)

    def current_ssim_weight(self) -> float:
        """Linear warmup for SSIM weight."""
        if self.ssim_warmup_steps <= 0:
            return self.ssim_weight_max
        progress = min(1.0, self.current_step / self.ssim_warmup_steps)
        return self.ssim_weight_max * progress

    def step(self, target_image: Union[np.ndarray, torch.Tensor], camera_pose: Optional[np.ndarray] = None) -> float:
        """Perform one training step on a single frame."""
        if camera_pose is not None:
            self.view_matrix = torch.from_numpy(camera_pose.astype(np.float32)).to(self.device)

        # Random background
        if self.random_background:
            self.background = torch.randint(0, 2, (3,), device=self.device, dtype=torch.float32)

        eff_deg = self.effective_sh_degree()

        # Render
        if self.use_cuda_rasterizer:
            means3D = self.gaussians.positions
            scales = torch.exp(self.gaussians.log_scales).repeat(1, 3)
            rotations = self.gaussians.rotations
            opacities = self.gaussians.opacities

            shs = self.gaussians.sh_coeffs if eff_deg > 0 else None
            colors_precomp = None if eff_deg > 0 else self.gaussians.sh_coeffs[:, 0, :]

            R = self.view_matrix[:3, :3]
            t = self.view_matrix[:3, 3]
            cam_pts = means3D @ R.T + t
            cam_x, cam_y, cam_z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]
            z_clip = torch.clamp(cam_z, min=0.01)
            fx = self.fx if not self.train_focal else self.fx
            fy = self.fy if not self.train_focal else self.fy
            u = fx * cam_x / z_clip + self.cx
            v = fy * cam_y / z_clip + self.cy
            means2D = torch.stack([u, v], dim=-1)

            viewmat = self.view_matrix[:3, :]
            projmat = self._build_projection_matrix()
            cam_center = self._get_camera_center(self.view_matrix)
            self._update_tanfov()

            settings = RasterizationSettings(
                image_height=self.image_height,
                image_width=self.image_width,
                tanfovx=self.tanfovx,
                tanfovy=self.tanfovy,
                bg=self.background,
                scale_modifier=1.0,
                viewmatrix=viewmat,
                projmatrix=projmat,
                sh_degree=eff_deg,
                campos=cam_center,
                prefiltered=False,
                debug=False,
            )
            rasterizer = CUDARasterizer(raster_settings=settings)
            rendered, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                colors_precomp=colors_precomp,
                shs=shs,
            )
        else:
            if self.rasterizer is None:
                raise RuntimeError("No rasterizer available")
            # Use SH0 only
            rendered, _ = self.rasterizer(
                self.gaussians.positions,
                self.gaussians.cov3d,
                self.gaussians.opacities,
                self.gaussians.sh_coeffs,
                self.view_matrix,
                self.K,
                self.background,
            )

        # Target
        if isinstance(target_image, torch.Tensor):
            target = target_image.float()
        else:
            target = torch.from_numpy(target_image).float().to(self.device)

        # Loss
        l1_loss = compute_weighted_l1_loss(rendered, target)
        ssim_loss = compute_ssim_loss(rendered, target)
        w_ssim = self.current_ssim_weight()
        loss = (1.0 - w_ssim) * l1_loss + w_ssim * ssim_loss

        # Backward
        for opt in self.optimizers.values():
            opt.zero_grad()
        loss.backward()

        params_to_clip = [
            self.gaussians.positions, self.gaussians.log_scales,
            self.gaussians.opacities_raw, self.gaussians.rotations,
            self.gaussians.sh_coeffs
        ]
        if self.train_focal:
            params_to_clip.append(self.fx)
            params_to_clip.append(self.fy)
        if self.enable_k1 and self.k1 is not None:
            params_to_clip.append(self.k1)
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_NORM)

        for opt in self.optimizers.values():
            opt.step()

        self.current_step += 1
        self._update_lr()

        if self.adaptive_density is not None:
            self.adaptive_density.step()

        return loss.item()

    def train_epoch(
        self,
        frames_iter,
        camera_poses: List[Optional[np.ndarray]],
        stop_event: Optional[threading.Event] = None,
        progress_callback: Optional[Callable] = None,
        loss_threshold: float = LOSS_THRESHOLD,
        checkpoint_path: Optional[str] = None,
    ) -> float:
        """Train for one epoch over all frames."""
        total_loss = 0.0
        n = len(frames_iter) if hasattr(frames_iter, '__len__') else 0

        for i, frame in enumerate(frames_iter):
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
            total_loss += loss

            if checkpoint_path and self.current_step % CHECKPOINT_INTERVAL_STEPS == 0:
                self.save_training_state(checkpoint_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if loss_threshold and loss > loss_threshold:
                if checkpoint_path:
                    self.save_training_state(checkpoint_path)
                raise LossDivergenceError(f"Loss {loss:.4f} > threshold {loss_threshold} at frame {i+1}")

            if progress_callback:
                progress_callback(i + 1, n if n > 0 else i + 1, loss)

        avg_loss = total_loss / max(i + 1, 1)

        # Density control
        if self.adaptive_density:
            if self.adaptive_density.should_densify():
                stats = self.adaptive_density.densify()
                print(f"  [DENSIFY] split={stats['split']}, duplicate={stats['duplicate']}")
            if self.adaptive_density.should_prune():
                n_pruned = self.adaptive_density.prune(min_opacity=self.adaptive_density.min_opacity)
                if n_pruned > 0:
                    print(f"  [PRUNE] removed {n_pruned} Gaussians")

        if checkpoint_path:
            self.save_training_state(checkpoint_path)

        return avg_loss

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get current Gaussian parameters as numpy arrays."""
        return {
            "positions": self.gaussians.positions.detach().cpu().numpy(),
            "scales": np.exp(self.gaussians.log_scales.detach().cpu().numpy()),
            "opacities": torch.sigmoid(self.gaussians.opacities_raw).detach().cpu().numpy(),
            "rotations": self.gaussians.rotations.detach().cpu().numpy(),
            "sh_coeffs": self.gaussians.sh_coeffs.detach().cpu().numpy(),
        }

    def save_training_state(self, path: str) -> None:
        """Save full training state (strict format)."""
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
            "background": self.background.detach().cpu(),
            "adaptive_density": {
                "step_count": self.adaptive_density._step_count,
                "opacity_accum": self.adaptive_density._opacity_accum.detach().cpu(),
                "grad_accum": self.adaptive_density._grad_accum.detach().cpu(),
                "max_gaussians": self.adaptive_density.max_gaussians,
            },
            "sh_degree": self.sh_degree,
            "train_focal": self.train_focal,
            "enable_k1": self.enable_k1,
            "fx": self.fx.detach().cpu() if isinstance(self.fx, torch.Tensor) else self.fx,
            "fy": self.fy.detach().cpu() if isinstance(self.fy, torch.Tensor) else self.fy,
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

    def load_training_state(self, path: str, device: str = "cpu") -> None:
        """Load training state. Expects all fields to be present."""
        state = torch.load(path, map_location=device)
        device = torch.device(device)

        # Gaussian parameters
        g = self.gaussians
        params = state["gaussian_params"]
        g.positions.data.copy_(params["positions"].to(device))
        g.log_scales.data.copy_(params["log_scales"].to(device))
        g.opacities_raw.data.copy_(params["opacities_raw"].to(device))
        g.rotations.data.copy_(params["rotations"].to(device))
        g.sh_coeffs.data.copy_(params["sh_coeffs"].to(device))

        # Restore optimizer states (assumes exact shape match)
        for name, opt in self.optimizers.items():
            if name in state["optimizer_states"]:
                opt.load_state_dict(state["optimizer_states"][name])

        self.current_step = state["step_count"]
        self.background = state["background"].to(device)

        self.sh_degree = state["sh_degree"]
        self.train_focal = state["train_focal"]
        self.enable_k1 = state["enable_k1"]

        self.fx = nn.Parameter(state["fx"].to(device))
        self.fy = nn.Parameter(state["fy"].to(device))
        if self.enable_k1:
            self.k1 = nn.Parameter(state["k1"].to(device))
        else:
            self.k1 = None

        self.sh_warmup_steps = state["sh_warmup_steps"]
        self.ssim_warmup_steps = state["ssim_warmup_steps"]
        self.ssim_weight_max = state["ssim_weight_max"]

        self.use_lr_schedule = state["use_lr_schedule"]
        self.lr_decay_steps = state["lr_decay_steps"]
        self.lr_decay_gamma = state["lr_decay_gamma"]

        # Density controller
        ad = self.adaptive_density
        ad._step_count = state["adaptive_density"]["step_count"]
        ad._opacity_accum = state["adaptive_density"]["opacity_accum"].to(device)
        ad._grad_accum = state["adaptive_density"]["grad_accum"].to(device)
        ad.max_gaussians = state["adaptive_density"]["max_gaussians"]
        ad.grad_thresh_base = state.get("grad_thresh_base", GRAD_THRESH_BASE)
        ad.scale_thresh = state.get("scale_thresh", SCALE_THRESH)
        ad.min_opacity = state.get("min_opacity", MIN_OPACITY)
        ad.densify_every = state.get("densify_every", DENSIFY_EVERY)
        ad.prune_every = state.get("prune_every", PRUNE_EVERY)

        # Rebuild optimizers if shapes mismatch (density control may have changed)
        # But we've already loaded states, so we trust they match.
        # If they don't, we'll let the next density control step rebuild them.
        self._update_tanfov()


# ---------- Adaptive Density Controller ----------
class AdaptiveDensityController:
    def __init__(
        self,
        trainer: Trainer,
        densify_every: int = DENSIFY_EVERY,
        prune_every: int = PRUNE_EVERY,
        max_gaussians: int = MAX_GAUSSIANS,
        grad_thresh_base: float = GRAD_THRESH_BASE,
        scale_thresh: float = SCALE_THRESH,
        min_opacity: float = MIN_OPACITY,
    ):
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

    def step(self) -> None:
        self._step_count += 1
        g = self.trainer.gaussians
        n = g.num_gaussians
        if n == 0:
            return

        if self._opacity_accum is None:
            self._opacity_accum = torch.zeros(n, device=g.positions.device)
            self._grad_accum = torch.zeros(n, device=g.positions.device)
        elif self._opacity_accum.shape[0] != n:
            # Resize accumulators
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

    def should_densify(self) -> bool:
        return self._step_count > 0 and self._step_count % self.densify_every == 0

    def should_prune(self) -> bool:
        return self._step_count > 0 and self._step_count % self.prune_every == 0

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

        # Adaptive gradient threshold
        avg_grad = self._grad_accum / max(1, self._step_count)
        median_grad = torch.median(avg_grad[avg_grad > 0]) if (avg_grad > 0).any() else torch.tensor(1.0, device=g.positions.device)
        grad_thresh = max(self.grad_thresh_base, median_grad * 0.5)
        avg_opacity = self._opacity_accum / max(1, self._step_count)

        split_mask = (avg_grad > grad_thresh) & (g.log_scales.squeeze(-1) > self.scale_thresh) & (avg_opacity > 0.01)
        duplicate_mask = (avg_grad > grad_thresh) & ~split_mask & (avg_opacity > 0.01)

        n_split = int(split_mask.sum())
        n_dup = int(duplicate_mask.sum())

        if n_split > 0:
            base_pos = g.positions[split_mask].detach().cpu().numpy()
            base_log_scales = g.log_scales[split_mask].detach().cpu().numpy()
            jitter = np.random.randn(n_split, 3).astype(np.float32) * 0.001
            new_pos = base_pos + jitter
            new_log_scales = np.log(np.exp(base_log_scales) * 0.8)
            new_opa = np.full((n_split,), np.log(0.1), dtype=np.float32)
            new_rot = np.tile([1.0, 0.0, 0.0, 0.0], (n_split, 1)).astype(np.float32)
            new_sh = g.sh_coeffs[split_mask].detach().cpu().numpy()
            self._replace_gaussians(~split_mask, new_pos, new_log_scales, new_opa, new_rot, new_sh)
            stats["split"] = n_split

        if n_dup > 0:
            dup_pos = g.positions[duplicate_mask].detach().cpu().numpy()
            dup_log_scales = g.log_scales[duplicate_mask].detach().cpu().numpy()
            dup_opa = g.opacities_raw[duplicate_mask].detach().cpu().numpy()
            dup_rot = g.rotations[duplicate_mask].detach().cpu().numpy()
            dup_sh = g.sh_coeffs[duplicate_mask].detach().cpu().numpy()
            self._replace_gaussians(~duplicate_mask, dup_pos, dup_log_scales, dup_opa, dup_rot, dup_sh)
            stats["duplicate"] = n_dup

        # Budget control
        n_current = g.num_gaussians
        if n_current > self.max_gaussians:
            n_remove = min(n_current - self.max_gaussians, int(n_current * 0.15))
            if n_remove > 0:
                self.prune(min_opacity=self.min_opacity, target_remove=n_remove)

        self.reset_accumulators()
        return stats

    def prune(self, min_opacity: float = MIN_OPACITY, target_remove: Optional[int] = None) -> int:
        g = self.trainer.gaussians
        n = g.num_gaussians
        if n == 0:
            return 0

        avg_opacity = self._opacity_accum / max(1, self._step_count) if self._opacity_accum is not None else g.opacities

        if target_remove is not None and target_remove < n:
            vals, indices = torch.topk(avg_opacity, k=target_remove, largest=False)
            prune_mask = torch.zeros(n, dtype=torch.bool, device=g.positions.device)
            prune_mask[indices] = True
        else:
            prune_mask = avg_opacity < min_opacity
            # Protect high‑gradient Gaussians
            if self._grad_accum is not None and n > 0:
                avg_grad = self._grad_accum / max(1, self._step_count)
                protect_mask = avg_grad > 0.005
                prune_mask = prune_mask & ~protect_mask

        n_pruned = int(prune_mask.sum())
        if n_pruned == 0:
            return 0

        keep = ~prune_mask
        self._replace_gaussians(
            keep,
            g.positions[keep].detach().cpu().numpy(),
            g.log_scales[keep].detach().cpu().numpy(),
            g.opacities_raw[keep].detach().cpu().numpy(),
            g.rotations[keep].detach().cpu().numpy(),
            g.sh_coeffs[keep].detach().cpu().numpy(),
        )
        return n_pruned

    def _replace_gaussians(
        self,
        keep_mask: torch.Tensor,
        positions: np.ndarray,
        log_scales: np.ndarray,
        opacities_raw: np.ndarray,
        rotations: np.ndarray,
        sh_coeffs: np.ndarray,
    ) -> None:
        """Replace Gaussians and rebuild optimizers (discard old optimizer states)."""
        g = self.trainer.gaussians
        device = g.positions.device

        # Concatenate kept + new
        g.positions = torch.cat([g.positions[keep_mask], torch.from_numpy(positions).float().to(device)], dim=0)
        g.log_scales = torch.cat([g.log_scales[keep_mask], torch.from_numpy(log_scales).float().to(device)], dim=0)
        g.opacities_raw = torch.cat([g.opacities_raw[keep_mask], torch.from_numpy(opacities_raw).float().to(device)])
        g.rotations = torch.cat([g.rotations[keep_mask], torch.from_numpy(rotations).float().to(device)], dim=0)
        g.sh_coeffs = torch.cat([g.sh_coeffs[keep_mask], torch.from_numpy(sh_coeffs).float().to(device)], dim=0)

        # Set requires_grad
        g.positions.requires_grad_(True)
        g.log_scales.requires_grad_(True)
        g.opacities_raw.requires_grad_(True)
        g.rotations.requires_grad_(True)
        g.sh_coeffs.requires_grad_(True)

        # Rebuild optimizers (discard old momentum)
        self.trainer._setup_optimizers()

        # Reset accumulators to new size
        n = g.num_gaussians
        self._opacity_accum = torch.zeros(n, device=device)
        self._grad_accum = torch.zeros(n, device=device)
        self._step_count = 0