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


# ---------- Frame loader ----------
def _load_frame_from_path(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class LazyFrames:
    def __init__(self, sources: List[Union[str, np.ndarray]]):
        self._sources = sources
    def __len__(self): return len(self._sources)
    def __iter__(self):
        for src in self._sources:
            yield self._load(src)
    def __getitem__(self, idx):
        n = len(self._sources)
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(n))]
        if idx < 0: idx += n
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
    return color


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

    def forward(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree=3):
        return self._render_batch(positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree)

    # ---------- 向量化光栅化（方向 A：排序式逐像素 splat） ----------
    def _render_batch(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree):
        """主实现：排序式逐像素 splat，详见 _render_batch_vectorized 的说明。"""
        return self._render_batch_vectorized(positions, cov3d, opacities, sh_coeffs,
                                             view_matrix, K, background, sh_degree)

    # ---------- 向量化光栅化（方向 A：排序式逐像素 splat） ----------
    def _render_batch_vectorized(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree):
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
        dirs = F.normalize(cam_positions[valid][order], dim=-1)
        colors = eval_sh(sh_degree, sh_coeffs[valid][order], dirs)

        # Ensure gradient connection even when all Gaussians land off-screen
        _graph_link = (positions.sum() if positions.numel() > 0 else opacities.sum()) * 0.0
        out_color = _graph_link.view(1,1,1).expand(H, W, 3).contiguous()
        out_alpha = _graph_link.view(1,1).expand(H, W).contiguous()

        # ---- 逐像素向量化合成（替代逐高斯 Python 循环） ----
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

        h_sizes = (y_max_b - y_min_b).int(); w_sizes = (x_max_b - x_min_b).int()
        max_h = int(h_sizes.max()); max_w = int(w_sizes.max())

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

        # 展平覆盖像素对：只保留 valid_mask 为真的格点
        # flat indices: (gauss_idx, y_in_tile, x_in_tile) -> pixel (gauss_idx, y, x)
        mask = valid_mask  # [B, max_h, max_w]
        flat_alpha = alpha[mask]
        n_pairs = flat_alpha.shape[0]
        if n_pairs == 0:
            return out_color + background.view(1,1,3) * (1.0 - out_alpha.unsqueeze(-1)), out_alpha

        # 重建每个覆盖格的全局像素坐标（y, x）
        gauss_idx_3d = torch.arange(batch_n, device=colors.device).view(-1, 1, 1).expand_as(mask)
        # 广播 y/x 网格到与 mask 相同的 [B, max_h, max_w] 形状
        y_3d = gy_g.expand_as(mask)
        x_3d = gx_g.expand_as(mask)
        gauss_ids = gauss_idx_3d[mask].long()
        y_coord = y_3d[mask].long()
        x_coord = x_3d[mask].long()
        pix = y_coord * W + x_coord  # 展平像素索引 [n_pairs]

        flat_color = col_b[gauss_ids]  # [n_pairs, 3]

        # 按像素 stable sort（保持深度序：gauss_ids 已按深度排序）
        pix_sorted, sort_idx = torch.sort(pix, stable=True)
        a_sorted = flat_alpha[sort_idx]
        c_sorted = flat_color[sort_idx]

        # 分段透射率：T_before(p) = exp(Σ_{j<i} log(1-α_j))，按像素分组
        log_ta = torch.log1p(-a_sorted)
        log_cum = torch.cumsum(log_ta, dim=0)

        # 段内 exclusive 前缀：log_T_before[i] = log_cum[i-1]（前一个元素的累计），段首为 0
        # torch.cumsum 给出含自身的累计，段内前移一位即得 exclusive 前缀
        log_cum_shift = torch.cat([torch.zeros(1, dtype=log_cum.dtype, device=log_cum.device), log_cum[:-1]])
        # 像素组边界：新组起始处重置为 0
        new_group = pix_sorted[1:] != pix_sorted[:-1]
        group_starts = torch.cat([torch.tensor([True], device=pix_sorted.device), new_group])
        # 组内 exclusive 前缀：
        #   log_T_before[j] = log_cum_shift[j] - offset[j]
        #   其中 offset[j] = 该元素所属组"组首元素前一个位置"的累计
        #                  = log_cum_shift[组首位置]（组首自身 offset 使 log_T=0，自洽）
        # 组首位置用 cummax 前向传播（不能用 cumsum-1，那给出的是组 id 而非组首索引）
        arange = torch.arange(group_starts.shape[0], device=pix_sorted.device)
        group_start_pos = torch.where(group_starts, arange, torch.zeros_like(arange))
        group_start_pos = torch.cummax(group_start_pos, dim=0).values
        seg_offset = log_cum_shift[group_start_pos]
        log_T_before = log_cum_shift - seg_offset

        T_before = torch.exp(log_T_before.clamp(min=-50.0))
        weight = a_sorted * T_before  # α_i * T_before(p)

        # 归约：color + alpha
        HpW = H * W
        acc_color = torch.zeros(HpW, 3, dtype=colors.dtype, device=colors.device)
        acc_color = acc_color.index_add_(0, pix_sorted, weight.unsqueeze(-1) * c_sorted)
        acc_log = torch.zeros(HpW, dtype=colors.dtype, device=colors.device)
        acc_log = acc_log.index_add_(0, pix_sorted, log_ta)

        out_color = out_color.view(HpW, 3) + acc_color
        out_alpha = out_alpha.view(HpW) + (1.0 - torch.exp(acc_log.clamp(min=-50.0)))
        out_color = out_color.view(H, W, 3)
        out_alpha = out_alpha.view(H, W)

        out_color = out_color + background.view(1,1,3) * (1.0 - out_alpha.unsqueeze(-1))
        return out_color, out_alpha

    # ---------- 方向 B 实现（保留：批量同步 + 逐高斯串行合成） ----------
    def _render_batch_batched(self, positions, cov3d, opacities, sh_coeffs, view_matrix, K, background, sh_degree):
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
        dirs = F.normalize(cam_positions[valid][order], dim=-1)
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
                 enable_k1: bool = False, use_lr_schedule: bool = USE_LR_SCHEDULE,
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
        self._setup_optimizers()
        self.current_step = 0
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
        if self.gaussians.num_gaussians < 2000:
            densify_initial_gaussians(self.gaussians, expansion_factor=8, noise_scale=0.02)
            print(f"  [INIT] Densified to {self.gaussians.num_gaussians} Gaussians")

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
            self.background = torch.randint(0, 2, (3,), device=self.device, dtype=torch.float32)
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
        cov3d = self.gaussians.cov3d

        eff_deg = self.effective_sh_degree()

        # Use PyTorch rasterizer
        viewmat = self.view_matrix
        K = torch.zeros(3, 3, dtype=torch.float32, device=self.device)
        K[0,0] = self.fx.item() if isinstance(self.fx, torch.Tensor) else self.fx
        K[1,1] = self.fy.item() if isinstance(self.fy, torch.Tensor) else self.fy
        K[0,2] = self.cx
        K[1,2] = self.cy
        K[2,2] = 1.0

        rendered, _ = self.rasterizer(
            means3D, cov3d, opacities, sh_coeffs, viewmat, K, self.background, sh_degree=eff_deg
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

        params_to_clip = [
            self.gaussians.positions, self.gaussians.log_scales,
            self.gaussians.opacities_raw, self.gaussians.rotations,
            self.gaussians.sh_coeffs
        ]
        if self.train_focal and isinstance(self.fx, nn.Parameter):
            params_to_clip.append(self.fx); params_to_clip.append(self.fy)
        if self.enable_k1 and self.k1 is not None and isinstance(self.k1, nn.Parameter):
            params_to_clip.append(self.k1)
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
            total_loss += loss
            processed_count += 1

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

        avg_loss = total_loss / max(processed_count, 1)
        if checkpoint_path:
            self.save_training_state(checkpoint_path)
        return avg_loss

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {
            "positions": self.gaussians.positions.detach().cpu().numpy(),
            "scales": np.exp(self.gaussians.log_scales.detach().cpu().numpy()),
            "opacities": torch.sigmoid(self.gaussians.opacities_raw).detach().cpu().numpy(),
            "rotations": self.gaussians.rotations.detach().cpu().numpy(),
            "sh_coeffs": self.gaussians.sh_coeffs.detach().cpu().numpy(),
        }

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
            "best_loss": self.best_loss,  # ===== 新增 =====
            "background": self.background.detach().cpu(),
            "adaptive_density": {
                "step_count": self.adaptive_density._step_count,
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

    # ===== 修复: load_training_state 增加 best_loss 恢复 =====
    def load_training_state(self, path: str, device: str = "cpu") -> None:
        state = torch.load(path, map_location=device, weights_only=False)
        device = torch.device(device)
        g = self.gaussians
        params = state["gaussian_params"]
        g.positions.data.copy_(params["positions"].to(device))
        g.log_scales.data.copy_(params["log_scales"].to(device))
        g.opacities_raw.data.copy_(params["opacities_raw"].to(device))
        g.rotations.data.copy_(params["rotations"].to(device))
        g.sh_coeffs.data.copy_(params["sh_coeffs"].to(device))

        self.current_step = state["step_count"]
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

        avg_grad = self._grad_accum / max(1, self._step_count)
        avg_opacity = self._opacity_accum / max(1, self._step_count)
        median_grad = torch.median(avg_grad[avg_grad > 0]) if (avg_grad > 0).any() else torch.tensor(1.0, device=g.positions.device)
        grad_thresh = max(self.grad_thresh_base, median_grad * 0.5)
        max_log_scale = torch.max(g.log_scales, dim=1).values
        split_mask = (avg_grad > grad_thresh) & (max_log_scale > self.scale_thresh) & (avg_opacity > 0.01)
        duplicate_mask = (avg_grad > grad_thresh) & ~split_mask & (avg_opacity > 0.01)
        keep_mask = ~(split_mask | duplicate_mask)
        keep_idx = torch.where(keep_mask)[0]
        split_idx = torch.where(split_mask)[0]
        dup_idx = torch.where(duplicate_mask)[0]

        new_pos = []; new_log_scales = []; new_opa = []; new_rot = []; new_sh = []

        if len(keep_idx) > 0:
            new_pos.append(g.positions[keep_idx].detach().cpu().numpy())
            new_log_scales.append(g.log_scales[keep_idx].detach().cpu().numpy())
            new_opa.append(g.opacities_raw[keep_idx].detach().cpu().numpy())
            new_rot.append(g.rotations[keep_idx].detach().cpu().numpy())
            new_sh.append(g.sh_coeffs[keep_idx].detach().cpu().numpy())

        if len(split_idx) > 0:
            base_pos = g.positions[split_idx].detach().cpu().numpy()
            base_log_scales = g.log_scales[split_idx].detach().cpu().numpy()
            base_opa = g.opacities_raw[split_idx].detach().cpu().numpy()
            base_rot = g.rotations[split_idx].detach().cpu().numpy()
            base_sh = g.sh_coeffs[split_idx].detach().cpu().numpy()
            for scale_factor in [0.8, 0.6]:
                jitter = np.random.randn(len(split_idx), 3).astype(np.float32) * 0.001
                new_pos.append(base_pos + jitter * (1 if scale_factor == 0.8 else -0.5))
                new_log_scales.append(base_log_scales + np.log(scale_factor))
                new_opa.append(base_opa + np.random.normal(0, 0.1, len(split_idx)).astype(np.float32))
                new_rot.append(base_rot + np.random.normal(0, 0.01, (len(split_idx), 4)).astype(np.float32))
                new_sh.append(base_sh + np.random.normal(0, 0.01, base_sh.shape).astype(np.float32))
            stats["split"] = len(split_idx) * 2

        if len(dup_idx) > 0:
            dup_pos = g.positions[dup_idx].detach().cpu().numpy()
            dup_log_scales = g.log_scales[dup_idx].detach().cpu().numpy()
            dup_opa = g.opacities_raw[dup_idx].detach().cpu().numpy()
            dup_rot = g.rotations[dup_idx].detach().cpu().numpy()
            dup_sh = g.sh_coeffs[dup_idx].detach().cpu().numpy()
            new_pos.append(dup_pos + np.random.randn(len(dup_idx), 3).astype(np.float32) * 0.001)
            new_log_scales.append(dup_log_scales)
            new_opa.append(dup_opa + np.random.normal(0, 0.1, len(dup_idx)).astype(np.float32))
            new_rot.append(dup_rot + np.random.normal(0, 0.01, (len(dup_idx), 4)).astype(np.float32))
            new_sh.append(dup_sh + np.random.normal(0, 0.01, dup_sh.shape).astype(np.float32))
            stats["duplicate"] = len(dup_idx)

        # ===== 修复: 防止 new_pos 为空导致 np.concatenate 报错 =====
        if not new_pos:
            return stats

        new_pos = np.concatenate(new_pos, axis=0)
        new_log_scales = np.concatenate(new_log_scales, axis=0)
        new_opa = np.concatenate(new_opa, axis=0)
        new_rot = np.concatenate(new_rot, axis=0)
        new_sh = np.concatenate(new_sh, axis=0)

        device = g.positions.device
        g.positions = torch.from_numpy(new_pos).float().to(device)
        g.log_scales = torch.from_numpy(new_log_scales).float().to(device)
        g.opacities_raw = torch.from_numpy(new_opa).float().to(device)
        g.rotations = torch.from_numpy(new_rot).float().to(device)
        g.sh_coeffs = torch.from_numpy(new_sh).float().to(device)
        for param in [g.positions, g.log_scales, g.opacities_raw, g.rotations, g.sh_coeffs]:
            param.requires_grad_(True)

        self.trainer._setup_optimizers()
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
                protect_mask = avg_grad > 0.005
                prune_mask = prune_mask & ~protect_mask

        n_pruned = int(prune_mask.sum())
        if n_pruned == 0:
            return 0

        keep_mask = ~prune_mask
        keep_idx = torch.where(keep_mask)[0]
        device = g.positions.device
        g.positions = g.positions[keep_idx].detach().clone()
        g.log_scales = g.log_scales[keep_idx].detach().clone()
        g.opacities_raw = g.opacities_raw[keep_idx].detach().clone()
        g.rotations = g.rotations[keep_idx].detach().clone()
        g.sh_coeffs = g.sh_coeffs[keep_idx].detach().clone()
        for param in [g.positions, g.log_scales, g.opacities_raw, g.rotations, g.sh_coeffs]:
            param.requires_grad_(True)

        self.trainer._setup_optimizers()
        self.reset_accumulators()
        return n_pruned