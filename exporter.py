"""
3D Gaussian Splatting 的 PLY 导出工具。

将高斯参数写入与官方 3DGS 查看器兼容的 .ply 文件
（https://github.com/graphdeco-inria/gaussian-splatting）。

字段名与数值约定完全对齐官方 save_ply：
  x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3
  nx/ny/nz = 0（法线未用）；scale 存 log σ；opacity 存 logit（未过 sigmoid）；
  SH 存原始系数（通道 0 已是 (RGB-0.5)/C0，求值方补 +0.5）；rot 存 (w,x,y,z)。
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def write_ply(
    output_path: str,
    positions: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    rotations: np.ndarray,
    sh_coeffs: np.ndarray,
    *,
    sh_degree: int = 0,
) -> None:
    """按官方 3DGS 约定写出高斯 PLY 文件。

    参数
    ----------
    output_path : str — 目标 .ply 文件路径。
    positions : (N, 3) float32 — XYZ 坐标。
    scales : (N,), (N, 1) 或 (N, 3) float32 — 原始 log σ（不要传 exp 后的线性尺度）。
    opacities : (N,) float32 — 原始 logit（不要传 sigmoid 后的概率）。
    rotations : (N, 4) float32 — 四元数 (w, x, y, z)。
    sh_coeffs : (N, num_bases, 3) float32 — 原始 SH 系数（通道 0 已按 (RGB-0.5)/C0）。
    sh_degree : int — 最大 SH 阶数；决定写入多少 f_rest 字段（((deg+1)²-1)×3）。
    """
    N = positions.shape[0]

    # 归一化 SH 系数形状
    if sh_coeffs.ndim == 2:
        sh_coeffs = sh_coeffs[:, np.newaxis, :]  # (N, 1, 3)

    # 按 sh_degree 截断/补齐，保证声明阶数与实际字段一致（--sh-degree 0 → 无 f_rest 字段）
    target_bases = (sh_degree + 1) ** 2
    current_bases = sh_coeffs.shape[1]
    if current_bases < target_bases:
        pad = np.zeros((N, target_bases - current_bases, 3), dtype=sh_coeffs.dtype)
        sh_coeffs = np.concatenate([sh_coeffs, pad], axis=1)
    elif current_bases > target_bases:
        sh_coeffs = sh_coeffs[:, :target_bases, :]

    # 归一化尺度（原始 log σ，[N,3]）
    if scales.ndim == 1:
        scales = np.stack([scales, scales, scales], axis=1)
    elif scales.shape[1] == 1:
        scales = np.hstack([scales, scales, scales])

    # f_rest 通道主序：rest [N, target_bases-1, 3] → [N,3,rest] → 扁平 45（SH3）
    # f_rest_{c*n+b} = sh_coeffs[:, 1+b, c]
    rest = sh_coeffs[:, 1:, :]                      # [N, n_rest_bases, 3]
    rest_cm = rest.transpose(0, 2, 1)               # [N, 3, n_rest_bases]
    n_rest = rest_cm.shape[1] * rest_cm.shape[2]
    rest_flat = rest_cm.reshape(N, n_rest)

    # PLY 头
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(n_rest):
        header_lines.append(f"property float f_rest_{i}")
    header_lines += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]

    dtype_fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
    ]
    dtype_fields += [(f"f_rest_{i}", "<f4") for i in range(n_rest)]
    dtype_fields += [
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]

    data = np.zeros(N, dtype=dtype_fields)

    # 位置
    data["x"] = positions[:, 0].astype(np.float32)
    data["y"] = positions[:, 1].astype(np.float32)
    data["z"] = positions[:, 2].astype(np.float32)
    # nx/ny/nz 保持 0（官方格式法线未用）

    # SH DC（通道 0，已是 (RGB-0.5)/C0）
    data["f_dc_0"] = sh_coeffs[:, 0, 0].astype(np.float32)
    data["f_dc_1"] = sh_coeffs[:, 0, 1].astype(np.float32)
    data["f_dc_2"] = sh_coeffs[:, 0, 2].astype(np.float32)
    # SH rest（通道主序）
    for i in range(n_rest):
        data[f"f_rest_{i}"] = rest_flat[:, i].astype(np.float32)

    # 透明度（原始 logit）
    data["opacity"] = opacities.astype(np.float32)
    # 尺度（原始 log σ）
    data["scale_0"] = scales[:, 0].astype(np.float32)
    data["scale_1"] = scales[:, 1].astype(np.float32)
    data["scale_2"] = scales[:, 2].astype(np.float32)
    # 旋转（w,x,y,z → 官方 rot_0..3 = w,x,y,z）
    data["rot_0"] = rotations[:, 0].astype(np.float32)
    data["rot_1"] = rotations[:, 1].astype(np.float32)
    data["rot_2"] = rotations[:, 2].astype(np.float32)
    data["rot_3"] = rotations[:, 3].astype(np.float32)

    with open(output_path, "wb") as f:
        header_text = "\n".join(header_lines) + "\n"
        f.write(header_text.encode("ascii"))
        f.write(data.tobytes())

    logger.info("已写出 %d 个高斯到 %s", N, output_path)


def export_training_checkpoint(
    trainer,
    output_path: str,
    sh_degree: Optional[int] = None,
) -> None:
    """把当前训练状态导出为 PLY。

    sh_degree 为 None 时使用 trainer.sh_degree（默认 3）。
    """
    if sh_degree is None:
        sh_degree = getattr(trainer, "sh_degree", 3)
    p = trainer.gaussians.export_ply_dict()
    write_ply(
        output_path,
        positions=p["positions"],
        scales=p["scales"],
        opacities=p["opacities"],
        rotations=p["rotations"],
        sh_coeffs=p["sh_coeffs"],
        sh_degree=sh_degree,
    )