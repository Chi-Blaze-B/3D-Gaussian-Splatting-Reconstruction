"""
PLY export utilities for 3D Gaussian Splatting.

Writes Gaussian parameters to a .ply file compatible with the official
3DGS viewer (https://github.com/graphdeco-inria/gaussian-splatting).

Updated to support SH degree 3 (16 bases).
"""

import numpy as np


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
    """Write Gaussian Splatting PLY file.

    Parameters
    ----------
    output_path : str — destination .ply file path.
    positions : (N, 3) float32 — XYZ coordinates.
    scales : (N,), (N, 1) or (N, 3) float32 — scale factors.
    opacities : (N,) float32 — opacity values.
    rotations : (N, 4) float32 — quaternions (w, x, y, z).
    sh_coeffs : (N, num_bases, 3) or (N, 3) float32 — SH color coefficients.
    sh_degree : int — maximum spherical harmonic degree used.
    """
    N = positions.shape[0]

    # --- Normalize SH coefficients shape ---
    if sh_coeffs.ndim == 2:
        sh_coeffs = sh_coeffs[:, np.newaxis, :]  # (N, 1, 3)

    # Ensure correct number of bases for requested degree
    target_bases = (sh_degree + 1) ** 2
    current_bases = sh_coeffs.shape[1]
    if current_bases < target_bases:
        pad = np.zeros((N, target_bases - current_bases, 3), dtype=sh_coeffs.dtype)
        sh_coeffs = np.concatenate([sh_coeffs, pad], axis=1)
    elif current_bases > target_bases:
        sh_coeffs = sh_coeffs[:, :target_bases, :]

    num_bases = sh_coeffs.shape[1]

    # --- Normalize scales ---
    if scales.ndim == 1:
        scales = np.stack([scales, scales, scales], axis=1)
    elif scales.shape[1] == 1:
        scales = np.hstack([scales, scales, scales])

    # --- Build PLY header ---
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",   # quaternion x
        "property float ny",   # quaternion y
        "property float nz",   # quaternion z
        "property float nw",   # quaternion w
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float opacity",
    ]

    for b in range(num_bases):
        header_lines.append(f"property float f_dc_{b}_r")
        header_lines.append(f"property float f_dc_{b}_g")
        header_lines.append(f"property float f_dc_{b}_b")

    header_lines.append("end_header")

    with open(output_path, "wb") as f:
        header_text = "\n".join(header_lines) + "\n"
        f.write(header_text.encode("ascii"))

        # --- Define structured data type ---
        dtype_fields = [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("nw", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("opacity", "<f4"),
        ]
        for b in range(num_bases):
            dtype_fields.append((f"f_dc_{b}_r", "<f4"))
            dtype_fields.append((f"f_dc_{b}_g", "<f4"))
            dtype_fields.append((f"f_dc_{b}_b", "<f4"))

        data = np.zeros(N, dtype=dtype_fields)

        # Positions
        data["x"] = positions[:, 0].astype(np.float32)
        data["y"] = positions[:, 1].astype(np.float32)
        data["z"] = positions[:, 2].astype(np.float32)

        # Quaternion: (w,x,y,z) → (x,y,z,w)
        data["nx"] = rotations[:, 1].astype(np.float32)
        data["ny"] = rotations[:, 2].astype(np.float32)
        data["nz"] = rotations[:, 3].astype(np.float32)
        data["nw"] = rotations[:, 0].astype(np.float32)

        # Scales
        data["scale_0"] = scales[:, 0].astype(np.float32)
        data["scale_1"] = scales[:, 1].astype(np.float32)
        data["scale_2"] = scales[:, 2].astype(np.float32)

        # Opacity
        data["opacity"] = opacities.astype(np.float32)

        # SH coefficients — write the actual coefficients for all bases
        for b in range(num_bases):
            data[f"f_dc_{b}_r"] = sh_coeffs[:, b, 0].astype(np.float32)
            data[f"f_dc_{b}_g"] = sh_coeffs[:, b, 1].astype(np.float32)
            data[f"f_dc_{b}_b"] = sh_coeffs[:, b, 2].astype(np.float32)

        f.write(data.tobytes())

    print(f"Wrote {N} Gaussians to {output_path}")


def export_training_checkpoint(
    trainer,
    output_path: str,
    sh_degree: int = None,
) -> None:
    """Export the current training state as a PLY file.

    sh_degree: if None, uses trainer.sh_degree (default 3).
    """
    if sh_degree is None:
        sh_degree = getattr(trainer, 'sh_degree', 3)
    params = trainer.get_parameters()
    write_ply(
        output_path,
        positions=params["positions"],
        scales=params["scales"],
        opacities=params["opacities"],
        rotations=params["rotations"],
        sh_coeffs=params["sh_coeffs"],
        sh_degree=sh_degree,
    )