"""
COLMAP-based pose estimation for 3D Gaussian Splatting (optimized).

Wraps COLMAP CLI to run feature extraction, matching, and mapping,
then parses the results into the same format as the ORB+EM pipeline
(CameraPose list + sparse_points).

Changes:
- Use symbolic links (or copy as fallback) to avoid duplicating frames.
- Robust images.txt parsing (line-pair logic).
- Expose SIFT/max_image_size parameters.
- Graceful temporary dir cleanup.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Local imports
try:
    from poses import CameraIntrinsics, CameraPose
except ImportError:
    CameraIntrinsics = None
    CameraPose = None


def _run_colmap(cmd: List[str], label: str, colmap_bin: str) -> subprocess.CompletedProcess:
    """Run a COLMAP command with proper Qt plugin path."""
    print(f"  [COLMAP] {label}...")
    env = dict(os.environ)

    # Clear Qt-related env vars set by parent process (PyQt5, etc.)
    for key in list(env.keys()):
        if key.startswith("QT_"):
            del env[key]

    # Prepend COLMAP bin so it finds its own DLLs first
    existing = env.get("PATH", "")
    env["PATH"] = colmap_bin + os.pathsep + existing

    # Set Qt plugin paths explicitly for COLMAP
    plugins_dir = os.path.join(colmap_bin, "..", "plugins")
    if os.path.isdir(plugins_dir):
        env["QT_PLUGIN_PATH"] = plugins_dir
        platforms_dir = os.path.join(plugins_dir, "platforms")
        if os.path.isdir(platforms_dir):
            env["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms_dir

    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=env, creationflags=creationflags)
    if result.returncode != 0:
        err = result.stderr.strip() if result.stderr else ""
        msg = f"COLMAP {label} failed with code {result.returncode}"
        if err:
            msg += f": {err[:300]}"
        raise RuntimeError(msg)
    return result


def _create_symlink_or_copy(src: str, dst: str):
    """Create symbolic link to src at dst; fallback to copy if symlink fails."""
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def estimate_poses_with_colmap(
    frame_paths: List[str],
    output_dir: str,
    *,
    colmap_exe: Optional[str] = None,
    image_path: Optional[str] = None,
    max_image_size: int = 2400,
    sift_peak_threshold: float = 0.005,
    sift_edge_threshold: int = 10,
    sift_max_num_features: int = 12000,
    matcher: str = "exhaustive",
    matcher_overlap: int = 10,
    loop_detection: bool = False,
) -> Tuple[CameraIntrinsics, List[Optional[CameraPose]], np.ndarray]:
    """Estimate camera poses using COLMAP.

    Args:
        frame_paths: list of absolute paths to input frames.
        output_dir: directory to store intermediate COLMAP results.
        colmap_exe: path to colmap executable (auto-detected if None).
        max_image_size: max image dimension for feature extraction.
            Default 2400 (from tuned optimum): high resolution retains small-scale
            texture → more/better SIFT matches → denser triangulated point cloud.
        sift_peak_threshold: SIFT peak threshold.
        sift_edge_threshold: SIFT edge threshold.
        sift_max_num_features: max number of SIFT features per image.
            Default 12000 (from tuned optimum): more features → better matching
            quality.  Note: raises runtime on long sequences, use
            --pose-estimator opencv or reduce features for very long videos.
        matcher: 'exhaustive' (match all pairs, robust for unordered image sets and
            for unevenly-spaced two-stage frames; O(n²) pairs) or 'sequential'
            (match only temporally adjacent frames, fastest for video).
            Default 'exhaustive' (matches the tuned optimum).
        matcher_overlap: sequential matcher overlap window (frames to either side).
        loop_detection: enable sequential matcher loop closure detection
            (matches far-away frames that revisit the same scene; requires a
            vocabulary tree — leave off otherwise, sequential_matcher hangs).

    Returns:
        intrinsics: CameraIntrinsics object.
        poses: list of CameraPose (or None for unregistered frames),
               length equals len(frame_paths), order matches frame_paths.
        sparse_points: (N,3) numpy array of 3D points.
    """
    if CameraIntrinsics is None or CameraPose is None:
        raise ImportError("CameraIntrinsics/CameraPose not found. Install poses module.")

    script_dir = str(Path(__file__).resolve().parent)

    # ------------------------------------------------------------------
    # Locate COLMAP executable
    # ------------------------------------------------------------------
    colmap_exe_path = colmap_exe
    if colmap_exe_path is None:
        bundled_bin = os.path.join(script_dir, "colmap-x64-windows-nocuda", "bin")
        bundled_exe = os.path.join(bundled_bin, "colmap.exe")
        if os.path.isfile(bundled_exe):
            colmap_exe_path = bundled_exe
            print(f"  [COLMAP] Using bundled COLMAP from {bundled_bin}")
        else:
            colmap_exe_path = shutil.which("colmap")

    if colmap_exe_path is None:
        raise RuntimeError(
            "COLMAP not found. Install it via conda:\n"
            "  conda install -c conda-forge colmap\n"
            "Or provide colmap_exe='path/to/colmap'"
        )
    colmap_bin_dir = os.path.dirname(colmap_exe_path)

    # ------------------------------------------------------------------
    # Setup directories
    # ------------------------------------------------------------------
    workdir = Path(output_dir) / "colmap_work"
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = str(workdir / "database.db")
    sparse_dir = str(workdir / "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Prepare sorted images directory using symlinks (or copies)
    # This keeps disk usage minimal and ensures predictable order.
    # ------------------------------------------------------------------
    tmp_img_dir = str(workdir / "sorted_images")
    if os.path.isdir(tmp_img_dir):
        shutil.rmtree(tmp_img_dir, ignore_errors=True)
    os.makedirs(tmp_img_dir, exist_ok=True)

    # Map original filename -> index in frame_paths
    name_to_frame_idx: Dict[str, int] = {}
    for idx, p in enumerate(frame_paths):
        name_to_frame_idx[Path(p).name] = idx

    # Sort by filename to guarantee mapping: 0000.png, 0001.png, ...
    sorted_names = sorted(Path(p).name for p in frame_paths)
    src_dir = Path(frame_paths[0]).parent  # assume all frames in same dir
    for idx, name in enumerate(sorted_names):
        src = src_dir / name
        if not src.exists() and image_path:
            src = Path(image_path) / name
        dst = os.path.join(tmp_img_dir, f"{idx:04d}.png")
        _create_symlink_or_copy(str(src), dst)

    try:
        # ------------------------------------------------------------------
        # Step 1: Feature extraction
        # ------------------------------------------------------------------
        if os.path.exists(db_path):
            os.remove(db_path)

        cmd = [colmap_exe_path, "feature_extractor",
               "--ImageReader.camera_model", "SIMPLE_RADIAL",
               "--ImageReader.single_camera", "1",
               "--SiftExtraction.peak_threshold", str(sift_peak_threshold),
               "--SiftExtraction.edge_threshold", str(sift_edge_threshold),
               "--SiftExtraction.max_num_features", str(sift_max_num_features),
               "--SiftExtraction.domain_size_pooling", "1",
               "--FeatureExtraction.max_image_size", str(max_image_size),
               "--database_path", db_path,
               "--image_path", tmp_img_dir]
        _run_colmap(cmd, "Feature extraction", colmap_bin_dir)

        # ------------------------------------------------------------------
        # Read image_id -> filename mapping from database
        # ------------------------------------------------------------------
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
            rows = cursor.fetchall()
            id_to_orig_name: Dict[int, str] = {}
            for image_id, seq_name in rows:
                try:
                    seq_idx = int(Path(seq_name).stem)      # "0000" -> 0
                    orig_name = sorted_names[seq_idx]        # original filename
                    id_to_orig_name[image_id] = orig_name
                except (ValueError, IndexError):
                    continue
        finally:
            conn.close()

        if not id_to_orig_name:
            raise RuntimeError("No valid images found in COLMAP database.")

        # ------------------------------------------------------------------
        # Step 2: Matching
        # ------------------------------------------------------------------
        if matcher == "sequential":
            # Best for video sequences: only match temporally adjacent frames.
            # O(n·overlap) pairs instead of O(n²), and avoids feeding
            # weak-baseline far-apart pairs to the mapper.
            matching_cmd = [
                colmap_exe_path, "sequential_matcher",
                "--database_path", db_path,
                "--SequentialMatching.overlap", str(matcher_overlap),
                "--SequentialMatching.loop_detection", "1" if loop_detection else "0",
                "--SequentialMatching.loop_detection_period", "10",
            ]
        else:
            matching_cmd = [colmap_exe_path, "exhaustive_matcher",
                            "--database_path", db_path]
        _run_colmap(matching_cmd, "Matching", colmap_bin_dir)

        # ------------------------------------------------------------------
        # Verify matching quality (detect COLMAP 4.x bug)
        # ------------------------------------------------------------------
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM two_view_geometries")
            total_pairs = c.fetchone()[0]
            if total_pairs == 0:
                raise RuntimeError(
                    "COLMAP matching produced zero geometric pairs. "
                    "Please check your input quality or use --pose-estimator opencv."
                )
            c.execute(
                "SELECT pair_id FROM two_view_geometries"
            )
            pair_ids = [r[0] for r in c.fetchall()]
            if pair_ids:
                # pair_id 编码：高 16 位 = image_id1，低 16 位 = image_id2（COLMAP 约定）
                img_ids = set()
                for pid in pair_ids:
                    img_ids.add(pid >> 16)
                    img_ids.add(pid & 0xFFFF)
                n_distinct_imgs = len(img_ids)
            else:
                n_distinct_imgs = 0
            if n_distinct_imgs < 2:
                raise RuntimeError(
                    f"COLMAP matching failed: only {n_distinct_imgs} distinct images "
                    f"involved ({total_pairs} pairs). All matches may point to the "
                    "same image_id (known COLMAP 4.x bug). "
                    "Please check your input quality or use --pose-estimator opencv."
                )
            print(f"  [COLMAP] Matching OK: {total_pairs} pairs, "
                  f"{n_distinct_imgs} distinct images")
        finally:
            conn.close()

        # ------------------------------------------------------------------
        # Step 3: Mapper (SfM)
        # ------------------------------------------------------------------
        cmd = [colmap_exe_path, "mapper",
               "--database_path", db_path,
               "--image_path", tmp_img_dir,
               "--output_path", sparse_dir]
        _run_colmap(cmd, "Mapper (SfM reconstruction)", colmap_bin_dir)

        # ------------------------------------------------------------------
        # Step 4: Select best reconstruction among all mapper output models.
        # The mapper may split the scene into multiple disconnected sub-models
        # (sparse/0, sparse/1, ...). Model 0 is NOT guaranteed to be the largest —
        # it can be a failed seed reconstruction with almost no images/points.
        # Convert every model to TXT and pick the one with the most registered
        # images (tie-break: most 3D points).
        # ------------------------------------------------------------------
        model_ids = sorted(
            int(d.name) for d in Path(sparse_dir).iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        if not model_ids:
            raise RuntimeError(
                "COLMAP mapper failed — no reconstruction produced.\n"
                "Please check COLMAP logs or use --pose-estimator opencv."
            )

        best_score = -1
        best_txt_dir = None
        best_meta = None
        for mid in model_ids:
            recon_path = Path(sparse_dir) / str(mid)
            txt_dir = str(recon_path) + "_txt"
            os.makedirs(txt_dir, exist_ok=True)
            cmd = [colmap_exe_path, "model_converter",
                   "--input_path", str(recon_path),
                   "--output_path", txt_dir,
                   "--output_type", "TXT"]
            _run_colmap(cmd, f"Model conversion (model {mid} -> TXT)", colmap_bin_dir)

            # Score = registered images, tie-broken by 3D point count.
            n_images = 0
            n_points = 0
            img_txt = os.path.join(txt_dir, "images.txt")
            pts_txt = os.path.join(txt_dir, "points3D.txt")
            if os.path.isfile(img_txt):
                with open(img_txt, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        # Registered image: 9+ tokens and not a 2D-observation line.
                        if s and not s.startswith("#") and len(s.split()) >= 9 \
                                and not s.split()[0].startswith("-"):
                            n_images += 1
            if os.path.isfile(pts_txt):
                with open(pts_txt, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        if s and not s.startswith("#") and len(s.split()) >= 4:
                            n_points += 1

            score = n_images * 10000 + n_points
            print(f"  [COLMAP] Model {mid}: {n_images} images, {n_points} points")
            if score > best_score:
                best_score = score
                best_txt_dir = txt_dir
                best_meta = (mid, n_images, n_points)

        if best_txt_dir is None:
            raise RuntimeError(
                "COLMAP mapper produced no parseable reconstruction.\n"
                "Please check COLMAP logs or use --pose-estimator opencv."
            )

        # ------------------------------------------------------------------
        # Step 5: Parse the best model's TXT and map poses back to original
        # ------------------------------------------------------------------
        intrinsics, poses_dict, sparse_points = _parse_colmap_txt(
            best_txt_dir, id_to_orig_name, name_to_frame_idx, len(frame_paths)
        )

        ordered_poses = [None] * len(frame_paths)
        for frame_idx, pose in poses_dict.items():
            ordered_poses[frame_idx] = pose

        # Save artifacts for resume compatibility
        np.save(workdir / "intrinsics.npy", intrinsics.K)
        valid_poses = [p for p in ordered_poses if p is not None]
        if valid_poses:
            np.save(workdir / "poses.npy", np.stack([p.RT for p in valid_poses]))
        if sparse_points.size > 0:
            np.save(workdir / "sparse_points.npy", sparse_points)

        best_mid = best_meta[0] if best_meta is not None else -1

        print(f"  [COLMAP] Estimated {len(valid_poses)} poses "
              f"(total frames: {len(frame_paths)}), "
              f"{len(sparse_points)} sparse 3D points "
              f"(selected model {best_mid})")
        return intrinsics, ordered_poses, sparse_points

    finally:
        # Cleanup temporary sorted images
        try:
            if os.path.isdir(tmp_img_dir):
                shutil.rmtree(tmp_img_dir, ignore_errors=True)
        except Exception:
            pass


# ------------------------------------------------------------------
# TXT parsers (line-pair logic for images.txt)
# ------------------------------------------------------------------

def _parse_colmap_txt(
    recon_txt_path: str,
    id_to_orig_name: Dict[int, str],
    name_to_frame_idx: Dict[str, int],
    num_frames: int,
) -> Tuple[CameraIntrinsics, Dict[int, CameraPose], np.ndarray]:
    """Parse COLMAP TXT files and map poses back to original frame indices."""
    cameras = _parse_cameras_txt(os.path.join(recon_txt_path, "cameras.txt"))
    if not cameras:
        raise RuntimeError("No camera model found in cameras.txt")
    images = _parse_images_txt(
        os.path.join(recon_txt_path, "images.txt"),
        id_to_orig_name,
        name_to_frame_idx,
    )
    points = _parse_points_txt(os.path.join(recon_txt_path, "points3D.txt"))

    # Use first camera model (single_camera=1)
    cam = cameras[0]
    intrinsics = CameraIntrinsics(fx=cam["fx"], fy=cam["fy"],
                                  cx=cam["cx"], cy=cam["cy"])
    return intrinsics, images, points


def _parse_cameras_txt(path: str) -> list:
    """Parse cameras.txt, supports multiple camera models."""
    MODEL_PARAMS = {
        "SIMPLE_RADIAL": lambda w, h, p: {"fx": p[0], "fy": p[0], "cx": p[1], "cy": p[2]},
        "SIMPLE_PINHOLE": lambda w, h, p: {"fx": p[0], "fy": p[0], "cx": p[1], "cy": p[2]},
        "PINHOLE": lambda w, h, p: {"fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3]},
        "OPENCV": lambda w, h, p: {"fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3]},
    }

    cameras = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]

            mapper = MODEL_PARAMS.get(model)
            if mapper:
                intr = mapper(width, height, params)
            else:
                # Fallback: assume first four params are fx, fy, cx, cy
                intr = {"fx": params[0], "fy": params[1] if len(params) > 1 else params[0],
                        "cx": params[2] if len(params) > 2 else 0.0,
                        "cy": params[3] if len(params) > 3 else 0.0}

            cameras.append({
                "camera_id": cam_id, "model": model,
                "width": width, "height": height, "params": params,
                **intr,
            })
    return cameras


def _parse_images_txt(
    path: str,
    id_to_orig_name: Dict[int, str],
    name_to_frame_idx: Dict[str, int],
) -> Dict[int, CameraPose]:
    """Parse images.txt using line-pair pattern."""
    poses_map: Dict[int, CameraPose] = {}
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        # First line of a pair: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        parts = line.split(maxsplit=9)
        if len(parts) < 9:
            continue
        try:
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            # camera_id = int(parts[8])  # not needed
            name = parts[9] if len(parts) > 9 else ""

            orig_name = id_to_orig_name.get(image_id)
            if orig_name is None:
                # Skip second line (points2D) and continue
                if i < len(lines):
                    i += 1
                continue
            frame_idx = name_to_frame_idx.get(orig_name)
            if frame_idx is None:
                if i < len(lines):
                    i += 1
                continue

            R_cam = _quat_to_rot(qw, qx, qy, qz)
            t_cam = np.array([tx, ty, tz])
            poses_map[frame_idx] = CameraPose(R=R_cam.copy(), t=t_cam.copy())

            # Skip the next line (points2D for this image)
            if i < len(lines):
                i += 1
        except (ValueError, IndexError):
            # Malformed line, try to continue
            continue

    return poses_map


def _parse_points_txt(path: str) -> np.ndarray:
    """Parse points3D.txt, each point line: POINT3D_ID X Y Z ..."""
    points = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                    points.append(xyz)
                except ValueError:
                    pass
    return np.array(points, dtype=np.float64) if points else np.empty((0, 3))


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ])