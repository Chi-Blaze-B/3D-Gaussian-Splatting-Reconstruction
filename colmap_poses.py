"""
基于 COLMAP 的位姿估计，用于 3D Gaussian Splatting（优化版）。

封装 COLMAP CLI，依次执行特征提取、匹配与稀疏重建，
再把结果解析为与 ORB+EM 流程一致的格式
（CameraPose 列表 + 稀疏点云）。
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 本地依赖
try:
    from poses import CameraIntrinsics, CameraPose
except ImportError:
    CameraIntrinsics = None
    CameraPose = None

logger = logging.getLogger(__name__)


def _run_colmap(cmd: List[str], label: str, colmap_bin: str) -> subprocess.CompletedProcess:
    """执行一条 COLMAP 命令，并正确设置 Qt 插件路径。"""
    logger.info("[COLMAP] %s...", label)
    env = dict(os.environ)

    # 清掉父进程（PyQt5 等）注入的 Qt 相关环境变量
    for key in list(env.keys()):
        if key.startswith("QT_"):
            del env[key]

    # 把 COLMAP bin 前置，保证优先加载它自带的 DLL
    existing = env.get("PATH", "")
    env["PATH"] = colmap_bin + os.pathsep + existing

    # 显式为 COLMAP 指定 Qt 插件路径
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
        msg = f"COLMAP {label} 失败，返回码 {result.returncode}"
        if err:
            msg += f": {err[:300]}"
        raise RuntimeError(msg)
    return result


def _create_symlink_or_copy(src: str, dst: str):
    """在 dst 处创建指向 src 的符号链接；失败时回退为复制。"""
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
    """用 COLMAP 估计相机位姿。

    参数：
        frame_paths: 输入帧的绝对路径列表。
        output_dir: 存放 COLMAP 中间结果的目录。
        colmap_exe: colmap 可执行文件路径（None 时自动探测）。
        max_image_size: 特征提取时的图像最大边长。默认 2400（调优最优值）：
            高分辨率能保留小尺度纹理 → SIFT 匹配更多更好 → 三角化点云更稠密。
        sift_peak_threshold: SIFT 峰值阈值。
        sift_edge_threshold: SIFT 边缘阈值。
        sift_max_num_features: 每张图 SIFT 特征数上限。默认 12000（调优最优值）：
            特征更多 → 匹配质量更好。注意：长序列会拉长运行时间，
            超长视频建议改用 --pose-estimator opencv 或降低特征数。
        matcher: 'exhaustive'（全对匹配，对无序图像集、两阶段采样间距不均的帧更稳，
            复杂度 O(n²)）或 'sequential'（仅匹配时间相邻帧，视频场景最快）。
            默认 'exhaustive'（对应调优最优值）。
        matcher_overlap: sequential 匹配器的重叠窗口（前后各取多少帧）。
        loop_detection: 启用 sequential 匹配器的回环检测
            （匹配重访同一场景的远距离帧；需要 vocabulary tree，
            否则 sequential_matcher 会卡住，故默认关闭）。

    返回：
        intrinsics: CameraIntrinsics 对象。
        poses: CameraPose 列表（未注册帧为 None），
               长度等于 len(frame_paths)，顺序与 frame_paths 一致。
        sparse_points: (N,3) numpy 数组，稀疏 3D 点。
    """
    if CameraIntrinsics is None or CameraPose is None:
        raise ImportError("CameraIntrinsics/CameraPose not found. Install poses module.")

    script_dir = str(Path(__file__).resolve().parent)

    # ------------------------------------------------------------------
    # 定位 COLMAP 可执行文件
    # ------------------------------------------------------------------
    colmap_exe_path = colmap_exe
    if colmap_exe_path is None:
        bundled_bin = os.path.join(script_dir, "colmap-x64-windows-nocuda", "bin")
        bundled_exe = os.path.join(bundled_bin, "colmap.exe")
        if os.path.isfile(bundled_exe):
            colmap_exe_path = bundled_exe
            logger.info("[COLMAP] 使用内置 COLMAP: %s", bundled_bin)
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
    # 目录准备
    # ------------------------------------------------------------------
    workdir = Path(output_dir) / "colmap_work"
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = str(workdir / "database.db")
    sparse_dir = str(workdir / "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 用符号链接（或复制）准备排序后的图像目录：
    # 既最小化磁盘占用，又保证顺序可预测。
    # ------------------------------------------------------------------
    tmp_img_dir = str(workdir / "sorted_images")
    if os.path.isdir(tmp_img_dir):
        shutil.rmtree(tmp_img_dir, ignore_errors=True)
    os.makedirs(tmp_img_dir, exist_ok=True)

    # 原始文件名 -> frame_paths 索引
    name_to_frame_idx: Dict[str, int] = {}
    for idx, p in enumerate(frame_paths):
        name_to_frame_idx[Path(p).name] = idx

    # 按文件名排序，保证映射关系：0000.png, 0001.png, ...
    sorted_names = sorted(Path(p).name for p in frame_paths)
    src_dir = Path(frame_paths[0]).parent  # 假定所有帧在同一目录
    for idx, name in enumerate(sorted_names):
        src = src_dir / name
        if not src.exists() and image_path:
            src = Path(image_path) / name
        dst = os.path.join(tmp_img_dir, f"{idx:04d}.png")
        _create_symlink_or_copy(str(src), dst)

    try:
        # ------------------------------------------------------------------
        # 步骤 1：特征提取
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
        _run_colmap(cmd, "特征提取", colmap_bin_dir)

        # ------------------------------------------------------------------
        # 从数据库读取 image_id -> 原始文件名映射
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
                    orig_name = sorted_names[seq_idx]        # 还原原始文件名
                    id_to_orig_name[image_id] = orig_name
                except (ValueError, IndexError):
                    continue
        finally:
            conn.close()

        if not id_to_orig_name:
            raise RuntimeError("No valid images found in COLMAP database.")

        # ------------------------------------------------------------------
        # 步骤 2：匹配
        # ------------------------------------------------------------------
        if matcher == "sequential":
            # 视频序列首选：只匹配时间相邻帧。
            # 复杂度 O(n·overlap) 而非 O(n²)，且避免把弱基线远距离帧对喂给 mapper。
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
        _run_colmap(matching_cmd, "匹配", colmap_bin_dir)

        # ------------------------------------------------------------------
        # 校验匹配质量（检测 COLMAP 4.x 的 bug）
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
            logger.info("[COLMAP] 匹配正常：%d 对，覆盖 %d 张不同图像",
                        total_pairs, n_distinct_imgs)
        finally:
            conn.close()

        # ------------------------------------------------------------------
        # 步骤 3：Mapper（SfM 重建）
        # ------------------------------------------------------------------
        cmd = [colmap_exe_path, "mapper",
               "--database_path", db_path,
               "--image_path", tmp_img_dir,
               "--output_path", sparse_dir]
        _run_colmap(cmd, "Mapper（SfM 重建）", colmap_bin_dir)

        # ------------------------------------------------------------------
        # 步骤 4：在 mapper 产出的所有模型中挑最好的。
        # mapper 可能把场景拆成多个互不连通的子模型（sparse/0, sparse/1, ...）。
        # model 0 未必最大 —— 可能只是一个几乎无图像/无点的失败种子。
        # 把每个模型都转成 TXT，选注册图像最多的（并列时比 3D 点数）。
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
            _run_colmap(cmd, f"模型转换（model {mid} -> TXT）", colmap_bin_dir)

            # 评分 = 注册图像数，并列时以 3D 点数打破平局
            n_images = 0
            n_points = 0
            img_txt = os.path.join(txt_dir, "images.txt")
            pts_txt = os.path.join(txt_dir, "points3D.txt")
            if os.path.isfile(img_txt):
                with open(img_txt, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        # 已注册图像行：9+ 个字段且不是 2D 观测行
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
            logger.info("[COLMAP] 模型 %d：%d 张图像，%d 个点", mid, n_images, n_points)
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
        # 步骤 5：解析最优模型的 TXT，并把位姿映射回原始帧索引
        # ------------------------------------------------------------------
        intrinsics, poses_dict, sparse_points = _parse_colmap_txt(
            best_txt_dir, id_to_orig_name, name_to_frame_idx, len(frame_paths)
        )

        ordered_poses = [None] * len(frame_paths)
        for frame_idx, pose in poses_dict.items():
            ordered_poses[frame_idx] = pose

        # 不再重复写 workdir/*.npy：顶层保存（定长 NaN 掩码）才是续训源，
        # 这里再写一份既无人读取，格式也可能与顶层不一致。
        valid_poses = [p for p in ordered_poses if p is not None]

        best_mid = best_meta[0] if best_meta is not None else -1

        logger.info("[COLMAP] 估计 %d 个位姿（总帧数 %d），%d 个稀疏 3D 点（选用模型 %d）",
                    len(valid_poses), len(frame_paths), len(sparse_points), best_mid)
        return intrinsics, ordered_poses, sparse_points

    finally:
        # 清理临时排序图像目录
        try:
            if os.path.isdir(tmp_img_dir):
                shutil.rmtree(tmp_img_dir, ignore_errors=True)
        except Exception:
            pass


# ------------------------------------------------------------------
# TXT 解析（images.txt 采用行对逻辑）
# ------------------------------------------------------------------

def _parse_colmap_txt(
    recon_txt_path: str,
    id_to_orig_name: Dict[int, str],
    name_to_frame_idx: Dict[str, int],
    num_frames: int,
) -> Tuple[CameraIntrinsics, Dict[int, CameraPose], np.ndarray]:
    """解析 COLMAP TXT 文件，并把位姿映射回原始帧索引。"""
    cameras = _parse_cameras_txt(os.path.join(recon_txt_path, "cameras.txt"))
    if not cameras:
        raise RuntimeError("No camera model found in cameras.txt")
    images = _parse_images_txt(
        os.path.join(recon_txt_path, "images.txt"),
        id_to_orig_name,
        name_to_frame_idx,
    )
    points = _parse_points_txt(os.path.join(recon_txt_path, "points3D.txt"))

    # 使用第一个相机模型（single_camera=1）
    cam = cameras[0]
    intrinsics = CameraIntrinsics(fx=cam["fx"], fy=cam["fy"],
                                  cx=cam["cx"], cy=cam["cy"])
    return intrinsics, images, points


def _parse_cameras_txt(path: str) -> list:
    """解析 cameras.txt，支持多种相机模型。"""
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
                # 回退：假定前四个参数为 fx, fy, cx, cy
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
    """按行对（line-pair）解析 images.txt。"""
    poses_map: Dict[int, CameraPose] = {}
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        # 行对的第一行：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        parts = line.split(maxsplit=9)
        if len(parts) < 9:
            continue
        try:
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            # camera_id = int(parts[8])  # 未使用
            name = parts[9] if len(parts) > 9 else ""

            orig_name = id_to_orig_name.get(image_id)
            if orig_name is None:
                # 跳过第二行（points2D）
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

            # 跳过下一行（该图像的 points2D）
            if i < len(lines):
                i += 1
        except (ValueError, IndexError):
            # 行格式异常，尝试继续
            continue

    return poses_map


def _parse_points_txt(path: str) -> np.ndarray:
    """解析 points3D.txt，每行格式：POINT3D_ID X Y Z ..."""
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
    """四元数 (w,x,y,z) → 3×3 旋转矩阵。"""
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ])