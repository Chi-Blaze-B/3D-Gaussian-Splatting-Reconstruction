"""视频转 3D 高斯泼溅 CLI 端。"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import psutil
import torch

import cv2

from frames import extract_frames
from poses import estimate_poses, CameraPose, FrameStatus   # [A1]
from point_cloud import initialize_gaussians, sample_point_colors
from gaussian import (
    Gaussian3D, Trainer, LazyFrames, LossDivergenceError,
    auto_tune_config,
)
from exporter import export_training_checkpoint

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def set_affinity_to_all_cores() -> None:
    try:
        p = psutil.Process(os.getpid())
        all_cpus = list(range(psutil.cpu_count()))
        p.cpu_affinity(all_cpus)
        logger.info("CPU 亲和性设置为 %d 个核心", len(all_cpus))
    except Exception as e:
        logger.warning("无法设置 CPU 亲和性: %s", e)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video-to-3D Gaussian Splatting Pipeline (simplified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入 / 输出
    parser.add_argument("--video", type=str, required=True, help="输入视频文件路径")
    parser.add_argument("--output", type=str, default="output.ply", help="输出 PLY 文件路径")
    parser.add_argument("--workdir", type=str, default="./workdir", help="中间文件工作目录")

    # 帧提取
    parser.add_argument(
        "--sampling-mode", type=str,
        choices=["uniform", "smart", "two-stage"], default="uniform",
        help="帧采样策略：uniform / smart（光流）/ two-stage（视差+光流+纹理）",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="目标帧率（均匀采样模式）")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放系数 (0<scale<=1)")
    parser.add_argument("--min-frames", type=int, default=30, help="最少提取帧数")
    parser.add_argument("--max-frames", type=int, default=200, help="最多提取帧数")

    # 位姿估计
    parser.add_argument(
        "--pose-estimator", type=str,
        choices=["opencv", "colmap"], default="opencv",
        help="相机位姿估计后端（opencv=ORB+EM，colmap=外部 COLMAP）",
    )
    parser.add_argument(
        "--feature-type", type=str,
        choices=["orb", "sift"], default="orb",
        help="OpenCV 位姿估计的特征描述子（orb=快速二进制，sift=鲁棒浮点，较慢）",
    )
    parser.add_argument(
        "--use-focal-guess", action="store_true",
        help="以 1.0×图像长边作为初始像素焦距（约 53° 长边方向 FOV）传给位姿估计器；"
            "仅在 --pose-estimator opencv 时生效",
    )
    # [B1] 回环 / PGO 开关
    parser.add_argument(
        "--no-loop", action="store_true",
        help="关闭回环检测（短序列 / 纯前向拍摄可关，加速且减少误回环风险）",
    )
    parser.add_argument(
        "--no-pgo", action="store_true",
        help="关闭位姿图优化（无回环时收益有限）",
    )

    # 训练
    parser.add_argument("--num-epochs", type=int, default=3000, help="训练轮数")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="运行设备",
    )
    parser.add_argument(
        "--max-gaussians", type=int, default=None,
        help="高斯数量上限；不指定则按训练设备自动选择（可用 --show-config 查看）",
    )
    parser.add_argument("--eval-every", type=int, default=500, help="每 N 轮打印一次损失")

    # 高级特性
    parser.add_argument(
        "--sh-degree", type=int, default=0, choices=[0, 1, 2, 3],
        help="球谐阶数（0=仅漫反射，3=完整视角相关）",
    )
    parser.add_argument("--sh-warmup-steps", type=int, default=200,
                        help="球谐阶数渐进提升的步数")
    parser.add_argument("--ssim-warmup-steps", type=int, default=500,
                        help="SSIM 权重线性提升的步数（0→0.2）")
    parser.add_argument("--ssim-weight-max", type=float, default=0.2,
                        help="预热后 SSIM 权重上限")
    parser.add_argument("--random-background", action="store_true",
                        help="训练时随机使用黑/白背景")
    parser.add_argument("--train-focal", action="store_true",
                        help="训练时学习焦距（自标定）")
    parser.add_argument("--amp", action="store_true",
                        help="混合精度训练（fp16，仅 CUDA + Tensor Core 有收益，光栅化器保持 fp32）")

    # 断点续训
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="从上次运行的 workdir 续训（需包含 training_state.pt）")

    # 信息输出
    parser.add_argument("--show-config", action="store_true",
                        help="按 --device 打印硬件自适应渲染配置后退出")

    return parser


def load_poses(poses_data: np.ndarray) -> list:
    """从定长 [n,4,4] 数组还原位姿列表，NaN 行表示位姿缺失。"""
    poses = []
    for p in poses_data:
        if np.isnan(p).any():
            poses.append(None)
        else:
            poses.append(CameraPose(R=p[:3, :3].copy(), t=p[:3, 3].copy()))
    return poses


def save_poses(poses: list, path: Path) -> None:
    """位姿列表保存为定长 [n,4,4] 数组，缺失位姿填 NaN。"""
    poses_arr = np.full((len(poses), 4, 4), np.nan, dtype=np.float32)
    for i, p in enumerate(poses):
        if p is not None:
            poses_arr[i] = p.RT
    np.save(path, poses_arr)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def print_render_config(device: str) -> None:
    cfg = auto_tune_config(device=device)
    print("硬件自适应渲染配置：")
    print(f"  设备:          {device}")
    print(f"  分档来源:      {cfg.source}")
    print(f"  硬件容量:      {cfg.hardware_gb:.1f} GB")
    print(f"  raster_chunk:  {cfg.raster_chunk}")
    print(f"  radius_max:    {cfg.radius_max}")
    print(f"  max_span:      {cfg.max_span}")
    print(f"  max_gaussians: {cfg.max_gaussians}")


def _estimate_frame_mb(paths) -> float:
    """估算单帧 RGB uint8 内存占用（MB），与 LazyFrames 缓存单位对齐。"""
    try:
        img = cv2.imread(paths[0], cv2.IMREAD_COLOR)
        if img is None:
            return 6.0  # 1080p 估算
        return img.shape[0] * img.shape[1] * 3 / (1024 ** 2)
    except Exception:
        return 6.0


# [B3] 稀疏点云导出（xyz + rgb），供外部工具检查位姿
def _write_xyzrgb_ply(path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """写二进制 PLY（x y z + red green blue）。xyz 期望 (N,3)，rgb 期望 (N,3) uint8。"""
    xyz = np.asarray(xyz, dtype=np.float32)
    n = int(xyz.shape[0])
    if n == 0:
        return
    if rgb is None:
        rgb = np.full((n, 3), 128, dtype=np.uint8)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape[0] != n:
        rgb = np.full((n, 3), 128, dtype=np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    arr = np.empty(n, dtype=dtype)
    arr["x"] = xyz[:, 0]
    arr["y"] = xyz[:, 1]
    arr["z"] = xyz[:, 2]
    arr["r"] = rgb[:, 0]
    arr["g"] = rgb[:, 1]
    arr["b"] = rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


# [A1] 依据 frame_status 过滤 LOST / INVALID 帧，构造训练用位姿列表
def _build_train_poses(poses, sfm_result, n_frames):
    """返回与 n_frames 对齐的位姿列表；LOST/INVALID 或 None 填 None。"""
    train_poses = []
    if sfm_result is not None and sfm_result.frame_status is not None:
        statuses = sfm_result.frame_status
        n_lost = 0
        for i in range(n_frames):
            p = poses[i] if i < len(poses) else None
            st = statuses[i] if i < len(statuses) else None
            if p is None or st in (FrameStatus.LOST, FrameStatus.INVALID):
                train_poses.append(None)
                n_lost += 1
            else:
                train_poses.append(p.RT.astype(np.float32))
        if n_lost > 0:
            logger.info("训练位姿过滤：跳过 %d/%d 帧（LOST / INVALID）",
                        n_lost, n_frames)
    else:
        for p in poses[:n_frames]:
            train_poses.append(p.RT.astype(np.float32) if p is not None else None)
    return train_poses


def run_pipeline(args: argparse.Namespace) -> None:
    overall_start = time.time()

    # --resume-dir 时 workdir 强制指向该目录
    workdir = Path(args.workdir)
    if args.resume_dir is not None:
        workdir = Path(args.resume_dir)
        if not workdir.exists():
            logger.error("续训目录不存在: %s", args.resume_dir)
            sys.exit(1)
        logger.info("从以下工作目录续训: %s", workdir)
    else:
        workdir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    logger.info("使用设备: %s", device)

    frame_dir = workdir / "frames"
    poses_dir = workdir / "poses"
    poses_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("  视频 → 3D 高斯泼溅")
    logger.info("=" * 60)

    # 步骤 1：提取帧
    logger.info("[1/5] 正在提取帧...")
    t0 = time.time()

    frame_paths_file = workdir / "frame_paths.txt"
    meta_path = workdir / "frame_meta.json"

    can_reuse_frames = frame_paths_file.exists()

    # 提取参数（video / scale / fps / sampling_mode / feature_type）任一变化
    # 都会改变下游语义，需要作废所有缓存。
    if can_reuse_frames and meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text())
        except Exception:
            old_meta = {}

        mismatch_reasons = []
        old_video = old_meta.get("video")
        old_scale = old_meta.get("scale")
        old_fps = old_meta.get("fps")
        old_sampling = old_meta.get("sampling_mode")
        old_feature = old_meta.get("feature_type")

        if old_video is not None and os.path.abspath(str(old_video)) != os.path.abspath(args.video):
            mismatch_reasons.append(
                f"video {os.path.basename(str(old_video))} → {os.path.basename(args.video)}"
            )
        if old_scale is not None and abs(float(old_scale) - float(args.scale)) >= 1e-6:
            mismatch_reasons.append(f"scale {old_scale} → {args.scale:.2f}")
        if old_fps is not None and abs(float(old_fps) - float(args.fps)) >= 1e-6:
            mismatch_reasons.append(f"fps {old_fps} → {args.fps}")
        if old_sampling is not None and old_sampling != args.sampling_mode:
            mismatch_reasons.append(
                f"sampling_mode {old_sampling} → {args.sampling_mode}"
            )
        if old_feature is not None and old_feature != args.feature_type:
            mismatch_reasons.append(
                f"feature_type {old_feature} → {args.feature_type}"
            )

        if mismatch_reasons:
            logger.warning(
                "  提取参数变化（%s），重新提取帧并作废下游缓存（位姿 / 高斯 / 检查点）",
                "; ".join(mismatch_reasons),
            )
            can_reuse_frames = False
            for stale in (
                workdir / "training_state.pt",
                workdir / "best_training_state.pt",
                workdir / "gaussian_params.npz",
                workdir / "intrinsics.npy",
                workdir / "poses.npy",
                workdir / "sparse_points.npy",
            ):
                try:
                    if stale.exists():
                        stale.unlink()
                        logger.info("  🗑  已删除过期文件 %s", stale.name)
                except OSError as e:
                    logger.warning("  ⚠️  无法删除过期文件 %s: %s", stale.name, e)
    elif can_reuse_frames:
        logger.info("  提示: 复用旧帧（无 frame_meta.json，未校验提取参数）")

    if can_reuse_frames:
        frame_paths = [p.strip() for p in frame_paths_file.read_text().splitlines()]
        logger.info("已从 %s 加载 %d 帧", workdir, len(frame_paths))
    else:
        smart_sampling = args.sampling_mode != "uniform"
        two_stage = args.sampling_mode == "two-stage"

        frame_paths = extract_frames(
            video_path=args.video,
            output_dir=str(frame_dir),
            fps=args.fps,
            scale=args.scale,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            smart_sampling=smart_sampling,
            two_stage=two_stage,
            poses_output_dir=str(poses_dir / "coarse_poses") if two_stage else None,
            optical_flow_method="farneback",
            feature_type=args.feature_type,
        )
        frame_paths_file.write_text("\n".join(frame_paths))
        meta_path.write_text(json.dumps({
            "video": os.path.abspath(args.video),
            "scale": args.scale,
            "fps": args.fps,
            "sampling_mode": args.sampling_mode,
            "feature_type": args.feature_type,
        }))
        logger.info("已提取 %d 帧（耗时 %.1fs）", len(frame_paths), time.time() - t0)

    if len(frame_paths) < 2:
        logger.error("至少需要 2 帧。")
        sys.exit(1)

    # ---- 帧容器：按估算内存选 preload / LRU ----
    per_frame_mb = _estimate_frame_mb(frame_paths)
    total_mb = per_frame_mb * len(frame_paths)
    if total_mb > 4096:
        # [B4] PGO / 多视图重三角化会非顺序访问关键帧，缓存随序列长度自适应
        cache_n = min(512, max(128, len(frame_paths) // 2))
        frames = LazyFrames(frame_paths, preload=False, cache_size=cache_n)
        logger.info("帧缓存约 %.0fMB > 4GB，启用 LRU 滑窗（%d 帧上限）",
                    total_mb, cache_n)
    else:
        frames = LazyFrames(frame_paths, preload=True)
        logger.info("帧缓存：全量预加载（约 %.0fMB）", total_mb)

    h, w = frames[0].shape[:2]
    logger.info("帧分辨率: %dx%d", w, h)

    # 步骤 2：估计相机位姿
    logger.info("[2/5] 正在估计相机位姿...")
    t0 = time.time()

    intrinsics_file = workdir / "intrinsics.npy"
    poses_file = workdir / "poses.npy"
    sparse_file = workdir / "sparse_points.npy"

    # 仅当本轮新跑 SfM 时才有值，用于后续复用 rgb / obs_per_point
    sfm_result = None

    if intrinsics_file.exists() and poses_file.exists() and sparse_file.exists():
        K = np.load(intrinsics_file)
        sparse_points = np.load(sparse_file)
        poses = load_poses(np.load(poses_file))
        logger.info("已从 %s 加载位姿", workdir)
    else:
        focal_guess = None
        if args.use_focal_guess:
            focal_guess = float(max(w, h))
            fov_deg = 2.0 * np.degrees(np.arctan(max(w, h) / (2.0 * focal_guess)))
            axis = "水平" if w >= h else "垂直"
            logger.info("初始焦距猜测: %.1fpx（约 %.0f° %s FOV）",
                        focal_guess, fov_deg, axis)

        if args.pose_estimator == "colmap":
            try:
                from colmap_poses import estimate_poses_with_colmap
                intrinsics, poses, sparse_points = estimate_poses_with_colmap(
                    frame_paths, str(workdir)
                )
                K = intrinsics.K
            except (ImportError, RuntimeError) as e:
                logger.error("COLMAP 失败: %s。请安装 COLMAP 或改用 --pose-estimator opencv。", e)
                sys.exit(1)
        else:  # opencv
            try:
                sfm_result = estimate_poses(
                    frame_paths,
                    min_inliers=25,
                    feature_type=args.feature_type,
                    focal_guess=focal_guess,
                    aspect_ratio=1.0,
                    # [B1] 用户可控的回环 / PGO 开关
                    enable_loop=not args.no_loop,
                    enable_pgo=not args.no_pgo,
                )
                K = sfm_result.intrinsics.K
                poses = sfm_result.poses
                sparse_points = sfm_result.xyz
            except RuntimeError as e:
                logger.error("位姿估算失败: %s", e)
                logger.error(
                    "建议：改用 --pose-estimator colmap，"
                    "或换用平移充分、纹理丰富、无大面积运动模糊的素材。"
                )
                sys.exit(1)

            # [B2] 结构化 SfM 诊断
            if sfm_result.frame_status is not None:
                st = Counter(sfm_result.frame_status)
                logger.info("SfM 帧状态: %s", dict(st))
                logger.info("SfM 关键帧: %d / %d",
                            len(sfm_result.keyframes), len(frame_paths))
                logger.info("SfM 回环: %d 处", len(sfm_result.loop_closures))
                if len(sfm_result.loop_closures) > 0:
                    for a, b in sfm_result.loop_closures[:10]:
                        logger.info("  loop: %d ↔ %d", a, b)
                n_lost = st.get(FrameStatus.LOST, 0) + st.get(FrameStatus.INVALID, 0)
                if n_lost > len(frame_paths) * 0.3:
                    logger.warning(
                        "超过 30%% 的帧丢失/无效（%d / %d），"
                        "建议改用 --pose-estimator colmap 或更换素材。",
                        n_lost, len(frame_paths),
                    )

            # [B3] 导出稀疏点云，便于外部工具检查 SfM 质量
            try:
                sparse_ply = workdir / "sparse_points.ply"
                _write_xyzrgb_ply(
                    str(sparse_ply),
                    sparse_points,
                    sfm_result.rgb if sfm_result.rgb is not None else None,
                )
                logger.info("稀疏点云已导出: %s（可用 MeshLab / CloudCompare 检查位姿）",
                            sparse_ply)
            except Exception as e:
                logger.warning("稀疏点云导出失败: %s", e)

        np.save(intrinsics_file, K)
        save_poses(poses, poses_file)
        # 空点云也写盘：下游缓存判定依赖此文件存在
        if sparse_points is None:
            sparse_points = np.zeros((0, 3), dtype=np.float32)
        np.save(sparse_file, sparse_points)

    while len(poses) < len(frame_paths):
        poses.append(None)

    # [A1] 用 frame_status 判定有效位姿数，而非 poses 里是否有 None
    if sfm_result is not None and sfm_result.frame_status is not None:
        lost_or_invalid = sum(
            1 for st in sfm_result.frame_status
            if st in (FrameStatus.LOST, FrameStatus.INVALID)
        )
        valid_count = len(frame_paths) - lost_or_invalid
    else:
        valid_count = sum(1 for p in poses if p is not None)
    logger.info("共 %d 帧，其中 %d 帧位姿有效（耗时 %.1fs）",
                len(frame_paths), valid_count, time.time() - t0)

    if valid_count < 3:
        logger.error("有效位姿过少。请检查视频质量，或改用 --pose-estimator colmap。")
        sys.exit(1)

    # 步骤 3：初始化高斯
    logger.info("[3/5] 正在初始化 3D 高斯...")
    t0 = time.time()

    gauss_init_file = workdir / "gaussian_params.npz"
    required_keys = ("positions", "scales", "opacities", "sh_coeffs", "rotations")

    gauss_init = None
    if gauss_init_file.exists():
        try:
            params = dict(np.load(gauss_init_file))
            missing = [k for k in required_keys if k not in params]
            if missing:
                logger.warning(
                    "gaussian_params.npz 缺少键 %s，重新初始化", missing
                )
            else:
                n_loaded = int(params["positions"].shape[0])
                if n_loaded == 0:
                    # 早期版本可能写入空 npz，加载它会训练出空 PLY
                    logger.warning(
                        "gaussian_params.npz 为空（0 个高斯），删除并重新初始化"
                    )
                    try:
                        gauss_init_file.unlink()
                    except OSError as e:
                        logger.warning("删除空 npz 失败: %s", e)
                else:
                    gauss_init = {k: params[k] for k in required_keys}
                    logger.info("已从 %s 加载 %d 个初始化高斯", workdir, n_loaded)
        except Exception as e:
            logger.warning("读取 gaussian_params.npz 失败: %s，重新初始化", e)

    if gauss_init is None:
        # 本轮刚跑完 SfM 时优先复用其内建颜色 / 观测计数，省掉全量帧读取
        use_sfm_colors = (
            sfm_result is not None
            and sfm_result.rgb is not None
            and len(sfm_result.rgb) == len(sparse_points)
        )

        if use_sfm_colors:
            colors = np.asarray(sfm_result.rgb, dtype=np.uint8)
            if (sfm_result.obs_per_point is not None
                    and len(sfm_result.obs_per_point) == len(sparse_points)):
                counts = np.array([len(o) for o in sfm_result.obs_per_point], dtype=np.int32)
            else:
                counts = np.ones(len(sparse_points), dtype=np.int32)
            logger.info("复用 SfM 内建颜色（%d 点，无需重采样）", len(colors))
        else:
            class _Intrinsics:
                pass
            _intr = _Intrinsics()
            _intr.K = K
            colors, counts = sample_point_colors(sparse_points, poses, frames, _intr)
            logger.info("从帧重采样颜色（%d 点）", len(colors))

        gauss_init = initialize_gaussians(sparse_points, colors, counts)

        num_gs = int(gauss_init["positions"].shape[0])
        if num_gs == 0:
            # 0 高斯会训练出无意义的空 PLY，直接退出
            logger.error(
                "初始化高斯数为 0：sparse_points=%d，colors=%d。"
                "可能原因：SfM 未产出有效点云，或点云被全部剔除。"
                "请检查视频质量，或改用 --pose-estimator colmap。",
                len(sparse_points), len(colors),
            )
            sys.exit(1)

        np.savez(gauss_init_file, **gauss_init)
        logger.info("已初始化 %d 个高斯（耗时 %.1fs）",
                    num_gs, time.time() - t0)

        if sfm_result is not None and sfm_result.keyframes:
            logger.info("  SfM 关键帧: %d / %d",
                        len(sfm_result.keyframes), len(frame_paths))
    else:
        logger.info("已加载 %d 个高斯（跳过初始化）",
                    gauss_init["positions"].shape[0])

    # 步骤 4：训练
    logger.info("[4/5] 正在训练 3D 高斯...")
    if args.sh_degree > 0:
        logger.info("SH 阶数: %d，升温步数: %d", args.sh_degree, args.sh_warmup_steps)
    if args.ssim_warmup_steps > 0:
        logger.info("SSIM 升温步数: %d，最大权重: %s",
                    args.ssim_warmup_steps, args.ssim_weight_max)
    if args.random_background:
        logger.info("随机背景: 已启用")
    if args.train_focal:
        logger.info("焦距自校准: 已启用")

    gaussians = Gaussian3D()
    gaussians.initialize_from_dict(gauss_init, device=device)

    # max_gaussians 只覆盖密度上限，光栅化器参数仍按设备自动分档
    render_config = auto_tune_config(device=device)
    if args.max_gaussians is not None and args.max_gaussians != render_config.max_gaussians:
        old_max = render_config.max_gaussians
        render_config = replace(render_config, max_gaussians=int(args.max_gaussians))
        logger.info("max_gaussians 覆盖：%d → %d", old_max, render_config.max_gaussians)

    logger.info("渲染配置（设备=%s）：raster_chunk=%d radius_max=%d max_span=%d "
                "max_gaussians=%d (来源=%s, 硬件 %.1fGB)",
                device, render_config.raster_chunk, render_config.radius_max,
                render_config.max_span, render_config.max_gaussians,
                render_config.source, render_config.hardware_gb)

    trainer = Trainer(
        gaussians=gaussians,
        K=K,
        image_width=w,
        image_height=h,
        device=device,
        rasterizer=None,
        sh_degree=args.sh_degree,
        random_background=args.random_background,
        train_focal=args.train_focal,
        render_config=render_config,
        sh_warmup_steps=args.sh_warmup_steps,
        ssim_warmup_steps=args.ssim_warmup_steps,
        ssim_weight_max=args.ssim_weight_max,
        use_amp=args.amp,
    )

    # [A1] LOST / INVALID 帧不喂给训练
    train_poses = _build_train_poses(poses, sfm_result, len(frame_paths))

    start_epoch = 1
    start_frame = 0
    pt_ckpt = workdir / "training_state.pt"
    best_loss = float("inf")
    training_start = time.time()

    if pt_ckpt.exists():
        try:
            trainer.load_training_state(str(pt_ckpt), device=device)
            best_loss = trainer.best_loss
            saved = trainer.current_step
            n_valid = sum(1 for p in train_poses if p is not None)
            start_epoch = max(1, saved // max(n_valid, 1) + 1)
            start_frame = (trainer.last_frame_index + 1) if (n_valid > 0 and saved % n_valid != 0) else 0
            if start_frame > 0:
                logger.info("已恢复：从第 %d 轮（步 %d）继续，接续帧 %d",
                            start_epoch, saved, start_frame)
            else:
                logger.info("已恢复：从第 %d 轮（步 %d）继续", start_epoch, saved)
        except Exception as e:
            logger.warning("加载训练状态失败: %s。将从头开始训练。", e)

    for epoch in range(start_epoch, args.num_epochs + 1):
        try:
            avg_loss = trainer.train_epoch(
                frames_iter=frames,
                camera_poses=train_poses,
                stop_event=None,
                progress_callback=None,
                loss_threshold=1.0,
                checkpoint_path=str(pt_ckpt),
                start_frame=start_frame,
            )
            start_frame = 0
        except LossDivergenceError as e:
            logger.warning("损失发散: %s", e)
            trainer.save_training_state(str(pt_ckpt))
            break
        except KeyboardInterrupt:
            logger.info("用户中断，正在保存检查点...")
            trainer.save_training_state(str(pt_ckpt))
            logger.info("检查点已保存。可用 --resume-dir %s 续训", workdir)
            sys.exit(0)

        if epoch % max(1, args.eval_every) == 0 or epoch == start_epoch:
            elapsed = time.time() - training_start
            logger.info("轮次 %5d/%d | 损失: %.6f | 耗时: %.1fs | 高斯数: %d",
                        epoch, args.num_epochs, avg_loss, elapsed,
                        trainer.gaussians.num_gaussians)

        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save_training_state(str(workdir / "best_training_state.pt"))

    total_train = time.time() - training_start
    logger.info("训练完成。最佳损失: %.6f（耗时 %.1fs）", best_loss, total_train)

    # 步骤 5：导出
    logger.info("[5/5] 正在导出 PLY...")
    export_training_checkpoint(trainer, args.output, sh_degree=args.sh_degree)

    total = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("完成！输出文件: %s", os.path.abspath(args.output))
    logger.info("总耗时: %.1fs", total)
    logger.info("=" * 60)


def cli(argv: list[str] = None) -> None:
    setup_logging()
    set_affinity_to_all_cores()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_config:
        print_render_config(resolve_device(args.device))
        sys.exit(0)

    if not os.path.isfile(args.video):
        logger.error("视频文件不存在: %s", args.video)
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    cli()