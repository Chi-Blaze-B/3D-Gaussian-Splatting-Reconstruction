"""视频转 3D 高斯泼溅 CLI 端。"""

import argparse
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import psutil
import torch

from frames import extract_frames
from poses import estimate_poses, CameraPose
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
    """将当前进程绑定到所有逻辑核心。"""
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
    parser.add_argument("--fps", type=float, default=15.0, help="目标帧率（均匀采样模式）")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放系数 (0<scale<=1)")
    parser.add_argument("--min-frames", type=int, default=30, help="最少提取帧数")
    parser.add_argument("--max-frames", type=int, default=200, help="最多提取帧数")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["uniform", "smart", "two-stage"],
        default="uniform",
        help="帧采样策略：uniform / smart（光流）/ two-stage（视差+光流+纹理）",
    )

    # 训练
    parser.add_argument("--num-epochs", type=int, default=3000, help="训练轮数")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--eval-every", type=int, default=500, help="每 N 轮打印一次损失")
    parser.add_argument("--max-gaussians", type=int, default=None,
                        help="高斯数量上限；不指定则按实际训练设备自动选择（可用 --show-config 查看）")

    # 高级特性
    parser.add_argument("--sh-degree", type=int, default=0, choices=[0, 1, 2, 3],
                        help="球谐阶数（0=仅漫反射，3=完整视角相关）")
    parser.add_argument("--sh-warmup-steps", type=int, default=1000,
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
                        help="混合精度训练（fp16，需 CUDA + 安培以上 GPU，光栅化器保持 fp32；CPU 无效）")

    # 位姿估计
    parser.add_argument(
        "--pose-estimator",
        type=str,
        choices=["opencv", "colmap"],
        default="opencv",
        help="相机位姿估计后端（opencv=ORB+EM，colmap=外部 COLMAP）",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["orb", "sift"],
        default="orb",
        help="OpenCV 位姿估计的特征描述子（orb=快速二进制，sift=鲁棒浮点，较慢）",
    )
    parser.add_argument("--focal-guess", type=float, default=None, help="初始焦距估计值（可选）")

    # 断点续训
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="从上次运行的 workdir 续训（需包含 training_state.pt）")

    # 信息输出
    parser.add_argument("--show-config", action="store_true",
                        help="按 --device 打印硬件自适应渲染配置后退出")

    return parser


def load_poses(poses_data: np.ndarray) -> list:
    """从定长 [n,4,4] 数组还原位姿列表，NaN 行表示该帧位姿缺失。"""
    poses = []
    for p in poses_data:
        if np.isnan(p).any():
            poses.append(None)
        else:
            poses.append(CameraPose(R=p[:3, :3].copy(), t=p[:3, 3].copy()))
    return poses


def save_poses(poses: list, path: Path) -> None:
    """将位姿列表保存为定长 [n,4,4] 数组，缺失位姿填 NaN。"""
    poses_arr = np.full((len(poses), 4, 4), np.nan, dtype=np.float32)
    for i, p in enumerate(poses):
        if p is not None:
            poses_arr[i] = p.RT
    np.save(path, poses_arr)


def resolve_device(device_arg: str) -> str:
    """把 "auto" 解析为实际设备字符串。"""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def print_render_config(device: str) -> None:
    """按指定设备打印硬件自适应渲染配置。"""
    cfg = auto_tune_config(device=device)
    print("硬件自适应渲染配置：")
    print(f"  设备:          {device}")
    print(f"  分档来源:      {cfg.source}")
    print(f"  硬件容量:      {cfg.hardware_gb:.1f} GB")
    print(f"  raster_chunk:  {cfg.raster_chunk}")
    print(f"  radius_max:    {cfg.radius_max}")
    print(f"  max_span:      {cfg.max_span}")
    print(f"  max_gaussians: {cfg.max_gaussians}")


def run_pipeline(args: argparse.Namespace) -> None:
    overall_start = time.time()

    # 指定 resume-dir 时，workdir 强制指向该目录
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

    # ---------- 步骤 1：提取帧 ----------
    logger.info("[1/5] 正在提取帧...")
    t0 = time.time()

    frame_paths_file = workdir / "frame_paths.txt"
    if frame_paths_file.exists():
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
        logger.info("已提取 %d 帧（耗时 %.1fs）", len(frame_paths), time.time() - t0)

    if len(frame_paths) < 2:
        logger.error("至少需要 2 帧。")
        sys.exit(1)

    frames = LazyFrames(frame_paths)
    h, w = frames[0].shape[:2]
    logger.info("帧分辨率: %dx%d", w, h)

    # ---------- 步骤 2：估计相机位姿 ----------
    logger.info("[2/5] 正在估计相机位姿...")
    t0 = time.time()

    intrinsics_file = workdir / "intrinsics.npy"
    poses_file = workdir / "poses.npy"
    sparse_file = workdir / "sparse_points.npy"

    if intrinsics_file.exists() and poses_file.exists() and sparse_file.exists():
        K = np.load(intrinsics_file)
        sparse_points = np.load(sparse_file)
        poses = load_poses(np.load(poses_file))
        logger.info("已从 %s 加载位姿", workdir)
    else:
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
            intrinsics, poses, sparse_points = estimate_poses(
                frame_paths,
                min_inliers=25,
                feature_type=args.feature_type,
                focal_guess=args.focal_guess,
                aspect_ratio=1.0,
            )
            K = intrinsics.K

        np.save(intrinsics_file, K)
        save_poses(poses, poses_file)
        if sparse_points is not None and sparse_points.size > 0:
            np.save(sparse_file, sparse_points)

    while len(poses) < len(frame_paths):
        poses.append(None)
    valid_count = sum(1 for p in poses if p is not None)
    logger.info("共 %d 帧，其中 %d 帧位姿有效（耗时 %.1fs）",
                len(frame_paths), valid_count, time.time() - t0)

    if valid_count < 3:
        logger.error("有效位姿过少。请检查视频质量，或改用 --pose-estimator colmap。")
        sys.exit(1)

    # ---------- 步骤 3：初始化高斯 ----------
    logger.info("[3/5] 正在初始化 3D 高斯...")
    t0 = time.time()

    gauss_init_file = workdir / "gaussian_params.npz"
    if gauss_init_file.exists():
        params = dict(np.load(gauss_init_file))
        gauss_init = {k: params[k] for k in ["positions", "scales", "opacities", "sh_coeffs", "rotations"]}
        logger.info("已从 %s 加载初始化高斯", workdir)
    else:
        class _Intrinsics:
            pass
        _intr = _Intrinsics()
        _intr.K = K
        colors, counts = sample_point_colors(sparse_points, poses, frames, _intr)
        gauss_init = initialize_gaussians(sparse_points, colors, counts)
        np.savez(gauss_init_file, **gauss_init)

    logger.info("已初始化 %d 个高斯（耗时 %.1fs）",
                gauss_init["positions"].shape[0], time.time() - t0)

    # ---------- 步骤 4：训练 ----------
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

    # 按实际训练设备分档；--max-gaussians 只覆盖密度上限，光栅化器参数不变。
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

    train_poses = [p.RT.astype(np.float32) if p is not None else None for p in poses]
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
            n_valid = sum(1 for p in poses if p is not None)
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

    # --show-config 只打印配置，不跑流程，也不要求 --video
    if args.show_config:
        print_render_config(resolve_device(args.device))
        sys.exit(0)

    if not os.path.isfile(args.video):
        logger.error("视频文件不存在: %s", args.video)
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    cli()