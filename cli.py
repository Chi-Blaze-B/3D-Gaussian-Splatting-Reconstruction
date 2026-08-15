"""
Command-line interface for the video-to-3DGS pipeline.

Usage examples:
    # Basic uniform sampling, ORB-based poses, train with SH0
    python cli.py --video input.mp4 --output out.ply

    # Smart sampling + SH3 + learnable focal
    python cli.py --video input.mp4 --output out.ply --sampling-mode smart --sh-degree 3 --train-focal

    # Two‑stage sampling + COLMAP + full features
    python cli.py --video input.mp4 --output out.ply --sampling-mode two-stage --pose-estimator colmap \
        --sh-degree 3 --random-background --train-focal --max-gaussians 500000
"""

import argparse
import os
import sys
import time
import signal
from pathlib import Path

import numpy as np
import torch

from frames import extract_frames
from poses import estimate_poses, CameraPose, CameraIntrinsics
from point_cloud import initialize_gaussians
from gaussian import Gaussian3D, DifferentiableRasterizer, Trainer, LazyFrames, LossDivergenceError
from exporter import export_training_checkpoint

import psutil
import os

def set_affinity_to_all_cores():
    """将当前进程绑定到所有逻辑核心"""
    try:
        p = psutil.Process(os.getpid())
        all_cpus = list(range(psutil.cpu_count()))
        p.cpu_affinity(all_cpus)
        print(f"[INFO] CPU 亲和性设置为 {len(all_cpus)} 个核心")
    except Exception as e:
        print(f"[WARN] 无法设置 CPU 亲和性: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video-to-3D Gaussian Splatting Pipeline (simplified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------- Input / Output ----------
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output.ply", help="Output PLY file path")
    parser.add_argument("--workdir", type=str, default="./workdir", help="Working directory for intermediate files")

    # ---------- Frame Extraction ----------
    parser.add_argument("--fps", type=float, default=15.0, help="Target frame rate (for uniform mode)")
    parser.add_argument("--scale", type=float, default=0.5, help="Resize scale factor (0<scale<=1)")
    parser.add_argument("--min-frames", type=int, default=30, help="Minimum number of frames to extract")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum number of frames to extract")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["uniform", "smart", "two-stage"],
        default="uniform",
        help="Frame sampling strategy: uniform, smart (flow-based), two-stage (parallax+flow+texture)"
    )

    # ---------- Training ----------
    parser.add_argument("--num-epochs", type=int, default=3000, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--eval-every", type=int, default=500, help="Print loss every N epochs")
    parser.add_argument("--max-gaussians", type=int, default=300000, help="Maximum number of Gaussians")

    # ---------- Advanced Features ----------
    parser.add_argument("--sh-degree", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Spherical Harmonics degree (0=diffuse only, 3=full view-dependent)")
    parser.add_argument("--sh-warmup-steps", type=int, default=1000,
                        help="Steps over which to gradually increase SH degree")
    parser.add_argument("--ssim-warmup-steps", type=int, default=500,
                        help="Steps over which to linearly increase SSIM weight (0→0.2)")
    parser.add_argument("--ssim-weight-max", type=float, default=0.2,
                        help="Maximum SSIM weight after warmup")
    parser.add_argument("--random-background", action="store_true",
                        help="Randomly sample black/white background during training")
    parser.add_argument("--train-focal", action="store_true",
                        help="Learn focal length during training (self‑calibration)")
    parser.add_argument("--enable-k1", action="store_true",
                        help="Learn radial distortion coefficient k1 (experimental)")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed precision (AMP / fp16) — requires CUDA GPU with fp16; "
                             "uses Tensor Cores on Ampere+. Rasterizer stays fp32. No effect on CPU.")

    # ---------- Pose Estimation ----------
    parser.add_argument(
        "--pose-estimator",
        type=str,
        choices=["opencv", "colmap"],
        default="opencv",
        help="Backend for camera pose estimation (opencv=ORB+EM, colmap=external COLMAP)"
    )
    parser.add_argument("--focal-guess", type=float, default=None, help="Initial focal length guess (optional)")

    # ---------- Resume ----------
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="Resume from a previous run's workdir (must contain training_state.pt)")

    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    overall_start = time.time()

    # ===== 修复: resume-dir 覆盖 workdir =====
    workdir = Path(args.workdir)
    if args.resume_dir is not None:
        # 如果用户显式指定了 resume-dir，则 workdir 强制指向该目录
        resume_path = Path(args.resume_dir)
        if not resume_path.exists():
            print(f"[ERROR] Resume directory not found: {args.resume_dir}")
            sys.exit(1)
        workdir = resume_path
        print(f"[INFO] Resuming from workdir: {workdir}")
    else:
        workdir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    frame_dir = workdir / "frames"
    poses_dir = workdir / "poses"
    poses_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Video → 3D Gaussian Splatting")
    print("=" * 60)

    # ---------- Step 1: Extract Frames ----------
    print("\n[1/5] Extracting frames...")
    t0 = time.time()

    # 优先从 resume-dir 加载，若不存在则提取
    frame_paths_file = workdir / "frame_paths.txt"
    if frame_paths_file.exists():
        frame_paths = [p.strip() for p in frame_paths_file.read_text().splitlines()]
        print(f"  Loaded {len(frame_paths)} frames from {workdir}")
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
        )
        frame_paths_file.write_text("\n".join(frame_paths))
        print(f"  Extracted {len(frame_paths)} frames ({time.time()-t0:.1f}s)")

    if len(frame_paths) < 2:
        print("Error: need at least 2 frames.")
        sys.exit(1)

    frames = LazyFrames(frame_paths)
    h, w = frames[0].shape[:2]
    print(f"  Frame resolution: {w}x{h}")

    # ---------- Step 2: Estimate Camera Poses ----------
    print("\n[2/5] Estimating camera poses...")
    t0 = time.time()

    intrinsics_file = workdir / "intrinsics.npy"
    poses_file = workdir / "poses.npy"
    sparse_file = workdir / "sparse_points.npy"

    if intrinsics_file.exists() and poses_file.exists() and sparse_file.exists():
        K = np.load(intrinsics_file)
        poses_data = np.load(poses_file)
        sparse_points = np.load(sparse_file)
        poses = [CameraPose(R=p[:3, :3].copy(), t=p[:3, 3].copy()) for p in poses_data]
        print(f"  Loaded poses from {workdir}")
    else:
        if args.pose_estimator == "colmap":
            try:
                from colmap_poses import estimate_poses_with_colmap
                intrinsics, poses, sparse_points = estimate_poses_with_colmap(
                    frame_paths, str(workdir)
                )
                K = intrinsics.K
            except (ImportError, RuntimeError) as e:
                print(f"  [ERROR] COLMAP failed: {e}. Please install COLMAP or use --pose-estimator opencv.")
                sys.exit(1)
        else:  # opencv
            intrinsics, poses, sparse_points = estimate_poses(
                frame_paths,
                str(poses_dir),
                min_inliers=25,
                feature_type="orb",
                focal_guess=args.focal_guess,
                aspect_ratio=1.0,
            )
            K = intrinsics.K

        # Save for potential resume
        np.save(intrinsics_file, K)
        valid_poses = [p for p in poses if p is not None]
        if valid_poses:
            np.save(poses_file, np.stack([p.RT for p in valid_poses]))
        if sparse_points is not None and sparse_points.size > 0:
            np.save(sparse_file, sparse_points)

    # Ensure poses list length matches frames
    while len(poses) < len(frame_paths):
        poses.append(None)
    valid_count = sum(1 for p in poses if p is not None)
    print(f"  Estimated {valid_count} valid poses out of {len(frame_paths)} frames ({time.time()-t0:.1f}s)")

    if valid_count < 3:
        print("Error: too few valid poses. Check video quality or try --pose-estimator colmap.")
        sys.exit(1)

    # ---------- Step 3: Initialize Gaussians ----------
    print("\n[3/5] Initializing 3D Gaussians...")
    t0 = time.time()

    gauss_init_file = workdir / "gaussian_params.npz"
    if gauss_init_file.exists():
        params = dict(np.load(gauss_init_file))
        gauss_init = {k: params[k] for k in ["positions", "scales", "opacities", "sh_coeffs", "rotations"]}
        print(f"  Loaded initialized Gaussians from {workdir}")
    else:
        class _Intrinsics:
            pass
        _intr = _Intrinsics()
        _intr.K = K
        gauss_init = initialize_gaussians(
            sparse_points=sparse_points,
            poses=poses,
            frame_paths=frame_paths,
            intrinsics=_intr,
        )
        np.savez(gauss_init_file, **gauss_init)

    print(f"  Initialized {gauss_init['positions'].shape[0]} Gaussians ({time.time()-t0:.1f}s)")

    # ---------- Step 4: Train ----------
    print(f"\n[4/5] Training 3D Gaussians...")
    print(f"  Device: {device}, Epochs: {args.num_epochs}, Max Gaussians: {args.max_gaussians}")
    if args.sh_degree > 0:
        print(f"  SH Degree: {args.sh_degree}, Warmup: {args.sh_warmup_steps} steps")
    if args.ssim_warmup_steps > 0:
        print(f"  SSIM Warmup: {args.ssim_warmup_steps} steps, max weight: {args.ssim_weight_max}")
    if args.random_background:
        print(f"  Random background: enabled")
    if args.train_focal:
        print(f"  Train focal: enabled")
    if args.enable_k1:
        print(f"  Train k1: enabled")

    gaussians = Gaussian3D()
    gaussians.initialize_from_dict(gauss_init, device=device)
    rasterizer = DifferentiableRasterizer(image_width=w, image_height=h)

    trainer = Trainer(
        gaussians=gaussians,
        rasterizer=rasterizer,
        K=K,
        image_width=w,
        image_height=h,
        device=device,
        sh_degree=args.sh_degree,
        random_background=args.random_background,
        train_focal=args.train_focal,
        max_gaussians=args.max_gaussians,
        sh_warmup_steps=args.sh_warmup_steps,
        ssim_warmup_steps=args.ssim_warmup_steps,
        ssim_weight_max=args.ssim_weight_max,
        enable_k1=args.enable_k1,
        use_amp=args.amp,
    )

    train_poses = [p.RT.astype(np.float32) if p is not None else None for p in poses]
    start_epoch = 1
    pt_ckpt = workdir / "training_state.pt"
    best_loss = float("inf")
    training_start = time.time()

    # Resume if checkpoint exists
    if pt_ckpt.exists():
        try:
            trainer.load_training_state(str(pt_ckpt), device=device)
            saved = trainer.current_step
            start_epoch = max(1, saved // max(len(frame_paths), 1))
            print(f"  Resumed from epoch {start_epoch} (step {saved})")
        except Exception as e:
            print(f"  [WARN] Failed to load training state: {e}. Starting from scratch.")

    # ===== 修复: 捕获 KeyboardInterrupt 并保存检查点 =====
    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            try:
                avg_loss = trainer.train_epoch(
                    frames_iter=frames,   # 传 LazyFrames 对象（内存缓存帧），避免每帧读盘
                    camera_poses=train_poses,
                    stop_event=None,   # CLI 无停止事件
                    progress_callback=None,
                    loss_threshold=1.0,
                    checkpoint_path=str(pt_ckpt),
                )
            except LossDivergenceError as e:
                print(f"\n  [LOSS DIVERGENCE] {e}")
                trainer.save_training_state(str(pt_ckpt))
                break
            except KeyboardInterrupt:
                # ===== 新增: Ctrl+C 时保存检查点 =====
                print("\n  [STOP] Interrupted by user, saving checkpoint...")
                trainer.save_training_state(str(pt_ckpt))
                print("  Checkpoint saved. To resume, use --resume-dir", workdir)
                sys.exit(0)

            if epoch % max(1, args.eval_every) == 0 or epoch == start_epoch:
                elapsed = time.time() - training_start
                print(f"  Epoch {epoch:>5d}/{args.num_epochs} | Loss: {avg_loss:.6f} | "
                      f"Time: {elapsed:.1f}s | Gaussians: {trainer.gaussians.num_gaussians}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                trainer.save_training_state(str(workdir / "best_training_state.pt"))

    except KeyboardInterrupt:
        # 外层防护（理论上不会触发）
        print("\n  [STOP] Interrupted, saving checkpoint...")
        trainer.save_training_state(str(pt_ckpt))
        print("  Checkpoint saved. To resume, use --resume-dir", workdir)
        sys.exit(0)

    total_train = time.time() - training_start
    print(f"\n  Training complete. Best loss: {best_loss:.6f} ({total_train:.1f}s)")

    # ---------- Step 5: Export ----------
    print("\n[5/5] Exporting PLY...")
    export_training_checkpoint(trainer, args.output, sh_degree=args.sh_degree)

    total = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  Done! Output: {os.path.abspath(args.output)}")
    print(f"  Total time: {total:.1f}s")
    print(f"{'='*60}")


def cli(argv: list[str] = None) -> None:
    set_affinity_to_all_cores()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.isfile(args.video):
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    cli()