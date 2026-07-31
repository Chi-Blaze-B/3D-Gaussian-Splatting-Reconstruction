# 3D Gaussian Splatting Reconstruction
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square&logo=apache&logoColor=white" alt="License">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/GUI-PySide6-8A2BE2?style=flat-square&logo=qt&logoColor=white" alt="GUI">
  <img src="https://img.shields.io/badge/SfM-OpenCV%20%7C%20COLMAP-brightgreen?style=flat-square" alt="SfM">
  <img src="https://img.shields.io/github/stars/Chi-Blaze-B/3D-Gaussian-Splatting-Reconstruction?style=flat-square&color=yellow" alt="Stars">
  <img src="https://img.shields.io/github/issues/Chi-Blaze-B/3D-Gaussian-Splatting-Reconstruction?style=flat-square" alt="Issues">
</p>

纯 Python + PyTorch 实现的视频转 3D 高斯溅射（3DGS）完整工作流。输入一段视频，输出一个 `.ply` 文件，可用官方 3DGS 查看器（https://github.com/graphdeco-inria/gaussian-splatting）浏览重建的三维场景。

- **纯 PyTorch 光栅化器**：完全基于 PyTorch 实现，无需编译 CUDA 扩展，支持 SH 0–3 阶球谐函数，开箱即用。
- **鲁棒姿态估计**：内置 ORB/SIFT 增量式 SfM（无 COLMAP 依赖），也可选用 COLMAP 作为后端。
- **智能采样**：均匀、光流驱动、两阶段（视差+光流+纹理）三种帧采样策略。
- **自适应密度控制**：训练中自动分裂/复制/修剪高斯，支持显存预算控制。
- **暗色主题 GUI**：基于 PySide6，实时损失曲线、帧预览、日志输出，可配置所有高级参数。
- **断点续训**：保存完整训练状态（参数、优化器、密度控制器、焦距、畸变系数、最佳损失等），随时恢复。

---

## 📦 安装

### 环境要求

- Python 3.11（推荐）
- CUDA 12.1（可选，CPU 也可运行）

### 步骤

1. **创建 Conda 环境**（可选）
   ```bash
   conda create -n gs python=3.11
   conda activate gs
   ```
   安装 PyTorch
   
   GPU 版本（CUDA 12.1）：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   
   CPU 版本：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
   
   安装 OpenCV（headless）
   ```bash
   pip install opencv-python-headless
   ```
   必须使用 headless 版本，避免与 PySide6 的 Qt 库冲突。
   
   安装其他依赖
   ```bash
   pip install numpy scipy PySide6 matplotlib psutil>=5.9.0
   ```
   （可选）COLMAP 后端
   
需自行安装 COLMAP，并确保 colmap 命令在 PATH 中可用。

## 🚀 使用方式

1. 命令行接口（CLI）

基本用法：
```bash
python cli.py --video input.mp4 --output output.ply
```

常用参数表：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 输入视频路径 | 必填 |
| `--output` | 输出 PLY 文件路径 | `output.ply` |
| `--workdir` | 工作目录（存放中间文件） | `./workdir` |
| `--fps` | 采样帧率（均匀采样） | `15.0` |
| `--scale` | 画面缩放（0~1） | `0.5` |
| `--min-frames / --max-frames` | 最少/最多提取帧数 | `30 / 200` |
| `--sampling-mode` | 采样模式：uniform / smart / two-stage | `uniform` |
| `--num-epochs` | 训练轮数 | `3000` |
| `--device` | 计算设备：auto / cpu / cuda | `auto` |
| `--max-gaussians` | 最大高斯数量 | `300000` |
| `--sh-degree` | 球谐阶数（0~3） | `0` |
| `--sh-warmup-steps` | SH 阶数升温步数 | `1000` |
| `--ssim-warmup-steps` | SSIM 权重升温步数 | `500` |
| `--ssim-weight-max` | SSIM 最大权重 | `0.2` |
| `--random-background` | 训练时随机黑白背景 | `False` |
| `--train-focal` | 训练中微调焦距 | `False` |
| `--enable-k1` | 训练径向畸变系数 k1 | `False` |
| `--pose-estimator` | 姿态估计后端：opencv / colmap | `opencv` |
| `--focal-guess` | 初始焦距猜测（像素） | `None` |
| `--resume-dir` | 从该工作目录恢复训练 | `None` |
| `--eval-every` | 每 N 轮打印一次日志 | `500` |

示例：
```bash
# 智能采样 + SH3 + 焦距自校准
python cli.py --video input.mp4 --output out.ply --sampling-mode smart --sh-degree 3 --train-focal

# 两阶段采样 + COLMAP + 完整功能
python cli.py --video input.mp4 --output out.ply --sampling-mode two-stage --pose-estimator colmap \
    --sh-degree 3 --random-background --train-focal --max-gaussians 500000
```

2. 图形界面（GUI）

启动 GUI：
```bash
python gui.py
```

图形界面提供完整的参数配置面板，操作直观：

- 选择视频、输出路径、工作目录
- 调整采样策略、训练轮次、高斯预算等
- 实时预览帧缩略图
- 训练过程中显示损失曲线（帧级和轮次级）
- 帧预览分页浏览：每页容量随窗口宽高自适应（列数 × 行数，默认约 24 帧），可翻页查看全部提取帧
- 日志输出窗口详细记录每步进度
- 支持中断训练并自动保存检查点

## 🧩 核心模块说明

| 模块 | 功能 |
|------|------|
| `frames.py` | 视频帧提取，支持三种采样策略，光流采用 Farneback/LK |
| `poses.py` | 纯 OpenCV 增量式 SfM（ORB/SIFT），含 BA 和点云过滤 |
| `colmap_poses.py` | COLMAP 封装，作为备选姿态估计后端 |
| `point_cloud.py` | 从稀疏点云初始化高斯参数（SH 0–3），自适应离群点剔除 |
| `gaussian.py` | 3DGS 核心：纯 PyTorch 光栅化器（含梯度图连接保护）、Trainer、密度控制、学习率调度 |
| `exporter.py` | 导出标准 PLY 格式，兼容官方查看器 |
| `gui.py` | PySide6 暗色主题图形界面，帧预览分页浏览（列数×行数随窗口宽高自适应，可查看全部帧） |
| `cli.py` | 命令行入口，集成完整流程 |

## 📈 训练细节

**损失函数**：`(1 - w_ssim) * L1 + w_ssim * SSIM`，SSIM 权重线性升温。

**密度控制**：每 `densify_every` 步根据梯度累积和尺度分裂/复制高斯，每 `prune_every` 步修剪低不透明度高斯，并自动限制总数量。

**初始高斯稠密化**：若初始化后高斯数量少于 2000 个，自动对每个高斯做 8 倍扩增（加噪声），确保训练有足够的起始高斯数。

**学习率衰减**：指数衰减，每 `lr_decay_steps` 步乘以 `lr_decay_gamma`。

**SH 升温**：前 `sh_warmup_steps` 步逐渐提升 SH 阶数，从 0 逐步增至设定值。

**梯度裁剪**：全局梯度范数限制为 10.0，防止训练发散。

**梯度图连接保护**：光栅化器在边缘情况（高斯数为 0、全部高斯在相机后方或投影到画面外）下返回与计算图保持连接的零张量，避免 `loss.backward()` 因缺少 `grad_fn` 而崩溃。

**Loss 发散保护**：若单步 loss 超过阈值（默认 1.0），自动保存检查点并中断训练，避免无限发散。

**焦距自校准**：若启用 `--train-focal`，在训练中优化焦距参数（fx, fy），适应实际内参。

**径向畸变**：实验性支持优化一阶径向畸变系数 k1（仅 `--enable-k1`）。

## 💾 断点续训

所有中间结果和训练状态保存在 `--workdir` 指定目录下：

| 文件 | 内容 |
|------|------|
| `frame_paths.txt` | 帧路径列表 |
| `intrinsics.npy`、`poses.npy`、`sparse_points.npy` | 姿态和稀疏点云 |
| `gaussian_params.npz` | 初始化后的高斯参数 |
| `training_state.pt` | 完整训练状态（参数、优化器、密度控制器、焦距、k1、SH 阶数等） |
| `best_training_state.pt` | 历史最优（最低 loss）训练状态，始终保留不覆盖 |

**最佳检查点保护**：`best_training_state.pt` 始终保留训练过程中的最优模型，与常规检查点分开保存，不会因后续训练震荡而被覆盖。

**恢复训练**：
```bash
python cli.py --video input.mp4 --resume-dir ./workdir --output restored.ply
```
或通过 GUI 直接选择相同的工作目录，程序自动检测并恢复。恢复时，训练会从上次中断的帧位置继续，而非从头开始该轮次。

## ⚙️ 高级参数调优建议
--sampling-mode two-stage：适用于快速运动或视角变化剧烈的视频，能更好保留细节。

--sh-degree 3：获得最强的视角相关效果，但训练时间略增。

--max-gaussians：根据显存设置，推荐 300k~500k（8GB 显存可尝试 300k，24GB 可到 1M）。

--train-focal：若视频本身运动估计不准，开启此选项可改善几何一致性。

--random-background：能提升前景物体重建质量，但背景透明区域可能受干扰。

--enable-k1：仅当镜头畸变明显时开启，否则可能引入噪声。

## 📝 注意事项

帧采样数量：通常 100~200 帧效果较好，过少会导致欠约束，过多增加训练时间。

姿态估计：若 ORB 方法失败，可尝试 --pose-estimator colmap（需安装 COLMAP）。COLMAP 更鲁棒但速度较慢。

显存管理：如果训练中显存溢出，程序会自动修剪高斯并降低上限，并保存检查点。

CPU 亲和性：启动时会自动绑定所有逻辑核心，提升多核利用效率（通过 psutil）。

光栅化器：本项目使用纯 PyTorch 实现的光栅化器，无需编译任何 CUDA 扩展，开箱即用。

## 📄 许可证
本项目采用 Apache-2.0 许可证，欢迎自由使用和修改。

## 🙏 致谢
3D Gaussian Splatting 原始论文和开源代码。

OpenCV、PyTorch、SciPy、PySide6 等优秀开源库。

如有问题，欢迎提 Issue 或 PR。Happy Splatting!