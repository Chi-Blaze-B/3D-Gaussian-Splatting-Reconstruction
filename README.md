# 3D Gaussian Splatting Reconstruction

纯 Python + PyTorch 实现的从视频到 3D Gaussian Splatting (.ply) 的完整工作流。输入一段视频，输出一个 .ply 文件，可以用官方 3DGS viewer 查看三维重建效果。

---

## ✨ 亮点

- **鲁棒的姿态估计**：基于 ORB（或 SIFT）特征的增量式 SfM 流程，包含鲁棒初始化、局部地图跟踪、关键帧管理、重定位和全局 BA，支持自动焦距校准（训练阶段）。
- **官方 CUDA 光栅化器集成**：训练速度比纯 PyTorch 实现快数十倍，支持 SH Degree 0-3。
- **SH Degree 3 视角相关颜色**：完整球谐函数支持，金属反光、高光等材质表现显著提升。
- **自适应密度控制**：训练中自动分裂/复制/修剪高斯，内置预算控制防显存爆炸。
- **智能帧采样（三种模式）**：
  - `uniform`：均匀采样
  - `smart`：基于光流的运动自适应采样
  - `two-stage`：基于视差、光流和纹理的综合评分采样（适用于复杂动态场景）
- **断点续训**：保存完整训练状态（参数+优化器+密度控制+焦距+k1+SH阶数），随时恢复。
- **暗色主题 GUI**：PySide6 实现，实时损失曲线、帧预览、进度日志、所有高级参数可视化配置。
- **轻量级无 COLMAP 依赖**（可选）：默认使用自研 ORB+EM SfM，也可选择 COLMAP 作为后端。

---

## 📁 项目结构

```
3D Gaussian Splatting Reconstruction/
├── gui.py                 # PySide6 GUI（暗色主题 + 损失曲线 + 高级参数配置）
├── cli.py                 # 命令行接口（精简参数，支持三种采样模式）
├── frames.py              # 视频帧提取（均匀/智能/两阶段采样，光流可选）
├── poses.py               # ORB/SIFT 增量式 SfM（无 COLMAP 依赖）
├── colmap_poses.py        # COLMAP 封装（可选后端）
├── point_cloud.py         # 稀疏重建 & 高斯初始化（SH Degree 3，自适应离群点过滤）
├── gaussian.py            # 3DGS 核心：CUDA/手写光栅化器 + Trainer + 密度控制 + 学习率调度
├── exporter.py            # PLY 导出（完整 SH 系数）
├── requirements.txt       # 依赖清单（含分步安装指引）
├── LICENSE                # MIT 许可证
└── README.md              # 本文件
---

## 🔧 安装

### 环境要求

- **Python 3.11** — 必须 3.11，不要用 3.12（Windows 上 PyTorch DLL 加载问题）。
- **conda 环境** — 推荐使用 conda 管理依赖。
- **CUDA Toolkit 12.1**（若使用 CUDA 光栅化器）。
- **Visual Studio 2022 生成工具**（若需编译 CUDA 扩展）。

### 基础安装（CPU / 手写光栅化器）

```powershell
# 1. 创建 conda 环境
conda create -n gs python=3.11
conda activate gs

# 2. 安装 PyTorch（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 OpenCV（必须 headless 版本，避免 Qt 冲突）
pip install opencv-python-headless

# 4. 安装其余核心依赖
pip install numpy scipy PySide6 matplotlib
```

重要：必须用 opencv-python-headless，不要用 conda 装的 opencv（有递归加载 bug），也不要装 opencv-python（会与 PySide6 的 Qt 冲突）。

提示：requirements.txt 仅作为依赖参考，请勿直接执行 `pip install -r requirements.txt`，因为 PyTorch 和 OpenCV 有特殊的版本要求。请严格按照上述步骤安装。

### 启用 CUDA 光栅化器（可选，强烈推荐）

安装官方 CUDA 光栅化器以获得数十倍训练加速：

1. 安装 Visual Studio 2022 生成工具（勾选"使用 C++ 的桌面开发"）。
2. 安装 CUDA Toolkit 12.1：

```powershell
conda install -c nvidia cuda-toolkit=12.1.0
```

设置环境变量（每次新终端需执行）：

```powershell
$env:CUDA_HOME = "$env:CONDA_PREFIX\Library"
$env:PATH += ";$env:CONDA_PREFIX\Library\bin"
```

安装光栅化器（可能需要代理）：

```powershell
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
```

COLMAP 已附带 `colmap-x64-windows-nocuda/`（CPU-only），无需额外安装。

---

## 🚀 用法

### 方式 1：GUI（推荐）

```powershell
python gui.py
```

GUI 提供：

- 视频路径选择、输出路径设置
- 采样模式下拉框（均匀 / 智能 / 两阶段）
- 帧率、缩放比例、帧数范围配置
- 高斯上限设置（默认 30 万）
- SH 阶数下拉框（0-3，默认 3）
- SH 升温步数、SSIM 升温步数、SSIM 最大权重
- 动态背景、焦距自校准、径向畸变 k1 开关
- 训练轮次、计算设备（自动 / CPU / CUDA）
- 姿态估算后端选择（ORB+EM / COLMAP）
- 进度条、实时日志、停止按钮
- 损失曲线可视化（当前轮次帧损失 + 历史轮次平均损失）
- 帧预览缩略图网格
- 自动从检查点恢复训练

### 方式 2：命令行（CLI）

```powershell
# 基础用法（均匀采样，SH0，固定背景，固定焦距）
python cli.py --video input.mp4 --output output.ply

# 智能采样 + SH3 + 学习焦距
python cli.py --video input.mp4 --output out.ply --sampling-mode smart --sh-degree 3 --train-focal

# 两阶段采样 + COLMAP + 全部高级特性
python cli.py --video input.mp4 --output out.ply --sampling-mode two-stage --pose-estimator colmap \
    --sh-degree 3 --random-background --train-focal --max-gaussians 500000
```

#### 完整参数表

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--video` | **必填** | 输入视频路径 |
| `--output` | `output.ply` | 输出 PLY 路径 |
| `--workdir` | `./workdir` | 中间文件存储目录 |
| `--fps` | `15.0` | 采样帧率（仅均匀模式使用） |
| `--scale` | `0.5` | 分辨率缩放比例 |
| `--min-frames` | `30` | 最少提取帧数 |
| `--max-frames` | `200` | 最多提取帧数 |
| `--sampling-mode` | `uniform` | 采样策略：`uniform`（均匀）、`smart`（光流）、`two-stage`（视差+流+纹理） |
| `--num-epochs` | `3000` | 训练轮次 |
| `--device` | `auto` | 计算设备：`auto`（自动）、`cpu`、`cuda` |
| `--eval-every` | `500` | 评估间隔（每 N 轮打印日志） |
| `--max-gaussians` | `300000` | 高斯数量上限 |
| `--sh-degree` | `0` | SH 阶数（0=漫反射，3=完整视角相关） |
| `--sh-warmup-steps` | `1000` | SH 升温步数（逐渐增加可用阶数） |
| `--ssim-warmup-steps` | `500` | SSIM 升温步数（线性增加权重至 `--ssim-weight-max`） |
| `--ssim-weight-max` | `0.2` | SSIM 最大权重 |
| `--random-background` | `False` | 启用动态背景混合（黑白随机） |
| `--train-focal` | `False` | 启用焦距自校准（训练阶段优化 fx, fy） |
| `--enable-k1` | `False` | 训练时优化径向畸变系数 k1（实验性） |
| `--pose-estimator` | `opencv` | 姿态估算后端：`opencv`（自研 ORB+EM）或 `colmap` |
| `--focal-guess` | `None` | 初始焦距猜测（像素），不指定则自动估算 |
| `--resume-dir` | `None` | 从指定目录恢复训练（需包含 `training_state.pt`） |

设计理念：CLI 默认使用最稳定配置（SH 0、均匀采样、固定焦距、固定背景），高级特性需用户显式启用。GUI 则默认开启常用高级特性并提供可视化开关，开箱即用。

---

## 🔄 断点续训

程序自动检测中间文件。若存在 `workdir/training_state.pt`，GUI 和 CLI 会从中断处恢复完整训练状态（包括高斯参数、Adam 动量、自适应密度控制信号、焦距、k1、SH 阶数、升温步数等），继续训练。CLI 可通过 `--resume-dir` 指定上次训练的 workdir。

---

## 📋 工作流步骤

| 步骤 | 模块 | 说明 |
| :--- | :--- | :--- |
| 1. 帧提取 | `frames.py` | 按模式提取帧（均匀/光流智能/两阶段），支持缩放 |
| 2. 姿态估计 | `poses.py` / `colmap_poses.py` | ORB+EM SfM（鲁棒初始化 + 关键帧管理 + 重定位 + BA）或 COLMAP |
| 3. 点云初始化 | `point_cloud.py` | 三角测量 → 颜色平均 → 高斯参数（SH Degree 3，自适应离群点过滤） |
| 4. 训练 | `gaussian.py` | CUDA/手写光栅化 + 加权 L1 + SSIM + Adam + 密度控制 + 预算控制 + 可选 k1 优化 |
| 5. 导出 | exporter.py | 标准 3DGS .**ply** 格式（完整 SH 系数） |

---

## 📦 输出

生成的 `.ply` 文件兼容 3DGS 官方查看器：

```powershell
pip install plyfile
python -c "from plyfile import PlyData; p=PlyData.read('output.ply'); print(p['vertex'].data.shape)"
```

或在浏览器中使用官方 web viewer。

---

## 🧩 架构细节

### 光栅化器

支持两种模式，自动切换：

1. **官方 CUDA 光栅化器**（diff-gaussian-rasterization）：GPU 原生实现，训练速度提升数十倍，支持 SH Degree 0-3。
2. **手写 PyTorch 光栅化器**（DifferentiableRasterizer）：纯 Python/torch 实现，仅支持 SH0（漫反射），作为 CUDA 不可用时的 fallback。

### 损失函数

- **加权 L1**：在图像梯度大的区域给更高权重（上限 1.2x），提升边缘质量。
- **SSIM**：结构相似性损失，捕捉感知质量。
- **组合**：`loss = (1 - w) * l1_loss + w * ssim_loss`，其中 `w` 从 0 线性升至 `--ssim-weight-max`（由 `ssim_warmup_steps` 控制）。
- **LPIPS**（可选）：若安装 `torchmetrics`，可在训练后期加入感知损失，进一步提升视觉质量（代码中已预留，但 GUI/CLI 默认未暴露）。

### 相机姿态估计（poses.py）

基于 ORB（或 SIFT）特征的增量式 SfM 流程：

- **鲁棒初始化**：自动检测平移不足的帧，延迟到合适帧对初始化。
- **局部地图跟踪**：通过与最近关键帧的 3D-2D 匹配 + PnP 精化位姿。
- **关键帧管理**：基于视角变化、平移距离、共视比综合决策；自动剔除冗余关键帧。
- **重定位**：描述子评分 + 几何验证（PnP RANSAC）二级筛选。
- **三角化过滤**：深度检验、视差角检验、重投影误差检验多重过滤。
- **Bundle Adjustment**：局部/全局 BA（优化位姿+内参+点云），带 Huber 损失和平滑深度屏障。

优势：无需外部 SfM 库，完全自包含；支持自动焦距校准（单参数 f，fx=fy）；适合短序列、纹理丰富的视频。

### COLMAP（备选）

[COLMAP](https://github.com/colmap/colmap) 是一款通用的 Structure-from-Motion (SfM) 和多视图立体 (MVS) 工具箱，适合处理长序列、大场景或需要精确畸变校正的任务。

本项目通过 `colmap_poses.py` 封装 COLMAP 命令行工具，提供以下功能：

- SIFT 特征提取（`max_image_size=1024`）
- exhaustive matching（带 `distinct_cols` 验证）
- Mapper（SfM reconstruction）

> **注意**：本项目的 `colmap_poses.py` 通过 `subprocess` 调用 COLMAP **命令行可执行文件**，而非 `pycolmap` Python 绑定。因此无需安装 `pycolmap` 包，但需要自行下载 COLMAP 二进制文件。
>
> 项目原本附带的 `colmap-x64-windows-nocuda/`（CPU-only）已从仓库中移除（体积过大），如需使用 COLMAP 后端，请从 [COLMAP 官方仓库](https://github.com/colmap/colmap) 或 [官方发布页](https://github.com/colmap/colmap/releases) 下载对应平台的版本。
>
> COLMAP 后端失败时会直接报错，不再自动回退，请检查输入质量或切换至 `--pose-estimator opencv`。

### SH 高阶颜色

SH Degree 3（16 个基函数）提供视角相关的高光、反射效果。GUI 默认开启（可调），CLI 需显式指定 `--sh-degree 3`。初期训练会主要使用低阶系数，高阶系数从零开始逐渐学习（`sh_warmup_steps` 控制升温速度）。

### 自适应密度控制

默认启用。每 100 步分裂/复制，每 1000 步修剪。内置预算控制：高斯数超过上限时自动紧急修剪不透明度最低的高斯（默认上限 30 万，GUI/CLI 可配置）。

### 检查点

每 CHECKPOINT_INTERVAL_STEPS（默认 500 步）保存一次完整训练状态到 `workdir/training_state.pt`。

包含：参数、优化器动量、密度控制信号、焦距、k1、SH 阶数、升温步数等。

支持跨设备迁移（GPU→CPU 恢复）。

保存时机：定期保存、用户取消、loss 发散、OOM 回退前。

### 学习率调度

| 参数 | 学习率 | 说明 |
| :--- | :--- | :--- |
| positions | 1.6e-4 | 原论文值 |
| log_scales | 5.0e-3 | 原论文值 |
| rotations | 1.0e-3 | 原论文值 |
| opacities | 5.0e-2 | 原论文值 |
| sh_coeffs | 2.5e-3 | 原论文值 |
| focal（可选） | 1.0e-5 | 焦距微调 |
| k1（可选） | 1.0e-5 | 径向畸变微调 |

默认启用指数衰减（每 1000 步乘以 0.998），可通过修改 `gaussian.py` 中的 `LR_DECAY_STEPS` 和 `LR_DECAY_GAMMA` 调整。

梯度裁剪全局范数阈值 10.0。

### 智能帧采样（三种模式）

| 模式 | 说明 |
| :--- | :--- |
| uniform | 均匀抽取固定数量帧 |
| smart | 基于光流变化率分配帧数，运动剧烈处多采 |
| two-stage | 粗采样约 30~50 帧 → 快速姿态估计 → 综合评分（0.4×光流 + 0.3×纹理 + 0.3×视差）→ 采样最终帧集 |

GUI 默认使用 two-stage 模式（可通过下拉框切换），CLI 需通过 `--sampling-mode two-stage` 指定。

---

## ⚠️ 已知限制与注意事项

| 限制 | 说明 |
| :--- | :--- |
| COLMAP 4.x 匹配 bug | 所有匹配对指向同一 image_id，已提交 issue #4562。当前会检测并报错，建议使用 ORB+EM 后端 |
| 姿态估计畸变支持 | 当前 `poses.py` 默认 k1=0，如需广角镜头畸变校正，建议在输入前预先去畸变，或使用 COLMAP（支持畸变模型） |
| SH Degree 3 显存消耗 | 16 通道系数比 3 通道多约 20-30% 显存，低端显卡可调低 SH 阶数 |
| ORB+EM 初始点云密度 | 通常生成 1~2 万点，但自适应密度控制可在训练中补充细节 |
| CUDA 光栅化器安装 | 需要 Visual Studio 2022 生成工具和 CUDA Toolkit 12.1 |
| 训练初期收敛慢 | 初始点云覆盖有限，前几轮损失可能停滞。继续训练即可 |
| 径向畸变 k1 训练 | 在畸变不明显或视差不足的场景中，k1 可能无法正确收敛，建议关闭相关选项 |
| COLMAP 后端失败 | GUI/CLI 中若 COLMAP 失败会直接报错，不再自动回退，需检查 COLMAP 安装或输入质量 |

---

## ❓ 常见问题

| 问题 | 解决方案 |
| :--- | :--- |
| DLL 加载失败 / Python 3.12 问题 | 必须使用 Python 3.11 |
| OpenCV 递归加载错误 | `pip uninstall opencv-python -y && pip install opencv-python-headless` |
| 修改代码后不生效 | `rm -rf __pycache__` |
| COLMAP 只出少量位姿 | 检查 `workdir/colmap_work/` 下的日志。删除 `colmap_work/` 目录后重试。若仍失败，使用 `--pose-estimator opencv` |
| 训练发散 / loss 飙升 | GUI：关闭"动态背景"和"焦距自校准"，或降低 SH 阶数。<br>CLI：使用 `--sh-degree 0`（不指定 `--random-background` 和 `--train-focal` 即为关闭）。降低高斯上限或减少帧数 |
| 训练初期大量帧损失相同 / 平均损失下降极慢 | 正常现象。初始点云覆盖有限，许多帧暂时看不到高斯。继续训练几十轮，高斯扩散后会明显下降 |
| 如何启用 CUDA 光栅化器？ | 见上文安装章节。成功后 GUI 选择 "CUDA" 设备或 CLI 使用 `--device cuda` |
| GUI 和 CLI 的高级特性有何区别？ | GUI：默认开启 SH Degree 3、动态背景、焦距自校准、两阶段采样，并可通过控件随时调整。<br>CLI：默认全部关闭（最稳定配置），需显式指定参数启用 |
| 径向畸变 k1 相关选项 | 若视频有明显桶形/枕形畸变，可尝试开启 `--enable-k1`（训练阶段优化）；否则建议保持关闭以避免不稳定 |
| 姿态估计初始化失败 | 确保视频中有足够的平移运动（纯旋转场景可能无法初始化），尝试增加帧数（`--max-frames`）或调整采样策略 |

---

## 📄 License

本项目代码基于 MIT License 开源。