# 3D Gaussian Splatting Reconstruction — 交接文档

纯 Python + PyTorch 实现的从视频到 3D Gaussian Splatting (.ply) 的完整工作流。输入一段视频，输出一个 .ply 文件，可以用官方 3DGS viewer 查看三维重建效果。

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
├── README.md              # 用户文档
└── HANDOFF.md             # 本文件（交接文档）
```

> **注意**：本项目采用**扁平脚本结构**（所有 `.py` 在同一目录），不包含 `__init__.py` 或 `src/` 布局。直接运行 `python gui.py` 或 `python cli.py` 即可，无需安装为包。

---

## 🔧 安装与环境配置

### 环境要求

| 组件 | 要求 |
| :--- | :--- |
| **Python** | 3.11（不要用 3.12，Windows 上 PyTorch DLL 加载有问题） |
| **包管理器** | conda（推荐） |
| **CUDA** | 12.1（仅当使用 CUDA 光栅化器时） |
| **编译器** | Visual Studio 2022 生成工具（仅当编译 CUDA 光栅化器时） |

### 安装步骤

```powershell
# 1. 创建 conda 环境
conda create -n gs python=3.11
conda activate gs

# 2. 安装 PyTorch（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 OpenCV（必须 headless 版本，避免 Qt 冲突）
pip install opencv-python-headless

# 4. 安装其余依赖
pip install numpy scipy PySide6 matplotlib
```

### （可选）启用 CUDA 光栅化器

> 强烈推荐，训练速度提升数十倍。

```powershell
# 1. 安装 CUDA Toolkit 12.1
conda install -c nvidia cuda-toolkit=12.1.0

# 2. 设置环境变量（每次新终端需执行）
$env:CUDA_HOME = "$env:CONDA_PREFIX\Library"
$env:PATH += ";$env:CONDA_PREFIX\Library\bin"

# 3. 安装光栅化器（需要 VS2022 生成工具）
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
```

### （可选）COLMAP 后端

项目已附带 `colmap-x64-windows-nocuda/`（CPU-only），无需额外安装。如需 GPU 加速，请自行下载 CUDA 版 COLMAP 并替换。

> **注意**：本项目的 `colmap_poses.py` 通过 `subprocess` 调用 COLMAP **命令行可执行文件**，而非 `pycolmap` Python 绑定。因此无需安装 `pycolmap` 包。

---

## 🚀 使用方式

### 方式一：GUI（推荐）

```powershell
python gui.py
```

GUI 提供：

- 视频路径选择、输出路径设置
- **采样模式**下拉框：均匀 / 智能（光流）/ 两阶段（视差+流+纹理）
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

### 方式二：命令行（CLI）

```powershell
# 基础用法（均匀采样，SH0，固定背景，固定焦距）
python cli.py --video input.mp4 --output output.ply

# 智能采样 + SH3 + 学习焦距
python cli.py --video input.mp4 --output out.ply --sampling-mode smart --sh-degree 3 --train-focal

# 两阶段采样 + COLMAP + 全部高级特性
python cli.py --video input.mp4 --output out.ply --sampling-mode two-stage --pose-estimator colmap \
    --sh-degree 3 --random-background --train-focal --max-gaussians 500000
```

#### 完整 CLI 参数表

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
| `--pose-estimator` | `opencv` | 姿态估算后端：`opencv`（自研 ORB+EM）或 `colmap` |
| `--focal-guess` | `None` | 初始焦距猜测（像素），不指定则自动估算 |
| `--resume-dir` | `None` | 从指定目录恢复训练（需包含 `training_state.pt`） |
| `--sh-degree` | `0` | SH 阶数（0=漫反射，3=完整视角相关） |
| `--sh-warmup-steps` | `1000` | SH 升温步数（逐渐增加可用阶数） |
| `--ssim-warmup-steps` | `500` | SSIM 升温步数（线性增加权重至 `--ssim-weight-max`） |
| `--ssim-weight-max` | `0.2` | SSIM 最大权重 |
| `--random-background` | `False` | 启用动态背景混合（黑白随机） |
| `--train-focal` | `False` | 启用焦距自校准（训练阶段优化 fx, fy） |
| `--enable-k1` | `False` | 训练时优化径向畸变系数 k1（实验性） |

> **设计理念**：CLI 默认使用最稳定配置（SH 0、均匀采样、固定焦距、固定背景），高级特性需用户显式启用。GUI 则默认开启常用高级特性并提供可视化开关，开箱即用。

---

## 🧩 各模块职责

| 模块 | 职责 | 关键特性 |
| :--- | :--- | :--- |
| `frames.py` | 视频帧提取 | 三种采样模式（均匀/智能/两阶段），光流方法可配置（Farneback/LK） |
| `poses.py` | 相机姿态估计 | ORB/SIFT 增量式 SfM，鲁棒初始化 + BA + 关键帧管理 + 重定位 |
| `colmap_poses.py` | COLMAP 后端封装 | 调用 COLMAP 命令行进行 SfM，失败时直接报错（不回退） |
| `point_cloud.py` | 高斯初始化 | 三角测量 → 颜色平均 → SH 系数初始化，自适应离群点过滤 |
| `gaussian.py` | 3DGS 核心 | 光栅化器（CUDA/手写），训练器，自适应密度控制，学习率调度 |
| `exporter.py` | PLY 导出 | 标准 3DGS .ply 格式，支持 SH Degree 0-3 |
| `gui.py` | 图形界面 | PySide6 暗色主题，损失曲线，实时日志，参数可视化配置 |
| `cli.py` | 命令行入口 | 精简参数，与 GUI 共享相同的后端逻辑 |

---

## 🔄 工作流

| 步骤 | 模块 | 说明 |
| :--- | :--- | :--- |
| 1. 帧提取 | `frames.py` | 按模式提取帧，支持缩放和光流采样 |
| 2. 姿态估计 | `poses.py` / `colmap_poses.py` | ORB+EM SfM 或 COLMAP |
| 3. 点云初始化 | `point_cloud.py` | 稀疏点云 → 高斯参数（位置、尺度、SH 系数） |
| 4. 训练 | `gaussian.py` | 光栅化 + 加权 L1/SSIM 损失 + 密度控制 |
| 5. 导出 | `exporter.py` | 写入 .ply 文件 |

---

## 🎯 关键设计决策

### 1. 姿态估计：自研 ORB+EM vs COLMAP

- **默认使用 `opencv`（自研）**：无外部依赖，适合短序列（< 200 帧），速度快。
- **COLMAP 作为备选**：适合长序列、大场景或需要畸变校正的场景。
- **不再自动回退**：若 COLMAP 失败，直接报错，提示用户检查输入或切换后端。

### 2. 采样策略：三种模式

| 模式 | 适用场景 | 原理 |
| :--- | :--- | :--- |
| `uniform` | 运动均匀的视频 | 固定间隔抽取 |
| `smart` | 运动分布不均的视频 | 基于光流变化率分配帧数 |
| `two-stage` | 复杂动态场景 | 粗姿态估计 → 综合评分（光流+纹理+视差）→ 精细采样 |

### 3. 训练优化

- **学习率调度**：默认指数衰减（每 1000 步 × 0.998）
- **自适应密度控制**：基于中位梯度动态调整分裂阈值
- **OOM 恢复**：自动降低高斯上限并修剪，而非切换 CPU
- **细粒度检查点**：每 500 步保存一次，避免 epoch 内崩溃丢失进度

### 4. 依赖管理

- **`requirements.txt` 不作为直接安装入口**：因为 PyTorch 需要指定 CUDA 版本，OpenCV 需要用 headless 版本。
- **分步安装指引**：在 `requirements.txt` 顶部注释中提供完整安装步骤。

---

## ⚠️ 已知限制与注意事项

| 限制 | 影响 | 缓解方案 |
| :--- | :--- | :--- |
| COLMAP 4.x 匹配 bug | 所有匹配对指向同一 `image_id` | 检测后报错，建议使用 ORB+EM 后端 |
| 姿态估计畸变支持 | `poses.py` 默认 k1=0 | 广角镜头建议输入前去畸变，或使用 COLMAP |
| SH Degree 3 显存消耗 | 比 SH0 多 20-30% 显存 | 低端显卡调低 SH 阶数 |
| ORB+EM 初始点云密度 | 通常 1~2 万点 | 自适应密度控制可在训练中补充 |
| 径向畸变 k1 训练 | 畸变不明显或视差不足时可能不收敛 | 关闭 `--enable-k1` |

---

## ❓ 常见问题排查

| 问题 | 解决方案 |
| :--- | :--- |
| DLL 加载失败 / Python 3.12 | 必须使用 Python 3.11 |
| OpenCV 递归加载错误 | `pip uninstall opencv-python -y && pip install opencv-python-headless` |
| 修改代码后不生效 | `rm -rf __pycache__` |
| COLMAP 只出少量位姿 | 检查 `workdir/colmap_work/` 日志。删除后重试。若仍失败，使用 `--pose-estimator opencv` |
| 训练发散 / loss 飙升 | GUI：关闭"动态背景"和"焦距自校准"；CLI：不指定 `--random-background` 和 `--train-focal`。降低高斯上限或减少帧数 |
| 训练初期损失下降慢 | 正常现象。初始点云覆盖有限，继续训练几十轮后会改善 |
| 如何启用 CUDA 光栅化器？ | 见安装章节。成功后 GUI 选择 "CUDA" 设备或 CLI 使用 `--device cuda` |
| GUI 和 CLI 高级特性差异 | GUI 默认开启 SH3、动态背景、焦距自校准、两阶段采样；CLI 默认全部关闭 |
| 姿态估计初始化失败 | 确保视频有足够的平移运动。尝试增加帧数（`--max-frames`）或调整采样策略 |

---

## 📌 后续维护建议

1. **定期更新 `colmap_poses.py`**：COLMAP 版本更新可能带来 API 变化（命令行参数），需同步测试。
2. **考虑添加 CI/CD**：用一个小型测试数据集（如 `camera-trap` 序列）验证各模块的兼容性。
3. **LPIPS 支持**：`gaussian.py` 中已预留 LPIPS 支持（`torchmetrics`），但未暴露给 GUI/CLI。如需启用，需添加参数控制。
4. **多 GPU 支持**：当前仅单卡训练。若需多卡，需修改 `gaussian.py` 中的数据并行逻辑。
5. **文档版本同步**：修改核心逻辑时，需同步更新 `README.md` 和本 `HANDOFF.md`。

---

## 📄 License

本项目基于 MIT License 开源。