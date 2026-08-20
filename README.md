<h1 align="center"><strong>3D Gaussian Splatting Reconstruction</strong></h1>
<p align="center">
  <!-- 项目标题 -->
  <img src="https://img.shields.io/badge/🌟_3D_Gaussian_Splatting-Reconstruction-FF6F00?style=flat-square&logo=github&logoColor=white" alt="Project">
  <br><br>
  <!-- 环境与核心依赖 -->
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <br><br>
  <!-- 核心功能模块 -->
  <img src="https://img.shields.io/badge/GUI-PySide6-8A2BE2?style=flat-square&logo=qt&logoColor=white" alt="GUI">
  <img src="https://img.shields.io/badge/SfM-COLMAP_|_OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="SfM">
  <img src="https://img.shields.io/badge/Rendering-Real_Time-FF4500?style=flat-square" alt="Rendering">
  <br><br>
  <!-- 兼容性与许可证 -->
  <img src="https://img.shields.io/badge/Platform-Windows_|_Linux_|_macOS-0078D4?style=flat-square&logo=windows&logoColor=white" alt="Platform">
  <img src="https://img.shields.io/badge/License-Apache_2.0-1E90FF?style=flat-square&logo=apache&logoColor=white" alt="License">
  <br><br>
  <!-- 社区动态（自动实时更新） -->
  <img src="https://img.shields.io/github/stars/Chi-Blaze-B/3D-Gaussian-Splatting-Reconstruction?style=flat-square&color=yellow&logo=github" alt="Stars">
  <img src="https://img.shields.io/github/forks/Chi-Blaze-B/3D-Gaussian-Splatting-Reconstruction?style=flat-square&color=blue&logo=github" alt="Forks">
  <img src="https://img.shields.io/github/issues/Chi-Blaze-B/3D-Gaussian-Splatting-Reconstruction?style=flat-square&color=red&logo=github" alt="Issues">
  <img src="https://img.shields.io/github/last-commit/Chi-Blaze-B/3D-Gaussian-Splatting-Reconstruction?style=flat-square&color=green&logo=github" alt="Last Commit">
</p>

纯 Python + PyTorch 实现的视频转 3D 高斯溅射（3DGS）完整工作流。输入一段视频，输出一个 `.ply` 文件，可用官方 3DGS 查看器（https://github.com/graphdeco-inria/gaussian-splatting ）浏览重建的三维场景。

- **纯 PyTorch 光栅化器**：完全基于 PyTorch 实现，无需编译 CUDA 扩展，支持 SH 0–3 阶球谐函数，排序式逐像素 splat 向量化（大幅加速），开箱即用。
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
2. 安装 PyTorch
   
   GPU 版本（CUDA 12.1）：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   
   CPU 版本：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
   
3. 安装 OpenCV（headless）
   ```bash
   pip install opencv-python-headless
   ```
   必须使用 headless 版本，避免与 PySide6 的 Qt 库冲突。
   
4. 安装其他依赖
   ```bash
   pip install numpy scipy PySide6 matplotlib psutil>=5.9.0
   ```
5. （可选）COLMAP 后端
   
   需自行安装 COLMAP（https://github.com/colmap/colmap ）。`colmap_poses.py` 查找顺序：项目内 `colmap-x64-windows-nocuda/bin/colmap.exe`（若有）→ 系统 PATH 中的 `colmap`。安装后将 colmap 加入 PATH 即可使用 `--pose-estimator colmap`。
   
6. 克隆本仓库
   ```bash
   git clone https://github.com/Chi-Blaze-B/3D-Gaussian-Splatting-Reconstruction
   ```

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
| `--amp` | 混合精度 fp16（需 CUDA + fp16 显卡，Ampere+ 可用 Tensor Core；光栅化器内部保持 fp32，无 Tensor Core 无收益） | `False` |
| `--pose-estimator` | 姿态估计后端：opencv / colmap | `opencv` |
| `--feature-type` | OpenCV 特征描述子：orb / sift（仅 opencv 后端生效） | `orb` |
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
| `poses.py` | 纯 OpenCV 增量式 SfM（ORB/SIFT），含 BA（深度障碍 + 基于场景尺度的相机边界 + 三角化尺度归一化 + PnP 尺度锚定）和点云过滤 |
| `colmap_poses.py` | COLMAP 封装，作为备选姿态估计后端 |
| `point_cloud.py` | 从稀疏点云初始化高斯参数（SH 0–3），自适应离群点剔除 |
| `gaussian.py` | 3DGS 核心：纯 PyTorch 光栅化器（排序式逐像素 splat 向量化，含梯度图连接保护）、LazyFrames 帧内存预加载、Trainer、密度控制、学习率调度 |
| `exporter.py` | 导出标准 PLY 格式，兼容官方查看器 |
| `gui.py` | PySide6 暗色主题图形界面，帧预览分页浏览（列数×行数随窗口宽高自适应，可查看全部帧） |
| `cli.py` | 命令行入口，集成完整流程 |

## 📈 训练细节

**损失函数**：`(1 - w_ssim) * L1 + w_ssim * SSIM`，SSIM 权重线性升温。

**密度控制**：每 `densify_every` 步根据梯度累积和尺度分裂/复制高斯，每 `prune_every` 步修剪低不透明度高斯，并自动限制总数量。分裂/复制阈值用**梯度分布的 p60 分位数自适应**（不依赖绝对量级，适配 loss 的 mean reduction；无绝对硬底，避免小梯度量级下永不触发）；复制时**保留原高斯 + 新增带微扰副本**（总数 = n + n_dup，与官方 densify_and_clone 一致）；修剪时保护梯度高于中位数的高斯；对需梯度的参数先 `.detach()` 再转 numpy，修剪后用 `.detach().clone()` 重建叶子张量。**优化器动量按行保留**：密度控制时新增高斯动量补零、幸存者按 mask 保留（官方语义），不再重建 Adam 清空全部动量，避免训练震荡。

**初始高斯稠密化**：若初始化后高斯数量少于 2000 个，自动对每个高斯做 8 倍扩增（加噪声），确保训练有足够的起始高斯数。

**学习率衰减**：指数衰减，每 `lr_decay_steps` 步乘以 `lr_decay_gamma`。

**SH 升温**：前 `sh_warmup_steps` 步逐渐提升 SH 阶数，从 0 逐步增至设定值。

**梯度裁剪**：全局梯度范数限制为 10.0，防止训练发散。

**光栅化器向量化**：像素级合成采用**排序式逐像素 splat**——把逐高斯 Python 内层循环重写为「展平覆盖像素对 → stable sort → 分段透射率 → scatter_add 归约」的纯张量算子，消除每颗高斯的 kernel launch 与 GPU→CPU 同步。实测 180x320+4000 高斯 forward 加速 **14.5x**、forward+backward **37.7x**；2160x3840+38665 高斯单帧约 1.7s（原为分钟级）。输出与旧实现逐元素一致（误差 < 1e-6）。

**光栅化器显存上界（分块）**：逐像素合成按深度有序高斯**分块**（每块至多 512 颗），块内覆盖网格只按块内最大包围盒物化——单颗大高斯（半径已 clamp 到 32）只撑大自己所在块，不再让全体陪跑。跨块透射率用**逐像素 log-transmittance 进位**（carry）累计，与整表算法在精确算术下等价（fp64 验证一致到 ~5e-13）。实测 256²、n=2000→4000 时峰值显存 **361→369MB 基本持平**（旧实现 1049→2099MB 翻倍）。附带收益：深堆叠像素上分块版比整表全局 cumsum 更准（整表大负数相减存在灾难性抵消，分块块内 cumsum 短）。

**混合精度（AMP）**：`--amp` 开启 fp16 混合精度，**仅 CUDA 生效**。cov3d 组合矩阵乘与 SSIM 卷积走 fp16（Ampere+ 可命中 Tensor Core），光栅化器内部保持 fp32（其 cumsum/scatter 不使用 Tensor Core，硬上 fp16 反而伤数值），配 GradScaler 动态损失缩放避免梯度下溢。无 Tensor Core 的显卡（如 GTX 10 系）开启无收益甚至略慢，**默认关闭**。

**帧内存预加载**：训练每 epoch 遍历全部帧，读盘 + PNG 解码是主要开销且伤硬盘。帧在训练开始前全部预解码为 **uint8 RGB** 缓存到内存（200 帧约 5GB，仅为 float32 的 1/4），训练期间**零磁盘读取**，访问时按需转 float32。实测 2 epoch × 200 帧从 57.3s（全读盘）降到 21.9s（内存缓存），**2.6x 加速且消除磁盘 IO**。

**梯度图连接保护**：光栅化器在边缘情况（高斯数为 0、全部高斯在相机后方或投影到画面外）下返回与计算图保持连接的零张量，避免 `loss.backward()` 因缺少 `grad_fn` 而崩溃。

**密度控制梯度处理**：密度控制（densify/prune）对需梯度的参数先 `.detach()` 再转 numpy，并用 `.detach().clone()` 重建修剪后的叶子张量，避免 `numpy()` 误用崩溃和修剪后参数被优化器静默跳过。

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
或通过 GUI 直接选择相同的工作目录，程序自动检测并恢复。恢复时，训练会**从上次中断的帧位置继续**（检查点记录 `last_frame_index`，配合有效位姿帧数计算），而非从头开始该轮次。检查点保存的**高斯基数与当前初始化数量不同也可恢复**（密度自适应会改变数量，恢复时直接采用检查点参数重建高斯与优化器）。

**位姿保存约定**：`poses.npy` 保存为**定长数组**（长度 = 帧数），缺失位姿的帧记为 `NaN` 行，恢复时按索引还原——中段存在未注册帧（COLMAP 常见）也不会错位。旧版 gap 压缩缓存无法还原中段对齐（自动退化为末尾补 None，不崩溃）。

**尺度缓存迁移**：`gaussian_params.npz` 带 `scale_domain` 标记；旧缓存（曾把 log 尺度存入 `scales` 键，导致初始化双重取 log、全部高斯塌缩为 1e-6）加载时自动迁移为线性尺度。

**SH 颜色约定**：已对齐官方 3DGS——DC 系数存 `(RGB-0.5)/C0`、求值补 `+0.5`、视角方向用世界系；导出的 `.ply` 可直接被官方查看器 / SuperSplat 加载。旧检查点的 SH 系数与新约定不兼容（预发布，加载处有注释）。

## ⚙️ 高级参数调优建议
--sampling-mode two-stage：适用于快速运动或视角变化剧烈的视频，能更好保留细节。

--sh-degree 3：获得最强的视角相关效果，但训练时间略增。

--max-gaussians：根据显存设置，推荐 300k~500k（8GB 显存可尝试 300k，24GB 可到 1M）。

--train-focal：若视频本身运动估计不准，开启此选项可改善几何一致性。

--random-background：能提升前景物体重建质量，但背景透明区域可能受干扰。

--enable-k1：仅当镜头畸变明显时开启，否则可能引入噪声。

## 📝 注意事项

帧采样数量：通常 100~200 帧效果较好，过少会导致欠约束，过多增加训练时间。

姿态估计：**长序列（≥60 帧）建议用 `--pose-estimator colmap`**——COLMAP 是工业级 SfM，实测点云质量与帧注册率远高于自研 ORB+EM（30 帧可见性 86.9% vs 37.5%）。COLMAP 默认用调优最优配置（`max_image_size=2400`、`sift_max_num_features=12000`、exhaustive matcher），60 帧实测 54/60 注册、27329 点。自研 ORB+EM 适合短序列（<60 帧），已修复 BA 深度障碍与尺度漂移问题。注意：COLMAP 对视频长序列的注册率低是 mapper 固有行为（只注册可稳定三角化的帧），但注册帧点云质量高，足以初始化高斯。mapper 可能把场景拆成多个子模型，程序会自动选择注册图像数最多的模型（修复：不再硬编码选模型 0）。对短序列或纹理不足场景，OpenCV 后端可换 `--feature-type sift`（SIFT 浮点描述子用 L2 距离匹配，比 ORB 更稳健但更慢）。

显存管理：如果训练中显存溢出，程序会自动修剪高斯并降低上限，并保存检查点。

CPU 亲和性：启动时会自动绑定所有逻辑核心，提升多核利用效率（通过 psutil）。

光栅化器：本项目使用纯 PyTorch 实现的光栅化器（排序式逐像素 splat 向量化，分块显存上界），无需编译任何 CUDA 扩展，开箱即用。

GPU 精度：`--amp` 混合精度仅对 Ampere+（RTX 30 系及以上）有 Tensor Core 收益；无 Tensor Core 的显卡（GTX 10 系等）请保持默认纯 FP32。

## 📄 许可证
本项目采用 Apache-2.0 许可证，欢迎自由使用和修改。

## 🙏 致谢
3D Gaussian Splatting 原始论文和开源代码。

OpenCV、PyTorch、SciPy、PySide6 等优秀开源库。

如有问题，欢迎提 Issue 或 PR。Happy Splatting!