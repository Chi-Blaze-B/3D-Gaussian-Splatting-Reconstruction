<h1 align="center"><strong>3D Gaussian Splatting Reconstruction</strong></h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/GUI-PySide6-8A2BE2?style=flat-square&logo=qt&logoColor=white" alt="GUI">
  <img src="https://img.shields.io/badge/License-Apache_2.0-1E90FF?style=flat-square&logo=apache&logoColor=white" alt="License">
</p>

基于 Python 的视频转 3D 高斯泼溅（3DGS）工作流。输入一段视频，输出一个 `.ply` 文件，可用官方 3DGS 查看器浏览重建的三维场景。

**核心光栅化器完全基于 PyTorch 实现**，无需编译 CUDA 扩展，支持 SH 0–3 阶球谐函数，排序式逐像素 splat 向量化。整体流程还依赖 OpenCV、SciPy、PySide6、psutil 等库。

- **纯 PyTorch 光栅化器**：无需编译 CUDA 扩展，支持 SH 0–3 阶，开箱即用。
- **鲁棒姿态估计**：内置 ORB/SIFT 增量式 SfM，也可选用 COLMAP 后端。
- **智能采样**：均匀、光流驱动（smart）、两阶段（视差+光流+清晰度）三种策略。
- **自适应密度控制**：训练中自动分裂/复制/修剪高斯。
- **硬件自适应配置**：按实际训练设备自动推导分块、半径与高斯数量上限。
- **暗色主题 GUI**：实时损失曲线、帧预览、日志输出。
- **断点续训**：保存完整训练状态，恢复时从上次中断帧继续。

---

## 📦 安装

### 环境要求

- Python 3.11（推荐）
- CUDA 11.8+（可选，CPU 也可运行）

### 步骤

1. **创建 Conda 环境**（可选）
   ```bash
   conda create -n gs python=3.11
   conda activate gs
   ```
2. **安装 PyTorch**
   ```bash
   # GPU 示例（CUDA 12.1）
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   # CPU
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
3. **安装 OpenCV（headless）**
   ```bash
   pip install opencv-python-headless
   ```
   必须使用 headless 版本，避免与 PySide6 的 Qt 库冲突。
4. **安装其他依赖**
   ```bash
   pip install numpy scipy PySide6 matplotlib "psutil>=5.9.0"
   ```
5. **（可选）COLMAP 后端**

   需自行安装 COLMAP 并加入 PATH。`colmap_poses.py` 查找顺序：项目内 `colmap-x64-windows-nocuda/bin/colmap.exe` → 系统 PATH 中的 `colmap`。
6. **克隆仓库**
   ```bash
   git clone https://github.com/Chi-Blaze-B/3D-Gaussian-Splatting-Reconstruction
   ```
## 🚀 使用方式

### 1. 命令行接口（CLI）

基本用法：
```bash
python cli.py --video input.mp4 --output output.ply
```

#### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 输入视频路径 | 必填 |
| `--output` | 输出 PLY 文件路径 | `output.ply` |
| `--workdir` | 工作目录 | `./workdir` |
| `--fps` | 采样帧率 | `15.0` |
| `--scale` | 画面缩放（0~1） | `0.5` |
| `--min-frames / --max-frames` | 最少/最多提取帧数 | `30 / 200` |
| `--sampling-mode` | uniform / smart / two-stage | `uniform` |
| `--num-epochs` | 训练轮数 | `3000` |
| `--device` | auto / cpu / cuda | `auto` |
| `--max-gaussians` | 高斯上限；不指定则按设备自动选择 | `None` |
| `--sh-degree` | 球谐阶数（0~3） | `0` |
| `--sh-warmup-steps` | SH 升温步数 | `1000` |
| `--ssim-warmup-steps` | SSIM 升温步数 | `500` |
| `--ssim-weight-max` | SSIM 最大权重 | `0.2` |
| `--random-background` | 随机黑白背景 | `False` |
| `--train-focal` | 训练中微调焦距 | `False` |
| `--amp` | 混合精度 fp16（需 CUDA + Ampere+） | `False` |
| `--pose-estimator` | opencv / colmap | `opencv` |
| `--feature-type` | orb / sift（仅 opencv 后端） | `orb` |
| `--focal-guess` | 初始焦距猜测（像素） | `None` |
| `--resume-dir` | 从工作目录续训 | `None` |
| `--eval-every` | 每 N 轮打印日志 | `500` |
| `--show-config` | 打印硬件自适应配置后退出 | `False` |

#### 示例

```bash
# 基础用法
python cli.py --video input.mp4 --output out.ply

# 智能采样 + SH3 + 焦距自校准
python cli.py --video input.mp4 --output out.ply --sampling-mode smart --sh-degree 3 --train-focal

# 两阶段采样 + COLMAP
python cli.py --video input.mp4 --output out.ply --sampling-mode two-stage --pose-estimator colmap \
    --sh-degree 3 --random-background --train-focal --max-gaussians 500000

# 查看当前设备自适应渲染配置
python cli.py --video input.mp4 --device cuda --show-config
```
### 2. 图形界面（GUI）

启动 GUI：
```bash
python gui.py
```

GUI 提供：

- 视频、输出路径、工作目录选择
- 采样策略、训练轮次、高斯预算等参数配置
- 高斯上限随设备（CPU/CUDA）自动刷新范围与默认值
- 帧缩略图预览、分页浏览
- 帧级和轮次级损失曲线
- 日志输出、中断训练并保存检查点

**注意**：GUI 暂未提供 `--focal-guess`、`--resume-dir`；`评估间隔`控件未生效。GUI 默认值与 CLI 有差异，以界面为准。

## 🧩 核心模块

| 模块 | 功能 |
|------|------|
| `frames.py` | 视频帧提取，支持 uniform / smart / two-stage 采样 |
| `poses.py` | 纯 OpenCV 增量式 SfM（ORB/SIFT），带鲁棒 BA |
| `colmap_poses.py` | COLMAP 封装，备选姿态估计后端 |
| `point_cloud.py` | 稀疏点云初始化高斯参数，离群点剔除 |
| `gaussian.py` | 3DGS 核心：纯 PyTorch 光栅化器、Trainer、密度控制、硬件自适应 |
| `exporter.py` | 导出标准 PLY，兼容官方查看器 |
| `gui.py` | PySide6 暗色主题图形界面 |
| `cli.py` | 命令行入口，集成完整流程 |

## 📋 输入数据要求

### 视角数

| 路径 | 最低视角数 | 推荐视角数 |
|------|-----------|-----------|
| OpenCV（自研 SfM） | 30 帧 | 50~60 帧 |
| COLMAP | 60 帧 | 100~200 帧 |

### 帧间重叠率

- 相邻两帧像素重叠建议 ≥70%。
- `--fps` 越高重叠越大；过高会导致 COLMAP 注册帧过少。
- 运动模糊会降低匹配成功率。

### 曝光 / 白平衡

- 自动曝光/白平衡跳变会干扰 SfM 与 BA，建议锁定或预处理。
- RAW 或手动曝光素材重建上限更高。

### 场景环绕度

- 单面可见只能重建拍摄范围，未拍摄面为空白。
- 360° 环绕建议每 ~30° 至少一个机位。
- 垂直方向缺失会导致地板/天花板欠拟合。

### 分辨率

- 特征匹配建议 ≥480p。
- `--scale 0.5` 时 1080p 训练为 960×540。
- 过高分辨率 + 大量高斯可能 OOM。

### 动态场景

3DGS 假设场景静态。移动物体会产生重影/形变，属于方法边界。
## 🎯 姿态估计后端选择

| 后端 | 命令 | 适用场景 |
|------|------|----------|
| OpenCV + ORB（默认） | `--pose-estimator opencv --feature-type orb` | 纹理丰富，速度最快 |
| OpenCV + SIFT | `--pose-estimator opencv --feature-type sift` | 纹理不足或短序列，更稳健但慢 |
| COLMAP | `--pose-estimator colmap` | 长序列（≥60 帧），质量更高 |

**推荐**：

- 短序列（<60 帧）：OpenCV 后端。
- 长序列（≥60 帧）：优先 COLMAP。
- 不确定：先用默认 OpenCV + ORB。

## 🧠 硬件自适应渲染配置

- `device` 决定分档依据：`cuda` → 显存；`cpu` → 系统内存；`auto` → 自动判断。
- 配置写入 `RenderConfig`：`raster_chunk` / `radius_max` / `max_gaussians` / `source` / `hardware_gb`。
- `--max-gaussians` 只覆盖高斯上限，其余渲染参数不变。
- 查看方式：
  ```bash
  python cli.py --video input.mp4 --device cuda --show-config
  ```

GPU 分档参考：<4GB 100k、<6GB 200k、<8GB 300k、<12GB 500k、<16GB 700k、<24GB 1M、≥24GB 1.5M。  
CPU 分档参考：<8GB 50k、<16GB 100k、<32GB 200k、<64GB 300k、<128GB 400k、≥128GB 600k。

`render_config` 会随检查点保存，续训时同步恢复。

## 📈 训练细节

- **损失**：`(1 - w_ssim) * L1 + w_ssim * SSIM`，SSIM 权重线性升温。
- **密度控制**：按梯度分位数自适应分裂/复制，修剪低不透明度高斯。
- **初始稠密化**：高斯少于 2000 时自动 8 倍扩增。
- **学习率衰减**：指数衰减。
- **SH 升温**：前 `sh_warmup_steps` 步逐步提升 SH 阶数。
- **梯度裁剪**：全局范数限制 10.0。
- **光栅化器**：排序式逐像素 splat 向量化，分块控制显存。
- **混合精度**：`--amp` 仅 CUDA + Ampere+ 有收益，默认关闭。
- **帧内存预加载**：训练前预解码为 uint8 RGB，减少磁盘 IO。
- **Loss 发散保护**：单步 loss 超过阈值时保存检查点并中断。
- **焦距自校准**：`--train-focal` 时优化 fx、fy。
## 💾 断点续训

工作目录保存：

| 文件 | 内容 |
|------|------|
| `frame_paths.txt` | 帧路径列表 |
| `frame_meta.json` | GUI 写入的帧元数据（scale/fps） |
| `intrinsics.npy`、`poses.npy`、`sparse_points.npy` | 内参、位姿、稀疏点云 |
| `gaussian_params.npz` | 初始化高斯参数 |
| `training_state.pt` | 完整训练状态 |
| `best_training_state.pt` | 历史最优训练状态 |

恢复训练：
```bash
python cli.py --video input.mp4 --resume-dir ./workdir --output restored.ply
```

恢复时从上次中断帧继续；高基数不同也可恢复；渲染配置同步恢复。  
`best_loss` 从 `training_state.pt` 恢复；`best_training_state.pt` 仅保存历史最优，恢复流程不会自动读取。

## ⚙️ 高级参数建议

- `--sampling-mode two-stage`：快速运动或视角变化剧烈时使用。
- `--sh-degree 3`：最强视角相关效果，训练时间略增。
- `--max-gaussians`：不指定则按设备自动选择；可手动下调。
- `--train-focal`：运动估计不准时改善几何一致性。
- `--random-background`：提升前景质量，背景透明区域可能受干扰。
- `--feature-type sift`：低纹理 / 短序列更稳健，速度慢。
- `--pose-estimator colmap`：长序列推荐。

## 📝 注意事项

- 通常 100~200 帧效果较好。
- 长序列优先 COLMAP；短序列 OpenCV 即可。
- GUI 遇到 CUDA OOM 会尝试降低高斯上限并修剪；CLI 需手动降低 `--max-gaussians`。
- 有显卡但想用 CPU 训练时务必显式传 `--device cpu`。
- 启动时自动绑定所有逻辑核心。
- 纯 PyTorch 光栅化器无需编译 CUDA 扩展。
- `--amp` 仅对 Ampere+ 有 Tensor Core 收益。

## ⚠️ 已知限制

- GUI 未暴露 `--focal-guess`、`--resume-dir`；评估间隔控件未生效。
- 性能数字因硬件、分辨率、场景而异。
- 动态场景 / 运动物体会产生重影或形变，需 4D-GS 类扩展。
- 反射 / 镜面表面会重建发雾或颜色错乱，属方法边界。
- 训练开销与推理/渲染 FPS 无关；导出的 `.ply` 可用官方 CUDA viewer 实时浏览。

## 📄 许可证

Apache-2.0，欢迎自由使用和修改。

## 🙏 致谢

3D Gaussian Splatting 原始论文与开源代码。  
OpenCV、PyTorch、SciPy、PySide6 等优秀开源库。

如有问题，欢迎提 Issue 或 PR。Happy Splatting!