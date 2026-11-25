# Z02 RLT Source

Z02 机器人（倒水任务）**RLT（RL Token）** 训练与真机部署源码合集，由两部分组成：

- **[RLinf](RLinf/)** — 通用强化学习基础设施，内含 Z02 RLT 两阶段（离线 + 真机在线）训练入口
- **[zoyi_brain](zoyi_brain/)** — 基于 openpi 的 π₀ / π₀.₅ 策略栈，含 Z02 数据适配、RECAP 价值模型与 ROS 桥接

> 本仓库是 2026-11-25 的工作快照，**不包含**训练数据、模型权重、归一化文件和 ROS overlay。

## 目录结构

```text
.
├── RLinf/                       # RLinf 0.4.0（Apache-2.0）
│   ├── rlinf/                   # 核心库：scheduler / workers / algorithms / envs / models
│   ├── examples/z02_rlt/        # ★ Z02 RLT 训练入口（config.sh / train.sh）
│   ├── requirements/            # embodied 安装脚本（含 z02 环境）
│   ├── docs/  tests/  toolkits/
│   └── README.md  README.zh-CN.md
└── zoyi_brain/                  # openpi 分支（Apache-2.0，Python 3.10.12 + uv）
    ├── src/openpi/              # π₀ / π₀.₅ 的 JAX 与 PyTorch 实现、Zoyi 数据配置
    ├── workflows/recap/         # RECAP 价值模型离线工作流
    ├── inference/               # Z02 推理客户端 + z02_rlt_bridge.py
    ├── scripts/                 # train.py / train_pytorch.py / serve_policy.py 等
    ├── train_*_z02*.sh          # 普通微调与 RECAP 一键脚本
    └── README.md  RECAP_HANDOVER.md
```

## 两个组成部分

### 1. RLinf（`rlinf` 0.4.0）

面向具身智能与 Agentic AI 的 RL 基础设施：Ray 调度、FSDP / Megatron / SGLang 混合引擎，覆盖 embodied、agentic、SFT、reward 等示例。

本仓库实际使用的是 **`examples/z02_rlt`**：

| 阶段 | 内容 | 入口 |
| --- | --- | --- |
| Stage 1（离线） | 归一化统计 → π₀.₅ 示范动作 + RLT 特征重建 | `compute_norm_stats.sh` → `train.sh` |
| Stage 2（真机在线） | actor / critic + replay，经 HTTP 桥接与真机交互 | `train.sh stage2` |

- 训练端环境（Python 3.11 + CUDA）：`bash requirements/install.sh embodied --model openpi --env z02`
- 桥接默认地址 `http://127.0.0.1:8765`，路径与超参统一在 `examples/z02_rlt/config.sh` 配置

### 2. zoyi_brain（openpi fork）

π₀ / π₀.₅ 的 JAX + PyTorch 实现，针对 Z02 的 **25 维观测 / 动作布局**（双臂 + 夹爪 + 头部）做了适配：

- **数据适配**：`ZoyiInputs` / `ZoyiOutputs`、`LeRobotZoyiDataConfig`；相机键 `cam2`=头部、`cam4`=左腕、`cam3`=右腕，状态/动作为 `joint_positions`
- **微调脚本**：`train_pi0_z02*.sh`、`train_pi05_z02*.sh`（含 LoRA 与 PyTorch 版本）
- **RECAP**：`workflows/recap/` 用三路图像 + 任务文本训练价值模型（SigLIP2 → Gemma3 → Gemma value expert），产出 `advantage` 标签，策略侧做 CFG 条件训练；统一入口 `train_recap_z02.sh`
- **推理**：`inference/pi0|pi05_infer_z02_{jax,pytorch}.py`，以及 `inference/z02_rlt_bridge.py`（ROS 2 Humble 桥接，独占关节命令发布，事件驱动：`ready` / `actor` / `success` / `failure` / `abort`）

## 数据流概览

```text
Stage 1   LeRobot 数据 ──▶ norm_stats ──▶ π₀.₅ + RLT 特征重建 ──▶ actor 权重
Stage 2   真机 ROS 桥接（相机 / 关节） ⇄ RLinf learner（actor / critic / replay） ──▶ 关节命令
RECAP     原始数据 ──▶ 价值模型训练与评估 ──▶ advantage 标签 ──▶ 带 CFG 的 π 训练 ──▶ 部署
```

## 快速开始

**训练端（RLinf）**

```bash
cd RLinf
bash requirements/install.sh embodied --model openpi --env z02
# 编辑 examples/z02_rlt/config.sh（数据目录、基座权重、GPU 数、步数）
cd examples/z02_rlt
bash compute_norm_stats.sh
bash train.sh stage1
# Stage 1 完成后把 STAGE1_ACTOR_DIR 指向 global_step_<N>/actor
bash train.sh stage2
```

**策略端（zoyi_brain）**

```bash
cd zoyi_brain
uv venv --python 3.10.12
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install transformers==4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.10/site-packages/transformers/

bash train_pi05_z02.sh          # 普通微调；RECAP 全流程先编辑 train_recap_z02.sh
```

**真机桥接**

```bash
# 真机端：source ROS 2 Humble 与机器人 workspace overlay
cd zoyi_brain && .venv/bin/python inference/z02_rlt_bridge.py
ros2 topic pub --once /z02_rlt/event std_msgs/msg/String '{data: ready}'
```

## 环境与数据要求

- **训练**：NVIDIA GPU（Stage 1 双卡可跑；π₀.₅ 全参微调建议 80GB，LoRA > 22.5GB），CUDA 12
- **Python**：训练端 3.11（RLinf z02 环境）；策略端 3.10.12（`uv`）
- **数据**：LeRobot v2.1，每个数据集目录直接含 `data/`、`meta/`、`videos/`
- **真机**：ROS 2 Humble，需自备 `zoyi_robot_interfaces` 与 `stereo_camera.frame_hub`（不在本仓库）
- **未包含**：数据集、模型权重、`norm_stats.json`、ROS overlay；`pi05_base` 可用 `gs://openpi-assets/checkpoints/pi05_base/params` 或本地路径

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [RLinf/README.md](RLinf/README.md) · [中文](RLinf/README.zh-CN.md) | RLinf 总览、特性与示例 |
| [RLinf/examples/z02_rlt/README.md](RLinf/examples/z02_rlt/README.md) | Z02 RLT 两阶段训练、桥接事件与迁移说明 |
| [zoyi_brain/README.md](zoyi_brain/README.md) | 安装、Z02 数据准备、微调与推理 |
| [zoyi_brain/RECAP_HANDOVER.md](zoyi_brain/RECAP_HANDOVER.md) | RECAP / CFG 训练与推理的实现记录 |
| [zoyi_brain/workflows/recap/README.md](zoyi_brain/workflows/recap/README.md) | RECAP 离线工作流 |
| [zoyi_brain/docs/](zoyi_brain/docs/) | 归一化、Docker、远程推理 |

## 来源与许可

- **RLinf**：Apache-2.0，上游 RLinf 项目 v0.4.0 快照，含本地 `examples/z02_rlt`
- **zoyi_brain**：Apache-2.0，基于 Physical Intelligence 的 openpi；`workflows/recap` 迁自 `yxt02/RECAP-value-function@549afc4`
- 快照日期：2026-11-25
