# Z02 倒水 RLT

## 训练端安装

从 RLinf 源码根目录安装 Python 3.11 的 NVIDIA CUDA 环境。

```bash
bash requirements/install.sh embodied --model openpi --env z02
```

安装器为 TorchCodec 固定配套的 PyTorch 版本，并固定可读取 LeRobot v2.1 的 LeRobot 提交。解码需要系统 FFmpeg 共享库。OpenPI 在 `${OPENPI_DATA_HOME:-$HOME/.cache/openpi}/big_vision/` 寻找 `paligemma_tokenizer.model`。离线迁移时可提前将该文件放入对应目录。

## Stage 1 归一化和离线训练

编辑 [config.sh](config.sh) 的 `DATA_DIRS` 和 `BASE_MODEL_PATH`。每个倒水数据目录直接包含 `data`、`meta` 和 `videos`。默认使用 `2026.11.17` 批次。`TASK_PROMPT` 应与倒水数据的任务描述一致。根据 GPU 数量和训练预算设置 `STAGE1_GPU_COUNT`、`STAGE1_STEPS` 和 `STAGE1_SAVE_INTERVAL`。

```bash
cd examples/z02_rlt
bash compute_norm_stats.sh
bash train.sh
```

归一化工具按同一套观测、动作和掩码计算统计量。数据路径或内容变化后重新计算。Stage 1 同时训练 π₀.₅ 的示范动作和 RLT 特征重建，并在输出目录保存配套的归一化文件。

```text
stage1/
├── norm_stats.json
└── checkpoints/
    └── global_step_<N>/
        └── actor/
            └── model_state_dict/full_weights.pt
```

## Stage 2 真机在线训练

在 `config.sh` 中把 `STAGE1_ACTOR_DIR` 指向选定的 `global_step_<N>/actor`。跨机器复制时保留上面的目录层级和同一轮的 `norm_stats.json`。`STAGE2_RESUME_DIR` 可填之前在线训练的 `global_step_<N>` 路径，用于恢复该 checkpoint 中保存的 actor、critic 和 replay。仍需指定原来的 Stage 1 权重。

在真机端，以 ROS Humble 对应的 Python 3.10 加载 Z02 overlay。它需要提供 `zoyi_robot_interfaces` 和 `stereo_camera.frame_hub`。相机采集进程与桥接共享本机的三路帧，桥接独占关节命令发布。进入 `zoyi_brain` 后启动桥接。

```bash
.venv/bin/python inference/z02_rlt_bridge.py
```

训练端和桥接同机时使用默认的 `BRIDGE_URL`。跨机器时可将训练端的本地端口通过 SSH 转发到真机桥接的 `127.0.0.1:8765`。进入 RLinf 的 `examples/z02_rlt` 后启动 Stage 2。

```bash
bash train.sh stage2
```

等桥接日志提示复位请求后，摆好倒水器具并让机器人处于准备姿态。在真机 ROS shell 发布事件。

```bash
ros2 topic pub --once /z02_rlt/event std_msgs/msg/String '{data: ready}'
```

| 事件 | 行为 |
| --- | --- |
| `ready` | 开始下一回合，先使用冻结的 π₀.₅ 参考动作 |
| `actor` | 本回合从下一组动作开始采用在线 actor |
| `success` | 倒水成功，终局奖励为 1 |
| `failure` | 倒水失败，终局奖励为 0 |
| `abort` | 中断回合并解除控制，异常回合不作为失败样本 |

默认累计两个 600 步采样窗口后开始更新。每个终局动作块提交前会等待下一轮 `ready`。在 learner 日志确认已有在线更新后再用 `actor` 评估修正策略。观测保留 25 维布局，真机只控制双臂、夹爪和头部，底盘速度字段置零。

## 迁移与打包

保留 RLinf 的源码、`examples`、`toolkits` 和 `requirements`，以及 `zoyi_brain` 中同目录的 `inference/z02_rlt_bridge.py` 和 `inference/utils.py`。从当前工作树归档或先提交改动。将训练数据、模型权重、归一化文件与 ROS overlay 根据需要单独迁移。用目标机器重建虚拟环境，并在 `config.sh` 填写目标机器的实际路径。
