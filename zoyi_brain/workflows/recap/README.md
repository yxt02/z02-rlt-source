# RECAP 离线工作流

当前价值模型、数据准备、评估、可视化和优势计算迁自 `RECAP-full-wrokflow`（来源见 [UPSTREAM.md](UPSTREAM.md)）。价值模型使用三路图像和任务文字，保留 SigLIP2 → Gemma3 → Gemma value expert 架构。旧平均特征模型、离线特征缓存训练和历史 benchmark 入口没有迁入。

流程：原始数据 → 工作副本与 train/val/test 划分 → 价值训练 → 评估 → 完整标签副本 → 策略归一化 → π 训练。各阶段可以独立运行。

## 两个独立环境

根目录的策略环境保持不变。此目录不在根 `uv` 工作区内，其 `pyproject.toml` 仅安装价值工作流的依赖，不安装 Python 包。

已有 `value_function` 环境可直接配置到 `VALUE_PYTHON`。重建环境时，在仓库根执行：

```bash
python3.11 -m venv workflows/recap/.venv
workflows/recap/.venv/bin/python -m pip install './workflows/recap[plot]'
```

价值环境使用 Transformers 4.57.6 / NumPy 2.2.6；策略环境沿用其自己的版本。不要把价值依赖安装到策略 `.venv`。调度器仅使用标准库，两个模型始终通过各自的解释器启动，交换数据文件和 checkpoint。

原始数据、SigLIP2/Gemma3 权重及生成产物不随代码提交。各入口先导入 `submodules`，由其设置 BLAS 线程环境，再导入模型依赖。

## 只在一个脚本里填写配置

统一编辑根目录的 [train_recap_z02.sh](../../train_recap_z02.sh)。价值和策略两个独立入口也读取它，不再分别填写路径。顶部按模型选择、优势标签、数据与产物、环境与权重、可选设置分区，并附中文注释。原有五个普通 `train_pi*_z02*.sh` 保持原来的用法。

**所有相对路径都从 `zoyi_brain` 仓库根目录算起**，与启动时所在目录无关；支持 `../`、带空格的目录、绝对路径及 JAX `gs://` 权重地址。无需重复写 `/home/zoyi/my_work/zoyi_brain`。例如：

```bash
VALUE_SOURCE_ROOT="data/raw"
VALUE_DATA_DIRS=("${VALUE_SOURCE_ROOT}/2026.09.30")
RECAP_WORK_DIR="outputs/recap_z02"
# 标签导出默认位于 <工作目录>/labeled/ 下；可提前填好预计生成的数据集路径。
POLICY_DATA_DIRS=("outputs/recap_z02/labeled/2026.09.30")
POLICY_PYTHON=".venv/bin/python"
POLICY_WEIGHT_PATH="checkpoints/pi05_base_pytorch"
EXP_NAME="recap_z02_01"
```

- `VALUE_DATA_DIRS`：完整原始 LeRobot 数据集，可填写多个；最后一级名称必须不同。
- `POLICY_DATA_DIRS`：RECAP 填完整标签副本的每个子目录；普通模式填原始数据。空数组保留 `config.py` 的 `repo_ids`，流程不会自动替换为刚导出的标签目录。不要把 `labeled/` 聚合目录当作单个数据集；`training_manifest.json` 列出实际导出路径。
- `RECAP_WORK_DIR`：工作副本、划分、价值 checkpoint 和报告目录。`RECAP_TRAINING_DIR` 为空时导出到 `<工作目录>/labeled`；`VALUE_CHECKPOINT` 为空时使用 `<工作目录>/value/best_model.pt`。
- `VALUE_PYTHON`、`POLICY_PYTHON`：两个独立环境的解释器。已有价值 conda 环境可用默认路径，仓库内价值环境可填 `workflows/recap/.venv/bin/python`。
- `SIGLIP_PATH`、`GEMMA3_PATH`、`TOKENIZER_PATH`：价值模型的本地权重，默认位于 `workflows/recap/models/`。`POLICY_WEIGHT_PATH` 为空沿用策略配置；PyTorch 使用转换后的权重目录，JAX 使用 `params` 路径，格式不同。
- `POLICY_ASSETS_DIR`、`POLICY_ASSET_ID`、`POLICY_CHECKPOINT_DIR`：为空时沿用选中 TrainConfig 的设置。普通与 RECAP 可以使用独立的统计目录和 asset id；修改 asset id 后，推理配置也应匹配。

模型选择使用 `TRAIN_BACKEND=jax/pytorch`、`MODEL=pi0/pi05/pi05_lora`、`USE_RECAP=true/false`。`POLICY_CONFIG` 为空时自动选择预设，也可填写自定义配置名。策略 batch、学习率、训练步数、精度和梯度检查点仍由 `src/openpi/training/config.py` 管理。

`RECAP_DATA_CONFIG` 指向回报与划分配置，默认 `workflows/recap/config/z02_data.yaml`；它保留原任务的回报规则和旧批次结果覆盖。新任务应核对 `max_episode_steps`、`failure_reward`、`gamma` 和结果来源；未知批次从原数据的 `is_success` / 终帧 `reward` 读取成功与失败，无法判断时停止。成功轨迹不会自动被当作全帧正标签的专家示范。

环境变量覆盖仍保留：`VALUE_SOURCE_ROOT`、`RECAP_WORK_DIR`、`VALUE_PYTHON`、`POLICY_PYTHON`、`POLICY_DATA_DIR`（单个策略目录）、`POLICY_ASSETS_DIR`、`POLICY_ASSET_ID`、`POLICY_WEIGHT_PATH`、`POLICY_CHECKPOINT_DIR`、`POLICY_EXP_NAME` 等；这些路径覆盖也按仓库根目录解析。多目录编辑统一脚本中的数组，或临时传入 `--data.repo-ids`。

## 分阶段与串联启动

先检查完整命令，不读取数据、加载模型或创建输出：

```bash
python3 scripts/recap_workflow.py --backend pytorch --dry-run
```

在统一脚本填写配置后，三个专用入口都可直接 bash 启动：

```bash
bash train_recap_value_z02.sh   # 默认跑准备、价值训练、评估、标签导出
bash train_recap_policy_z02.sh  # 默认跑统计、RECAP pi0.5 PyTorch 训练（2 卡）
bash train_recap_z02.sh         # 默认串联全部六个阶段
```

统一脚本中的 `STAGE` 控制完整流程；`VALUE_STAGE` 控制单独启动价值入口，`POLICY_STAGE` 控制单独启动策略入口。`DRY_RUN`、`REUSE` 控制预览和已完成产物复用。三个入口共用同一组路径、模型选择和实验名；完整流程仍按人工填写的目录训练，不自动替换数据。

需要临时覆盖时，原有阶段参数仍可使用：

```bash
bash train_recap_value_z02.sh --stage prepare
bash train_recap_value_z02.sh --stage value
bash train_recap_value_z02.sh --stage evaluate
bash train_recap_value_z02.sh --stage export
```

评估会生成 JSON、NPZ 和 HTML 报告。优势导出同时保留分析报告。额外的轨迹图可在价值环境运行 `workflows/recap/scripts/visualize_episode.py`，参数见 `--help`。价值超参数在 `workflows/recap/config/train_value.yaml`，多卡进程数由 `VALUE_NPROC` 控制。

普通 π0.5 保留原来的方式：在 TrainConfig 中填写数据路径并计算统计，然后直接启动原脚本：

```bash
python scripts/compute_norm_stats.py --config-name pi05_zoyi_z02
bash train_pi05_z02.sh
bash train_pi05_z02_pytorch.sh
```

RECAP 策略在统一脚本 `train_recap_z02.sh` 填写标签副本目录，设置 `USE_RECAP=true`（默认），选择框架后直接启动：

```bash
bash train_recap_policy_z02.sh
```

统一配置的 `POLICY_STAGE=norm` 只计算统计，`train` 只启动训练，`all`（默认）顺序执行两者，也可用 `--stage` 临时覆盖。归一化使用同一套最终训练参数，包括额外 CLI 覆盖，并写到训练实际加载的 `POLICY_ASSETS_DIR/POLICY_ASSET_ID`。沿用恒定维度 loss mask、固定值恢复和罕见运动的分位数退化保护；默认统计遍历全部帧。改变 `POLICY_ASSET_ID` 时，推理配置也需使用相同 ID，以读取 checkpoint 内的统计量。

统一入口在路径填写完成后串联全部六个阶段：

```bash
bash train_recap_z02.sh
# 可以只运行某一步。
python3 scripts/recap_workflow.py --stage norm --backend jax --policy-config pi05_zoyi_z02_recap
```

`--model` 支持 `pi0`、`pi05`、`pi05_lora`，LoRA 沿用现有 JAX 入口。统一入口默认 `--recap`，普通模式使用 `--no-recap` 并选择需要的阶段。**策略始终读取你填写的目录，不自动套用价值训练划分，也不自动替换为刚生成的标签目录。**

## 数据合同

准备阶段复制 `data/` 和 `meta/`，链接原视频，只在工作副本写回报与适配信息。维度从真实帧及元信息读取，并与配置核对，保留当前 25 维及更高维数据。价值训练严格读取准备阶段的 train/val/test；评估和优势计算重载 checkpoint 内的配置及训练回报尺度。

策略默认 32 维模型容纳 25 维数据，0–6、8–14 为增量，其余绝对值。后续扩大策略维度还需在 `TrainConfig` 中设置 `joint_dim`、完整增量 mask、足够的 `model.action_dim` 和独立统计量；这次迁移的通用数据检查不会自动调整预训练模型容量。

三个启动脚本共用 `train_recap_z02.sh` 顶部的优势标签配置，无需每次传启动参数：

```bash
RECAP_HORIZON="50"                       # 未来 50 帧的优势
RECAP_LABEL_RULE="percentile"            # percentile 或 fixed
RECAP_POSITIVE_PERCENT="40"              # 优势最高的约 40% 判正，不是第 40 百分位
RECAP_THRESHOLD="0.0"                    # 仅 fixed 模式生效：A > 0
RECAP_REFERENCE_SPLIT="train"            # 用训练划分确定分位阈值
RECAP_FORCE_INTERVENTION_POSITIVE="true"  # 接管帧强制置正
```

分位数模式对参考划分中的全部优势分数取阈值，再将同一阈值用于所有待导出帧；`40` 对应第 60 百分位。阈值可以是负数，接管置正和阈值处同分会让最终正例比例高于目标值。当前多任务数据仍共用一个阈值，没有自动按任务分别计算。`RECAP_THRESHOLD` 在分位数模式不生效。默认优势公式与上游一致，前瞻长度为 `50`。

完整流程和单独价值入口均使用上述配置。单独价值入口仍可临时覆盖，例如 `bash train_recap_value_z02.sh --stage export --label-rule fixed --threshold 0`；直接运行底层 Python 工具时保留旧默认值，需显式传入所需标签参数。回报及划分规则仍在 `RECAP_DATA_CONFIG` 指向的 YAML，价值训练超参数在 `RECAP_VALUE_CONFIG` 指向的 YAML，策略训练超参数在对应 `TrainConfig`；统一脚本中有这三处配置的位置说明。

完整训练副本写入连续 `advantage`（float32）和布尔 `advantage_positive`，更新字段与标签统计元信息，并链接原视频。发布前检查所有帧的 episode/frame 对齐、标签类型、有限数值、轨迹长度、维度、总帧数及视频帧数/帧率/解码。覆盖不完整时不会发布目标目录。部分轨迹可通过 `calculate_advantage.py --split ... --episodes ...` 生成分析结果；该用途不应启用 `--write-training-dataset`。

链接视频要求原视频路径持续可读取。此实现对接当前 LeRobot v2 帧文件与元信息格式；不会自动迁移到 v3 存储格式。

## 记录与产物复用

价值阶段记录 `<work>/run_manifest.json`，策略阶段记录 `<work>/policy_runs.json`。记录包含解析后的配置/命令、输入身份、阶段状态和产物身份；帧/元信息或代码使用哈希或文件属性，视频及本地大权重记录路径、大小与修改时间。

用相同路径、参数和实验名执行 `--reuse`，可跳过已完成且输入/产物一致的阶段：

```bash
POLICY_EXP_NAME=recap_z02_01 python3 scripts/recap_workflow.py --backend pytorch --reuse
POLICY_EXP_NAME=recap_z02_01 bash train_recap_policy_z02.sh --reuse
```

改变数据、几何、代码或产物后不能复用旧记录，应选择新的工作目录/统计目录/实验名。失败阶段记录为 `failed`。价值训练暂未实现优化器内部断点续训；`--reuse` 只跳过完整产物。统计脚本另写 `norm_stats_source.json`，启动训练时核对数据和几何来源；旧统计文件没有来源记录时保持兼容，但训练新模型应重算统计量。

## 验证范围

价值环境：

```bash
cd workflows/recap
PYTHONDONTWRITEBYTECODE=1 "$VALUE_PYTHON" -m unittest discover -s tests -v
```

策略环境在仓库根运行跨环境集成测试：

```bash
RECAP_VALUE_PYTHON="$VALUE_PYTHON" PYTHONPATH=src:. JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest scripts/recap_workflow_test.py -q
```

测试覆盖 25/40 维准备、原始数据保留、逐帧导出、视频解码、元信息、阶段复用、两框架普通/RECAP batch，以及带真实数据管线的小模型策略更新与 checkpoint。价值链路使用小型 SigLIP/Gemma 架构完成训练、重载、评估、优势和发布；策略短训练使用小型替代骨干。GPU 冒烟、完整预训练权重、真实任务成功率需在训练机另行验证，流程测试通过不代表模型效果已达标。
