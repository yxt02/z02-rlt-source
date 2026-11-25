# RECAP/CFG 训练、推理与恒定维度屏蔽

> 更新：2026-10-13 · 状态：**普通/RECAP 两框架训练、25 维布局、统计屏蔽与独立价值工作流已接通；小型完整数据集通过 CPU 验证，GPU/真实模型效果待验证。最新工作流见 §20。**
>
> 目标分支：`dev_RECAP`。以下早期章节保留当时的实施和验证记录；最新实现与限制以 §16–20 为准。

---

## 1. 任务背景

复现 π\*0.6 的 RECAP：用户的 `calculate_advantage.py` 已给 LeRobot 数据的每个 `episode_00000*.parquet` 帧写入 `advantage`（float）与 `advantage_positive`（bool）两列（已在 `meta/info.json` features 注册）。初期把 bool 标签接入 openpi 训练管线（`scripts/train.py` + `scripts/train_pytorch.py`）；2026-10-12 追加 JAX/PyTorch 双路 CFG 推理，详见 §12。

**参考文档**：
- 需求 MD：`/home/zoyi/Downloads/pi05_RECAP_提示词与训练流程.md`
- 用户 RECAP 实现：`/home/zoyi/my_work/RECAP-value-function-main`
- 参考实现（计划中对照）：`/home/zoyi/my_work/RLinf_official`（RLinf fa1579c）
- 实施计划（含逐文件改动设计）：`/home/zoyi/.claude/plans/valiant-noodling-dolphin.md`

**明确不在范围**：sidecar advantages 导出、`CfgMixtureDataset`、价值模型。MD §6 双路 CFG 推理现已接入。
**硬约束**：非 CFG 配置行为逐位不变；真训练在训练机跑，本机只做全冻结冒烟。

## 2. 核心数据流与算法合同

```
parquet 帧列 advantage_positive (bool)
  → AdvantagePreservingDataset：transform 后按索引恢复 sample["advantage"]
  → DataLoaderImpl：cfg 时 yield 3 元组 (Observation, actions, advantage[B] bool)
  → TokenizePromptWithGuidance：1 个 prompt → 3 份 tokenize（6 键）
       original / f"{prompt}\nAdvantage: positive" / f"{prompt}\nAdvantage: negative"
  → 模型 forward/compute_loss(advantage=...)：
       每前向抽 r ~ U(0,1) 一个/样本 → 7 个 routing mask → 按 mask 替换 tokenized_prompt
  → 分组 loss 指标（14 raw 键）→ 窗口均值 → derive_cfg_ratios → 日志/wandb
```

**路由语义**（`positive_only_conditional=True`，`unconditional_prob=p=0.1`）：
- `pos_cond = advantage & (r > p)` → 正样本以 90% 概率条件、10% 无条件
- `neg_cond ≡ False` → 负样本**永不做条件**，全部无条件
- 期望：`positive_effective_conditional_ratio ≈ 0.9`、`negative_effective_conditional_ratio = 0`、`conditional_ratio ≈ q×0.9`（q = 正样本占比）
- 三种 prompt 都始终产出（negative 即使不路由也保留，"完整接口"要求）

**非 CFG 不变性**：batch 保持 2 元组、`advantage=None` 不进 cfg 分支；路由 RNG 用 `fold_in(rng, 3)`——`split(rng,3)` 的三个子流就是 `fold_in(rng, 0..2)`，datum 3 是独立流，原有 preprocess/noise/time 流逐位不变。

## 3. 改动文件清单

以下清单记录初期训练接入；追加的推理改动见 §12。

### 新增

| 文件 | 内容 |
|---|---|
| `src/openpi/models/cfg.py` (220 行) | JAX 侧：`CFG_RAW_KEYS`（14 键，与 RLinf fsdp_cfg_worker 键名一致）、`draw_routing_random(rng, B)`（fold_in datum 3）、`compute_routing_masks`（7 mask）、`select_guidance_tokens`、`compute_cfg_metrics`（14 raw）、`derive_cfg_ratios`（纯 Python 推导头条键） |
| `src/openpi/models/cfg_test.py` (241 行) | 13 个测试：路由表逐格、full 模式、token 选择、metrics/derivation、确定性、**非 CFG 逐位回归**（p=1.0+全 False 标签 == 非 CFG 模型） |
| `src/openpi/models_pytorch/cfg.py` (157 行) | torch 镜像（不 import jax）；`select_guidance_tokens` 在 raw observation 上操作（torch 版 forward 在 preprocess 前路由） |
| `src/openpi/models_pytorch/cfg_test.py` (141 行) | 6 个测试，含键集合同步断言（`== openpi.models.cfg.CFG_RAW_KEYS` 防漂移） |

### 修改（按管线顺序）

| 文件 | 改动 |
|---|---|
| `src/openpi/models/pi0_config.py` | `Pi0Config` 新增 5 字段：`cfg_guidance=False`、`positive_only_conditional=True`、`unconditional_prob=0.1`、`guidance_type="positive"`、`cfgrl_guidance_scale=1.0`（后两个现已用于 §12 推理）；`__post_init__` 校验；`cfg_guidance` 时 `inputs_spec()` 追加 4 个 guidance spec |
| `src/openpi/training/config.py` | `DataConfig.read_advantage`；`ModelTransformFactory` 按 cfg 选 `TokenizePromptWithGuidance`；`LeRobotZoyiDataConfig.create`/`FakeDataConfig.create` 自动接线 `read_advantage`；新增 `pi05_zoyi_z02_recap`、`pi05_zoyi_z02_recap_pytorch`、`debug_recap` 三个配置（详见 §4） |
| `src/openpi/transforms.py` | 新增 `TokenizePromptWithGuidance`（与 `TokenizePrompt` 同构，pop prompt 后 tokenize 三份） |
| `src/openpi/transforms_test.py` | +44 行：6 键、形状、三份互异、与 `tokenizer.tokenize(...)` 自洽（覆盖 `\n`→空格清洗） |
| `src/openpi/models/model.py` | `Observation` 加 4 个可选 guidance 字段 + 成对校验；`preprocess_observation` 转发 4 字段；`from_dict` 图像循环增加 torch float32 channels-last → NCHW 分支（修 FakeDataset 布局，JAX NHWC 不受影响） |
| `src/openpi/models/pi0.py` | `compute_loss(..., advantage=None)`：cfg 分支在 `split(rng,3)` 之后、embed 之前路由（advantage 为 None 但 cfg 开启时清晰报错）；存 3 个 cfg 属性为 NNX 静态 |
| `src/openpi/training/data_loader.py` | 新增 `AdvantagePreservingDataset`（构建期遍历 transform 前数据集读整列 → `dict[int,bool]`，只持 dict + 可 pickle 的 transformed，spawn worker 安全；缺列/多 `advantage_positive_*` 列/长度不匹配均报错）；`FakeDataset` cfg 时合成 `advantage = bool(index % 2)`；`DataLoaderImpl.__iter__` read_advantage 时 yield 3 元组 |
| `src/openpi/training/data_loader_test.py` | +139 行：列恢复、缺列报错、ConcatDataset offset 对齐、3/2 元组切换 |
| `scripts/train.py` | `train_step` 接 3 元组 + `advantage` kwarg + `has_aux=True`（aux = per-sample loss `[B]`）；同 RNG 流重算 routing → `compute_cfg_metrics` 入 info；日志块 `derive_cfg_ratios` + 头条键；bf16 0-d 数组用 `float()` 包裹再 format（否则抛误导性 `Unknown format code 'f' ... object of type 'str'`） |
| `scripts/train_pytorch.py` | sample/loader 解包兼容 3 元组；forward **先解包 `(losses, cfg_metrics)` 再进原 stack 分支**；14 raw 键跨 rank `all_reduce(SUM)`（所有 rank 参与，rank0 记录）；日志块推导 + 头条键 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `forward(..., advantage=None)`：draw 顺序 **aug → noise → time → `torch.rand(B)` 路由**（与 RLinf 一致）；cfg 返回 `(per_element_loss, metrics_dict)`（detached），非 CFG 返回原 tensor；`_preprocess_observation` 返回 9 元组 |
| `src/openpi/models_pytorch/preprocessing_pytorch.py` | `SimpleProcessedObservation` 显式带 4 字段 |
| `src/openpi/models_pytorch/gemma_pytorch.py` | `projection_dim = vlm_config.width`（原硬编码 2048，dummy 变体 cat 2048 vs 64 崩；真实模型仍 2048。**此 bug 与 RECAP 无关，属顺手修复**） |

## 4. 新增配置

### `pi05_zoyi_z02_recap`（JAX）/ `pi05_zoyi_z02_recap_pytorch`

- `model=Pi0Config(pi05=True, action_horizon=50, discrete_state_input=False, cfg_guidance=True, positive_only_conditional=True, unconditional_prob=0.1, guidance_type="positive", cfgrl_guidance_scale=1.0)`
- pytorch 版另有 `pytorch_compile_mode="max-autotune-no-cudagraphs"`（与基础 pytorch 配置一致）
- `data`：与基础配置同 repo_ids，**`AssetsConfig(assets_dir="./assets/pi05_zoyi_z02", asset_id="z02")` 复用已有 norm stats**，`read_advantage` 由 `cfg_guidance` 自动接线
- `lr_schedule=CosineDecaySchedule(warmup=5000, peak=1e-5, decay=30000, decay_lr=0)`，`AdamW(clip=1.0)`，batch=16，30k steps，seed=42，save/keep=2000
- pytorch 版：`pytorch_weight_path="./checkpoints/pi05_base_pytorch"`、`ema_decay=None`
- ⚠️ **repo_ids 内已有 TODO**：必须改成指向"带 `advantage_positive` 列且含 videos"的数据（当前指向基础配置路径，未验证带标签）

### `debug_recap`（本机冒烟专用，勿用于真训练）

- dummy 变体 + `FakeDataConfig` + `cfg_guidance=True`
- `freeze_filter=PathRegex(".*")`（JAX 全冻结：所有参数转 bf16、trainable 为空）
- `model.pytorch_compile_mode=None`（本机 CPU 上 torch.compile 装饰会间歇段错误，见 §7；`sample_actions` 训练期本就不调用）

## 5. 本机验证记录（全部完成）

| 项 | 结果 |
|---|---|
| CFG 单测（两框架） | **19/19**（JAX 13 + torch 6） |
| `cfg_test + transforms_test + data_loader_test` | **43 passed / 1 failed**——仅 `test_with_real_dataset`（环境缺 libtorchcodec/FFmpeg，与改动无关） |
| `model_test` 回归 | **9/9**（含 `test_model_restore`，见 §8 缓存修复） |
| JAX 全冻结冒烟 `debug_recap` | **通过**：`grad_norm=0.0000`（全冻结生效）、ckpt 步 9；step 5 输出 `conditional_ratio=0.3000 positive_label_ratio=0.3000 positive_effective=1.0000 negative_effective=0.0000 conditional_loss=2.2748 unconditional_loss=2.2945` |
| PyTorch `debug_recap` 冒烟 | **exit=0**：`read_advantage=True`；step 5 输出 `conditional_ratio=0.3000 positive_label_ratio=0.3000 positive_effective=1.0000 negative_effective=0.0000 conditional_loss=1.6737 unconditional_loss=1.6933`；ckpt 步 9 |
| PyTorch 基线 `debug`（非 CFG） | **exit=0**：日志零 CFG 键，2 元组路径完好 |
| 首批次探针 | 3 元组、`advantage (B,) bool`、4 个 guidance 键 `(B,200)`、三份 prompt 两两互异 |
| ruff | 改动文件 5 个告警**在 HEAD 上全部存在**（0 新增）；新文件 format 干净（本机 ruff 0.15.1，仓库 pin 0.8.6） |

说明：小 batch（B=2）下比率是量化值，`positive_effective=1.0` 而非 0.9 属正常（窗口内少数正样本恰好全抽到 r>0.1）；大 batch/长窗口才会趋近 0.9。

**复现命令**（本机）：

```bash
# 单测
uv run pytest src/openpi/models/cfg_test.py src/openpi/models_pytorch/cfg_test.py -q
uv run pytest src/openpi/transforms_test.py src/openpi/training/data_loader_test.py -q

# 冒烟（一次只跑一个，本机曾因并行/内存问题卡死）
uv run scripts/train.py debug_recap --exp-name smoke --overwrite --num_train_steps=10 --log_interval=5 --no-wandb_enabled
uv run scripts/train_pytorch.py debug_recap --exp_name smoke --num_train_steps=10 --log_interval=5 --no-wandb_enabled
# 基线（非 CFG）；--model.pytorch-compile-mode None 绕开本机 torch.compile 段错误
uv run scripts/train_pytorch.py debug --model.pytorch-compile-mode None --exp_name smoke_baseline --overwrite --num_train_steps=10 --log_interval=5 --no-wandb_enabled
```

## 6. 与计划的偏差（均已在方案内显式化）

1. **无梯度累积**（openpi 两 trainer 均无；K=1，batch=16 数学自洽）；MD 的全局 512 示例不适用。
2. 直接读帧列 `advantage_positive`，未导出 sidecar（MD §2 允许）；单 `ConcatDataset`，无 `CfgMixtureDataset` 权重混合。
3. 路由 RNG 计划为 `fold_in(rng,1)` → 实际 **`fold_in(rng,3)`**（datum 1 会撞 noise 流，见 §2）。
4. `freeze_filter` 实现为 `PathRegex(".*")`（等效 `nnx.everything()`）。
5. **PyTorch trainer 不实现 freeze_filter**（上游就没有该机制）→ `debug_recap` 的冻结只对 JAX 生效；PyTorch 冒烟实际会更新 dummy 权重（grad_norm≠0），checkpoint 仅本地意义。
6. `nnx.value_and_grad(has_aux=True, argnums=DiffState)` 在 flax 0.10.2 直接可用，未启用计划中的闭包回退方案。
7. cfg + pi0_fast 组合不支持（pi0_fast 无 advantage kwarg）——现有 recap 配置均 Pi0Config；不加防御代码，错误在 TypeError 显性暴露。
8. 初期 `guidance_type`/`cfgrl_guidance_scale` 仅存储+校验；现已用于双路 CFG 推理（见 §12）。
9. 顺手修复了 3 个与 RECAP 无关的既有问题：`from_dict` float32 torch 布局、`projection_dim` 硬编码 2048、bf16 format 崩溃（均先用基线复现确认为既有问题）。

## 7. 关键实现细节与坑（接手必读）

- **`Observation.from_dict` 图像布局**：uint8 → 归一化（numpy 保持 NHWC / torch 同时 permute NCHW）；float32 torch channels-last → 仅 permute（FakeDataset 路径）。JAX 管线永远 NHWC，torch 管线到 vision tower 前必须 NCHW。loader 框架不变量：`TorchDataLoader` 在 `framework="pytorch"` 时才发 torch tensor，所以分支不会串。
- **bf16 日志格式化**：`f"{bf16_0d_array:.4f}"` 抛 `Unknown format code 'f' for object of type 'str'`（误导性报错），必须 `float(...)` 包裹。`train.py` 已修；`train_pytorch` 走 float 无此问题。
- **PyTorch tuple 解包顺序**：`model(...)` 返回 `(losses, metrics)` 必须**先进入 `isinstance(losses, (list,tuple))` 判定之前**解包，否则 stack 崩。
- **metrics 聚合的 rank 参与**：`all_reduce` 必须所有 rank 调用，只在 `is_main` 里做会死锁；当前实现是先 reduce 后 rank0 记录。
- **torch.compile**：`torch.compile(fn, mode=...)` 装饰时就 eager 调 `_optimize → backend.get_compiler_config()` → 导入 `torch._inductor.compile_fx` → hipify `_pattern` 深递归。本机（CPU-only）间歇段错误；`debug_recap` 已设 `pytorch_compile_mode=None`。**绕过方式**：CLI `--model.pytorch-compile-mode None`（tyro 字段名是 `--model.pytorch-compile-mode`，kebab-case，不是 `--config.model...`）。
- **退出码坑**：`cmd | grep ...; echo $?` 拿到的是 grep 的退出码；判断段错误（139）要用文件重定向 + 直接 `$?`。
- **stdin heredoc / 探针脚本**：直接 `python - <<EOF` 会让 DataLoader worker 异常退出，探针必须写成文件再跑（如 `/tmp/probe_guidance_batch.py`，必要时 `num_workers=0`）。
- **RNG 顺序（PyTorch）**：forward 内 draw 顺序 aug → noise → time → 路由 `torch.rand`；路由在 forward 作用域，不在 `_apply_checkpoint` 内（`preserve_rng_state=False` 无碍）。
- **路由重算一致性（JAX）**：`train_step` 里 metrics 用同一 `train_rng` 重调 `draw_routing_random`——函数纯，mask 与前向完全一致。

## 8. 环境问题（重要，影响稳定性判断）

1. **疑似硬件不稳定**：内核日志 3 次 python segfault **全部落在同一物理核（CPU 10 / core 20 / socket 0）**，`error 4/14`（空指针读/跳转 0），另伴随整机卡死一次、两次瞬态标准库诡异错误（`email.feedparser`/`sre_parse` 报 `'str' object ...`，典型内存被破坏症状）。温度与 MCE 日志正常。**建议正式长训前跑 `memtest86+` 或全核压测**；训练机若在模型构建处无故退出，先排除 compile/硬件。
2. **`pi0_base` 下载缓存修复**：`~/.cache/openpi/openpi-assets/checkpoints/pi0_base/params/` 曾被截断（卡死打断下载）→ 已把截断树移到同级 `_truncated_backup_20261012/`（**4.1G，确认无误后可删**），完整 12G 树从 `params/params/` 提升上来，`test_model_restore` 随之通过。
3. **`transformers_replace` 补丁**已在本 venv 生效（CLAUDE.md 的 setup 步骤）；改动过 uv cache，需要回退时 `uv cache clean transformers`。

## 9. 训练机待办 checklist（按序）

1. **环境**：按 CLAUDE.md setup（uv sync + transformers==4.53.2 + transformers_replace 拷贝）。
2. **单测**：
   ```bash
   uv run pytest src/openpi/models/cfg_test.py src/openpi/models_pytorch/cfg_test.py -q
   uv run pytest src/openpi/transforms_test.py src/openpi/training/data_loader_test.py -q
   uv run pytest src/openpi/models/pi0_test.py src/openpi/models/model_test.py -q
   uv run pytest --strict-markers -m "not manual"        # CI 同款全量
   ```
3. **数据**：把 `pi05_zoyi_z02_recap` / `..._pytorch` 的 repo_ids 指向带 `advantage_positive` 且**含 videos** 的数据（labeled_frames 无 videos，不能直接喂 LeRobot 视频读取）。确认列已进 `meta/info.json`。
4. **真数据短冒烟**（不冻结，验证真实标签链路）：
   ```bash
   uv run scripts/train.py pi05_zoyi_z02_recap --exp-name smoke --overwrite --num_train_steps=20 --log_interval=5 --no-wandb_enabled
   uv run scripts/train_pytorch.py pi05_zoyi_z02_recap_pytorch --exp_name smoke --num_train_steps=20 --log_interval=5 --no-wandb_enabled
   ```
5. **人工核对（MD §5/§8）**：
   - 首帧日志 batch 树含 4 个 `tokenized_*guidance*` 键
   - `positive_effective_conditional_ratio ≈ 0.9`、`negative_effective_conditional_ratio = 0`、`conditional_ratio ≈ q×0.9`
   - `conditional_loss` / `unconditional_loss` 有限且可区分
   - checkpoint 存→取回环
   - 非 CFG 配置（如 `debug`/基础配置）日志无 CFG 键、行为与改前一致
6. **≥8 卡时**留意 DDP `static_graph`（`world_size>=8` 自动开）与 tuple 返回的组合，先跑一次短冒烟。
7. 正式训练；观察 `conditional_ratio` 是否 ≈ 期望值（q×0.9），据此校验数据标签分布。
8. 提醒：lr=1e-5 配 batch=16（RLinf 参考为 512 全局），效果需按训练曲线自行调参。

## 10. git 状态与后续

**未提交**（按要求"先不动"）。工作区：

```
M  scripts/train.py                      M  src/openpi/training/config.py
M  scripts/train_pytorch.py              M  src/openpi/training/data_loader.py
M  src/openpi/models/model.py            M  src/openpi/training/data_loader_test.py
M  src/openpi/models/pi0.py              M  src/openpi/transforms.py
M  src/openpi/models/pi0_config.py       M  src/openpi/transforms_test.py
M  src/openpi/models_pytorch/gemma_pytorch.py
M  src/openpi/models_pytorch/pi0_pytorch.py
M  src/openpi/models_pytorch/preprocessing_pytorch.py
?? src/openpi/models/cfg.py              ?? src/openpi/models/cfg_test.py
?? src/openpi/models_pytorch/cfg.py       ?? src/openpi/models_pytorch/cfg_test.py
?? CLAUDE.md                             ?? RECAP_HANDOVER.md（本文档）
```

- `CLAUDE.md` 是 `/init` 生成的项目说明，与本任务无关（也未提交）。
- 提交时按仓库惯例结尾加 `Co-Authored-By: Claude Code <noreply@anthropic.com>`；建议按逻辑分 1–2 个 commit（核心管线 / 顺手修复），或按团队习惯。
- checkpoints、norm stats（`checkpoints/`、`assets/`）均 gitignored，本地冒烟产物可直接删。

## 11. 一图总览（改动如何串起来）

```
pi0_config.py ── cfg_guidance 开关 ──┐
                                     ▼
config.py ── read_advantage + TokenizePromptWithGuidance 选型 + *_recap 配置
                                     │
transforms.py ── 1 prompt → 6 键 ────┤
                                     ▼
model.py ── Observation +4 字段（校验/转发）── 两框架共用合同
                                     │
data_loader.py ── advantage 列恢复 → 3 元组 batch
                                     │
              ┌──────────────────────┴──────────────────────┐
              ▼                                             ▼
        JAX: pi0.py compute_loss                     PyTorch: pi0_pytorch.forward
        + models/cfg.py 路由/指标                     + models_pytorch/cfg.py 镜像
              │                                             │
              ▼                                             ▼
        train.py: has_aux+DiffState                 train_pytorch.py: 解包+all_reduce
        窗口均值 → derive_cfg_ratios → 日志/wandb（两边同一套键名与公式）
```

## 12. 双路 CFG 推理（2026-10-12 追加）

已直接应用到 `/home/zoyi/my_work/zoyi_brain`，保留此前工作区改动，未 commit、未 push。

已按减少改动文件的要求收敛：本次推理扩展的运行时代码仅修改 `pi0.py` 与 `pi0_pytorch.py`；`pi0_config.py` 恢复到追加推理前的内容，直接复用已有配置字段。推理测试并入现有 `src/openpi/models/cfg_test.py`，取消单独的 `cfg_inference_test.py`。连同本文档，本次推理扩展相对既有训练接入共调整 4 个已有文件，未新增仓库文件。

- JAX `src/openpi/models/pi0.py::sample_actions` 与 PyTorch `src/openpi/models_pytorch/pi0_pytorch.py::sample_actions` 分别为原始 prompt、所选 advantage prompt 建立独立前缀 KV cache。
- 每个去噪步骤用相同的观测、动作 `x_t` 与时间 `t` 计算两路预测，先混合速度，再更新动作：`v = (1-w) * v_original + w * v_guided`，`x_next = x_t + dt * v`。
- `cfg_guidance=True` 且 `guidance_type="positive"` 时使用正 advantage 条件；`w=1.0` 对应正条件速度，`w>1.0` 支持增强引导。当前两个 RECAP 配置默认已经是 positive / 1.0，因此加载匹配的配置后自动启用。
- `cfg_guidance=False` 或 `guidance_type="no_guide"` 保持单路采样。缺少条件 token/mask 时明确报错。仅当 `positive_only_conditional=False` 时允许 negative 引导。
- 两框架的采样入口在启用引导时校验 `cfgrl_guidance_scale` 必须有限且非负，复用现有 `Pi0Config` 字段。
- 推理使用训练配置中的 guidance 设置，无需实时输入 advantage 布尔标签，也无需运行价值模型。

加载已训练好的 RECAP checkpoint 时使用对应配置：JAX 为 `pi05_zoyi_z02_recap`，PyTorch 为 `pi05_zoyi_z02_recap_pytorch`。如需 `w=1.5`，在创建 policy 前替换 model config：

```python
import dataclasses

from openpi.policies import policy_config
from openpi.training import config

train_config = config.get_config("pi05_zoyi_z02_recap")  # PyTorch 改用 ..._pytorch
train_config = dataclasses.replace(
    train_config,
    model=dataclasses.replace(
        train_config.model,
        guidance_type="positive",
        cfgrl_guidance_scale=1.5,
    ),
)
policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
```

现有 `src/openpi/models/cfg_test.py` 中的推理测试用非线性轻量骨干调用实际两框架采样函数，覆盖逐步混合、独立缓存、相同噪声与时间、不同 token mask、JAX policy JIT、禁用引导、缺少条件、negative 模式及非法 scale。图像预处理的测试替换仅对新增推理测试生效，不影响原有训练测试。

本轮验证结果：

- 精简后 CPU 针对性测试 **58/58 通过**：推理 23 项（非法 scale 改为验证两框架采样入口）、两框架训练 CFG 19 项、transforms 12 项、Pi0 配置/冻结 4 项。
- 对照应用前保存的采样函数，轻量骨干下两框架 `cfg_guidance=False` 和 `guidance_type="no_guide"` 的输出逐位一致；JAX JIT 输出也一致。
- 改动 Python 文件的 Ruff 检查、格式检查与 `git diff --check` 通过。两个模型文件、既有 CFG 测试及本文档之外的既有工作区文件哈希未变，`pi0_config.py` 与追加推理前逐字节一致。
- 上一轮扩展测试中，普通 FakeDataset 的一个加载子进程曾在 Python 标准库 dataclasses 初始化处异常退出，单独复跑该测试 **1/1 通过**。真实外部数据集测试未完成，不计入上述 58 项。
- 尚未运行 GPU、完整模型 checkpoint 或机器人实机验证。

本轮不变更训练数据路径、专家示范置正规则或 `discrete_state_input`。本地两个 RECAP 配置的 `discrete_state_input` 当前仍为 `False`；训练和推理必须使用与 checkpoint 一致的状态设置。

## 13. 统计恒定维度的自动屏蔽（2026-10-13 追加）

已接入 π0 / π0.5 的 JAX、PyTorch 路径，仅保存在本地，未 commit、未 push。沿用先计算统计量再训练的流程，不在 batch 内重新判断哪些关节恒定。

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_zoyi_z02 --constant-atol 1e-6
```

`constant_atol` 默认 `1e-6`，单位与各维度在 data transforms 后的数值相同。判定为观测到的 `max - min <= constant_atol`，固定值取均值。`0` 只识别严格恒定，`None` 禁用自动屏蔽。增大容差前应查看打印出的 state/actions 维度编号和保存的 min/max；阈值是容许的运动范围，不是通用的关节精度。

路径说明：Zoyi π0.5 RECAP 的 `data.assets.assets_dir` 引用 `./assets/pi05_zoyi_z02`，所以应对实际引用的基础配置计算统计量，并确认其数据路径、关节维度和增量规则与训练数据相同。当前 JAX 25 维配置使用 `z02_25d`，未更新的 PyTorch 16 维预设仍使用 `z02`。统计脚本始终保存到所选配置自己的 assets 目录；选择 `pi05_zoyi_z02_recap` 不会自动替换基础配置的统计文件。若使用不同数据集或关节布局，应显式设置各配置的统计量引用，避免共享不匹配的统计量。

- `norm_stats.json` 保存 min/max、constant_mask、constant_atol，均值复用原 mean 字段。state 和 actions 分别判定，不用 q01/q99 相等来认定恒定，以保留偶尔运动的关节。统计内部使用 float64，避免低精度平方累计误差影响统计。
- 恒定维度归一化为 0。actions 的有效维度标记通过 Observation 到达两框架；模型补零维度随 mask 补 False。训练的动作与噪声在屏蔽维度归零，loss 仅按有效维度数平均；全部屏蔽时 loss 为 0。推理每个去噪步骤也维持这些维度为 0，反归一化强制恢复对应动作均值。归一化另外保留原始 state，policy 输出衔接后由 Unnormalize 恢复，避免归零丢失当前状态而影响增量动作恢复。
- 两套 checkpoint 保存代码已会保存整份 NormStats，无需再改训练入口或 policy 创建代码。旧统计文件没有 constant_mask，保持原来的缩放、loss 和采样行为。要启用新行为，需要重新计算统计量，并用它进行训练；不要只给已有模型替换推理统计量。
- Zoyi 的统计在 DeltaActions 后计算，因此固定的增量均值 0 表示保持当前关节位置；它不代表恢复成某个固定的绝对角度。绝对动作维度恢复为统计均值，再沿现有输出流程返回。
- LeRobot 统计扫描包含最后不足一个 batch 的帧。使用 `--max-frames` 时只对采样帧负责，可能漏掉低频运动，应优先全量扫描。RLDS 路径仍使用其现有批次读取方式，本轮未验证。

本轮核心运行时代码修改 7 个现有文件：统计脚本、shared/normalize.py、transforms.py、models/model.py、两框架的 pi0 模型文件，以及 policy.py 的两行原始 state 传递；测试并入既有 normalize_test.py、transforms_test.py、models/cfg_test.py，没有增加仓库文件。

CPU 针对性测试 **89/89 通过**，覆盖统计分类与序列化、旧 JSON 兼容、偶发运动、最后一个 batch、两种归一化、固定值恢复、增量转换、loss/梯度屏蔽、普通与 CFG 采样、JAX JIT，以及两框架 policy 使用真实当前状态恢复增量动作。Ruff 检查、格式检查与 git diff --check 通过。尚无实际训练数据统计、GPU 或机器人实机验证。

## 14. dev_RECAP 提交前验证（2026-10-13）

以远端 `dev_RECAP` 的 `8019633` 为基础，保留已有相机话题更新。提交分为 RECAP 训练/CFG 推理，以及统计恒定维度屏蔽两部分。21 个修改 Python 文件通过当前本地 Ruff 和 CI 固定版本 Ruff 0.8.6 的检查与格式验证。

扩展 CPU 检查加入 data_loader_test.py，排除需要外部数据的 test_with_real_dataset：100 项通过，test_torch_data_loader_parallel 的子进程发生 SIGSEGV；该单项隔离复跑 1/1 通过。多进程稳定性仍需在实际训练环境确认，不能将此记录视为一次完整无失败的测试运行。

远端 `8019633` 的红叉对应 [pre-commit run 38248455525](https://github.com/Zoyi-Tech/zoyi_brain/actions/runs/38248455525)：uv-lock 通过，Ruff 33 个问题中自动修复 6 个、剩余 27 个，格式检查改写 5 个文件。主要涉及已有 ROS 推理代码，本次没有进行无关文件的批量格式修改。现有 pre-commit 工作流仅对 main push 和 PR 触发；Test 工作流仅对 PR 触发，并使用 openpi-verylarge runner。直接推送 dev_RECAP 不会自动运行这两个工作流。

## 15. 同一配置切换普通训练与 RECAP（2026-10-13）

训练入口现在支持 `--recap` / `--no-recap`，两者都是现有 `model.cfg_guidance` 的命令行别名，没有增加第二个模型开关。JAX、PyTorch 的 π0 / π0.5 以及 JAX LoRA 启动脚本均可传入额外参数。不传开关时沿用所选配置的默认值：普通配置关闭，`*_recap` 配置开启。

```bash
# 普通训练：不读取 advantage 标签，使用原始任务提示词。
bash train_pi05_z02.sh --no-recap --exp-name sft_state --model.discrete-state-input

# RECAP：相同数据、状态设置和超参数，读取标签并训练条件提示词。
bash train_pi05_z02.sh --recap --exp-name recap_state --model.discrete-state-input

# PyTorch 同样支持；π0 使用对应 train_pi0_z02*.sh。
bash train_pi05_z02_pytorch.sh --recap --exp-name recap_state --model.discrete-state-input
```

直接运行训练入口也支持，例如 `uv run scripts/train.py pi05_zoyi_z02_recap --no-recap --exp-name sft_control`。关闭时输出普通 `(observation, actions)` batch；开启时输出 `(observation, actions, advantage)`，启用原条件、正条件、负条件提示词和既有路由规则。开启不负责生成标签，真实数据仍须有完整的 `advantage_positive` 布尔列；关闭后无需该列。

开关不会改学习率、数据路径、状态输入、归一化策略或恒定维度 mask。若希望使用 RECAP 预设的学习率和 warmup，仍选择 `*_recap` 配置；同一配置切换适用于保持其余条件一致的对照。两个模式应使用不同 `exp_name`，因为已有启动脚本带 `--overwrite`，使用相同实验名会覆盖旧训练目录。

推理也使用 `TrainConfig`：服务的 `--policy.config` 通过 `get_config` 读取当前 `src/openpi/training/config.py`，由该配置创建模型和输入/输出 transforms；不会自动恢复训练命令行对 RECAP 或状态输入的覆盖。使用同一配置名、同一版配置文件且重新启动服务时，文件内的设置是一致的。归一化统计量默认从 checkpoint 的 `assets` 读取。

推理服务新增两个可选覆盖项，未提供时保留配置默认值。下面命令对应上述 RECAP + state=True 训练：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_zoyi_z02 \
  --policy.dir /实际的/checkpoint/目录 \
  --policy.recap True \
  --policy.discrete-state-input True
```

普通模型将 `--policy.recap` 设为 `False`；状态输入必须与该 checkpoint 的训练设置一致。服务会打印 `Policy input settings: recap=..., discrete_state_input=...` 便于核对。两项均不改变权重，不能用推理开关将普通 checkpoint 转成学会优势条件的 RECAP 模型。

本轮针对开关的 CPU 检查 41 项通过：7 个训练预设的两个 CLI 模式、两框架的 batch/条件字段切换、Zoyi 状态 transform 独立性、推理参数覆盖及非 π0 配置校验。初次测试在 pytest 的 AST 重写和 tyro 的随机缓存自检处出现环境异常；CLI 用与真实训练入口相同的独立进程验证后全部通过。没有运行真实数据训练或机器人验证，状态 True 实验成功率为 0 的原因尚未确认。

原有归一化、transforms、两框架 CFG 和 Pi0 检查另行复跑 89/89 通过。4 个修改的 Python 文件通过 CI 固定版 Ruff 0.8.6 的代码与格式检查，5 个启动脚本通过 bash 语法检查，git diff --check 通过。

## 16. 25 维全身布局及更高维度（2026-10-13）

用户确认的顺序为：0–6 左臂、7 左手、8–14 右臂、15 右手；16–24 保持绝对值。JAX `pi05_zoyi_z02` 与 `pi05_zoyi_z02_recap` 现设置 `joint_dim=25`，`delta_action_mask=make_bool_mask(7, -1, 7, -10)`。这 25 个状态/动作坐标全部保留，补到模型默认的 32 维，policy 反归一化及增量恢复后返回 25 维。数据短于配置维度时明确报错。

两个配置改用独立的 `asset_id="z02_25d"`，必须重新计算统计量：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_zoyi_z02 --constant-atol 1e-6
bash train_pi05_z02.sh --no-recap --exp-name sft_25d
```

统计文件为 `assets/pi05_zoyi_z02/z02_25d/norm_stats.json`，checkpoint 保存并读取相同 asset id。不要将旧 `z02` 的 16 维控制统计直接复制到此处：旧文件可能把 16–24 识别为补零维度，从而屏蔽真实身体状态和动作。训练与推理使用相同维度、增量 mask、状态输入设置及 checkpoint 内的统计量。

`LeRobotZoyiDataConfig` 新增可配置的 `joint_dim` 和完整 `delta_action_mask`。未设置维度时保留旧的双臂 16 / 单臂 8 维默认值，当时其他 LoRA/PyTorch 预设沿用旧布局；π0.5 PyTorch 预设现已按第 19 节对齐。自定义增量布局必须提供与 `joint_dim` 等长的布尔 mask；绝对动作模式可关闭 `use_delta_actions`。通用处理支持大于 32 的维度，但需要同步增加 `model.action_dim`，不能超过模型容量。

未来例如 40 维，可在同一配置文件设置 `joint_dim=40`、`Pi0Config(action_dim=64, max_token_len=512, ...)`、40 项增量 mask 和新的 asset id。JAX `CheckpointWeightLoader` 新增显式 `allow_action_dim_resize=True`，仅调整 `action_in_proj/kernel`、`action_out_proj/kernel`/`bias` 及 π0 的 `state_proj/kernel` 的动作坐标轴；保留重叠坐标，新增坐标初始化为 0，其余维度及骨干权重继续严格校验。不启用时保留原来的严格加载行为。扩大后的模型需重新训练新动作维度，推理使用训练好的完整模型维度。PyTorch 预训练权重扩展不在本轮 JAX 工作范围内。

新增维度/权重测试与既有训练开关、推理覆盖测试合计 59 项通过。另用真实小型 JAX π0.5 架构验证了 32→64 的完整参数加载、40 个有效坐标、新增坐标的非零训练梯度、补零坐标的零梯度及 JIT 采样输出，均通过；未加载完整预训练权重或运行 GPU/实机。

原有统计量、归一化、transforms、两框架 CFG 和 Pi0 测试另行复跑 89/89 通过；5 个修改的 Python 文件通过 Ruff 0.8.6 代码与格式检查，git diff --check 通过。

本节的模型维度扩展最初未包含 ROS 映射。用户随后提供了采集源代码；JAX 客户端的对应映射已在第 17 节补齐。

此前发现的非恒定维度 `q99-q01` 退化问题已按第 18 节补齐。需用更新后的脚本重新计算统计量，再训练新模型。

## 17. 对齐 Z02 采集布局的 JAX ROS 推理

源代码：[collection.py，commit 24f1c954](https://github.com/Zoyi-Tech/zoyi_data_collection/blob/24f1c9549c3e48d316f51d9ee02711a306f27c00/src/z02_data_collection/scripts/collection.py#L235)。后 9 维实际为 **4 个身体坐标 + 2 个头部坐标 + 3 个底盘速度**，不是 9 个 JointState 关节或 6 维 Twist。

| 模型索引 | 状态来源 | 动作执行 |
| --- | --- | --- |
| 0–6 / 7 | `position[4:11]` / 左手二值状态 | 左臂 / 左手 |
| 8–14 / 15 | `position[11:18]` / 右手二值状态 | 右臂 / 右手 |
| 16–19 | `position[:4]` | `joint_positions[:4]` |
| 20–21 | `position[18:20]` | `joint_positions[18:20]` |
| 22–24 | `/slamware_ros_sdk_server_node/odom` 的 `twist.twist.linear.x/y`、`angular.z` | `/cmd_vel` 的 `linear.x/y`、`angular.z` |

用户确认线上采集源代码的右臂 `[7:15]` 切片错误已在实际采集版本修正为 `[8:15]`，状态和动作均如此。未修改外部采集仓库，也未据此认定既有数据损坏。用户明确要求前 3 个身体坐标主动控制，因此 25 维模式将其 `control_mode` 开启；16 维模式保留旧设置。

仅修改原 `inference/pi05_infer_z02_jax.py` 完成实际控制接入：从训练配置读取维度，warmup 用相同长度，25 维模式同步里程计、组装完整状态、将关节和底盘动作分别路由到原关节话题及 `/cmd_vel`。仅支持已知 16/25 维布局，其他维度明确报错，不能猜测硬件映射。新增 ROS 参数 `config`、`policy_dir`，应指定新训练的 25 维 checkpoint；原默认路径仍是旧示例，不能直接当作 25 维模型使用。

关闭推理、复位、动作队列为空、同步观测超过 1 秒未更新或退出节点时发布零底盘速度；非有限或维度不匹配的动作在写入关节命令前拒绝。ROS 回调及初始化测试不依赖实际 ROS，覆盖两个布局、完整映射、前 3 个控制模式、配置与 warmup 一致、停止条件和无效输入。当时 PyTorch ROS 文件保持旧模式；现已按第 19 节补齐。未进行 ROS/GPU/机器人实测。

## 18. 罕见运动导致分位数退化的保护

仅靠恒定维度 mask 不能处理“1000 帧中仅 1 帧运动”的有效维度：全范围大于阈值，维度正确保留，但 `q01=q99` 时旧分母会把动作放大到约 200 万。

`RunningStats.get_statistics` 为新统计量记录 `quantile_fallback_mask`。非恒定坐标的 `q99-q01 <= 1e-6` 时，Normalize/Unnormalize 使用同一份完整 min/max 作为缩放区间。原 q01/q99 仍保存为实际统计分位数，不覆盖成极值；正常分位数坐标保持原公式。此处的 fallback 只改变缩放，不影响 loss mask，不会冻结有运动的维度。真正恒定坐标仍按全观测范围与 `constant_atol` 分类，归零、屏蔽并恢复均值。

新字段可序列化、保存到 checkpoint 并随推理加载。旧 JSON 没有此字段时保持旧归一化行为，即使已经包含 min/max；避免仅更新推理代码就改变旧模型的输入尺度。要启用修复，必须重新计算统计量并重新训练。关闭恒定维度屏蔽时，分位数 fallback 仍可正常工作。统计元信息会校验 mask 的布尔类型、长度及 min/max 是否齐全。

本轮统计/归一化/维度映射/ROS 模拟/两框架 CFG 测试 **117 项通过**，另用独立进程执行 Pi0 架构检查 **4 项通过**。稀有运动探针由约 200 万的归一化幅度降至 1，往返误差为 0，且该维度保持有效。初次将所有测试合在同一进程时，进入 Pi0 架构检查发生 JAX 底层段错误；拆分进程后两组均通过，未宣称该进程崩溃的根因已经解决。6 个修改的 Python 文件均通过 Ruff 0.8.6 代码检查，5 个核心/测试文件通过格式检查；为限制无关改动，ROS 文件沿用原有排版，没有对整文件重排。`git diff --check` 通过。真实训练数据、完整 checkpoint、ROS/GPU/实机尚未验证。


## 19. PyTorch π0.5 与配置来源对齐（2026-10-13）

`pi05_zoyi_z02_pytorch` 与 `pi05_zoyi_z02_recap_pytorch` 均在 `config.py` 显式设置 25 维、`make_bool_mask(7, -1, 7, -10)`、`z02_25d`、共享统计目录及 boil_water 数据路径。模型继续为 32 维，后 7 维由统计 mask 屏蔽。普通配置不读优势标签；RECAP 配置读取 `advantage_positive`，保留其独立学习率与 5000 步 warmup。两配置 `discrete_state_input=False`。

训练参数统一从对应 TrainConfig 读取，沿用 `batch_size`、`num_train_steps`、`num_workers`、`pytorch_training_precision`、`pytorch_weight_path` 等字段；新增 `pytorch_gradient_checkpointing=True`，默认保持原先开启行为，可在配置或 CLI 关闭。两个 Z02 π0.5 PyTorch 预设关闭 wandb，保持此前启动脚本的本地训练默认。启动脚本只选择配置、启动多卡、设置默认实验名和转发覆盖参数，移除写死的 batch/训练步数/学习率等设置，原脚本的 20000 步不再覆盖配置的 30000 步。RECAP 可用首参数 `pi05_zoyi_z02_recap_pytorch` 选择，也可沿用 `--recap` 开关。

`batch_size` 为全局 batch，需为进程数的正整数倍，否则在改动 checkpoint 前报错。配置默认仍为 16，可在对应预设改为 64（2 卡各 32）；没有添加梯度累积。模型精度通过复制配置设置，避免修改全局预设；float32 模式不再强制进入 bfloat16 autocast。DDP 在每张卡复制模型，能否使用更大 batch 取决于实际显存，未验证完整模型的 GPU 容量。

PyTorch ROS 客户端按第 17 节的同一布局接入 25 维，包括身体、头部、vx/vy/wz、主动控制前 3 个坐标和底盘停止条件。保留 PyTorch 模型设备选择、`inference_mode` 和 safetensors 加载。它通过 ROS `config` / `policy_dir` 参数选择同训练配置与 checkpoint，warmup 使用配置维度；16 维兼容模式仍可用。

验证覆盖 171 个不同测试用例：统计/归一化/两框架 CFG/策略与 ROS 模拟 128 项通过，PyTorch 训练/启动/推理配置覆盖 21 项通过，训练开关及 loader 合约 22 项均通过（其中 14 项 CLI 已整组复跑）。普通/RECAP 小模型各在 batch 64 运行 2 步，分别覆盖梯度检查点开/关，验证恒定坐标梯度为零、罕见转动坐标有梯度、参数更新及 checkpoint 统计元信息保存。首次开关测试中一个 CLI 子进程在导入依赖时出现 `TypeError: list_iterator object is not callable`；该用例单独重跑及全部 14 项 CLI 整组重跑均通过，未认定其根因已修复。5 个修改的 Python 文件 Ruff 0.8.6 检查通过，4 个核心/测试文件格式检查通过，ROS 文件保留既有排版；启动脚本语法及 patch 应用检查通过。真实数据、完整预训练 checkpoint、GPU/ROS/机器人尚未验证；运行前需重新生成带 mask 和分位数 fallback 的 25 维统计量，并准备转换后的 PyTorch 基础权重。


## 20. 独立 RECAP 价值工作流合并（2026-10-13）

`workflows/recap/` 迁入价值仓库 `RECAP-full-wrokflow` 的当前图像/文字主流程，来源提交 `549afc40693e604c17d6a08d82619dae0d22f832`。保留 SigLIP2、Gemma3、Gemma value expert、回报准备、价值训练、评估、可视化及原优势公式；移除旧平均特征模型、离线特征缓存训练和历史 benchmark 入口。独立 `pyproject.toml` 不加入根 uv workspace，价值与策略在不同解释器进程运行。

统一入口 `scripts/recap_workflow.py` 支持 prepare/value/evaluate/export/norm/policy/all、框架选择、`--dry-run` 和 `--reuse`。价值与策略的路径、模型选择和实验设置现统一在 `train_recap_z02.sh` 顶部，两个独立入口共用；相对路径以仓库根目录为基准，见本节末尾的最新调整。原有五个普通启动脚本恢复合并前用法，见下一段。专用工作流的策略超参数读取选中的 TrainConfig，策略始终读取用户配置的目录，不自动使用价值划分或改写路径。

准备阶段复制帧文件和元信息、链接原视频，只在工作副本写回报和适配信息；按真实数据与配置校验维度，25/40 维验证通过。价值训练严格读取生成的 train/val/test，评估与优势重载 checkpoint 的配置与训练回报尺度。完整标签副本写 `advantage` float32、`advantage_positive` bool，复制数据/元信息、链接视频，并更新标签统计。默认 horizon=50、threshold=0、接管帧置正，保留 train 分位阈值选项。帧覆盖、索引、轨迹长度、维度、视频帧数/帧率/解码或元信息不一致时停止发布；部分导出仅供分析。

`compute_norm_stats.py` 接收路径覆盖或完整训练 CLI 参数，使用同一配置和变换，并写到实际 assets 目录/asset id。沿用恒定维度 mask 与罕见运动保护，新增 `norm_stats_source.json`；训练启动前核对数据和几何来源。没有来源记录的旧统计量保持兼容，新训练应重算。价值和策略分别记录阶段命令/配置、输入身份、产物及 running/failed/complete 状态；`--reuse` 仅跳过输入/产物一致的已完成阶段，不实现价值优化器内部续训。

验证：价值环境 30 项通过，包含小型真实 SigLIP/Gemma 训练、checkpoint 重载、评估/HTML、32 帧优势及完整发布。策略跨环境新增 10 项通过，直接读取该小模型生成的标签副本，覆盖 JAX/PyTorch × 普通/RECAP 四种真实视频 batch、统计来源一致性、恒定坐标、PyTorch 两模式各 2 步更新与 checkpoint，以及只读调度和复用检查；另复跑 4 个现有启动用例。原有归一化/transforms/两框架 CFG/策略 128 项及 PyTorch 训练相关 21 项通过（启动用例与前述 4 项重叠），本轮合计 189 个不同用例。本轮 Python 文件的 Ruff 0.8.6 代码/格式检查、shell 语法和 patch 检查通过。全仓 Ruff 另有 16 处既存问题（旧 π0 ROS、inference/utils 与 websocket 服务），与合并前逐项一致；本轮没有修复这些无关文件，因此不宣称整个 CI 已转绿。

CPU 视频数据读取使用 PyAV；本机可选 TorchCodec/FFmpeg 组合不可用。完整链路测试使用小型模型及临时完整数据集，未加载真实预训练大模型；GPU 驱动不可用，GPU 冒烟、真实数据效果及机器人成功率未验证。新工作流目前对接 LeRobot v2，不自动转换 v3。运行说明见 `workflows/recap/README.md`。


### 专用启动入口调整（2026-10-13）

按用户后续要求，五个 `train_pi0/05*_z02*.sh` 已逐字恢复到工作流合并前（`4dafcb6`）的版本，保留普通训练原有覆盖参数和独立统计流程。新流程只通过专用入口启动：`bash train_recap_value_z02.sh` 默认价值四阶段，`bash train_recap_policy_z02.sh` 默认统计 + π0.5 PyTorch RECAP 训练（2 卡），`bash train_recap_z02.sh` 默认全部六阶段。阶段、框架、模型、模式、实验名、预览与复用可在专用脚本顶部填写，命令行覆盖仍保留。调度器改为调用专用策略入口，原脚本不再承接新阶段参数；数据路径仍由人工配置。

此次 18 个针对性测试通过，含旧 PyTorch 启动行为、JAX/PyTorch × 普通/RECAP 无参数启动顺序、统计失败停止训练，以及已有跨环境小型完整链路；Python 代码/格式、全部 shell 语法和 patch 检查通过。未启动真实大模型训练或 GPU/机器人运行。


### 相对路径与统一配置（2026-10-13）

按用户要求，RECAP 三个入口的配置集中到 `train_recap_z02.sh`，不新增配置文件。价值与策略独立入口 source 统一脚本时只读取配置，不触发全流程。顶部按用途分区并补充中文注释；相对数据、环境、权重、统计及输出路径在调度前转为仓库根目录下的路径，支持从其他目录启动、带空格的路径、`../`、绝对路径和远程权重 URI。环境变量及 CLI 覆盖继续可用。普通启动脚本未修改；策略数组仍需人工填写，不自动使用价值划分或标签产物。

本次 12 项启动相关测试通过，包括原有 PyTorch 启动行为、两框架普通/RECAP 启动、统计失败停止以及三个入口共用相对路径；覆盖无关启动目录、空格、多数据集、独立环境、相对权重/配置路径和远程 JAX 权重 URI。改动的 Python 文件通过 Ruff 代码/格式检查，三个 shell 语法检查通过。此次仅调整启动配置，未运行正式训练。
