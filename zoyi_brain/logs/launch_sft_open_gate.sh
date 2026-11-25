#!/usr/bin/env bash
# open_the_gate π0.5 全量微调：discrete_state_input=True，25 维全身，恒定维屏蔽
# 数据：new_2026.10.16（86 集 / 125,133 帧，含底盘运动插值标签）
# 目标：让模型学习底盘运动 → 屏蔽仅 [8-15, 20, 21, 23]，保留左臂/左手/腰 16-19/底盘 22,24
# ⚠️ 推理时必须对应 --policy.discrete-state-input True
cd /home/zoyi/Projects/manipulation/zoyi_brain

# 20000 步 × 64 = 128 万样本 ≈ 10.2 轮（125,133 帧）；loss 走平可用 checkpoint 提前收
STEPS=20000

NPROC=2 bash train_pi05_z02_pytorch.sh \
  --exp-name=sft_open_gate_25d_state \
  --batch-size=64 \
  --num-train-steps=${STEPS} \
  --lr-schedule.peak-lr=2.0e-5 \
  --lr-schedule.decay-steps=${STEPS} \
  --save-interval=5000 \
  --model.discrete-state-input \
  --data.repo-ids /home/zoyi/Projects/manipulation/training_data/lerobot/Z02/open_the_gate/new_2026.10.16 \
  --data.assets.asset-id z02_gate_25d \
  >> /home/zoyi/Projects/manipulation/zoyi_brain/logs/sft_open_gate_25d_state.log 2>&1
