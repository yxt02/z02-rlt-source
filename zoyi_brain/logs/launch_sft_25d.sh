#!/usr/bin/env bash
# sft_25d_state 正式训练：discrete_state_input=True（状态离散化进 prompt，对齐 pi05 预训练配方）
# ⚠️ 推理时必须对应 --policy.discrete-state-input True
cd /home/zoyi/Projects/manipulation/zoyi_brain
NPROC=2 bash train_pi05_z02_pytorch.sh \
  --exp-name=sft_25d_state \
  --batch-size=64 \
  --num-train-steps=20000 \
  --lr-schedule.peak-lr=2.0e-5 \
  --lr-schedule.decay-steps=20000 \
  --save-interval=5000 \
  --model.discrete-state-input \
  >> /home/zoyi/Projects/manipulation/zoyi_brain/logs/sft_25d_state.log 2>&1
