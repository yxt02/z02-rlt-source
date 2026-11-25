# LeRobot 数据根目录。每个目录直接包含 data、meta 和 videos。
# 默认只使用新采集的倒水批次。可在数组里添加同一动作定义的其他批次。
DATA_DIRS=(
  "${RLINF_ROOT}/../training_data/lerobot/Z02/pour_water/2026.11.17"
)
TASK_PROMPT="Pick up the french press and pour water into the cup."

# Stage 1 的 π₀.₅ 初始化权重。填写包含 model.safetensors 的目录。
BASE_MODEL_PATH="${RLINF_ROOT}/.rlt-local/checkpoints/pi05_base_pytorch"
OUTPUT_ROOT="${RLINF_ROOT}/logs/z02_rlt_pour_water"
NORM_STATS_PATH="${OUTPUT_ROOT}/norm_stats.json"
CONSTANT_ATOL=0.05

STAGE1_GPU_COUNT=2
STAGE1_STEPS=4000
STAGE1_SAVE_INTERVAL=500

# Stage 1 完成后，填写所选 global_step_<N>/actor 目录。
STAGE1_ACTOR_DIR=""
BRIDGE_URL="http://127.0.0.1:8765"
STAGE2_STEPS=1000
STAGE2_SAVE_INTERVAL=50
# 可选。继续 Stage 2 训练时填写之前的 global_step_<N> 目录。
STAGE2_RESUME_DIR=""
