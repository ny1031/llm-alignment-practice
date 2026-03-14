#!/bin/bash
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_NAME="${MODEL_DIR:-/path/to/models}/qwen3-1.7b-sft-by-tulu3-subsets"
DATASET_NAME="${DATA_DIR:-/path/to/datasets}/RLVR-IFeval,${DATA_DIR:-/path/to/datasets}/IF_multi_constraints_upto5"
SPLIT="train"
OUTPUT_DIR="$WORKDIR/checkpoints/rlvr"
MAX_COMPLETION_LENGTH=1024
NUM_GENERATIONS=8
MAX_STEPS=3000
PER_DEVICE_BATCH_SIZE=4
SAVE_STEPS=300
GRADIENT_ACCUMULATION_STEPS=4
GRADIENT_CHECKPOINTING=True
BF16=True
OPTIM="adamw_torch"
LEARNING_RATE=1e-6
MIN_LEARNING_RATE=1e-7
LR_SCHEDULER_TYPE="cosine_with_min_lr"
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
KL_BETA=0.1
LOSS_TYPE="grpo"
REWARD_TYPE="ifeval"
TEMPERATURE=0.8
EVAL_RATIO=0.01
EVAL_STEPS=100

# Load API keys from .env
source "$WORKDIR/.env"
export WANDB_API_KEY
export WANDB_PROJECT="lmalign-2026-rlvr-grpo-ifeval"

DATE=$(date +%Y%m%d%H%M%S)
LOG_DIR="$WORKDIR/rlvr-scripts/logs"
mkdir -p $LOG_DIR

DATASET_NAME_SHORT=$(echo $DATASET_NAME | sed 's|[^,]*/||g' | tr ',' '+')

# Start vLLM server
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../modules/trl:${PYTHONPATH}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python3 -m trl.cli vllm-serve \
    --model $MODEL_NAME \
    --port 8000 \
    --max-model-len 2048 \
    --gpu_memory_utilization 0.9 \
    --enable-prefix-caching True &
VLLM_PID=$!

echo "Waiting for vLLM server (PID: $VLLM_PID) to start..."
HOSTNAME=$(hostname -f)
while ! curl -s "http://${HOSTNAME}:8000/health/" > /dev/null 2>&1; do
    sleep 3
done
echo "vLLM server started"

# Run GRPO
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6, \
accelerate launch --num_processes=7 --gpu_ids="0,1,2,3,4,5,6" grpo_open_instruct.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --split $SPLIT \
    --output_dir $OUTPUT_DIR \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --num_generations $NUM_GENERATIONS \
    --max_steps $MAX_STEPS \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --bf16 $BF16 \
    --optim $OPTIM \
    --learning_rate $LEARNING_RATE \
    --min_learning_rate $MIN_LEARNING_RATE \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --beta $KL_BETA \
    --loss_type $LOSS_TYPE \
    --reward_type $REWARD_TYPE \
    --temperature $TEMPERATURE \
    --use_vllm \
    --vllm_mode "server" \
    --eval_ratio $EVAL_RATIO \
    --eval_steps $EVAL_STEPS \
    --vllm_server_host "${HOSTNAME}" \
    --vllm_server_port 8000 \
    --resume_from_checkpoint True \
    >> ${LOG_DIR}/grpo_$(basename ${MODEL_NAME})_${DATASET_NAME_SHORT}_${MAX_STEPS}_steps_${LEARNING_RATE}_${DATE}.log 2>&1

# Kill vLLM server and all its child processes
kill -- -$(ps -o pgid= -p $VLLM_PID | tr -d ' ') 2>/dev/null
echo "vLLM server stopped."
