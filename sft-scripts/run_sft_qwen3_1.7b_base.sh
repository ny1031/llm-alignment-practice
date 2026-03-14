WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_NAME="${MODEL_DIR:-/path/to/models}/Qwen3-1.7B-Base"
DATASET_NAME="${DATA_DIR:-/path/to/datasets}/tulu-3-sft-mixture"
CHAT_TEMPLATE_PATH="${DATA_DIR:-/path/to/data}/chat_templates/Qwen3_assistant_only"
SPLIT="train"
OUTPUT_DIR="$WORKDIR/checkpoints/sft"
MAX_SEQ_LENGTH=4096
MAX_STEPS=7000
PER_DEVICE_BATCH_SIZE=4
SAVE_STEPS=1000
GRADIENT_ACCUMULATION_STEPS=4
GRADIENT_CHECKPOINTING=True
BF16=True
OPTIM="adamw_torch"
LEARNING_RATE=2e-5
MIN_LEARNING_RATE=1e-6
LR_SCHEDULER_TYPE="cosine_with_min_lr"
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
ASSISTANT_ONLY_LOSS=True
PACKING=True

DATE=$(date +%Y%m%d%H%M%S)
LOG_DIR="$WORKDIR/sft-scripts/logs"
mkdir -p $LOG_DIR

accelerate launch --num_processes=8 sft.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --split $SPLIT \
    --chat_template_path $CHAT_TEMPLATE_PATH \
    --output_dir $OUTPUT_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
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
    --assistant_only_loss $ASSISTANT_ONLY_LOSS \
    --packing $PACKING \
    >> ${LOG_DIR}/sft_$(basename ${MODEL_NAME})_$(basename ${DATASET_NAME})_${SPLIT}_${MAX_STEPS}_steps_${LEARNING_RATE}_${MIN_LEARNING_RATE}_${DATE}.log 2>&1
