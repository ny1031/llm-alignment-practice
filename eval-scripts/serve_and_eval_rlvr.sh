#!/bin/bash
set -e

# ============================================================
# vLLM 서버 띄우기 + NeMo Skills 평가 통합 스크립트
# GPU별 vLLM 인스턴스를 띄워 벤치마크를 병렬 평가 (pseudo-DP)
#
# Usage:
#   bash serve_and_eval_rlvr.sh                  # 300~3000, 300 간격
#   bash serve_and_eval_rlvr.sh 1200             # 1200 스텝만
#   bash serve_and_eval_rlvr.sh 900 1800         # 900~1800, 300 간격
#   bash serve_and_eval_rlvr.sh 900 1800 600     # 900~1800, 600 간격
#
# Environment variables:
#   CKPT_DIR  — 체크포인트 디렉토리 (default: $WORKDIR/checkpoints/rlvr/qwen3-1.7b-sft-by-tulu3-subsets_RLVR-IFeval+IF_multi_constraints_upto5_3000)
# ============================================================

export WORKDIR=$(dirname $(realpath $0))/..

# --- 공통 설정 ---
STEP_START=${1:-300}
STEP_END=${2:-$STEP_START}
STEP_INTERVAL=${3:-300}

# 인자가 0개면 전체 범위
if [ $# -eq 0 ]; then
    STEP_START=300
    STEP_END=3000
    STEP_INTERVAL=300
fi

export PHASE=RLVR-MATH-GSM
export MODEL_NAME=Qwen3-1.7B
CKPT_DIR=${CKPT_DIR:-$WORKDIR/checkpoints/rlvr/qwen3-1.7b-sft-by-tulu3-subsets_RLVR-MATH+RLVR-GSM_400}
BASE_PORT=8080

# --- GPU별 벤치마크 분배 ---
# 7개 벤치마크를 GPU 7장에 분산
GPU_IDS=(0 1 2 3 4 5 6)
BENCHMARKS_PER_GPU=(
    "ifeval"
    "ifbench"
    "gsm8k"
    "hendrycks_math"
    "human-eval"
    "mbpp"
    "arena-hard"
)
NUM_WORKERS=${#BENCHMARKS_PER_GPU[@]}

# --- 평가 설정 ---
export NEMO_SKILLS_CONFIG_DIR=$WORKDIR/configs
export NEMO_SKILLS_DATA_DIR=$WORKDIR/datasets
export RESULT_DIR=$WORKDIR/eval-results
export LOG_DIR=$WORKDIR/eval-scripts/logs
# Load API keys from .env
source $WORKDIR/.env
export OPENAI_API_KEY
export HUGGING_FACE_HUB_TOKEN

mkdir -p $NEMO_SKILLS_CONFIG_DIR
mkdir -p $NEMO_SKILLS_DATA_DIR
mkdir -p $LOG_DIR

HOSTNAME=$(hostname -f)

# --- 데이터셋 준비 (없으면 자동 다운로드) ---
PREPARE_DATE=$(date +%Y%m%d%H%M%S)
ALL_BENCHMARKS=$(printf "%s " "${BENCHMARKS_PER_GPU[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' ')
NEED_PREPARE=false
for bench in $ALL_BENCHMARKS; do
    if [ ! -f "$NEMO_SKILLS_DATA_DIR/$bench/test.jsonl" ]; then
        echo "[$(date)] 데이터셋 미준비: $bench (test.jsonl 없음) — prepare_data 실행"
        NEED_PREPARE=true
        break
    fi
done
# --- antlr4 버전 전환 (ns는 omegaconf → antlr4==4.9.3 필요, 학습은 4.11 필요) ---
echo "[$(date)] antlr4 → 4.9.3 (omegaconf 호환) 전환..."
pip install -q 'antlr4-python3-runtime==4.9.3'

if [ "$NEED_PREPARE" = true ]; then
    echo "[$(date)] ns prepare_data 시작..."
    ns prepare_data --cluster=local --data_dir $NEMO_SKILLS_DATA_DIR $ALL_BENCHMARKS \
        >> $LOG_DIR/prepare_data_$PREPARE_DATE.log 2>&1
    echo "[$(date)] ns prepare_data 완료"
fi

# ============================================================
# 헬퍼 함수: 단일 vLLM 인스턴스 시작 + health check
# ============================================================
start_vllm_instance() {
    local GPU_ID=$1
    local PORT=$2
    local STEPS=$3
    local DATE=$4
    local VLLM_LOG=$LOG_DIR/vllm_gpu${GPU_ID}_${MODEL_NAME}_${PHASE}_${STEPS}_${DATE}.log

    echo "[$(date)] [Step ${STEPS}] GPU ${GPU_ID}, 포트 ${PORT}에서 vLLM 시작..."

    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve \
        --model $CKPT_DIR/checkpoint-${STEPS} \
        --served-model-name $MODEL_NAME \
        --port $PORT \
        --gpu-memory-utilization 0.9 \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --chat-template $WORKDIR/chat-templates/qwen3/qwen3_nonthinking.jinja \
        >> $VLLM_LOG 2>&1 &

    _LAST_VLLM_PID=$!
}

wait_for_health() {
    local PORT=$1
    local PID=$2
    local STEPS=$3
    local GPU_ID=$4
    local MAX_WAIT=600
    local WAITED=0
    local INTERVAL=10

    while [ $WAITED -lt $MAX_WAIT ]; do
        if ! kill -0 $PID 2>/dev/null; then
            echo "[$(date)] [Step ${STEPS}] ERROR: GPU ${GPU_ID} vLLM 프로세스 종료됨"
            return 1
        fi
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "[$(date)] [Step ${STEPS}] GPU ${GPU_ID} (포트 ${PORT}) 준비 완료! (${WAITED}초)"
            return 0
        fi
        sleep $INTERVAL
        WAITED=$((WAITED + INTERVAL))
    done

    echo "[$(date)] [Step ${STEPS}] ERROR: GPU ${GPU_ID} 서버 타임아웃 (${MAX_WAIT}초)"
    kill $PID 2>/dev/null
    return 1
}

# ============================================================
# 헬퍼 함수: 모든 vLLM 인스턴스 종료
# ============================================================
stop_all_vllm() {
    local STEPS=$1
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local PID=${VLLM_PIDS[$i]}
        if [ -n "$PID" ] && kill -0 $PID 2>/dev/null; then
            echo "[$(date)] [Step ${STEPS}] GPU ${GPU_IDS[$i]} vLLM 종료 (PID: $PID)"
            kill $PID 2>/dev/null
            wait $PID 2>/dev/null || true
        fi
    done
}

# ============================================================
# 메인 루프
# ============================================================
echo "============================================================"
echo "평가 대상 스텝: $(seq $STEP_START $STEP_INTERVAL $STEP_END | tr '\n' ' ')"
echo "GPU ${NUM_WORKERS}장 병렬 평가"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "  GPU ${GPU_IDS[$i]} (포트 $((BASE_PORT + i))): ${BENCHMARKS_PER_GPU[$i]}"
done
echo "============================================================"

for STEPS in $(seq $STEP_START $STEP_INTERVAL $STEP_END); do
    echo ""
    echo "============================================================"
    echo "[$(date)] Step ${STEPS} 평가 시작"
    echo "============================================================"

    DATE=$(date +%Y%m%d%H%M%S)
    mkdir -p $RESULT_DIR/$MODEL_NAME-${PHASE}-${STEPS}

    # --- 1. 모든 vLLM 인스턴스 시작 ---
    VLLM_PIDS=()
    ALL_STARTED=true
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        PORT=$((BASE_PORT + i))
        start_vllm_instance ${GPU_IDS[$i]} $PORT $STEPS $DATE
        VLLM_PIDS+=($_LAST_VLLM_PID)
    done

    # --- 2. 모든 인스턴스 health check ---
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        PORT=$((BASE_PORT + i))
        if ! wait_for_health $PORT ${VLLM_PIDS[$i]} $STEPS ${GPU_IDS[$i]}; then
            ALL_STARTED=false
            break
        fi
    done

    if [ "$ALL_STARTED" = false ]; then
        echo "[$(date)] [Step ${STEPS}] 서버 시작 실패, 다음 스텝으로 넘어갑니다."
        stop_all_vllm $STEPS
        continue
    fi

    # --- 3. 벤치마크 병렬 실행 ---
    echo "[$(date)] [Step ${STEPS}] 벤치마크 병렬 평가 시작..."
    EVAL_PIDS=()
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        PORT=$((BASE_PORT + i))
        SERVER_ADDR="http://${HOSTNAME}:${PORT}/v1"
        EVAL_LOG=$LOG_DIR/eval_gpu${GPU_IDS[$i]}_${MODEL_NAME}_${PHASE}_${STEPS}_${DATE}.log

        echo "[$(date)] [Step ${STEPS}] GPU ${GPU_IDS[$i]}: ${BENCHMARKS_PER_GPU[$i]} → 포트 ${PORT}"

        ns eval \
            --cluster=local \
            --server_type=vllm \
            --model=$MODEL_NAME \
            --server_address=$SERVER_ADDR \
            --benchmarks=${BENCHMARKS_PER_GPU[$i]} \
            --output_dir=$RESULT_DIR/$MODEL_NAME-${PHASE}-${STEPS} \
            --expname=eval_${PHASE}_${STEPS}_gpu${GPU_IDS[$i]} \
            ++inference.temperature=0.7 \
            ++inference.top_p=0.8 \
            ++inference.top_k=20 \
            ++inference.min_p=0 \
            ++inference.tokens_to_generate=1024 \
            ++inference.endpoint_type=chat \
            >> $EVAL_LOG 2>&1 &

        EVAL_PIDS+=($!)
    done

    # --- 4. 모든 평가 완료 대기 ---
    EVAL_FAILED=false
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        if ! wait ${EVAL_PIDS[$i]}; then
            echo "[$(date)] [Step ${STEPS}] WARNING: GPU ${GPU_IDS[$i]} 평가 실패 (${BENCHMARKS_PER_GPU[$i]})"
            EVAL_FAILED=true
        fi
    done

    if [ "$EVAL_FAILED" = true ]; then
        echo "[$(date)] [Step ${STEPS}] 일부 평가 실패. 로그를 확인하세요."
    else
        echo "[$(date)] [Step ${STEPS}] 모든 평가 완료!"
    fi

    # --- 5. 서버 종료 ---
    stop_all_vllm $STEPS
done

# --- antlr4 버전 복원 (학습용) ---
echo "[$(date)] antlr4 → 4.11 (MathVerifier 호환) 복원..."
pip install -q 'antlr4-python3-runtime==4.11'

echo ""
echo "============================================================"
echo "[$(date)] 모든 스텝 평가 완료!"
echo "============================================================"
