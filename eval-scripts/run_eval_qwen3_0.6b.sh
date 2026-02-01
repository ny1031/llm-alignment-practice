MODEL_NAME="Qwen3-0.6B"
COMPLETIONS_URL="http://instance-32550.prj-2516.ws-1223.svc.cluster.local:8080/v1/completions"

# Timeout 설정 (초 단위) - 기본값이 너무 짧아서 늘림
export AIOHTTP_CLIENT_TIMEOUT=300
export LM_HARNESS_REQUEST_TIMEOUT=300
RESULT_DIR="/data/ib-huawei-nas-lmt_980/users/lana/workspace/lmalign-2026/llm-alignment-practice/eval-results"
LOG_DIR="/data/ib-huawei-nas-lmt_980/users/lana/workspace/lmalign-2026/llm-alignment-practice/eval-scripts/logs"
ENDPOINT_TYPE="completions"
TASK=$1     # "humaneval", "mbpp"
PARALLELISM=1  # 타임아웃 발생시 1로 줄이기
TEMPERATURE=0.7
TOP_P=0.8

DATE=$(date +%Y%m%d%H%M%S)
# max_saved_requests=5
# gen_kwargs: {'until': ['Question:', '</s>', '<|im_end|>'], 'do_sample': False, 'temperature': 0.7, 'top_p': 0.8}
# generation_kwargs: {'temperature': 0.7, 'top_p': 0.8} specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!
python3 eval.py --model_name $MODEL_NAME --completions_url $COMPLETIONS_URL --result_dir $RESULT_DIR \
    --endpoint_type $ENDPOINT_TYPE --task $TASK --parallelism $PARALLELISM --temperature $TEMPERATURE --top_p $TOP_P \
    >> $LOG_DIR/eval_${MODEL_NAME}_${TASK}_${DATE}.log 2>&1