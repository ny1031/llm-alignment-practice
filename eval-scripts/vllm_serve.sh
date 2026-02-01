# tested with 0.11.1+cu129
vllm serve --model /data/ib-a100-cluster-a-pri-lmt_967/models/Qwen3-0.6B --served-model-name Qwen3-0.6B \
    --port 8080 --gpu-memory-utilization 0.7 --tensor-parallel-size 4