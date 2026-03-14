# llm-alignment-practice
LLM Post-training(SFT → RLVR) 파이프라인 구축 및 평가 아카이브

## Pipeline Overview
- Qwen3-1.7B Base 모델로부터 post-training 진행

```
Qwen3-1.7B-Base
    │
    ▼  SFT (tulu-3-sft-mixture, 7000 steps)
Qwen3-1.7B-SFT
    │
    ▼  RLVR-IF (RLVR-IFeval + IF_multi_constraints_upto5, GRPO 3000 steps)
Qwen3-1.7B-RLVR
```

## Project Structure

```
├── sft-scripts/              # SFT 학습 (TRL SFTTrainer)
│   ├── sft.py
│   └── run_sft_qwen3_1.7b_base.sh
├── rlvr-scripts/             # RLVR 학습 (TRL GRPO + open-instruct verifier)
│   ├── grpo_open_instruct.py
│   └── run_grpo_open_instruct_qwen3_1.7b.sh
├── eval-scripts/             # 평가 (NeMo Skills + vLLM)
│   ├── serve_and_eval_sft.sh     # SFT 모델 서빙+평가 통합
│   └── serve_and_eval_rlvr.sh    # RLVR 체크포인트 서빙+평가 통합
├── chat-templates/           # Jinja chat templates (Qwen3)
├── configs/                  # NeMo Skills cluster config
├── datasets/                 # NeMo Skills benchmark data
├── reports/                  # 평가 결과 리포트
├── modules/                  # Git submodules
│   ├── trl/                  #   HuggingFace TRL
│   ├── Skills/               #   NVIDIA NeMo Skills
│   └── open-instruct/        #   Ai2 Open-Instruct (verifier)
├── checkpoints/              # 학습된 모델 체크포인트 (gitignored)
└── eval-results/             # 평가 결과 JSON (gitignored)
```

## Quick Start

### 환경 설정

```bash
# NeMo Skills 평가용
pip install -r requirements-nemo-skills.txt

# IFEval 평가 시 google-research 데이터 필요
sudo mkdir -p /opt/benchmarks && sudo chown -R $(whoami):$(whoami) /opt/benchmarks
git clone https://github.com/google-research/google-research.git /opt/benchmarks/google-research --depth=1
```

### SFT 학습

```bash
bash sft-scripts/run_sft_qwen3_1.7b_base.sh
```

8-GPU DDP, [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) 데이터셋, assistant-only loss + packing. 자세한 내용은 [sft-scripts/README.md](sft-scripts/README.md) 참조.

### RLVR 학습

```bash
cd rlvr-scripts
bash run_grpo_open_instruct_qwen3_1.7b.sh
```

GPU 0에서 vLLM 서버, GPU 1-7에서 GRPO 학습. [RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval) + [IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5) 데이터셋으로 IFEval verifier reward 기반 학습. 자세한 내용은 [rlvr-scripts/README.md](rlvr-scripts/README.md) 참조.

### 평가

```bash
# SFT 모델 평가 (GPU 7장 병렬, 벤치마크별 vLLM 인스턴스)
bash eval-scripts/serve_and_eval_sft.sh

# RLVR 체크포인트 평가 (step 300~3000)
bash eval-scripts/serve_and_eval_rlvr.sh
```

7개 벤치마크를 GPU별 TP=1 vLLM 인스턴스로 병렬 평가:

| GPU | 벤치마크 |
|-----|---------|
| 0 | IFEval |
| 1 | IFBench |
| 2 | GSM8K |
| 3 | MATH |
| 4 | HumanEval |
| 5 | MBPP |
| 6 | Arena-Hard |

## Evaluation Reports

- [reports/Qwen3-1.7B-Base-SFT.md](reports/Qwen3-1.7B-Base-SFT.md) — SFT 학습 평가 결과
- [reports/Qwen3-1.7B-Base-RLVR-IF.md](reports/Qwen3-1.7B-Base-RLVR-IF.md) — IF-RLVR 학습 평가 결과
