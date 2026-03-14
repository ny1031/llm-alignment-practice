# RLVR Scripts — GRPO with Verifiable Rewards

TRL의 `GRPOTrainer `를 사용하여 SFT를 진행한 Qwen3-1.7B-Base 모델에 대해 GRPO(Group Relative Policy Optimization)를 수행
Open-instruct의 verifier를 reward function으로 사용하여, 모델의 응답이 정답/제약 조건을 만족하는지에 따른 Reward 부여.

### RLVR Tasks
- **IF-RLVR**: Instruction Following 제약 조건
- **Math-RLVR**: 수학 문제 풀이 (GSM8K + MATH)

## 파일 구조

```
rlvr-scripts/
├── grpo_open_instruct.py                  # IF-RLVR 학습 스크립트 (IFEvalVerifier)
├── run_grpo_open_instruct_qwen3_1.7b.sh   # IF-RLVR 실행 셸 스크립트
├── grpo_math.py                           # Math-RLVR 학습 스크립트 (MathVerifier)
├── run_grpo_math_qwen3_1.7b.sh            # Math-RLVR 실행 셸 스크립트
├── logs/                                  # 학습 로그 (gitignored)
└── wandb/                                 # W&B 로그 (gitignored)
```

## 작동 방식

### 전체 흐름

```
┌─────────────────────────────────────────────────────────┐
│  GPU 7: vLLM Server (추론 전용)                            │ 
│  - trl vllm-serve로 모델 서빙                              │
│  - 학습 중 weight sync로 최신 파라미터 반영                    │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (generation) + NCCL (weight sync)
┌──────────────────────▼──────────────────────────────────┐
│  GPU 0-6: GRPO Training (accelerate multi-GPU)          │
│                                                         │
│  매 step:                                                │
│  1. vLLM 서버에 프롬프트 전송 → completion 생성 (G=8)         │
│  2. Verifier로 각 completion의 reward 계산                 │
│  3. Group 내 reward 기반으로 advantage 계산                 │
│  4. Policy gradient로 모델 업데이트                         │
│  5. 업데이트된 weight를 vLLM 서버에 NCCL로 sync               │
└─────────────────────────────────────────────────────────┘
```

### GRPO 학습 루프 상세

각 training step에서:

1. **Generation**: 배치에서 프롬프트를 가져와 vLLM 서버로 전송. 프롬프트당 `num_generations`(8)개의 completion을 샘플링.
2. **Reward 계산**: open-instruct verifier가 각 completion을 평가. 태스크별로 다른 verifier 사용 (아래 참조).
3. **Advantage 계산**: 같은 프롬프트에서 나온 G개 completion의 reward를 정규화하여 advantage 산출. Reward가 모두 같은 그룹(std=0)은 학습에 기여하지 않음.
4. **Policy Gradient**: advantage가 높은 completion의 확률을 높이고, 낮은 completion의 확률을 낮추는 방향으로 업데이트.
5. **KL Penalty**: `beta` > 0이면 reference model 대비 KL divergence를 penalty로 추가하여 과도한 policy 변화를 방지.

### GPU 할당과 Weight Sync

```
vLLM 서버:  CUDA_VISIBLE_DEVICES=7         (GPU 7에서만 실행)
학습:       CUDA_VISIBLE_DEVICES=0,1,...,6  (GPU 0-6 보이게 설정)
            accelerate --gpu_ids="0,...,6"  (실제 학습은 GPU 0-6)
```

---

## IF-RLVR (`grpo_open_instruct.py`)

Instruction Following 제약 조건을 학습.

### 데이터셋 & Reward

두 가지 IF 데이터셋의 single/multi constraints reward를 분기로 처리 :

| 데이터셋 | ground_truth 형식 | Verifier | Reward |
|---|---|---|---|
| [RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval) (14.9K) | `{"func_name": "validate_lowercase", "N": null, ...}` | `IFEvalVerifierOld` | 단일 제약 → binary (0 or 1) |
| [IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5) (95.3K) | `[{"instruction_id": [...], "kwargs": [...]}]` | `IFEvalVerifier` | 다중 제약 → 만족 비율 (0~1) |

`ground_truth`가 `{`로 시작하면 단일 제약(Old), `[`로 시작하면 다중 제약(New)으로 dispatch.

### 다중 데이터셋 혼합

`--dataset_name`에 쉼표로 구분된 경로를 전달하면:
1. 각 데이터셋을 로드
2. 공통 컬럼만 추출 (교집합)
3. `concatenate_datasets`로 합친 뒤 `shuffle(seed=42)`
4. `eval_ratio` 비율로 train/eval split

### 실행 방법

```bash
cd rlvr-scripts
bash run_grpo_open_instruct_qwen3_1.7b.sh
```

---

## Math-RLVR (`grpo_math.py`)

수학 문제 풀이 능력을 학습. RLVR-MATH와 RLVR-GSM을 혼합하여 학습하고, `MathVerifier`로 정답 여부를 검증.

### 데이터셋 & Reward

| 데이터셋 | 크기 | ground_truth 형식 | 용도 |
|---|---|---|---|
| [RLVR-MATH](https://huggingface.co/datasets/allenai/RLVR-MATH) | 7,500 (train only) | 정답 문자열 (e.g. `"8"`) | train + eval 분리 |
| [RLVR-GSM](https://huggingface.co/datasets/allenai/RLVR-GSM) | 7,473 train / 1,319 test | 정답 숫자 (e.g. `"1104"`) | train / test 그대로 사용 |

`MathVerifier`가 모델 응답에서 정답을 추출하여 ground_truth와 비교. 정답이면 1.0, 오답이면 0.0.

### 데이터셋 혼합 & Eval 구성

RLVR-MATH에는 test split이 없으므로, 다음과 같이 eval set을 구성:

```
RLVR-MATH train (7,500)
    ├── 95% → train (~7,125)  ─┐
    └──  5% → eval  (~375)    ─┤
                               ├── 합쳐서 shuffle
RLVR-GSM train (7,473) ────────┘         │
RLVR-GSM test  (1,319) ──────────────────┘

Train: MATH train + GSM train (~14,598)
Eval:  MATH eval + GSM test   (~1,694)
```

두 데이터셋의 공통 컬럼(`messages`, `ground_truth`, `dataset`)만 사용.

### 실행 방법

```bash
cd rlvr-scripts
bash run_grpo_math_qwen3_1.7b.sh
```

---

## Hyperparameters

IF-RLVR과 Math-RLVR 모두 동일한 하이퍼파라미터를 사용:

| 파라미터 | 값 | 설명 |
|---|---|---|
| `per_device_batch_size` | 4 | GPU당 프롬프트 수 |
| `num_generations` | 8 | 프롬프트당 생성 수 |
| `gradient_accumulation_steps` | 4 | Gradient 누적 횟수 |
| `max_completion_length` | 1024 | 최대 생성 토큰 수 |
| `beta` | 0.1 | KL divergence penalty 계수 |
| `temperature` | 0.8 | 생성 temperature |
| `learning_rate` | 1e-6 | 최대 학습률 |
| `lr_scheduler_type` | cosine_with_min_lr | 스케줄러 |
| IF-RLVR `max_steps` | 3000 (약 3 epochs) | 총 학습 step 수 |
| MATH-RLVR `max_steps` | 400 (약 3 epochs) | 총 학습 step 수 |

Effective batch size = `per_device_batch_size` × `num_processes` × `gradient_accumulation_steps` = 4 × 7 × 4 = **112 prompts/step**
