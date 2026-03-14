# SFT Scripts — Supervised Fine-Tuning with TRL

TRL의 `SFTTrainer`를 사용하여 Qwen3-1.7B-Base 모델에 대해 Supervised Fine-Tuning을 수행
HuggingFace Hub 또는 로컬 경로의 chat 데이터셋을 로드하고, chat template을 적용하여 학습한다.

## 파일 구조

```
sft-scripts/
├── sft.py                        # SFT 학습 스크립트 (TRL SFTTrainer)
├── run_sft_qwen3_1.7b_base.sh   # Qwen3-1.7B 학습 실행 셸 스크립트
└── logs/                         # 학습 로그 (gitignored)
```

## 작동 방식

### 전체 흐름

```
┌─────────────────────────────────────────────────────────────┐
│  accelerate launch --num_processes=8                        │
│                                                             │
│  1. 데이터셋 로드 (HuggingFace Hub 또는 로컬 경로)                 │
│     - 최대 1,000,000 샘플로 제한                                │
│  2. Chat template 적용하여 토크나이징 (48 workers 병렬)           │
│  3. Packing으로 시퀀스를 max_seq_length까지 연결                  │
│  4. 8-GPU DDP로 SFT 학습                                      │
│  5. save_steps 간격으로 체크포인트 저장                           │
└─────────────────────────────────────────────────────────────┘
```

### Important Notes

- **Chat Template + Assistant-Only Loss**: `--chat_template_path`로 커스텀 chat template을 지정함으로써 **loss를 assistant turn에만 적용**. 
- 커스텀 template(`Qwen3_assistant_only`)은 assistant 응답 부분을 `{%- generation %}...{%- endgeneration %}` 태그로 감싸므로 TRL의 `SFTTrainer`는 이 태그를 파싱하여 해당 구간의 토큰에만 loss mask를 생성한다. 
- `--assistant_only_loss True`와 함께 사용하면 system/user 턴은 loss 계산에서 제외되어 모델이 **응답 생성 능력만 학습**하게 됨.

  ```
  토크나이즈된 시퀀스:
  <|im_start|>user\n질문<|im_end|>\n<|im_start|>assistant\n{%- generation %}답변{%- endgeneration %}<|im_end|>
  Loss mask:
  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1]
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^
                                                      generation 태그 안쪽만 loss 계산
  ```

- **Packing**: `--packing True`이면 짧은 시퀀스를 하나의 `max_seq_length` 시퀀스로 연결하여 GPU utilization을 높임.
- **Gradient Checkpointing**: 메모리 절약을 위해 중간 activation을 저장하지 않고 backward pass에서 재계산.

### TRL Module Import

TRL은 pip 설치가 아닌 submodule 직접 참조 방식으로 사용:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../modules/trl"))
from trl import SFTTrainer, SFTConfig
```

## 실행 방법

```bash
cd sft-scripts
bash run_sft_qwen3_1.7b_base.sh
```

1. 하이퍼파라미터 설정
2. `accelerate launch --num_processes=8`로 8-GPU 분산 학습 시작
3. 로그를 `logs/`에 저장

## Hyperparameters

| 파라미터 | 값 | 설명 |
|---|---|---|
| `model_name` | Qwen3-1.7B-Base | 베이스 모델 |
| `dataset_name` | [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | SFT 데이터셋 |
| `per_device_batch_size` | 4 | GPU당 배치 크기 |
| `gradient_accumulation_steps` | 4 | Gradient 누적 횟수 |
| `max_seq_length` | 4096 | 최대 시퀀스 길이 |
| `max_steps` | 7000 | 총 학습 step 수 |
| `save_steps` | 1000 | 체크포인트 저장 간격 |
| `learning_rate` | 2e-5 | 최대 학습률 |
| `min_learning_rate` | 1e-6 | 최소 학습률 |
| `lr_scheduler_type` | cosine_with_min_lr | 스케줄러 |
| `warmup_ratio` | 0.1 | Warmup 비율 |
| `weight_decay` | 0.01 | Weight decay |
| `assistant_only_loss` | True | Assistant 턴에만 loss 계산 |
| `packing` | True | 시퀀스 패킹 |

Effective batch size = `per_device_batch_size` × `num_processes` × `gradient_accumulation_steps` = 4 × 8 × 4 = **128 samples/step**

## 체크포인트 저장 경로

```
checkpoints/sft/Qwen3-1.7B-Base_tulu-3-sft-mixture_7000/
├── checkpoint-1000/
├── checkpoint-2000/
├── ...
└── checkpoint-7000/
```
