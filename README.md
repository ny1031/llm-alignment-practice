# llm-alignment-practice
LLM Post-training(SFT, RLVR, RLHF) 파이프라인 구축 및 평가 실습 아카이브

### Install Requirements
```
pip install -r requirements.txt
```

## Evaluation
### Nemo Evaluator
run vllm sever
```
bash eval-scripts/vllm_seve.sh
```
run evaluation
```
bash eval-scripts/run_eval_qwen3_0.6b.sh
```

### Nemo-Skills
TBD

---
## Train
### TRL SFT
```
bash sft-scripts/run_sft_qwen3_0.6b_base.sh
```

---
## Evaluation Results
### Nemo-Evaluator
| Model | Benchmark | Score |
|-------|:-----------:|:-------:|
| Qwen3-0.6B | MBPP (pass@1) | 20.96 |
| Qwen3-0.6B | HumanEval (pass@1) | 21.31 |
| Qwen3-0.6B | GSM8K (em, strict) | 40.86 |
<<<<<<< HEAD
=======
| Qwen3-0.6B | IFEval (strict prompt) | 33.09 |
>>>>>>> 87ef9bb (nemo eval and trl sft)

### Nemo-Skills
TBD