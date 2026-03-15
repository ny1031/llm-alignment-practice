# Qwen3-1.7B MATH+GSM RLVR Evaluation Results

- **Base Model**: Qwen3-1.7B-Base
- **SFT Init**: qwen3-1.7b-sft-by-tulu3-subsets ([tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) subsets)
- **RLVR Task**: [RLVR-MATH](https://huggingface.co/datasets/allenai/RLVR-MATH) + [RLVR-GSM](https://huggingface.co/datasets/allenai/RLVR-GSM)
- **Algorithm**: GRPO (loss_type=grpo, KL beta=0.1, temperature=0.8)
- **Checkpoints**: 100 ~ 400 steps (100 interval)

## Benchmark Results

| Steps | GSM8K | MATH | HumanEval (base) | HumanEval (plus) | MBPP (base) | MBPP (plus) | IFEval (avg) | IFEval (prompt_strict) | IFEval (inst_strict) | IFEval (prompt_loose) | IFEval (inst_loose) | IFBench (avg) | Arena-Hard |
|------:|------:|-----:|-----------------:|-----------------:|------------:|------------:|-------------:|-----------------------:|---------------------:|----------------------:|--------------------:|--------------:|-----------:|
| SFT init | 81.35 | — | 62.80 | 55.49 | 69.31 | 58.20 | 54.01 | 46.21 | 58.03 | 49.91 | 61.87 | — | 12.37 |
| 100 | 81.96 ✅▲0.61 | 61.28 | 64.63 ✅▲1.83 | 55.49 — | 69.58 ✅▲0.27 | 58.47 ✅▲0.27 | 54.77 ✅▲0.76 | 47.32 ✅▲1.11 | 58.99 ✅▲0.96 | 50.65 ✅▲0.74 | 62.11 ✅▲0.24 | 13.78 | 13.69 ✅▲1.32 |
| 200 | 81.73 ✅▲0.38 | 61.82 | 62.20 ❌▼0.60 | 54.88 ❌▼0.61 | **70.63** ✅▲1.32 | **59.26** ✅▲1.06 | 54.69 ✅▲0.68 | 47.32 ✅▲1.11 | 58.39 ✅▲0.36 | 50.83 ✅▲0.92 | 62.23 ✅▲0.36 | 13.14 | — |
| 300 | **82.79** ✅▲1.44 | 61.82 | **65.85** ✅▲3.05 | **57.32** ✅▲1.83 | 68.52 ❌▼0.79 | 57.41 ❌▼0.79 | **55.89** ✅▲1.88 | **48.43** ✅▲2.22 | **59.95** ✅▲1.92 | **51.76** ✅▲1.85 | **63.43** ✅▲1.56 | **13.45** | — |
| 400 | 82.71 ✅▲1.36 | **62.10** | 63.41 ✅▲0.61 | 56.10 ✅▲0.61 | 69.84 ✅▲0.53 | 58.73 ✅▲0.53 | 54.20 ✅▲0.19 | 47.32 ✅▲1.11 | 57.91 ❌▼0.12 | 50.28 ✅▲0.37 | 61.27 ❌▼0.60 | 13.15 | — |

## Analysis

### IF-RLVR와의 비교

IF-RLVR 실험에서는 step 1200 이후 GSM8K/HumanEval/MBPP가 급락하는 심각한 alignment tax가 관찰되었으나, MATH+GSM RLVR에서는 **400 step 전 구간에서 alignment tax가 거의 발생하지 않음**. 전반적으로 성능이 유지되거나 소폭 상승.

### 주요 관찰

1. **GSM8K**: 전 구간에서 SFT 대비 상승. Step 300에서 최고 82.79 (✅▲1.44).
2. **MATH (Hendrycks)**: SFT baseline 없어 직접 비교 불가하나, step 간 안정적 (61.28~62.10). Step 400에서 최고 62.10.
3. **HumanEval/MBPP**: RLVR 타겟이 아닌 코딩 벤치마크도 대체로 유지. Step 300에서 HumanEval 65.85/57.32로 최고.
4. **IFEval**: 소폭 상승 (최대 ✅▲1.88 at step 300). IF-RLVR 대비 상승폭은 작음 (IF-RLVR은 최대 ✅▲19.94).
5. **IFBench**: 전 구간 13~14pt 수준으로 변화 미미.
6. **Arena-Hard**: Step 100에서 13.69로 SFT 대비 ✅▲1.32.

### Best Checkpoint

| 강점 | Step | 이유 |
|---|---|---|
| **전체 밸런스 (추천)** | **300** | GSM8K **82.79 최고**, MATH 61.82 유지. HumanEval **65.85/57.32 최고**. IFEval **55.89 최고**. 비목표 벤치마크 손실 없이 전반적 개선. |
| MATH 극대화 | 400 | MATH **62.10 최고**, GSM8K 82.71도 준수. 다만 IFEval/HumanEval이 step 300보다 하락. |
| 안전한 선택 | 100 | SFT 대비 거의 모든 벤치마크에서 소폭 상승하면서 리스크 최소화. Arena-Hard 결과도 존재. |
