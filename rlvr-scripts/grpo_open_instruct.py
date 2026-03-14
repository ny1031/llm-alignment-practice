import os
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../modules/trl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../modules/open-instruct"))

from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

# Stub out heavy transitive dependencies that ground_truth_utils pulls in
# via open_instruct.utils (beaker, ray, etc.)
import types
for _mod_name in ("beaker", "ray", "ray.util", "ray.util.state"):
    if _mod_name not in sys.modules:
        mod = types.ModuleType(_mod_name)
        sys.modules[_mod_name] = mod
sys.modules["ray"].util = sys.modules["ray.util"]
sys.modules["ray.util"].state = sys.modules["ray.util.state"]
sys.modules["ray"].ObjectRef = type("ObjectRef", (), {})

from open_instruct.ground_truth_utils import (
    GSM8KVerifier,
    MathVerifier,
    StrictMathVerifier,
    IFEvalVerifier,
    IFEvalVerifierOld,
    FlanVerifier,
    StringMatcherVerifier,
    F1Verifier,
    PuzzleMatcherVerifier,
)

VERIFIER_REGISTRY = {
    "gsm8k": GSM8KVerifier,
    "math": MathVerifier,
    "strict_math": StrictMathVerifier,
    "ifeval": IFEvalVerifier,
    "ifeval_old": IFEvalVerifierOld,
    "flan": FlanVerifier,
    "string_matcher": StringMatcherVerifier,
    "string_f1": F1Verifier,
    "puzzle": PuzzleMatcherVerifier,
}

def make_reward_fn(verifier_cls):
    """Wrap an open-instruct VerifierFunction into a TRL reward function.

    For ifeval reward_type, automatically dispatches between IFEvalVerifier
    (multi-constraint format) and IFEvalVerifierOld (single-constraint format)
    based on the ground_truth format.
    """
    verifier = verifier_cls()
    # If using IFEvalVerifier, also create IFEvalVerifierOld for single-constraint samples
    ifeval_old_verifier = None
    if verifier_cls is IFEvalVerifier:
        ifeval_old_verifier = IFEvalVerifierOld()

    def reward_fn(prompts, completions, ground_truth, **kwargs):
        rewards = []
        for completion, gt in zip(completions, ground_truth):
            try:
                if isinstance(completion, list):
                    response = completion[-1]["content"]
                else:
                    response = str(completion)

                # Dispatch based on ground_truth format
                if ifeval_old_verifier is not None and gt.lstrip().startswith("{"):
                    result = ifeval_old_verifier(
                        tokenized_prediction=[],
                        prediction=response,
                        label=gt,
                    )
                else:
                    result = verifier(
                        tokenized_prediction=[],
                        prediction=response,
                        label=gt,
                    )
                rewards.append(result.score)
            except Exception as e:
                logging.warning(f"Reward computation failed: {e}")
                rewards.append(0.0)
        return rewards

    return reward_fn

def prepare_dataset(dataset):
    """Convert messages format to prompt-only format for GRPOTrainer."""
    def convert(example):
        example["prompt"] = example["messages"]
        return example

    dataset = dataset.map(convert)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset path(s), comma-separated for mixing multiple datasets")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--reward_type", type=str, required=True,
                        choices=list(VERIFIER_REGISTRY.keys()),
                        help="Verifier type to use as reward function")
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--min_learning_rate", type=float, default=1e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL divergence coefficient (0 disables KL penalty)")
    parser.add_argument("--loss_type", type=str, default="grpo")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="server",
                        choices=["server", "colocate"],
                        help="vLLM integration mode: 'server' (separate vLLM process) or 'colocate' (in-process)")
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                        help="GPU memory utilization for colocate mode")
    parser.add_argument("--vllm_max_model_length", type=int, default=None,
                        help="Max model length for colocate mode")
    parser.add_argument("--eval_ratio", type=float, default=0.02,
                        help="Fraction of train data to hold out for evaluation")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Run evaluation every N steps")
    parser.add_argument("--log_completions", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from, or 'True' to resume from latest")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    model_name = args.model_name.split("/")[-1]

    dataset_paths = [p.strip() for p in args.dataset_name.split(",")]
    dataset_name = "+".join(p.split("/")[-1] for p in dataset_paths)

    datasets_list = []
    for path in dataset_paths:
        ds = load_dataset(path, split=args.split)
        logging.info(f"Loaded {path}: {len(ds)} examples")
        datasets_list.append(ds)

    if len(datasets_list) > 1:
        common_cols = set(datasets_list[0].column_names)
        for ds in datasets_list[1:]:
            common_cols &= set(ds.column_names)
        datasets_list = [ds.select_columns(list(common_cols)) for ds in datasets_list]
        dataset = concatenate_datasets(datasets_list).shuffle(seed=42)
        logging.info(f"Mixed dataset: {len(dataset)} total examples (columns: {list(common_cols)})")
    else:
        dataset = datasets_list[0]

    # Filter out samples where the prompt exceeds 2048 tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_prompt_tokens = 2048

    def token_length(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return len(tokenizer.encode(text)) <= max_prompt_tokens

    before = len(dataset)
    dataset = dataset.filter(token_length)
    logging.info(f"Filtered {before - len(dataset)} samples exceeding {max_prompt_tokens} tokens ({before} -> {len(dataset)})")

    split_dataset = dataset.train_test_split(test_size=args.eval_ratio, seed=42)
    train_dataset = prepare_dataset(split_dataset["train"])
    eval_dataset = prepare_dataset(split_dataset["test"])
    logging.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    reward_fn = make_reward_fn(VERIFIER_REGISTRY[args.reward_type])

    grpo_config = GRPOConfig(
        run_name=f"{model_name}_{dataset_name}_{args.max_steps}",
        output_dir=f"{args.output_dir}/{model_name}_{dataset_name}_{args.max_steps}",

        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,

        # Training steps
        max_steps=args.max_steps,

        # Batch Size
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Learning Rate & Scheduler
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        lr_scheduler_kwargs={"min_lr_rate": args.min_learning_rate},
        warmup_ratio=args.warmup_ratio,

        # Optimizer
        optim=args.optim,
        weight_decay=args.weight_decay,

        # GRPO-specific
        beta=args.beta,
        loss_type=args.loss_type,

        # Logging, Evaluation & Saving
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        report_to="wandb",
        log_completions=args.log_completions,

        # Mixed Precision
        bf16=args.bf16,

        # Gradient Checkpointing
        gradient_checkpointing=args.gradient_checkpointing,

        # vLLM acceleration
        # use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        # Server mode settings
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        # Colocate mode settings
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Handle resume_from_checkpoint
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is not None:
        if resume_ckpt.lower() == "true":
            resume_ckpt = True  # Auto-detect latest checkpoint
        elif resume_ckpt.lower() == "false":
            resume_ckpt = None
    
    trainer.train(resume_from_checkpoint=resume_ckpt)
