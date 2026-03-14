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

from open_instruct.ground_truth_utils import MathVerifier


def make_reward_fn():
    """Wrap MathVerifier into a TRL reward function."""
    verifier = MathVerifier()

    def reward_fn(prompts, completions, ground_truth, **kwargs):
        rewards = []
        for completion, gt in zip(completions, ground_truth):
            try:
                if isinstance(completion, list):
                    response = completion[-1]["content"]
                else:
                    response = str(completion)

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


def load_and_mix_datasets(math_path, gsm_path, math_eval_ratio, seed=42):
    """Load RLVR-MATH and RLVR-GSM, create train/eval splits.

    - RLVR-MATH: train split only → split into train + eval
    - RLVR-GSM: train split for training, test split for eval
    - Train: MATH train + GSM train (shuffled)
    - Eval: MATH eval + GSM test (shuffled)
    """
    math_ds = load_dataset(math_path, split="train")
    gsm_train = load_dataset(gsm_path, split="train")
    gsm_test = load_dataset(gsm_path, split="test")

    logging.info(f"RLVR-MATH: {len(math_ds)} examples")
    logging.info(f"RLVR-GSM train: {len(gsm_train)}, test: {len(gsm_test)}")

    # Split MATH into train/eval
    math_split = math_ds.train_test_split(test_size=math_eval_ratio, seed=seed)
    math_train = math_split["train"]
    math_eval = math_split["test"]
    logging.info(f"RLVR-MATH split → train: {len(math_train)}, eval: {len(math_eval)}")

    # Use common columns only (MATH has extra constraint_type, constraint columns)
    common_cols = sorted(
        set(math_train.column_names)
        & set(gsm_train.column_names)
        & set(gsm_test.column_names)
    )
    logging.info(f"Common columns: {common_cols}")

    train_dataset = concatenate_datasets([
        math_train.select_columns(common_cols),
        gsm_train.select_columns(common_cols),
    ]).shuffle(seed=seed)

    eval_dataset = concatenate_datasets([
        math_eval.select_columns(common_cols),
        gsm_test.select_columns(common_cols),
    ]).shuffle(seed=seed)

    logging.info(f"Combined train: {len(train_dataset)}, eval: {len(eval_dataset)}")
    return train_dataset, eval_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--math_dataset", type=str, required=True,
                        help="Path to RLVR-MATH dataset")
    parser.add_argument("--gsm_dataset", type=str, required=True,
                        help="Path to RLVR-GSM dataset")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=300)
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
                        choices=["server", "colocate"])
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--vllm_max_model_length", type=int, default=None)
    parser.add_argument("--math_eval_ratio", type=float, default=0.05,
                        help="Fraction of RLVR-MATH to hold out for evaluation")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Run evaluation every N steps")
    parser.add_argument("--log_completions", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from, or 'True' to resume from latest")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    model_name = args.model_name.split("/")[-1]
    dataset_name = "RLVR-MATH+RLVR-GSM"

    # Load and mix datasets
    train_dataset, eval_dataset = load_and_mix_datasets(
        args.math_dataset, args.gsm_dataset, args.math_eval_ratio,
    )

    # Filter out samples where the prompt exceeds 2048 tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_prompt_tokens = 2048

    def token_length(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return len(tokenizer.encode(text)) <= max_prompt_tokens

    before_train = len(train_dataset)
    train_dataset = train_dataset.filter(token_length)
    logging.info(f"Train filtered: {before_train} -> {len(train_dataset)}")

    before_eval = len(eval_dataset)
    eval_dataset = eval_dataset.filter(token_length)
    logging.info(f"Eval filtered: {before_eval} -> {len(eval_dataset)}")

    train_dataset = prepare_dataset(train_dataset)
    eval_dataset = prepare_dataset(eval_dataset)

    reward_fn = make_reward_fn()

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
            resume_ckpt = True
        elif resume_ckpt.lower() == "false":
            resume_ckpt = None

    trainer.train(resume_from_checkpoint=resume_ckpt)
