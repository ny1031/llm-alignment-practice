import os
import argparse
import logging
import sys
sys.path.insert(0, "/data/ib-huawei-nas-lmt_980/users/lana/workspace/lmalign-2026/llm-alignment-practice/modules/trl")
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="/data/ib-huawei-nas-lmt_980/users/lana/workspace/lmalign-2026/llm-alignment-practice/checkpoints/sft")
    parser.add_argument("--chat_template_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--assistant_only_loss", type=bool, default=False)
    parser.add_argument("--packing", type=bool, default=False)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]

    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.select(range(min(1000000, len(dataset))))

    trainer_config = SFTConfig(
        run_name=f"{model_name}_{dataset_name}_{args.max_steps}",
        output_dir=f"{args.output_dir}/{model_name}_{dataset_name}_{args.max_steps}",
        chat_template_path=args.chat_template_path,

        max_steps=args.max_steps,                      # max steps
        
        # Batch Size
        per_device_train_batch_size=args.per_device_batch_size,         
        per_device_eval_batch_size=args.per_device_batch_size,         
        gradient_accumulation_steps=args.gradient_accumulation_steps,         # gradient accumulation (effective bsz = 4 * 4 = 16)
        
        # Learning Rate & Scheduler
        lr_scheduler_type=args.lr_scheduler_type,            # scheduler: linear, cosine, constant, etc.
        learning_rate=args.learning_rate,                    # max learning rate
        lr_scheduler_kwargs={"min_lr_rate": args.min_learning_rate},
        warmup_ratio=args.warmup_ratio,                      # warmup ratio 
        
        # Optimizer
        optim=args.optim,                                   # optimizer
        weight_decay=args.weight_decay,                     # weight decay
        
        # Logging & Saving
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="tensorboard",
        
        # Mixed Precision
        bf16=args.bf16,                             # bfloat16 
        
        # Gradient Checkpointing 
        gradient_checkpointing=args.gradient_checkpointing,
        
        # Sequence Length 
        max_length=args.max_seq_length,                   # max sequence length
        packing=args.packing,
        assistant_only_loss=args.assistant_only_loss,
        dataset_num_proc=16,                              # tokenizing parallel processing
        )

    trainer = SFTTrainer(
        model=args.model_name,
        args=trainer_config,
        train_dataset=dataset,
    )
    trainer.train()