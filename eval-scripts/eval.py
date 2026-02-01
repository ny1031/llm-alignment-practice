import os
import argparse
import logging
import signal
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../modules/Evaluator/packages/nemo-evaluator/src"))

from nemo_evaluator.api import check_endpoint, evaluate, show_available_tasks
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
)

# show_available_tasks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-0.6B")
    parser.add_argument("--completions_url", type=str, default="http://instance-32550.prj-2516.ws-1223.svc.cluster.local:8000/v1/completions")
    parser.add_argument("--result_dir", type=str, default="/data/ib-huawei-nas-lmt_980/users/lana/workspace/lmalign-2026/llm-alignment-practice/eval-results")
    parser.add_argument("--endpoint_type", type=str, default="completions")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)


    os.makedirs(f"{args.result_dir}/{args.model_name}/{args.task}-results", exist_ok=True)

    endpoint_type = EndpointType.COMPLETIONS if args.endpoint_type == "completions" else EndpointType.CHAT

    target_config = EvaluationTarget(
        api_endpoint=ApiEndpoint(
            url=args.completions_url, type=endpoint_type, model_id=args.model_name
        )
    )

    eval_config = EvaluationConfig(
        type=args.task,
        params=ConfigParams(temperature=args.temperature, top_p=args.top_p, parallelism=args.parallelism),
        output_dir=f"{args.result_dir}/{args.model_name}/{args.task}-results",
    )

    completions_results = evaluate(target_cfg=target_config, eval_cfg=eval_config)

    print(completions_results)

