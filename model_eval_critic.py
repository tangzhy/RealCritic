import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from smashed.utils.io_utils import recursively_list_files
import numpy as np
from utils import save_jsonl

PROMPT_TEMPLATE = """You are a mathematics expert. You will be provided with a question, a solution, and a critique given by a teacher model. The critique will follow this format: If the reasoning is found to be incorrect, first identify the errors in the reasoning to form feedback, and then correct the erroneous reasoning based on the provided feedback.

Your task is to evaluate whether the teacher model's critique exhibits any of the following non-compliant behaviors:

1. The model does not adhere to the paradigm of critiquing before providing its own solution; instead, it solves the problem first and provides a critique afterward.
2. During the critique, the model is capable of identifying errors but fails to correct the flawed reasoning based on the identified mistakes; instead, it circumvents the pointed-out issues in a deceptive manner, providing its own solution.

If any of the above non-compliant behaviors are present, the final answer should be \\boxed{Unqualified}.

If there are no such non-compliant behaviors, the final answer should be \\boxed{Qualified}.

Please reason step-by-step with your analysis and provide the final answer with \\boxed{Unqualified} or \\boxed{Qualified}. You can output only these two results."""

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--multi_turn",type=int, default=1)
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args

def extract_answer(pred_str):
    pred = ""
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    return pred

def loads_data(data_dir, data_name):
    data = []
    file_list = [path for path in recursively_list_files(data_dir)]
    for file in file_list:
        # TODO: 之后在这里补上区分multi_turn的逻辑，或者什么情况下全部读入
        file_data_name = file.split('/')[-2]
        if data_name == file_data_name and "metrics" not in file:
            with open(file, "r", encoding='utf-8') as fr:
                for line in fr:
                    line = json.loads(line)
                    data.append(line)
            break 
    return data

def load_vllm(args):
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    return llm, tokenizer

def construct_prompt(json_data, tokenizer):
    user = json_data["question"] + "\n\nCritique\n\n" + json_data["critic"]
    message = [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {"role": "user", "content": user}
    ]

    return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

def setup(args):
    llm, tokenizer = load_vllm(args)
    data_list = args.data_names.split(',')
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args)) 
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def main(llm, tokenizer, data_name, args):
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    datas = loads_data(args.data_dir, data_name)
    prompts = []
    for json_data in datas:
        prompts.append(construct_prompt(json_data, tokenizer))
    print(prompts[0])
    outputs = llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens_per_call,
                        n=1,
                        stop=stop_words,
                        stop_token_ids=(
                            [151645, 151643]
                            if "qwen2" in args.model_name_or_path.lower()
                            else None
                        ),
                    ),
                )
    outputs = sorted(
        outputs, key=lambda x: int(x.request_id)
    )  # sort outputs by request_id
    outputs = [output.outputs[0].text for output in outputs]

    qualified_num, unsolve_num = 0, 0
    
    for idx, output in enumerate(outputs):
        pred_answer = extract_answer(output)
        if pred_answer == "Unqualified":
            datas[idx]["model_eval_critic_score"] = False
        elif pred_answer == "Qualified":
            datas[idx]["model_eval_critic_score"] = True
            qualified_num += 1
        else:
            datas[idx]["model_eval_critic_score"] = False
            unsolve_num += 1

    for idx, output in enumerate(outputs):
        datas[idx]["model_eval_critic"] = output
    
    output_dir = args.output_dir
    output_dir += args.data_dir
    if "model_eval_critic" not in output_dir:
        output_dir += "/model_eval_critic"
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file_prefix = f"seed{args.seed}_t{args.temperature}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(datas, out_file)

    result_json = {"num_samples": len(datas), "num_scores": len(datas) - unsolve_num, "acc": round(qualified_num / len(datas) * 100, 1)}
    with open(
        out_file.replace(".jsonl", f"_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
