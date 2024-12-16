import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
from pebble import ProcessPool
from transformers import AutoTokenizer
import json
from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_message, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions, get_client_response
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="math_output", type=str)
    parser.add_argument("--prompt_type", default="direct-cot", type=str)
    parser.add_argument("--user_prompt_type", default="default-critic", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--multi_turn",type=int, default=1)
    parser.add_argument("--multi_turn_critic_prompt", type=str, default="cot")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    output_dir = args.output_dir
    if "math_eval" not in output_dir:
        output_dir += "/math_eval"
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    if "critic" in args.prompt_type:
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_turn{args.multi_turn}.jsonl"
    else:
        out_file = f"{output_dir}/{data_name}/test.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    processed_samples = []
    if not args.overwrite:
        processed_samples = []

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    if args.multi_turn > 1 and "tora" in args.prompt_type:
        print("The program has automatically exited because it is not recommended to use tora for multi-turn critic.")
        exit()
    if args.use_vllm:
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
    elif args.use_api:
        llm = None
        tokenizer = None
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
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


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    messages = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        message = construct_message(args, example)
        messages.append(message)
        if args.use_api:
            # 统一框架产生的冗余代码，使用api的时候不需要full prompt
            full_prompt = example["question"]
        else:
            full_prompt = construct_prompt(message, tokenizer)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": example["gt_cot"],
            "gt": example["gt"],
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)
    
    ##
    # samples = samples[:50]
    # messages = messages[:50]
    ##

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    input_messages = [
        message for message in messages for _ in range(args.n_sampling)
    ]
    input_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]
    remain_prompts = input_prompts
    input_messages = [(i, message) for i, message in enumerate(input_messages)]
    remain_messages = input_messages
    end_prompts = []
    end_messages = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tora", "tora-critic", "tora-judging-critic"]:
        stop_words.extend(["\n\n---", "```output", "\n---", "```\n"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    # measure time use
    start_time = time.time()
    multi_turn = args.multi_turn if "critic" in args.prompt_type else 1
    for turn in range(multi_turn):
        for epoch in range(max_func_call):
            print("-" * 20, "Epoch", epoch)
            current_prompts = remain_prompts
            current_messages = remain_messages
            if len(current_prompts) == 0:
                break
            prompts = [item[1] for item in current_prompts]
            if args.use_vllm:
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
            elif args.use_api:
                outputs = []
                timeout_cnt = 0
                client_messages = [{"idx": item[0], "message": item[1]} for item in current_messages]
                get_client_response_paltial = partial(get_client_response, args=args, stop=stop_words)
                batch_size = 100
                with ProcessPool(max_workers=10) as pool:
                    for i in range(0, len(client_messages), batch_size):
                        batch_messages = client_messages[i: i + batch_size]
                        future = pool.map(get_client_response_paltial, batch_messages, timeout=240)
                        iterator = future.result()
                        with tqdm(total=len(batch_messages), desc="Evaluate") as progress_bar:
                            while True:
                                try:
                                    result = next(iterator)
                                    outputs.append(result)
                                except StopIteration:
                                    break
                                except TimeoutError as error:
                                    print(error)
                                    timeout_cnt += 1
                                except Exception as error:
                                    print(error)
                                    exit()
                                progress_bar.update(1)
                outputs = sorted(outputs, key=lambda x: x[0])
                outputs = [output[1] for output in outputs]
            else:
                outputs = generate_completions(
                    model=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_tokens_per_call,
                    batch_size=16,
                    stop_id_sequences=stop_words,
                )

            assert len(outputs) == len(current_prompts)

            # process all outputs
            remain_prompts = []
            remain_codes = []
            remain_messages = []
            for (i, query), (j, message), output in zip(current_prompts, current_messages, outputs):
                assert i == j
                output = output.rstrip()
                if "```python" in output:
                    remain_prompts.append((i, query))
                    message.append({"role": "assistant", "content": output})
                    output = extract_program(output)
                    remain_codes.append(output)
                    remain_messages.append((i, message))
                elif args.prompt_type == "direct-cot":
                    if turn != multi_turn - 1:
                        message = construct_message(args=args, message=message, assistant=output)
                        if not args.use_api:
                            query = construct_prompt(message, tokenizer)
                        end_prompts.append((i, query))
                    else:
                        end_prompts.append((i, query + output))
                    end_messages.append((i, message))
                elif "boxed" not in output and output.endswith("```"):
                    program = extract_program(query)
                    remain_prompts.append((i, query))
                    remain_codes.append(program)
                    remain_messages.append((i, message))
                else:
                    if turn != multi_turn - 1:
                        message = construct_message(args=args, message=message, assistant=output)
                        if not args.use_api:
                            query = construct_prompt(message, tokenizer)
                        end_prompts.append((i, query))
                    else:
                        end_prompts.append((i, query + output))
                    end_messages.append((i, message))

            # execute the remain prompts
            remain_results = executor.batch_apply(remain_codes)
            for k in range(len(remain_prompts)):
                i, query = remain_prompts[k]
                i, message = remain_messages[k]
                res, report = remain_results[k]
                exec_result = res if res else report
                if "pal" in args.prompt_type:
                    exec_result = "\\boxed{" + exec_result + "}"
                exec_result = f"\n```output\n{exec_result}\n```\n"
                query += exec_result
                message.append({"role": "user", "content": exec_result})
                # not end
                if epoch == max_func_call - 1:
                    query += "\nReach max function call limit."
                remain_prompts[k] = (i, query)
                remain_messages[k] = (i, message)
        print("Unsolved samples:", len(remain_prompts))
        end_prompts.extend(remain_prompts)
        end_messages.extend(remain_messages)
        if turn != multi_turn - 1:
            input_prompts = end_prompts
            remain_prompts = end_prompts
            remain_messages = end_messages
            end_prompts = []
            end_messages = []
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    end_messages = sorted(end_messages, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i][1])[-1].strip()
        # for stop_word in stop_words:
        #     if stop_word in code:
        #         code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        message = messages[i]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        if "critic" in args.prompt_type:
            sample.update({"critic": code, "pred": preds, "report": reports, "message": message})
        else:
            sample.update({"reasoning": code, "pred": preds, "report": reports, "message": message})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
