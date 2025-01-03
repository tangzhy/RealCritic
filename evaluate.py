import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]
    
    scores = []
    timeout_cnt = 0 
    previous_scores = [sample.get('previous_score', [False])[0] for sample in samples]

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1)

    idx = 0
    score_mat = []
    for sample in samples:
        score_name = "score" if "critic" in prompt_type else "previous_score"
        sample[score_name] = scores[idx: idx+len(sample['pred'])]
        assert len(sample[score_name]) == len(sample['pred'])
        score_mat.append(sample[score_name])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0]
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample[score_name][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    if "critic" in prompt_type:
        # add critic_information
        originally_correct = sum(1 for x in previous_scores if x)
        originally_incorrect = len(previous_scores) - originally_correct
            
        correct_to_correct = sum(1 for prev, curr in zip(previous_scores, scores) if prev and curr)
        correct_to_incorrect = sum(1 for prev, curr in zip(previous_scores, scores) if prev and not curr)
        incorrect_to_correct = sum(1 for prev, curr in zip(previous_scores, scores) if not prev and curr)
        incorrect_to_incorrect = sum(1 for prev, curr in zip(previous_scores, scores) if not prev and not curr)
        

        result_json['critic_information'] = {
            "total_samples": len(previous_scores),
            "originally_correct": originally_correct,
            "originally_incorrect": originally_incorrect,
            "correct_to_correct_count": correct_to_correct,
            "correct_to_incorrect_count": correct_to_incorrect,
            "incorrect_to_correct_count": incorrect_to_correct,
            "incorrect_to_incorrect_count": incorrect_to_incorrect,
            "correct_to_correct_ratio": (correct_to_correct/originally_correct*100) if originally_correct else 0,
            "correct_to_incorrect_ratio": (correct_to_incorrect/originally_correct*100) if originally_correct else 0,
            "incorrect_to_correct_ratio": (incorrect_to_correct/originally_incorrect*100) if originally_incorrect else 0,
            "incorrect_to_incorrect_ratio": (incorrect_to_incorrect/originally_incorrect*100) if originally_incorrect else 0
        }

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
