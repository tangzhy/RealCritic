import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example

SYSTEM_PROMPT_TEMPLATES = {
    "direct-cot": "Please reason step by step, and put your final answer within \\boxed{}.",
    "cot-critic": "You will be provided with a question and an incorrect reasoning. First, you need to point out the errors in the reasoning to form feedback, and then correct the erroneous reasoning based on the feedback you provide. Put your final answer within \\boxed{{}}.",
    "cot-judging-critic": "You will be provided with a question and a reasoning. The reasoning may be correct or incorrect. You need to make a judgment; if you think the reasoning is correct, put the answer within \\boxed{{}}. If you think the reasoning is wrong, first you need to point out the errors in the reasoning to form feedback, and then correct the erroneous reasoning based on the feedback you provide. Put your final answer within \\boxed{{}}.",
    "critic_and_correct": r"""
I will provide a math problem along with a student solution.

Conduct a step-by-step critique of the student's solution. For each step, use proper verification, recalculation, and reflection to determine whether it is logically and mathematically valid. Please elaborate on the analysis process carefully. If an error is detected in any step, describe the nature and cause of the error in detail, and suggest how to correct it or provide the correct approach. Once a step is found to contain an error, stop further analysis of subsequent steps (as they may depend on the identified error) and directly conclude with "Incorrect".

Finally, provide a conclusion of your critique and a correction for the student solution.

Format your response as follows:

```
## Critique of Student Solution Steps
### Critique of Step 1
[critique of step 1]

...

### Critique of Step n
[critique of step n]

## Conclusion of Critique
[For correct student solution]
The solution is Correct.
The first error step number: -1

[For incorrect student solution]
The solution is Incorrect.
The first error step number: [first step containing error]

## Correction of Student Solution
[For correct student solution]
The student solution is correct and well-reasoned. The final answer is \boxed{[answer]}.

[For incorrect student solution]
Here's the corrected solution, following the original approach up to step [error step - 1] and then proceeding with the proper correction:
===CORRECTION_START===
[Steps 1 to (error step - 1) from original student solution]
[Corrected version of error step]
[Subsequent steps following from the correction to reach the final answer]
Therefore, the final answer is \boxed{[corrected answer]}.
===CORRECTION_END===
```
""".strip(), 
    "tora": "Please integrate natural language reasoning with python programs to solve the problem above, and put your final answer within \\boxed{}.",
    "tora-critic": "You will be provided with a question and an incorrect reasoning. First, you need to point out the errors in the reasoning to form feedback, and then correct the erroneous reasoning based on the feedback you provide. Please integrate natural language reasoning with python programs to critic the reasoning above, and put your final answer within \\boxed{}.",
    "tora-judging-critic": "You will be provided with a question and a reasoning. The reasoning may be correct or incorrect. You need to make a judgment; if you think the reasoning is correct, put the answer within \\boxed{{}}. If you think the reasoning is wrong, first you need to point out the errors in the reasoning to form feedback, and then correct the erroneous reasoning based on the feedback you provide. Please integrate natural language reasoning with python programs to critic the reasoning above, and put your final answer within \\boxed{}."
}

MULTI_TURN_CRITIC = {
    "cot": "Are you sure your critic is correct? Please reconsider all the content above and identify any possible errors. Place the final answer in \\boxed{{}}.",
    "tora": "Are you sure your critic is correct? Please integrate natural language reasoning with python programs to reconsider all the content above and identify any possible errors. Place the final answer in \\boxed{{}}."
}

USER_PROMPT_TEMPLATE = {
    "default-cot": "{question}",
    "default-critic": "## Question\n\n{question}\n\nReasoning\n\n{reasoning}", 
    "critic_and_correct": r"""
The following is the data for your task:

## Math Problem
{question}

## Student Solution
{reasoning}
""".strip()
}

def construct_message(args, example=None, message=[], assistant = None):
    if len(message) == 0:
        prompt_type = args.prompt_type
        user_prompt_type = args.user_prompt_type
        assert example is not None
        prompt_template = SYSTEM_PROMPT_TEMPLATES[prompt_type]
        user_prompt_template = USER_PROMPT_TEMPLATE[user_prompt_type]
        if "critic" in user_prompt_type:
            if isinstance(example["reasoning"], list):
                assert len(example["reasoning"]) == 1
                example["reasoning"] = example["reasoning"][0]
            user_prompt = user_prompt_template.format(question=example["question"], reasoning=example["reasoning"])
        elif "cot" in user_prompt_type:
            user_prompt = user_prompt_template.format(question=example["question"])
        else:
            raise NotImplementedError
        message = [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt}
        ]
    else:
        message.append({"role": "assistant", "content": assistant})
        # TODO：多轮critic的prompt也需要能够定制
        if "cot" in args.prompt_type:
            message.append({"role": "user", "content": MULTI_TURN_CRITIC["cot"]})
        elif "tora" in args.prompt_type:
            message.append({"role": "user", "content": MULTI_TURN_CRITIC["tora"]})
        else:
            raise
    return message

def construct_prompt(message, tokenizer):
    return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

#     if prompt_type == "tora":
#         full_prompt = (
#             """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

# - Analyze the question and write functions to solve the problem; the function should not take any arguments.
# - Present the final result in LaTeX using a `\boxed{}` without any units.
# - Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

# Here are some examples you may refer to:

# ---

# """
#             + full_prompt
#         )

    return full_prompt.strip(" ")  # important!


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
