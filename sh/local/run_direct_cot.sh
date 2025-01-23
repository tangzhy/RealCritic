#!/bin/bash

PROMPT_TYPE=$1
USER_PROMPT_TYPE=$2
MODEL_NAME_OR_PATH=$3
DATA_DIR=$4
DATASETS=$5

output_dir="outputs/multi_turn/direct_cot/${MODEL_NAME_OR_PATH}/turn1/math_eval"

PYTHONPATH=$(pwd) python math_eval.py \
    --data_dir "$DATA_DIR" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "$output_dir" \
    --data_name "$DATASETS" \
    --prompt_type "$PROMPT_TYPE" \
    --user_prompt_type "$USER_PROMPT_TYPE" \
    --multi_turn 1 \
    --use_vllm \
    --save_outputs \
    --overwrite 