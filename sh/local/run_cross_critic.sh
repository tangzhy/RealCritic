#!/bin/bash

PROMPT_TYPE=$1
USER_PROMPT_TYPE=$2
MODEL_NAME_OR_PATH=$3
DATA_DIR=$4
DATASETS=$5
MULTI_TURN=$6

output_dir="outputs/multi_turn/cross_critic/${MODEL_NAME_OR_PATH}/turn${MULTI_TURN}/math_eval"

PYTHONPATH=$(pwd) python math_eval.py \
    --data_dir "$DATA_DIR" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "$output_dir" \
    --data_name "$DATASETS" \
    --prompt_type "$PROMPT_TYPE" \
    --user_prompt_type "$USER_PROMPT_TYPE" \
    --multi_turn $MULTI_TURN \
    --use_vllm \
    --save_outputs \
    --overwrite 