#!/bin/bash

PROMPT_TYPE="direct-cot"
USER_PROMPT_TYPE="default-cot"
MODEL_NAME_OR_PATH="gpt-4o"
DATA_DIR="data/mix_data"
DATASETS="arc-challenge,college_math,gpqa,gsm8k,math,minerva_math,mmlu_stem,olympiadbench"
MULTI_TURN=1

export DASHSCOPE_API_KEY="your_api_key"
export DASHSCOPE_BASE_URL="https://api.dashscope.com/v1"

bash sh/run_direct_cot.sh $PROMPT_TYPE $USER_PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR $DATASETS

# bash sh/run_cross_critic.sh $PROMPT_TYPE $USER_PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR $DATASETS $MULTI_TURN

# bash sh/run_self_critic.sh $PROMPT_TYPE $USER_PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR $DATASETS $MULTI_TURN
