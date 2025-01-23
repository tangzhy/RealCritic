#!/bin/bash

PROMPT_TYPE="critic_and_correct"
USER_PROMPT_TYPE="critic_and_correct"
MODEL_NAME_OR_PATH="path/to/your/model"
DATA_DIR="data/mix_data"
DATASETS="arc-challenge,college_math,gpqa,gsm8k,math,minerva_math,mmlu_stem,olympiadbench"
MULTI_TURN=1

export CUDA_VISIBLE_DEVICES="0,1"

bash sh/local/run_cross_critic.sh $PROMPT_TYPE $USER_PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR $DATASETS $MULTI_TURN

bash sh/local/run_self_critic.sh $PROMPT_TYPE $USER_PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR $DATASETS $MULTI_TURN

bash sh/local/run_direct_cot.sh $PROMPT_TYPE $USER_PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR $DATASETS