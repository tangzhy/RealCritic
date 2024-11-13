#!/bin/sh

set -e 
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export CUDA_VISIBLE_DEVICES="0"

# model_name="meta-llama/Meta-Llama-3.1-70B-Instruct"
# model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name="Qwen/Qwen2.5-7B-Instruct"
model_path="/220049033/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/"
api_key="EMPTY"
num_gpus=1

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --enforce-eager