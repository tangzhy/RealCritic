# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-cot"

# Qwen2.5-Math-1.5B-Instruct
MODEL_NAME_OR_PATH="/220049033/huggingface/hub/models--Qwen--Qwen2.5-Math-72B-Instruct/snapshots/8fcf92b1eac8eca48c3d49778ad3dd1b394c9a17/"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2.5-Math-7B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2.5-Math-72B-Instruct
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-72B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH


# # Evaluate Qwen2-Math-Instruct
# PROMPT_TYPE="qwen-boxed"

# # Qwen2-Math-1.5B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-1.5B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2-Math-7B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-7B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2-Math-72B-Instruct
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-72B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
