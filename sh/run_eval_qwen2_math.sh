# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-critic"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="/220049033/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/"
DATA_DIR="/220049033/project/xzy/data/critic_benchmark/bad_solution_data/500"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR

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
