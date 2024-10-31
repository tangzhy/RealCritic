# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-critic"

# Qwen2.5-Math-1.5B-Instruct
MODEL_NAME_OR_PATH="qwen-max-latest"
DATA_DIR="./data"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR