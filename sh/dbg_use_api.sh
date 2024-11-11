# WARNING!!! Plz first start the api service!!!
# sh sh/serve_qwen2.5-math-7b-instruct_api.sh

# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-critic"

# Qwen2.5-Math-1.5B-Instruct
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
export DASHSCOPE_API_KEY="EMPTY"
BASE_URL="http://localhost:8000/v1"
DATA_DIR="./data"

MAX_SAMPLE=300
OUTPUT_DIR=${MAX_SAMPLE}/${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="arc-challenge"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --data_dir ${DATA_DIR} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_api \
    --save_outputs \
    --overwrite \
    --critic \
    --base_url $BASE_URL