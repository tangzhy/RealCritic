# WARNING!!! Plz first start the api service!!!
# sh sh/serve_qwen2.5-math-7b-instruct_api.sh

PROMPT_TYPE="api-judging-critic"

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-7B-Instruct"
export DASHSCOPE_API_KEY="EMPTY"
export DASHSCOPE_BASE_URL="http://0.0.0.0:8000/v1"
DATA_DIR="./outputs/api/api-cot/Qwen/Qwen2.5-7B-Instruct/math_eval"

OUTPUT_DIR=api/${PROMPT_TYPE}/${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="math"
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