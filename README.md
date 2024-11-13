### API Debug

First start your api service, please change the model path or name as you like

```bash
sh sh/serve_qwen2.5-math-7b-instruct_api.sh
```

Then try to request the api to evaluate. 

- We change the batch_size per request to 1, since we are already doing multiprocessing at data point level, grouping data point into large batches seems unneccessary.
- We modify the get_client_reponse function which support temperature, top_p and so on. Otherwise, you may get totally unexpected results.

```bash
sh sh/dbg_use_api
```


### Requirements

You can install the required packages with the following command:

```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
```

### Evaluation

You can evaluate use API with the following command:

```bash
# for critic bench 1
PROMPT_TYPE="api-cot"

MODEL_NAME_OR_PATH=...
export DASHSCOPE_API_KEY=...
export DASHSCOPE_BASE_URL=...
DATA_DIR="./data/wrong_data"

OUTPUT_DIR=api/${PROMPT_TYPE}/${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="arc-challenge,college_math,gpqa,gsm8k,math,minerva_math,mmlu_stem,olympiadbench"
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
    
# for critic bench 2
PROMPT_TYPE="api-critic"

MODEL_NAME_OR_PATH=...
export DASHSCOPE_API_KEY=...
export DASHSCOPE_BASE_URL=...
DATA_DIR="./data/wrong_data"

OUTPUT_DIR=api/${PROMPT_TYPE}/${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="arc-challenge,college_math,gpqa,gsm8k,math,minerva_math,mmlu_stem,olympiadbench"
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
    
# for critic bench 3
PROMPT_TYPE="api-judging-critic"

MODEL_NAME_OR_PATH=...
export DASHSCOPE_API_KEY=...
export DASHSCOPE_BASE_URL=...
DATA_DIR="./data/random"

OUTPUT_DIR=api/${PROMPT_TYPE}/${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="arc-challenge,college_math,gpqa,gsm8k,math,minerva_math,mmlu_stem,olympiadbench"
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
    
# for critic bench 4
PROMPT_TYPE="api-judging-critic"

MODEL_NAME_OR_PATH=...
export DASHSCOPE_API_KEY=...
export DASHSCOPE_BASE_URL=...
DATA_DIR="critic/bench1/outputdir/math_eval"

OUTPUT_DIR=api/${PROMPT_TYPE}_self/${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="arc-challenge,college_math,gpqa,gsm8k,math,minerva_math,mmlu_stem,olympiadbench"
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
```

## Acknowledgement

The codebase is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).