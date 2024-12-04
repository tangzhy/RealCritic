# 定制prompt

在utils.py中，存在三个PROMPT_TEMPLATES集，需要定制自己的prompt可以找到对应的集合，在里面加入自己的prompt。

### 定制system prompt

在SYSTEM_PROMPT_TEMPLATES添加自己的prompt。prompt总体上可以分为cot和critic两种。如果要添加critic类型的prompt，记得prompt type里一定要带上"critic"。代码需要检查prompt type里是否有"critic"字段，有的话会跳转到critic对应的逻辑。



### 定制user prompt

在USER_PROMPT_TEMPLATE添加自己的prompt。同样，prompt总体上可以分成cot和critic两种。其中cot类型只需要填入question，critic类型需要填入question和reasoning。在构造自己的user prompt template时，cot类型的prompt，user prompt type要带上"cot"字段，critic类型的prompt同理，要带上"critic"字段。



# 一键启动脚本

``` bash
export CUDA_VISIBLE_DEVICES=...
MODEL_NAME_OR_PATH=...
PROMPT_TYPE=xx-cot
USER_PROMPT_TYPE=xxx-cot
OUTPUT_PATH=... #需要自己制定output path，如果output path不存在，则在outputs文件夹下拼接该output path然后makedirs
DATA_NAME="arc-challenge,college_math,gpqa,gsm8k,math,minerva_math,mmlu_stem,olympiadbench"
# cot
python3 -u math_eval.py \
	--data_dir data/wrong_data \
	--model_name_or_path ${MODEL_NAME_OR_PATH \
	--output_dir ${OUTPUT_PATH} \
	--prompt_type ${PROMPT_TYPE} \
	--user_prompt_type ${USER_PROMPT_TYPE} \
	--data_name ${DATA_NAME} \
	--use_vllm \
	--save_outputs \
	--overwrite
	

# benchmark 1: 全错的critic
PROMPT_TYPE=xx-critic
USER_PROMPT_TYPE=xxx-critic
OUTPUT_PATH=...
python3 -u math_eval.py \
	--data_dir data/wrong_data \
	--model_name_or_path ${MODEL_NAME_OR_PATH \
	--output_dir ${OUTPUT_PATH} \
	--prompt_type ${PROMPT_TYPE} \
	--user_prompt_type ${USER_PROMPT_TYPE} \
	--data_name ${DATA_NAME} \
	--use_vllm \
	--save_outputs \
	--overwrite
	
# benchmark 2: 一半错一半对的critic，选择自己要的prompt type（默认是cot judging critic），主要是改data_dir，改成data/random
PROMPT_TYPE=xx-critic
USER_PROMPT_TYPE=xxx-critic
OUTPUT_PATH=...
python3 -u math_eval.py \
	--data_dir data/random \
	--model_name_or_path ${MODEL_NAME_OR_PATH \
	--output_dir ${OUTPUT_PATH} \
	--prompt_type ${PROMPT_TYPE} \
	--user_prompt_type ${USER_PROMPT_TYPE} \
	--data_name ${DATA_NAME} \
	--use_vllm \
	--save_outputs \
	--overwrite
	
# self-critic，改data_dir，改成cot运行完后的output_path，那里会保存一个matric json和一个test json，程序会读取里面的test json进行critic
PROMPT_TYPE=xx-critic
USER_PROMPT_TYPE=xxx-critic
OUTPUT_PATH=...
python3 -u math_eval.py \
	--data_dir cot/output/path \
	--model_name_or_path ${MODEL_NAME_OR_PATH \
	--output_dir ${OUTPUT_PATH} \
	--prompt_type ${PROMPT_TYPE} \
	--user_prompt_type ${USER_PROMPT_TYPE} \
	--data_name ${DATA_NAME} \
	--use_vllm \
	--save_outputs \
	--overwrite

# post check
python3 -u model_eval_critic.py \
	--data_dir path/for/critic/math_eval \
	--model_name_or_path /post/check/model \
	--output_dir /output/dir/ \
	--data_name ${DATA_NAME} \
	--seed 0 \
	--temperature 0 \
	--top_p 1 \
```

下面的API和多轮critic暂时不要运行，还没来得及测试user prompt逻辑改变带来的影响。

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
export CUDA_VISIBLE_DEVICES=0
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