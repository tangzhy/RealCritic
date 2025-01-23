# 定制prompt

在utils.py中，存在三个PROMPT_TEMPLATES集，需要定制自己的prompt可以找到对应的集合，在里面加入自己的prompt。

### 定制system prompt

在SYSTEM_PROMPT_TEMPLATES添加自己的prompt。prompt总体上可以分为cot和critic两种。如果要添加critic类型的prompt，记得prompt type里一定要带上"critic"。代码需要检查prompt type里是否有"critic"字段，有的话会跳转到critic对应的逻辑。



### 定制user prompt

在USER_PROMPT_TEMPLATE添加自己的prompt。同样，prompt总体上可以分成cot和critic两种。其中cot类型只需要填入question，critic类型需要填入question和reasoning。在构造自己的user prompt template时，cot类型的prompt，user prompt type要带上"cot"字段，critic类型的prompt同理，要带上"critic"字段。



### 定制multi turn prompt

multi turn prompt用来在每一轮的开始添加进message的末尾，以某种方式合适的方式提示模型需要再次进行critic



# 一键启动脚本

所有示例脚本的位置在sh/中

只需要在config.sh中，按照示例填入对应的值，然后运行该脚本即可。



### api

``` sh 
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
```

值得说明的是，调用api评测时，model_name_or_path对应的是调用api时的模型名。



### 本地调用

``` sh
#!/bin/bash

PROMPT_TYPE="critic_and_correct"
USER_PROMPT_TYPE="critic_and_correct"
MODEL_NAME_OR_PATH="path/to/your/model"
DATA_DIR="data/mix_data"
DATASETS="arc-challenge,college_math,gpqa,gsm8k,math,minerva_math,mmlu_stem,olympiadbench"
MULTI_TURN=1

export CUDA_VISIBLE_DEVICES="0,1"

bash sh/local/run_cross_critic.sh $PROMPT_TYPE $USER_PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR $DATASETS $MULTI_TURN

# bash sh/local/run_self_critic.sh $PROMPT_TYPE $USER_PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR $DATASETS $MULTI_TURN

# bash sh/local/run_direct_cot.sh $PROMPT_TYPE $USER_PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_DIR $DATASETS
```







### post check

``` sh
python3 -u model_eval_critic.py \
	--data_dir path/for/critic/math_eval \
	--model_name_or_path /post/check/model \
	--output_dir /output/dir/ \
	--data_name ${DATA_NAME} \
	--seed 0 \
	--temperature 0 \
	--top_p 1 \
```



