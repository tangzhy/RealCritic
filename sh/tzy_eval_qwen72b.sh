export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
MODEL_NAME_OR_PATH=../hf_cache/Qwen2.5-72B-Instruct
PROMPT_TYPE=direct-cot
USER_PROMPT_TYPE=default-cot
COT_OUTPUT_PATH=$MODEL_NAME_OR_PATH/critic_bench_cot #需要自己制定output path，如果output path不存在，则在outputs文件夹下拼接该output path然后makedirs
DATA_NAME="arc-challenge,college_math,gpqa,gsm8k,math,minerva_math,mmlu_stem,olympiadbench"
# DATA_NAME="math"
# cot
python3 -u math_eval.py \
	--data_dir data/wrong_data \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--output_dir ${COT_OUTPUT_PATH} \
	--prompt_type ${PROMPT_TYPE} \
	--user_prompt_type ${USER_PROMPT_TYPE} \
	--data_name ${DATA_NAME} \
	--use_vllm \
	--save_outputs \
	--overwrite
	

# benchmark 1: 全错的critic
PROMPT_TYPE=critic_and_correct
USER_PROMPT_TYPE=critic_and_correct
OUTPUT_PATH=$MODEL_NAME_OR_PATH/critic_bench_critic_on_wrong
python3 -u math_eval.py \
	--data_dir data/wrong_data \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--output_dir ${OUTPUT_PATH} \
	--prompt_type ${PROMPT_TYPE} \
	--user_prompt_type ${USER_PROMPT_TYPE} \
	--data_name ${DATA_NAME} \
	--use_vllm \
	--save_outputs \
	--overwrite
	
# benchmark 2: 一半错一半对的critic，选择自己要的prompt type（默认是cot judging critic），主要是改data_dir，改成data/random
PROMPT_TYPE=critic_and_correct
USER_PROMPT_TYPE=critic_and_correct
OUTPUT_PATH=$MODEL_NAME_OR_PATH/critic_bench_critic_on_half_correct_half_wrong
python3 -u math_eval.py \
	--data_dir data/random \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--output_dir ${OUTPUT_PATH} \
	--prompt_type ${PROMPT_TYPE} \
	--user_prompt_type ${USER_PROMPT_TYPE} \
	--data_name ${DATA_NAME} \
	--use_vllm \
	--save_outputs \
	--overwrite
	
# self-critic，改data_dir，改成cot运行完后的output_path，那里会保存一个matric json和一个test json，程序会读取里面的test json进行critic
PROMPT_TYPE=critic_and_correct
USER_PROMPT_TYPE=critic_and_correct
OUTPUT_PATH=$MODEL_NAME_OR_PATH/critic_bench_critic_on_self
echo $COT_OUTPUT_PATH/math_eval
python3 -u math_eval.py \
	--data_dir outputs/$COT_OUTPUT_PATH/math_eval \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--output_dir ${OUTPUT_PATH} \
	--prompt_type ${PROMPT_TYPE} \
	--user_prompt_type ${USER_PROMPT_TYPE} \
	--data_name ${DATA_NAME} \
	--use_vllm \
	--save_outputs \
	--overwrite

# # post check
# python3 -u model_eval_critic.py \
# 	--data_dir path/for/critic/math_eval \
# 	--model_name_or_path /post/check/model \
# 	--output_dir /output/dir/ \
# 	--data_name ${DATA_NAME} \
# 	--seed 0 \
# 	--temperature 0 \
# 	--top_p 1 \