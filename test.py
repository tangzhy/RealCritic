from model_utils import load_hf_lm_and_tokenizer, generate_completions, get_client_response
from functools import partial

get_client_response_paltial = partial(get_client_response, model_name="qwen_max_latest", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", critic=True)

current_prompt = {"idx": 1, "prompt": "jjjdjj"}

get_client_response_paltial(client_prompt = current_prompt)

print("k")