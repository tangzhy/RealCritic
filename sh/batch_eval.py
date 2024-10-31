import os
import json
import click

@click.command()
@click.option(
    "--max_sample",
    type=int
)
@click.option(
    "--math_datas",
    type=str
)
def main(max_sample, math_datas):
    math_models = [
        {"Qwen2.5-Math-72B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-Math-72B-Instruct/snapshots/8fcf92b1eac8eca48c3d49778ad3dd1b394c9a17/"},
        {"Qwen2.5-72B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842/"},
        {"Llama-3.1-70B-Instruct": "/220049033/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393/"},
        {"Qwen2.5-Math-7B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/"},
        {"Qwen2.5-7B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/"},
        # {"Deepseek-math-7b-instruct": "/220049033/huggingface/hub/models--deepseek-ai--deepseek-math-7b-instruct/snapshots/0a5828f800a36df0fd7f0ed581b983246c0677ff/"},
        {"Llama-3.1-8B-Instruct": "/220049033/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"},
        # {"Qwen2.5-Math-1.5B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35/"},
        # {"Qwen2-Math-1.5B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2-Math-1.5B-Instruct/snapshots/19bd485b6474f7263545b4d5a1c29d78bcbf3a4f/"},
        # {"Qwen2-0.5B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/"},
        # {"Qwen1.5-0.5B-Chat": "/220049033/huggingface/hub/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/4d14e384a4b037942bb3f3016665157c8bcb70ea/"}
    ]

    data_dir = f"/220049033/project/xzy/data/critic_benchmark/bad_solution_data/{max_sample}"
    def run_command(model_name, model_path, data_name, data_path, prompt_type="cot", critic=False):
        output_dir = str(max_sample) + "/" + model_path + "/math_eval"
        if critic:
            command = f"PYTHONPATH=$(pwd) nohup python evaluation/math_eval.py --data_dir={data_path} --model_name_or_path={model_path} --output_dir={output_dir} --data_name={data_name} --prompt_type={prompt_type} --use_vllm --save_outputs --critic --overwrite> log/eval_{model_name}-{data_name}-critic.txt 2>&1"
        else:
            command = f"PYTHONPATH=$(pwd) nohup python evaluation/math_eval.py --data_dir={data_path} --model_name_or_path={model_path} --output_dir={output_dir} --data_name={data_name} --prompt_type={prompt_type} --use_vllm --save_outputs --overwrite > log/eval_{model_name}-{data_name}.txt 2>&1"
        print(command)
        os.system(command)

    for model in math_models:
        try:
            for (model_name, model_path) in model.items():
                for critic in (True, False):
                    if critic:
                        if "qwen" in model_name.lower():
                            run_command(model_name, model_path, math_datas, data_dir, prompt_type="qwen25-math-critic", critic=critic)
                        elif "llama-3.1" in model_name.lower():
                            run_command(model_name, model_path, math_datas, data_dir, prompt_type="llama-3.1-critic", critic=critic) 
                    else:
                        if "qwen" in model_name.lower():
                            run_command(model_name, model_path, math_datas, data_dir, prompt_type="qwen25-math-cot")
                        elif "llama-3.1" in model_name.lower():
                            run_command(model_name, model_path, math_datas, data_dir, prompt_type="llama-3.1-cot")
                        # elif "llama-3" in model_name.lower():
                        #     run_command(model_name, model_path, math_datas, data_dir, prompt_type="llama-3-cot")
                        # elif "deepseek" in model_name.lower():
                        #     run_command(model_name, model_path, math_datas, data_dir, prompt_type="deepseek-math")
                        # else:
                        #     raise
        except:
            break

if __name__ == "__main__":
    main()