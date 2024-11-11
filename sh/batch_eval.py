import os
import click
import time

@click.command()
@click.option(
    "--math_datas",
    type=str
)
def main(math_datas):
    math_models = [
        # {"Qwen2.5-7B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/"},
        # {"Qwen2.5-Math-72B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-Math-72B-Instruct/snapshots/8fcf92b1eac8eca48c3d49778ad3dd1b394c9a17/"},
        {"Qwen2.5-72B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842/"},
        {"Llama-3.1-70B-Instruct": "/220049033/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393/"},
        {"Qwen2.5-Math-7B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/"},
        {"Llama-3.1-8B-Instruct": "/220049033/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"},
        {"mistral-123B": "/220049033/huggingface/hub/models--mistralai--Mistral-Large-Instruct-2407/snapshots/c6a230e7b1070914dd28bd6bc3b9c7c3d9e612ae/"}
    ]
    types = ["cot", "critic", "judging-critic"]
    # types = ["cot"]
    prompt_type = {
        "qwen": {"cot": "qwen25-math-cot", "critic": "qwen25-math-critic", "judging-critic": "qwen25-math-judging-critic"},
        "llama-3.1": {"cot": "llama-3.1-cot", "critic": "llama-3.1-critic", "judging-critic": "llama-3.1-judging-critic"},
        "deepseek-math": {"cot": "deepseek-math-cot", "critic": "deepseek-math-critic", "judging-critic": "deepseek-math-judging-critic"},
        "mistral": {"cot": "mistral-cot", "critic": "mistral-critic", "judging-critic": "mistral-judging-critic"}
    }

    def run_command(model_name, model_path, data_name, data_path, prompt_type="cot", is_self=False):
        output_dir = prompt_type + f"/{str(is_self)}" + model_path + "/math_eval"
        log_file = f"log/eval_{model_name}-{data_name}-{prompt_type}-is_self-{is_self}.txt"
        command = f"PYTHONPATH=$(pwd) nohup python math_eval.py --data_dir={data_path} --model_name_or_path={model_path} --output_dir={output_dir} --data_name={data_name} --prompt_type={prompt_type} --use_vllm --save_outputs --overwrite> {log_file} 2>&1"
        print(command)
        os.system(command)
        time.sleep(10)
        

    data_dir_root = f"/220049033/project/xzy/data/critic_benchmark/evaluation/data"
    for model in math_models:
        try:
            for (model_name, model_path) in model.items():
                for type in types:
                    if type == "critic" or type == "cot":
                        data_dir = data_dir_root + "/wrong_data"
                    else:
                        data_dir = data_dir_root + "/random"
                    if "qwen" in model_name.lower():
                        run_command(model_name, model_path, math_datas, data_dir, prompt_type=prompt_type["qwen"][type])
                    elif "llama-3.1" in model_name.lower():
                        run_command(model_name, model_path, math_datas, data_dir, prompt_type=prompt_type["llama-3.1"][type])
                    elif "deepseek-math" in model_name.lower():
                        run_command(model_name, model_path, math_datas, data_dir, prompt_type=prompt_type["deepseek-math"][type])
                    elif "mistral" in model_name.lower():
                        run_command(model_name, model_path, math_datas, data_dir, prompt_type=prompt_type["mistral"][type])
                    else:
                        raise
        except:
            break

    math_models = [
        {"Qwen2.5-7B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/"},
        {"Qwen2.5-Math-72B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-Math-72B-Instruct/snapshots/8fcf92b1eac8eca48c3d49778ad3dd1b394c9a17/"},
        {"Qwen2.5-72B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842/"},
        {"Llama-3.1-70B-Instruct": "/220049033/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393/"},
        {"Qwen2.5-Math-7B-Instruct": "/220049033/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/"},
        {"Llama-3.1-8B-Instruct": "/220049033/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"},
        {"mistral-123B": "/220049033/huggingface/hub/models--mistralai--Mistral-Large-Instruct-2407/snapshots/c6a230e7b1070914dd28bd6bc3b9c7c3d9e612ae/"}
    ]
    
    for model in math_models:
        try:
            data_dir_root = "/220049033/project/xzy/data/critic_benchmark/evaluation/outputs"
            for (model_name, model_path) in model.items():
                if "qwen" in model_name.lower():
                    data_dir = data_dir_root + "/" + prompt_type["qwen"]["cot"] + f"/False/{model_path}/math_eval/"
                    model_type = "qwen"
                elif "llama-3.1" in model_name.lower():
                    data_dir = data_dir_root + "/" + prompt_type["llama-3.1"]["cot"] + f"/False/{model_path}/math_eval/"
                    model_type = "llama-3.1"
                elif "deepseek-math" in model_name.lower():
                    data_dir = data_dir_root + "/" + prompt_type["deepseek-math"]["cot"] + f"/False/{model_path}/math_eval/"
                    model_type = "deepseek-math"
                elif "mistral" in model_name.lower():
                    data_dir = data_dir_root + "/" + prompt_type["mistral"]["cot"] + f"/False/{model_path}/math_eval/"
                    model_type = "mistral"
                else:
                    raise
                run_command(model_name, model_path, math_datas, data_dir, prompt_type=prompt_type[model_type]["judging-critic"], is_self=True)
        except:
            break
        
    

if __name__ == "__main__":
    main()