# RealCritic: Towards Effectiveness-Driven Evaluation of Language Model Critiques

<p align="center" width="100%">
<a ><img src="./imgs/RealCritic.png" alt="RealCritic" style="width: 80%; min-width: 300px; display: block; margin: auto;"></a>
</p> 

RealCritic is a novel benchmark for evaluating language models' critique capabilities through a closed-loop, effectiveness-driven approach. Unlike traditional open-loop evaluations, we measure critique quality based on its ability to drive solution improvement.

## Key Contributions

1. **Closed-loop Evaluation**: Measures critique quality through improvement in refined solutions rather than isolated judgments
2. **Comprehensive Assessment**: First benchmark to evaluate self-critique, cross-critique, and iterative critique capabilities
3. **Novel Findings**: Reveals significant gaps between reasoning-based and traditional LLMs in critique abilities

## Benchmark Details

### Tasks
- ARC Challenge: Multiple choice science questions
- GSM8K: Grade school math problems
- MATH: Advanced mathematics
- College Math: Undergraduate mathematics
- Minerva Math: Scientific reasoning
- GPQA: Physics questions
- MMLU STEM: Science/tech/engineering/math topics
- Olympiad Bench: Competition-level problems

### Evaluation Modes
- **Direct CoT**: Baseline problem-solving ability
- **Self-Critique**: Model critiques its own solutions
- **Cross-Critique**: Model critiques other models' solutions
- **Iterative Critique**: Multi-round refinement process

## Results

| Model | Task Type | ARC | GSM8K | MATH | College Math | Minerva Math | GPQA | MMLU STEM | Olympiad | Avg |
|-------|-----------|------|--------|-------|--------------|--------------|-------|------------|-----------|-----|
| LLaMA-3.1-70B | Direct | 88.6 | 93.4 | 65.3 | 40.1 | 33.7 | 32.6 | 63.0 | 31.1 | 56.0 |
| | Self-Critique | 87.5 | 91.9 | 59.0 | 37.3 | 33.7 | 22.1 | 53.8 | 28.5 | 51.7 |
| | Δ(Self-Direct) | -1.1 | -1.5 | -6.3 | -2.8 | 0.0 | -10.5 | -9.2 | -2.6 | -4.3 |
| | Cross-Critique | 82.5 | 88.6 | 64.9 | 49.1 | 42.1 | 35.3 | 42.8 | 40.8 | 55.8 |
| | Δ(Cross-Base) | +32.5 | +38.6 | +14.9 | -0.9 | -7.9 | -14.7 | -7.2 | -9.2 | +5.8 |
| Mistral-Large | Direct | 86.7 | 94.5 | 70.1 | 41.6 | 35.6 | 41.1 | 66.8 | 34.5 | 58.9 |
| | Self-Critique | 85.2 | 93.4 | 70.9 | 41.6 | 36.8 | 36.3 | 56.5 | 36.0 | 57.1 |
| | Δ(Self-Direct) | -1.5 | -1.1 | +0.8 | 0.0 | +1.2 | -4.8 | -10.3 | +1.5 | -1.8 |
| | Cross-Critique | 84.8 | 90.8 | 69.0 | 50.9 | 49.4 | 40.5 | 53.8 | 48.7 | 61.0 |
| | Δ(Cross-Base) | +34.8 | +40.8 | +19.0 | +0.9 | -0.6 | -9.5 | +3.8 | -1.3 | +11.0 |
[Similar rows for other models...]
| o1-mini | Direct | 68.1 | 89.0 | 81.3 | 20.1 | 28.4 | 56.3 | 81.8 | 49.4 | 59.3 |
| | Self-Critique | 87.5 | 93.8 | 91.4 | 44.1 | 41.0 | 33.2 | 53.8 | 55.8 | 62.6 |
| | Δ(Self-Direct) | +19.4 | +4.8 | +10.1 | +24.0 | +12.6 | -23.1 | -28.0 | +6.4 | +3.3 |
| | Cross-Critique | 85.2 | 93.8 | 85.1 | 52.7 | 49.8 | 44.7 | 56.5 | 57.3 | 65.6 |
| | Δ(Cross-Base) | +35.2 | +43.8 | +35.1 | +2.7 | -0.2 | -5.3 | +6.5 | +7.3 | +15.6 |

Key findings from delta analysis:
1. Self-Critique Impact:
   - Traditional LLMs show negative deltas (-1.8% to -5.1%)
   - Only o1-mini achieves positive improvement (+3.3%)
   - Largest improvements in College Math (+24.0%) and ARC (+19.4%) for o1-mini

2. Cross-Critique Performance:
   - All models show positive average deltas
   - o1-mini leads with +15.6% average improvement
   - Strong performance in mathematical tasks (GSM8K: +43.8%, MATH: +35.1%)

## Usage

### Installation
Please follow the installation instructions in the [Qwen2.5-Math repository](https://github.com/QwenLM/Qwen2.5-Math.git).

### Local Evaluation

```bash
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

### API Evaluation

```bash
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

### Post Check (Optional)

```
python3 -u model_eval_critic.py \
	--data_dir path/for/critic/math_eval \
	--model_name_or_path /post/check/model \
	--output_dir /output/dir/ \
	--data_name ${DATA_NAME} \
	--seed 0 \
	--temperature 0 \
	--top_p 1 \
```

### Customizing Prompts

All prompts are defined in `utils.py`:

```python
# System prompts
SYSTEM_PROMPT_TEMPLATES = {
    "direct-cot": "...",
    "critic_and_correct": "...",  # Must include "critic" keyword
}

# User prompts
USER_PROMPT_TEMPLATE = {
    "default-cot": {
        "type": "cot",
        "template": "..."
    },
    "critic_and_correct": {
        "type": "critic",
        "template": "..."
    }
}
```

Acknowledgements
The code in this repository is built upon [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math.git). We thank the Qwen team for their excellent work and open-source contributions.