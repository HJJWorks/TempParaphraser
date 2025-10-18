# TempParaphraser: "Heating Up" Text to Evade AI-Text Detection through Paraphrasing 🔥📝
EMNLP 2025

## How to Use

The `attack` directory contains two key types of code:

1. **Experimental test code** for validating the model’s performance
2. **Custom attack code** that supports single-text input (accepts a single text segment) and returns the paraphrased output. This code can be adapted to meet your specific experimental requirements.


### Step-by-Step Guide to Reproduce Our Experiments

Follow these steps exactly to replicate the experiments presented in our paper:

#### 1. Download the Model and Launch the VLLM Backend via Llamafactory
Visit https://huggingface.co/huangjj877/TempParaphraser to obtain our fine-tuned paraphrasing model (used in the main experiments).

We use Llamafactory to start the VLLM backend. Refer to https://github.com/hiyouga/LLaMA-Factory and follow its installation instructions, ensuring the vllm functionality is installed.

Run the following command in your terminal:
```bash
API_PORT=10001 llamafactory-cli api attack/start_paraphrasing_model_vllm.yaml
```
This launches the inference API for the paraphrasing model.


#### 2. Run the Main Experiment Script

After launching the backend, execute the script to process the test set and generate paraphrased text:

1. First, open the `attack/attack_for_experiment.sh` file and confirm (or set) the `API_PORT` parameter to `10001` (to match the backend port in Step 1).

2. Run the script with the following command:
```bash
bash attack/attack_for_experiment.sh
```

**What the script does:**
- Automatically iterates through every GPT-generated text sample in the test set.
- Applies TempParaphraser’s paraphrasing to each sample (while preserving semantic meaning).
- Saves the fully paraphrased dataset to your specified directory (check the script for output path details).


### How to Customize Your Experiments
The `rewrite_text` function in `attack/attack_for_experiment.py` serves as the entry point for paraphrasing single text segments. You can freely adapt it to your experimental needs by modifying the dataset iteration method in the `main` function.


### Acknowledgements
Some code and data are derived from the following open-source repositories:
- https://github.com/zhouying20/HMGC
- https://github.com/superannotateai/generated_text_detector
- https://github.com/textstat/textstat
- https://github.com/baoguangsheng/fast-detect-gpt
- https://github.com/hiyouga/LLaMA-Factory

We thank the authors of these repositories for their valuable research resources.


### Code Usage Restrictions
This code is restricted to academic research use only. For any questions, please contact junjie2001@stu.xmu.edu.cn.