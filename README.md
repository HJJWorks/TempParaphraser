TempParaphraser: "Heating Up" Text to Evade AI-Text Detection through Paraphrasing 🔥📝
EMNLP 2025

## How to Use

The `attack` directory houses two key types of code:



1. **Experimental test code** for validating the model’s performance

2. **Custom attack code** that enables single-text input (takes a single text segment) and returns the paraphrased output. This code can be adapted to fit your specific experimental requirements.

### Step-by-Step Guide to Reproduce Our Experiments

Follow these exact steps to replicate the experiments presented in our paper:

#### 1. Launch the VLLM Backend via Llamafactory

We use Llamafactory to start the VLLM backend (a high-performance inference engine for large language models). Run the following command in your terminal:



```
API\_PORT=8000 llamafactory-cli api attack/start\_paraphrasing\_model\_vllm.yaml infer\_backend=vllm vllm\_enforce\_eager=true
```


#### 2. Run the Experiment Script

After launching the backend, execute the script to process the test set and generate paraphrased text:



1. First, open the `attack/attack_for_experiment.sh` file and confirm (or set) the `API_PORT` parameter to `8000` (to match the backend port in Step 1).

2. Run the script using the following command:



```
bash attack/attack\_for\_experiment.sh
```



1. What the script does:

* Automatically iterates through every GPT-generated text sample in the test set.

* Applies TempParaphraser’s paraphrasing to each sample (while preserving semantic meaning).

* Saves the fully paraphrased dataset to your specified directory (check the script for output path details).


