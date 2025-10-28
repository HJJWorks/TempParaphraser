# TempParaphraser: "Heating Up" Text to Evade AI-Text Detection through Paraphrasing 🔥📝

**EMNLP 2025**

## Overview

TempParaphraser is a fine-tuned paraphrasing model designed to “heat up” the text representation and evade AI-text detectors while preserving the semantic content.
This repository provides all scripts and configuration files to **reproduce the results** from our EMNLP 2025 paper.


## ⚙️ Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HJJWorks/TempParaphraser.git
cd TempParaphraser
```

### 2. Create and Activate the Conda Environment

```bash
conda create -n tp python=3.10
conda activate tp
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


## 📦 Model Download and Placement

Before running any experiments, you must download our fine-tuned paraphrasing model:

1. Visit [**huangjj877/TempParaphraser**](https://huggingface.co/huangjj877/TempParaphraser)
2. **Place the entire folder under:**

   ```
   TempParaphraser/model/
   ```


## 🧠 Install and Configure LLaMA-Factory (Required)

**TempParaphraser relies on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to launch the VLLM backend**,
which enables *high-throughput paraphrasing* and *multi-round text rewriting*.
To avoid dependency conflicts, we **recommend installing LLaMA-Factory in a separate Conda environment.**

### Step 1. Create and Activate a New Environment for LLaMA-Factory

```bash
conda create -n llamafactory python=3.10
conda activate llamafactory
```

### Step 2. Install LLaMA-Factory with VLLM Support

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[vllm]"
```

After installation, verify it by running:

```bash
llamafactory-cli --help
```

If the help message appears correctly, you’re ready to launch the paraphrasing backend.


## 🚀 Run the Paraphrasing Backend via LLaMA-Factory

Switch to your LLaMA-Factory environment (if not already active) and start the model backend:

```bash
conda activate llama
cd TempParaphraser
API_PORT=10001 llamafactory-cli api attack/start_paraphrasing_model_vllm.yaml
```

✅ If successful, you’ll see:

```
Uvicorn running on http://0.0.0.0:10001 (Press CTRL+C to quit)
```

This backend provides the HTTP API endpoint used by all TempParaphraser scripts for inference.

## Reproducing the Experiments

Follow these steps to replicate our paper’s experiments.

### Run the Main Experiment Script

After the backend is running:

1. Open `attack/attack_for_experiment.sh`

   * Make sure `API_PORT=10001` matches your backend port.
   * Adjust dataset input/output paths if needed.

2. Execute the batch script:

   ```bash
   bash attack/attack_for_experiment.sh
   ```

This script will:

* Iterate through all GPT-generated test samples.
* Paraphrase each sample using TempParaphraser (preserving semantics).
* Save the fully paraphrased dataset to your defined output directory.

## 🧠 Customizing Your Experiment

You can modify:

* **`attack/attack_for_experiment.py` → `rewrite_text()`**
  This is the entry point for paraphrasing single text segments.
* **`main()`**
  Adapt the dataset iteration logic to fit your own corpus or evaluation pipeline.


## 🤝 Acknowledgements

Some code and data are derived from the following open-source repositories:

* [HMGC](https://github.com/zhouying20/HMGC)
* [Generated-Text-Detector](https://github.com/superannotateai/generated_text_detector)
* [textstat](https://github.com/textstat/textstat)
* [Fast-Detect-GPT](https://github.com/baoguangsheng/fast-detect-gpt)
* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

We thank the authors of these repositories for their valuable research resources.


## 📜 License & Usage Restriction

This code is released **for academic research purposes only**.
Commercial use is strictly prohibited.

For any questions or collaborations, please contact:
📧 **[junjie2001@stu.xmu.edu.cn](mailto:junjie2001@stu.xmu.edu.cn)**
