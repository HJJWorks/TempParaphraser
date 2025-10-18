import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
import json
import argparse
import openai

parser = argparse.ArgumentParser()
parser.add_argument('--eval_model_name', type=str, default="hc3")
parser.add_argument('--rewrite_times', type=int, default=5)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--input_path', type=str, default='/data/HC3/test_rnd10k.jsonl')
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--api', type=str, required=True)
args = parser.parse_args()

eval_model_name = args.eval_model_name
rewrite_times = args.rewrite_times
temperature = args.temperature
input_path = args.input_path
save_path = args.save_path

# Create save directory if it doesn't exist
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))

# Define system prompt for rewriting text
system_prompt = """Rewrite the following text to sound more natural and human-like. Keep the same information and overall structure, but use more casual language, varied sentence structures, and add subtle personal touches."""

device = torch.device("cuda")

if rewrite_times != 1:
    if eval_model_name == "SA":
        eval_model_path = "SuperAnnotate/roberta-large-llm-content-detector"
        eval_model = RobertaClassifier.from_pretrained(eval_model_path).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_path)
    elif eval_model_name == "hc3":
        eval_model_path = "Hello-SimpleAI/chatgpt-detector-roberta"
        eval_model_config = RobertaConfig.from_pretrained(eval_model_path, num_labels=2)
        eval_tokenizer = RobertaTokenizer.from_pretrained(eval_model_path, use_fast=False)
        eval_model = RobertaForSequenceClassification.from_pretrained(eval_model_path, config=eval_model_config).to(device)
    else:
        raise NotImplementedError("Unknown eval model")

openai.api_key = ''
openai.api_base = args.api

# Batch inference function
def batch_inference(input_text, temperature=1.2, rewrite_times=5):
    """
    Perform batch inference on input text to generate paraphrased versions.
    :param input_text: The input text to be rewritten
    :param temperature: Temperature for model inference (default is 1.2)
    :param rewrite_times: Number of times to generate rewrites (default is 5)
    :return: A list of rewritten text
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}]
    
    try:
        response = openai.ChatCompletion.create(
            model="",
            messages=messages,
            temperature=temperature,
            n=rewrite_times
        )
        # Extract and return the rewritten responses
        responses = [choice['message']['content'] for choice in response['choices']]
        return responses

    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# Rewrite a single sentence using batch inference and evaluation model
def rewrite_sentence(input_text, before_text, rewrite_times=5, temperature=0.7):
    batch_generations = batch_inference(input_text, temperature, rewrite_times)
    batch_generations.insert(0, input_text)  # Insert the original text at the beginning
    before_and_batch_generations = [before_text + gen for gen in batch_generations]

    if eval_model_name == "SA":
        tokens = eval_tokenizer.batch_encode_plus(
            before_and_batch_generations,
            add_special_tokens=True,
            max_length=512,
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        ).to(device)

        _, logits = eval_model(**tokens)
        logits = torch.sigmoid(logits)
    elif eval_model_name == "hc3":
        tokens = eval_tokenizer(before_and_batch_generations, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
        with torch.no_grad():
            logits = eval_model(**tokens).logits
            logits = torch.sigmoid(logits)
        logits = logits[:, 1]  # Take the second column of logits

    # Find the minimum probability and return the corresponding rewritten text
    min_index = torch.argmin(logits)
    return batch_generations[min_index].strip()

# Rewrite input text with optional evaluation flag
def rewrite_text(input_text, eval_flag=False, rewrite_times=5, temp=1.2):
    output_text = ""
    for segment in input_text.split("."):
        if len(segment.strip()) == 0:
            continue
        segment = segment.strip() + '.'
        output = ""
        
        for line in segment.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line.split(" ")) > 4:
                rewrite_text_result = batch_inference(line, temp, 1)[0] if rewrite_times == 1 else rewrite_sentence(line, output_text, rewrite_times, temp)
            else:
                rewrite_text_result = line
            
            output += "\n" + rewrite_text_result if output else rewrite_text_result
            output_text += "\n" + rewrite_text_result if output else rewrite_text_result
            
        output_text += " "

    # Evaluate rewritten text if flag is set
    if eval_flag:
        input_tokens = eval_tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=512,
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        ).to(device)

        _, input_logits = eval_model(**input_tokens)
        input_proba = F.sigmoid(input_logits).squeeze(1).item()
        output_text += f"（Original Probability: {input_proba:.4f} "

        tokens = eval_tokenizer.encode_plus(
            output_text,
            add_special_tokens=True,
            max_length=512,
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        ).to(device)

        _, logits = eval_model(**tokens)
        proba = F.sigmoid(logits).squeeze(1).item()
        output_text += f"｜ Rewritten Probability: {proba:.4f}）"

    return output_text.strip()

# Load data from file
def load_data(file_path):
    if "jsonl" in file_path:
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    else:
        with open(file_path, 'r') as f:
            return json.load(f)

# Load existing results from file if available
def load_existing_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Save results to file
def save_results(file_path, results):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

# Main function to process the input data and apply text rewriting
if __name__ == "__main__":
    data = load_data(input_path)
    results = load_existing_results(save_path)
    existing_uids = {item['uid'] for item in results}

    # Iterate through the data and process each item
    for item in tqdm(data, desc=f"Attack ｜T{temperature}｜R{rewrite_times}"):
        # Check if the result already exists
        if item['uid'] in existing_uids:
            continue
        result_json = {
            'uid': item['uid'],
            'id': item['id'],
            'label': item['label'],
            'source': item['source'],
            'prefix': item['question'],
            'origin_text': item['answer']
        }

        # If label is 'gpt', rewrite the text
        if item['label'] == 'gpt':
            result_text = rewrite_text(item['answer'], eval_flag=False, rewrite_times=rewrite_times, temp=temperature)
            result_json['attacked_text'] = result_text

        results.append(result_json)
        
        if len(results) % 100 == 0:
            save_results(save_path, results)
    
    # Save the final results
    save_results(save_path, results)
