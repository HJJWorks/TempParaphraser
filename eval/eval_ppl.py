import os
import json
import torch
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy import spatial
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def load_text_pairs(test_file):
    text_pairs = list()
    if "jsonl" in test_file:
        with open(test_file, "r") as rf:
            for line in rf:
                lj = json.loads(line)
                if lj["label"] == "gpt":
                    # in case of None generated from attacking
                    attacked_text = lj["attacked_text"]
                    text_pairs.append((lj["origin_text"], attacked_text))
    elif "json" in test_file:
        with open(test_file, "r") as rf:
            lj = json.load(rf)
            for sample in lj:
                if sample["label"] == "gpt":
                    attacked_text = sample["attacked_text"]
                    text_pairs.append((sample["origin_text"], attacked_text))
    return text_pairs

def calculate_ppl(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()


def main(args):
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_files = list()
    for test_pattern in args.tests:
        if os.path.isdir(test_pattern):
            for root, dirs, files in os.walk(test_pattern):
                for file in files:
                    if file.endswith(".json") or file.endswith(".jsonl"):
                        test_files.append(os.path.join(root, file))
        else:
            test_files.append(test_pattern)

    for file in sorted(test_files):
        text_pairs = load_text_pairs(file)
        ppls = list()
        for ori_text, per_text in tqdm(text_pairs, dynamic_ncols=True):
            if not isinstance(per_text, str):
                continue
            if len(per_text) < 10:
                continue
            ppl = calculate_ppl(per_text, model, tokenizer)
            ppls.append(ppl)

        avg_ppl = np.mean(ppls)
        print(
            "*******************************\n"
            f"perturbed: {file}\n"
            f"gpt2 ppl:{avg_ppl}[delta：{avg_ppl-35.83644}]\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--tests", nargs='+', default=["human.json"])
    args = parser.parse_args()
    main(args)