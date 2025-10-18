import os
import json
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from glob import glob
from pathlib import Path
from datasets import load_dataset
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from transformers import AutoTokenizer

from metrics import calc_classification_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def prepare_data(config, data_file, tokenizer, label2id, padding="max_length", max_seq_length=512, batch_size=64, shuffle=False, pin_memory=False, num_workers=0):
    ori_key = config.origin_key
    att_key = config.attacked_key
    prefix_key = config.prefix_key
    label_key = config.label_key

    def tokenize_func(examples):
        token_args = list()
        for idx, lb in enumerate(examples[label_key]):
            if lb == "gpt":  # only gpt samples were attacked
                _text = examples[att_key][idx] or examples[ori_key][idx]
                token_args.append(_text if prefix_key is None else (examples[prefix_key][idx], _text))
            else:
                token_args.append(examples[ori_key][idx] if prefix_key is None else (examples[prefix_key][idx], examples[ori_key][idx]))
        result = tokenizer(token_args, padding=padding, max_length=max_seq_length, truncation=True)

        if label2id is not None and label_key in examples:
            result["label"] = [(label2id[l] if l != -1 else -1) for l in examples[label_key]]
        return result

    def label_split_collate_fn(batch):
        input_ids = torch.stack([d['input_ids'] for d in batch], dim=0).to(device)
        attn_masks = torch.stack([d['attention_mask'] for d in batch], dim=0).to(device)
        labels = [d['label'] for d in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attn_masks
        }, labels

    _data = {"data": data_file}
    dataset = load_dataset("json", data_files=_data)["data"]
    dataset = dataset.map(tokenize_func, batched=True, desc="Running tokenizer on dataset")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, collate_fn=label_split_collate_fn)


class SAEvaluator(object):
    def __init__(self, config) -> None:
        self.config = config
        self.model_path = 'SuperAnnotate/roberta-large-llm-content-detector'

        self.batch_size = config.batch_size

        self.label2id = {
            "human": 0,
            "gpt": 1,
        }
        self.labels = ["human", "gpt"]

        model = RobertaClassifier.from_pretrained(self.model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = model
        self.tokenizer = tokenizer

    def do_eval(self, test_file):
        model = self.model
        dataloader = prepare_data(self.config, test_file, self.tokenizer, self.label2id, padding="max_length", max_seq_length=512, batch_size=self.batch_size)

        all_labels = list()
        all_preds = list()
        total_logits = 0
        for batch_inputs, batch_labels in tqdm(dataloader):
            all_labels += batch_labels

            with torch.no_grad():
                _, logits_text = model(**batch_inputs)
                logits = F.sigmoid(logits_text).cpu()

            preds = (logits > 0.5).int().tolist()
            all_preds += preds

        metric_report = calc_classification_metrics(all_preds, all_labels, target_names=self.labels)
        print("*************************")
        print(f"[{self.model_path}]-[{test_file}]-len({len(all_labels)})")
        #print(json.dumps(metric_report, indent=4, ensure_ascii=False))
        print(metric_report['gpt_acc'])

        return metric_report

class HC3Evaluator(object):
    def __init__(self, config) -> None:
        self.config = config
        self.model_path = "Hello-SimpleAI/chatgpt-detector-roberta"

        self.batch_size = config.batch_size

        self.label2id = {
            "human": 0,
            "gpt": 1,
        }
        self.labels = ["human", "gpt"]

        model_config = RobertaConfig.from_pretrained(self.model_path, num_labels=2)
        tokenizer = RobertaTokenizer.from_pretrained(self.model_path, use_fast=False)
        model = RobertaForSequenceClassification.from_pretrained(self.model_path, config=model_config).to(device)

        self.model = model
        self.tokenizer = tokenizer


    def do_eval(self, test_file):
        model = self.model
        dataloader = prepare_data(self.config, test_file, self.tokenizer, self.label2id, padding="max_length", max_seq_length=512, batch_size=self.batch_size)

        all_labels = list()
        all_preds = list()
        for batch_inputs, batch_labels in tqdm(dataloader):
            all_labels += batch_labels

            with torch.no_grad():
                logits = model(**batch_inputs).logits
                logits = logits.cpu().numpy()

            preds = np.argmax(logits, axis=1).tolist()
            all_preds += preds

        metric_report = calc_classification_metrics(all_preds, all_labels, target_names=self.labels)
        print("*************************")
        print(f"[{self.model_path}]-[{test_file}]-len({len(all_labels)})")
        print(metric_report['gpt_acc'])

        return metric_report


def main(args):
    if args.detector == "sa":
        evaluator = SAEvaluator(args)
    elif args.detector == "hc3":
        evaluator = HC3Evaluator(args)
    else:
        raise ValueError(f"Unsupported detector type: {args.detector}")

    test_files = list()
    for test_pattern in args.tests:
        if os.path.isdir(test_pattern):
            test_files.extend(
                [os.path.join(root, file) for root, _, files in os.walk(test_pattern) 
                 for file in files if file.endswith((".json", ".jsonl"))]
            )
        else:
            test_files.append(test_pattern)

    res_evals = list()
    for test in sorted(test_files):
        eval_meta = {
            "test": Path(test).stem,
            "model": args.detector,
        }
        eval_metric = evaluator.do_eval(test)
        res_evals.append(dict(list(eval_meta.items()) + list(eval_metric.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1,)
    parser.add_argument("--seed", type=int, default=42,)

    parser.add_argument("--detector", type=str, default="sa")
    parser.add_argument("--tests", nargs='+', default=[""])
    parser.add_argument("--output_file", type=str, default=None,)

    parser.add_argument("--batch_size", type=int, default=256,)
    parser.add_argument("--origin_key", type=str, default="origin_text",)
    parser.add_argument("--attacked_key", type=str, default="attacked_text",)
    parser.add_argument("--prefix_key", type=str, default=None,)
    parser.add_argument("--label_key", type=str, default="label",)
    args = parser.parse_args()

    set_seed(args)
    main(args)
