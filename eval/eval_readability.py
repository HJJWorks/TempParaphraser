import json
import argparse
import textstat
import numpy as np
import os
import re

from glob import glob


def count_words_advanced(sentence: str) -> int:
    return len(re.sub(r"[^\w\s]", "", sentence).split())


class TextStatEvaluator:

    def calc_metrics(self, samples):
        res_each_scores = {metric: [] for metric in [
            "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog", "smog_index", 
            "automated_readability_index", "coleman_liau_index", "linsear_write_formula", 
            "dale_chall_readability_score", "text_standard", "spache_readability", 
            "difficult_words_rate"]}

        for sample in samples:
            res_each_scores["flesch_reading_ease"].append(textstat.flesch_reading_ease(sample))
            res_each_scores["flesch_kincaid_grade"].append(textstat.flesch_kincaid_grade(sample))
            res_each_scores["gunning_fog"].append(textstat.gunning_fog(sample))
            res_each_scores["smog_index"].append(textstat.smog_index(sample))
            res_each_scores["automated_readability_index"].append(textstat.automated_readability_index(sample))
            res_each_scores["coleman_liau_index"].append(textstat.coleman_liau_index(sample))
            res_each_scores["linsear_write_formula"].append(textstat.linsear_write_formula(sample))
            res_each_scores["dale_chall_readability_score"].append(textstat.dale_chall_readability_score(sample))
            res_each_scores["text_standard"].append(textstat.text_standard(sample, float_output=True))
            res_each_scores["spache_readability"].append(textstat.spache_readability(sample))
            res_each_scores["difficult_words_rate"].append(textstat.difficult_words(sample) / count_words_advanced(sample))

        return res_each_scores

    def do_eval(self, samples):
        perturbed_scores = self.calc_metrics(samples)
        return {metric: np.mean(scores) for metric, scores in perturbed_scores.items()}


def load_attacked_texts(test_file):
    res_texts = []
    with open(test_file, "r") as rf:
        if test_file.endswith("jsonl"):
            for line in rf:
                lj = json.loads(line)
                if lj["label"] == "gpt":
                    attacked_text = lj["attacked_text"].strip()
                    if isinstance(attacked_text, str):
                        res_texts.append(attacked_text)
        elif test_file.endswith("json"):
            lj = json.load(rf)
            for sample in lj:
                if sample["label"] == "gpt":
                    attacked_text = sample["attacked_text"].strip()
                    if isinstance(attacked_text, str):
                        res_texts.append(attacked_text)

    return res_texts


def main(args):
    test_files = []
    for test_pattern in args.tests:
        if os.path.isdir(test_pattern):
            test_files.extend(
                [os.path.join(root, file) for root, _, files in os.walk(test_pattern) 
                 for file in files if file.endswith((".json", ".jsonl"))]
            )
        else:
            test_files.append(test_pattern)

    text_stat = TextStatEvaluator()
    for file in sorted(test_files):
        attacked_samples = load_attacked_texts(file)
        metric_report = text_stat.do_eval(attacked_samples)

        print(f"*************************\n[{file}]\n"
              f"Flesch Reading Ease: {metric_report['flesch_reading_ease']} (Delta: {metric_report['flesch_reading_ease'] - 57.5483})\n"
              f"Difficult Words Rate: {metric_report['difficult_words_rate']} (DW: {metric_report['difficult_words_rate'] - 0.17779})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--tests", nargs='+', default=["/home2/jjhuang/TempParaphraser/data/human.json"])
    args = parser.parse_args()
    main(args)
