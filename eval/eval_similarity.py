import os
import json
import torch
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy import spatial
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def cosine_similarity(X, Y):
    return 1 - spatial.distance.cosine(X, Y)


class SimilarityComparer(object):
    def __init__(
        self,
        model_name_or_path="princeton-nlp/sup-simcse-roberta-large",
        similarity_method="cosine",
        pooling_method="cls"
    ) -> None:
        self.model_name = model_name_or_path
        if similarity_method == "cosine":
            self.similarity_func = cosine_similarity
        else:
            raise NotImplementedError(f"not implemented similarity function -> {similarity_method}")
        self.pooling_method = pooling_method
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)

    def get_similarity(self, origin_text, perturbed_text):
        vec_pair = []
        for text in [origin_text, perturbed_text]:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512,truncation=True).to(self.device)
            with torch.no_grad():
                output = self.model(**inputs, return_dict=True, output_hidden_states=True)

            last_hidden = output.last_hidden_state
            pooler_output = output.pooler_output
            hidden_states = output.hidden_states
            if self.pooling_method == 'cls':
                out_vec = pooler_output.cpu()
            elif self.pooling_method == 'last_avg':
                out_vec = last_hidden.mean(dim=1)
            elif self.pooling_method == 'last2_avg':
                out_vec = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            elif self.pooling_method == 'first_last_avg':
                out_vec = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            else:
                raise NotImplementedError("unknown pooling {}".format(self.pooling_method))

            vec_pair.append(out_vec.cpu().numpy()[0])

        res_score = self.similarity_func(vec_pair[0], vec_pair[1])
        return res_score


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
                    attacked_text = sample["attacked_text"] or sample["origin_text"]
                    text_pairs.append((sample["origin_text"], attacked_text))
    return text_pairs



def main(args):
    sim_eval = SimilarityComparer(model_name_or_path=args.similarity_model)

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

        sims = list()
        #ppls = list()
        for ori_text, per_text in tqdm(text_pairs, dynamic_ncols=True):
            sim_score = sim_eval.get_similarity(ori_text, per_text)
            sims.append(sim_score)

        print(
            "*******************************\n"
            f"perturbed: {file}\n"
            f"similarity:{np.mean(sims)}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--similarity_model", default="princeton-nlp/sup-simcse-roberta-large", help="Huggingface Model for evaluate the similarity score")
    parser.add_argument("--tests", nargs='+', default=[""])
    args = parser.parse_args()
    main(args)