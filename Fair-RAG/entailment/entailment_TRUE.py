from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class NLI_TRUE():
    def __init__(self, model_name, device="auto"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)

    def evaluate(self, passage: str, claim: str) -> int:
        # cut the input if it's too long, but try to keep the claim
        MAX_LENGTH = 512
        SAFE_BUFFER = 8
        passage_tokens = self.tokenizer(passage, return_tensors="pt").input_ids
        claim_tokens = self.tokenizer(claim, return_tensors="pt").input_ids
        passage_n_tokens = passage_tokens.size(1)
        claim_n_tokens = claim_tokens.size(1)
        if passage_n_tokens + claim_n_tokens + SAFE_BUFFER > MAX_LENGTH:
            if claim_n_tokens > MAX_LENGTH // 2:
                claim_tokens = claim_tokens[:, : MAX_LENGTH // 2]
                claim = self.tokenizer.decode(claim_tokens[0], skip_special_tokens=True)
                claim_n_tokens = claim_tokens.size(1)
            if passage_n_tokens + claim_n_tokens + SAFE_BUFFER > MAX_LENGTH:
                passage_tokens = passage_tokens[:, : MAX_LENGTH - claim_n_tokens - SAFE_BUFFER]
                passage = self.tokenizer.decode(passage_tokens[0], skip_special_tokens=True)
                passage_n_tokens = passage_tokens.size(1)

        prompt = f"premise: {passage} hypothesis: {claim}"
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids
        input_ids = input_ids.to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(input_ids, max_new_tokens=10)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if len(result) > 1:
            result = result[0]
        if result not in ["0", "1"]:
            print(f'warning: NLI AutoAIS returned "{result}" instead of 0 or 1')
            return 0
        return int(result)


"""
Perform LM inference either with
a) 0 profile: to get the baseline perforamance
b) 1 profile: to get the utility-gain of one item (profile) for a specific generator
"""
import os
import sys

# Path of the current file
CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# One level up (parent directory of CUR_DIR_PATH)
PARENT_DIR = os.path.dirname(CUR_DIR_PATH)
# Add the parent directory to sys.path
sys.path.append(PARENT_DIR)

import argparse
import pandas as pd
from transformers import pipeline, AutoTokenizer
import json
from tqdm import tqdm
from utils import models_info

def main(args):
    RAG_MODEL_NAME: str = args.rag_model_name
    ENTAILMENT_MODEL_NAME: str = args.entailment_model_name
    LAMP_NUM: int = args.lamp_num    
    nt = NLI_TRUE(model_name=models_info[ENTAILMENT_MODEL_NAME]["model_id"])

    dataset_name = args.dataset
    if dataset_name == "":
        print("you must pass a valid value for --dataset arg.")
        return
    
    retriever_name = args.retriever
    if retriever_name == "":
        print("you must pass a valid value for --retriever arg.")
        return
    
    rag_output_path = os.path.join(
        PARENT_DIR,
        "utility_labels",
        "inference_results",
        dataset_name,
        retriever_name,
        RAG_MODEL_NAME
    )
    dtype_spec = {"qid": str, "pid": str, "answer": str, "target": str}
    rag_result = pd.read_csv(os.path.join(rag_output_path, f"{LAMP_NUM}_output_rag.log"), delimiter="\t", dtype=dtype_spec)

    rag_input_path = os.path.join(
        PARENT_DIR,
        "data",
        "lamp",
        dataset_name,
        retriever_name
    )
    with open(os.path.join(rag_input_path, f"{LAMP_NUM}_user_dev_inputs.json"), 'r') as f:
        data = json.load(f)
    
    entailment_output_path = os.path.join(
        CUR_DIR_PATH,
        dataset_name,
        retriever_name,
        RAG_MODEL_NAME,
        ENTAILMENT_MODEL_NAME
    )
    os.makedirs(entailment_output_path, exist_ok=True)
    
    entailment_dict = {"doc_id":[], "profile_doc_id":[], "score":[]}
    counter = 0
    for item in tqdm(data):
        # if counter > 3:
        #     break
        premise = rag_result[rag_result['qid'] == item["id"]]["answer"].tolist()[0]
        # print(premise)
        # print("*************")    
        for context in item["profile"]:
            context_id = context["id"]
            if pd.isna(premise):
                entailment_dict["doc_id"].append(item["id"])
                entailment_dict["profile_doc_id"].append(context_id)
                entailment_dict["score"].append(0)
                continue
            
            context_title = context["title"]
            context_abstract = context["abstract"]
            hypotheses = context_title + "\n" + context_abstract
            # print(hypotheses)
            # print("++++++++++++++++")
            class_label = nt.evaluate(premise, hypotheses)
            # print(f"class label: {class_label}")
            entailment_dict["doc_id"].append(item["id"])
            entailment_dict["profile_doc_id"].append(context_id)
            entailment_dict["score"].append(class_label)
        counter += 1
    pd.DataFrame(entailment_dict).to_csv(os.path.join(entailment_output_path, "entailment.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rag_model_name",
        type=str,
        default="flanT5XXL",
        help="LLM used in the RAG system",
    )

    parser.add_argument(
        "--entailment_model_name",
        type=str,
        default="",
        help="Entailment model used for calculate context usage",
    )
    
    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Help with nameing output dir creation",
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default="",
        help="Help with reading input dataset from the correct directory",
    )
    
    args = parser.parse_args()

    main(args)
