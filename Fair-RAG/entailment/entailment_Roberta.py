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
    tokenizer = AutoTokenizer.from_pretrained(models_info[ENTAILMENT_MODEL_NAME]["model_id"])
    classifier = pipeline(models_info[ENTAILMENT_MODEL_NAME]["pipeline_task"], model=models_info[ENTAILMENT_MODEL_NAME]["model_id"])
    dataset_name = args.dataset
    if dataset_name == "":
        print("you must pass a valid value for --dataset arg.")
        return
    
    retriever_name = args.retriever
    if retriever_name == "":
        print("you must pass a valid value for --retriever arg.")
        return

    query_topic = args.query_topic
    if retriever_name == "":
        print("you must pass a valid value for --query_topic arg.")
        return

    task = args.task
    if retriever_name == "":
        print("you must pass a valid value for --task arg.")
        return

    
    rag_output_path = os.path.join(
        PARENT_DIR,
        "utility_labels",
        "inference_results",
        dataset_name,
        task,
        query_topic,
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
        task,
        query_topic,
        retriever_name
    )
    with open(os.path.join(rag_input_path, f"{LAMP_NUM}_user_dev_inputs.json"), 'r') as f:
        data = json.load(f)
    
    entailment_output_path = os.path.join(
        CUR_DIR_PATH,
        dataset_name,
        task,
        query_topic,
        retriever_name,
        RAG_MODEL_NAME,
        ENTAILMENT_MODEL_NAME
    )
    os.makedirs(entailment_output_path, exist_ok=True)

    entailment_dict = {"doc_id":[], "profile_doc_id":[], "score":[]}
    counter = 0
    for item in tqdm(data):
        premise = rag_result[rag_result['qid'] == item["id"]]["answer"].tolist()[0]
        # print(premise)
        # print("*************")    
        for context in item["profile"]:
            context_id = context["id"]
            if pd.isna(premise) or premise.strip() == "":
                entailment_dict["doc_id"].append(item["id"])
                entailment_dict["profile_doc_id"].append(context_id)
                entailment_dict["score"].append(0)
                continue
            
            context_title = context["title"]
            context_abstract = context["abstract"]
            hypotheses = context_title + "\n" + context_abstract

            hypothese_len = len(tokenizer.tokenize(hypotheses))
            premise_len = len(tokenizer.tokenize(premise))
            if hypothese_len + premise_len > 512:
                if premise_len > 254 and hypothese_len > 254:
                    premise = tokenizer.tokenize(premise)[:252]
                    premise = tokenizer.convert_tokens_to_ids(premise)
                    premise = tokenizer.decode(premise, skip_special_tokens=True)
                    hypotheses = tokenizer.tokenize(hypotheses)[:252]
                    hypotheses = tokenizer.convert_tokens_to_ids(hypotheses)
                    hypotheses = tokenizer.decode(hypotheses, skip_special_tokens=True)
                elif premise_len < 254:
                    hypotheses = tokenizer.tokenize(hypotheses)[:502 - len(tokenizer.tokenize(premise))]
                    hypotheses = tokenizer.convert_tokens_to_ids(hypotheses)
                    hypotheses = tokenizer.decode(hypotheses, skip_special_tokens=True)
                elif hypothese_len < 254:
                    premise = tokenizer.tokenize(premise)[:502 - len(tokenizer.tokenize(hypotheses))]
                    premise = tokenizer.convert_tokens_to_ids(premise)
                    premise = tokenizer.decode(premise, skip_special_tokens=True)
            # print(hypotheses)
            # inputs = tokenizer(premise, hypotheses, truncation=True, return_tensors="pt")
            # if inputs['input_ids'].shape[1] > 512:
            #     # You may log or skip if this still occurs, but it shouldn't with truncation
            #     hypotheses = tokenizer.tokenize(hypotheses)[:510-len(tokenizer.tokenize(premise))]
            #     hypotheses = tokenizer.convert_tokens_to_ids(hypotheses)
            #     hypotheses = tokenizer.decode(hypotheses, skip_special_tokens=True)
                
            # if len(tokenizer.tokenize(hypotheses)) + len(tokenizer.tokenize(premise)) > 512:
            #     # print("*************")
            #     # print(hypotheses)
            #     # print("*************")
            #     hypotheses = tokenizer.tokenize(hypotheses)[:505-len(tokenizer.tokenize(premise))]
            #     hypotheses = tokenizer.convert_tokens_to_ids(hypotheses)
            #     hypotheses = tokenizer.decode(hypotheses, skip_special_tokens=True)
            #     # print(hypotheses)
            #     # print("*************")
            result = classifier(premise, hypotheses, truncation=True, max_length=512, multi_label=False)
            premise = rag_result[rag_result['qid'] == item["id"]]["answer"].tolist()[0]
            entailment_dict["doc_id"].append(item["id"])
            entailment_dict["profile_doc_id"].append(context_id)
            entailment_dict["score"].append(result["scores"][0])
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

    parser.add_argument(
        "--query_topic",
        type=str,
        default="",
        help="Help with reading input dataset from the correct directory",
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Help with reading input dataset from the correct directory",
    )
    args = parser.parse_args()

    main(args)
