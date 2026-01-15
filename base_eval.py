# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/eval/evaluation.py

import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm 
import numpy as np

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR_PATH = os.path.dirname(CUR_DIR_PATH)
sys.path.append(PARENT_DIR_PATH)

# Get absolute path to Fair-RAG folder
fairrag_path = os.path.abspath('Fair-RAG')
if fairrag_path not in sys.path:
    sys.path.insert(0, fairrag_path)


def calculate_rouge_over_fairness_groups(corpus, delta_df, bias, dataset_name, EVAL_RESULTS_DIR_PATH, LAMP_NUM):
    bias_values = set(corpus[bias].tolist())
    print(bias_values)
    bias_category_rouge = {key: 0 for key in bias_values}
    bias_category_count = {key: 0 for key in bias_values}

    grouped = delta_df.groupby('qid')
    for qid, group in grouped:
        baseline_score = 0
        for eachrow in group.itertuples(index=False):
            if np.isnan(eachrow.baseline_score):
                continue
            else:
                baseline_score = float(eachrow.baseline_score)
                break
        bias_value = corpus[corpus["docno"]==int(qid)][bias].values[0]
        bias_category_rouge[str(bias_value)] += baseline_score
        bias_category_count[str(bias_value)] += 1

    bias_category_avg = dict()
    # print(bias_category_rouge)
    for key in bias_values:
        count = bias_category_count[key]
        if count == 0:
            bias_category_avg[key] = 0  # or float('nan') if you prefer
        else:
            bias_category_avg[key] = bias_category_rouge[key] / count
    
    print(bias_category_avg)
    
    with open(os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_baseline_rouge.txt"), "a") as res_file:
        res_file.write(f"{bias}\n")
        res_file.write(str(bias_values))
        res_file.write("\n")
        res_file.write(str(bias_category_avg))
        res_file.write("\n")


def main(args):
    MODEL_NAME: str = args.model_name
    LAMP_NUM: int = args.lamp_num

    dataset_name = args.dataset
    if dataset_name == "":
        print("you must pass a valid value for --dataset arg.")
        return
    
    retriever_name = args.retriever
    if retriever_name == "":
        print("you must pass a valid value for --retriever arg.")
        return

    query_topic = args.query_topic
    if query_topic == "":
        print("you must pass a valid value for --query_topic arg.")
        return

    task = args.task
    if task == "":
        print("you must pass a valid value for --task arg.")
        return

    INF_RESULTS_DIR_PATH = os.path.join(
        CUR_DIR_PATH,
        "Fair-RAG",
        "utility_labels", 
        "inference_results",
        dataset_name,
        task,
        query_topic,
        retriever_name,
        MODEL_NAME,
    )
    EVAL_RESULTS_DIR_PATH = os.path.join(
        CUR_DIR_PATH,
        "Fair-RAG",
        "utility_labels", 
        "eval_results",
        dataset_name,
        task,
        query_topic,
        retriever_name,
        MODEL_NAME,
    )

    corpus = pd.read_csv(os.path.join(CUR_DIR_PATH, dataset_name, "document_features.csv"))
    
    os.makedirs(EVAL_RESULTS_DIR_PATH, exist_ok=True)
    delta_df = pd.read_csv(os.path.join(EVAL_RESULTS_DIR_PATH, "5_delta.tsv"), delimiter="\t")

    if os.path.exists(os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_baseline_rouge.txt")):
        os.remove(os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_baseline_rouge.txt"))
    
    calculate_rouge_over_fairness_groups(corpus, delta_df, "creation_date_category", dataset_name, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
    calculate_rouge_over_fairness_groups(corpus, delta_df, "years_category", dataset_name, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
    calculate_rouge_over_fairness_groups(corpus, delta_df, "relative_pageviews_category", dataset_name, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
    calculate_rouge_over_fairness_groups(corpus, delta_df, "first_letter_category", dataset_name, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
    
    
    grouped = delta_df.groupby('qid')
    total_scores = 0
    for qid, group in grouped:
        for eachrow in group.itertuples(index=False):
            if np.isnan(eachrow.baseline_score):
                continue
            else:
                total_scores += float(eachrow.baseline_score)
                break
        
    print(total_scores/len(grouped))

    with open(os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_baseline_rouge.txt"), "a") as res_file:
        res_file.write(f"AVG baseline score: {total_scores/len(grouped)}\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lamp_split_type",
        type=str,
        default="user",
        help="data split type of LaMP: either 'user' or 'time'",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Model nickname of HF model",
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
