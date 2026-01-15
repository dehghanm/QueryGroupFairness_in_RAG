# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/eval/evaluation.py

import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm 
import ast

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR_PATH = os.path.dirname(CUR_DIR_PATH)
sys.path.append(PARENT_DIR_PATH)

# Get absolute path to Fair-RAG folder
fairrag_path = os.path.abspath('Fair-RAG')
if fairrag_path not in sys.path:
    sys.path.insert(0, fairrag_path)

from eval.lamp_metrics import get_metric_fn_accuracy, get_metric_fn_rouge_L



def calculate_rouge_over_fairness_groups(corpus, rag_df, bias, dataset_name, augment_scores, EVAL_RESULTS_DIR_PATH, LAMP_NUM):
    if bias == "general_regions":
        bias_values = set()
        bias_values_lists = corpus[bias].tolist()
        for bias_values_list in bias_values_lists:
            for bias_value in ast.literal_eval(bias_values_list):
                bias_values.add(bias_value)
    else:
        bias_values = set(corpus[bias].tolist())
    bias_category_rouge = {key: 0 for key in bias_values}
    bias_category_count = {key: 0 for key in bias_values}

    idx = 0
    for eachrow in rag_df.itertuples(index=False):
        qid = eachrow.qid
        if dataset_name == "news":
            bias_value = corpus[corpus["docno"]==qid][bias].values[0]
        elif dataset_name == "X":
            bias_value = corpus[corpus["docno"]==int(qid)][bias].values[0]
        else:
            bias_value = corpus[corpus["docno"]==int(qid)][bias].values[0]
        if bias == "general_regions":
            for bv in ast.literal_eval(bias_value):
                bias_category_rouge[bv] += augment_scores[idx]
                bias_category_count[bv] += 1
        else:
            bias_category_rouge[bias_value] += augment_scores[idx]
            bias_category_count[bias_value] += 1
        idx += 1

    bias_category_avg = dict()
    for key in bias_values:
        count = bias_category_count[key]
        if count == 0:
            bias_category_avg[key] = 0  # or float('nan') if you prefer
        else:
            bias_category_avg[key] = bias_category_rouge[key] / count
    
    print(bias_category_avg)
    
    with open(os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_rouge.txt"), "a") as res_file:
        res_file.write(f"{bias}\n")
        res_file.write(str(bias_values))
        res_file.write("\n")
        res_file.write(str(bias_category_avg))
        res_file.write("\n")

def load_df(fp: str) -> pd.DataFrame:
    with open(fp, 'r', encoding='utf-8') as f:
        # Read header and split into column names
        header = f.readline().strip().split('\t')
        
        rows = []
        for i, line in tqdm(enumerate(f, start=2)):  # start=2 for correct line numbering
            line = line.strip()
            if not line:
                print(f"Skipping blank line at {i}")
                continue
            
            values = line.split('\t')
            if len(values) != len(header):
                print(f"Malformed row at line {i}: Expected {len(header)} fields, got {len(values)}")
                continue

            rows.append(values)
    
    # Create DataFrame from list of rows
    df = pd.DataFrame(rows, columns=header)

    # Optionally enforce column types
    df = df.astype({
        "qid": str,
        "pid": str,
        "answer": str,
        "target": str
    })  
    # Must read qid and pid as string not as int
    # dtype_spec = {"qid": str, "pid": str, "answer": str, "target": str}
    # df = pd.read_csv(fp, delimiter="\t", dtype=dtype_spec)
    return df


def get_labels(lamp_num):
    if lamp_num == 1:
        return ["[1]", "[2]"]
    elif lamp_num == 2:
        return [
            "sci-fi",
            "based on a book",
            "comedy",
            "action",
            "twist ending",
            "dystopia",
            "dark comedy",
            "classic",
            "psychology",
            "fantasy",
            "romance",
            "thought-provoking",
            "social commentary",
            "violence",
            "true story",
        ]
    elif lamp_num == 3:
        return ["1", "2", "3", "4", "5"]
    else:
        raise ValueError(f"LaMP {lamp_num} is not classification task")


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
     
    os.makedirs(EVAL_RESULTS_DIR_PATH, exist_ok=True)
    print("loading data ...")
    # load log as dataframe
    rag_df = load_df(
        os.path.join(INF_RESULTS_DIR_PATH, f"{LAMP_NUM}_output_rag.log")
    )
    
    # set corresponding metric function for a LaMP task
    if LAMP_NUM in {1, 2, 3}:
        metric_fn = get_metric_fn_accuracy(get_labels(LAMP_NUM))
    else:
        metric_fn = get_metric_fn_rouge_L()
    
    if (dataset_name == "trec-2021") or (dataset_name == "trec-2022") or (dataset_name == "final_trec_2022"):
        corpus = pd.read_csv(os.path.join(CUR_DIR_PATH, dataset_name, "document_features.csv"))
    else:
        corpus = pd.read_csv(os.path.join(CUR_DIR_PATH, dataset_name, f"{dataset_name}_collection.csv"))
    
    # get augment metric score (u_j) and attach to augment_df
    print("calculating augmented answer qualities ...")
    augment_answers = rag_df["answer"].tolist()
    augment_targets = rag_df["target"].tolist()
    augment_scores: list = metric_fn(augment_answers, augment_targets)
    with open(os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_per_ranking_rouge.txt"), "w") as res_file:
        for idx, augment_score in enumerate(augment_scores):
            res_file.write(f"{idx},{augment_score}\n")

    if (dataset_name == "trec-2022") or (dataset_name == "final_trec_2022"):
        calculate_rouge_over_fairness_groups(corpus, rag_df, "creation_date_category", dataset_name, augment_scores, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
        calculate_rouge_over_fairness_groups(corpus, rag_df, "years_category", dataset_name, augment_scores, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
        calculate_rouge_over_fairness_groups(corpus, rag_df, "relative_pageviews_category", dataset_name, augment_scores, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
        calculate_rouge_over_fairness_groups(corpus, rag_df, "first_letter_category", dataset_name, augment_scores, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
        calculate_rouge_over_fairness_groups(corpus, rag_df, "general_regions", dataset_name, augment_scores, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
    elif dataset_name == "trec-2021":
        calculate_rouge_over_fairness_groups(corpus, rag_df, "geographic_locations", dataset_name, augment_scores, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
    else:
        calculate_rouge_over_fairness_groups(corpus, rag_df, "bias", dataset_name, augment_scores, EVAL_RESULTS_DIR_PATH, LAMP_NUM)
    # # print(augment_scores)
    # average_score = sum(augment_scores) / len(augment_scores)
    # print(average_score)
    
    # bias_values = set(corpus["bias"].tolist())
    # bias_category_rouge = {key: 0 for key in bias_values}
    # bias_category_count = {key: 0 for key in bias_values}

    # idx = 0
    # for eachrow in rag_df.itertuples(index=False):
    #     qid = eachrow.qid
    #     if dataset_name == "news":
    #         bias_value = corpus[corpus["docno"]==qid]["bias"].values[0]
    #     else:
    #         bias_value = corpus[corpus["docno"]==int(qid)]["bias"].values[0]
    #     bias_category_rouge[bias_value] += augment_scores[idx]
    #     bias_category_count[bias_value] += 1
    #     idx += 1

    # bias_category_avg = dict()
    # for key in bias_values:
    #     count = bias_category_count[key]
    #     if count == 0:
    #         bias_category_avg[key] = 0  # or float('nan') if you prefer
    #     else:
    #         bias_category_avg[key] = bias_category_rouge[key] / count
    
    # print(bias_category_avg)
    
    # with open(os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_rouge.txt"), "w") as res_file:
    #     res_file.write(str(bias_values))
    #     res_file.write("\n")
    #     res_file.write(str(average_score))
    #     res_file.write("\n")
    #     res_file.write(str(bias_category_avg))
    #     res_file.write("\n")
    #     res_file.write(str(total_scores/len(grouped)))

    average_score = sum(augment_scores) / len(augment_scores)
    
    delta = pd.read_csv(os.path.join(EVAL_RESULTS_DIR_PATH, "5_delta.tsv"), delimiter="\t")
    grouped = delta.groupby('qid')
    total_scores = 0
    for qid, group in grouped:
        for eachrow in group.itertuples(index=False):
            total_scores += float(eachrow.baseline_score)
            break
    print(total_scores/len(grouped))
    with open(os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_rouge.txt"), "a") as res_file:
        res_file.write(f"AVG baseline score: {total_scores/len(grouped)}\n")
        res_file.write(f"AVG RAG score: {average_score}\n")
    

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
