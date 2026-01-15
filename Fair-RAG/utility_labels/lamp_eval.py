#This script is taken from https://github.com/kimdanny/Fair-RAG

# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/eval/evaluation.py

import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm 

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR_PATH = os.path.dirname(CUR_DIR_PATH)
sys.path.append(PARENT_DIR_PATH)

from eval.lamp_metrics import get_metric_fn_accuracy, get_metric_fn_rouge_L


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
            if len(values) > 4:
                # Merge values[2:-1] (from third to second last) into one string
                merged = '\t'.join(values[2:-1])
                # Reconstruct the list with exactly 4 elements
                values = [values[0], values[1], merged, values[-1]]
                
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
        "inference_results",
        dataset_name,
        task,
        query_topic,
        retriever_name,
        MODEL_NAME,
    )
    EVAL_RESULTS_DIR_PATH = os.path.join(
        CUR_DIR_PATH,
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
    baseline_df = load_df(
        os.path.join(INF_RESULTS_DIR_PATH, f"{LAMP_NUM}_output_baseline.log")
    )
    augment_df = load_df(
        os.path.join(INF_RESULTS_DIR_PATH, f"{LAMP_NUM}_output_augment.log")
    )
    print(len(augment_df))
    # set corresponding metric function for a LaMP task
    if LAMP_NUM in {1, 2, 3}:
        metric_fn = get_metric_fn_accuracy(get_labels(LAMP_NUM))
    else:
        metric_fn = get_metric_fn_rouge_L()


    # get augment metric score (u_j) and attach to augment_df
    print("calculating augmented answer qualities ...")
    augment_answers = augment_df["answer"].tolist()
    augment_targets = augment_df["target"].tolist()
    augment_scores: list = metric_fn(augment_answers, augment_targets)
    augment_df["augment_score"] = augment_scores
    
    # get baseline metric score (u_i) and attach to baseline_df
    # print(baseline_df)
    baseline_answers = baseline_df["answer"].tolist()
    baseline_targets = baseline_df["target"].tolist()
    print("calculating baseline answer qualities ...")
    baseline_scores: list = metric_fn(baseline_answers, baseline_targets)
    baseline_df["baseline_score"] = baseline_scores



    # join baseline_score column to augment_df by qid
    augment_df_joined = pd.merge(
        augment_df, baseline_df[["qid", "baseline_score"]], on="qid", how="left"
    )
    del baseline_df, augment_df

    augment_df_joined = augment_df_joined[
        ["qid", "pid", "augment_score", "baseline_score"]
    ]
    print("calculating differences ...")
    # calculate delta (\delta_j) = (u_j - u_i) and attach to augment_df_joined
    augment_df_joined["delta"] = (
        augment_df_joined["augment_score"] - augment_df_joined["baseline_score"]
    )

    # save to dir
    augment_df_joined.to_csv(
        os.path.join(EVAL_RESULTS_DIR_PATH, f"{LAMP_NUM}_delta.tsv"),
        sep="\t",
        index=False,
    )


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
        help="Help with reading input dataset from the correct directory",
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Help with reading input dataset from the correct directory",
    )

    args = parser.parse_args()

    main(args)
