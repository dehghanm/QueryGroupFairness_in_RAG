from tqdm import tqdm
import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import argparse
import pandas as pd
import json

def load_df(fp: str) -> pd.DataFrame:
    # Must read qid and pid as string not as int
    dtype_spec = {"qid": str, "pid": str, "answer": str, "target": str}
    df = pd.read_csv(fp, delimiter="\t", dtype=dtype_spec)
    return df

def main(args):
    retriever_name = args.retriever
    if retriever_name == "":
        print("you must pass a valid value for --retriever arg.")
        return

    dataset_name = args.dataset
    if dataset_name == "":
        print("you must pass a valid value for --dataset arg.")
        return

    llm_name = args.llm
    if llm_name == "":
        print("you must pass a valid value for --dataset arg.")
        return
        
    data_path = os.path.join(
        CUR_DIR_PATH,
        dataset_name,
        retriever_name,
        llm_name
    )
    df = load_df(os.path.join(data_path, f"5_output_augment.log"))
    print(len(df))
    mask = df.isnull() | (df == "")
    rows_with_issues = df[mask.any(axis=1)]
    print(rows_with_issues)
    for eachrow in df.itertuples(index=False):
        if eachrow.answer == " ":
            print("here")
        

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        type=str,
        default="",
        help="Help with reading input dataset from the correct directory",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Help with nameing output dir creation",
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="",
        help="Help with nameing output dir creation",
    )
    
    args = parser.parse_args()

    main(args)
