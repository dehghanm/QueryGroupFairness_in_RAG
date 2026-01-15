from tqdm import tqdm   
import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def analysis(performance, unfairness, path, unfairness_method):
    # Basic sanity check
    assert len(performance) == len(unfairness), "Mismatch in number of queries."
    
    # Scatter plot with regression line
    plt.figure(figsize=(8, 6))
    sns.regplot(x=unfairness, y=performance, scatter_kws={"s": 40}, line_kws={"color": "red"})
    plt.xlabel("Unfairness Score (Squared L2 Norm)")
    plt.ylabel("Model Performance (e.g., ROUGE-L)")
    plt.title("Correlation Between Fairness Misalignment and Model Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{unfairness_method}_fig.png"))
    plt.show()

    with open(os.path.join(path, f"{unfairness_method}_correlation.txt"), "w") as correlation_file:
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(unfairness, performance)
        print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
        correlation_file.write(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})\n")
        # Spearman correlation (non-parametric)
        spearman_corr, spearman_p = spearmanr(unfairness, performance)
        print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
        correlation_file.write(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})\n")



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
    if dataset_name == "":
        print("you must pass a valid value for --llm arg.")
        return
    
    entailment_model = args.entailment_model
    if entailment_model == "":
        print("you must pass a valid value for --entailment_model arg.")
        return

    
    # Load model performance scores and unfairness scores
    # These should be 1D lists or numpy arrays of the same length
    # For example:
    # performance = [0.76, 0.55, ...]
    # unfairness = [0.03, 0.27, ...]
    writing_result_path = os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model)
    input_path = os.path.join(CUR_DIR_PATH, "Fair-RAG/utility_labels/eval_results", dataset_name, retriever_name, llm_name)
    performance = np.loadtxt(os.path.join(input_path, "5_per_ranking_rouge.txt"), delimiter=",", usecols=1)
    
    unfairness = np.loadtxt(os.path.join(writing_result_path, "cosine_per_row.txt"), delimiter=",", usecols=1)
    analysis(performance, unfairness, writing_result_path, unfairness_method="cosine")

    unfairness = np.loadtxt(os.path.join(writing_result_path, "attribution_disparity_per_row.txt"), delimiter=",", usecols=1)
    analysis(performance, unfairness, writing_result_path, unfairness_method="attribution_disparity")

    unfairness = np.loadtxt(os.path.join(writing_result_path, "utility_attribution_alignment_disparity_per_row.txt"), delimiter=",", usecols=1)
    analysis(performance, unfairness, writing_result_path, unfairness_method="utility_attribution_alignment_disparity")

    unfairness = np.loadtxt(os.path.join(writing_result_path, "euclidian_per_row.txt"), delimiter=",", usecols=1)
    analysis(performance, unfairness, writing_result_path, unfairness_method="euclidian")

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

    parser.add_argument(
        "--entailment_model",
        type=str,
        default="",
        help="Help with nameing output dir creation",
    )
    args = parser.parse_args()

    main(args)
