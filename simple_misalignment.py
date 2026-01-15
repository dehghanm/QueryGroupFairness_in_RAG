from tqdm import tqdm   
import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import argparse
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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



    import pandas as pd
    import numpy as np
    from collections import defaultdict
    
    # Load data
    entailment = pd.read_csv('/mnt/data/entailment.csv')
    delta = pd.read_csv('/mnt/data/delta.csv')
    corpus = pd.read_csv('/mnt/data/corpus.csv')
    
    # Step 1: Label usage from entailment (usage = 1 if score > 0.5)
    entailment['usage'] = entailment['score'] > 0.5
    
    # Step 2: Label relevance from delta (relevant = 1 if delta > 0)
    delta['relevant'] = delta['delta'] > 0
    
    # Step 3: Merge everything into a single DataFrame
    merged = entailment.merge(delta, left_on=['doc_id', 'profile_doc_id'], right_on=['qid', 'pid'])
    merged = merged.merge(corpus[['docno', 'bias']], left_on='profile_doc_id', right_on='docno')
    
    # Step 4: Compute per-query bias score
    all_groups = merged['bias'].unique().tolist()
    
    def compute_bias_score(df):
        group_to_idx = {g: i for i, g in enumerate(all_groups)}
        rel_dist = np.zeros(len(all_groups))
        use_dist = np.zeros(len(all_groups))
    
        for _, row in df.iterrows():
            idx = group_to_idx[row['bias']]
            rel_dist[idx] += int(row['relevant'])
            use_dist[idx] += int(row['usage'])
    
        if rel_dist.sum() > 0:
            rel_dist /= rel_dist.sum()
        if use_dist.sum() > 0:
            use_dist /= use_dist.sum()
    
        return np.sum((rel_dist - use_dist) ** 2)
    
    bias_scores = merged.groupby('doc_id').apply(compute_bias_score).reset_index()
    bias_scores.columns = ['query_id', 'bias_score']
    
    # Step 5: Compute per-group precision and recall
    def per_group_precision(df):
        group_correct = defaultdict(int)
        group_total = defaultdict(int)
        for _, row in df.iterrows():
            if row['usage']:
                group_total[row['bias']] += 1
                if row['relevant']:
                    group_correct[row['bias']] += 1
        return {g: group_correct[g] / group_total[g] if group_total[g] > 0 else None for g in group_total}
    
    def per_group_recall(df):
        group_relevant = defaultdict(int)
        group_used_relevant = defaultdict(int)
        for _, row in df.iterrows():
            if row['relevant']:
                group_relevant[row['bias']] += 1
                if row['usage']:
                    group_used_relevant[row['bias']] += 1
        return {g: group_used_relevant[g] / group_relevant[g] if group_relevant[g] > 0 else None for g in group_relevant}
    
    precision_per_group = per_group_precision(merged)
    recall_per_group = per_group_recall(merged)
    
    bias_scores.head(), precision_per_group, recall_per_group
    
    










    
    corpus = pd.read_csv(os.path.join(CUR_DIR_PATH, dataset_name, f"{dataset_name}_collection.csv"))
    bias_values = set(corpus["bias"].tolist())
    
    entailment = pd.read_csv(os.path.join(CUR_DIR_PATH, "Fair-RAG/entailment", dataset_name, retriever_name, llm_name, entailment_model, "entailment.csv"))
    utility = pd.read_csv(os.path.join(CUR_DIR_PATH, "Fair-RAG/utility_labels/eval_results", dataset_name, retriever_name, llm_name, "5_delta.tsv"), delimiter="\t")
    
    bias_category_utility = pd.DataFrame(columns=list(bias_values))
    bias_category_entailment = pd.DataFrame(columns=list(bias_values))
    bias_category_exposure = pd.DataFrame(columns=list(bias_values))
    grouped = entailment.groupby('doc_id')

    # Iterate through the groups
    for group_name, group_df in tqdm(grouped):
        doc_id = group_name
        bias_category_dict = dict()
        for value in bias_values:
            bias_category_dict[value] = {"utility":0, "entailment": 0, "exposure": 0, "count": 0}
    
        rank_position = 1
        for eachrow in group_df.itertuples(index=False):
            profile_doc_id = eachrow.profile_doc_id
            entailment_score = int(eachrow.score > 0.5) #eachrow.score int(eachrow.score > 0.5)
            utility_score = utility[(utility["qid"] == doc_id) & (utility["pid"] == profile_doc_id)]["delta"].values[0]
            bias_value = corpus[corpus["docno"]==profile_doc_id]["bias"].values[0]
            bias_category_dict[bias_value]["entailment"] += float(entailment_score)
            bias_category_dict[bias_value]["utility"] += int(float(utility_score) > 0) # (int(utility_score) * (1/(math.log2(rank_position)+1))) max(0, float(utility_score)) int(float(utility_score) > 0)
            bias_category_dict[bias_value]["count"] += 1
            rank_position += 1

        for key, vals in bias_category_dict.items():
            count = vals['count']
            if count == 0:
                continue
            vals['utility'] /= count
            vals['entailment'] /= count
            vals['exposure'] = count/len(group_df)

        # Normalize utility
        utility_total = sum(vals['utility'] for vals in bias_category_dict.values())
        if utility_total == 0:
            pass
        else:
            for vals in bias_category_dict.values():
                vals['utility'] /= utility_total
        
        # Normalize entailment
        entailment_total = sum(vals['entailment'] for vals in bias_category_dict.values())
        if entailment_total == 0:
            pass
        else:
            for vals in bias_category_dict.values():
                vals['entailment'] /= entailment_total

        utility_scores = {k: v['utility'] for k, v in bias_category_dict.items()}
        bias_category_utility.loc[len(bias_category_utility)] = utility_scores

        entailment_scores = {k: v['entailment'] for k, v in bias_category_dict.items()}
        bias_category_entailment.loc[len(bias_category_entailment)] = entailment_scores

        exposure_scores = {k: v['exposure'] for k, v in bias_category_dict.items()}
        bias_category_exposure.loc[len(bias_category_exposure)] = exposure_scores
    if dataset_name == "news":
        bias_order = ['left', 'center', 'right']
    else:
        bias_order = ['Left', 'Center', 'Right']

    os.makedirs(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model), exist_ok=True)
    utility_matrix = bias_category_utility[bias_order].to_numpy()
    entailment_matrix = bias_category_entailment[bias_order].to_numpy()
    cos_sim = average_cosine_similarity(utility_matrix, entailment_matrix, os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "cosine_per_row.txt"))
    print("AUM: ", cos_sim)

    eu_dist = average_euclidean_distance(utility_matrix, entailment_matrix, os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "euclidian_per_row.txt"))
    print("AED: ", eu_dist)
    
    column_distances = avg_column_distance(utility_matrix, entailment_matrix)
    print(column_distances)

    average_l2 = compute_average_squared_l2_norm(entailment_matrix, os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "attribution_disparity_per_row.txt"))
    print("GAD", average_l2)
    
    alignment_scores = compute_utility_entailment_misalignment(utility_matrix, entailment_matrix)
    print("UAAD", alignment_scores.mean())
    with open(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "utility_attribution_alignment_disparity_per_row.txt"), "w") as f:
        for i, val in enumerate(alignment_scores):
            f.write(f"{i}, {val}\n")


    with open(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "other_metrics.txt"), "w") as other_metric_file:
        other_metric_file.write(f"{bias_order}\n")
        other_metric_file.write(f"Distance based on Cosine metric: {cos_sim} \n")
        other_metric_file.write(f"Average distance between utility and consumption for each category: {column_distances} \n")
        other_metric_file.write(f"Average squared L2 norm over entailment matrix (unfairness score): {average_l2} \n")
        other_metric_file.write(f"Average squared L2 norm over utility-entailment matrix: {alignment_scores.mean()} \n")

    bias_category_utility.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_utility.csv"))
    bias_category_entailment.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_entailment.csv"))
    bias_category_exposure.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_exposure.csv"))
    
    aggregated_result = general_result_analysis(bias_category_utility, bias_category_entailment, bias_category_exposure)
    aggregated_result.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "aggregated_result.csv"))
 
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
