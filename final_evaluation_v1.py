from tqdm import tqdm   
import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import argparse
import pandas as pd
import numpy as np
import pandas as pd
import ast


def general_result_analysis(category_utility, category_entailment, category_exposure, bias_category_idx):
    # Averages
    # avg_utility = category_utility.mean(numeric_only=True).to_frame(name='AVG Utility')
    # Collect values based on dict
    selected = []
    for row_idx, col in bias_category_idx.items():
        selected.append({"column": col, "value": category_utility.loc[row_idx, col]})
    selected_df = pd.DataFrame(selected)
    # Calculate average per column
    avg_utility = selected_df.groupby("column")["value"].mean()
        
    avg_consumption = category_entailment.mean(numeric_only=True).to_frame(name='AVG Consumption')    
    avg_exposure = category_exposure.mean(numeric_only=True).to_frame(name='Prob Exposure')

    # Raw consumption over utility ratios
    consumption_utility_ratio = (
        category_entailment.mean(numeric_only=True) / category_utility.mean(numeric_only=True)
    ).to_frame(name='consumption_to_utility_ratio')

    # Normalize the consumption-to-utility ratio into a probability distribution
    consumption_utility_prob = (consumption_utility_ratio['consumption_to_utility_ratio'] /
                                consumption_utility_ratio['consumption_to_utility_ratio'].sum()
                               ).to_frame(name='Consumption-to-Utility Ratio Prob')
    
    #raw consumption over exposure ratio
    consumption_exposure_ratio = (
        category_entailment.mean(numeric_only=True) / category_exposure.mean(numeric_only=True)
    ).to_frame(name='consumption_to_exposure_ratio')

    # Normalize the consumption-to-exposure ratio into a probability distribution
    consumption_exposure_prob = (consumption_exposure_ratio['consumption_to_exposure_ratio'] /
                                consumption_exposure_ratio['consumption_to_exposure_ratio'].sum()
                               ).to_frame(name='Consumption-to-Exposure Ratio Prob')

    
    # Probability distributions
    utility_prob = (avg_utility['AVG Utility'] / avg_utility['AVG Utility'].sum()).to_frame(name='Prob Utility')
    consumption_prob = (avg_consumption['AVG Consumption'] / avg_consumption['AVG Consumption'].sum()).to_frame(name='Prob Consumption')
    consumption_to_utility_prob_ratio = (consumption_prob['Prob Consumption'] / utility_prob['Prob Utility']).to_frame(name='C-to-U Ratio')


    consumption_exposure_prob_ratio = (
        consumption_prob['Prob Consumption'] / avg_exposure['Prob Exposure']
    ).to_frame(name='C-to-E Ratio')

    # Combine all
    return pd.concat([
        avg_utility.round(3),
        avg_consumption.round(3),
        avg_exposure.round(3),
        utility_prob.round(3),
        consumption_prob.round(3),
        consumption_utility_ratio.round(3),
        consumption_to_utility_prob_ratio.round(3),
        consumption_exposure_prob_ratio.round(3),
    ], axis=1)


def matrix_calculation(dataset_name, fairness_feature, retriever_name, llm_name, entailment_model, query_topic, task):
    if "trec" in dataset_name:
        corpus = pd.read_csv(os.path.join(CUR_DIR_PATH, dataset_name, f"document_features.csv"))
        if fairness_feature == "general_regions":
            bias_values = set()
            bias_values_lists = corpus[fairness_feature].tolist()
            for bias_values_list in bias_values_lists:
                for bias_value in ast.literal_eval(bias_values_list):
                    bias_values.add(bias_value)
        else:
            bias_values = set(corpus[fairness_feature].tolist())
    else:
        corpus = pd.read_csv(os.path.join(CUR_DIR_PATH, dataset_name, f"{dataset_name}_collection.csv"))
        bias_values = set(corpus[fairness_feature].tolist())

    entailment = pd.read_csv(os.path.join(CUR_DIR_PATH, "Fair-RAG/entailment", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, "entailment.csv"))
    utility = pd.read_csv(os.path.join(CUR_DIR_PATH, "Fair-RAG/utility_labels/eval_results", dataset_name, task, query_topic, retriever_name, llm_name, "5_delta.tsv"), delimiter="\t")
    
    bias_category_utility = pd.DataFrame(columns=list(bias_values))
    bias_category_entailment = pd.DataFrame(columns=list(bias_values))
    bias_category_exposure = pd.DataFrame(columns=list(bias_values))
    bias_category_idx = dict()
    grouped = entailment.groupby('doc_id')

    query_idx = 0
    # Iterate through the groups
    for group_name, group_df in tqdm(grouped):
        doc_id = group_name
        
        bias_category_dict = dict()
        for value in bias_values:
            bias_category_dict[value] = {"utility":0, "entailment": 0, "exposure": 0, "count": 0}

        query_bias_value = corpus[corpus["docno"]==doc_id][fairness_feature].values[0]
        bias_category_idx[query_idx] = query_bias_value
        query_idx += 1
        
        rank_position = 1
        for eachrow in group_df.itertuples(index=False):
            profile_doc_id = eachrow.profile_doc_id
            entailment_score = int(eachrow.score >= 0.5) #eachrow.score int(eachrow.score > 0.5)
            utility_score = utility[(utility["qid"] == doc_id) & (utility["pid"] == profile_doc_id)]["delta"].values[0]
            bias_value = corpus[corpus["docno"]==profile_doc_id][fairness_feature].values[0]
            if fairness_feature == "general_regions":
                for bv in ast.literal_eval(bias_value):
                    bias_category_dict[bv]["entailment"] += float(entailment_score)
                    bias_category_dict[bv]["utility"] += max(0, float(utility_score)) # (int(utility_score) * (1/(math.log2(rank_position)+1))) max(0, float(utility_score)) int(float(utility_score) > 0)
                    bias_category_dict[bv]["count"] += 1
            else:
                if bias_value != query_bias_value:
                    continue
                bias_category_dict[bias_value]["entailment"] += float(entailment_score)
                bias_category_dict[bias_value]["utility"] +=  max(0, float(utility_score)) # (int(utility_score) * (1/(math.log2(rank_position)+1))) max(0, float(utility_score)) int(float(utility_score) > 0)
                bias_category_dict[bias_value]["count"] += 1
            rank_position += 1

        for key, vals in bias_category_dict.items():
            count = vals['count']
            if count == 0:
                continue
            # vals['utility'] /= count
            # vals['entailment'] /= count
            vals['exposure'] = count/len(group_df)

        # # Normalize utility
        # utility_total = sum(vals['utility'] for vals in bias_category_dict.values())
        # if utility_total == 0:
        #     pass
        # else:
        #     for vals in bias_category_dict.values():
        #         vals['utility'] /= utility_total
        
        # # Normalize entailment
        # entailment_total = sum(vals['entailment'] for vals in bias_category_dict.values())
        # if entailment_total == 0:
        #     pass
        # else:
        #     for vals in bias_category_dict.values():
        #         vals['entailment'] /= entailment_total

        utility_scores = {k: v['utility'] for k, v in bias_category_dict.items()}
        bias_category_utility.loc[len(bias_category_utility)] = utility_scores

        entailment_scores = {k: v['entailment'] for k, v in bias_category_dict.items()}
        bias_category_entailment.loc[len(bias_category_entailment)] = entailment_scores

        exposure_scores = {k: v['exposure'] for k, v in bias_category_dict.items()}
        bias_category_exposure.loc[len(bias_category_exposure)] = exposure_scores
        
    return bias_category_utility, bias_category_entailment, bias_category_exposure, bias_category_idx


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

    query_topic = args.query_topic
    if query_topic == "":
        print("you must pass a valid value for --query_topic arg.")
        return

    task = args.task
    if task == "":
        print("you must pass a valid value for --task arg.")
        return
    
    entailment_model = args.entailment_model
    if entailment_model == "":
        print("you must pass a valid value for --entailment_model arg.")
        return

    os.makedirs(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model), exist_ok=True)
    
    if "trec" in dataset_name:
        fairness_feature = "years_category"
        bias_category_utility, bias_category_entailment, bias_category_exposure, bias_category_idx = matrix_calculation(dataset_name, fairness_feature, retriever_name, llm_name, entailment_model, query_topic, task)
        bias_category_utility.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_utility.csv"))
        bias_category_entailment.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_entailment.csv"))
        bias_category_exposure.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_exposure.csv"))
        aggregated_result = general_result_analysis(bias_category_utility, bias_category_entailment, bias_category_exposure, bias_category_idx)
        aggregated_result.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_aggregated_result.csv"))

        fairness_feature = "creation_date_category"
        bias_category_utility, bias_category_entailment, bias_category_exposure = matrix_calculation(dataset_name, fairness_feature, retriever_name, llm_name, entailment_model, query_topic, task)
        bias_category_utility.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_utility.csv"))
        bias_category_entailment.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_entailment.csv"))
        bias_category_exposure.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_exposure.csv"))
        aggregated_result = general_result_analysis(bias_category_utility, bias_category_entailment, bias_category_exposure)
        aggregated_result.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_aggregated_result.csv"))

        fairness_feature = "relative_pageviews_category"
        bias_category_utility, bias_category_entailment, bias_category_exposure = matrix_calculation(dataset_name, fairness_feature, retriever_name, llm_name, entailment_model, query_topic, task)
        bias_category_utility.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_utility.csv"))
        bias_category_entailment.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_entailment.csv"))
        bias_category_exposure.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_exposure.csv"))
        aggregated_result = general_result_analysis(bias_category_utility, bias_category_entailment, bias_category_exposure)
        aggregated_result.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_aggregated_result.csv"))

        fairness_feature = "first_letter_category"
        bias_category_utility, bias_category_entailment, bias_category_exposure = matrix_calculation(dataset_name, fairness_feature, retriever_name, llm_name, entailment_model, query_topic, task)
        bias_category_utility.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_utility.csv"))
        bias_category_entailment.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_entailment.csv"))
        bias_category_exposure.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_exposure.csv"))
        aggregated_result = general_result_analysis(bias_category_utility, bias_category_entailment, bias_category_exposure)
        aggregated_result.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_aggregated_result.csv"))

        # fairness_feature = "general_regions"
        # bias_category_utility, bias_category_entailment, bias_category_exposure = matrix_calculation(dataset_name, fairness_feature, retriever_name, llm_name, entailment_model, query_topic, task)
        # bias_category_utility.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_utility.csv"))
        # bias_category_entailment.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_entailment.csv"))
        # bias_category_exposure.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_exposure.csv"))
        # aggregated_result = general_result_analysis(bias_category_utility, bias_category_entailment, bias_category_exposure)
        # aggregated_result.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, task, query_topic, retriever_name, llm_name, entailment_model, f"{fairness_feature}_aggregated_result.csv"))
        
    elif dataset_name == "news":
        bias_category_utility, bias_category_entailment, bias_category_exposure = matrix_calculation(dataset_name, "bias", retriever_name, llm_name, entailment_model)
        bias_category_utility.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_utility.csv"))
        bias_category_entailment.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_entailment.csv"))
        bias_category_exposure.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_exposure.csv"))
        aggregated_result = general_result_analysis(bias_category_utility, bias_category_entailment, bias_category_exposure)
        aggregated_result.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_aggregated_result.csv"))
    else:
        bias_category_utility, bias_category_entailment, bias_category_exposure = matrix_calculation(dataset_name, "bias", retriever_name, llm_name, entailment_model)
        bias_category_utility.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_utility.csv"))
        bias_category_entailment.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_entailment.csv"))
        bias_category_exposure.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_exposure.csv"))
        aggregated_result = general_result_analysis(bias_category_utility, bias_category_entailment, bias_category_exposure)
        aggregated_result.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, entailment_model, "bias_aggregated_result.csv"))
 
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