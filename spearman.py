import ast
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import dcor


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

def calculate_kendaltau(all_file_matrices, all_entailment_file_matrices, fairness_category):
    # Step 1: Get the Rouge matrix
    category_to_merge = fairness_category
    rouge_df = pd.concat(
        [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
    )
    
    # Step 2: Get the Attention matrix (C-to-U Ratio)
    attention_df = pd.concat(
        [file_data[category_to_merge] for file_data in all_entailment_file_matrices.values() if category_to_merge in file_data]
    )
    
    # Step 3: Align row orders (important if loaded from different sources)
    rouge_df = rouge_df.sort_index()
    # print(rouge_df)
    attention_df = attention_df.sort_index()
    # print(attention_df)
    
    # Step 4: Convert to NumPy arrays
    rouge_matrix = rouge_df.to_numpy()
    attention_matrix = attention_df.to_numpy()
    
    kendall_correlations = []
    p_value_correlations = []
    for i in range(rouge_matrix.shape[0]):
        rouge_scores = rouge_matrix[i]
        attention_scores = attention_matrix[i]
    
        tau, p_value = kendalltau(rouge_scores, attention_scores)
        p_value_correlations.append(p_value)
        kendall_correlations.append(tau)
    
    # print("Kendall's Tau per condition:", kendall_correlations)
    print("Average Kendall's Tau:", np.mean(kendall_correlations))
    # print(p_value_correlations)


def calcualte_correlation(all_file_matrices, all_entailment_file_matrices, fairness_category):
    # Step 1: Get the Rouge matrix
    category_to_merge = fairness_category
    rouge_df = pd.concat(
        [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
    )
    
    # Step 2: Get the Attention matrix (C-to-U Ratio)
    attention_df = pd.concat(
        [file_data[category_to_merge] for file_data in all_entailment_file_matrices.values() if category_to_merge in file_data]
    )
    
    # Step 3: Align row orders (important if loaded from different sources)
    rouge_df = rouge_df.sort_index()
    # print(rouge_df)
    attention_df = attention_df.sort_index()
    # print(attention_df)
    
    # Step 4: Convert to NumPy arrays
    rouge_matrix = rouge_df.to_numpy()
    attention_matrix = attention_df.to_numpy()
    
    # Step 5: Compute Spearman rank correlations row-wise
    correlations = []
    p_values = []
    for i in range(rouge_matrix.shape[0]):
        rouge_scores = rouge_matrix[i]
        attention_scores = attention_matrix[i]
        corr, p_value = spearmanr(rouge_scores, attention_scores)
        p_values.append(p_value)
        correlations.append(corr)
    
    # Step 6: Compute average correlation
    average_corr = np.mean(correlations)
    # if average_corr < 0:
    #     print(rouge_matrix)
    # average_p_value = np.mean(p_values)

    # Step 7: Print results
    # print("Spearman correlations per condition:", correlations)
    print("Average Spearman rank correlation:", average_corr)
    # print("Average Spearman rank coorelation p_values:", average_p_value)



def calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, all_entailment_file_matrices, fairness_category):
    # Step 1: Get the Rouge matrix
    category_to_merge = fairness_category
    rouge_df = pd.concat(
        [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
    )

    baseline_rouge_df = pd.concat(
        [file_data[category_to_merge] for file_data in all_file_matrices_baseline.values() if category_to_merge in file_data]
    )
    
    # Step 2: Get the Attention matrix (C-to-U Ratio)
    attention_df = pd.concat(
        [file_data[category_to_merge] for file_data in all_entailment_file_matrices.values() if category_to_merge in file_data]
    )
    
    # Step 3: Align row orders (important if loaded from different sources)
    rouge_df = rouge_df.sort_index()
    baseline_rouge_df = baseline_rouge_df.sort_index()
    # print(rouge_df)
    attention_df = attention_df.sort_index()
    # print(attention_df)
    
    # Step 4: Convert to NumPy arrays
    rouge_matrix = rouge_df.to_numpy()
    baseline_rouge_matrix = baseline_rouge_df.to_numpy()
    # difference_rouge_matrix = np.where(
    #     baseline_rouge_matrix != 0,
    #     ((rouge_matrix - baseline_rouge_matrix) / baseline_rouge_matrix) * 100,
    #     0  # or np.nan
    # )    

    difference_rouge_matrix = rouge_matrix - baseline_rouge_matrix
    attention_matrix = attention_df.to_numpy()
    
    # Step 5: Compute Spearman rank correlations row-wise
    correlations = []
    p_values = []
    for i in range(difference_rouge_matrix.shape[0]):
        rouge_scores = difference_rouge_matrix[i]
        attention_scores = attention_matrix[i]
        corr, p_value = spearmanr(rouge_scores, attention_scores)
        p_values.append(p_value)
        correlations.append(corr)
    
    # Step 6: Compute average correlation
    average_corr = np.mean(correlations)
    # if average_corr < 0:
    #     print(difference_rouge_matrix)
    # average_p_value = np.mean(p_values)

    # Step 7: Print results
    # print("Spearman correlations per condition:", correlations)
    print("Average Spearman rank correlation:", average_corr)
    # print("Average Spearman rank coorelation p_values:", average_p_value)    


def get_entailment_to_utility(entailment_path):
    entailment = pd.read_csv(entailment_path, keep_default_na=False)
    group_labels = entailment['Unnamed: 0'].tolist()
    # group_labels = ['None' if x != x else x for x in group_labels]  # NaN != NaN is True
    # print(group_labels)
    exposure_data = {}
    consumption_data = {}
    utility_data = {}
    consumption_to_utility_data = {}
    consumption_to_exposure_data = {}
    for group_label in group_labels:
        exposure_data[group_label] = entailment[entailment['Unnamed: 0'] == group_label]["Prob Exposure"].to_list()[0]
        consumption_data[group_label] = entailment[entailment['Unnamed: 0'] == group_label]["Prob Consumption"].to_list()[0]
        utility_data[group_label] = entailment[entailment['Unnamed: 0'] == group_label]["Prob Utility"].to_list()[0]
        consumption_to_utility_data[group_label] = entailment[entailment['Unnamed: 0'] == group_label]["C-to-U Ratio"].to_list()[0]
        consumption_to_exposure_data[group_label] = entailment[entailment['Unnamed: 0'] == group_label]["C-to-E Ratio"].to_list()[0]
        # "Prob Consumption" "Prob Utility" "Prob Exposure" "C-to-U Ratio" "C-to-E Ratio"
    # dataframe = pd.DataFrame([data], index=[index_name])
    return exposure_data, consumption_data, utility_data, consumption_to_utility_data, consumption_to_exposure_data
    
dataset_name = "final_trec_2022"# "trec-2022" ""final_trec_2022
query_topics = ["666", "1102", "1630"] # "666" , "1630", "1102" "10"
retrievers = ["bm25", "contriver", "splade"] # "bm25", "contriver", "splade"
llms = ["gemma2-9b"] # "gemma2-9b", "llama31-8b"
entailment_model = "roberta-large-mnli"
task = "title_generation" # "title_generation" "article_generation"
print(f"Query topic: {query_topics}")
print(f"LLM: {llms}")
print(f"Task: {task}")
print("*******************")
all_file_matrices = {}  # {filename: {category: DataFrame}}
all_file_matrices_baseline = {}  # {filename: {category: DataFrame}}
utility_all_entailment_file_matrices = {}  # {filename: {category: DataFrame}}
exposure_all_entailment_file_matrices = {}  # {filename: {category: DataFrame}}
consumption_all_entailment_file_matrices = {}  # {filename: {category: DataFrame}} 
consumption_to_utility__all_entailment_file_matrices = {} # {filename: {category: DataFrame}}
consumption_to_exposure__all_entailment_file_matrices = {} # {filename: {category: DataFrame}}
for llm in llms:
    for retriever in retrievers:
        for query_topic in query_topics:    
            rouge_path = os.path.join(
                CUR_DIR_PATH,
                "Fair-RAG",
                "utility_labels",
                "eval_results",
                dataset_name,
                task,
                query_topic,
                retriever,
                llm,
                "5_rouge.txt"
            )
            fairness_data = {}
        
            # === Read and parse each file ===
            with open(rouge_path, "r") as f:
                lines = f.readlines()
        
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.endswith("_category") or (line == "general_regions"):
                    category_name = line
                    group_order = list(ast.literal_eval(lines[i + 1].strip()))
                    scores_dict = ast.literal_eval(lines[i + 2].strip())
                    fairness_data[category_name] = {
                        "group_order": group_order,
                        "scores": scores_dict
                    }
                    i += 3
                else:
                    i += 1
            
            # === Create per-category matrix for this file ===
            file_matrices = {}
            for category, data in fairness_data.items():
                order = data["group_order"]
                scores = data["scores"]
                row = [scores[group] for group in order]
                df = pd.DataFrame([row], columns=order, index=[f"{llm}_{retriever}_{query_topic}"])
                file_matrices[category] = df
        
            all_file_matrices[f"{llm}_{retriever}_{query_topic}"] = file_matrices


            #baseline rouges:
            rouge_path = os.path.join(
                CUR_DIR_PATH,
                "Fair-RAG",
                "utility_labels",
                "eval_results",
                dataset_name,
                task,
                query_topic,
                retriever,
                llm,
                "5_baseline_rouge.txt"
            )
            fairness_data = {}
        
            # === Read and parse each file ===
            with open(rouge_path, "r") as f:
                lines = f.readlines()
        
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.endswith("_category") or (line == "general_regions"):
                    category_name = line
                    group_order = list(ast.literal_eval(lines[i + 1].strip()))
                    scores_dict = ast.literal_eval(lines[i + 2].strip())
                    fairness_data[category_name] = {
                        "group_order": group_order,
                        "scores": scores_dict
                    }
                    i += 3
                else:
                    i += 1
            
            # === Create per-category matrix for this file ===
            file_matrices = {}
            for category, data in fairness_data.items():
                order = data["group_order"]
                scores = data["scores"]
                row = [scores[group] for group in order]
                df = pd.DataFrame([row], columns=order, index=[f"{llm}_{retriever}_{query_topic}"])
                file_matrices[category] = df
        
            all_file_matrices_baseline[f"{llm}_{retriever}_{query_topic}"] = file_matrices
            
            utility_file_matrices = {}
            consumption_file_matrices = {}
            exposure_file_matrices = {}
            consumption_to_utility_file_matrices = {}
            consumption_to_exposure_file_matrices = {}

            for fairness_category in fairness_data.keys():
                entailment_path = os.path.join(
                    CUR_DIR_PATH,
                    "final_result",
                    dataset_name,
                    task,
                    query_topic,
                    retriever,
                    llm,
                    entailment_model,
                    f"{fairness_category}_aggregated_result.csv"
                )
                # print(entailment_path)
                exposure_data_scores, consumption_data_scores, utility_data_scores, consumption_to_utility_data_score, consumption_to_exposure_data_score = get_entailment_to_utility(entailment_path)
                order = fairness_data[fairness_category]["group_order"]
                
                row = [exposure_data_scores[group] for group in order]
                df = pd.DataFrame([row], columns=order, index=[f"{llm}_{retriever}_{query_topic}"])
                exposure_file_matrices[fairness_category] = df

                row = [consumption_data_scores[group] for group in order]
                df = pd.DataFrame([row], columns=order, index=[f"{llm}_{retriever}_{query_topic}"])
                consumption_file_matrices[fairness_category] = df

                row = [utility_data_scores[group] for group in order]
                df = pd.DataFrame([row], columns=order, index=[f"{llm}_{retriever}_{query_topic}"])
                utility_file_matrices[fairness_category] = df

                row = [consumption_to_utility_data_score[group] for group in order]
                df = pd.DataFrame([row], columns=order, index=[f"{llm}_{retriever}_{query_topic}"])
                consumption_to_utility_file_matrices[fairness_category] = df

                row = [consumption_to_exposure_data_score[group] for group in order]
                df = pd.DataFrame([row], columns=order, index=[f"{llm}_{retriever}_{query_topic}"])
                consumption_to_exposure_file_matrices[fairness_category] = df
                
            utility_all_entailment_file_matrices[f"{llm}_{retriever}_{query_topic}"] = utility_file_matrices
            exposure_all_entailment_file_matrices[f"{llm}_{retriever}_{query_topic}"] = exposure_file_matrices
            consumption_all_entailment_file_matrices[f"{llm}_{retriever}_{query_topic}"] = consumption_file_matrices
            consumption_to_utility__all_entailment_file_matrices[f"{llm}_{retriever}_{query_topic}"] = consumption_to_utility_file_matrices
            consumption_to_exposure__all_entailment_file_matrices[f"{llm}_{retriever}_{query_topic}"] = consumption_to_exposure_file_matrices
            

fairness_category = "creation_date_category"
print(fairness_category)
# calculate_pearson(all_file_matrices, all_entailment_file_matrices, fairness_category)
# calculate_peasron_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
# calculate_correlation_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
print("utility vs performence")
calcualte_correlation(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, utility_all_entailment_file_matrices, fairness_category)

print("exposure vs performence")
calcualte_correlation(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, exposure_all_entailment_file_matrices, fairness_category)

print("consumption vs performence")
calcualte_correlation(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_all_entailment_file_matrices, fairness_category)

print("c-to-u vs performence")
calcualte_correlation(all_file_matrices, consumption_to_utility__all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_to_utility__all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_to_utility__all_entailment_file_matrices, fairness_category)

print("c-to-e vs performence")
calcualte_correlation(all_file_matrices, consumption_to_exposure__all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_to_exposure__all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_to_exposure__all_entailment_file_matrices, fairness_category)

print("utility vs consumption")
calcualte_correlation(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)

print("utility vs exposure")
calcualte_correlation(utility_all_entailment_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(utility_all_entailment_file_matrices, exposure_all_entailment_file_matrices, fairness_category)

print("exposure vs consumption")
calcualte_correlation(exposure_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(exposure_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)


print("*****************")

fairness_category = "years_category"
print(fairness_category)
# calculate_pearson(all_file_matrices, all_entailment_file_matrices, fairness_category)
# calculate_peasron_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
# calculate_correlation_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
print("utility vs performence")
calcualte_correlation(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, utility_all_entailment_file_matrices, fairness_category)

print("exposure vs performence")
calcualte_correlation(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, exposure_all_entailment_file_matrices, fairness_category)

print("consumption vs performence")
calcualte_correlation(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_all_entailment_file_matrices, fairness_category)

print("c-to-u vs performence")
calcualte_correlation(all_file_matrices, consumption_to_utility__all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_to_utility__all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_to_utility__all_entailment_file_matrices, fairness_category)

print("c-to-e vs performence")
calcualte_correlation(all_file_matrices, consumption_to_exposure__all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_to_exposure__all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_to_exposure__all_entailment_file_matrices, fairness_category)

print("utility vs consumption")
calcualte_correlation(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)

print("utility vs exposure")
calcualte_correlation(utility_all_entailment_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(utility_all_entailment_file_matrices, exposure_all_entailment_file_matrices, fairness_category)

print("exposure vs consumption")
calcualte_correlation(exposure_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(exposure_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)

print("*****************")

fairness_category = "first_letter_category"
print(fairness_category)
# calculate_pearson(all_file_matrices, all_entailment_file_matrices, fairness_category)
# calculate_peasron_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
# calculate_correlation_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
print("utility vs performence")
calcualte_correlation(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, utility_all_entailment_file_matrices, fairness_category)

print("exposure vs performence")
calcualte_correlation(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, exposure_all_entailment_file_matrices, fairness_category)

print("consumption vs performence")
calcualte_correlation(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_all_entailment_file_matrices, fairness_category)

print("c-to-u vs performence")
calcualte_correlation(all_file_matrices, consumption_to_utility__all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_to_utility__all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_to_utility__all_entailment_file_matrices, fairness_category)

print("c-to-e vs performence")
calcualte_correlation(all_file_matrices, consumption_to_exposure__all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_to_exposure__all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_to_exposure__all_entailment_file_matrices, fairness_category)

print("utility vs consumption")
calcualte_correlation(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)

print("utility vs exposure")
calcualte_correlation(utility_all_entailment_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(utility_all_entailment_file_matrices, exposure_all_entailment_file_matrices, fairness_category)

print("exposure vs consumption")
calcualte_correlation(exposure_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(exposure_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)


print("*****************")

fairness_category = "relative_pageviews_category"
print(fairness_category)
# calculate_pearson(all_file_matrices, all_entailment_file_matrices, fairness_category)
# calculate_peasron_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
# calculate_correlation_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
print("utility vs performence")
calcualte_correlation(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, utility_all_entailment_file_matrices, fairness_category)

print("exposure vs performence")
calcualte_correlation(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, exposure_all_entailment_file_matrices, fairness_category)

print("consumption vs performence")
calcualte_correlation(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_all_entailment_file_matrices, fairness_category)

print("c-to-u vs performence")
calcualte_correlation(all_file_matrices, consumption_to_utility__all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_to_utility__all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_to_utility__all_entailment_file_matrices, fairness_category)

print("c-to-e vs performence")
calcualte_correlation(all_file_matrices, consumption_to_exposure__all_entailment_file_matrices, fairness_category)
calculate_kendaltau(all_file_matrices, consumption_to_exposure__all_entailment_file_matrices, fairness_category)
calcualte_difference_correlation(all_file_matrices, all_file_matrices_baseline, consumption_to_exposure__all_entailment_file_matrices, fairness_category)

print("utility vs consumption")
calcualte_correlation(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)

print("utility vs exposure")
calcualte_correlation(utility_all_entailment_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(utility_all_entailment_file_matrices, exposure_all_entailment_file_matrices, fairness_category)

print("exposure vs consumption")
calcualte_correlation(exposure_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
calculate_kendaltau(exposure_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)

# fairness_category = "general_regions"
# print(fairness_category)
# # calculate_pearson(all_file_matrices, all_entailment_file_matrices, fairness_category)
# # calculate_peasron_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
# # calculate_correlation_2(all_file_matrices, all_entailment_file_matrices, fairness_category)
# print("utility")
# calcualte_correlation(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)
# calculate_kendaltau(all_file_matrices, utility_all_entailment_file_matrices, fairness_category)

# print("exposure")
# calcualte_correlation(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)
# calculate_kendaltau(all_file_matrices, exposure_all_entailment_file_matrices, fairness_category)

# print("consumption")
# calcualte_correlation(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
# calculate_kendaltau(all_file_matrices, consumption_all_entailment_file_matrices, fairness_category)

# print("utility vs consumption")
# calcualte_correlation(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)
# calculate_kendaltau(utility_all_entailment_file_matrices, consumption_all_entailment_file_matrices, fairness_category)

# category_to_merge = "years_category"
# merged_df = pd.concat(
#     [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
# )

# print(f"\nCombined matrix for {category_to_merge}:\n")
# print(merged_df)


# category_to_merge = "first_letter_category"
# merged_df = pd.concat(
#     [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
# )

# print(f"\nCombined matrix for {category_to_merge}:\n")
# print(merged_df)

# category_to_merge = "relative_pageviews_category"
# merged_df = pd.concat(
#     [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
# )

# print(f"\nCombined matrix for {category_to_merge}:\n")
# print(merged_df)








# def calculate_dcor(all_file_matrices, all_entailment_file_matrices, fairness_category):
#     # Step 1: Get the Rouge matrix
#     category_to_merge = fairness_category
#     rouge_df = pd.concat(
#         [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
#     )
    
#     # Step 2: Get the Attention matrix (C-to-U Ratio)
#     attention_df = pd.concat(
#         [file_data[category_to_merge] for file_data in all_entailment_file_matrices.values() if category_to_merge in file_data]
#     )
    
#     # Step 3: Align row orders (important if loaded from different sources)
#     rouge_df = rouge_df.sort_index()
#     # print(rouge_df)
#     attention_df = attention_df.sort_index()
#     # print(attention_df)
    
#     # Step 4: Convert to NumPy arrays
#     rouge_matrix = rouge_df.to_numpy()
#     attention_matrix = attention_df.to_numpy()
#     dcor_correlations = []
#     for i in range(rouge_matrix.shape[0]):
#         rouge_scores = rouge_matrix[i]
#         attention_scores = attention_matrix[i]
    
#         dcor_value = dcor.distance_correlation(rouge_matrix, attention_matrix)
#         dcor_correlations.append(dcor_value)
    
#     # print("Kendall's Tau per condition:", kendall_correlations)
#     print("Average DCor:", np.mean(dcor_correlations))
#     # dcor_value = dcor.distance_correlation(rouge_matrix, attention_matrix)

#     # print("dCore:", dcor_value)



# def calculate_correlation_2(all_file_matrices, all_entailment_file_matrices, fairness_category):
#     # Step 1: Get the Rouge matrix
#     category_to_merge = fairness_category
#     rouge_df = pd.concat(
#         [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
#     )
    
#     # Step 2: Get the Attention matrix (C-to-U Ratio)
#     attention_df = pd.concat(
#         [file_data[category_to_merge] for file_data in all_entailment_file_matrices.values() if category_to_merge in file_data]
#     )
    
#     # Step 1: Sort and align
#     rouge_df = rouge_df.sort_index()
#     attention_df = attention_df.sort_index()
    
#     # Step 2: Average pairwise distance in Rouge-L
#     rouge_disparity = []
#     for row in rouge_df.to_numpy():
#         dists = [abs(a - b) for a, b in combinations(row, 2)]
#         avg_dist = np.mean(dists)
#         rouge_disparity.append(avg_dist)
    
#     # Step 3: L2 squared norm of attention probabilities
#     attention_disparity = []
#     for row in attention_df.to_numpy():
#         prob_row = row / np.sum(row)  # Normalize to probability
#         l2_squared = np.sum(prob_row ** 2)
#         attention_disparity.append(l2_squared)
    
#     # Step 4: Correlation between disparities
#     corr, p_value = spearmanr(rouge_disparity, attention_disparity)
    
#     # Step 5: Print results
#     # print("Rouge disparity (avg. pairwise distances):", rouge_disparity)
#     # print("Attention disparity (L2 squared norms):", attention_disparity)
#     print("Spearman rank correlation between avg group distance and c-to-u disparity:", corr)
#     print("P-value:", p_value)



# def calculate_pearson(all_file_matrices, all_entailment_file_matrices, fairness_category):
# # Step 1: Get the Rouge matrix
#     category_to_merge = fairness_category
#     rouge_df = pd.concat(
#         [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
#     )
    
#     # Step 2: Get the Attention matrix (C-to-U Ratio)
#     attention_df = pd.concat(
#         [file_data[category_to_merge] for file_data in all_entailment_file_matrices.values() if category_to_merge in file_data]
#     )
    
#     # Step 3: Align row orders (important if loaded from different sources)
#     rouge_df = rouge_df.sort_index()
#     # print(rouge_df)
#     attention_df = attention_df.sort_index()
#     # print(attention_df)
    
#     # Step 4: Convert to NumPy arrays
#     rouge_matrix = rouge_df.to_numpy()
#     attention_matrix = attention_df.to_numpy()
    
#     # Step 5: Compute Spearman rank correlations row-wise
#     correlations = []
#     p_values = []
#     for i in range(rouge_matrix.shape[0]):
#         rouge_scores = rouge_matrix[i]
#         attention_scores = attention_matrix[i]
#         corr, p_value = pearsonr(rouge_scores, attention_scores)
#         correlations.append(corr)
#         p_values.append(p_value)
    
#     # Step 6: Compute average correlation
#     average_corr = np.mean(correlations)
#     # average_p_value = np.mean(p_values)
#     print(average_corr) #, average_p_value
    



# def calculate_peasron_2(all_file_matrices, all_entailment_file_matrices, fairness_category):
#     # Step 1: Get the Rouge matrix
#     category_to_merge = fairness_category
#     rouge_df = pd.concat(
#         [file_data[category_to_merge] for file_data in all_file_matrices.values() if category_to_merge in file_data]
#     )
    
#     # Step 2: Get the Attention matrix (C-to-U Ratio)
#     attention_df = pd.concat(
#         [file_data[category_to_merge] for file_data in all_entailment_file_matrices.values() if category_to_merge in file_data]
#     )
    
#     # Step 1: Sort and align
#     rouge_df = rouge_df.sort_index()
#     attention_df = attention_df.sort_index()
    
#     # Step 2: Average pairwise distance in Rouge-L
#     rouge_disparity = []
#     for row in rouge_df.to_numpy():
#         dists = [abs(a - b) for a, b in combinations(row, 2)]
#         avg_dist = np.mean(dists)
#         rouge_disparity.append(avg_dist)
    
#     # Step 3: L2 squared norm of attention probabilities
#     attention_disparity = []
#     for row in attention_df.to_numpy():
#         prob_row = row / np.sum(row)  # Normalize to probability
#         l2_squared = np.sum(prob_row ** 2)
#         attention_disparity.append(l2_squared)
    
#     # Step 4: Correlation between disparities
#     corr, p_value = pearsonr(rouge_disparity, attention_disparity)
    
#     # Step 5: Print results
#     # print("Rouge disparity (avg. pairwise distances):", rouge_disparity)
#     # print("Attention disparity (L2 squared norms):", attention_disparity)
#     print("Pearson correlation between avg group distance and c-to-u disparity:", corr)
#     print("P-value:", p_value)
