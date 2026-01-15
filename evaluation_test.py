from tqdm import tqdm   
import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import argparse
import pandas as pd
import numpy as np
# def general_result_analysis(category_utility, category_entailment, category_exposure):
#     avg_utility = category_utility.mean(numeric_only=True).to_frame(name='AVG Utility')
#     avg_consumption = category_entailment.mean(numeric_only=True).to_frame(name='AVG Consumption')
#     avg_exposure = category_exposure.mean(numeric_only=True).to_frame(name='AVG Exposure')
#     consumption_utility_ratio = (category_entailment.mean(numeric_only=True)/category_utility.mean(numeric_only=True)).to_frame(name='consumption_to_utility_ratio')
#     consumption_exposure_ratio = (category_entailment.mean(numeric_only=True)/category_exposure.mean(numeric_only=True)).to_frame(name='consumption_to_exposure_ratio')
#     return pd.concat([avg_utility, avg_consumption, avg_exposure, consumption_utility_ratio, consumption_exposure_ratio], axis=1)

import pandas as pd

# def general_result_analysis(category_utility, category_entailment, category_exposure):
#     # Averages
#     avg_utility = category_utility.mean(numeric_only=True).to_frame(name='AVG Utility')
#     avg_consumption = category_entailment.mean(numeric_only=True).to_frame(name='AVG Consumption')
#     avg_exposure = category_exposure.mean(numeric_only=True).to_frame(name='AVG Exposure')

#     # Probability distributions
#     utility_prob = (avg_utility['AVG Utility'] / avg_utility['AVG Utility'].sum()).to_frame(name='Utility Prob')
#     consumption_prob = (avg_consumption['AVG Consumption'] / avg_consumption['AVG Consumption'].sum()).to_frame(name='Consumption Prob')

#     # Ratio of probabilities
#     prob_ratio = (consumption_prob['Consumption Prob'] / utility_prob['Utility Prob']).to_frame(name='Consumption/Utility Prob Ratio')

#     # Raw ratios
#     consumption_utility_ratio = (
#         category_entailment.mean(numeric_only=True) / category_utility.mean(numeric_only=True)
#     ).to_frame(name='consumption_to_utility_ratio')

#     consumption_exposure_ratio = (
#         category_entailment.mean(numeric_only=True) / category_exposure.mean(numeric_only=True)
#     ).to_frame(name='consumption_to_exposure_ratio')

#     # Combine all results
#     return pd.concat([
#         avg_utility,
#         avg_consumption,
#         avg_exposure,
#         utility_prob,
#         consumption_prob,
#         prob_ratio,
#         consumption_utility_ratio,
#         consumption_exposure_ratio
#     ], axis=1)


import numpy as np

def compute_utility_entailment_misalignment(utility_matrix, entailment_matrix):
    """
    Computes squared L2 norm between utility and entailment rows.

    Parameters:
    - utility_matrix: numpy array of shape (n_rows, 3)
    - entailment_matrix: numpy array of shape (n_rows, 3)

    Returns:
    - alignment_scores: numpy array of shape (n_rows,)
      (Lower = better alignment, Higher = more misalignment)
    """
    differences = utility_matrix - entailment_matrix
    squared_diffs = np.square(differences)
    rowwise_misalignment = np.sum(squared_diffs, axis=1)
    return rowwise_misalignment


import numpy as np

def compute_average_squared_l2_norm(matrix):
    """
    Computes the average squared L2 norm across rows in the matrix.

    Parameters:
    - matrix: numpy array of shape (n_rows, 3), each row sums to 1.

    Returns:
    - average_l2: a single float, average of all rowwise squared L2 norms
    """
    l2_norms = np.sum(np.square(matrix), axis=1)
    average_l2 = np.mean(l2_norms)
    return average_l2


from sklearn.metrics.pairwise import cosine_similarity

def average_cosine_similarity(utility, entailment):
    scores = []
    for u, e in zip(utility, entailment):
        sim = 1-cosine_similarity([u], [e])[0][0]
        scores.append(sim)
        # if np.all(u == 0) and np.all(e == 0):
        #     scores.append(1.0)
        # elif np.all(u == 0) or np.all(e == 0):
        #     scores.append(0.0)
        # else:
        #     sim = cosine_similarity([u], [e])[0][0]
        #     scores.append(sim)
    return np.mean(scores)


def avg_column_distance(utility_matrix, entailment_matrix):
    distances = np.abs(utility_matrix - entailment_matrix)
    return distances.mean(axis=0)  # Average across rows, for each column


# from scipy.spatial.distance import jensenshannon

# def average_jsd_similarity(utility, entailment):
#     scores = []
#     idx = 0
#     for u, e in zip(utility, entailment):
#         if np.all(u == 0) and np.all(e == 0):
#             scores.append(1.0)
#         # elif np.all(u == 0) or np.all(e == 0):
#         #     scores.append(0.0)
#         else:
#             dist = jensenshannon(u, e, base=2)
#             if np.isnan(dist):
#                 print(f"⚠️ NaN JSD at row {idx}")
#                 print(f"utility:   {u}")
#                 print(sum(u))
#                 print(f"entailment:{e}")
#                 print(sum(e))
#                 scores.append(0.0)
#             else:
#                 scores.append(1 - dist)
#         idx += 1
#     return np.mean(scores)



def general_result_analysis(category_utility, category_entailment, category_exposure):
    # Averages
    avg_utility = category_utility.mean(numeric_only=True).to_frame(name='AVG Utility')
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
#     normalized_consumption_to_utility_prob_ratio = (
#     consumption_to_utility_prob_ratio['Consumption-to-Utility Prob Ratio'] /
#     consumption_to_utility_prob_ratio['Consumption-to-Utility Prob Ratio'].sum()
# ).to_frame(name='Normalized Consumption-to-Utility Prob Ratio')


    consumption_exposure_prob_ratio = (
        consumption_prob['Prob Consumption'] / avg_exposure['Prob Exposure']
    ).to_frame(name='C-to-E Ratio')

#     normalized_consumption_to_exposure_prob_ratio = (
#     consumption_exposure_prob_ratio['Consumption-to-Exposure Prob Ratio'] /
#     consumption_exposure_prob_ratio['Consumption-to-Exposure Prob Ratio'].sum()
# ).to_frame(name='Normalized Consumption-to-Exposure Prob Ratio')


    # Combine all
    return pd.concat([
        avg_utility.round(3),
        avg_consumption.round(3),
        avg_exposure.round(3),
        utility_prob.round(3),
        consumption_prob.round(3),
        # consumption_utility_ratio.round(3),
        # consumption_utility_prob.round(3),
        # consumption_exposure_ratio.round(3),
        # consumption_exposure_prob.round(3),
        consumption_to_utility_prob_ratio.round(3),
        # normalized_consumption_to_utility_prob_ratio.round(3),
        consumption_exposure_prob_ratio.round(3),
        # normalized_consumption_to_exposure_prob_ratio.round(3)
    ], axis=1)




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
    utility_matrix = bias_category_utility[bias_order].to_numpy()
    entailment_matrix = bias_category_entailment[bias_order].to_numpy()

    cos_sim = average_cosine_similarity(utility_matrix, entailment_matrix)
    
    print("Distance based on Cosine metric:", cos_sim)

    column_distances = avg_column_distance(utility_matrix, entailment_matrix)
    print(column_distances)

    average_l2 = compute_average_squared_l2_norm(entailment_matrix)
    print("Average squared L2 norm (unfairness score):", average_l2)
    
    alignment_scores = compute_utility_entailment_misalignment(utility_matrix, entailment_matrix)
    # print("Misalignment per row:", alignment_scores)
    print("Average misalignment:", alignment_scores.mean())
        
    # # Example: utility and entailment are matrices of shape (num_rows, 3)
    # difference = utility_matrix - entailment_matrix  # shape: (num_rows, 3)
    
    # # Compute L2 norm row-wise (disparity per row)
    # # disparity_per_row = np.linalg.norm(difference, axis=1)  # shape: (num_rows,)
    
    # # If you want a single summary score for all rows combined:
    # total_disparity = np.linalg.norm(difference)
    
    # # print("Disparity per row:", disparity_per_row)
    # print("Total disparity:", total_disparity)


    
    os.makedirs(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name), exist_ok=True)

    bias_category_utility.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, "bias_utility.csv"))
    bias_category_entailment.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, "bias_entailment.csv"))
    bias_category_exposure.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, "bias_exposure.csv"))
    
    aggregated_result = general_result_analysis(bias_category_utility, bias_category_entailment, bias_category_exposure)
    aggregated_result.to_csv(os.path.join(CUR_DIR_PATH, "final_result", dataset_name, retriever_name, llm_name, "aggregated_result.csv"))
 
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
