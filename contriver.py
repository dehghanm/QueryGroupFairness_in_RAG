import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import re
import pandas as pd
import ast
import json
import os
import argparse


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def batchify(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]

@torch.no_grad()
def encode_corpus(contriver, tokenizer, corpus, batch_size, embedding_path):
    counter = 0
    os.makedirs(f"./{embedding_path}/embeddings/", exist_ok=True)
    for batch in tqdm(batchify(corpus, batch_size)):
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda:0")
        outputs = contriver(**tokens)
        embeddings = mean_pooling(outputs.last_hidden_state, tokens["attention_mask"])
        torch.save(embeddings, f"./{embedding_path}/embeddings/batch_{counter}.pt")
        counter += 1
    return counter


@torch.no_grad()
def retrieve_top_k_with_contriver(contriver, tokenizer, profile, query, k, number_of_batches, path) -> list[tuple]:
    """
    Uses precomputed corpus_embeddings to retrieve top-k documents for a given query.
    """
    # Compute query embedding
    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors="pt").to("cuda:0")
    output_query = contriver(**query_tokens)
    query_embedding = mean_pooling(output_query.last_hidden_state, query_tokens["attention_mask"])  # (1, dim)
    scores = []
    for i in range(number_of_batches):
        batch_embeddings = torch.load(f"./{path}/embeddings/batch_{i}.pt").to("cuda:0")
        temp_scores = query_embedding.squeeze() @ batch_embeddings.T
        scores.extend(temp_scores.tolist())

    topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
    topk_values = topk_values.tolist()
    topk_indices = topk_indices.tolist()
    return [(profile[i], score) for score, i in zip(topk_values, topk_indices)]

def preprocess(doc):
    cleaned_text = str(doc).replace("\n", " ").replace("\r", " ")
    clean_text = re.sub(r'\s+', ' ', cleaned_text)
    return clean_text



def main(args):
    dataset_name = args.dataset
    if dataset_name == "":
        print("you must pass a valid value for --dataset arg.")
        return
    
    query_topic = args.query_topic
    if query_topic == "":
        print("you must pass a valid value for --query_topic arg.")
        return

    task = args.task
    if task == "":
        print("you must pass a valid value for --task arg.")
        return
    do_embedding: bool = args.embedding

    if dataset_name == "trec-2021":
        os.makedirs(f"./trec-2021/{task}/{query_topic}/contriver", exist_ok=True)
    elif dataset_name == "trec-2022":
        os.makedirs(f"./trec-2022/{task}/{query_topic}/contriver", exist_ok=True)
    
    if dataset_name == "trec-2021":
        corpus_path = "./trec-2021/corpus.csv"
        corpus = pd.read_csv(corpus_path)
    elif dataset_name == "trec-2022":
        corpus_path = "./trec-2022/corpus.csv"
        corpus = pd.read_csv(corpus_path)
        
    corpus["document"] = corpus["title"] + "\n" + corpus["text"]
    corpus["processed_text"] = corpus["text"].apply(preprocess)
    corpus["processed_title"] = corpus["title"].apply(preprocess)
    corpus["processed_document"] = corpus["processed_title"] + "\n" + corpus["processed_text"]
    document_list = corpus["processed_document"].tolist()
    profile = corpus[["docno", "document"]].to_dict(orient='records')
    
    contriver_checkpoint = "facebook/contriever"
    tokenizer = AutoTokenizer.from_pretrained(contriver_checkpoint)
    contriver = AutoModel.from_pretrained(contriver_checkpoint).to("cuda:0")
    contriver.eval()
    
    batch_size = 1500
    if dataset_name == "trec-2021":
        embedding_path = "trec-2021/contriver"
    elif dataset_name == "trec-2022":
        embedding_path = "trec-2022/contriver"
    if do_embedding:
        number_of_batches = encode_corpus(contriver, tokenizer, document_list, batch_size, embedding_path)
    else:
        number_of_batches = len([f for f in os.listdir(f"./{embedding_path}/embeddings/") if os.path.isfile(os.path.join(f"./{embedding_path}/embeddings", f))])

    
    contriver_result = []
    if dataset_name == "trec-2021":
        query_df = pd.read_csv(f"./trec-2021/query_ids_{query_topic}.csv")
    elif dataset_name == "trec-2022":
        query_df = pd.read_csv(f"./trec-2022/query_ids_{query_topic}.csv")

    for eachrow in tqdm(query_df.itertuples(index=False)):
        query_docno = eachrow.docno
        if task == "title_generation":
            query = corpus.loc[corpus["docno"] == query_docno, "text"].values[0]
            preprocessed_query = preprocess(query)
            if dataset_name == "trec-2021":
                selected_profs_with_scores = retrieve_top_k_with_contriver(contriver, tokenizer, profile, preprocessed_query, 200, number_of_batches, "trec-2021/contriver")
            elif dataset_name == "trec-2022":
                selected_profs_with_scores = retrieve_top_k_with_contriver(contriver, tokenizer, profile, preprocessed_query, 200, number_of_batches, "trec-2022/contriver")
        else:
            query = corpus.loc[corpus["docno"] == query_docno, "title"].values[0]
            preprocessed_query = preprocess(query)
            if dataset_name == "trec-2021":
                selected_profs_with_scores = retrieve_top_k_with_contriver(contriver, tokenizer, profile, preprocessed_query, 200, number_of_batches, "trec-2021/contriver")
            elif dataset_name == "trec-2022":
                selected_profs_with_scores = retrieve_top_k_with_contriver(contriver, tokenizer, profile, preprocessed_query, 200, number_of_batches, "trec-2022/contriver")
            
        related_doc_ids = set()
        for i in range(200):
            if str(query_docno) == str(selected_profs_with_scores[i][0]["docno"]):
                continue

            if str(selected_profs_with_scores[i][0]["docno"]) in related_doc_ids:
                continue

            related_doc_ids.add(selected_profs_with_scores[i][0]["docno"])
            if len(related_doc_ids) == 10:
                break
        if len(related_doc_ids) < 10:
            print(f"something is wrong with this query: {query_docno}")
        contriver_result.append(related_doc_ids)
    
    for query_idx, res in enumerate(contriver_result):
        if len(set(res)) < 10:
            print(f"found a mistake in a query: {query_idx}")
    
    if dataset_name == "trec-2021":
        if task == "title_generation":
            query_df["rel_docno"] = contriver_result
            query_df.to_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_contriver.csv", index=False)
        else:
            query_df["rel_docno"] = contriver_result
            query_df.to_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_contriver.csv", index=False)
    elif dataset_name == "trec-2022":
        if task == "title_generation":
            query_df["rel_docno"] = contriver_result
            query_df.to_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_contriver.csv", index=False)
        else:
            query_df["rel_docno"] = contriver_result
            query_df.to_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_contriver.csv", index=False)
    
    if dataset_name == "trec-2021":
        if task == "title_generation":
            query_set_with_contriver = pd.read_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_contriver.csv")
        else:
            query_set_with_contriver = pd.read_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_contriver.csv")
    elif dataset_name == "trec-2022":
        if task == "title_generation":
            query_set_with_contriver = pd.read_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_contriver.csv")
        else:
            query_set_with_contriver = pd.read_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_contriver.csv")
        
    
    input_list = []
    output_data_dict = {"task":"", "golds": []}
    output_data_dict["task"] = "LaMP_5"
    for i, eachrow in tqdm(enumerate(query_set_with_contriver.itertuples(index=False))):
        output_inner_dict = {"id":"", "output": ""}
        output_inner_dict["id"] = str(eachrow.docno)
        if task == "title_generation":
            output_inner_dict["output"] = preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "title"].values[0])
        else:
            output_inner_dict["output"] = preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "text"].values[0])
        output_data_dict["golds"].append(output_inner_dict)

        input_data_dict = {"id":"", "input":"", "profile": []}
        input_data_dict["id"] = str(eachrow.docno)
        if task == "title_generation":
            input_data_dict["input"] = "Output only the exact Wikipedia article title. Do not add any additional text, formatting, or explanation. Your response must be a single line consisting solely of the title. Article:" + preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "text"].values[0])
        else:
            input_data_dict["input"] = "Output only a Wikipedia article. Do not add any additional text, formatting, or explanation. Your response must be a single document related to the following title. Title:" + preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "title"].values[0])
            
        rel_docno_list = ast.literal_eval(eachrow.rel_docno)
        for doc_id in rel_docno_list:
            input_inner_dict = {"title":"", "abstract": "", "id": ""}
            input_inner_dict["title"] = preprocess(corpus[corpus['docno'] == int(doc_id)]["title"].to_list()[0])
            input_inner_dict["abstract"] = preprocess(corpus[corpus['docno'] == int(doc_id)]["text"].to_list()[0])
            input_inner_dict["id"] = str(doc_id)
            input_data_dict["profile"].append(input_inner_dict)
        input_list.append(input_data_dict)
    # Save to JSON file
    if dataset_name == "trec-2021":
        with open(f"./trec-2021/{task}/{query_topic}/contriver/5_user_dev_inputs.json", "w", encoding="utf-8") as f:
            json.dump(input_list, f, indent=2, ensure_ascii=False)
    
        # Save to JSON file
        with open(f"./trec-2021/{task}/{query_topic}/contriver/5_user_dev_outputs.json", "w", encoding="utf-8") as f:
            json.dump(output_data_dict, f, indent=2, ensure_ascii=False)
    elif dataset_name == "trec-2022":
        with open(f"./trec-2022/{task}/{query_topic}/contriver/5_user_dev_inputs.json", "w", encoding="utf-8") as f:
            json.dump(input_list, f, indent=2, ensure_ascii=False)
    
        # Save to JSON file
        with open(f"./trec-2022/{task}/{query_topic}/contriver/5_user_dev_outputs.json", "w", encoding="utf-8") as f:
            json.dump(output_data_dict, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query_topic",
        type=int,
        help="Query topic",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="TREC",
        help="Name of the dataset you want to run contriver",
    )

    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Enable genearting embeddings",
    )  
    
    parser.add_argument(
        "--task",
        type=str,
        default="TREC",
        help="Type of the task including title generation and article generation.",
    )
    args = parser.parse_args()

    main(args)


