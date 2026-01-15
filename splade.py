from sparsembed import model, retrieve
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import re
import pandas as pd
import json
import os
import argparse
import ast

def preprocess(doc):
    cleaned_text = str(doc).replace("\n", " ").replace("\r", " ")
    clean_text = re.sub(r'\s+', ' ', cleaned_text)
    return clean_text

def retrieve_top_k_with_splade(retriever, query, k, batch_size=16):
    output = retriever(
        [query],  # can pass multiple queries but passing a single query here
        # k_tokens=20,
        k=k,  # Number of documents to retrieve.
        batch_size=batch_size,
    )
    return [({"id": x["id"]}, x["similarity"]) for x in output[0]]

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

    if dataset_name == "trec-2021":
        os.makedirs(f"./trec-2021/{task}/{query_topic}/splade", exist_ok=True)
    elif dataset_name == "trec-2022":
        os.makedirs(f"./trec-2022/{task}/{query_topic}/splade", exist_ok=True)
    
    if dataset_name == "trec-2021":
        corpus_path = "./trec-2021/corpus.csv"
        corpus = pd.read_csv(corpus_path)
    elif dataset_name == "trec-2022":
        corpus_path = "./trec-2022/corpus.csv"
        corpus = pd.read_csv(corpus_path)
        
    corpus = corpus.rename(columns={'text': 'body'})
    corpus["processed_body"] = corpus["body"].apply(preprocess)
    corpus["processed_title"] = corpus["title"].apply(preprocess)
    corpus["text"] = corpus["processed_title"] + "\n" + corpus["processed_body"]
    splade_corpus = [
            {"id": eachrow.docno, "text": eachrow.text} for eachrow in corpus.itertuples(index=False)
    ]
    
    splade_checkpoint = "naver/splade_v2_max"
    splade_model = model.Splade(model=AutoModelForMaskedLM.from_pretrained(splade_checkpoint).to("cuda:0"),
                               tokenizer=AutoTokenizer.from_pretrained(splade_checkpoint),device="cuda:0",)
    batch_size = 350
    retriever = retrieve.SpladeRetriever(
        key="id",  # Key identifier of each document
        on=["text"],  # Fields to search.
        model=splade_model,
        )
    retriever = retriever.add(documents=splade_corpus, batch_size=batch_size) # k_tokens=256 # Number of activated tokens.    

    splade_result = []
    if dataset_name == "trec-2021":
        query_df = pd.read_csv(f"./trec-2021/query_ids_{query_topic}.csv")
    elif dataset_name == "trec-2022":
        query_df = pd.read_csv(f"./trec-2022/query_ids_{query_topic}.csv")
        
    batch_size = 350
    for eachrow in tqdm(query_df.itertuples(index=False)):
        query_docno = eachrow.docno
        if task == "title_generation":
            query = corpus.loc[corpus["docno"] == query_docno, "body"].values[0]
        else:
            query = corpus.loc[corpus["docno"] == query_docno, "title"].values[0]
            
        preprocessed_query = preprocess(query)
        selected_profs_with_scores = retrieve_top_k_with_splade(retriever, query, 200, batch_size)
        related_doc_ids = set()
        for i in range(200):
            if str(query_docno) == str(selected_profs_with_scores[i][0]["id"]):
                continue
            if str(selected_profs_with_scores[i][0]["id"]) in related_doc_ids:
                continue

            related_doc_ids.add(selected_profs_with_scores[i][0]["id"])
            if len(related_doc_ids) == 10:
                break
        if len(related_doc_ids) < 10:
            print(f"something is wrong with this query: {query_docno}")
        splade_result.append(related_doc_ids)
    for query_index, res in enumerate(splade_result):
        if len(set(res)) < 10:
            print(f"found a mistake in query: {query_index}")

    if dataset_name == "trec-2021":
        query_df["rel_docno"] = splade_result
        query_df.to_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_splade.csv", index=False)
    elif dataset_name == "trec-2022":
        query_df["rel_docno"] = splade_result
        query_df.to_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_splade.csv", index=False)

    if dataset_name == "trec-2021":
        query_set_with_spalde = pd.read_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_splade.csv")
    elif dataset_name == "trec-2022":
        query_set_with_spalde = pd.read_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_splade.csv")

    
    input_list = []
    output_data_dict = {"task":"", "golds": []}
    output_data_dict["task"] = "LaMP_5"

    for i, eachrow in tqdm(enumerate(query_set_with_spalde.itertuples(index=False))):
        output_inner_dict = {"id":"", "output": ""}
        output_inner_dict["id"] = str(eachrow.docno)
        if task == "title_generation":
            output_inner_dict["output"] = preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "title"].values[0])
        else:
            output_inner_dict["output"] = preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "body"].values[0])
        output_data_dict["golds"].append(output_inner_dict)

        input_data_dict = {"id":"", "input":"", "profile": []}
        input_data_dict["id"] = str(eachrow.docno)
        if task == "title_generation":
            input_data_dict["input"] = "Output only the exact Wikipedia article title. Do not add any additional text, formatting, or explanation. Your response must be a single line consisting solely of the title. Article:" + preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "body"].values[0])
        else:
            input_data_dict["input"] = "Output only a Wikipedia article. Do not add any additional text, formatting, or explanation. Your response must be a single article related to the following title. Title:" + preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "title"].values[0])
            
        rel_docno_list = ast.literal_eval(eachrow.rel_docno)
        for doc_id in rel_docno_list:
            input_inner_dict = {"title":"", "abstract": "", "id": ""}
            input_inner_dict["title"] = preprocess(corpus[corpus['docno'] == int(doc_id)]["title"].to_list()[0])
            input_inner_dict["abstract"] = preprocess(corpus[corpus['docno'] == int(doc_id)]["body"].to_list()[0])
            input_inner_dict["id"] = str(doc_id)
            input_data_dict["profile"].append(input_inner_dict)
        input_list.append(input_data_dict)

    if dataset_name == "trec-2021":
        # Save to JSON file
        with open(f"./trec-2021/{task}/{query_topic}/splade/5_user_dev_inputs.json", "w", encoding="utf-8") as f:
            json.dump(input_list, f, indent=2, ensure_ascii=False)
    
        # Save to JSON file
        with open(f"./trec-2021/{task}/{query_topic}/splade/5_user_dev_outputs.json", "w", encoding="utf-8") as f:
            json.dump(output_data_dict, f, indent=2, ensure_ascii=False)
    elif dataset_name == "trec-2022":
        # Save to JSON file
        with open(f"./trec-2022/{task}/{query_topic}/splade/5_user_dev_inputs.json", "w", encoding="utf-8") as f:
            json.dump(input_list, f, indent=2, ensure_ascii=False)
    
        # Save to JSON file
        with open(f"./trec-2022/{task}/{query_topic}/splade/5_user_dev_outputs.json", "w", encoding="utf-8") as f:
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
        "--task",
        type=str,
        default="TREC",
        help="Type of the task including title generation and article generation.",
    )
    
    args = parser.parse_args()

    main(args)


