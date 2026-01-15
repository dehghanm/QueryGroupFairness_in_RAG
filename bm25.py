# Adapted from https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py
#!/usr/bin/env python

import math
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re
import pandas as pd
import ast
import argparse
import json

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned.
"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

import pyterrier as pt
if not pt.started():
    pt.init()


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

    # return with scores
    def get_top_n_with_scores(self, query, documents, n=5):

        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [(documents[i], scores[i]) for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score.tolist()

stopwords = set()
with open("./stopword-list.txt", "r", encoding="utf-8") as f:
    for line in f:
        stopwords.add(line.strip())

stemmer = pt.TerrierStemmer.porter


def preprocess(doc):
    cleaned_text = str(doc).replace("\n", " ").replace("\r", " ")
    clean_text = re.sub(r'\s+', ' ', cleaned_text)
    clean_text = clean_text.split()
    temp_clean_text = []
    for word in clean_text:
        if word not in stopwords:
            temp_clean_text.append(stemmer.stem(word))

    clean_text = " ".join(temp_clean_text)
    return clean_text

def simple_preprocess(text):
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text


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

    documents = corpus[["docno", "document"]].to_dict(orient='records')

    tokenized_corpus = [x.split() for x in document_list]
    bm25 = BM25Okapi(tokenized_corpus)


    if dataset_name == "trec-2021":    
        os.makedirs(f"./trec-2021/{task}/{query_topic}/bm25", exist_ok=True)
    elif dataset_name == "trec-2022":
        os.makedirs(f"./trec-2022/{task}/{query_topic}/bm25", exist_ok=True)
        
    bm25_result = []
    if dataset_name == "trec-2021":
        query_df = pd.read_csv(f"./trec-2021/query_ids_{query_topic}.csv")
    elif dataset_name == "trec-2022":
        query_df = pd.read_csv(f"./trec-2022/query_ids_{query_topic}.csv")
    
    for eachrow in tqdm(query_df.itertuples(index=False)):
        query_id = eachrow.docno
        if task == "title_generation":
            query = preprocess(corpus.loc[corpus["docno"] == query_id, "text"].values[0])
            tokenized_query = query.split()
            selected_profs_with_scores = bm25.get_top_n_with_scores(tokenized_query, documents, n=50)
        else:
            query = preprocess(corpus.loc[corpus["docno"] == query_id, "title"].values[0])
            tokenized_query = query.split()
            selected_profs_with_scores = bm25.get_top_n_with_scores(tokenized_query, documents, n=50)
        
        related_doc_ids = set()
        for i in range(0, 50):
            if str(query_id) == str(selected_profs_with_scores[i][0]["docno"]):
                continue
            if str(selected_profs_with_scores[i][0]["docno"]) in related_doc_ids:
                continue
            related_doc_ids.add(selected_profs_with_scores[i][0]["docno"])
            if len(related_doc_ids) == 10:
                break
        if len(related_doc_ids) < 10:
            print(f"something is wrong with this query: {query_id}")
            
        bm25_result.append(related_doc_ids)


    for query_idx, res in enumerate(bm25_result):
        if len(set(res)) < 10:
            print(f"found a mistake in query number: {query_idx}")


    if dataset_name == "trec-2021":
        if task == "title_generation":
            query_df['rel_docno'] = bm25_result
            query_df.to_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_bm25.csv", index=False)
        else:
            query_df['rel_docno'] = bm25_result
            query_df.to_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_bm25.csv", index=False)
    elif dataset_name == "trec-2022":
        if task == "title_generation":
            query_df['rel_docno'] = bm25_result
            query_df.to_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_bm25.csv", index=False)
        else:
            query_df['rel_docno'] = bm25_result
            query_df.to_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_bm25.csv", index=False)

    if dataset_name == "trec-2021":
        if task == "title_generation":
            query_set_with_bm25 = pd.read_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_bm25.csv")
        else:
            query_set_with_bm25 = pd.read_csv(f"./trec-2021/{task}/query_ids_{query_topic}_with_bm25.csv")
    elif dataset_name == "trec-2022":
        if task == "title_generation":
            query_set_with_bm25 = pd.read_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_bm25.csv")
        else:
            query_set_with_bm25 = pd.read_csv(f"./trec-2022/{task}/query_ids_{query_topic}_with_bm25.csv")

    input_list = []
    output_data_dict = {"task":"", "golds": []}
    output_data_dict["task"] = "LaMP_5"
    if dataset_name == "trec-2021":
        corpus_path = "./trec-2021/corpus.csv"
        corpus = pd.read_csv(corpus_path)
    elif dataset_name == "trec-2022":
        corpus_path = "./trec-2022/corpus.csv"
        corpus = pd.read_csv(corpus_path)
        
    for i, eachrow in tqdm(enumerate(query_set_with_bm25.itertuples(index=False))):
        output_inner_dict = {"id":"", "output": ""}
        output_inner_dict["id"] = str(eachrow.docno)
        if task == "title_generation":
            output_inner_dict["output"] = simple_preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "title"].values[0])
        else:
            output_inner_dict["output"] = simple_preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "text"].values[0])
        output_data_dict["golds"].append(output_inner_dict)

        input_data_dict = {"id":"", "input":"", "profile": []}
        input_data_dict["id"] = str(eachrow.docno)
        if task == "title_generation":
            input_data_dict["input"] = "Output only the exact Wikipedia article title. Do not add any additional text, formatting, or explanation. Your response must be a single line consisting solely of the title. Article:" + simple_preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "text"].values[0])
        else:
            input_data_dict["input"] = "Output only a Wikipedia article. Do not add any additional text, formatting, or explanation. Your response must be a single document related to the following title. Title:" + simple_preprocess(corpus.loc[corpus["docno"] == eachrow.docno, "title"].values[0])
        rel_docno_list = ast.literal_eval(eachrow.rel_docno)
        for doc_id in rel_docno_list:
            input_inner_dict = {"title":"", "abstract": "", "id": ""}
            input_inner_dict["title"] = simple_preprocess(corpus[corpus['docno'] == int(doc_id)]["title"].to_list()[0])
            input_inner_dict["abstract"] = simple_preprocess(corpus[corpus['docno'] == int(doc_id)]["text"].to_list()[0])
            input_inner_dict["id"] = str(doc_id)
            input_data_dict["profile"].append(input_inner_dict)
        input_list.append(input_data_dict)
    if dataset_name == "trec-2021":
        # Save to JSON file
        with open(f"./trec-2021/{task}/{query_topic}/bm25/5_user_dev_inputs.json", "w", encoding="utf-8") as f:
            json.dump(input_list, f, indent=2, ensure_ascii=False)
    
        # Save to JSON file
        with open(f"./trec-2021/{task}/{query_topic}/bm25/5_user_dev_outputs.json", "w", encoding="utf-8") as f:
            json.dump(output_data_dict, f, indent=2, ensure_ascii=False)
    elif dataset_name == "trec-2022":
        # Save to JSON file
        with open(f"./trec-2022/{task}/{query_topic}/bm25/5_user_dev_inputs.json", "w", encoding="utf-8") as f:
            json.dump(input_list, f, indent=2, ensure_ascii=False)
    
        # Save to JSON file
        with open(f"./trec-2022/{task}/{query_topic}/bm25/5_user_dev_outputs.json", "w", encoding="utf-8") as f:
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
        help="Name of the dataset you want to run BM25",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="TREC",
        help="Type of the task including title generation and article generation.",
    )
    
    args = parser.parse_args()

    main(args)

