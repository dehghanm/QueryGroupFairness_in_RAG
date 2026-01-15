#!/bin/bash

datasets=("trec-2022")
llms=("llama31-8b")
retrievers=("bm25" "contriver" "splade")
query_topics=("10")
task="article_generation"
for dataset in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    for query_topic in "${query_topics[@]}"; do
      for retriever in "${retrievers[@]}"; do
        echo "Running with dataset=$dataset, llm=$llm, retriever=$retriever"
        python entailment_Roberta.py \
          --rag_model_name "$llm" \
          --entailment_model_name roberta-large-mnli \
          --dataset "$dataset" \
          --retriever "$retriever" \
          --lamp_num 5 \
          --query_topic "$query_topic" \
          --task "$task"
      done
    done
  done
done