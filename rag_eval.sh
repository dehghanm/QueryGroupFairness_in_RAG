#!/bin/bash

datasets=("final_trec_2022")
llms=("gemma2-9b" "llama31-8b")
retrievers=("bm25" "contriver" "splade")
query_topics=("666" "1102" "1630")
task="article_generation"

for dataset in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    for retriever in "${retrievers[@]}"; do
      for query_topic in "${query_topics[@]}"; do
        echo "Running with dataset=$dataset, llm=$llm, retriever=$retriever"
        python rag_eval.py \
          --dataset "$dataset" \
          --model_name "$llm" \
          --retriever "$retriever" \
          --lamp_num 5 \
          --query_topic "$query_topic" \
          --task "$task"
      done
    done
  done
done
