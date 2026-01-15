#!/bin/bash

datasets=("trec-2022")
retrievers=("contriver")
llms=("llama31-8b")
query_topics=("10")
task="article_generation"

for dataset in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    for query_topic in "${query_topics[@]}"; do
      for retriever in "${retrievers[@]}"; do
        echo "Running with dataset=$dataset, query_topic=$query_topic, llm=$llm, retriever=$retriever"
        python lamp_eval.py \
          --dataset "$dataset" \
          --model_name "$llm" \
          --lamp_num 5 \
          --retriever "$retriever" \
          --query_topic "$query_topic" \
          --task "$task"
      done
    done
  done
done
