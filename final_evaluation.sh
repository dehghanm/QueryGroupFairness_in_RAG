#!/bin/bash

datasets=("final_trec_2022")
llms=("gemma2-9b" "llama31-8b")
retrievers=("bm25" "contriver" "splade")
entailment_models=("roberta-large-mnli")
query_topics=("1102" "666" "1630")
task="title_generation"

for dataset in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    for retriever in "${retrievers[@]}"; do
      for entailment_model in "${entailment_models[@]}"; do
        for query_topic in "${query_topics[@]}"; do
          echo "Running with dataset=$dataset, llm=$llm, retriever=$retriever, entailment_model=$entailment_model"
          python final_evaluation.py \
            --dataset "$dataset" \
            --llm "$llm" \
            --entailment_model $entailment_model \
            --retriever "$retriever" \
            --query_topic "$query_topic" \
            --task "$task"
        done
      done
    done
  done
done