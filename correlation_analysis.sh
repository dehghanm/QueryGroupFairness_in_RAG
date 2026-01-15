#!/bin/bash

datasets=("X" "news")
llms=("gemma2-9b")
retrievers=("bm25" "contriver")
entailment_models=("roberta-large-mnli" "t5-xxl-nli")

for dataset in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    for retriever in "${retrievers[@]}"; do
      for entailment_model in "${entailment_models[@]}"; do
        echo "Running with dataset=$dataset, llm=$llm, retriever=$retriever, entailment_model=$entailment_model"
        python correlation_analysis.py \
          --dataset "$dataset" \
          --llm "$llm" \
          --entailment_model $entailment_model \
          --retriever "$retriever"
      done
    done
  done
done
