#!/bin/bash

datasets=("trec_title")
llms=("gemma2-9b" "llama31-8b")
retrievers=("bm25" "contriver")
entailment_models=("roberta-large-mnli" "t5-xxl-nli")

for dataset in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    for retriever in "${retrievers[@]}"; do
      for entailment_model in "${entailment_models[@]}"; do
        echo "Running with dataset=$dataset, llm=$llm, retriever=$retriever, entailment_model=$entailment_model"
        python evaluation.py \
          --dataset "$dataset" \
          --llm "$llm" \
          --entailment_model $entailment_model \
          --retriever "$retriever"
      done
    done
  done
done
