#!/bin/bash

datasets=("trec_title")
llms=("gemma2-9b" "llama31-8b")
retrievers=("splade")

for dataset in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    for retriever in "${retrievers[@]}"; do
      echo "Running with dataset=$dataset, llm=$llm, retriever=$retriever"
      python entailment_TRUE.py \
        --rag_model_name "$llm" \
        --entailment_model_name t5-xxl-nli\
        --dataset "$dataset" \
        --retriever "$retriever" \
        --lamp_num 5
    done
  done
done