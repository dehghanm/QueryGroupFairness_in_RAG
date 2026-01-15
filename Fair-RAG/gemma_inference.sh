#!/bin/bash

nohup python -W ignore ./utility_labels/inference.py --model_name gemma2-9b --lamp_num 5 --k 1 --dataset trec-2022 --query_topic 10 --retriever contriver --task article_generation > g29b_k_t22_ag_7_c.log 2>&1 &

nohup python -W ignore ./utility_labels/inference.py --model_name gemma2-9b --lamp_num 5 --k 10 --dataset trec-2022 --query_topic 10 --retriever bm25 --task title_generation > g29b_rag_t22_tg_7_b.log 2>&1 &
