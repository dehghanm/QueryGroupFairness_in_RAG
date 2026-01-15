# Group-Query-Fairness

# Query Group Fairness in RAG

This repository contains the code and datasets for our experiments on **investigating the problem of Group Query Fairness in RAG (Retrieval-Augmented Generation) settings**. 

Group query fairness examines whether a RAG system is systematically more accurate for queries associated with particular groups within a fairness category, or whether the inclusion of the retriever component in RAG leads to greater accuracy improvements for certain query groups.

---

## Datasets

We use the **TREC 2022 Fair Ranking Track** test collection, which is well-suited for evaluating **group-level fairness in RAG systems**, to construct three datasets for our research questions. 

This repository includes:

- The datasets with queries and ground truth
- The corpus used for these experiments

You can find these files in the `final_trec_2022` directory.  

---

## Running Retriever Models

To run the retrievers used in our experiments (BM25, SPLADE, and Contriever), you can use the following scripts:

- `bm25.py`  
- `splade.py`  
- `contriever.py`  

Each script includes important arguments that can be configured before running.  

**Example usage:**

```bash
python bm25.py --arg1 value --arg2 value
