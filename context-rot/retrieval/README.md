# Retrieval Module

This folder contains all retrieval baselines used for evaluating needle-in-a-haystack (NIAH) benchmarks.

Each retrieval method has:
- A Python script (`.py`) that runs retrieval
- A CSV file with the retrieved passages and scores
- A consistent interface so results can be compared across methods

The goal of this module is to measure how well different retrieval methods can extract the “needle” (answer span) from a long haystack before sending it to an LLM for RAG evaluation.

## Motivation
Large Language Models perform poorly at searching through long contexts. Therefore, before sending text into an LLM for RAG generation, we evaluate: 
- Which retrieval method finds the gold answer more reliably?
- How much text should be passed to the LLM?
- How retrieval quality affects final RAG accuracy?

This provides full reproducibility and debugging visibility.

## Retrieval Methods Implemented

1. BM25
Lexical keyword-based retrieval using `rank-bm25`.

2. FAISS (Cosine Similarity)
Dense embedding retrieval using sentence-transformers + FAISS index (cosine metric).

3. FAISS (Euclidean Distance)
Same as above but using L2 distance.

4. Hybrid Retrieval
Weighted combination of BM25 lexical similarity and FAISS dense similarity. Uses an `alpha` parameter (default 0.5) to combine scores.

## Input Data
All retrieval scripts load needle-in-a-haystack format:
- **prompt**: the full haystack
- **question**: what we want the model to answer
- **answer**: the needle

## Output Data
After running the retrieval the information is stored in csv files containing:
- **csv_file**: The source CSV
- **needle_type**: The needle type
- **needle**: The original needle
- **question**: The original question
- **found**: Whether the needle was found
- **top_passages**: Top-k retrieved passaged (sorted from highest score to lowest)

## How to Reproduce the Results
BM25 example: 

```bash
python bm25/bm25_needle_retrieval.py 
```
Each method can be run similarly by supplying the `.py` files which contains the input NIAH CSV and the output path.
