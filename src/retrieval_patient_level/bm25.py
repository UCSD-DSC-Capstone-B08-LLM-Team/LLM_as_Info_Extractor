import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import numpy as np
import nltk
import os
import argparse
from tqdm import tqdm

nltk.download('punkt')

def retrieve_needles(haystack, needle, top_k=3, window_size=3):
    """
    Retrieve top_k passages from haystack that are most relevant to needle using BM25.

    Args:
        haystack (str): The text to search within.
        needle (str): The text to search for.
        top_k (int): Number of top passages to retrieve.
        window_size (int): Number of sentences per passage.

    Returns:
        top_passages: List of strings of top_k passages most relevant to the needle.
        num_passages: Total number of passages created from the haystack.
    """

    sentences = sent_tokenize(str(haystack))

    # Handle short or empty notes
    if len(sentences) == 0:
        return [], 0

    # Create passages
    if window_size <= 0:
        passages = [' '.join(sentences)]
    elif len(sentences) < window_size:
        passages = [' '.join(sentences)]
    else:
        passages = [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]

    num_passages = len(passages)

    # BM25 retrieval
    tokenized_passages = [word_tokenize(p.lower()) for p in passages]
    if len(tokenized_passages) == 0:
        return [], num_passages

    bm25 = BM25Okapi(tokenized_passages)
    tokenized_needle = word_tokenize(str(needle).lower())
    scores = bm25.get_scores(tokenized_needle)

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_passages = [passages[i] for i in top_indices]

    return top_passages, num_passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-level BM25 retrieval")
    parser.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    parser.add_argument("--output_csv", type=str, default="src/retrieval_patient_level/outputs/bm25_patient_results.csv")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.haystack_csv)
    results = []
    found_count = 0

    # Use tqdm to track progress
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BM25 patient-level retrieval"):
        haystack = row['PATIENT_RECORD']
        needle = row['NEEDLE_INSERTED']

        top_passages, num_passages = retrieve_needles(
            haystack, needle, top_k=args.top_k, window_size=args.window_size
        )

        # Determine rank of needle if found
        needle_rank = np.nan
        for i, passage in enumerate(top_passages):
            if needle in passage:
                needle_rank = i + 1
                break

        # Check if needle retrieved within top-k
        found = not np.isnan(needle_rank)
        if found:
            found_count += 1

        results.append({
            "SUBJECT_ID": row["SUBJECT_ID"],
            "NUM_NOTES": row["NUM_NOTES"],
            "needle": needle,
            "needle_rank": needle_rank,
            "found": found,
            "num_passages": num_passages,
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages
        })

        tqdm.write(f"Progress: {_+1}/{len(df)}, Current Needle Rank: {needle_rank}, Current Recall@{args.top_k}: {found_count/(_+1):.2f}")

    results_df = pd.DataFrame(results)

    # Mean rank (only where needle was found)
    mean_rank = results_df["needle_rank"].dropna().mean()
    # % at rank 1
    pct_rank_1 = (results_df["needle_rank"] == 1).mean()
    # % at rank <= K
    K = args.top_k
    pct_rank_k = (results_df["needle_rank"] <= K).mean()
    print("\n===== Retrieval Rank Metrics =====")
    print(f"Mean needle rank: {mean_rank:.2f}")
    print(f"% needles at rank 1: {pct_rank_1:.2%}")
    print(f"% needles at rank ≤ {K}: {pct_rank_k:.2%}")
    # Overall accuracy / recall@k
    recall_at_k = results_df["found"].mean()
    print(f"Recall@{args.top_k}: {recall_at_k:.4f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved BM25 results to {args.output_csv}")