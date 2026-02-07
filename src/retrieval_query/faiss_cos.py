from tqdm import tqdm
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from nltk.tokenize import sent_tokenize
import nltk
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

nltk.download("punkt")

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_faiss_cos(haystack, query, top_k=5, window_size=3):
    """
    Retrieve top_k passages from haystack using FAISS cosine similarity.
    Search is driven by QUERY, not the needle.
    """

    sentences = sent_tokenize(str(haystack))
    if len(sentences) == 0:
        return [], 0

    # Create passages
    if window_size <= 0 or len(sentences) < window_size:
        passages = [" ".join(sentences)]
    else:
        passages = [
            " ".join(sentences[i : i + window_size])
            for i in range(len(sentences) - window_size + 1)
        ]

    num_passages = len(passages)

    # Encode passages + query
    passage_embeddings = model.encode(passages, convert_to_numpy=True, show_progress_bar=False)
    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)

    # Normalize for cosine similarity
    faiss.normalize_L2(passage_embeddings)
    faiss.normalize_L2(query_embedding)

    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)

    current_k = min(top_k, num_passages)
    scores, indices = index.search(query_embedding, current_k)

    top_passages = [passages[i] for i in indices[0]]

    return top_passages, num_passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-level FAISS cosine retrieval (query-based)")
    parser.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    parser.add_argument("--output_csv", type=str, default="src/retrieval_query/outputs/faiss_cos_patient_results.csv")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.haystack_csv)
    results = []
    found_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="FAISS cosine retrieval"):
        haystack = row["PATIENT_RECORD"]
        query = row["QUERY"]
        needle = row["NEEDLE_INSERTED"]

        top_passages, num_passages = retrieve_faiss_cos(
            haystack,
            query,
            top_k=args.top_k,
            window_size=args.window_size,
        )

        # Needle rank
        needle_rank = np.nan
        for j, passage in enumerate(top_passages):
            if needle in passage:
                needle_rank = j + 1
                break

        found = not np.isnan(needle_rank)
        if found:
            found_count += 1

        results.append(
            {
                "SUBJECT_ID": row["SUBJECT_ID"],
                "NUM_NOTES": row["NUM_NOTES"],
                "query": query,
                "needle": needle,
                "needle_rank": needle_rank,
                "found": found,
                "num_passages": num_passages,
                "haystack_len_chars": len(haystack),
                "top_passages": top_passages,
            }
        )

        tqdm.write(
            f"Progress: {i+1}/{len(df)}, "
            f"Current Needle Rank: {needle_rank}, "
            f"Recall@{args.top_k}: {found_count/(i+1):.2f}"
        )

    results_df = pd.DataFrame(results)

    print("\n===== Retrieval Rank Metrics =====")
    print(f"Mean needle rank: {results_df['needle_rank'].dropna().mean():.2f}")
    print(f"% needles at rank 1: {(results_df['needle_rank'] == 1).mean():.2%}")
    print(f"% needles at rank ≤ {args.top_k}: {(results_df['needle_rank'] <= args.top_k).mean():.2%}")
    print(f"Recall@{args.top_k}: {results_df['found'].mean():.4f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved FAISS cosine results to {args.output_csv}")