from tqdm import tqdm
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from nltk.tokenize import sent_tokenize
import nltk
import os
import argparse

# Thread + tokenizer control for determinism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

nltk.download("punkt")

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def mmr_rerank(
    query_embedding,
    passage_embeddings,
    passages,
    lambda_param=0.7,
    top_k=5,
):
    """
    Apply Maximal Marginal Relevance (MMR) re-ranking.
    Assumes cosine similarity (embeddings already normalized).
    """

    selected_indices = []
    candidate_indices = list(range(len(passages)))

    # Precompute similarities
    query_sims = np.dot(passage_embeddings, query_embedding.T).squeeze()
    passage_sims = np.dot(passage_embeddings, passage_embeddings.T)

    for _ in range(min(top_k, len(passages))):
        mmr_scores = []

        for idx in candidate_indices:
            relevance = query_sims[idx]

            if not selected_indices:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(passage_sims[idx][selected_indices])

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * diversity_penalty
            )
            mmr_scores.append((mmr_score, idx))

        _, selected_idx = max(mmr_scores, key=lambda x: x[0])
        selected_indices.append(selected_idx)
        candidate_indices.remove(selected_idx)

    return [passages[i] for i in selected_indices]


def retrieve_faiss_mmr(
    haystack,
    query,
    top_k=5,
    window_size=3,
    mmr_lambda=0.7,
    mmr_candidates=20,
):
    """
    Retrieve top_k passages from haystack using FAISS cosine similarity
    followed by MMR re-ranking. Search is driven by QUERY.
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

    # Encode passages and query
    passage_embeddings = model.encode(
        passages, convert_to_numpy=True, show_progress_bar=False
    )
    query_embedding = model.encode(
        [query], convert_to_numpy=True, show_progress_bar=False
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(passage_embeddings)
    faiss.normalize_L2(query_embedding)

    # Build FAISS index
    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)

    # FAISS candidate retrieval
    candidate_k = min(mmr_candidates, num_passages)
    _, indices = index.search(query_embedding, candidate_k)

    candidate_passages = [passages[i] for i in indices[0]]
    candidate_embeddings = passage_embeddings[indices[0]]

    # MMR re-ranking
    top_passages = mmr_rerank(
        query_embedding=query_embedding,
        passage_embeddings=candidate_embeddings,
        passages=candidate_passages,
        lambda_param=mmr_lambda,
        top_k=top_k,
    )

    return top_passages, num_passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-level FAISS + MMR retrieval (query-based)")
    parser.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    parser.add_argument("--output_csv", type=str, default="src/retrieval_query/outputs/faiss_mmr_patient_results.csv")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--mmr_lambda", type=float, default=0.7)
    parser.add_argument("--mmr_candidates", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.haystack_csv)
    results = []
    found_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="FAISS with MMR retrieval"):
        haystack = row["PATIENT_RECORD"]
        query = row["QUERY"]
        needle = row["NEEDLE_INSERTED"]

        top_passages, num_passages = retrieve_faiss_mmr(
            haystack=haystack,
            query=query,
            top_k=args.top_k,
            window_size=args.window_size,
            mmr_lambda=args.mmr_lambda,
            mmr_candidates=args.mmr_candidates,
        )

        # Compute needle rank
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
            f"Recall@{args.top_k}: {found_count / (i + 1):.2f}"
        )

    results_df = pd.DataFrame(results)

    print("\n===== Retrieval Rank Metrics =====")
    print(f"Mean needle rank: {results_df['needle_rank'].dropna().mean():.2f}")
    print(f"% needles at rank 1: {(results_df['needle_rank'] == 1).mean():.2%}")
    print(f"% needles at rank ≤ {args.top_k}: {(results_df['needle_rank'] <= args.top_k).mean():.2%}")
    print(f"Recall@{args.top_k}: {results_df['found'].mean():.4f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved FAISS with MMR results to {args.output_csv}")