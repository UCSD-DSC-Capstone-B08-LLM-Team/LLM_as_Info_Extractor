import os
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
import nltk
import argparse
import numpy as np

nltk.download('punkt')

# Thread control for deterministic behavior
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

model = SentenceTransformer('all-MiniLM-L6-v2')


def build_indices(haystack, window_size=3):
    """
    Build BM25 and FAISS indices for all passages in a haystack.
    Returns passages, bm25_index, faiss_index, passage_embeddings, num_passages
    """
    sentences = sent_tokenize(str(haystack))

    if len(sentences) == 0:
        return [], None, None, None, 0

    # Create overlapping passages
    if window_size <= 0 or len(sentences) < window_size:
        passages = [' '.join(sentences)]
    else:
        passages = [' '.join(sentences[i:i + window_size]) for i in range(len(sentences) - window_size + 1)]

    num_passages = len(passages)

    # BM25 index
    tokenized_passages = [word_tokenize(p.lower()) for p in passages]
    bm25_index = BM25Okapi(tokenized_passages) if tokenized_passages else None

    # FAISS index
    passage_embeddings = model.encode(passages, convert_to_numpy=True)
    faiss.normalize_L2(passage_embeddings)
    dim = passage_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(passage_embeddings)

    return passages, bm25_index, faiss_index, passage_embeddings, num_passages


def hybrid_retrieve(query, passages, bm25_index, faiss_index, passage_embeddings,
                    top_k=5, bm25_weight=0.5, faiss_weight=0.5):
    """
    Hybrid BM25 + FAISS retrieval using query (not needle) to score passages.
    Returns top_k passages ranked by combined normalized score.
    """
    if not passages:
        return []

    n = len(passages)

    # BM25 scores
    tokenized_query = word_tokenize(str(query).lower())
    bm25_scores = bm25_index.get_scores(tokenized_query) if bm25_index else np.zeros(n)

    # Normalize BM25 scores to [0,1]
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()  

    # FAISS scores
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    faiss_scores_all = np.zeros(n)
    scores, indices = faiss_index.search(q_emb, n)
    for score, idx in zip(scores[0], indices[0]):
        faiss_scores_all[idx] = score

    # Normalize FAISS scores to [0,1]   
    if faiss_scores_all.max() > 0:
        faiss_scores_all = faiss_scores_all / faiss_scores_all.max() 

    # Combined weighted score
    combined_scores = bm25_weight * bm25_scores + faiss_weight * faiss_scores_all
    ranked_indices = np.argsort(combined_scores)[::-1][:top_k]

    return [passages[i] for i in ranked_indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-level hybrid BM25 + FAISS retrieval (query-driven)")
    parser.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    parser.add_argument("--output_csv", type=str, default="src/retrieval_query/outputs/hybrid_patient_results.csv")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--bm25_weight", type=float, default=0.5)
    parser.add_argument("--faiss_weight", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.haystack_csv)
    all_results = []
    found_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Hybrid Patient Retrieval"):
        haystack = row['PATIENT_RECORD']
        needle = row['NEEDLE_INSERTED']
        query = row['QUERY']  

        passages, bm25_idx, faiss_idx, passage_embeddings, num_passages = build_indices(haystack, args.window_size)
        top_passages = hybrid_retrieve(
            query, passages, bm25_idx, faiss_idx, passage_embeddings,
            top_k=args.top_k, bm25_weight=args.bm25_weight, faiss_weight=args.faiss_weight
        )

        # Compute needle rank
        needle_rank = np.nan
        for i, passage in enumerate(top_passages):
            if needle in passage:
                needle_rank = i + 1
                break

        found = not np.isnan(needle_rank)
        if found:
            found_count += 1

        all_results.append({
            "SUBJECT_ID": row["SUBJECT_ID"],
            "NUM_NOTES": row["NUM_NOTES"],
            "query": query,
            "NEEDLE_INSERTED": needle,
            "needle_rank": needle_rank,
            "found": found,
            "num_passages": num_passages,
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages
        })

        tqdm.write(f"Progress: {idx+1}/{len(df)}, Current Needle Rank: {needle_rank}, Current Recall@{args.top_k}: {found_count/(idx+1):.2f}")

    results_df = pd.DataFrame(all_results)

    # Rank metrics
    mean_rank = results_df["needle_rank"].dropna().mean()
    pct_rank_1 = (results_df["needle_rank"] == 1).mean()
    pct_rank_k = (results_df["needle_rank"] <= args.top_k).mean()
    recall_at_k = results_df["found"].mean()

    print("\n===== Retrieval Rank Metrics =====")
    print(f"Mean needle rank: {mean_rank:.2f}")
    print(f"% needles at rank 1: {pct_rank_1:.2%}")
    print(f"% needles at rank ≤ {args.top_k}: {pct_rank_k:.2%}")
    print(f"Final Recall@{args.top_k}: {recall_at_k:.4f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved Hybrid BM25+FAISS results to {args.output_csv}")