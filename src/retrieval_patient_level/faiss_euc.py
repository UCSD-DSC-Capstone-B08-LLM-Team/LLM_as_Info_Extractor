import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from nltk.tokenize import sent_tokenize
import nltk
import os
from tqdm import tqdm
import argparse
import numpy as np

nltk.download('punkt')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

model = SentenceTransformer('all-MiniLM-L6-v2')


def retrieve_needle_faiss_euc(haystack, needle, top_k=5, window_size=3):
    """
    Retrieve top_k passages from haystack that are most similar to needle using FAISS Euclidean distance.
    """
    sentences = sent_tokenize(haystack)

    # Handle short or empty notes
    if len(sentences) == 0:
        return []

    # Create passages
    if window_size <= 0 or len(sentences) < window_size:
        passages = [' '.join(sentences)]
    else:
        passages = [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]

    # FAISS retrieval
    passage_embeddings = model.encode(passages, convert_to_numpy=True)
    needle_embedding = model.encode([needle], convert_to_numpy=True)

    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(passage_embeddings)

    current_k = min(top_k, len(passages))
    distances, indices = index.search(needle_embedding, current_k)
    top_passages = [passages[i] for i in indices[0]]

    return top_passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-level FAISS Euclidean retrieval")
    parser.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    parser.add_argument("--output_csv", type=str, default="src/retrieval_patient_level/outputs/faiss_euc_patient_results.csv")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.haystack_csv)
    all_results = []
    found_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="FAISS Euclidean Retrieval"):
        haystack = row['PATIENT_RECORD']
        needle = row['NEEDLE_INSERTED']

        top_passages = retrieve_needle_faiss_euc(haystack, needle, top_k=args.top_k, window_size=args.window_size)

        if args.window_size <= 0:
            num_passages = 1
        else:
            num_passages = max(len(sent_tokenize(haystack)) - args.window_size + 1, 1)

        # Determine needle rank
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
            "needle": needle,
            "needle_rank": needle_rank,
            "found": found,
            "num_passages": num_passages,
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages
        })

        tqdm.write(f"Progress: {_+1}/{len(df)}, Current Needle Rank: {needle_rank}, Current Recall@{args.top_k}: {found_count/(_+1):.2f}")

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
    print(f"\nSaved FAISS Euclidean results to {args.output_csv}")