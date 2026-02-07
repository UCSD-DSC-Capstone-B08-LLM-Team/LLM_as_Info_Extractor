import pandas as pd
import numpy as np
from ragatouille import RAGPretrainedModel
from tqdm import tqdm
import argparse
import torch

def run_colbert_retrieval(input_csv, output_csv, k=5, chunk_size=256):
    """
    Runs ColBERT retrieval on a 'needle in a haystack' dataset.
    """

    print("Loading ColBERT model...")
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    df = pd.read_csv(input_csv)
    results = []
    found_count = 0 

    print(f"Processing {len(df)} haystack records...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        haystack = str(row["PATIENT_RECORD"])
        query = str(row["QUERY"])
        needle = str(row["NEEDLE_INSERTED"])
        subject_id = row["SUBJECT_ID"]
        num_notes = row["NUM_NOTES"]

        # Chunk haystack with 10% overlap
        words = haystack.split()
        overlap = int(chunk_size * 0.1)
        step = chunk_size - overlap

        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]

        # Deduplicate chunks while preserving order
        unique_chunks = list(dict.fromkeys(chunks))
        if len(unique_chunks) < len(chunks):
            print(f"WARNING: {len(chunks) - len(unique_chunks)} duplicate chunk(s) removed for SUBJECT_ID {subject_id}")
        chunks = unique_chunks

        num_chunks = len(chunks)
        current_k = min(k, num_chunks)

        # Retrieve top passages
        try:
            if num_chunks == 0:
                top_passages = []
            else:
                retrieval_results = RAG.rerank(
                    query=query,
                    documents=chunks,
                    k=current_k
                )

                if retrieval_results is None:
                    top_passages = []
                else:
                    top_passages = [res["content"] for res in retrieval_results]

        except Exception as e:
            print(f"Error on SUBJECT_ID {subject_id}: {e}")
            top_passages = []

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
            "SUBJECT_ID": subject_id,
            "NUM_NOTES": num_notes,
            "query": query,
            "needle": needle,
            "needle_rank": needle_rank,
            "found": found,
            "num_passages": num_chunks,
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages
        })

        tqdm.write(f"Progress: {_+1}/{len(df)}, Current Needle Rank: {needle_rank}, Current Recall@{k}: {found_count/(_+1):.2f}")

    # Save results
    results_df = pd.DataFrame(results)

    # Mean rank (only where needle was found)
    mean_rank = results_df["needle_rank"].dropna().mean()
    # % at rank 1
    pct_rank_1 = (results_df["needle_rank"] == 1).mean()
    # % at rank <= K
    pct_rank_k = (results_df["needle_rank"] <= k).mean()

    print("\n===== Retrieval Rank Metrics =====")
    print(f"Mean needle rank: {mean_rank:.2f}")
    print(f"% needles at rank 1: {pct_rank_1:.2%}")
    print(f"% needles at rank ≤ {k}: {pct_rank_k:.2%}")
    print(f"Final Recall@{k}: {results_df['found'].mean():.4f}")

    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved ColBERT results to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColBERT Needle-in-a-Haystack Retriever (query-based)")
    parser.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    parser.add_argument("--output_csv", type=str, default="src/retrieval_query/outputs/colbert_patient_results.csv")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()

    run_colbert_retrieval(
        input_csv=args.haystack_csv,
        output_csv=args.output_csv,
        k=args.top_k,
        chunk_size=args.chunk_size
    )


    # python src/retrieval_query_level/colbert_retrieval.py \
    # --haystack_csv src/haystacks/mimic_haystack.csv \
    # --output_csv src/retrieval_query/outputs/colbert_patient_results.csv \
    # --top_k 5 \
    # --chunk_size 256

    # 2, 512 and 5, 256 and 10, 128