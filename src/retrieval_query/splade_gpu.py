import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

DEFAULT_SPLADE = "naver/splade-cocondenser-ensembledistil"


def chunk_sliding_sentences(text: str, window_size: int = 3):
    """
    Splits text into sentences and creates overlapping chunks of sentences.
    
    Args:
        text (str): The input text to be chunked.
        window_size (int): The number of sentences in each chunk.

    Returns:
        List[str]: A list of sentence chunks.
    """
    sents = sent_tokenize(str(text))
    if not sents:
        return []
    if window_size <= 0 or len(sents) <= window_size:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + window_size]) for i in range(len(sents) - window_size + 1)]


@torch.no_grad()
def splade_encode(texts, tokenizer, model, device, max_length=256):
    """
    Encodes a list of texts into sparse vectors using SPLADE.
    
    Args:
        texts (List[str]): List of input texts to encode.
        tokenizer: The SPLADE tokenizer.
        model: The SPLADE model.
        device: The device to run the model on.
        
    Returns:
        np.ndarray: A 2D array where each row is the sparse vector representation of the corresponding input text.
    """
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    logits = model(**tok).logits  # [B, T, V]
    weights = torch.log1p(torch.relu(logits))
    sparse = torch.max(weights, dim=1).values  # [B, V]

    return sparse.cpu().numpy()


def retrieve_topk(chunks, query, tokenizer, model, device, top_k=5, max_length=256, batch_size=8):
    """
    Retrieves the top-k most relevant chunks for a given query using SPLADE.
    
    Args:
        chunks (List[str]): List of text chunks to search through.
        query (str): The query string to match against the chunks.
        tokenizer: The SPLADE tokenizer.
        model: The SPLADE model.
        device: The device to run the model on.
        top_k (int): The number of top relevant chunks to return.
        max_length (int): Maximum token length for encoding.
        batch_size (int): Number of chunks to encode at once.
        
    Returns:
        List[str]: The top-k most relevant chunks based on SPLADE similarity to the query.
    """
    if not chunks:
        return []

    q_vec = splade_encode([query], tokenizer, model, device, max_length=max_length)[0]

    doc_vecs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        doc_vecs.append(splade_encode(batch, tokenizer, model, device, max_length=max_length))
    doc_mat = np.vstack(doc_vecs)

    scores = doc_mat @ q_vec
    k = min(top_k, len(chunks))
    idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    parser.add_argument("--output_csv", type=str, default="src/retrieval_query/outputs/splade_patient_results.csv")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--model_name", type=str, default=DEFAULT_SPLADE)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=50, help="Save results every N patients")
    args = parser.parse_args()

    df = pd.read_csv(args.haystack_csv)
    if args.limit_rows:
        df = df.head(args.limit_rows)

    # Load previous results if output CSV exists
    if os.path.exists(args.output_csv):
        results_df = pd.read_csv(args.output_csv)
        processed_ids = set(results_df["SUBJECT_ID"].tolist())
        print(f"Resuming: {len(processed_ids)} patients already processed.")
    else:
        results_df = pd.DataFrame()
        processed_ids = set()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    model.eval()

    found_count = results_df["found"].sum() if not results_df.empty else 0
    batch_results = []

    # Process only rows that haven't been processed yet
    for i, row in tqdm(df.iterrows(), total=len(df), desc="SPLADE retrieval"):
        if row["SUBJECT_ID"] in processed_ids:
            continue

        haystack = str(row["PATIENT_RECORD"])
        query = str(row["QUERY"])         
        needle = str(row["NEEDLE_INSERTED"])

        chunks = chunk_sliding_sentences(haystack, window_size=args.window_size)
        top_passages = retrieve_topk(
            chunks, query, tokenizer, model, device,
            top_k=args.top_k, max_length=args.max_length, batch_size=args.batch_size
        )

        needle_rank = np.nan
        for rank, passage in enumerate(top_passages):
            if needle in passage:
                needle_rank = rank + 1
                break

        found = not np.isnan(needle_rank)
        if found:
            found_count += 1

        result_row = {
            "SUBJECT_ID": row["SUBJECT_ID"],
            "NUM_NOTES": row["NUM_NOTES"],
            "query": query,
            "needle": needle,
            "needle_rank": needle_rank,
            "found": found,
            "num_passages": len(chunks),
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages,
        }

        batch_results.append(result_row)
        processed_ids.add(row["SUBJECT_ID"])

        # Save every N patients
        if len(batch_results) >= args.save_every:
            results_df = pd.concat([results_df, pd.DataFrame(batch_results)], ignore_index=True)
            os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
            results_df.to_csv(args.output_csv, index=False)
            batch_results = []
            tqdm.write(f"Saved progress after {len(results_df)} patients. Current Recall@{args.top_k}: {found_count/len(results_df):.2f}")

    # Save any remaining results
    if batch_results:
        results_df = pd.concat([results_df, pd.DataFrame(batch_results)], ignore_index=True)
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        results_df.to_csv(args.output_csv, index=False)
        tqdm.write(f"Saved final batch. Total patients processed: {len(results_df)}")

    # Final metrics
    print("\n===== Retrieval Rank Metrics =====")
    print(f"Mean needle rank: {results_df['needle_rank'].dropna().mean():.2f}")
    print(f"% needles at rank 1: {(results_df['needle_rank'] == 1).mean():.2%}")
    print(f"% needles at rank ≤ {args.top_k}: {(results_df['needle_rank'] <= args.top_k).mean():.2%}")
    print(f"Recall@{args.top_k}: {results_df['found'].mean():.4f}")

    print(f"\nSaved SPLADE results to {args.output_csv}")


if __name__ == "__main__":
    main()