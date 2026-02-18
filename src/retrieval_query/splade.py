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
    sents = sent_tokenize(str(text))
    if not sents:
        return []
    if window_size <= 0 or len(sents) <= window_size:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + window_size]) for i in range(len(sents) - window_size + 1)]


@torch.no_grad()
def splade_encode(texts, tokenizer, model, device, max_length=256):

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
    args = parser.parse_args()

    df = pd.read_csv(args.haystack_csv)
    if args.limit_rows:
        df = df.head(args.limit_rows)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    model.eval()

    results = []
    found_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="SPLADE retrieval"):
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

        results.append({
            "SUBJECT_ID": row["SUBJECT_ID"],
            "NUM_NOTES": row["NUM_NOTES"],
            "query": query,
            "needle": needle,
            "needle_rank": needle_rank,
            "found": found,
            "num_passages": len(chunks),
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages,
        })

        tqdm.write(f"Progress: {i+1}/{len(df)}, Current Needle Rank: {needle_rank}, Current Recall@{args.top_k}: {found_count/(i+1):.2f}")

    results_df = pd.DataFrame(results)

    print("\n===== Retrieval Rank Metrics =====")
    print(f"Mean needle rank: {results_df['needle_rank'].dropna().mean():.2f}")
    print(f"% needles at rank 1: {(results_df['needle_rank'] == 1).mean():.2%}")
    print(f"% needles at rank ≤ {args.top_k}: {(results_df['needle_rank'] <= args.top_k).mean():.2%}")
    print(f"Recall@{args.top_k}: {results_df['found'].mean():.4f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)

    print(f"\nSaved SPLADE results to {args.output_csv}")


if __name__ == "__main__":
    main()