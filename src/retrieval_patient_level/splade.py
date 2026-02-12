"""Q2 Omid - Patient-level SPLADE Retriever (sparse expansion)

Uses a SPLADE masked-LM to build sparse vectors for sentence-window chunks.
Start with --limit_rows 25–50.

Input CSV columns expected:
- SUBJECT_ID
- NUM_NOTES
- PATIENT_RECORD
- NEEDLE_INSERTED
"""

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
    return [" ".join(sents[i:i+window_size]) for i in range(0, len(sents) - window_size + 1)]


@torch.no_grad()
def splade_encode(texts, tokenizer, model, device, max_length=256):
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    out = model(**tok).logits  # [B, T, V]
    weights = torch.log1p(torch.relu(out))
    sparse = torch.max(weights, dim=1).values  # [B, V]
    return sparse.cpu().numpy()


def retrieve_topk(chunks, query, tokenizer, model, device, top_k=5, max_length=256, batch_size=8):
    if not chunks:
        return []

    q_vec = splade_encode([query], tokenizer, model, device, max_length=max_length)[0]

    doc_vecs = []
    for i in range(0, len(chunks), batch_size):
        doc_vecs.append(splade_encode(chunks[i:i+batch_size], tokenizer, model, device, max_length=max_length))
    doc_mat = np.vstack(doc_vecs)

    scores = doc_mat @ q_vec
    k = min(top_k, len(chunks))
    idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    ap.add_argument("--output_csv", type=str, default="src/retrieval_patient_level/outputs/splade_patient_results.csv")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--model_name", type=str, default=DEFAULT_SPLADE)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit_rows", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.haystack_csv)
    if args.limit_rows is not None:
        df = df.head(args.limit_rows)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    model.eval()

    results = []
    found_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="SPLADE retrieval"):
        haystack = str(row["PATIENT_RECORD"])
        needle = str(row["NEEDLE_INSERTED"])

        chunks = chunk_sliding_sentences(haystack, window_size=args.window_size)
        top_passages = retrieve_topk(
            chunks, needle, tokenizer, model, device,
            top_k=args.top_k, max_length=args.max_length, batch_size=args.batch_size
        )

        needle_rank = np.nan
        for i, passage in enumerate(top_passages):
            if needle in passage:
                needle_rank = i + 1
                break

        found = not np.isnan(needle_rank)
        if found:
            found_count += 1

        results.append({
            "SUBJECT_ID": row.get("SUBJECT_ID", np.nan),
            "NUM_NOTES": row.get("NUM_NOTES", np.nan),
            "needle": needle,
            "needle_rank": needle_rank,
            "found": found,
            "num_passages": len(chunks),
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages
        })

        if (idx + 1) % 10 == 0:
            tqdm.write(f"Progress: {idx+1}/{len(df)} | Recall@{args.top_k}: {found_count/(idx+1):.2f}")

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
