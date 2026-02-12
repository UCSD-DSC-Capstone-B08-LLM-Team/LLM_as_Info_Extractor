"""Q2 Omid - Patient-level Semantic Chunking Retriever

Input CSV must have columns:
- SUBJECT_ID
- NUM_NOTES
- PATIENT_RECORD
- NEEDLE_INSERTED

Outputs a retrieval-results CSV compatible with your existing answer generator:
- top_passages (list[str])
- needle_rank, found, num_passages, haystack_len_chars
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

from sentence_transformers import SentenceTransformer
import faiss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DEFAULT_MODEL = "all-MiniLM-L6-v2"

def chunk_by_similarity(text: str, model: SentenceTransformer, max_sents: int, similarity_threshold: float):
    sents = sent_tokenize(str(text))
    if not sents:
        return []

    emb = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(emb)

    chunks = []
    cur = [sents[0]]
    cur_centroid = emb[0].copy()

    def cos(a, b):
        return float(np.dot(a, b))

    for i in range(1, len(sents)):
        sim = cos(cur_centroid, emb[i])
        if (len(cur) >= max_sents) or (sim < similarity_threshold):
            chunks.append(" ".join(cur))
            cur = [sents[i]]
            cur_centroid = emb[i].copy()
        else:
            cur.append(sents[i])
            cur_centroid = cur_centroid + emb[i]
            cur_centroid = cur_centroid / (np.linalg.norm(cur_centroid) + 1e-12)

    if cur:
        chunks.append(" ".join(cur))

    chunks = list(dict.fromkeys([c.strip() for c in chunks if c.strip()]))
    return chunks


def retrieve_topk_cosine(chunks, query, model, top_k: int):
    if not chunks:
        return []

    doc_emb = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)

    faiss.normalize_L2(doc_emb)
    faiss.normalize_L2(q_emb)

    dim = doc_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_emb)

    k = min(top_k, len(chunks))
    _, idx = index.search(q_emb, k)
    return [chunks[i] for i in idx[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    ap.add_argument("--output_csv", type=str, default="src/retrieval_patient_level/outputs/semantic_chunking_patient_results.csv")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_sents", type=int, default=5)
    ap.add_argument("--sim_threshold", type=float, default=0.55)
    ap.add_argument("--embed_model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--limit_rows", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.haystack_csv)
    if args.limit_rows is not None:
        df = df.head(args.limit_rows)

    model = SentenceTransformer(args.embed_model)

    results = []
    found_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Semantic chunking retrieval"):
        haystack = str(row["PATIENT_RECORD"])
        needle = str(row["NEEDLE_INSERTED"])

        chunks = chunk_by_similarity(haystack, model=model, max_sents=args.max_sents, similarity_threshold=args.sim_threshold)
        top_passages = retrieve_topk_cosine(chunks, needle, model, top_k=args.top_k)

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

        if (idx + 1) % 25 == 0:
            tqdm.write(f"Progress: {idx+1}/{len(df)} | Recall@{args.top_k}: {found_count/(idx+1):.2f}")

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
