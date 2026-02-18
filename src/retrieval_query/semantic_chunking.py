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

    return list(dict.fromkeys([c.strip() for c in chunks if c.strip()]))


def retrieve_topk_cosine(chunks, query, model, top_k: int):
    if not chunks:
        return []

    doc_emb = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)

    faiss.normalize_L2(doc_emb)
    faiss.normalize_L2(q_emb)

    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)

    k = min(top_k, len(chunks))
    _, idx = index.search(q_emb, k)
    return [chunks[i] for i in idx[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    ap.add_argument("--output_csv", type=str, default="src/retrieval_query/outputs/semantic_chunking_patient_results.csv")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_sents", type=int, default=5)
    ap.add_argument("--sim_threshold", type=float, default=0.55)
    ap.add_argument("--embed_model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--limit_rows", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.haystack_csv)
    if args.limit_rows:
        df = df.head(args.limit_rows)

    model = SentenceTransformer(args.embed_model)

    results = []
    found_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Semantic chunking retrieval"):
        haystack = str(row["PATIENT_RECORD"])
        needle = str(row["NEEDLE_INSERTED"])   
        query = str(row["QUERY"])              

        chunks = chunk_by_similarity(haystack, model=model, max_sents=args.max_sents, similarity_threshold=args.sim_threshold)
        top_passages = retrieve_topk_cosine(chunks, query, model, top_k=args.top_k)

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
            "query": query,
            "needle": needle,
            "needle_rank": needle_rank,
            "found": found,
            "num_passages": len(chunks),
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages
        })

        tqdm.write(f"Progress: {idx+1}/{len(df)}, Current Needle Rank: {needle_rank}, Current Recall@{args.top_k}: {found_count/(idx+1):.2f}")

    results_df = pd.DataFrame(results)
    print("\n===== Retrieval Rank Metrics =====")
    print(f"Mean needle rank: {results_df['needle_rank'].dropna().mean():.2f}")
    print(f"% needles at rank 1: {(results_df['needle_rank'] == 1).mean():.2%}")
    print(f"% needles at rank ≤ {args.top_k}: {(results_df['needle_rank'] <= args.top_k).mean():.2%}")
    print(f"Recall@{args.top_k}: {results_df['found'].mean():.4f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
 