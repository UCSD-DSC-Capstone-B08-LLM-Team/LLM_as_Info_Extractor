from tqdm import tqdm
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from nltk.tokenize import sent_tokenize
import nltk
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

nltk.download('punkt')

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  

def retrieve_needle_faiss_cos(haystack, needle, top_k=5, window_size=3):
    """
    Retrieve top_k passages from haystack that are most similar to needle using FAISS cosine similarity.
    
    Args:
        haystack (str): The text to search within.
        needle (str): The text to search for.
        top_k (int): Number of top passages to retrieve.
        window_size (int): Number of sentences per passage.

    Returns:
        top_passages: List of strings with top_k passages most similar to the needle.
    """
    
    # Create passages
    sentences = sent_tokenize(haystack)
    if len(sentences) < window_size:
        passages = [' '.join(sentences)]
    else:
        passages = [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]
    
    # Handle empty passages
    if not passages:
        return []

    # FAISS retrieval (cosine similarity)
    passage_embeddings = model.encode(passages, convert_to_numpy=True)
    needle_embedding = model.encode([needle], convert_to_numpy=True)

    faiss.normalize_L2(passage_embeddings)
    faiss.normalize_L2(needle_embedding)

    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)

    scores, indices = index.search(needle_embedding, top_k)
    top_passages = [passages[i] for i in indices[0]]

    return top_passages

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Patient-level FAISS cosine retrieval")
    parser.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv")
    parser.add_argument("--output_csv", type=str, default="src/retrieval_patient_level/outputs/faiss_cos_patient_results.csv")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.haystack_csv)
    all_results = []
    found_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing MIMIC Patient Records"):
        haystack = row['PATIENT_RECORD']
        needle = row['NEEDLE_INSERTED']

        top_passages = retrieve_needle_faiss_cos(haystack, needle, top_k=args.top_k, window_size=args.window_size)
        num_passages = len(sent_tokenize(haystack)) - args.window_size + 1 if len(sent_tokenize(haystack)) >= args.window_size else 1

        found = any(needle in passage for passage in top_passages)
        if found:
            found_count += 1

        all_results.append({
            "SUBJECT_ID": row["SUBJECT_ID"],
            "NUM_NOTES": row["NUM_NOTES"],
            "needle": needle,
            "found": found,
            "num_passages": num_passages,
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages
        })

        tqdm.write(f"Progress: {_+1}/{len(df)}, Current Accuracy: {found_count/(_+1):.2f}")

    results_df = pd.DataFrame(all_results)
    accuracy = results_df["found"].mean()
    print(f"Overall top-{args.top_k} FAISS cosine retrieval accuracy: {accuracy:.2f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")