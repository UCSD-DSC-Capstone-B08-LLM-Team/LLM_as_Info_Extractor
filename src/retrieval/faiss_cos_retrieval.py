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
    
    sentences = sent_tokenize(haystack)
    passages = [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]
    
    if not passages:
        return []

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

def evaluate_faiss_cos_on_mimic(haystack_csv, top_k=5, window_size=3):
    """
    Evaluate FAISS cosine retrieval on mimic_haystack.csv

    Args:
        haystack_csv (str): Path to the CSV file containing haystack and needle data.
        top_k (int): Number of top passages to retrieve.
        window_size (int): Number of sentences per passage.

    Returns:
        results_df: DataFrame containing evaluation results.
    """

    df = pd.read_csv(haystack_csv)
    all_results = []
    found_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing MIMIC Notes"):
        haystack = row['MODIFIED_NOTE']
        needle = row['NEEDLE_INSERTED']

        top_passages = retrieve_needle_faiss_cos(haystack, needle, top_k=top_k, window_size=window_size)
        found = any(needle in passage for passage in top_passages)

        if found:
            found_count += 1

        all_results.append({
            "HADM_ID": row['HADM_ID'],
            "SUBJECT_ID": row['SUBJECT_ID'],
            "CATEGORY": row['CATEGORY'],
            "needle": needle,
            "found": found,
            "top_passages": top_passages
        })

        tqdm.write(f"Progress: {_+1}/{len(df)}, Current Accuracy: {found_count/(_+1):.2f}")

    results_df = pd.DataFrame(all_results)
    accuracy = results_df["found"].mean()
    print(f"Overall top-{top_k} FAISS cosine retrieval accuracy: {accuracy:.2f}")
    return results_df


if __name__ == "__main__":
    haystack_file = os.path.join("src", "haystacks", "mimic_haystack.csv")
    results_df = evaluate_faiss_cos_on_mimic(haystack_file, top_k=2, window_size=3)

    output_dir = os.path.join("src", "retrieval", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "faiss_cos_mimic_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")