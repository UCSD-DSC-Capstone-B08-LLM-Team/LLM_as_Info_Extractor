import pandas as pd
import glob
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

model = SentenceTransformer('all-MiniLM-L6-v2')  

def retrieve_needles_faiss(haystack, question, top_k=3, window_size=3):
    """
    Retrieve top-k passages using FAISS cosine similarity.

    Args:     
        haystack (str): the text to search within
        question (str): the query text
        top_k (int): number of top passages to retrieve
        window_size (int): number of sentences per passage

    Returns: list of top-k passages
    """
    # split haystack into overlapping passages
    sentences = sent_tokenize(haystack)
    passages = [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]
    
    if not passages:
        return []

    # compute embeddings
    passage_embeddings = model.encode(passages, convert_to_numpy=True)
    question_embedding = model.encode([question], convert_to_numpy=True)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(passage_embeddings)
    faiss.normalize_L2(question_embedding)

    # FAISS index
    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity if vectors normalized
    index.add(passage_embeddings)

    # Search top_k
    scores, indices = index.search(question_embedding, top_k)
    top_passages = [passages[i] for i in indices[0]]

    return top_passages

def evaluate_faiss_retrieval(csv_pattern="./data/niah_prompts/needle_*/niah_prompts_sequential.csv", top_k=3, window_size=3):
    """
    Evaluate FAISS retrieval on needle datasets.

    Args:
        csv_pattern: str, glob pattern to locate CSV files
        top_k: int, number of top passages to retrieve
        window_size: int, number of sentences per passage

    Returns:
        results_df: pd.DataFrame, detailed retrieval results
        accuracy: float, overall retrieval accuracy
    """
    all_results = []
    count = 0

    for csv_file in glob.glob(csv_pattern):
        df = pd.read_csv(csv_file)

        for idx, row in df.iterrows():
            question = row['question']
            haystack = row['prompt']
            needle = row['answer']

            top_passages = retrieve_needles_faiss(haystack, question, top_k=top_k, window_size=window_size)
            found = any(needle in passage for passage in top_passages)

            all_results.append({
                "csv_file": csv_file,
                "needle_type": csv_file.split("/")[-2],
                "needle": needle,
                "question": question,
                "found": found,
                "top_passages": top_passages
            })
        count += 1

    results_df = pd.DataFrame(all_results)
    accuracy = results_df["found"].mean()

    print(f"Overall top-{top_k} FAISS retrieval accuracy: {accuracy:.2f}")
    print(f"Per Needle Accuracy: {results_df.groupby('csv_file')['found'].mean()}")

    return results_df, accuracy


if __name__ == "__main__":
    # increasing window size tends to increase accuracy as passages are longer
    results_df, accuracy = evaluate_faiss_retrieval(top_k=5, window_size=4)
    results_df.to_csv("retrieval/faiss_cosine/faiss_cos_retrieval_results.csv", index=False)