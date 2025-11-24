import pandas as pd
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_needles_faiss(haystack, question, top_k=3):
    """
    Retrieve top-k passages from haystack relevant to the question using FAISS (Euclidean distance).

    Args:
        haystack (str): The text containing multiple passages.
        question (str): The question to find relevant passages for.
        top_k (int): Number of top passages to retrieve.

    Returns: List of top-k relevant passages.
    """
    # split haystack into overlapping passages
    sentences = sent_tokenize(haystack)
    window_size = 3
    passages = [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]
    
    if not passages:
        return []

    # encode passages
    passage_embeddings = model.encode(passages, convert_to_numpy=True)

    # FAISS index (Euclidean)
    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(passage_embeddings)

    # encode question
    q_emb = model.encode([question], convert_to_numpy=True)

    # search
    distances, indices = index.search(q_emb, top_k)
    top_passages = [passages[i] for i in indices[0]]
    return top_passages


def evaluate_faiss_retrieval(csv_pattern="./data/niah_prompts/needle_*/niah_prompts_sequential.csv", top_k=3):
    """
    Evaluate FAISS Euclidean retrieval on needle-in-haystack datasets.

    Args:
        csv_pattern (str): Glob pattern to locate CSV files.
        top_k (int): Number of top passages to retrieve.
        
    Returns: DataFrame with retrieval results and accuracy.
    """
    all_results = []

    for csv_file in glob.glob(csv_pattern):
        df = pd.read_csv(csv_file)

        for idx, row in df.iterrows():
            question = row['question']
            haystack = row['prompt']
            needle = row['answer']

            top_passages = retrieve_needles_faiss(haystack, question, top_k=top_k)
            found = any(needle in passage for passage in top_passages)

            all_results.append({
                "csv_file": csv_file,
                "needle_type": csv_file.split("/")[-2],
                "needle": needle,
                "question": question,
                "found": found,
                "top_passages": top_passages
            })

    results_df = pd.DataFrame(all_results)
    accuracy = results_df["found"].mean()

    print(f"Overall top-{top_k} FAISS Euclidean retrieval accuracy: {accuracy:.2f}")
    print(f"Per Needle Accuracy: {results_df.groupby('csv_file')['found'].mean()}")

    return results_df


if __name__ == "__main__":
    results_df = evaluate_faiss_retrieval(top_k=5)
    results_df.to_csv("retrieval/faiss_euclidean/faiss_euc_retrieval_results.csv", index=False)