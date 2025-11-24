import pandas as pd
import glob
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_indices(haystack, window_size=3):
    """
    Build both BM25 and FAISS indices for a given haystack.

    Args:
        haystack (str): The text to index.
        window_size (int): Number of sentences per passage.

    Returns:
        passages, bm25_index, faiss_index, passage_embeddings
    """
    # split into overlapping passages
    sentences = sent_tokenize(haystack)
    passages = [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]
    
    # BM25
    tokenized_passages = [word_tokenize(p.lower()) for p in passages]
    bm25_index = BM25Okapi(tokenized_passages)
    
    # FAISS embeddings
    passage_embeddings = model.encode(passages, convert_to_numpy=True)
    faiss.normalize_L2(passage_embeddings)  # for cosine similarity
    dim = passage_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(passage_embeddings)
    
    return passages, bm25_index, faiss_index, passage_embeddings

def hybrid_retrieve(question, passages, bm25_index, faiss_index, passage_embeddings,
                    top_k=5, bm25_weight=0.5, faiss_weight=0.5):
    """
    Retrieve top passages using hybrid BM25 + FAISS.

    Args:
        question (str): The query question.
        passages (list): List of passages.
        bm25_index: BM25 index object.
        faiss_index: FAISS index object.
        passage_embeddings: Embeddings of passages.
        top_k (int): Number of top passages to retrieve.
        bm25_weight (float): Weight for BM25 scores.
        faiss_weight (float): Weight for FAISS scores.

    Returns: Top-k retrieved passages.
    """
    # BM25 scores
    tokenized_question = word_tokenize(question.lower())
    bm25_scores = bm25_index.get_scores(tokenized_question)
    
    # FAISS scores
    q_emb = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    faiss_scores, faiss_indices = faiss_index.search(q_emb, top_k)
    
    # combine scores
    scores_dict = {}
    for i, s in enumerate(bm25_scores):
        scores_dict[passages[i]] = bm25_weight * s
    for j, idx in enumerate(faiss_indices[0]):
        scores_dict[passages[idx]] = scores_dict.get(passages[idx], 0) + faiss_weight * faiss_scores[0][j]
    
    # return top_k passages
    sorted_passages = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return [p for p, _ in sorted_passages[:top_k]]

def evaluate_hybrid(csv_pattern="./data/niah_prompts/needle_*/niah_prompts_sequential.csv",
                    top_k=5, window_size=3, bm25_weight=0.5, faiss_weight=0.5):
    """
    Run hybrid retrieval on multiple CSVs and compute top-k accuracy efficiently.

    Args:
        csv_pattern (str): Glob pattern to find CSV files.
        top_k (int): Number of top passages to consider.
        window_size (int): Number of sentences per passage.
        bm25_weight (float): Weight for BM25 scores.
        faiss_weight (float): Weight for FAISS scores.
        
    Returns:
        results_df : DataFrame with retrieval results.
        accuracy : Overall top-k accuracy.
    """
    all_results = []

    for csv_file in glob.glob(csv_pattern):
        df = pd.read_csv(csv_file)
        # precompute indices for this haystack assuming it's the same for all rows
        haystack = df.iloc[0]['prompt']
        passages, bm25_index, faiss_index, passage_embeddings = build_indices(haystack, window_size)
        
        for idx, row in df.iterrows():
            question = row['question']
            needle = row['answer']
            
            top_passages = hybrid_retrieve(question, passages, bm25_index, faiss_index, passage_embeddings,
                                           top_k=top_k, bm25_weight=bm25_weight, faiss_weight=faiss_weight)
            found = any(needle in p for p in top_passages)
            
            all_results.append({
                "csv_file": csv_file,
                "needle_type": csv_file.split("/")[-2],
                "question": question,
                "needle": needle,
                "found": found,
                "top_passages": top_passages
            })
    
    results_df = pd.DataFrame(all_results)
    accuracy = results_df["found"].mean()
    print(f"Overall top-{top_k} hybrid retrieval accuracy: {accuracy:.2f}")
    print(f"Per Needle Accuracy:\n{results_df.groupby('csv_file')['found'].mean()}")
    return results_df, accuracy

if __name__ == "__main__":
    results_df, accuracy = evaluate_hybrid(top_k=5, window_size=3, bm25_weight=0.5, faiss_weight=0.5)
    results_df.to_csv("retrieval/hybrid_bm25_faiss/hybrid_bm25_faiss_retrieval_results.csv", index=False)