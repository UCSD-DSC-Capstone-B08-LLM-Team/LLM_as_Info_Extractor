import os
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

# Initialize sentence-transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def build_indices(haystack, window_size=3):
    """
    Build BM25 and FAISS indices for a given haystack.

    Args:
        haystack (str): The text to build indices from.
        window_size (int): Number of sentences per passage.

    Returns:
        passages: List of passages.
        bm25_index: BM25 index built from passages.
        faiss_index: FAISS index built from passage embeddings.
        passage_embeddings: Numpy array of embeddings of the passages.
    """
    
    sentences = sent_tokenize(haystack)
    passages = [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]

    # BM25 index
    tokenized_passages = [word_tokenize(p.lower()) for p in passages]
    bm25_index = BM25Okapi(tokenized_passages)

    # FAISS embeddings (cosine similarity)
    passage_embeddings = model.encode(passages, convert_to_numpy=True)
    faiss.normalize_L2(passage_embeddings)
    dim = passage_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(passage_embeddings)

    return passages, bm25_index, faiss_index, passage_embeddings


def hybrid_retrieve(question, passages, bm25_index, faiss_index, passage_embeddings,
                    top_k=5, bm25_weight=0.5, faiss_weight=0.5):
    """
    Retrieve top passages using hybrid BM25 + FAISS cosine similarity.

    Args:
        question (str): The query question.
        passages (list): List of passages.
        bm25_index: BM25 index.
        faiss_index: FAISS index.
        passage_embeddings: Numpy array of passage embeddings.
        top_k (int): Number of top passages to retrieve.
        bm25_weight (float): Weight for BM25 scores.
        faiss_weight (float): Weight for FAISS scores.

    Returns:
        List of top-k passages.
    """

    # BM25 scores
    tokenized_question = word_tokenize(question.lower())
    bm25_scores = bm25_index.get_scores(tokenized_question)

    # FAISS scores
    q_emb = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    faiss_scores, faiss_indices = faiss_index.search(q_emb, top_k)

    # Combine scores
    scores_dict = {}
    for i, s in enumerate(bm25_scores):
        scores_dict[passages[i]] = bm25_weight * s
    for j, idx in enumerate(faiss_indices[0]):
        scores_dict[passages[idx]] = scores_dict.get(passages[idx], 0) + faiss_weight * faiss_scores[0][j]

    # Return top-k passages
    sorted_passages = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return [p for p, _ in sorted_passages[:top_k]]


def evaluate_hybrid_on_mimic(haystack_csv, top_k=5, window_size=3, bm25_weight=0.5, faiss_weight=0.5):
    """
    Evaluate hybrid BM25 + FAISS retrieval on MIMIC haystack.

    Args:
        haystack_csv (str): Path to the MIMIC haystack CSV file.
        top_k (int): Number of top passages to retrieve.
        window_size (int): Number of sentences per passage.
        bm25_weight (float): Weight for BM25 scores.
        faiss_weight (float): Weight for FAISS scores.

    Returns:
        results_df: DataFrame with evaluation results.
    """

    df = pd.read_csv(haystack_csv)
    all_results = []
    found_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing MIMIC Notes"):
        haystack = row['MODIFIED_NOTE']
        needle = row['NEEDLE_INSERTED']

        # Build indices for this note
        passages, bm25_index, faiss_index, passage_embeddings = build_indices(haystack, window_size)

        # Retrieve top passages
        top_passages = hybrid_retrieve(needle, passages, bm25_index, faiss_index, passage_embeddings,
                                       top_k=top_k, bm25_weight=bm25_weight, faiss_weight=faiss_weight)
        found = any(needle in p for p in top_passages)
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
    print(f"Overall top-{top_k} hybrid retrieval accuracy: {accuracy:.2f}")
    return results_df


if __name__ == "__main__":
    haystack_file = os.path.join("src", "haystacks", "mimic_haystack.csv")
    results_df = evaluate_hybrid_on_mimic(haystack_file, top_k=2, window_size=3, bm25_weight=0.5, faiss_weight=0.5)

    output_dir = os.path.join("src", "retrieval", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "hybrid_mimic_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")