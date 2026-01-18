import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import numpy as np
import nltk
import os

nltk.download('punkt')

def retrieve_needles(haystack, needle, top_k=3, window_size=3):
    """
    Retrieve top_k passages from haystack that are most relevant to needle using BM25.
    
    Args:
        haystack (str): The text to search within.
        needle (str): The text to search for.
        top_k (int): Number of top passages to retrieve.
        window_size (int): Number of sentences per passage.
        
    Returns:
        top_passages: List of strings of top_k passages most relevant to the needle.
    """

    sentences = sent_tokenize(str(haystack))

    # handle short or empty notes
    if len(sentences) == 0:
        return []
    
    if len(sentences) < window_size:
        passages = [' '.join(sentences)]
    else:
        passages = [' '.join(sentences[i:i+window_size])
                    for i in range(len(sentences) - window_size + 1)]

    tokenized_passages = [word_tokenize(p.lower()) for p in passages]

    # tokenization edge cases
    if len(tokenized_passages) == 0:
        return []

    bm25 = BM25Okapi(tokenized_passages)
    tokenized_needle = word_tokenize(str(needle).lower())

    scores = bm25.get_scores(tokenized_needle)

    if len(scores) == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [passages[i] for i in top_indices]

# Load your CSV
df = pd.read_csv("src/haystacks/mimic_haystack.csv")

results = []

for idx, row in df.iterrows():
    haystack = row['MODIFIED_NOTE']      # the note text
    needle = row['NEEDLE_INSERTED']      # the "needle" you're trying to retrieve

    top_passages = retrieve_needles(haystack, needle, top_k=2, window_size=3)

    needle_tokens = set(word_tokenize(needle.lower()))
    def contains_needle(passage):
        passage_tokens = set(word_tokenize(passage.lower()))
        return len(needle_tokens & passage_tokens) / len(needle_tokens) > 0.6

    found = any(contains_needle(p) for p in top_passages)

    results.append({
        "HADM_ID": row["HADM_ID"],
        "SUBJECT_ID": row["SUBJECT_ID"],
        "CATEGORY": row["CATEGORY"],
        "needle": needle,
        "found": found,
        "top_passages": top_passages
    })

results_df = pd.DataFrame(results)
accuracy = results_df["found"].mean()
print(f"Overall retrieval accuracy: {accuracy:.4f}")

output_path = "src/retrieval/outputs/bm25_mimic_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"Saved BM25 results to {output_path}")