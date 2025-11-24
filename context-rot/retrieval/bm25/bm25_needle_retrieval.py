import pandas as pd
import glob
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import numpy as np
import nltk

nltk.download('punkt')


def retrieve_needles(haystack, question, top_k=3, window_size=3):
    """
    Retrieve top-k passages from haystack relevant to the question using BM25.

    Args:
        haystack (str): The text to search within.
        question (str): The query to search for.
        top_k (int): Number of top passages to retrieve.
        window_size (int): Number of sentences per passage.

    Returns: list Top-k retrieved passages.
    """
    # split haystack into overlapping passages
    sentences = sent_tokenize(haystack)
    passages = [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]
    
    # tokenize passages
    tokenized_passages = [word_tokenize(p.lower()) for p in passages]
    
    # BM25 index
    bm25 = BM25Okapi(tokenized_passages)
    
    # tokenize question
    tokenized_question = word_tokenize(question.lower())
    
    # scores and top-k indices
    scores = bm25.get_scores(tokenized_question)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    top_passages = [passages[i] for i in top_indices]
    return top_passages


def evaluate_bm25_retrieval(csv_pattern="./data/niah_prompts/needle_*/niah_prompts_sequential.csv",
                            top_k=3, window_size=3):
    """
    Run BM25 retrieval on multiple needle CSVs and compute top-k accuracy.
    Returns a summary dataframe and overall accuracy.
    """
    all_results = []

    for csv_file in glob.glob(csv_pattern):
        df = pd.read_csv(csv_file)

        for idx, row in df.iterrows():
            question = row['question']
            haystack = row['prompt']
            needle = row['answer']

            top_passages = retrieve_needles(haystack, question, top_k=top_k, window_size=window_size)
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

    print(results_df["needle_type"])
    print(f"Overall top-{top_k} retrieval accuracy: {accuracy:.2f}")
    print(f"Per Needle Accuracy: {results_df.groupby("csv_file")["found"].mean()}")

    return results_df, accuracy

if __name__ == "__main__":
    # as top_k increases, accuracy increase bc odds that the needle appears in any of the top-k passages increases
    results_df, accuracy = evaluate_bm25_retrieval(top_k=5, window_size=3)
    results_df.to_csv("retrieval/bm25/bm25_retrieval_results.csv", index=False)