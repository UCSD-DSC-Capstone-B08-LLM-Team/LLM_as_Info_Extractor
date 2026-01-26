import pandas as pd
from ragatouille import RAGPretrainedModel
from tqdm import tqdm
import argparse

def run_colbert_retrieval(input_csv, output_csv, k=5, chunk_size=256):
    """
    Runs ColBERT retrieval on a 'needle in a haystack' dataset.
    
    Args:
        input_csv (str): Path to mimic_haystack.csv
        output_csv (str): Path to save results
        k (int): Number of top passages to retrieve
        chunk_size (int): Word count for splitting the haystack (ColBERT likes < 300 words)
    """
    
    print("Loading ColBERT model...")
    # 'colbert-ir/colbertv2.0' is the standard checkpoint for general purpose
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    
    df = pd.read_csv(input_csv)
    results = []

    print(f"Processing {len(df)} haystack records...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        haystack = str(row['PATIENT_RECORD'])
        needle = str(row['NEEDLE_INSERTED'])
        subject_id = row['SUBJECT_ID']
        num_notes = row['NUM_NOTES']

        # Chunk the Haystack with 10% overlap
        words = haystack.split()
        overlap = int(chunk_size * 0.1)
        step = chunk_size - overlap
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), step)]

        num_chunks = len(chunks)
        current_k = min(k, num_chunks)
        
        # Retrieve top passages using ColBERT
        try:
            if num_chunks == 0:
                top_passages = []
            else:
                # RAG.rerank returns a list of dicts: [{'content': '...', 'score': 0.9, 'rank': 1}, ...]
                retrieval_results = RAG.rerank(query=needle, documents=chunks, k=current_k)
                
                # Extract just the text of the top passages
                if retrieval_results is None:
                    top_passages = []
                else:
                    top_passages = [res['content'] for res in retrieval_results]
        except Exception as e:
            print(f"Error on Subject {subject_id}: {e}")
            top_passages = []

        # Check if needle is in any of the top passages
        combined_retrieved_text = " ".join(top_passages)
        found = needle in combined_retrieved_text

        results.append({
            "SUBJECT_ID": subject_id,
            "NUM_NOTES": num_notes,
            "needle": needle,
            "found": found,
            "num_chunks": len(chunks), 
            "num_passages": len(top_passages),
            "haystack_len_chars": len(haystack),
            "top_passages": top_passages
        })

    # Save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved ColBERT results to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColBERT Needle in a Haystack Retriever")
    
    # Add Arguments
    parser.add_argument("--haystack_csv", type=str, default="src/haystacks/mimic_haystack.csv", help="Path to input CSV with 'haystack' and 'needle' columns")
    parser.add_argument("--output_csv", type=str, default="src/retrieval_patient_level/outputs/colbert_patient_results.csv", help="Path to save output CSV")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top passages to retrieve")
    parser.add_argument("--chunk_size", type=int, default=256, help="Word count per chunk")

    args = parser.parse_args()

    # Run function
    run_colbert_retrieval(input_csv=args.haystack_csv, output_csv=args.output_csv, k=args.top_k, chunk_size=args.chunk_size)

    # python src/retrieval_patient_level/colbert_retrieval.py \
    # --haystack_csv src/haystacks/mimic_haystack.csv \
    # --output_csv src/retrieval_patient_level/outputs/colbert_patient_results.csv \
    # --top_k 5 \
    # --chunk_size 256

    # 2, 512 and 5, 256 and 10, 128