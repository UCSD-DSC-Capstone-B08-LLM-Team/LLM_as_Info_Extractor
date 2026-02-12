import pandas as pd
import os
import ast
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Create a full-context style retrieval output
    results = []
    for _, row in df.iterrows():
        full_context = str(row["PATIENT_RECORD"])
        results.append({
            "SUBJECT_ID": row["SUBJECT_ID"],
            "query": row["QUERY"],
            "needle": row["NEEDLE_INSERTED"],
            "top_passages": [full_context]  # full context as a single "chunk"
        })

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved full-context retrieval-style CSV to {args.output_csv}")