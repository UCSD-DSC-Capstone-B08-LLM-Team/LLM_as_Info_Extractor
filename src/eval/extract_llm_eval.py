import pandas as pd
from fuzzywuzzy import fuzz
import os

def evaluate_extraction(responses_dict, output_file="src/eval/extract/all_methods_llm_eval.csv"):
    """
    Evaluate extraction accuracy for multiple retrieval methods.

    Args:
        responses_dict (dict): {strategy_name: pd.DataFrame} 
            Each DataFrame should have columns:
                - 'needle': the reference answer
                - 'bedrock_response': the LLM output
        output_file (str): CSV file to save summary metrics

    Returns:
        None
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    summary_rows = []

    for strategy, df in responses_dict.items():
        df = df.copy()
        # Compute exact match
        df['exact_match'] = df.apply(lambda row: row['needle'].strip() == row['bedrock_response'].strip(), axis=1)
        # Compute fuzzy match ratio
        df['fuzzy_ratio'] = df.apply(lambda row: fuzz.ratio(row['needle'], row['bedrock_response']), axis=1)
        df['fuzzy_match_80'] = df['fuzzy_ratio'] >= 80  # threshold for "good enough"

        summary_rows.append({
            'strategy': strategy,
            'num_items': len(df),
            'overall_exact_match': df['exact_match'].mean(),
            'overall_fuzzy_80_match': df['fuzzy_match_80'].mean()
        })

        # save per-method detailed CSV
        per_method_file = os.path.join(os.path.dirname(output_file), f"{strategy}_llm_eval.csv")
        df.to_csv(per_method_file, index=False)
        print(f"Saved detailed per-item eval for {strategy} to {per_method_file}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary evaluation for all methods to {output_file}")
    print(summary_df)

# Load responses for each method (can add more methods)
responses_dict = {
    "bm25": pd.read_csv("src/bedrock_pipeline/bedrock_responses/extract/bm25_responses.csv"),
    "faiss_cos": pd.read_csv("src/bedrock_pipeline/bedrock_responses/extract/faiss_cos_responses.csv"),
}

evaluate_extraction(responses_dict)