import pandas as pd
from fuzzywuzzy import fuzz
import os
import json
import re

def normalize_response(resp):
    if not isinstance(resp, str):
        return ""

    resp = resp.strip()

    # Try parsing JSON list
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, list):
            # Deduplicate after normalization
            parsed = [normalize_needle(p) for p in parsed]
            parsed = list(set(parsed))
            resp = " ".join(parsed)
    except:
        pass

    # Normalize whitespace & punctuation
    resp = resp.lower()
    resp = re.sub(r"\s+", " ", resp)
    resp = resp.strip()

    return resp


def normalize_needle(needle):
    if not isinstance(needle, str):
        return ""
    return re.sub(r"\s+", " ", needle.lower()).strip()


def evaluate_extraction(responses_dict, output_file="src/eval/patient_level/extract/all_methods_llm_eval.csv"):
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
        # Handle None values
        df['bedrock_response'] = df['bedrock_response'].fillna("")
        df['needle'] = df['needle'].fillna("")
        # Normalize responses and needles
        df['norm_response'] = df['bedrock_response'].apply(normalize_response)
        df['norm_needle'] = df['needle'].apply(normalize_needle)
        # Compute exact match
        df['exact_match'] = df.apply(lambda r: r['norm_needle'] in r['norm_response'], axis=1)
        # Compute fuzzy match ratio
        df['fuzzy_score'] = df.apply(lambda r: fuzz.token_set_ratio(r['norm_needle'], r['norm_response']), axis=1)
        df['fuzzy_match_80'] = df['fuzzy_score'] >= 80  # threshold for "good enough"
        # Compute empty extraction rate
        df['empty_extraction'] = df['norm_response'].isin(["", "none"])


        summary_rows.append({
            'strategy': strategy,
            'num_patients': len(df),
            'overall_exact_match': df['exact_match'].mean(),
            'overall_fuzzy_80_match': df['fuzzy_match_80'].mean(),
            'empty_extraction_rate': df['empty_extraction'].mean()
        })

        # save per-method detailed CSV
        per_method_file = os.path.join(os.path.dirname(output_file), f"{strategy}_eval.csv")
        df.to_csv(per_method_file, index=False)
        print(f"Saved detailed per-patient eval for {strategy} to {per_method_file}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary evaluation for all methods to {output_file}")
    print(summary_df)

# Load responses for each method (can add more methods)
responses_dict = {
    "bm25": pd.read_csv("src/bedrock_pipeline/bedrock_responses/extract/bm25_responses.csv"),
    "colbert": pd.read_csv("src/bedrock_pipeline/bedrock_responses/extract/colbert_responses.csv"),
    "faiss_cos": pd.read_csv("src/bedrock_pipeline/bedrock_responses/extract/faiss_cos_responses.csv"),
    "faiss_euc": pd.read_csv("src/bedrock_pipeline/bedrock_responses/extract/faiss_euc_responses.csv"),
    "hybrid": pd.read_csv("src/bedrock_pipeline/bedrock_responses/extract/hybrid_responses.csv"),
}

evaluate_extraction(responses_dict)