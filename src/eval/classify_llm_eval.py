import pandas as pd
import os

def merge_llm_results(task, method):
    """
    Merge LLM classification results with retrieval results.
    Args:
        task: "classify"
        method: "bm25", "faiss_cos", "faiss_euc", "hybrid"

    Returns:
        None
    """

    retrieval_file = f"src/retrieval/outputs/{method}_mimic_results.csv"
    response_file = f"src/bedrock_pipeline/bedrock_responses/{task}/{method}_responses.csv"

    retrieval = pd.read_csv(retrieval_file)
    responses = pd.read_csv(response_file)

    # merge on "needle"
    merged = retrieval.merge(
        responses[["needle", "bedrock_response"]],
        on="needle",
        how="left"
    )

    # classify correctness automatically
    merged["llm_correct"] = merged["bedrock_response"].str.lower().str.contains("yes")

    output_file = f"src/eval/{task}/{method}_llm_eval.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)

    print(f"Saved {output_file}")


# can add faiss_euc, and hybrid
methods = ["bm25", "faiss_cos"]
task = "classify"
for method in methods:
    merge_llm_results(task, method)