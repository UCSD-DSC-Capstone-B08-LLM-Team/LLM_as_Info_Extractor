import pandas as pd
import os

def merge_llm_results(task, method):
    """
    Merge LLM classification results with retrieval results.
    Args:
        task: "classify"
        method: "bm25", "colbert", "faiss", "faiss_mmr", "hybrid"

    Returns:
        None
    """
    if method == "sliding_window_baseline" or method == "full_context_baseline":
        retrieval_file = f"src/llm/outputs/{method}.csv"
    elif method == "faiss":
        retrieval_file = f"src/retrieval_query/outputs/faiss_cos_patient_results.csv"
    elif method == "colbert":
        retrieval_file = f"src/retrieval_query/outputs/colbert_patient_results_5.csv"
    else:
        retrieval_file = f"src/retrieval_query/outputs/{method}_patient_results.csv"

    response_file = f"src/bedrock_pipeline/bedrock_responses/{task}/{method}_responses.csv"

    retrieval = pd.read_csv(retrieval_file)
    responses = pd.read_csv(response_file)

    # merge on SUBJECT_ID and needle
    merged = retrieval.merge(
        responses[["SUBJECT_ID", "needle", "bedrock_response"]],
        on=["SUBJECT_ID", "needle"],
        how="left",
        validate="one_to_one"
    )

    # classify correctness automatically
    merged["llm_correct"] = merged["bedrock_response"].str.lower().str.contains("yes")

    output_file = f"src/eval/patient_level/{task}/{method}_eval.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)

    print(f"Saved {output_file}")

# Run the merging for all methods
methods = ["bm25", "colbert", "faiss", "faiss_mmr", "hybrid", "full_context_baseline", "semantic_chunking", "splade"]
task = "classify"
for method in methods:
    merge_llm_results(task, method)


def summarize_classify_eval(methods, task="classify"):
    rows = []

    for method in methods:
        eval_file = f"src/eval/patient_level/{task}/{method}_eval.csv"
        df = pd.read_csv(eval_file)

        rows.append({
            "strategy": method,
            "num_patients": df["SUBJECT_ID"].nunique(),
            "accuracy": df["llm_correct"].mean()
        })

    summary_df = pd.DataFrame(rows)

    output_file = f"src/eval/patient_level/{task}/all_methods_llm_eval.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_df.to_csv(output_file, index=False)

    print(f"Saved {output_file}")
    print(summary_df)

summarize_classify_eval(methods)
