import pandas as pd
import os

def merge_llm_results(task, method, element):
    """
    Merge LLM classification results with retrieval results.
    Args:
        task: "classify"
        method: "bm25", "colbert", "faiss", "faiss_mmr", "hybrid"

    Returns:
        None
    """
    if  method == "full_context_baseline":
        retrieval_file = f"src/llm/outputs/baseline_{element}.csv"
    elif method == "faiss":
        retrieval_file = f"src/retrieval_query/outputs/{element}/faiss_cos_patient_results.csv"
    elif method == "colbert":
        retrieval_file = f"src/retrieval_query/outputs/{element}/colbert_patient_results.csv"
    else:
        retrieval_file = f"src/retrieval_query/outputs/{element}/{method}_patient_results.csv"

    response_file = f"src/bedrock_pipeline/bedrock_responses/{task}/{element}/{method}_responses.csv"

    retrieval = pd.read_csv(retrieval_file)
    responses = pd.read_csv(response_file)

    # merge on SUBJECT_ID and needle
    merged = retrieval.merge(
        responses[["SUBJECT_ID", "needle", "needle_in_top_k", "bedrock_response"]],
        on=["SUBJECT_ID", "needle"],
        how="left",
        validate="one_to_one"
    )

    # classify correctness automatically
    merged["llm_correct"] = merged["bedrock_response"].str.lower().str.contains("yes")  

    output_file = f"src/eval/patient_level/{task}/{element}/{method}_eval.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)

    print(f"Saved {output_file}")

# Run the merging for all methods
methods = ["full_context_baseline"]#["bm25", "colbert", "faiss", "faiss_mmr", "hybrid", "full_context_baseline", "semantic_chunking", "splade"]
task = "classify"
element = "clinical_trial"
for method in methods:
    merge_llm_results(task, method, element)


def summarize_classify_eval(methods, task, element):
    rows = []

    for method in methods:
        eval_file = f"src/eval/patient_level/{task}/{element}/{method}_eval.csv"
        df = pd.read_csv(eval_file)

        rows.append({
            "strategy": method,
            "num_patients": df["SUBJECT_ID"].nunique(),
            "retrieval_success": df["needle_in_top_k"].mean(),
            "accuracy": df["llm_correct"].mean()
        })

    summary_df = pd.DataFrame(rows)

    output_file = f"src/eval/patient_level/{task}/{element}/all_methods_llm_eval.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_df.to_csv(output_file, index=False)

    print(f"Saved {output_file}")
    print(summary_df)

summarize_classify_eval(methods, task, element)
