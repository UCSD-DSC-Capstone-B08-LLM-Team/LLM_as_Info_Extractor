import pandas as pd
import os
import argparse

def merge_llm_results(task, method, element):
    """
    Merges retrieval results with LLM responses for a given task, method, and clinical element.
    
    Args:
        task (str): The classification task, "classify".
        method (str): The retrieval method used, e.g., "bm25", "faiss", etc.
        element (str): The clinical element being evaluated, e.g., "comfort_care".
        
    Returns:
        None: Saves the merged results to a CSV file.
    """
    if method == "full_context_baseline":
        retrieval_file = f"src/llm/outputs/baseline_{element}.csv"
    elif method == "faiss":
        retrieval_file = f"src/retrieval_query/outputs/{element}/faiss_cos_patient_results.csv"
    else:
        retrieval_file = f"src/retrieval_query/outputs/{element}/{method}_patient_results.csv"

    response_file = f"src/bedrock_pipeline/bedrock_responses/{task}/{element}/{method}_responses.csv"

    retrieval = pd.read_csv(retrieval_file)
    responses = pd.read_csv(response_file)

    merged = retrieval.merge(
        responses[["SUBJECT_ID", "needle", "needle_in_top_k", "bedrock_response"]],
        on=["SUBJECT_ID", "needle"],
        how="left",
        validate="one_to_one"
    )

    merged["llm_correct"] = merged["bedrock_response"].str.lower().str.contains("yes")

    output_file = f"src/eval/patient_level/{task}/{element}/{method}_eval.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)
    print(f"Saved {output_file}")


def summarize_classify_eval(methods, task, element):
    rows = []
    for method in methods:
        eval_file = f"src/eval/patient_level/{task}/{element}/{method}_eval.csv"
        df = pd.read_csv(eval_file)

        rows.append({
            "element": element,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LLM results and summarize evaluation")
    parser.add_argument("--element", required=True, help="Clinical element to process, e.g., comfort_care")
    args = parser.parse_args()

    element = args.element
    task = "classify"
    methods = [
        "bm25",
        "faiss",
        "faiss_mmr",
        "hybrid",
        "full_context_baseline",
        "semantic_chunking",
        "splade"
    ]

    for method in methods:
        merge_llm_results(task, method, element)

    summarize_classify_eval(methods, task, element)