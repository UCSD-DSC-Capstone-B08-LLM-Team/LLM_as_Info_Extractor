import pandas as pd
import os
from rapidfuzz import fuzz

# Row-level evaluation
def eval_summary_row(row, needle_thresh=70, context_thresh=60):
    needle = str(row.get("needle", "") or "")
    summary = str(row.get("bedrock_response", "") or "")
    context = str(row.get("top_passages", "") or "")

    # Handle empty / None summaries
    if summary.strip() == "":
        return pd.Series({
            "needle_sim": 0,
            "context_sim": 0,
            "needle_covered": False,
            "faithful": False,
            "focused": False
        })

    needle_sim = fuzz.token_set_ratio(needle, summary)
    context_sim = fuzz.token_set_ratio(summary, context)

    needle_covered = needle_sim >= needle_thresh
    faithful = context_sim >= context_thresh
    focused = needle_covered and faithful

    return pd.Series({
        "needle_sim": needle_sim,
        "context_sim": context_sim,
        "needle_covered": needle_covered,
        "faithful": faithful,
        "focused": focused
    })


# Per-method evaluation
def evaluate_summarize_method(method, input_path, output_dir):
    df = pd.read_csv(input_path)

    eval_df = df.apply(eval_summary_row, axis=1)
    df_eval = pd.concat([df, eval_df], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    per_method_file = os.path.join(output_dir, f"{method}_eval.csv")
    df_eval.to_csv(per_method_file, index=False)

    summary = {
        "strategy": method,
        "num_patients": len(df_eval),
        # Proportion of summaries that semantically mention the needle
        "needle_coverage_rate": df_eval["needle_covered"].mean(),
        # Proportion of summaries supported by retrieved context
        "faithfulness_rate": df_eval["faithful"].mean(),
        # Proportion of summaries both needle-covered and faithful
        "focused_summary_rate": df_eval["focused"].mean(),
        # Average similarity scores between needle and summary
        "avg_needle_sim": df_eval["needle_sim"].mean(),
        # Average similarity scores between retrieved context and summary
        "avg_context_sim": df_eval["context_sim"].mean()
    }

    print(f"Saved detailed eval → {per_method_file}")
    return summary


# Run all methods
def evaluate_all_methods():
    methods = ["bm25", "faiss_cos", "faiss_euc", "hybrid"]
    summaries = []

    for method in methods:
        input_path = f"src/bedrock_pipeline/bedrock_responses/summarize/{method}_responses.csv"
        output_dir = "src/eval/patient_level/summarize"

        summary = evaluate_summarize_method(
            method=method,
            input_path=input_path,
            output_dir=output_dir
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_file = "src/eval/patient_level/summarize/all_methods_llm_eval.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n=== SUMMARY METRICS ===")
    print(summary_df)
    print(f"\nSaved summary table → {summary_file}")


if __name__ == "__main__":
    evaluate_all_methods()