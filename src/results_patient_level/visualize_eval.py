import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Directories
BASE_EVAL_DIR = "src/eval/patient_level"
SAVE_DIR = "src/results_patient_level/llm_visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)


# Helper functions
def load_eval(task):
    """
    Load all-method evaluation CSVs for a task.
    
    Args:
        task (str): "classify", "extract", or "summarize"

    Returns:
        pd.DataFrame
    """
    
    file_map = {
        "classify": "all_methods_llm_eval.csv",
        "extract": "all_methods_llm_eval.csv",
        "summarize": "all_methods_llm_eval.csv"
    }
    path = os.path.join(BASE_EVAL_DIR, task, file_map[task])
    return pd.read_csv(path)

def plot_classify(df):
    plt.figure(figsize=(6,4))
    plt.bar(df["strategy"], df["accuracy"], color='skyblue')
    plt.ylim(0,1)
    plt.ylabel("LLM Accuracy")
    plt.xlabel("Retrieval Strategy")
    plt.title("Classification Accuracy by Retrieval Method")
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "classify_accuracy_by_method.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")

def plot_extract(df):
    x = range(len(df))
    width = 0.35
    plt.figure(figsize=(7,4))
    plt.bar(x, df["overall_exact_match"], width, label="Exact Match")
    plt.bar([i + width for i in x], df["overall_fuzzy_80_match"], width, label="Fuzzy ≥80")
    plt.xticks([i + width/2 for i in x], df["strategy"])
    plt.ylim(0,1)
    plt.ylabel("Match Rate")
    plt.xlabel("Retrieval Strategy")
    plt.title("Extraction Performance by Retrieval Method")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "extract_exact_vs_fuzzy.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")

def plot_summarize(df):
    # Quality metrics (line plot)
    rate_cols = ["needle_coverage_rate", "faithfulness_rate", "focused_summary_rate"]
    df_rates = df.melt(id_vars="strategy", value_vars=rate_cols, var_name="metric", value_name="value")
    plt.figure(figsize=(8,4))
    for metric in rate_cols:
        subset = df_rates[df_rates["metric"] == metric]
        plt.plot(subset["strategy"], subset["value"], marker="o", label=metric)
    plt.ylim(0,1)
    plt.ylabel("Rate")
    plt.xlabel("Retrieval Strategy")
    plt.title("Summarization Quality Metrics")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "summarize_quality_rates.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")

    # Similarity metrics
    plt.figure(figsize=(6,4))
    plt.plot(df["strategy"], df["avg_needle_sim"], marker="o", label="Needle Similarity")
    plt.plot(df["strategy"], df["avg_context_sim"], marker="o", label="Context Similarity")
    plt.ylabel("Similarity Score")
    plt.xlabel("Retrieval Strategy")
    plt.title("Summarization Similarity Scores")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "summarize_similarity_scores.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


# Combined Heatmap
def build_combined_metrics():
    rows = []

    # classify 
    classify = load_eval("classify")
    for _, r in classify.iterrows():
        rows.append({"strategy": r["strategy"], "metric": "Classify\nAccuracy", "value": r["accuracy"]})

    # extract
    extract = load_eval("extract")
    for _, r in extract.iterrows():
        rows.append({"strategy": r["strategy"], "metric": "Extract\nFuzzy ≥80", "value": r["overall_fuzzy_80_match"]})

    # summarize
    summarize = load_eval("summarize")
    for _, r in summarize.iterrows():
        rows.append({"strategy": r["strategy"], "metric": "Summarize\nNeedle Coverage", "value": r["needle_coverage_rate"]})

    return pd.DataFrame(rows)

def normalize_within_metric(df):
    df = df.copy()
    df["relative"] = df.groupby("metric")["value"].transform(lambda x: x / x.max())
    return df

def plot_relative_heatmap(df):
    heatmap_rel = df.pivot(index="strategy", columns="metric", values="relative")
    heatmap_abs = df.pivot(index="strategy", columns="metric", values="value")

    plt.figure(figsize=(8,4))
    sns.heatmap(
        heatmap_rel,
        annot=heatmap_abs,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Relative to Best in Task"},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("LLM Performance by Task (Relative Coloring, Absolute Scores)")
    plt.ylabel("Retrieval Strategy")
    plt.xlabel("Task / Metric")
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "llm_relative_task_heatmap.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    # Individual task plots
    plot_classify(load_eval("classify"))
    plot_extract(load_eval("extract"))
    plot_summarize(load_eval("summarize"))

    # Combined heatmap
    df_combined = build_combined_metrics()
    df_norm = normalize_within_metric(df_combined)
    plot_relative_heatmap(df_norm)