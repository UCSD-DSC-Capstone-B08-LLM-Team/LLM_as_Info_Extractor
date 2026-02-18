import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_EVAL_DIR = "src/eval/patient_level"
SAVE_DIR = "src/results_patient_level/llm_visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_eval(task):
    file_map = {
        "classify": "all_methods_llm_eval.csv",
        "extract": "all_methods_llm_eval.csv",
        "summarize": "all_methods_llm_eval.csv",
    }
    path = os.path.join(BASE_EVAL_DIR, task, file_map[task])
    return pd.read_csv(path)


def wrap_labels(labels, width=14):
    return ["\n".join(textwrap.wrap(str(l), width)) for l in labels]


# =============================
# CLASSIFICATION
# =============================
def plot_classify(df):
    plt.figure(figsize=(12,5))

    labels = wrap_labels(df["strategy"])

    plt.bar(labels, df["accuracy"])
    plt.ylim(0,1)

    plt.ylabel("LLM Accuracy")
    plt.xlabel("Retrieval Strategy")
    plt.title("Classification Accuracy by Retrieval Method")

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    out = os.path.join(SAVE_DIR, "classify_accuracy_by_method.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


# =============================
# EXTRACTION
# =============================
def plot_extract(df):
    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(12,5))

    plt.bar(x, df["overall_exact_match"], width, label="Exact Match")
    plt.bar([i + width for i in x], df["overall_fuzzy_80_match"], width, label="Fuzzy ≥80")

    labels = wrap_labels(df["strategy"])

    plt.xticks([i + width/2 for i in x], labels, rotation=30, ha="right")

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


# =============================
# SUMMARIZATION
# =============================
def plot_summarize(df):

    # ---------- QUALITY METRICS ----------
    rate_cols = [
        "needle_coverage_rate",
        "faithfulness_rate",
        "focused_summary_rate",
    ]

    df_rates = df.melt(
        id_vars="strategy",
        value_vars=rate_cols,
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(12,5))

    for metric in rate_cols:
        subset = df_rates[df_rates["metric"] == metric]
        plt.plot(
            wrap_labels(subset["strategy"]),
            subset["value"],
            marker="o",
            label=metric,
        )

    plt.ylim(0,1)
    plt.ylabel("Rate")
    plt.xlabel("Retrieval Strategy")
    plt.title("Summarization Quality Metrics")
    plt.legend()

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    out = os.path.join(SAVE_DIR, "summarize_quality_rates.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")

    # ---------- SIMILARITY METRICS ----------
    plt.figure(figsize=(12,5))

    labels = wrap_labels(df["strategy"])

    plt.plot(labels, df["avg_needle_sim"], marker="o", label="Needle Similarity")
    plt.plot(labels, df["avg_context_sim"], marker="o", label="Context Similarity")

    plt.ylabel("Similarity Score")
    plt.xlabel("Retrieval Strategy")
    plt.title("Summarization Similarity Scores")
    plt.legend()

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    out = os.path.join(SAVE_DIR, "summarize_similarity_scores.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


# =============================
# HEATMAP
# =============================
def build_combined_metrics():
    rows = []

    classify = load_eval("classify")
    for _, r in classify.iterrows():
        rows.append({
            "strategy": r["strategy"],
            "metric": "Classify\nAccuracy",
            "value": r["accuracy"],
        })

    extract = load_eval("extract")
    for _, r in extract.iterrows():
        rows.append({
            "strategy": r["strategy"],
            "metric": "Extract\nFuzzy ≥80",
            "value": r["overall_fuzzy_80_match"],
        })

    summarize = load_eval("summarize")
    for _, r in summarize.iterrows():
        rows.append({
            "strategy": r["strategy"],
            "metric": "Summarize\nNeedle Coverage",
            "value": r["needle_coverage_rate"],
        })

    return pd.DataFrame(rows)


def normalize_within_metric(df):
    df = df.copy()
    df["relative"] = df.groupby("metric")["value"].transform(lambda x: x / x.max())
    return df


def plot_relative_heatmap(df):

    heatmap_rel = df.pivot(index="strategy", columns="metric", values="relative")
    heatmap_abs = df.pivot(index="strategy", columns="metric", values="value")

    plt.figure(figsize=(10,5))

    sns.heatmap(
        heatmap_rel,
        annot=heatmap_abs,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Relative to Best in Task"},
        linewidths=0.5,
        linecolor="gray",
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

    plot_classify(load_eval("classify"))
    plot_extract(load_eval("extract"))
    plot_summarize(load_eval("summarize"))

    df_combined = build_combined_metrics()
    df_norm = normalize_within_metric(df_combined)
    plot_relative_heatmap(df_norm)