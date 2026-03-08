import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import argparse


def wrap_labels(labels, width=12):
    return ["\n".join(textwrap.wrap(str(l), width)) for l in labels]

def load_retrieval_csv(method, base_dir):
    """
    Load retrieval CSV for a given method and element directory.
    """
    path = os.path.join(base_dir, f"{method}_patient_results.csv")
    return pd.read_csv(path)

def plot_coverage(method_dfs, save_dir):
    """
    Plot retrieval coverage (found rate) per method.
    """
    coverage = {method: df["found"].mean() for method, df in method_dfs.items()}
    cov_df = (
        pd.DataFrame.from_dict(coverage, orient="index", columns=["coverage"])
        .sort_values("coverage", ascending=False)
        .reset_index()
        .rename(columns={"index": "method"})
    )

    plt.figure(figsize=(12,5))
    sns.set_theme(style="whitegrid")

    bars = plt.bar(
        wrap_labels(cov_df["method"]),
        cov_df["coverage"]
    )

    plt.ylim(0, 1)
    plt.ylabel("Fraction of Patients with Needle Found", fontsize=12)
    plt.xlabel("Retrieval Method", fontsize=12)
    plt.title("Clinical Trial Retrieval Coverage by Method", fontsize=14)
    plt.xticks(rotation=25, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "retrieval_coverage.png"), dpi=300)
    plt.close()
    print("Saved retrieval coverage plot")

def plot_summary_table(method_dfs, save_dir):
    """
    Create a summary table with coverage, avg num_passages, avg haystack length.
    """
    summary = []
    for method, df in method_dfs.items():
        summary.append({
            "method": method,
            "num_patients": len(df),
            "coverage_rate": df['found'].mean(),
            "avg_num_passages": df['num_passages'].mean(),
            "avg_haystack_len": df['haystack_len_chars'].mean()
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(save_dir, "retrieval_summary_table.csv"), index=False)
    print("Saved summary table CSV")
    return summary_df

def plot_summary_heatmap(summary_df, save_dir):
    """
    Heatmap of coverage, avg_num_passages, avg_haystack_len.
    """
    heatmap_data = summary_df.set_index("method")[["coverage_rate","avg_num_passages","avg_haystack_len"]]
    plt.figure(figsize=(8,4))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Clinical Trial Retrieval Summary Metrics by Method")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "retrieval_summary_heatmap.png"), dpi=300)
    plt.close()
    print("Saved summary heatmap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize retrieval results for a clinical element")
    parser.add_argument("--element", required=True, help="Clinical element, e.g., 'comfort_care'")
    args = parser.parse_args()

    element = args.element
    BASE_RETRIEVAL_DIR = f"src/retrieval_query/outputs/{element}"
    SAVE_DIR = f"src/results_patient_level/retrieval_query_visualizations/{element}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    methods = ["bm25", "faiss_cos", "faiss_mmr", "hybrid", "semantic_chunking", "splade"]
    method_dfs = {method: load_retrieval_csv(method, BASE_RETRIEVAL_DIR) for method in methods}

    # Coverage plot
    plot_coverage(method_dfs, SAVE_DIR)

    # Summary table & heatmap
    summary_df = plot_summary_table(method_dfs, SAVE_DIR)
    plot_summary_heatmap(summary_df, SAVE_DIR)