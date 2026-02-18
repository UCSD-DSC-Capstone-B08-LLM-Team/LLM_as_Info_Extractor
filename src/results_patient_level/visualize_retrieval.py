import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


# Directories
BASE_RETRIEVAL_DIR = "src/retrieval_query/outputs"
SAVE_DIR = "src/results_patient_level/retrieval_query_visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)


def wrap_labels(labels, width=12):
    return ["\n".join(textwrap.wrap(str(l), width)) for l in labels]

# Helper Functions
def load_retrieval_csv(method):
    """
    Load retrieval CSV for a given method.
    """
    if method == "colbert":
        path = os.path.join(BASE_RETRIEVAL_DIR, f"{method}_patient_results_5.csv")
    else:
        path = os.path.join(BASE_RETRIEVAL_DIR, f"{method}_patient_results.csv")
    return pd.read_csv(path)

def plot_coverage(method_dfs):
    """
    Plot retrieval coverage (found rate) per method.
    """

    coverage = {
        method: df["found"].mean()
        for method, df in method_dfs.items()
    }

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
    plt.title("Retrieval Coverage by Method", fontsize=14)

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

    plt.savefig(os.path.join(SAVE_DIR, "retrieval_coverage.png"), dpi=300)
    plt.close()

    print("Saved retrieval coverage plot")

def plot_summary_table(method_dfs):
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
    summary_df.to_csv(os.path.join(SAVE_DIR, "retrieval_summary_table.csv"), index=False)
    print("Saved summary table CSV")
    return summary_df

def plot_summary_heatmap(summary_df):
    """
    Heatmap of coverage, avg_num_passages, avg_haystack_len.
    """
    heatmap_data = summary_df.set_index("method")[["coverage_rate","avg_num_passages","avg_haystack_len"]]
    plt.figure(figsize=(8,4))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Retrieval Summary Metrics by Method")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "retrieval_summary_heatmap.png"), dpi=300)
    plt.close()
    print("Saved summary heatmap")


def plot_colbert_results(csv_file):
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    plot = sns.scatterplot(
        data=df, 
        x="num_passages", 
        y="found", 
        hue="found", 
        palette={True: "#2ecc71", False: "#e74c3c"},
        alpha=0.6,
        s=100
    )

    # Add a "Recall Rate" line
    # This shows the moving average of success as records get longer
    df['found_int'] = df['found'].astype(int)
    sns.lineplot(
        data=df, 
        x="num_passages", 
        y="found_int", 
        color="#3498db", 
        label="Success Rate Trend"
    )

    plt.title("ColBERT 'Needle in a Haystack' Performance", fontsize=15)
    plt.xlabel("Haystack Size", fontsize=12)
    plt.ylabel("Needle Found? (1 = Yes, 0 = No)", fontsize=12)
    plt.yticks([0, 1], ["Missed", "Found"])
    plt.legend(title="Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "colbert_retrieval.png"), dpi=300)
    plt.close()
    print("Saved colbert retrieval plot")


if __name__ == "__main__":
    methods = ["bm25", "colbert", "faiss_cos", "faiss_euc", "faiss_mmr", "hybrid", "semantic_chunking", "splade"]
    method_dfs = {method: load_retrieval_csv(method) for method in methods}

    # Coverage plot
    plot_coverage(method_dfs)

    # Summary table & heatmap
    summary_df = plot_summary_table(method_dfs)
    plot_summary_heatmap(summary_df)

    # Detailed haystack plot for ColBERT
    plot_colbert_results("src/retrieval_query/outputs/colbert_patient_results_5.csv")
