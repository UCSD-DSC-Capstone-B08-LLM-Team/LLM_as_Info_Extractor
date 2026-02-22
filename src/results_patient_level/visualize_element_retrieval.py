import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot method comparison by clinical element
def plot_method_comparison_by_element(df):
    plt.figure(figsize=(12,6))
    sns.set_theme(style="whitegrid")

    sns.barplot(
        data=df,
        x="element",
        y="recall",
        hue="method"
    )

    plt.ylim(0,1)
    plt.title("Retrieval Recall per Clinical Element")
    plt.ylabel("Recall@k")
    plt.xlabel("Clinical Element")

    plt.legend(title="Method", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig("src/results_patient_level/retrieval_query_visualizations/recall_by_element_grouped.png", dpi=300)

# Function to plot heatmap of recall by retrieval method and element
def plot_recall_heatmap(df):
    pivot = df.pivot(index="method", columns="element", values="recall")

    plt.figure(figsize=(8,5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5
    )

    plt.title("Recall Across Retrieval Methods and Clinical Elements")
    plt.ylabel("Retrieval Method")
    plt.xlabel("Clinical Element")

    plt.tight_layout()
    plt.savefig("src/results_patient_level/retrieval_query_visualizations/recall_heatmap.png", dpi=300)

# Function to plot average recall by method across all elements
def plot_avg_method_performance(df):
    avg_df = (
        df.groupby("method")["recall"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(10,5))
    sns.barplot(data=avg_df, x="method", y="recall")

    plt.ylim(0,1)
    plt.title("Average Recall Across Clinical Elements")
    plt.ylabel("Mean Recall")
    plt.xlabel("Method")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("src/results_patient_level/retrieval_query_visualizations/avg_method_recall.png", dpi=300)

elements = {
    "clinical_trial": "src/results_patient_level/retrieval_query_visualizations/results/clinical_trial_results.csv",
    "comfort_care": "src/results_patient_level/retrieval_query_visualizations/results/comfort_care_results.csv",
    "severe_sepsis": "src/results_patient_level/retrieval_query_visualizations/results/severe_sepsis_results.csv",
    "vasopressor": "src/results_patient_level/retrieval_query_visualizations/results/vasopressor_results.csv"
}

dfs = []
for element, path in elements.items():
    df = pd.read_csv(path)
    df["element"] = element
    dfs.append(df)

all_df = pd.concat(dfs)
full_df = all_df[["element", "Method", "Recall@k"]].rename(
    columns={
        "Method": "method",
        "Recall@k": "recall"
    }
)

plot_method_comparison_by_element(full_df)
plot_recall_heatmap(full_df)
plot_avg_method_performance(full_df)

