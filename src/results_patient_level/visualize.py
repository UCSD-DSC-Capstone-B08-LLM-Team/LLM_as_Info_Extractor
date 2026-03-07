import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

output_dir = "src/results_patient_level/llm_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Read and combine
csv_files = [
    "src/eval/patient_level/classify/clinical_trial/all_methods_llm_eval.csv",
    "src/eval/patient_level/classify/comfort_care/all_methods_llm_eval.csv",
    "src/eval/patient_level/classify/contra_care/all_methods_llm_eval.csv",
    "src/eval/patient_level/classify/severe_sepsis/all_methods_llm_eval.csv", 
    "src/eval/patient_level/classify/vasopressor/all_methods_llm_eval.csv"
]
dfs = []
for file in csv_files:
    df_temp = pd.read_csv(file)
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)


# Scatter Plot (Retrieval Success vs LLM Accuracy)
# how retrieval quality affects LLM performance.
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="retrieval_success",
    y="accuracy",
    hue="strategy",
    style="element",
    # hue="element",
    # style="strategy",
    s=140,
    alpha=0.8
)

sns.regplot(
    data=df,
    x="retrieval_success",
    y="accuracy",
    scatter=False,
    color="black"
)

plt.xlabel("Retrieval Success")
plt.ylabel("LLM Accuracy")
plt.title("Retrieval Performance vs LLM Accuracy")

plt.grid(True)
plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout()

plt.savefig(f"{output_dir}/retrieval_vs_llm_accuracy_scatter.png", dpi=300)
plt.close()

# Grouped Bar Chart (Method Comparison)
# Shows retrieval vs LLM accuracy side-by-side.
import numpy as np

elements = ["comfort_care", "contra_care", "severe_sepsis", "vasopressor"]
for element in elements:
    df_plot = df[df["element"] == element]

    x = np.arange(len(df_plot["strategy"]))
    width = 0.35

    plt.figure(figsize=(10,6))

    plt.bar(x - width/2, df_plot["retrieval_success"], width, label="Retrieval Success")
    plt.bar(x + width/2, df_plot["accuracy"], width, label="LLM Accuracy")

    plt.xticks(x, df_plot["strategy"], rotation=45)
    plt.ylabel("Score")
    plt.title(f"Retrieval vs LLM Accuracy ({element})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{element}_retrieval_vs_accuracy.png", dpi=300)
    plt.close()


# Heatmap 
# Shows method performance across all elements.

# LLM accuracy heatmap
pivot_acc = df.pivot(
    index="strategy",
    columns="element",
    values="accuracy"
)

plt.figure(figsize=(10,6))

sns.heatmap(
    pivot_acc,
    annot=True,
    fmt=".2f",
    cmap="Blues"
)

plt.title("LLM Accuracy Across Retrieval Methods and Elements")
plt.ylabel("Retrieval Strategy")
plt.xlabel("Clinical Element")

plt.tight_layout()
plt.savefig(f"{output_dir}/llm_accuracy_heatmap.png", dpi=300)
plt.close()


# retrieval success heatmap
pivot_ret = df.pivot(
    index="strategy",
    columns="element",
    values="retrieval_success"
)

plt.figure(figsize=(10,6))

sns.heatmap(
    pivot_ret,
    annot=True,
    fmt=".2f",
    cmap="Greens"
)

plt.title("Retrieval Success Across Methods and Elements")
plt.ylabel("Retrieval Strategy")
plt.xlabel("Clinical Element")

plt.tight_layout()
plt.savefig(f"{output_dir}/retrieval_success_heatmap.png", dpi=300)
plt.close()


# Performance Gap Plot
# This shows how much LLM performance drops after retrieval
# positive gap then retrieval is better- LLM failed even when correct passage was found
# negative gap then LLM is better- retrieval failed but LLM randomly got it right anyway
df["gap"] = df["retrieval_success"] - df["accuracy"]

plt.figure(figsize=(10,6))

sns.barplot(
    data=df,
    x="strategy",
    y="gap",
    hue="element"
)

plt.xticks(rotation=45)
plt.ylabel("Retrieval - Accuracy Gap")
plt.title("Performance Gap Between Retrieval and LLM")

plt.tight_layout()
plt.savefig(f"{output_dir}/retrieval_llm_gap.png", dpi=300)
plt.close()


# Mean LLM accuracy per retrieval method
# which retrieval strategies lead to better LLM performance on average, regardless of element?
method_perf = df.groupby("strategy")["accuracy"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,6))

sns.barplot(
    x=method_perf.index,
    y=method_perf.values
)

plt.xticks(rotation=45)
plt.ylabel("Mean LLM Accuracy")
plt.title("Average LLM Performance by Retrieval Strategy")

plt.tight_layout()
plt.savefig(f"{output_dir}/mean_strategy_accuracy.png", dpi=300)
plt.close()


# Scatter Plot (Retrieval Methods Only)
df_retrieval = df[df["strategy"] != "full_context_baseline"]

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df_retrieval,
    x="retrieval_success",
    y="accuracy",
    hue="strategy",
    style="element",
    s=140,
    alpha=0.85
)

sns.regplot(
    data=df_retrieval,
    x="retrieval_success",
    y="accuracy",
    scatter=False,
    color="black"
)

plt.xlabel("Retrieval Success")
plt.ylabel("LLM Accuracy")
plt.title("Retrieval Performance vs LLM Accuracy (Retrieval Methods Only)")

plt.grid(True)
plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout()

plt.savefig(f"{output_dir}/retrieval_methods_scatter.png", dpi=300)
plt.close()


# Scatter Plot (Baseline Comparison)
df["type"] = df["strategy"].apply(
    lambda x: "Baseline (Full Context)" if x == "full_context_baseline" else "Retrieval Method"
)

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="retrieval_success",
    y="accuracy",
    hue="type",
    style="element",
    s=160,
    alpha=0.85
)

plt.xlabel("Retrieval Success")
plt.ylabel("LLM Accuracy")
plt.title("Baseline vs Retrieval-Based Methods")

plt.grid(True)
plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout()

plt.savefig(f"{output_dir}/baseline_vs_retrieval_scatter.png", dpi=300)
plt.close()


# Improvement Over Baseline Plot
# Create a copy so we don’t modify original
df_improve = df.copy()

# Get baseline accuracy per element
baseline_acc = df_improve[df_improve["strategy"] == "full_context_baseline"] \
    .set_index("element")["accuracy"].to_dict()

# Compute improvement for each retrieval method (exclude baseline)
df_improve = df_improve[df_improve["strategy"] != "full_context_baseline"]
df_improve["accuracy_over_baseline"] = df_improve.apply(
    lambda row: row["accuracy"] - baseline_acc[row["element"]],
    axis=1
)

# bar plot of improvement over baseline by method and element
plt.figure(figsize=(10,6))

sns.barplot(
    data=df_improve,
    x="strategy",
    y="accuracy_over_baseline",
    hue="element"
)

plt.xticks(rotation=45)
plt.ylabel("Accuracy Improvement vs Baseline")
plt.title("How Retrieval Methods Improve LLM Accuracy Over Full-Context Baseline")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(bbox_to_anchor=(1.05,1))
plt.axhline(0, color="black", linestyle="--")
plt.tight_layout()

plt.savefig(f"{output_dir}/improvement_vs_baseline.png", dpi=300)
plt.close()