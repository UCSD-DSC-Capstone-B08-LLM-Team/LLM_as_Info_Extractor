import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_EVAL_DIR = "src/eval"
SAVE_DIR = "src/results/llm_visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

TASKS = ["classify", "extract"]

def load_task_eval(task):
    """
    Load all CSVs for a given task into a dict.

    Args:
        task: "classify" or "extract"

    Returns: {method_name: dataframe}
    """
    base_dir = os.path.join(BASE_EVAL_DIR, task)
    csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
    eval_dict = {}
    for file in csv_files:
        method = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(base_dir, file))
        eval_dict[method] = df
    return eval_dict

def compute_accuracy(task, df):
    """
    Computes overall accuracy based on task and CSV type.

    Args:
        task: "classify" or "extract"
        df: pd.DataFrame loaded from eval CSV
    
    Returns:
        accuracy as a float
    """
    if task == "classify":
        if 'llm_correct' in df.columns:
            return df['llm_correct'].mean()
        else:
            return 0
    else:  # extract
        if 'overall_fuzzy_80_match' in df.columns:
            # summary CSV
            return df['overall_fuzzy_80_match'].iloc[0]
        elif 'fuzzy_match_80' in df.columns:
            # per-needle CSV
            return df['fuzzy_match_80'].mean()
        else:
            return 0

def visualize_task(eval_dict, task):
    """
    Creates a bar plot for a single task.

    Args:
        eval_dict: {method_name: dataframe}
        task: "classify" or "extract"

    Returns:
        None
    """

    summary = []
    for method, df in eval_dict.items():
        acc = compute_accuracy(task, df)
        summary.append({"method": method, "accuracy": acc})
    
    summary_df = pd.DataFrame(summary)
    summary_df.plot(
        kind='bar',
        x='method',
        y='accuracy',
        figsize=(6,4),
        legend=False,
        color='skyblue' if task=="classify" else 'salmon'
    )
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title(f"LLM {task.capitalize()} Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{task}_overall_accuracy.png"))
    plt.close()
    print(f"Saved {task} overall accuracy plot")

def create_heatmap(eval_dicts, methods_to_include=None):
    """
    Creates a combined heatmap for selected tasks and methods.

    Args:
        eval_dicts: {"classify": {method: df}, "extract": {method: df}}
        methods_to_include: list of methods to include in the heatmap (default all)

    Returns:
        None
    """

    summary_rows = []
    for task, methods in eval_dicts.items():
        for method, df in methods.items():
            if methods_to_include and method not in methods_to_include:
                continue
            acc = compute_accuracy(task, df)
            summary_rows.append({"strategy": method, "task": task, "accuracy": acc})
    
    summary_df = pd.DataFrame(summary_rows)
    heatmap_data = summary_df.pivot(index="strategy", columns="task", values="accuracy")
    
    plt.figure(figsize=(6,4))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("LLM Evaluation Accuracy by Task & Retrieval Strategy")
    plt.ylabel("Retrieval Strategy")
    plt.xlabel("Task")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "llm_eval_heatmap.png"), dpi=300)
    plt.close()
    print("Saved combined LLM evaluation heatmap (filtered methods)")

if __name__ == "__main__":
    eval_dicts = {}
    for task in TASKS:
        eval_dicts[task] = load_task_eval(task)
        visualize_task(eval_dicts[task], task)
    
    create_heatmap(eval_dicts, methods_to_include=["bm25_llm_eval", "faiss_cos_llm_eval"])