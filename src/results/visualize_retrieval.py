import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def compute_and_visualize_retrieval(results_df, top_k=5, strategy_name='BM25', save_dir="src/results/retrieval_visualizations"):
    """
    Compute retrieval stats and save visualizations with strategy name in filenames.

    Args:
        results_df (pd.DataFrame): DataFrame with columns ['needle', 'found', 'CATEGORY', ...]
        top_k (int): Top-K used in retrieval.
        strategy_name (str): Name of the retrieval strategy (for filenames).
        save_dir (str): Directory to save the plots.

    Returns:
        stats_dict: Dictionary with overall, per-category, and per-needle accuracy
    """

    os.makedirs(save_dir, exist_ok=True)

    # Overall accuracy
    overall_acc = results_df['found'].mean()

    # Accuracy per category
    category_acc = results_df.groupby('CATEGORY')['found'].mean()

    # Accuracy per needle
    needle_acc = results_df.groupby('needle')['found'].mean()

    stats_dict = {
        'overall_accuracy': overall_acc,
        'category_accuracy': category_acc,
        'needle_accuracy': needle_acc
    }

    print(f"Overall top-{top_k} retrieval accuracy ({strategy_name}): {overall_acc:.2f}\n")
    print("Per Category Accuracy:")
    print(category_acc, "\n")
    print("Per Needle Accuracy:")
    print(needle_acc, "\n")

    # Plot per-category accuracy
    plt.figure(figsize=(8,5))
    category_acc.plot(kind='bar', color='skyblue')
    plt.ylim(0,1)
    plt.ylabel('Accuracy')
    plt.title(f'Top-{top_k} Retrieval Accuracy by Category ({strategy_name})')
    category_plot_file = os.path.join(save_dir, f'{strategy_name}_top{top_k}_accuracy_by_category.png')
    plt.tight_layout()
    plt.savefig(category_plot_file)
    plt.close()
    print(f"Saved per-category accuracy plot to {category_plot_file}")

    return stats_dict


def compare_strategies_category(dfs_dict, top_k=5, save_dir="src/results/retrieval_visualizations"):
    """
    Compare multiple retrieval strategies by category in a single plot.

    Args:
        dfs_dict: dict of {strategy_name: results_df}
        top_k (int): Top-K used in retrieval.
        save_dir (str): Directory to save the plot.
    
    Returns:
        None
    """

    os.makedirs(save_dir, exist_ok=True)
    combined = pd.DataFrame({name: df.groupby('CATEGORY')['found'].mean() for name, df in dfs_dict.items()})
    combined.plot(kind='bar', figsize=(10,6))
    plt.ylim(0,1)
    plt.ylabel('Accuracy')
    plt.title(f'Top-{top_k} Retrieval Accuracy by Category')
    plt.tight_layout()
    file = os.path.join(save_dir, f"all_strategies_top{top_k}_accuracy_by_category.png")
    plt.savefig(file)
    plt.close()
    print(f"Saved combined category plot to {file}")


def plot_category_strategy_heatmap(results_dict, output_dir="src/results/retrieval_visualizations"):
    """
    Create a Category × Strategy heatmap from multiple retrieval results.

    Args:
        results_dict (dict): Dictionary with strategy name as key and results DataFrame as value.
                             Each DataFrame should have columns ['CATEGORY', 'found', ...].
        output_dir (str): Directory to save the heatmap.

    Returns:
        None
    """

    all_data = []

    # Aggregate per-category accuracy for each strategy
    for strategy, df in results_dict.items():
        per_category = df.groupby('CATEGORY')['found'].mean().reset_index()
        per_category['STRATEGY'] = strategy
        all_data.append(per_category)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Pivot for heatmap
    heatmap_data = combined_df.pivot(index='CATEGORY', columns='STRATEGY', values='found')

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Retrieval Accuracy by Category and Strategy")
    plt.ylabel("Category")
    plt.xlabel("Retrieval Strategy")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "category_strategy_heatmap.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved Category × Strategy heatmap to {output_file}")


def plot_found_not_found(results_dict, output_dir="src/results/retrieval_visualizations"):
    """
    Plot found vs. not found needles for multiple retrieval strategies.

    Args:
        results_dict (dict): Dictionary with strategy name as key and results DataFrame as value.
                             Each DataFrame should have 'found' column (True/False).
        output_dir (str): Directory to save the plot.

    Returns:
        None
    """
    summary = []

    # Aggregate found/not found counts per strategy
    for strategy, df in results_dict.items():
        counts = df['found'].value_counts().to_dict()
        summary.append({
            'strategy': strategy,
            'found': counts.get(True, 0),
            'not_found': counts.get(False, 0)
        })

    summary_df = pd.DataFrame(summary)
    summary_df.set_index('strategy', inplace=True)

    # Plot stacked bar chart
    summary_df.plot(kind='bar', stacked=True, color=['green', 'red'], figsize=(8, 6))
    plt.ylabel("Number of Needles")
    plt.title("Found vs. Not Found Needles per Retrieval Strategy")
    plt.xticks(rotation=45)
    plt.legend(title="Needle Status", loc='upper right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "found_not_found_per_strategy.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved Found/Not Found plot to {output_file}")


if __name__ == "__main__":
    # visualization for bm25
    bm25_file = os.path.join("src", "retrieval", "outputs", "bm25_mimic_results.csv")
    results_df_bm25 = pd.read_csv(bm25_file)
    stats_bm25 = compute_and_visualize_retrieval(results_df_bm25, top_k=2, strategy_name="BM25")

    # visualization for FAISS cosine
    faiss_cos_file = os.path.join("src", "retrieval", "outputs", "faiss_cos_mimic_results.csv")
    results_df_faiss_cos = pd.read_csv(faiss_cos_file)
    stats_faiss_cos = compute_and_visualize_retrieval(results_df_faiss_cos, top_k=2, strategy_name="FAISS_cos")

    # visualization FAISS euclidean
    faiss_euc_file = os.path.join("src", "retrieval", "outputs", "faiss_euc_mimic_results.csv")
    results_df_faiss_euc = pd.read_csv(faiss_euc_file)
    stats_faiss_euc = compute_and_visualize_retrieval(results_df_faiss_euc, top_k=2, strategy_name="FAISS_euc")

    # visualization Hybrid
    hybrid_file = os.path.join("src", "retrieval", "outputs", "hybrid_mimic_results.csv")
    results_df_hybrid = pd.read_csv(hybrid_file)
    stats_hybrid = compute_and_visualize_retrieval(results_df_hybrid, top_k=2, strategy_name="Hybrid")

    # visualization to compare retrieval strategies by category
    compare_strategies_category({
        "BM25": results_df_bm25,
        "FAISS_cos": results_df_faiss_cos,
        "FAISS_euc": results_df_faiss_euc,
        "Hybrid": results_df_hybrid
    }, top_k=2)

    # Create Category × Strategy heatmap
    strategies = {
        "BM25": "src/retrieval/outputs/bm25_mimic_results.csv",
        "FAISS_cos": "src/retrieval/outputs/faiss_cos_mimic_results.csv",
        "FAISS_euc": "src/retrieval/outputs/faiss_euc_mimic_results.csv",
        "Hybrid": "src/retrieval/outputs/hybrid_mimic_results.csv"
    }
    cat_strat_dict = {}
    for strategy, file_path in strategies.items():
        df = pd.read_csv(file_path)
        cat_strat_dict[strategy] = df
    plot_category_strategy_heatmap(cat_strat_dict)

    # compare found vs not found across strategies
    found_not_found_dict = {s: pd.read_csv(f) for s, f in strategies.items()}
    plot_found_not_found(found_not_found_dict)