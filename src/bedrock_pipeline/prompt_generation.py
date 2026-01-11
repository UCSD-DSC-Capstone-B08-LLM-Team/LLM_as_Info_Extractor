import pandas as pd
import os

def generate_bedrock_prompts(results_df, task="classify", output_file=None):
    """
    Generate Bedrock prompts from retrieval results.

    Args:
        results_df (pd.DataFrame): Retrieval results with columns ['needle', 'top_passages', ...].
                                   top_passages should be a list of passages or a stringified list.
        task (str): Type of instruction for Bedrock ('summarize', 'classify', 'extract', etc.).
        output_file (str): Optional path to save prompts CSV. If None, will not save.

    Returns:
        prompts_df: DataFrame with columns ['needle', 'top_passages', 'bedrock_prompt'].
    """
    
    prompts = []
    
    for idx, row in results_df.iterrows():
        needle = row['needle']
        # if top_passages is a string representation of a list, eval it
        passages = row['top_passages']
        if isinstance(passages, str):
            try:
                passages = eval(passages)
            except:
                passages = [passages]

        context = "\n\n".join(passages)

        if task == "summarize":
            prompt = (
                f"Here are clinical notes:\n{context}\n\n"
                f"Summarize the information relevant to the following criteria:\n{needle}"
            )
        elif task == "classify":
            prompt = (
                f"Here are clinical notes:\n{context}\n\n"
                f"Does the patient meet the following criteria? Answer 'Yes' or 'No':\n{needle}"
            )
        elif task == "extract":
            prompt = (
                f"Here are clinical notes:\n{context}\n\n"
                f"Extract the text that matches the following criteria:\n{needle}"
            )
        else:
            prompt = f"Context:\n{context}\n\nNeedle:\n{needle}"

        prompts.append({
            'needle': needle,
            'top_passages': passages,
            'bedrock_prompt': prompt
        })
    
    prompts_df = pd.DataFrame(prompts)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        prompts_df.to_csv(output_file, index=False)
        print(f"Saved Bedrock prompts to {output_file}")

    return prompts_df

if __name__ == "__main__":
    # Load retrieval CSV
    retrieval_file = "src/retrieval/outputs/bm25_mimic_results.csv"
    results_df = pd.read_csv(retrieval_file)

    prompts_df = generate_bedrock_prompts(
        results_df,
        task="classify",
        output_file="src/bedrock_pipeline/bedrock_prompts/classify/bm25_prompts.csv"
    )