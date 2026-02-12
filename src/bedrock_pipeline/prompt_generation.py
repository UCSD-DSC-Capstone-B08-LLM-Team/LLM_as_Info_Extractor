import pandas as pd
import os
import ast
import argparse


TASK_TEMPLATES = {
    "classify": (
        "Instruction: Determine if the clinical notes below contain information that answers the 'Query'. "
        "Answer only 'Yes' or 'No'.\n"
        "Input: {context}\n"
        "Query: {query}"
    ),

    "extract": (
        "Instruction: You are a clinical data extractor. Extract the exact text spans from the 'Input' "
        "that answer the 'Query'. Return ONLY the extracted text strings in a list format. "
        "If no text matches, return 'None'.\n"
        "Input: {context}\n"
        "Query: {query}"
    ),

    "summarize": (
        "Instruction: Summarize all information in the 'Input' notes that is relevant to the 'Query'. "
        "Do not include unrelated medical history. Keep the summary concise and clinical.\n"
        "Input: {context}\n"
        "Query: {query}"
    ),
}


def generate_bedrock_prompts(results_df, task):
    """
    Generate Bedrock prompts from retrieval results.

    Args:
        results_df (pd.DataFrame): Retrieval results containing at least
                                   ['needle', 'top_passages'].
        task (str): One of ['classify', 'extract', 'summarize'].

    Returns:
        pd.DataFrame with columns:
        ['needle', 'task', 'top_passages', 'bedrock_prompt']
    """

    if task not in TASK_TEMPLATES:
        raise ValueError(f"Unsupported task '{task}'. Valid tasks: {list(TASK_TEMPLATES.keys())}")

    prompts = []

    for _, row in results_df.iterrows():
        query = row['query']
        needle = row["needle"]
        passages = row["top_passages"]

        # Safely parse list stored as string
        if isinstance(passages, str):
            try:
                passages = ast.literal_eval(passages)
            except Exception:
                passages = [passages]

        context = " ".join(sorted(set(passages), key=lambda x: passages.index(x)))

        # Limit context to around 2000 words
        max_words = 2000
        context_words = context.split()
        if len(context_words) > max_words:
            print(f"Context for SUBJECT_ID {row['SUBJECT_ID']} exceeds {max_words} words. Truncating.")
            context = " ".join(context_words[:max_words])


        prompt = TASK_TEMPLATES[task].format(
            context=context,
            query=query
        )

        prompts.append({
            "SUBJECT_ID": row["SUBJECT_ID"],
            "query": query,
            "needle": needle,
            "task": task,
            "top_passages": passages,
            "bedrock_prompt": prompt
        })

    return pd.DataFrame(prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Bedrock prompts from retrieval outputs")
    parser.add_argument(
        "--retrieval_csv",
        type=str,
        required=True,
        help="Path to retrieval results CSV"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save generated Bedrock prompts CSV"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classify", "extract", "summarize"],
        required=True,
        help="Task type for Bedrock prompting"
    )

    args = parser.parse_args()

    results_df = pd.read_csv(args.retrieval_csv)

    prompts_df = generate_bedrock_prompts(
        results_df=results_df,
        task=args.task
    )

    output_path = os.path.join(args.output_csv,)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prompts_df.to_csv(output_path, index=False)

    print(f"Saved Bedrock prompts to {output_path}")

    # Example usage:
    # python src/bedrock_pipeline/prompt_generation.py \
    #   --retrieval_csv src/retrieval_patient_level/outputs/bm25_patient_results.csv \
    #   --output_csv src/bedrock_pipeline/bedrock_prompts/classify/bm25_prompts.csv \
    #   --task classify
