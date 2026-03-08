import pandas as pd
import os
import ast
import argparse
import tiktoken


MAX_INPUT_TOKENS = 163840        # DeepSeek v3 max input
SAFETY_BUFFER = 2000             # Safety buffer to avoid overflow
TOKENIZER_NAME = "cl100k_base"   # tiktoken encoding


TASK_TEMPLATES = {
    "classify": (
        "Instruction: Determine if the clinical notes below contain information that answers the 'Query'. "
        "Answer only 'Yes' or 'No'.\n"
        "Input: {context}\n"
        "Query: {query}"
    )
}


# Tokenizer setup
enc = tiktoken.get_encoding(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    """Count the number of tokens in a given text."""
    return len(enc.encode(text))


def trim_to_token_limit(text: str, max_tokens: int) -> str:
    """Trim text to a max number of tokens."""
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens])


def pack_passages_to_tokens(passages, available_tokens):
    """
    Concatenate passages up to the available token limit.
    Stops before exceeding max tokens.
    """
    packed = []
    used_tokens = 0

    for p in passages:
        p_tokens = count_tokens(p)
        if used_tokens + p_tokens > available_tokens:
            break
        packed.append(p)
        used_tokens += p_tokens

    return " ".join(packed)


# Prompt generation
def generate_bedrock_prompts(results_df, task):
    """
    Generate Bedrock prompts from retrieval results.

    Args:
        results_df (pd.DataFrame): Retrieval results containing at least
                                   ['needle', 'top_passages'].
        task (str): 'classify'

    Returns:
        pd.DataFrame with columns:
        - 'SUBJECT_ID': Identifier for the patient/query
        - 'query': The original query/needle
        - 'needle': The clinical scenario or information need
        - 'task': The task type (classify)
        - 'top_passages': List of top retrieved passages
        - 'needle_in_top_k': Boolean indicating if needle is in any top passage 
        - 'bedrock_prompt': The final prompt formatted for Bedrock
        - 'tokens_used': Number of tokens in the final prompt
    """
    if task not in TASK_TEMPLATES:
        raise ValueError(f"Unsupported task '{task}'. Valid tasks: {list(TASK_TEMPLATES.keys())}")

    prompts = []

    for _, row in results_df.iterrows():
        query = row["query"]
        needle = row["needle"]
        passages = row["top_passages"]

        # Safely parse list stored as string
        if isinstance(passages, str):
            try:
                passages = ast.literal_eval(passages)
            except Exception:
                passages = [passages]

        # Preserve original order; remove duplicates
        passages = [p for i, p in enumerate(passages) if p not in passages[:i]]

        # check if needle exists in any of the top passages 
        needle_in_top_k = any(needle in p for p in passages)

        template = TASK_TEMPLATES[task]

        # Tokens used by template and query
        base_prompt = template.format(context="", query=query)
        overhead_tokens = count_tokens(base_prompt)

        available_tokens_for_context = MAX_INPUT_TOKENS - overhead_tokens - SAFETY_BUFFER
        if available_tokens_for_context <= 0:
            raise ValueError(
                f"Prompt overhead too large for SUBJECT_ID {row.get('SUBJECT_ID', 'unknown')}"
            )

        # Pack passages up to available token budget
        context = pack_passages_to_tokens(passages, available_tokens_for_context)

        # Build final prompt
        prompt = template.format(context=context, query=query)
        tokens_used = count_tokens(prompt)

        prompts.append({
            "SUBJECT_ID": row["SUBJECT_ID"],
            "query": query,
            "needle": needle,
            "task": task,
            "top_passages": passages,
            "needle_in_top_k": needle_in_top_k,
            "bedrock_prompt": prompt,
            "tokens_used": tokens_used
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
        choices=["classify"],
        required=True,
        help="Task type for Bedrock prompting"
    )

    args = parser.parse_args()

    results_df = pd.read_csv(args.retrieval_csv)

    prompts_df = generate_bedrock_prompts(
        results_df=results_df,
        task=args.task
    )

    output_path = args.output_csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prompts_df.to_csv(output_path, index=False)

    print(f"Saved Bedrock prompts to {output_path}")
    print(f"Example: SUBJECT_ID {prompts_df.iloc[0]['SUBJECT_ID']}, "
          f"needle_in_top_k = {prompts_df.iloc[0]['needle_in_top_k']}, "
          f"tokens_used = {prompts_df.iloc[0]['tokens_used']}")
