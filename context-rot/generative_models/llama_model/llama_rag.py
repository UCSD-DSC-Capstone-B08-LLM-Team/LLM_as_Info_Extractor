import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch
from tqdm import tqdm

# LLaMA generative model setup
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
device = model.device

# Answer generation function
def generate_answer(question: str, context: str, max_input_tokens: int = 500, max_new_tokens: int = 150):
    """
    Generate answer using LLaMA given a question and retrieved snippet.
    Args:
        question: The input question string.
        context: The retrieved document text string.
        max_input_tokens: Maximum number of input tokens to keep.
        max_new_tokens: Maximum number of new tokens to generate.
    Returns:
        answer: The generated answer string.
    """
    # tokenize context and truncate if too long
    context_tokens = tokenizer(context, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    context_text = tokenizer.decode(context_tokens['input_ids'][0], skip_special_tokens=True)
    
    prompt = f"Question: {question}\nContext: {context_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ----------------------------
# Main RAG script
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True, help="Input retrieval_results.csv")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV with generated answers")
    parser.add_argument("--methods", nargs="+", default=["bm25"], help="Retrieval methods to generate answers for")
    parser.add_argument("--max-snippet-tokens", type=int, default=500, help="Max tokens for retrieved snippet")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Max tokens for generated answer")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Add columns for generated answers
    for method in args.methods:
        df[f"generated_answer_{method}"] = ""

    print(f"Total rows: {len(df)}")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        for method in args.methods:
            idx_col = f"retrieved_idx_{method}"
            if idx_col not in row or pd.isna(row[idx_col]):
                continue

            retrieved_idx = int(row[idx_col])
            retrieved_text = df.loc[retrieved_idx, "prompt"]

            # Generate answer using only the snippet
            answer = generate_answer(
                question, 
                retrieved_text, 
                max_input_tokens=args.max_snippet_tokens,
                max_new_tokens=args.max_new_tokens
            )
            df.at[idx, f"generated_answer_{method}"] = answer

    df.to_csv(args.output_csv, index=False)
    print(f"Saved RAG answers to {args.output_csv}")

if __name__ == "__main__":
    main()

# Example command to run the script:

# python llama_model/llama_rag.py \
#     --input-csv retrieval_results.csv \
#     --output-csv llama_model/bm25_rag_results.csv \
#     --methods bm25
