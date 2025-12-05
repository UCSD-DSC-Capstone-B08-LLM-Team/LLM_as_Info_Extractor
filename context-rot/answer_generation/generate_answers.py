import os
import ast
import time
import pandas as pd
from openai import OpenAI

# ---------- CONFIG ----------

# 1) Where your CSVs live
INPUT_FILES = {
    "bm25": "../retrieval/bm25/bm25_retrieval_results.csv",
    "faiss_cos": "../retrieval/faiss_cos/faiss_cos_retrieval_results.csv",
    "faiss_euc": "../retrieval/faiss_euc/faiss_euc_retrieval_results.csv",
    "hybrid_bm25_faiss": "../retrieval/hybrid_bm25_faiss/hybrid_bm25_faiss_retrieval_results.csv",
}

# 2) Where to save generated answer CSVs
OUTPUT_DIR = "generated_answers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3) LLM config – change model name to whatever you’re allowed to use
client = OpenAI()  # expects OPENAI_API_KEY in env
MODEL_NAME = "gpt-4o-mini"

# Optional: limit rows while testing so you don't burn a ton of tokens accidentally
MAX_ROWS_PER_FILE = None  # e.g. set to 5 while testing, then None for all rows


# ---------- HELPER FUNCTIONS ----------

def reorder_passages(passages):
    """
    Reorder passages to help with 'lost in the middle':
    - Keep best passage (index 0) first
    - Other middle passages stay in the middle (2,3,...)
    - Put second-best (index 1) last
    """
    if len(passages) <= 2:
        return passages  # nothing fancy to do
    return [passages[0]] + passages[2:] + [passages[1]]


def build_prompt(question, passages):
    """
    Build the user prompt for the LLM.
    You can tweak wording later if you want to experiment.
    """
    context_blocks = []
    for i, p in enumerate(passages, start=1):
        context_blocks.append(f"Passage {i}:\n{p.strip()}\n")

    context_text = "\n".join(context_blocks)

    prompt = f"""
You are helping with a needle-in-a-haystack evaluation.

You are given:
- A question
- Several passages retrieved from a larger document

Using ONLY the information in the passages, answer the question as precisely as possible.
If the answer is not contained in the passages, reply with exactly: NOT_FOUND

Question:
{question.strip()}

Retrieved passages:
{context_text}

Now provide your answer in 1–3 sentences (or NOT_FOUND).
""".strip()

    return prompt


def call_llm(prompt):
    """
    Single call to the chat model.
    If you're in a restricted environment, you'll replace this
    with whatever LLM interface you have there.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful assistant that answers questions "
                    "ONLY using the provided passages."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def process_retrieval_file(retrieval_name, input_path):
    """
    For one retrieval method:
    - load the CSV
    - parse & reorder passages
    - build prompts + get LLM answers
    - write output CSV
    """
    print(f"\n=== Processing {retrieval_name}: {input_path} ===")
    df = pd.read_csv(input_path)

    # Storage for answers and reordered context
    llm_answers = []
    reordered_passages_list = []

    n_rows = len(df) if MAX_ROWS_PER_FILE is None else min(len(df), MAX_ROWS_PER_FILE)

    for idx, row in df.iloc[:n_rows].iterrows():
        question = row["question"]
        top_passages_str = row["top_passages"]

        # Parse the stringified list -> Python list
        try:
            passages = ast.literal_eval(top_passages_str)
        except Exception as e:
            print(f"Row {idx}: error parsing top_passages – {e}")
            passages = []

        # Reorder passages (Leah's suggestion)
        reordered = reorder_passages(passages)
        reordered_passages_list.append(reordered)

        # Build prompt + call LLM
        prompt = build_prompt(question, reordered)
        answer = call_llm(prompt)
        llm_answers.append(answer)

        print(f"[{retrieval_name}] Row {idx} done.")

        # If you're worried about rate limits, you can uncomment:
        # time.sleep(0.2)

    # Attach results to dataframe (only the rows we actually processed)
    df_out = df.iloc[:n_rows].copy()
    df_out["reordered_passages"] = reordered_passages_list
    df_out["llm_answer"] = llm_answers
    df_out["retrieval_method"] = retrieval_name

    # Save
    output_path = os.path.join(
        OUTPUT_DIR, f"generated_answers_{retrieval_name}.csv"
    )
    df_out.to_csv(output_path, index=False)
    print(f"Saved {retrieval_name} answers to: {output_path}")


# ---------- MAIN LOOP ----------

for retrieval_name, input_path in INPUT_FILES.items():
    if not os.path.exists(input_path):
        print(f"WARNING: {input_path} not found, skipping.")
        continue
    process_retrieval_file(retrieval_name, input_path)

print("\nAll retrieval methods processed.")
