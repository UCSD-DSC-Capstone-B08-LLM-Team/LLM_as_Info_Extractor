import pandas as pd
import boto3
import json
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def infer_task_and_retrieval(prompt_csv_path):
    parts = prompt_csv_path.replace("\\", "/").split("/")
    task = parts[-2]
    filename = os.path.basename(prompt_csv_path)
    retrieval = filename.split("_prompts")[0]
    return task, retrieval

def call_bedrock_row_with_retry(client, prompt, max_retries=5):
    """Call Bedrock for a single prompt with retry for throttling."""
    wait = 1  # initial backoff in seconds
    for attempt in range(max_retries):
        try:
            request_body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.0
            }
            resp = client.invoke_model(
                modelId="deepseek.v3-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            response_body = json.loads(resp["body"].read())
            if "choices" in response_body:
                return response_body["choices"][0]["message"]["content"]
            elif "outputs" in response_body:
                return response_body["outputs"][0]["text"]
            else:
                return str(response_body)

        except Exception as e:
            if "ThrottlingException" in str(e):
                print(f"Throttled, retrying in {wait}s... (attempt {attempt+1})")
                time.sleep(wait)
                wait *= 2  # exponential backoff
            else:
                return f"ERROR: {e}"

    return f"ERROR: ThrottlingException after {max_retries} retries"

def call_bedrock_and_save_parallel(prompt_csv, output_csv=None, max_workers=10, sleep_time=0.0, region="us-west-2"):
    df = pd.read_csv(prompt_csv)
    client = boto3.client("bedrock-runtime", region_name=region)
    task, retrieval = infer_task_and_retrieval(prompt_csv)

    if output_csv is None:
        output_csv = f"src/bedrock_pipeline/bedrock_responses/{task}/{retrieval}_responses.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    responses = [None] * len(df)

    failed_indices = []

    # Submit jobs to thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(call_bedrock_row_with_retry, client, row["bedrock_prompt"]): idx
            for idx, row in df.iterrows()
        }

        # Progress bar
        for future in tqdm(as_completed(future_to_idx), total=len(df), desc="Calling Bedrock"):
            idx = future_to_idx[future]
            result = future.result()
            responses[idx] = result
            if "ERROR" in str(result):
                failed_indices.append(idx)
            print(f"Patient {idx}: {result}")

            if sleep_time > 0:
                time.sleep(sleep_time)

    df["bedrock_response"] = responses
    df.to_csv(output_csv, index=False)
    print(f"\nSaved Bedrock responses to {output_csv}")

    if failed_indices:
        print(f"\nFailed rows: {failed_indices}")
        # Optional: save a CSV with only failed prompts to rerun
        failed_file = output_csv.replace(".csv", "_failed.csv")
        df.iloc[failed_indices].to_csv(failed_file, index=False)
        print(f"Saved failed prompts to {failed_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bedrock on prompt CSV (parallel)")
    parser.add_argument("--prompt_csv", required=True, help="Path to input prompt CSV file")
    parser.add_argument("--output_csv", default=None, help="Optional explicit output CSV path")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent API calls")
    parser.add_argument("--sleep_time", type=float, default=0.0, help="Optional delay between calls")
    args = parser.parse_args()

    call_bedrock_and_save_parallel(
        prompt_csv=args.prompt_csv,
        output_csv=args.output_csv,
        max_workers=args.workers,
        sleep_time=args.sleep_time
    )

# python bedrock_parallel.py \
#     --prompt_csv src/bedrock_pipeline/bedrock_prompts/classify/severe_sepsis/bm25_prompts.csv \
#     --output_csv src/bedrock_pipeline/bedrock_responses/classify/severe_sepsis/bm25_responses.csv \
#     --workers 10 \
#     --sleep_time 0.5