from tqdm import tqdm
import pandas as pd
import boto3
import time
import json
import os
import argparse

def infer_task_and_retrieval(prompt_csv_path):
    """
    Infers task and retrieval method from standardized path.

    Args:
        prompt_csv_path (str): Path to the prompt CSV file.

    Returns:
        tuple: (task, retrieval) inferred from the path.
    """

    parts = prompt_csv_path.replace("\\", "/").split("/")
    task = parts[-2]
    retrieval = os.path.basename(prompt_csv_path).replace("_prompts.csv", "")
    return task, retrieval


def call_bedrock_and_save(prompt_csv, model_id="deepseek.v3-v1:0", max_tokens=512, temperature=0.0, sleep_time=0.5, region="us-west-2"):
    """
    Calls Bedrock model on prompts from CSV and saves responses.
    
    Args:
        prompt_csv (str): Path to input prompt CSV file.
        model_id (str): Bedrock model ID to use.
        max_tokens (int): Maximum tokens for Bedrock response.
        temperature (float): Temperature setting for Bedrock model.
        sleep_time (float): Sleep time between API calls to avoid rate limits.
        region (str): AWS region for Bedrock client.

    Returns:
        None
    """

    df = pd.read_csv(prompt_csv)
    client = boto3.client("bedrock-runtime", region_name=region)

    task, retrieval = infer_task_and_retrieval(prompt_csv)

    output_csv = f"src/bedrock_pipeline/bedrock_responses/{task}/{retrieval}_responses.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    responses = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calling DeepSeek"):
        prompt = row["bedrock_prompt"]

        try:
            request_body = {
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            resp = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(resp["body"].read())
            
            # Extract response based on DeepSeek v3 format
            if "choices" in response_body:
                output = response_body["choices"][0]["message"]["content"]
            # Fallback for other DeepSeek variations
            elif "outputs" in response_body:
                output = response_body["outputs"][0]["text"]
            else:
                output = str(response_body) 
                
        except Exception as e:
            output = f"ERROR: {e}"

        responses.append(output)
        time.sleep(sleep_time)

    df["bedrock_response"] = responses
    df.to_csv(output_csv, index=False)

    print(f"\nSaved Bedrock responses to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bedrock on prompt CSV")
    parser.add_argument("--prompt_csv", required=True, help="Path to input prompt CSV file")

    args = parser.parse_args()

    call_bedrock_and_save(
        prompt_csv=args.prompt_csv
    )

# Run the script with:
# python src/bedrock_pipeline/call_bedrock.py --prompt_csv src/bedrock_pipeline/bedrock_prompts/classify/bm25_prompts.csv