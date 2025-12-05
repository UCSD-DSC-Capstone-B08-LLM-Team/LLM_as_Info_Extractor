from tqdm import tqdm
import pandas as pd
import boto3
import time
import json
import os

# Bedrock client (region must match your model!)
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

def call_bedrock_and_save(csv_file, output_file, model_id="anthropic.claude-3-sonnet", sleep_time=0.5):
    """
    Calls Bedrock for each prompt in CSV and saves responses.
    
    CSV must have column: 'bedrock_prompt'
    Optional columns: 'needle', 'top_passages'
    """
    df = pd.read_csv(csv_file)
    responses = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calling Bedrock"):
        prompt = row["bedrock_prompt"]

        try:
            resp = bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 512,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                })
            )

            response_body = json.loads(resp["body"].read())
            output = response_body["content"][0]["text"]

        except Exception as e:
            output = f"ERROR: {e}"

        responses.append(output)
        time.sleep(sleep_time)

    df["bedrock_response"] = responses

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\nSaved Bedrock responses to {output_file}")

def check_bedrock_connection(model_id, region="us-west-2"):
    """Tests connection and model ID validity by invoking a simple prompt.
    
    Args:
        model_id (str): Bedrock model ID to test.
        region (str): AWS region for Bedrock.
    
    Returns:
        bool: True if connection and model ID are valid, False otherwise.
    """
    
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        
        # Use a minimal request body
        client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            })
        )
        print(f"\nConnection Successful! Model ID '{model_id}' is valid in {region}.")
        return True
    
    except client.exceptions.ResourceNotFoundException:
        print(f"\nERROR: Model ID '{model_id}' is invalid or access is denied. Check AWS Model Access page.")
        return False
    except Exception as e:
        print(f"\nERROR: Failed to connect to Bedrock. Check API Key and Region. Details: {e}")
        return False

# Example usage before the loop:
model_to_use = "anthropic.claude-3-sonnet-20240229-v1:0" 

if check_bedrock_connection(model_to_use):
    call_bedrock_and_save(
        csv_file="src/bedrock_pipeline/bedrock_prompts/classify/bm25_prompts.csv",
        output_file="src/bedrock_pipeline/bedrock_responses/classify/bm25_responses.csv",
        model_id=model_to_use,
        sleep_time=0.5
    )