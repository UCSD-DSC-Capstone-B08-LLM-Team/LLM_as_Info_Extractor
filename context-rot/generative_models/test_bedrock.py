import boto3
import json

# Replace with the AWS Region where your Bedrock access is configured
REGION_NAME = "us-west-2" 
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
PROMPT = "Explain the difference between a short-term and long-term Bedrock API key in one sentence."

try:
    # Boto3 client automatically uses the AWS_BEARER_TOKEN_BEDROCK environment variable
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION_NAME
    )

    # The payload structure depends on the model (this is for Anthropic Claude)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": PROMPT}]
            }
        ]
    }

    response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    # Process and print the response
    response_body = json.loads(response.get("body").read())
    
    # Extract the generated text
    if response_body.get("content"):
        generated_text = response_body["content"][0]["text"]
        print(f"--- Prompt ---\n{PROMPT}\n")
        print(f"--- Response ---\n{generated_text}")
    else:
        print("Model returned an empty response.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("\nTroubleshooting: Ensure you have model access, the correct region, and your API key is properly set as the environment variable.")