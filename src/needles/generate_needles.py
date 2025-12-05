import boto3
import json
import csv
import time
import random

REGION_NAME = "us-west-2" 
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
OUTPUT_FILE = "mimic_data/needles/synthetic_needles.csv"
NUM_NEEDLES_PER_TYPE = 50

# These prompts force the LLM to follow strict NHSN surveillance criteria.
PROMPT_CAUTI = """
Generate a SINGLE short clinical statement (1–3 sentences) that satisfies ALL of:
1. Patient has an indwelling urinary catheter (Foley) in place for >2 days.
2. Fever > 38.0 C.
3. Urine culture positive for E. coli (>100,000 CFU/ml).
4. Patient complains of suprapubic tenderness or dysuria.
Output ONLY the short statement. No SOAP note. No medical record formatting.
"""

PROMPT_CLABSI = """
Generate a SINGLE short clinical statement (1–3 sentences) that satisfies ALL of:
1. Patient has a Central Venous Catheter (CVC) in place for >2 days.
2. Fever > 38.0 C.
3. Blood culture positive for Staph aureus.
4. EXPLICITLY STATE there is no infection at other sites (lungs clear, urine clear) to rule out secondary infection.
Output ONLY the short statement. No SOAP note. No medical record formatting.
"""

def generate_note(client, prompt_template):
    """Calls Bedrock to generate a single note.
    Args:
        client: Boto3 Bedrock client.
        prompt_template (str): The prompt template to use for generation.
        
    Returns:
        generated_note (str): The generated medical note text.
    """
    
    # Add random seed to the prompt to stop the LLM from generating identical text every time
    random_seed = f" Random seed: {random.randint(1, 10000)}"
    final_prompt = prompt_template + random_seed

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.7, 
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": final_prompt}]}
        ]
    }

    try:
        response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        response_body = json.loads(response.get("body").read())
        return response_body["content"][0]["text"].strip()
    except Exception as e:
        print(f"Error generating note: {e}")
        return None

def main():
    print(f"Starting generation of {NUM_NEEDLES_PER_TYPE * 2} synthetic needles...")
    
    # Initialize Bedrock Client
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=REGION_NAME)
    
    data_rows = []

    # Generate CAUTI Needles
    print("\n--- Generating CAUTI Needles ---")
    for i in range(NUM_NEEDLES_PER_TYPE):
        print(f"Generating CAUTI note {i+1}/{NUM_NEEDLES_PER_TYPE}...")
        note = generate_note(bedrock, PROMPT_CAUTI)
        if note:
            data_rows.append({
                "medical_note": note,
                "true_condition": "CAUTI",
                "needle_found": True
            })
        time.sleep(0.5) 

    # Generate CLABSI Needles
    print("\n--- Generating CLABSI Needles ---")
    for i in range(NUM_NEEDLES_PER_TYPE):
        print(f"Generating CLABSI note {i+1}/{NUM_NEEDLES_PER_TYPE}...")
        note = generate_note(bedrock, PROMPT_CLABSI)
        if note:
            data_rows.append({
                "medical_note": note,
                "true_condition": "CLABSI",
                "needle_found": True
            })
        time.sleep(0.5)

    # save to CSV
    print(f"\nSaving {len(data_rows)} notes to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['medical_note', 'true_condition', 'needle_found']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)
        
    print("Done! You now have your synthetic needles.")

if __name__ == "__main__":
    main()