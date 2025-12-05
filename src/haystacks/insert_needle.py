from tqdm import tqdm
import pandas as pd
import random
import csv
import os
import sys

# Load MIMIC data to use for haystack
real_data_path = os.path.join("data", "mimic", "NOTEEVENTS.csv.gz")

try:
    print("Attempting to load real MIMIC data...")
    # Raise FileNotFoundError if the file is missing
    notes = pd.read_csv(real_data_path, low_memory=False)
    print("SUCCESS: Using real MIMIC data.")
except FileNotFoundError:
    print(f"\nERROR: MIMIC data not found at '{real_data_path}'.")
    print("ACTION REQUIRED: Need to get access to the official MIMIC data first.")
    # Stop the script immediately
    sys.exit(1)

# Filter to include first 500 notes only for quick testing (category = 'Discharge summary')
notes_subset = notes.head(500).copy()

# Load synthetic needles
needle_file = "src/needles/synthetic_needles.csv"
needles_df = pd.read_csv(needle_file)

# Convert list of strings
real_needles = list(needles_df["medical_note"].dropna())
print(f"Loaded {len(real_needles)} synthetic needles.")

def insert_needle(note_text, needles):
    """Insert one random needle into the note text.
    
    Args:
        note_text (str): The original medical note text.
        needles (list): List of needle strings to choose from.
        
    Returns:
        modified_text (str): The note text with the needle inserted.
        needle (str): The needle that was inserted.

    """
    needle = random.choice(needles)

    # Split note by newline for realistic insertion
    parts = note_text.split("\n")

    if len(parts) > 1:
        insert_idx = random.randint(0, len(parts) - 1)
        parts.insert(insert_idx, needle)
    else:
        # If note is a single chunk
        parts.append(needle)

    return "\n".join(parts), needle

# Build the haystack dataset
output_rows = []

print("\nInserting needles into MIMIC notes...")
for _, row in tqdm(notes_subset.iterrows(), total=len(notes_subset), desc="Processing Notes"):
    original_text = row["TEXT"]
    modified_text, needle_inserted = insert_needle(original_text, real_needles)

    output_rows.append({
        "HADM_ID": row["HADM_ID"],
        "SUBJECT_ID": row["SUBJECT_ID"],
        "CATEGORY": row["CATEGORY"],
        "ORIGINAL_NOTE": original_text,
        "MODIFIED_NOTE": modified_text,
        "NEEDLE_INSERTED": needle_inserted
    })

# Save haystack CSV
output_file = os.path.join("src", "haystacks", "mimic_haystack.csv")

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
    writer.writeheader()
    writer.writerows(output_rows)

print(f"Saved {len(output_rows)} notes with REAL synthetic needles to {output_file}")