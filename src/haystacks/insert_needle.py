from tqdm import tqdm
import pandas as pd
import random
import csv
import os
import sys
import argparse

# define command line arguments
parser = argparse.ArgumentParser(description="Insert synthetic needles into MIMIC notes")
parser.add_argument(
    "--categories",
    type=str,
    nargs="+",
    required=True,
    help="One or more note categories (ECG, Echo, Discharge, etc)"
)
parser.add_argument(
    "--n_notes",
    type=int,
    default=500,
    help="Number of notes to sample from the category"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility"
)
args = parser.parse_args()
random.seed(args.seed)


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


# Strip whitespace from both MIMIC CATEGORY column and input categories
notes["CATEGORY"] = notes["CATEGORY"].astype(str).str.strip()
selected_categories = set([cat.strip() for cat in args.categories])

# Filter notes by cleaned categories
category_notes = notes[notes["CATEGORY"].isin(selected_categories)].copy()

if len(category_notes) == 0:
    raise ValueError(f"No notes found for categories {selected_categories}")

print("Category counts:")
print(category_notes["CATEGORY"].value_counts())

# Sample notes evenly across categories
per_cat = args.n_notes // len(selected_categories)
notes_subset = (
    category_notes
    .groupby("CATEGORY", group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), per_cat), random_state=args.seed))
    .reset_index(drop=True)
)
remaining = args.n_notes - len(notes_subset)

if remaining > 0:
    extra = category_notes.drop(notes_subset.index, errors="ignore") \
        .sample(n=remaining, random_state=args.seed)
    notes_subset = pd.concat([notes_subset, extra], ignore_index=True)

print("Total notes sampled:", len(notes_subset))
print(notes_subset["CATEGORY"].value_counts())

# MIMIC notes have varying timestamp availability across categories,
# so define a unified event time using chart time when available and chart date otherwise 
notes_subset["CHARTTIME"] = pd.to_datetime(notes_subset["CHARTTIME"], errors="coerce")
notes_subset["CHARTDATE"] = pd.to_datetime(notes_subset["CHARTDATE"], errors="coerce")
notes_subset["EVENT_TIME"] = notes_subset["CHARTTIME"].fillna(notes_subset["CHARTDATE"])


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
        "EVENT_TIME": row["EVENT_TIME"],
        "CHARTDATE": row["CHARTDATE"],
        "CHARTTIME": row["CHARTTIME"],
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

# to run: 
# python src/haystacks/insert_needle.py --categories 'ECG' 'Echo' 'Discharge summary' --n_notes 1000 --seed 42