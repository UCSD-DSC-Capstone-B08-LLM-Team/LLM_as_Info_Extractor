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
    nargs="*",
    default=None,
    help="Optional list of note categories. If omitted, all categories are used."
)
parser.add_argument(
    "--n_patients",
    type=int,
    default=100,
    help="Number of patients (each becomes one document with one needle)"
)
parser.add_argument(
    "--min_notes_per_patient",
    type=int,
    default=1,
    help="Minimum number of notes required per patient"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility"
)
parser.add_argument(
    "--needle_file",
    type=str,
    default="src/needles/contra_care_needles.csv",
    help="Path to synthetic needle CSV"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="src/haystack/outputs/mimic_haystack.csv",
    help="Path to output haystack CSV"
)
args = parser.parse_args()
random.seed(args.seed)
print(f"Using needle file: {args.needle_file}")



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



# Filter by categories

# strip whitespace from mimic category column
notes["CATEGORY"] = notes["CATEGORY"].astype(str).str.strip()

if args.categories is None or len(args.categories) == 0:
     # Use all categories
    category_notes = notes.copy()
    print("Using ALL categories")
else:
    # Strip whitespace from selected categories
    selected_categories = set(cat.strip() for cat in args.categories)
    # Filter notes by selected categories
    category_notes = notes[notes["CATEGORY"].isin(selected_categories)].copy()
    print("Using categories:", selected_categories)

if len(category_notes) == 0:
    raise ValueError("No notes found after category filtering")



# Compute EVENT_TIME

# MIMIC notes have varying timestamp availability across categories,
# so define a unified event time using chart time when available and chart date otherwise 
category_notes["CHARTTIME"] = pd.to_datetime(category_notes["CHARTTIME"], errors="coerce")
category_notes["CHARTDATE"] = pd.to_datetime(category_notes["CHARTDATE"], errors="coerce")

category_notes["EVENT_TIME"] = category_notes["CHARTTIME"].fillna(category_notes["CHARTDATE"])



# Remove notes without EVENT_TIME
before = len(category_notes)
category_notes = category_notes[category_notes["EVENT_TIME"].notna()]
after = len(category_notes)

print(f"Dropped {before - after} notes with no timestamp")



# Group by patient
patient_groups = category_notes.groupby("SUBJECT_ID")

eligible_patients = [
    pid for pid, g in patient_groups
    if len(g) >= args.min_notes_per_patient
]

print(f"Eligible patients with ≥{args.min_notes_per_patient} notes: {len(eligible_patients)}")

if len(eligible_patients) == 0:
    raise ValueError("No patients meet the minimum note requirement")



# Sample patients
random.seed(args.seed)

selected_patients = random.sample(
    eligible_patients,
    min(args.n_patients, len(eligible_patients))
)

notes_subset = category_notes[
    category_notes["SUBJECT_ID"].isin(selected_patients)
].copy()

print(f"Selected {len(selected_patients)} patients")
print("Notes per category in sample:")
print(notes_subset["CATEGORY"].value_counts())



# Load synthetic needles
if not os.path.exists(args.needle_file):
    raise FileNotFoundError(f"Needle file not found: {args.needle_file}")

needles_df = pd.read_csv(args.needle_file)

needles = needles_df[
    ["DATA_ELEMENT", "QUERY", "NEEDLE_TEXT"]
].dropna()

print(f"Loaded {len(needles)} synthetic needles.")



def insert_needle(note_text, needle_df):
    """Insert one random needle from the dataframe into the note text."""

    row = needle_df.sample(n=1).iloc[0]

    needle_text = row["NEEDLE_TEXT"]
    query = row["QUERY"]
    data_element = row["DATA_ELEMENT"]

    parts = note_text.split("\n")

    if len(parts) > 1:
        insert_idx = random.randint(0, len(parts) - 1)
        parts.insert(insert_idx, needle_text)
    else:
        parts.append(needle_text)

    return (
        "\n".join(parts),
        needle_text,
        query,
        data_element
    )



# Build the haystack dataset
output_rows = []

print("\nBuilding patient-level haystack...")

for subject_id, patient_notes in tqdm(
    notes_subset.groupby("SUBJECT_ID"),
    desc="Processing Patients"
):
    patient_notes = patient_notes.copy()

    # pick one random note for this patient
    chosen_idx = random.choice(patient_notes.index.tolist())
    original_note = patient_notes.loc[chosen_idx, "TEXT"]

    modified_note, needle_inserted, query, data_element = insert_needle(original_note, needles)
    patient_notes.loc[chosen_idx, "TEXT"] = modified_note

    # build full patient timeline
    patient_notes = patient_notes.sort_values("EVENT_TIME")

    combined_text = "\n\n".join(
        patient_notes.apply(
            lambda r: f"[{r['EVENT_TIME']} | {r['CATEGORY']}]\n{r['TEXT']}",
            axis=1
        )
    )

    output_rows.append({
        "SUBJECT_ID": subject_id,
        "NUM_NOTES": len(patient_notes),
        "DATA_ELEMENT": data_element,
        "QUERY": query,
        "NEEDLE_INSERTED": needle_inserted,
        "PATIENT_RECORD": combined_text
    })



# Save haystack CSV
output_file = args.output_file

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
    writer.writeheader()
    writer.writerows(output_rows)

print(f"Saved {len(output_rows)} notes with REAL synthetic needles to {output_file}")

# to run: 
# python src/haystacks/create_haystack_query.py \
#   --categories \
#   --n_patients 100 \
#   --min_notes_per_patient 2 \
#   --seed 42