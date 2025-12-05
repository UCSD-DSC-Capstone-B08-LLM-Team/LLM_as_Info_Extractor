# MIMIC-III Data Setup

This folder is intended to hold the MIMIC-III Clinical Database data required for running the LLM as Information Extractor project. MIMIC data is restricted and cannot be publicly shared. Users must obtain access via official channels.

## Accessing MIMIC-III

To use this project, you must first get access to MIMIC-III:

1. Complete the required CITI “Data or Specimens Only Research” course (Human Subjects training).
2. Create a PhysioNet account: https://physionet.org/
3. Request access to MIMIC-III (requires agreeing to the data use agreement).
4. Download the dataset (usually in CSV or SQL format).

2. Folder Structure
After downloading, organize your MIMIC-III files in the following structure:

```bash
    data/mimic/
    ├── NOTEEVENTS.csv          # Free-text clinical notes
    ├── PATIENTS.csv            # Patient demographic info
    ├── ADMISSIONS.csv          # Admission details
    ├── ICUSTAYS.csv            # ICU stay info
    ├── other CSV files...      # e.g., LABEVENTS.csv, DIAGNOSES_ICD.csv
```
Important: The filenames above are examples—adjust according to your downloaded files. All scripts assume the folder data/mimic/ contains these CSVs.

## Missing Files & Placeholders

If you do not have certain tables, you can create empty placeholders with the same name to prevent script errors:
```bash 
    touch data/mimic/NOTEEVENTS.csv
```
Scripts that rely on these files will not run correctly with empty placeholders—they are only for folder structure purposes.

## Notes on Privacy & Security

- Do not commit MIMIC files to GitHub or share them.
- Make sure `.gitignore` includes  `data/mimic/`

This ensures your private clinical data remains secure.

## Verifying Setup
After placing the CSVs, you can test the setup by running a simple script:
```bash
    python src/haystacks/insert_needle.py --test
```
This will check for required files without processing any sensitive data.