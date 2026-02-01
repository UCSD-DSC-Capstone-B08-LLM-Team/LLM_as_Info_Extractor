# LLMs as Information Extractors

This project explores the use of Large Language Models (LLMs) to extract and classify information from unstructured clinical notes, such as those found in Electronic Health Records (EHRs). By combining LLMs with retrieval-augmented generation (RAG) techniques, we aim to improve extraction accuracy for long and complex clinical notes.

## Introduction

Medical notes are a critical part of healthcare, providing comprehensive patient information from allergies to past complications. However, these notes are often unstructured, contain domain specific abbreviations, and require significant manual effort to analyze. Hospitals spend millions annually on maintaining and reporting medical notes, consuming thousands of hours that could be more beneifical if spend on treating patients.

LLMs, such as GPT-4, Claude 3, and Gemini Advanced, have demonstrated the ability to accurately identify key details in medical text, often matching or surpassing traditional machine learning methods. In this project, we investigate LLM performance on MIMIC-III clinical data, exploring how RAG and other retrieval techniques can enhance reliability for nuanced or context-rich information and whether a "needle-in-a-haystack" approach can help to benchmark LLM reliability.

## Project Structure
<pre>
    <code>
        LLM_as_Info_Extractor/
        ├── LICENSE
        ├── README.md                       # Main project overview
        ├── requirements.txt
        ├── .gitignore
        ├── data/
        │   └── mimic/                      # MIMIC data (instructions in data/mimic/README.md)
        ├── src/
        │   ├── bedrock_pipeline/           # LLM prompt & Bedrock pipeline
        │   ├── needles/                    # Synthetic needle generation
        │   ├── eval/                       # LLM evaluation scripts
        │   ├── results/                    # Visualization scripts & output
        │   ├── retrieval/                  # Retrieval methods (Note level)
        |   ├── retrieval_patient_level/    # Retrieval methods (Patient level)
        │   └── haystacks/                  # Functions for inserting needles and constructing haystacks
        └── Support_Docs                    # Past EDA & Literature Review
    </code>
</pre>

## Features
- Evaluate LLMs on information extraction from clinical notes. 
- Extracts information from unstructured clinical notes
- Supports multiple LLMs (GPT-4, Claude 3, Gemini Advanced). 
- Retrieval-augmented generation (RAG) using multiple retrieval strategies:
    - BM25
    - FAISS using euclidean distance
    - FAISS using cosine similarity
    - Hybrid: Combination of BM25 and FAISS where weights are by default 0.5/0.5
- Generate and test Needle-in-a-Haystack querires
- LLM Evaluation can perform three tasks: **Extract** key information, **classify** note contents, and **summarize** notes (more resource-intensive)
    - Focus for this experiment is on extraction and classification tasks
- Generates visualizations for accuracy and retrieval performance
- Can be extended to other clinical datasets or synthetic notes

## Methodology

1. **Data Preparation**: Generate synthetic needles using `src/needles/generate_needles.py`. 
    - Synthetic needles were created by reviewing adverse events described in the NHSN manual, passing those descriptions to an LLM, and prompting it to generate realistic clinical notes containing the corresponding events.  
    - Create patient-level haystacks of clinical notes by inserting needles into MIMIC-III structured and unstructured data (specifically notes from MIMIC-III `NOTEEVENTS` table). 
2. **Retrieval**: Implements and evaluates retrieval methods (BM25, FAISS Euclidean, FAISS Cosine, and hybrid) to select relevant note segments before feeding to LLMs.  
3. **LLM Prompting**: Use AWS Bedrock to query LLMs for extracting and classifying relevant information. LLM evaluation scripts support summarization in addition to extract and classify tasks. Summarization can provide high-level overviews of medical notes but is more energy-intensive and may take longer to run.
4. **Evaluation**: Compare LLM predictions against structured labels or known needles.  
5. **Visualization**: Results are visualized by generating plots for overall accuracy, heatmaps, and strategy comparison.

## Installation & Setup
1. Clone the repository:

```bash
git clone https://github.com/UCSD-DSC-Capstone-B08-LLM-Team/LLM_as_Info_Extractor.git
cd LLM_as_Info_Extractor
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies: `pip install -r requirements.txt`

4. Place MIMIC-III data in `data/mimic/` following instructions in `data/mimic/README.md`.

5. Configure **AWS Bedrock** credentials (if using AWS LLMs). 

## Dataset

- **MIMIC-III**: Deidentified clinical data for ~46,000 patients at Beth Israel Deaconess Medical Center (2001–2012). Includes structured data (demographics, vitals, labs, medications) and unstructured clinical notes.
- **Synthetic Needles**: Generated data (through prompting of Bedrock) used to test retrieval and extraction without patient privacy concerns.
**Important**: MIMIC data is sensitive and cannot be shared publicly. Do not commit MIMIC files to GitHub.

#### Subset used in this project:
Filtered  `NOTEEVENTS.csv` from `data/mimic/` to only include first 500 rows where all the selected notes are from a single category: **Discharge Summary**. This subset was used to test LLM extraction performance without exposing large amounts of sensitive data. Only the `NOTEEVENTS` table was primarily used in this experiment. The full MIMIC-III dataset is not included in this repository due to privacy restrictions. For full access, see the `data/mimic/README.md` instructions.

## Results & Visualization
- Extraction and classification accuracy per LLM
- Strategy performance (BM25, FAISS, Hybrid)
- Heatmaps comparing overall and per-category performance
- Visualizations highlight LLM strengths and limitations for nuanced text

## Steps to Run: 

### STEP 1: Place MIMIC-III data in `data/mimic/`. 

### STEP 2: Configure AWS 

### STEP 3: Generate synthetic needles
```bash
python src/needles/generate_needles.py
```

### STEP 4: Create Haystacks by inserting Needles

We support two ways of constructing haystacks from `NOTEEVENTS.csv`:

#### 1. Note-level Needle Insertion:

Insert synthetic needles into individual notes by sampling notes from specified MIMIC note categories.

```bash
python src/haystacks/insert_needle.py \
    --categories "Echo" "ECG" "Discharge summary" \
    --n_notes 500 \
    --seed 42
```
**Parameters**
- `--categories` (required): One or more note categories to sample from (e.g., Discharge, ECG, Echo)
- `--n_notes`: Number of notes to sample (default: 500)
- `--seed`: Random seed for reproducibility (default: 42)

Each sampled note becomes a separate haystack document with a single inserted needle.

#### 2. Patient-level Needle Insertion:

Insert needles at the patient level, where all notes from a single patient are concatenated into one document and exactly one needle is inserted per patient.

```bash
python src/haystacks/create_haystack_patient_level.py \
  --categories "Echo" "ECG" "Discharge summary" \
  --n_patients 100 \
  --min_notes_per_patient 2 \
  --seed 42
``` 

**Parameters:**
- `--categories`: Optional list of note categories to include (default: all categories)
- `--n_patients`: Number of patients to sample (default: 100)
- `--min_notes_per_patient`: Minimum number of notes required per patient (default: 1)
- `--seed`: Random seed for reproducibility (default: 42)

Each patient corresponds to one haystack document containing all qualifying notes and one inserted needle.

### STEP 5: Run retrieval methods

#### 1. Note-level retrieval: 
Run BM25, FAISS with cosine similarity, FAISS with euclidean distance, and hybrid retrieval:
```bash
python src/retrieval/bm25_retrieval.py
python src/retrieval/faiss_cos_retrieval.py
python src/retrieval/faiss_cos_retrieval.py
python src/retrieval/hybrid_retrieval.py
```

#### 2. Patient-level retrieval:
Run BM25 retrieval: 
```bash
python src/retrieval_patient_level/bm25.py \
  --haystack_csv src/haystacks/mimic_haystack.csv \
  --output_csv src/retrieval_patient_level/outputs/bm25_patient_results.csv \
  --top_k 2 \
  --window_size 3
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save BM25 retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)


Run FAISS with cosine similarity retrieval:
```bash
python src/retrieval_patient_level/faiss_cos.py \
  --haystack_csv src/haystacks/mimic_haystack.csv \
  --output_csv src/retrieval_patient_level/outputs/faiss_cos_patient_results.csv \
  --top_k 2 \
  --window_size 3
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save BM25 retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)


Run FAISS with euclidean distance retrieval: 
```bash
python src/retrieval_patient_level/faiss_euc.py \
  --haystack_csv src/haystacks/mimic_haystack.csv \
  --output_csv src/retrieval_patient_level/outputs/faiss_euc_patient_results.csv \
  --top_k 2 \
  --window_size 3
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save BM25 retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)


Run hybrid retrieval: 
```bash
python src/retrieval_patient_level/hybrid.py \
  --haystack_csv src/haystacks/mimic_haystack.csv \
  --output_csv src/retrieval_patient_level/outputs/hybrid_patient_results.csv \
  --top_k 2 \
  --window_size 3 \
  --bm25_weight 0.5 \
  --faiss_weight 0.5
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save BM25 retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)
- `--bm25_weight`: Weight assigned to BM25 lexical scores (optional)
- `--faiss_weight`: Weight assigned to FAISS cosine similarity scores (optional)


### STEP 6: Generate prompts and run Bedrock LLM based on prompts by using `bedrock_pipeline` folder.

#### 1. Generate Prompts:
`prompt_generation.py` converts retrieval outputs (e.g., BM25, FAISS, or hybrid retrieval results) into structured prompts suitable for Amazon Bedrock / LLM inference. It takes the top retrieved passages for each patient-level query (“needle”) and formats them into task-specific prompts for downstream evaluation.

The script currently supports three clinical NLP tasks:

1. `classify`: Binary classification of whether a clinical scenario is present. The prompt intends to check whether the retrieved notes contain the clinical scenario described by the needle with an expected output of yes or no. 

2. `extract`: Exact text span extraction from the retrieved context. The prompt intends to force a precise extraction instead of summarizing or paraphrasing with an expected output of a python style list of the extracted text. 

3. `summarize`: Focused summarization of relevant information only. The prompt intends to filter out unrelated history while maintaining medically relevant details with an expected output of a clinical summary of the topics specified by the needle. 

To generate the prompts, run: 
```bash
python src/bedrock_pipeline/prompt_generation.py \
  --retrieval_csv src/retrieval_patient_level/outputs/bm25_patient_results.csv \
  --output_csv src/bedrock_pipeline/bedrock_prompts/classify/bm25_prompts.csv \
  --task classify
```
**Parameters:**
- `--retrieval_csv`: Path to retrieval csv 
- `--output_csv`: Path to save generated Bedrock prompt csv
- `--task`: One of `classify`, `extract`, `summarize`

#### 2. Call Bedrock: 
`call_bedrock.py` sends generated prompts to an Amazon Bedrock hosted LLM (e.g., DeepSeek) and saves the model responses to disk. It is designed to operate on prompt CSVs produced by `prompt_generation.py` and follows a standardized directory structure to automatically infer task and retrieval method.

The scipt expects the prompt CSV path to follow this structure:
```php-template
src/bedrock_pipeline/bedrock_prompts/<task>/<retrieval_method>_prompts*.csv
```
From this path, the script automatically infers the task (classify, extract, or summarize) and the retrieval method (bm25, faiss_cos, faiss_euc, or hybrid). 

To call a Bedrock model, run:
```bash
python src/bedrock_pipeline/call_bedrock.py \
    --prompt_csv src/bedrock_pipeline/bedrock_prompts/classify/bm25_prompts.csv
```
**Parameter:**
- `--prompt_csv`: Path to a generated `bedrock_prompts` CSV containing prompts for a specific task and retrieval method.


### STEP 7: Evaluate Bedrock LLM
(TODO: add patient-level maybe input/output csv)
#### 1. For classification methods:
```bash
python src/eval/classify_llm_eval.py
```
#### 2. For extraction methods:
```bash
python src/eval/extract_llm_eval.py
```
#### 3. For summarization methods:
(TODO)

### STEP 8: Generate Visualizations

Visualizations for Bedrock evaluation can be generated by:
```bash
python src/results/visualize_eval.py
```
Visualizations for retrieval evaluation can be generated by:
```bash
python src/results/visualize_retrieval.py
```

## Notes
- `.gitignore` is configured to exclude `.DS_Store`, `__pycache__`, and sensitive MIMIC data. The project demonstrates methodology on a small subset of the data; scaling to full MIMIC requires additional setup and approvals.

- The NHSN manual used to generate synthetic needles in "needle-in-a-haystack" approach: 
[NHSN_manual](https://www.cdc.gov/nhsn/pdfs/pscmanual/pcsmanual_current.pdf) 

### Contributions: 
- Leah: Worked on the main branch, llm-needle-experiments branch, and medqa-prep branch. Specifically on llm-needle-experiments branch, worked on generating needles and haystacks and implementing and testing retrieval methods. On main branch, generated synthetic needles using LLMs and inserted them into MIMIC notes. Tested them on several retrieval methods and generate responses using AWS Bedrock to protect MIMIC privacy concerns. Used medqa-prep branch to determine if data could be used as benchmark. 
- Lewis: Worked on the generate needles branch. Implemented the pipeline using DeepSeek API to generate needles and haystacks and retrieve needles from haystacks. Added configuration for temperature, subtlety, and number of samples. Clean the LLM output and wrote into csv files for future analysis.
- Omid: Worked on the answer-generation component of the project and contributed to the LLM-retrieval pipeline. Specifically, created the full pipeline that takes retrieved passages from BM25, FAISS-cosine, FAISS-euclidean, and hybrid methods and generates final LLM answers for all queries. Implemented passage reordering to reduce lost-in-the-middle effects, designed standardized prompting, and ran gpt-4o-mini across all retrieval outputs. Produced structured CSV outputs for each method and added all code/results to the answer_generation branch. Also assisted with repository organization and ensured compatibility between retrieval outputs and downstream evaluation.
- Kaijie: Before leaving the group, worked on the llm-needle-experiments branch.
