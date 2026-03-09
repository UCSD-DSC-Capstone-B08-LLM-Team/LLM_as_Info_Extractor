# LLMs as Information Extractors

Doctors and hospitals rely on medical records to make life-or-death decisions. These records contain vital information such as diagnoses, treatments, medications, lab results, and clinical observations that guide patient care.

But medical records are long, complex, and often messy. Finding the important details can take hours of careful review, and mistakes or delays can affect patient outcomes. Accurate information extraction is not only critical for doctors and patients, but it also supports hospital operations, research, and reporting to agencies like the Centers for Medicare and Medicaid Services (CMS).

AI tools, like large language models (LLMs), can help speed up information extraction from medical records. But they have a problem: when given long and messy notes, they often get confused and miss the most important details. This limits how much hospitals can rely on AI for critical tasks. Retrieval-augmented generation (RAG) can help mitigate this challenge by reducing the amount of information providing to LLMs, however, this field is underexplored. Thus, we introduce a benchmark for evaluating and comparing retrieval methods for EHR information extraction. To help LLMs focus on the right information, we tested different retrieval strategies that guide the model to the most relevant parts of the records before it answers questions. Through this framework, researchers are able to compare different strategies and see which ones work best in medical settings.

We tested our methods using real patient records from the MIMIC-III database, which contains thousands of de-identified unstructured clinical notes from a hospital’s intensive care units. These notes come in many forms such as nursing notes, doctor’s observations, lab reports, and discharge summaries, and they vary in length and style.

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
        │   ├── eval/                       # LLM evaluation scripts
        │   ├── haystacks/                  # Functions for inserting needles and constructing haystacks
        │   ├── llm/                        # Baseline LLM
        │   ├── needles/                    # Synthetic needle generation
        │   ├── results_patient_level/      # Visualization scripts & output 
        |   └── retrieval_query/            # Retrieval methods
        └── Support_Docs                    # Past EDA & Literature Review
    </code>
</pre>

## Features
- "Needle-in-a-haystack" benchmark where synthetic clinical statements (“needles”) are inserted into long patient records (“haystacks”) for clinical information retrieval
- Comparison of different retrieval strategies
- Insights on how to make LLMs more accurate and reliable in healthcare settings
- Privacy preserving LLM using DeepSeek-V3 model on AWS Bedrock
- Retrieval-augmented generation (RAG) using multiple retrieval strategies:
    - BM25
    - FAISS using euclidean distance and cosine similarity
    - FAISS using maximal marginal relevance (MMR)
    - Hybrid: Combination of BM25 and FAISS
    - Semantic Chunking
    - SPLADE
- LLM Evaluation can perform two tasks: 
    - Classification: determining whether relevant information exists
    - Extraction: identifying the specific evidence in clinical notes
- Generates visualizations for accuracy and retrieval performance

## Methodology

1. **Data Preparation**: Generate synthetic needles using several files under `needles/` folder. 
    - Synthetic needles were generated using five elements from the Severe Sepsis and Septic Shock
    Early Management Bundle (SEP-1) from the Centers for Medicare and Medicaid Services (CMS) specifications manual.
    - Create patient-level haystacks of clinical notes by inserting needles into MIMIC-III structured and unstructured data (specifically notes from MIMIC-III `NOTEEVENTS` table). 
2. **Retrieval**: Implements and evaluates retrieval methods (BM25, FAISS, FAISS with MMR, Hybrid, Semantic Chunking, and SPLADE) to select relevant note segments before feeding to LLMs. 
3. **Baseline LLM (No Retrieval)**: To measure the impact of retrieval-augmented generation (RAG), we run the same classification and extraction tasks using LLMs without any retrieval step. In this setting, the model receives the full patient note directly in the prompt. This provides a direct comparison between LLM-only performance and RAG-enhanced performance. The baseline scripts are located in `src/llm/`.
4. **LLM Prompting**: Use AWS Bedrock to query LLMs for extracting and classifying relevant information.
5. **Evaluation**: Compare LLM predictions against structured labels or known needles.  
6. **Visualization**: Results are visualized by generating plots for overall accuracy, heatmaps, and strategy comparison.

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

5. Configure **AWS Bedrock** credentials. 

## Dataset

- **MIMIC-III**: Deidentified clinical data for ~46,000 patients at Beth Israel Deaconess Medical Center (2001–2012). Includes structured data (demographics, vitals, labs, medications) and unstructured clinical notes.
- **Synthetic Needles**: Generated data by using CMS Specifications and used as a ground truth to test retrieval and extraction without patient privacy concerns.
**Important**: MIMIC data is sensitive and cannot be shared publicly. Do not commit MIMIC files to GitHub.

#### Subset used in this project:
Used `NOTEEVENTS.csv` from `data/mimic/` where only the `NOTEEVENTS` table was primarily used in this experiment. The full MIMIC-III dataset is not included in this repository due to privacy restrictions. For full access, see the `data/mimic/README.md` instructions.

## Results & Visualization
- Extraction and classification accuracy per LLM
- Strategy performance (BM25, FAISS, FAISS MMR, Hybrid, Semantic Chunking, SPLADE)
- Heatmaps comparing overall and per-category performance
- Visualizations highlight LLM strengths and limitations for nuanced text

## Steps to Run: 

### STEP 1: Place MIMIC-III data in `data/mimic/`. 

### STEP 2: Configure AWS Bedrock
Sign in to AWS console. In the search bar, look up Bedrock and click on the first option titled "Bedrock". Under "Get started by using API Keys", select "View API keys." Select the long-term API key and then select "Generate long-term API keys". Save this key in a secure location.

### STEP 3: Generate synthetic needles

Synthetic needles were created using clinical definitions from version 5.18a of the Centers for Medicare & Medicaid Services (CMS) *Specifications Manual for National Hospital Inpatient Quality Measures*. We used the **SEP-1 1b-Alpha Data Dictionary (AlphaDD) abstraction specifications** to define clinically meaningful query concepts for Severe Sepsis.

From this manual, we focused on five element types:

1. Administrative Contraindication to Care — Severe Sepsis  
2. Directive for Comfort Care or Palliative Care — Severe Sepsis  
3. Clinical Trial — Severe Sepsis  
4. Severe Sepsis Present  
5. Vasopressor Administration — Severe Sepsis  

Each needle is a short ground-truth text span inserted into the patient timeline. Needles are constructed directly from CMS clinical criteria to ensure medical validity, reproducibility, and concept-level retrieval evaluation. To generate needles for each of the five elements, run the code below. 

#### 1. Administrative Contraindication to Care — Severe Sepsis:
```bash
python src/needles/generate_sep1_needles_contra_care.py
```

#### 2. Directive for Comfort Care or Palliative Care — Severe Sepsis:
```bash
python src/needles/generate_sep1_needles_comfort.py
```

#### 3. Clinical Trial — Severe Sepsis:
```bash
python src/needles/generate_sep1_needles_clinical_trial.py
```

#### 4. Severe Sepsis Present:
```bash
python src/needles/generate_sep1_needles_severe_sepsis_present.py
```

#### 5. Vasopressor Administration — Severe Sepsis:
```bash
python src/needles/generate_sep1_needles_vaso.py
```

After creating five types of needles, we show how to run the benchmark pipeline for one type of needle, administrative contraindication to care. To run on all five needles, repeat steps 4-8 using different needle files. 

### STEP 4: Create Haystacks by inserting Needles

Patient-level needle insertion is our main method of needle insertion as it insert needles at the patient level, where all notes from a single patient are concatenated into one document and exactly one needle is inserted per patient. Before inserting the needles, the Bedrock API key from before needs to be exported.

```bash
export AWS_BEARER_TOKEN_BEDROCK={YOUR_API_KEY}
```

```bash
python src/haystacks/create_haystack_query.py \
  --categories "Echo" "ECG" "Discharge summary" \
  --n_patients 100 \
  --min_notes_per_patient 2 \
  --seed 42 \
  --needle_file "src/needles/contra_care_needles.csv" \
  --output_file "src/haystack/outputs/mimic_haystack_contra_care.csv"
``` 

**Parameters:**
- `--categories`: Optional list of note categories to include (default: all categories)
- `--n_patients`: Number of patients to sample 
- `--min_notes_per_patient`: Minimum number of notes required per patient
- `--seed`: Random seed for reproducibility 
- `--needle_file`: Path to synthetic needle CSV
- `--output_file`: Path to output haystack CSV

Each patient corresponds to one haystack document containing all qualifying notes and one inserted needle.

### STEP 5: Run Baseline LLMs:
This step evaluates LLM performance without retrieval to quantify the benefit of RAG.

Run the baseline file to create a "retrieval" csv:
```bash
python src/llm/full_context_baseline.py \
  --input_csv src/haystacks/mimic_haystack_contra_care.csv \
  --output_csv src/llm/outputs/baseline_contra_care.csv
```

We then use this baseline retrieval file to prompt the LLM and run Bedrock. See STEP 7 for more information on how to create prompts and run the Bedrock model.

### STEP 6: Run retrieval methods

Run BM25 retrieval: 
```bash
python src/retrieval_query/bm25.py \
  --haystack_csv src/haystacks/mimic_haystack_contra_care.csv \
  --output_csv src/retrieval_query/outputs/contra_care/bm25_patient_results.csv \
  --top_k 10 \
  --window_size 1
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save BM25 retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)


Run FAISS with cosine similarity retrieval:
```bash
python src/retrieval_query/faiss_cos.py \
  --haystack_csv src/haystacks/mimic_haystack_contra_care.csv \
  --output_csv src/retrieval_query/outputs/contra_care/faiss_cos_patient_results.csv \
  --top_k 10 \
  --window_size 1
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save FAISS retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)


Run FAISS with euclidean distance retrieval: 
```bash
python src/retrieval_query/faiss_euc.py \
  --haystack_csv src/haystacks/mimic_haystack_contra_care.csv \
  --output_csv src/retrieval_query/outputs/contra_care/faiss_euc_patient_results.csv \
  --top_k 2 \
  --window_size 3
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save FAISS retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)


Run FAISS with maximal marginal relevance (MMR) retrieval:
```bash
python src/retrieval_query/faiss_mmr.py \
  --haystack_csv src/haystacks/mimic_haystack_contra_care.csv \
  --output_csv src/retrieval_query/outputs/contra_care/faiss_mmr_patient_results.csv \
  --top_k 10 \
  --window_size 1 \
  --mmr_lamda 0.7 \
  --mmr_candidates 20
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save FAISS retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)
- `--mmr_lambda`: Relevance–diversity trade-off for MMR re-ranking (0 = more diverse, 1 = more relevant) (optional)
- `--mmr_candidates`: Number of initially retrieved passages considered for MMR before selecting final top_k (optional)


Run hybrid retrieval: 
```bash
python src/retrieval_query/hybrid.py \
  --haystack_csv src/haystacks/mimic_haystack_contra_care.csv \
  --output_csv src/retrieval_query/outputs/contra_care/hybrid_patient_results.csv \
  --top_k 10 \
  --window_size 1 \
  --bm25_weight 0.3 \
  --faiss_weight 0.7
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save BM25 retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)
- `--bm25_weight`: Weight assigned to BM25 lexical scores (optional)
- `--faiss_weight`: Weight assigned to FAISS cosine similarity scores (optional)


Run Semantic Chunking retrieval: 
```bash
python src/retrieval_patient_level/semantic_chunking.py \
  --haystack_csv src/haystacks/mimic_haystack_contra_care.csv \
  --output_csv src/retrieval_patient_level/outputs/contra_care/hybrid_patient_results.csv \
  --top_k 10 \
  --max_sents 5 
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save BM25 retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--max_sents`: Maximum number of sentences allowed in semantic chunk (optional)
- `--sim_threshold`: Similarity threshold used to decide whether consecutive sentences belong in the same chunk (optional)
- `--embed_model`: Embedding model used to compute sentence similarity (optional)
- `--limit_rows`: Limit on the number of patients processed (optional)


Run SPLADE retrieval: 
```bash
python src/retrieval_patient_level/splade.py \
  --haystack_csv src/haystacks/mimic_haystack_contra_care.csv \
  --output_csv src/retrieval_patient_level/outputs/contra_care/splade_patient_results.csv \
  --top_k 5 \
  --window_size 1 \
  --max_length 128 \
  --batch_size 8
```
**Parameters:**
- `--haystack_csv`: Path to patient-level haystack CSV (optional)
- `--output_csv`: Path to save BM25 retrieval results (optional)
- `--top_k`: Number of top passages retrieved per patient (optional)
- `--window_size`: Number of sentences per passage window (optional)
- `--model_name`: Path of pretrained SPLADE model used to generate sparse representations (optional)
- `--max_length`: Maximum number of tokens per passage after tokenization (optional)
- `--batch_size`: Number of passages encoded at once during inference (optional)
- `--limit_rows`: Limit on the number of patients processed (optional)

SPLADE retrieval takes longest to run, so to upscale on more patients GPU use may be necessary. If access to a GPU, splade can be run by using `splade_gpu.py` with the same inputs used on `splade.py`. For other retrieval methods, changes would need to be made to run on a GPU.

### STEP 7: Generate prompts and run Bedrock LLM based on prompts by using `bedrock_pipeline` folder.

#### 1. Generate Prompts:
`prompt_generation.py` converts retrieval outputs (e.g., BM25, FAISS) into structured prompts suitable for Amazon Bedrock / LLM inference. It takes the top retrieved passages for each patient-level query (“needle”) and formats them into task-specific prompts for downstream evaluation.

The script supports two clinical NLP tasks:

1. `classify`: Binary classification of whether a clinical scenario is present. The prompt intends to check whether the retrieved notes contain the clinical scenario described by the needle with an expected output of yes or no. 

2. `extract`: Boolean value of whether the exact text span was extracted from the retrieved context. The prompt determines whether the needle was retrieved from the haystack. 

To generate the prompts, run: 
```bash
python src/bedrock_pipeline/prompt_generation.py \
  --retrieval_csv src/retrieval_query/outputs/contra_care/bm25_patient_results.csv \
  --output_csv src/bedrock_pipeline/bedrock_prompts/classify/contra_care/bm25_prompts.csv \
  --task classify
```
**Parameters:**
- `--retrieval_csv`: Path to retrieval csv (required)
- `--output_csv`: Path to save generated Bedrock prompt csv (required)
- `--task`: Contains 'classify' (required)

To create prompts for the Baseline LLMs, use the csv that was generated in STEP 5 in the folder `src/llm/outputs/` as the retrieval csv.

#### 2. Call Bedrock: 
`call_bedrock.py` sends generated prompts to an Amazon Bedrock hosted LLM (e.g., DeepSeek) and saves the model responses to disk. It is designed to operate on prompt CSVs produced by `prompt_generation.py` and follows a standardized directory structure to automatically infer task and retrieval method.

The scipt expects the prompt CSV path to follow this structure:
```php-template
src/bedrock_pipeline/bedrock_prompts/<task>/<retrieval_method>_prompts*.csv
```
From this path, the script automatically infers the task (classify) and the retrieval method (BM25, FAISS, hybrid, baseline, etc). 

To call a Bedrock model, run:
```bash
python src/bedrock_pipeline/call_bedrock.py \
  --prompt_csv src/bedrock_pipeline/bedrock_prompts/classify/contra_care/bm25_prompts.csv \ 
  --output_csv src/bedrock_pipeline/bedrock_responses/classify/contra_care/bm25_responses.csv 
```
**Parameters:**
- `--prompt_csv`: Path to a generated `bedrock_prompts` CSV containing prompts for a specific task and retrieval method (required)
- `--output_csv`: Path to save bedrock responses (required)

When running on large amounts of patients, use `bedrock_parallel.py` to parallelize theh calls: 
```bash
python src/bedrock_pipeline/bedrock_parallel.py \
  --prompt_csv src/bedrock_pipeline/bedrock_prompts/classify/contra_care/bm25_prompts.csv \ 
  --output_csv src/bedrock_pipeline/bedrock_responses/classify/contra_care/bm25_responses.csv \
  --workers 10
```
**Parameters:**
- `--prompt_csv`: Path to a generated `bedrock_prompts` CSV containing prompts for a specific task and retrieval method (required)
- `--output_csv`: Path to save bedrock responses (required)
- `workers`: Number of concurrent API calls where limit is 20 (optional)
- `sleep_time`: Delay between calls (optional)


### STEP 8: Evaluate Bedrock LLM

#### 1. For classification methods:
```bash
python src/eval/patient_level/classify_eval.py \
  --element "contra_care"
```

**Parameters:**
- `element`: Clinical element to process (required)

### STEP 9: Generate Visualizations
Assuming that steps 4-8 were run for all five elements then visualizations can be created.

Visualizations for Bedrock evaluation can be generated by:
```bash
python src/results_patient_level/visualize.py
python src/results_patient_level/visualize_element_retrieval.py
```

Visualizations for retrieval evaluation can be generated by:
```bash
python src/results_patient_level/visualize_retrieval.py \
 --element "contra_care"
```

**Parameters:**
- `element`: Clinical element to process (required)

## Notes
- `.gitignore` is configured to exclude `.DS_Store`, `__pycache__`, and sensitive MIMIC data. The project demonstrates methodology on a small subset of the data; scaling to full MIMIC requires additional setup and approvals.

- The Specifications Manual for National Hospital Inpatient Quality Measures used to generate synthetic needles in "needle-in-a-haystack" approach can be found at this website (Version 5.18a): 
[CMS Manual](https://qualitynet.cms.gov/inpatient/specifications-manuals)