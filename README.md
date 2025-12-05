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
        ├── README.md                 # Main project overview
        ├── requirements.txt
        ├── .gitignore
        ├── data/
        │   └── mimic/                # MIMIC data (instructions in data/mimic/README.md)
        ├── src/
        │   ├── bedrock_pipeline/     # LLM prompt & Bedrock pipeline
        │   ├── needles/              # Synthetic needle generation
        │   ├── eval/                 # LLM evaluation scripts
        │   ├── results/              # Visualization scripts & output
        │   ├── retrieval/            # Retrieval methods
        │   └── haystacks/            # Functions for inserting needles and constructing haystacks
        └── Support_Docs              # Past EDA & Literature Review
    </code>
</pre>

## Features
- Evaluate LLMs on information extraction from clinical notes. 
- Extracts information from unstructured clinical notes
- Supports multiple LLMs (GPT-4, Claude 3, Gemini Advanced). 
- Retrieval-augmented generation (RAG) using multiple retrieval strategies:
    - BM25: 
    - FAISS using euclidean distance:
    - FAISS using cosine similarity:
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

3. Install Dependencies: `pip install requirements.txt`

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

1. Place MIMIC-III data in `data/mimic/`. 

2. Generate synthetic needles using `src/needles/generate_needles.py`. The needles were 

3. Create haystack by inserting needles into notes in `NOTEEVENTS.csv` by using `src/haystacks/insert_needle.py`.

4. Run retrieval methods:
```bash
python src/retrieval/bm25_retrieval.py
python src/retrieval/faiss_cos_retrieval.py
python src/retrieval/faiss_cos_retrieval.py
python src/retrieval/hybrid_retrieval.py
```
5. Generate prompts and run Bedrock LLM based on prompts by using `src/bedrock_pipeline/`:

Prompts are generated on the top-k retrieved passages by running:
```bash
python src/bedrock_pipeline/prompt_generation.py
```
where the task could be "classify", "extract", or "summarize" and the retrieval file could be from any of the four retrieval methods.

Bedrock is called using any of the generated prompts by running:
```bash
python src/bedrock_pipeline/call_bedrock.py
```

6. Evaluate Bedrock LLM for classification or extraction methods:
```bash
python src/eval/extract_llm_eval.py
python src/eval/classify_llm_eval.py
```
7. Generate Visualizations:

Visualizations for Bedrock evaluation can be generated by:
```bash
python src/results/visualize_eval.py
```
Visualizations for retrieval evaluation can be generated by:
```bash
python src/results/visualize_retrieval.py
```

## Notes
`.gitignore` is configured to exclude `.DS_Store`, `__pycache__`, and sensitive MIMIC data. The project demonstrates methodology on a small subset of the data; scaling to full MIMIC requires additional setup and approvals.