# LLMs as Information Extractors

This project explores the use of Large Language Models (LLMs) to extract and classify information from unstructured clinical notes, such as those found in Electronic Health Records (EHRs). By combining LLMs with retrieval-augmented generation (RAG) techniques, we aim to improve extraction accuracy for long and complex clinical notes.

## Introduction

Medical notes are a critical part of healthcare, providing comprehensive patient information from allergies to past complications. However, these notes are often unstructured, contain domain specific abbreviations, and require significant manual effort to analyze. Hospitals spend millions annually on maintaining and reporting medical notes, consuming thousands of hours that could be more beneifical if spend on treating patients.

LLMs, such as GPT-4, Claude 3, and Gemini Advanced, have demonstrated the ability to accurately identify key details in medical text, often matching or surpassing traditional machine learning methods. In this project, we investigate LLM performance on MIMIC-III clinical data, exploring how RAG and other retrieval techniques can enhance reliability for nuanced or context-rich information.

## Features
- Extracts information from unstructured clinical notes
- Supports multiple retrieval strategies (BM25, FAISS using euclidean distance, FAISS using cosine similarity, Hybrid)
- Evaluates LLM performance on classification and extraction tasks
- Generates visualizations for accuracy and retrieval performance
- Can be extended to other clinical datasets or synthetic notes

## Methodology

1. **Data Preparation**: Patient-level haystacks of clinical notes are created using MIMIC-III structured and unstructured data. Synthetic needles are also used for testing. Inserting the synthetic needles into the haystacks allows for a Needle in a Haystack (NIAH) approach. 
2. **Retrieval**: Implements BM25, FAISS (Euclidean and Cosine), and hybrid retrievers to provide relevant context for LLMs.
3. **LLM Evaluation**: Uses AWS Bedrock or other LLMs to extract or classify information. Evaluations include both extraction and classification tasks.
4. **Visualization**: Results are visualized via heatmaps, accuracy plots, and other summary charts.

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

## Dataset

- **MIMIC-III**: Deidentified clinical data for ~46,000 patients at Beth Israel Deaconess Medical Center (2001–2012). Includes structured data (demographics, vitals, labs, medications) and unstructured clinical notes.
- **Synthetic Needles**: Generated data (through prompting of Bedrock) used to test retrieval and extraction without patient privacy concerns.
**Important**: MIMIC data is sensitive and cannot be shared publicly. Do not commit MIMIC files to GitHub.

## Results & Visualization
- Extraction and classification accuracy per LLM
- Strategy performance (BM25, FAISS, Hybrid)
- Heatmaps comparing overall and per-category performance
- Visualizations highlight LLM strengths and limitations for nuanced text