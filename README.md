# RAG Pipeline for Malaysian Financial Disclosures

> Master's thesis project — Optimizing Retrieval-Augmented Generation (RAG) for Malaysian Financial Disclosures using Layout-aware Processing and Hybrid Retrieval

---

## Overview

Financial reports published by Bursa Malaysia-listed companies are structurally complex documents containing dense financial tables, regulatory abbreviations, and multi-column layouts that challenge standard RAG systems. This project implements and evaluates three RAG pipeline configurations in a controlled ablation study to identify which combination of document parsing, chunking, and retrieval strategy best serves financial document question answering.

The study evaluates 30 questions across 14 companies from 10 industry sectors using the RAGAS framework, with results stratified by question type (FIN / RISK / GOV / OPS / SUS / REG) and document layout sensitivity (High / Medium / Low).

---

## Pipeline Design

Three pipelines are compared in an incremental A → B → C ablation sequence. Each path introduces one change over the previous while holding all other components constant.

| Component | Path A — Baseline | Path B — Layout-Aware | Path C — Hybrid Retrieval |
|---|---|---|---|
| **Extraction** | PyMuPDF + pdfplumber fallback | LlamaParse (markdown mode) | Same as Path B |
| **Chunking** | Fixed-token (300 tok, 50 overlap) | Structural / table-aware | Same as Path B |
| **Retrieval** | FAISS dense, top-5 | FAISS dense, top-10 | BM25 + FAISS fusion + cross-encoder reranking, top-10 |
| **Embedding** | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| **Generator** | llama3.1:8b (Ollama) | llama3.1:8b (Ollama) | llama3.1:8b (Ollama) |
| **Evaluation** | RAGAS v0.1.21 | RAGAS v0.1.21 | RAGAS v0.1.21 |

**Score fusion formula (Path C):**
`hybrid_score = 0.7 × dense_norm + 0.3 × BM25_norm`

---

## Results

**Aggregate RAGAS scores across all 30 questions:**

| Metric | Path A | Path B | Path C | A→C Δ |
|---|---|---|---|---|
| Context Precision | 0.841 | 0.790 | **0.877** | +0.036 |
| Context Recall | 0.615 | 0.620 | **0.703** | +0.088 |
| Faithfulness | 0.636 | 0.646 | **0.727** | +0.090 |
| Answer Relevancy | 0.386 | 0.505 | **0.534** | +0.148 |
| Answer Rate | 67% | 73% | **83%** | +16% |

Path C wins across all four metrics. The B→C retrieval transition contributes gains an order of magnitude larger than the A→B document processing transition.

**Key finding — RISK questions:** Dense-only retrieval records near-zero context recall (0.125) for regulatory abbreviation queries (CET1, GIL, LCR). BM25 exact-match retrieval recovers context recall to 0.750 without embedding fine-tuning.

---

## Repository Structure

```
rag_research/
├── src/
│   ├── extraction/         # Path A (PyMuPDF) and Path B (LlamaParse) extractors
│   ├── chunking/           # Fixed-token chunker (Path A) and structural chunker (Path B)
│   ├── embeddings/         # Sentence-transformer embedder (shared)
│   ├── vectorstore/        # FAISS index build and search (shared)
│   ├── retrieval/          # Dense retriever, BM25 store, hybrid retriever, reranker
│   ├── generation/         # Ollama LLM generator (shared)
│   └── evaluation/         # RAGAS evaluator and report generator
├── configs/
│   └── config.yaml         # All pipeline parameters (no hardcoded values)
├── questions/
│   ├── eval_questions.json # 30 evaluation questions with ground truth answers
│   └── eval_question.csv   # Appendix-ready question table
├── results/
│   ├── path_a/             # Path A RAGAS scores (CSV + JSON)
│   ├── path_b/             # Path B RAGAS scores (CSV + JSON)
│   ├── path_c/             # Path C RAGAS scores (CSV + JSON)
│   └── reports/            # Comparison report, stratified report, charts
├── run_pipeline.py         # Main entry point
├── test_pipeline.py        # Pipeline smoke tests
└── requirements.txt        # Python dependencies
```

> **Note:** `data/raw/` (source PDFs) is excluded from this repository due to copyright. Download annual reports directly from [Bursa Malaysia's public disclosure portal](https://www.bursamalaysia.com).

---

## Setup

**Requirements:** Python 3.11, [Ollama](https://ollama.com) installed and running locally.

```bash
# 1. Clone the repo
git clone https://github.com/xuanxy20/rag-malaysian-financial-disclosures.git
cd rag-malaysian-financial-disclosures

# 2. Create and activate environment
conda create -n rag-fin python=3.11
conda activate rag-fin

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM
ollama pull llama3.1:8b

# 5. Add your LlamaParse API key (required for Path B and C)
echo "LLAMA_CLOUD_API_KEY=your_key_here" > .env

# 6. Place source PDFs in data/raw/ following the filenames in configs/config.yaml
```

---

## Running the Pipeline

```bash
# Run all three paths end-to-end
python run_pipeline.py --path all

# Run a specific path
python run_pipeline.py --path a
python run_pipeline.py --path b
python run_pipeline.py --path c

# Run evaluation only (if pipeline outputs already exist)
python run_pipeline.py --eval-only
```

Results are saved to `results/path_a/`, `results/path_b/`, `results/path_c/` and a comparison report to `results/reports/`.

---

## Evaluation Questions

30 manually authored questions across 14 Bursa Malaysia-listed companies, annotated by question type and layout sensitivity. See [`questions/eval_question.csv`](questions/eval_question.csv) for the full question set with ground truth answers and source page references.

| Type | n | Description |
|---|---|---|
| FIN | 18 | Financial metrics, revenue, profit, ratios |
| OPS | 4 | Operational highlights, segment performance |
| GOV | 3 | Board composition, remuneration |
| RISK | 2 | Capital adequacy (CET1, LCR, GIL) |
| SUS | 2 | ESG and sustainability disclosures |
| REG | 1 | Regulatory framework compliance |

---

## Tech Stack

- **Extraction:** PyMuPDF, pdfplumber, LlamaParse
- **Chunking:** tiktoken (cl100k_base tokenizer)
- **Embeddings:** sentence-transformers / all-MiniLM-L6-v2
- **Vector index:** FAISS IndexFlatIP
- **Sparse retrieval:** rank-bm25 (BM25Okapi)
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM:** llama3.1:8b via Ollama (temperature = 0)
- **Evaluation:** RAGAS v0.1.21

---

## Citation

If you reference this work, please cite:

```
Lee, X. Y. (2026). Optimizing Retrieval-Augmented Generation (RAG) for Malaysian Financial
Disclosures using Layout-aware Processing and Hybrid Retrieval [Master's thesis].
```
