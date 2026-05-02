# NCTBench: When the Textbook Becomes the Oracle
## A Dual-Language Retrieval-Augmented QA Benchmark for Low-Resource Educational NLP

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/ShihabReza/nctb-qa)

**NCTBench** is a dual-language (Bengali/English) question-answering benchmark built from
official NCTB (National Curriculum and Textbook Board, Bangladesh) pilot textbooks for
Classes 6–9, covering ICT and Science. The repository contains the full pipeline to reproduce
the dataset and all 15 RAG system evaluations reported in our paper.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Setup](#setup)
- [Reproduction](#reproduction)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Problem Statement

Most QA benchmarks are built from English Wikipedia or web text. Low-resource educational
domains — particularly those involving South Asian scripts — lack validated benchmarks for
RAG evaluation. This forces practitioners to use ad-hoc evaluation methods without ground
truth.

**NCTBench** fills this gap by providing:

1. **12,403 validated QA pairs** from NCTB textbooks with evidence spans and full metadata
2. A **reproducible construction pipeline** with four-stage answer validation (AMSV)
3. **15 RAG configurations** benchmarked on a 200-item stratified test set, covering BM25,
   dense, hybrid, and reranked retrieval

---

## Architecture

### Stage A — Dataset Construction

```
16 NCTB PDFs
    │
    ▼
Text Extraction
  ├── English: PyMuPDF (digital text, no OCR)
  └── Bengali: EasyOCR (GPU, 200 DPI, lang=["en","bn"])
    │                          → 1,374 pages
    ▼
Text Chunking
  └── Sliding window: 1,000 chars / 200 overlap / sentence boundaries
    │                          → 6,887 chunks
    ▼
Semantic Embedding
  └── intfloat/multilingual-e5-large → ChromaDB (cosine, L2-normalized)
    │
    ▼
QA Generation
  └── Llama-3.3-70B-Instruct-Turbo (Together AI), 2 pairs/chunk
    │                          → 24,853 raw pairs
    ▼
AMSV Validation (4 filters)
  ├── S1: Semantic overlap (cosine ≥ 0.25)       → 19,639
  ├── S2: Answer quality (no stop-words/stubs)   → 15,095
  ├── S3: Deduplication (cosine < 0.97)          → 13,622
  └── S4: Retrieval grounding (source in top-3)  → 12,403
    │                          CMAR = 91.05%
    ▼
Train / Test Split
  ├── Train: 12,203 pairs
  └── Test:    200 pairs (stratified: language × subject × grade)
```

### Stage B — RAG Evaluation

```
Test Query
    │
    ▼
Retriever
  ├── BM25:   BM25Okapi + Bengali stemming + query expansion
  ├── Dense:  multilingual-e5-large + ChromaDB (top-10)
  ├── Hybrid: Reciprocal Rank Fusion (0.7×dense + 0.3×BM25, top-50)
  └── Rerank: First-stage top-50 → cross-encoder (ms-marco-MiniLM-L-6-v2)
              → score fusion → MMR (λ=0.6) → top-5
    │
    ▼
Generator (top-5 passages as context)
  ├── Llama-3.3-70B-Instruct-Turbo (grounded / open)
  └── Qwen-2.5-7B-Instruct-Turbo (grounded / open)
    │
    ▼
Evaluation Metrics
  Token F1 | ROUGE-L | BERTScore | Hallucination Rate | Recall@K
```

---

## Dataset

The dataset is available on Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("ShihabReza/nctb-qa")
```

**Fields:**

| Field | Type | Description |
|---|---|---|
| `question` | string | Question in Bengali or English |
| `answer` | string | Gold answer from the textbook |
| `evidence` | string | Verbatim textbook sentence supporting the answer |
| `source_text` | string | Full source chunk |
| `chunk_id` | int | Source chunk ID |
| `source` | string | Source PDF filename |
| `subject` | string | `ICT` or `Science` |
| `language` | string | `bn` or `en` |
| `class_num` | int | 6, 7, 8, or 9 |
| `grade_band` | string | `junior_sec` or `ssc` |
| `page_num` | int | Page number in source PDF |

**Statistics:**

| Property | Value |
|---|---|
| Total pairs | 12,403 |
| Train | 12,203 |
| Test | 200 |
| CMAR | 91.05% |
| Languages | Bengali (`bn`), English (`en`) |
| Subjects | ICT, Science |
| Classes | 6, 7, 8, 9 |
| Source books | 16 NCTB PDFs |
| Source chunks | 6,887 |

---

## Setup

### Requirements

- Python 3.9+
- CUDA GPU (strongly recommended for EasyOCR and embedding steps)
- Together AI API key (for QA generation; not needed for re-evaluation)

### Installation

```bash
git clone https://github.com/ShihabReza/nctb-bench-rag-benchmark.git
cd nctb-bench-rag-benchmark

# Create conda environment
conda create -n nctbench python=3.9 -y
conda activate nctbench

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy and edit the environment file
cp pipeline/.env.example pipeline/.env
# Add your Together AI API key to pipeline/.env:
# TOGETHER_API_KEY=your_key_here
```

### Environment variables

| Variable | Description |
|---|---|
| `TOGETHER_API_KEY` | Together AI API key (Step 5 only) |
| `CHROMADB_PATH` | ChromaDB storage path (default: `./chromadb`) |

---

## Reproduction

The full pipeline can be run end-to-end with:

```bash
bash run_pipeline.sh
```

Or step by step:

```bash
# Stage A: Dataset Construction
python src/step1_audit.py          # Corpus audit → manifest.json
python src/step2_extract.py        # Text extraction → ocr_results/
python src/step3_chunk.py          # Chunking → chunks/all_chunks.json
python src/step4_embed.py          # Embedding → ChromaDB collection
python src/step5_generate.py       # QA generation → raw_qa_pairs.json
python src/step6_validate.py       # AMSV validation → validated_qa.json

# Stage B: RAG Evaluation
python evaluation/run_evaluation.py          # Phase 1: 12 base configs
python evaluation/run_evaluation_enhanced.py # Phase 2: 3 reranking configs

# Rescore existing results with all metrics
python evaluation/rescore_existing.py

# Aggregate final results
python evaluation/finalize_results.py
```

### Expected outputs

| Step | Output | Size |
|---|---|---|
| Step 1 | `data/manifest.json` | 16 records |
| Step 2 | `data/ocr_results/` | 1,374 page JSONs |
| Step 3 | `data/chunks/all_chunks.json` | 6,887 chunks |
| Step 4 | ChromaDB collection `nctb_pilot` | 6,887 vectors |
| Step 5 | `data/raw_qa_pairs.json` | ~24,853 pairs |
| Step 6 | `data/validated_qa.json` | 12,403 pairs |
| Eval | `evaluation/results/enriched_summary_all.csv` | 15 configs |

---

## Results

All results are on the 200-item stratified test set.

### Phase 2 — Enhanced (Cross-encoder Reranking + MMR + Strict Prompt)

| Configuration | Token F1 | ROUGE-L | BERTScore | Hall. Rate |
|---|---|---|---|---|
| **Hybrid + Rerank + MMR + Llama** | **0.386** | **0.377** | 0.876 | 38.5% |
| Dense + Rerank + MMR + Llama | 0.376 | 0.367 | 0.874 | 38.0% |
| BM25 + Rerank + MMR + Llama | 0.341 | 0.335 | 0.863 | 32.5% |

### Phase 1 — Base Configurations (top grounded results)

| Configuration | Token F1 | ROUGE-L | BERTScore | Hall. Rate |
|---|---|---|---|---|
| Dense + Llama + Grounded | 0.374 | 0.421 | 0.892 | 50.8% |
| Hybrid + Llama + Grounded | 0.337 | 0.430 | **0.894** | 51.0% |
| BM25 + Llama + Grounded | 0.000 | 0.405 | 0.880 | 0.0%* |

> *BM25+Grounded Token F1=0.000: the strict grounded prompt causes the model to return
> "Not found in context." when the exact answer passage is not in the top-3. See paper §7.1.

### Language breakdown (Best config: C14)

| Language | Token F1 | ROUGE-L | BERTScore |
|---|---|---|---|
| English | 0.516 | 0.505 | 0.908 |
| Bengali | 0.238 | 0.230 | 0.839 |

Full results with confidence intervals and subject breakdowns:
`evaluation/results/enriched_summary_all.csv`

---

## Repository Structure

```
nctb-bench-rag-benchmark/
├── src/                        # Pipeline source code
│   ├── step1_audit.py          # Corpus audit
│   ├── step2_extract.py        # PDF text extraction (PyMuPDF + EasyOCR)
│   ├── step3_chunk.py          # Sliding window chunking
│   ├── step4_embed.py          # Embedding + ChromaDB indexing
│   ├── step5_generate.py       # QA pair generation (Llama-3.3-70B)
│   └── step6_validate.py       # AMSV 4-filter validation
├── evaluation/
│   ├── metrics.py              # Token F1, ROUGE-L, BERTScore, CI95
│   ├── retrievers.py           # BM25, Dense, Hybrid, Reranking retrievers
│   ├── generators.py           # Grounded and open prompt generators
│   ├── run_evaluation.py       # Phase 1: 12 base configurations
│   ├── run_evaluation_enhanced.py  # Phase 2: 3 reranking configurations
│   ├── rescore_existing.py     # Add ROUGE-L/BERTScore to existing results
│   ├── finalize_results.py     # Aggregate all 15 configs
│   └── results/
│       ├── config_XX_results.json  # Per-item results (15 files)
│       ├── enriched_summary.csv    # Configs 1–12 with all metrics
│       └── enriched_summary_all.csv  # All 15 configs (final)
├── notebooks/
│   ├── 01_audit.ipynb              # Corpus exploration
│   ├── 02_extract.ipynb            # OCR quality checks
│   ├── 03_chunk_and_embed.ipynb    # Chunk distribution analysis
│   ├── 04_qa_generation.ipynb      # QA sample inspection
│   ├── 05_amsv_validation.ipynb    # AMSV filter-by-filter analysis
│   ├── 06_rag_evaluation.ipynb     # RAG evaluation runs
│   └── 07_enhanced_results.ipynb  # Final results visualization
├── data/
│   ├── manifest.json           # Book metadata (16 records)
│   ├── chunks/
│   │   └── all_chunks.json     # 6,887 chunks with metadata
│   ├── qa_pairs/
│   │   ├── validated_qa.json   # 12,403 validated pairs
│   │   ├── train.json          # 12,203 training pairs
│   │   └── test.json           # 200 gold test items
│   └── ocr_results/            # Per-book extracted text
│       └── {class}_{lang}/
│           └── *.json
├── configs/
│   ├── phase1.yaml             # 12 base configuration specs
│   └── phase2.yaml             # 3 enhanced configuration specs
├── requirements.txt
├── run_pipeline.sh             # End-to-end pipeline runner
├── CITATION.cff                # Machine-readable citation (GitHub "Cite this repository")
├── CONTRIBUTING.md             # Contribution guidelines
├── REPRODUCIBILITY.md          # Step-by-step reproduction guide
├── CODE_OF_CONDUCT.md
└── README.md
```

---

## Authors

| Name | Affiliation | Role |
|---|---|---|
| **Md Shihab Reza** | North South University, Dhaka, Bangladesh | Lead Author |
| **Sk Muktadir Hossain** | American International University-Bangladesh (AIUB) | Co-Author |
| **Kazi Abdur Rahman** | American International University-Bangladesh (AIUB) | Co-Author |
| **Dr. Muhammad Hasibur Rashid Chayon** | American International University-Bangladesh (AIUB) | Supervisor |

---

## Citation

```bibtex
@dataset{nctb_qa_2026,
  title     = {NCTBench: When the Textbook Becomes the Oracle — A Dual-Language
               Retrieval-Augmented QA Benchmark for Low-Resource Educational NLP},
  author    = {Reza, Md Shihab and
               Hossain, Sk Muktadir and
               Rahman, Kazi Abdur and
               Chayon, Muhammad Hasibur Rashid},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/ShihabReza/nctb-qa},
  license   = {CC BY 4.0},
  note      = {North South University; American International University-Bangladesh (AIUB)}
}
```

---

## License

| Component | License |
|---|---|
| **Code** | [MIT License](LICENSE) |
| **Dataset** | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |

The source textbooks are published by the National Curriculum and Textbook Board (NCTB),
Bangladesh, and are publicly available for educational use.

---

## Contact

**Md Shihab Reza** — shihab.reza@northsouth.edu | shihabreza.nsu@gmail.com
North South University, Dhaka, Bangladesh
