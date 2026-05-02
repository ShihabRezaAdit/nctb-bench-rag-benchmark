# Reproducibility Guide

This document provides everything needed to exactly reproduce the results reported
in the NCTBench paper from scratch.

---

## Hardware Used

| Component | Specification |
|---|---|
| GPU | NVIDIA (CUDA-enabled, ≥8 GB VRAM recommended) |
| RAM | ≥16 GB |
| Storage | ≥10 GB free (OCR outputs + ChromaDB) |
| OS | Linux / Windows (tested on both) |

CPU-only execution is possible but Step 2 (EasyOCR) will be significantly slower.

---

## Software Versions

| Package | Version |
|---|---|
| Python | 3.9+ |
| PyMuPDF (fitz) | ≥1.23 |
| EasyOCR | ≥1.7 |
| sentence-transformers | ≥2.6 |
| chromadb | ≥0.4 |
| rank-bm25 | ≥0.2.2 |
| together | ≥1.1 |
| datasets | ≥2.18 |
| evaluate | ≥0.4 |
| bert-score | ≥0.3.13 |
| rouge-score | ≥0.1.2 |

See `requirements.txt` for the full pinned list.

---

## Step-by-Step Reproduction

### Prerequisites

1. Clone this repository:
   ```bash
   git clone https://github.com/ShihabRezaAdit/nctb-bench-rag-benchmark.git
   cd nctb-bench-rag-benchmark
   ```

2. Create and activate environment:
   ```bash
   conda create -n nctbench python=3.9 -y
   conda activate nctbench
   pip install -r requirements.txt
   ```

3. Set up API key:
   ```bash
   cp .env.example .env
   # Edit .env and add: TOGETHER_API_KEY=your_key_here
   ```

4. Obtain NCTB PDFs — 16 pilot textbooks (Classes 6–9, ICT + Science, Bengali + English)
   and place them in `data/pdfs/` following the naming convention:
   ```
   {Subject}_{lang}_class{N}.pdf
   e.g.  ICT_ben_class7.pdf   Science_eng_class9.pdf
   ```

---

### Stage A — Dataset Construction

```bash
# Step 1: Corpus audit — validates all 16 PDFs are present and readable
python src/step1_audit.py
# Output: data/manifest.json (16 records)

# Step 2: Text extraction (GPU recommended for Bengali OCR)
python src/step2_extract.py
# Output: data/ocr_results/ + data/extracted_text/ (1,374 page JSONs)

# Step 3: Sliding-window chunking
python src/step3_chunk.py
# Output: data/chunks/all_chunks.json (6,887 chunks)

# Step 4: Semantic embedding → ChromaDB
python src/step4_embed.py
# Output: chromadb/ collection "nctb_pilot" (6,887 vectors, 1024-dim)

# Step 5: QA generation (requires TOGETHER_API_KEY, ~2–4 hours)
python src/step5_generate.py
# Output: data/qa_pairs/qa_*.json (~24,853 raw pairs)

# Step 6: AMSV 4-stage validation
python src/step6_validate.py
# Output: data/qa_pairs/validated_pairs.json (12,403 pairs)
#         data/qa_pairs/test_set.json (200 pairs)
```

---

### Stage B — RAG Evaluation

The validated dataset is available directly from HuggingFace — you do **not** need
to re-run Stage A to reproduce evaluation results:

```python
from datasets import load_dataset
ds = load_dataset("ShihabReza/nctb-qa")
# train: 12,203 pairs | test: 200 pairs
```

```bash
# Phase 1: 12 base configurations (BM25 / Dense / Hybrid × Llama / Qwen × Grounded / Open)
python evaluation/run_evaluation.py
# Output: evaluation/results/config_0[1-12]_results.json

# Phase 2: 3 enhanced configurations (cross-encoder reranking + MMR)
python evaluation/run_evaluation_enhanced.py
# Output: evaluation/results/config_1[3-5]_results.json

# Add ROUGE-L and BERTScore to Phase 1 results
python evaluation/rescore_existing.py
# Output: evaluation/results/enriched_summary.csv

# Aggregate all 15 configs into final results table
python evaluation/finalize_results.py
# Output: evaluation/results/enriched_summary_all.csv
```

Or run everything at once:
```bash
bash run_pipeline.sh
```

---

## Expected Results

After running the full pipeline, `evaluation/results/enriched_summary_all.csv`
should match the values below (within rounding):

### Phase 2 — Best Configurations

| Config | Retriever | Generator | Token F1 | ROUGE-L | BERTScore | Hall. Rate |
|---|---|---|---|---|---|---|
| C14 | Hybrid+Rerank+MMR | Llama-3.3-70B | 0.386 | 0.377 | 0.876 | 38.5% |
| C13 | Dense+Rerank+MMR  | Llama-3.3-70B | 0.376 | 0.367 | 0.874 | 38.0% |
| C15 | BM25+Rerank+MMR   | Llama-3.3-70B | 0.341 | 0.335 | 0.863 | 32.5% |

### Phase 1 — Top Grounded Configurations

| Config | Retriever | Generator | Prompt | Token F1 | ROUGE-L | BERTScore |
|---|---|---|---|---|---|---|
| C5  | Dense  | Llama-3.3-70B | Grounded | 0.374 | 0.421 | 0.892 |
| C11 | Hybrid | Llama-3.3-70B | Grounded | 0.337 | 0.430 | 0.894 |

Pre-computed results are committed at `evaluation/results/enriched_summary_all.csv`
so you can verify evaluation scripts without re-running Stage A or Phase 1.

---

## Known Sources of Non-Determinism

| Source | Impact | Mitigation |
|---|---|---|
| Together AI LLM output | QA generation (Step 5) varies across runs | Use pre-built dataset from HuggingFace |
| EasyOCR GPU batching | Minor text differences across GPU models | Results reported from NVIDIA GPU run |
| Generator responses | Evaluation answers vary slightly | Reported metrics are means over 200 items |

Evaluation metrics (Token F1, ROUGE-L, BERTScore) are **deterministic** once
the test set and model outputs are fixed. The test set is frozen in
`data/qa_pairs/test_set.json` and on HuggingFace.

---

## Skipping Stage A (Recommended for Evaluation-Only)

If you only want to reproduce evaluation results without re-building the dataset:

```bash
# 1. Download the dataset from HuggingFace (automatic via datasets library)
# 2. Run evaluation directly — no PDFs or API key needed
python evaluation/run_evaluation.py
python evaluation/run_evaluation_enhanced.py
python evaluation/finalize_results.py
```

The evaluation scripts load the test set from HuggingFace automatically if
local `data/qa_pairs/test_set.json` is not found.

---

## Contact

For reproducibility questions, open a GitHub Issue or contact:
**Md Shihab Reza** — shihab.reza@northsouth.edu
