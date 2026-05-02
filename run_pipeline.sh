#!/usr/bin/env bash
# NCTBench — End-to-end pipeline runner
# Run from the repository root: bash run_pipeline.sh
# GPU strongly recommended (EasyOCR + embedding steps).
#
# All paths are controlled by src/config.py.
# Place your NCTB PDFs in data/pdfs/ before running.
# Add TOGETHER_API_KEY=your_key to a .env file in this directory.

set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "================================================================"
echo "NCTBench Pipeline"
echo "================================================================"

# ── Stage A: Dataset Construction ────────────────────────────────────────────

echo ""
echo "=== Step 1: Corpus Audit ==="
cd src && $PYTHON step1_audit.py && cd ..
echo "Step 1 complete -> data/manifest.json"

echo ""
echo "=== Step 2: Text Extraction ==="
cd src && $PYTHON step2_extract.py && cd ..
echo "Step 2 complete -> data/ocr_results/ and data/extracted_text/"

echo ""
echo "=== Step 3: Text Chunking ==="
cd src && $PYTHON step3_chunk.py && cd ..
echo "Step 3 complete -> data/chunks/all_chunks.json"

echo ""
echo "=== Step 4: Semantic Embedding and Indexing ==="
cd src && $PYTHON step4_embed.py && cd ..
echo "Step 4 complete -> chromadb/ (collection: nctb_pilot)"

echo ""
echo "=== Step 5: QA Pair Generation ==="
echo "NOTE: Requires TOGETHER_API_KEY in .env"
cd src && $PYTHON step5_generate.py && cd ..
echo "Step 5 complete -> data/qa_pairs/qa_*.json"

echo ""
echo "=== Step 6: AMSV Validation ==="
cd src && $PYTHON step6_validate.py && cd ..
echo "Step 6 complete -> data/qa_pairs/validated_pairs.json + test_set.json"

# ── Stage B: RAG Evaluation ───────────────────────────────────────────────────

echo ""
echo "=== RAG Evaluation: Phase 1 (12 base configurations) ==="
cd evaluation && $PYTHON run_evaluation.py && cd ..
echo "Phase 1 complete -> evaluation/results/config_0[1-12]_results.json"

echo ""
echo "=== RAG Evaluation: Phase 2 (3 enhanced configurations) ==="
cd evaluation && $PYTHON run_evaluation_enhanced.py && cd ..
echo "Phase 2 complete -> evaluation/results/config_1[3-5]_results.json"

echo ""
echo "=== Rescoring with full metrics (ROUGE-L, BERTScore, CI95) ==="
cd evaluation && $PYTHON rescore_existing.py && cd ..
echo "Rescoring complete -> evaluation/results/enriched_summary.csv"

echo ""
echo "=== Finalizing all 15 configs ==="
cd evaluation && $PYTHON finalize_results.py && cd ..
echo "Finalization complete -> evaluation/results/enriched_summary_all.csv"

echo ""
echo "================================================================"
echo "NCTBench pipeline complete."
echo "  Dataset : data/qa_pairs/validated_pairs.json"
echo "  Test set: data/qa_pairs/test_set.json"
echo "  Results : evaluation/results/enriched_summary_all.csv"
echo "================================================================"
