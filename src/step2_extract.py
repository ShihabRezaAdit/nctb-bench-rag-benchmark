# step2_extract.py
"""
Extract text from PDFs.
  - PyMuPDF  → English PDFs with embedded text
  - EasyOCR  → Bengali PDFs (image-based)

Key design decisions:
  1. EasyOCR model is loaded ONCE per run, not per page (saves 30-60s per PDF)
  2. GPU memory is explicitly released after each OCR PDF
  3. Already-extracted files are skipped (idempotency)
  4. Output written atomically via save_json
"""
import gc
import logging
import fitz
import numpy as np
from pathlib import Path

from config import (NCTB_ROOT, EXTRACTED_DIR, OCR_DIR, MANIFEST_PATH,
                    OCR_LANGUAGES, OCR_GPU, OCR_BATCH_SIZE, PDF_DPI, LOGS_DIR)
from utils import load_json, save_json, make_output_path, setup_logging

log = setup_logging("extract", LOGS_DIR)


# ── PyMuPDF Extraction ────────────────────────────────────────────────────────
def extract_pymupdf(full_path: str) -> list[dict]:
    """Returns list of {page_num, text} dicts."""
    doc   = fitz.open(full_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


# ── EasyOCR Extraction ────────────────────────────────────────────────────────
def extract_easyocr(full_path: str, reader) -> list[dict]:
    """
    Render each PDF page to an image, then run EasyOCR.
    reader is passed in (created ONCE outside this function).
    """
    doc = fitz.open(full_path)
    mat = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)
    pages = []

    for i in range(len(doc)):
        pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        # Convert pixmap to numpy array (RGB uint8)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        results = reader.readtext(img, detail=0, paragraph=True)
        text    = "\n".join(results).strip()
        pages.append({"page_num": i + 1, "text": text})

        if (i + 1) % 10 == 0:
            log.info("  OCR progress: page %d / %d", i + 1, len(doc))

    doc.close()
    return pages


# ── Main Runner ───────────────────────────────────────────────────────────────
def run_extraction() -> None:
    manifest = load_json(MANIFEST_PATH)
    if not manifest:
        log.error("manifest.json not found — run step 1 (audit) first")
        return

    entries = [v for v in manifest.values() if v.get("status") == "audited"]
    if not entries:
        log.error("No audited entries in manifest")
        return

    # Separate by method
    pymupdf_entries = [e for e in entries if e["extraction_method"] == "pymupdf"]
    ocr_entries     = [e for e in entries if e["extraction_method"] == "easyocr"]

    # ── PyMuPDF pass ──────────────────────────────────────────────────────────
    for entry in pymupdf_entries:
        key      = entry["filename"]
        out_path = make_output_path(EXTRACTED_DIR, entry["folder_key"],
                                    Path(entry["filename"]).stem, ".json")
        if out_path.exists():
            log.info("SKIP (already extracted): %s", key)
            manifest[key]["status"] = "extracted"
            continue
        try:
            pages = extract_pymupdf(entry["full_path"])
            save_json(out_path, {"meta": entry, "pages": pages})
            manifest[key]["status"]         = "extracted"
            manifest[key]["extracted_path"] = str(out_path)
            log.info("✓ PyMuPDF: %s (%d pages)", key, len(pages))
        except Exception as e:
            manifest[key]["status"] = "extract_failed"
            manifest[key]["error"]  = str(e)
            log.error("✗ PyMuPDF failed: %s | %s", key, e)

    # ── EasyOCR pass ──────────────────────────────────────────────────────────
    if ocr_entries:
        log.info("Loading EasyOCR model (once for all OCR PDFs)...")
        import easyocr
        reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU)
        log.info("EasyOCR model loaded ✓")

        for entry in ocr_entries:
            key      = entry["filename"]
            out_path = make_output_path(OCR_DIR, entry["folder_key"],
                                        Path(entry["filename"]).stem, ".json")
            if out_path.exists():
                log.info("SKIP (already OCR'd): %s", key)
                manifest[key]["status"] = "extracted"
                continue
            try:
                log.info("OCR start: %s (%d pages)", key, entry["page_count"])
                pages = extract_easyocr(entry["full_path"], reader)
                save_json(out_path, {"meta": entry, "pages": pages})
                manifest[key]["status"]         = "extracted"
                manifest[key]["extracted_path"] = str(out_path)
                log.info("✓ EasyOCR: %s (%d pages)", key, len(pages))
            except Exception as e:
                manifest[key]["status"] = "extract_failed"
                manifest[key]["error"]  = str(e)
                log.error("✗ OCR failed: %s | %s", key, e)

            # Release VRAM between PDFs to avoid CUDA OOM on 6GB card
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    save_json(MANIFEST_PATH, manifest)
    done = sum(1 for v in manifest.values() if v.get("status") == "extracted")
    log.info("Extraction complete: %d/%d extracted", done, len(manifest))


if __name__ == "__main__":
    run_extraction()
