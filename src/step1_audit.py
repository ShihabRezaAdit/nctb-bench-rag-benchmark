# step1_audit.py
"""
Audit every PDF: page count, file size, text extractability.
Marks each PDF as needing PyMuPDF (direct) or EasyOCR.
Writes results into manifest.json.
"""
import fitz  # PyMuPDF
import logging
from pathlib import Path

from config import NCTB_ROOT, MANIFEST_PATH, MIN_TEXT_LENGTH, LOGS_DIR
from utils import get_all_pdfs, save_json, load_json, setup_logging

log = setup_logging("audit", LOGS_DIR)


def audit_pdf(entry: dict) -> dict:
    path = Path(entry["full_path"])
    result = {
        **entry,
        "status":             "pending",
        "page_count":         0,
        "file_size_mb":       round(path.stat().st_size / 1_048_576, 2),
        "extraction_method":  None,
        "error":              None,
    }
    try:
        doc = fitz.open(str(path))
        result["page_count"] = len(doc)

        # Sample up to 5 pages to decide extraction method
        text_pages = 0
        sample = list(range(min(5, len(doc))))
        for i in sample:
            page_text = doc[i].get_text("text").strip()
            if len(page_text) >= MIN_TEXT_LENGTH:
                text_pages += 1
        doc.close()

        # Bengali PDFs are image-based → always OCR
        # English PDFs: use direct extraction unless text is sparse
        if entry["lang_code"] == "bn":
            result["extraction_method"] = "easyocr"
        elif text_pages >= len(sample) * 0.6:
            result["extraction_method"] = "pymupdf"
        else:
            result["extraction_method"] = "easyocr"
            log.warning("%s: English PDF has sparse text — using OCR", entry["filename"])

        result["status"] = "audited"
        log.info("✓ %s | %d pages | %.1f MB | method=%s",
                 entry["filename"], result["page_count"],
                 result["file_size_mb"], result["extraction_method"])

    except Exception as e:
        result["error"]  = str(e)
        result["status"] = "audit_failed"
        log.error("✗ %s | %s", entry["filename"], e)

    return result


def run_audit() -> None:
    pdfs = get_all_pdfs(NCTB_ROOT)
    if not pdfs:
        log.error("No PDFs found under %s", NCTB_ROOT)
        return

    # Load existing manifest to preserve prior state (idempotent)
    manifest = load_json(MANIFEST_PATH) or {}

    audited = []
    for entry in pdfs:
        key = entry["filename"]
        # Skip if already successfully audited
        if manifest.get(key, {}).get("status") == "audited":
            log.info("SKIP (already audited): %s", key)
            audited.append(manifest[key])
            continue
        result = audit_pdf(entry)
        manifest[key] = result
        audited.append(result)

    save_json(MANIFEST_PATH, manifest)
    success = sum(1 for r in audited if r["status"] == "audited")
    log.info("Audit complete: %d/%d succeeded", success, len(audited))


if __name__ == "__main__":
    run_audit()
