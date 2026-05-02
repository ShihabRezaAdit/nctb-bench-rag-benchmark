# step3_chunk.py
"""
Split extracted text into overlapping chunks.
Each chunk carries full source metadata for traceability back to
the original PDF, subject, class, and page number.
"""
import logging
from pathlib import Path

from config import (EXTRACTED_DIR, OCR_DIR, CHUNKS_DIR,
                    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS,
                    MANIFEST_PATH, LOGS_DIR,
                    SUBJECT_CODE_MAP, GRADE_BAND_MAP, SCRIPT_MAP)
from utils import load_json, save_json, setup_logging

log = setup_logging("chunk", LOGS_DIR)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Character-level sliding window chunker.
    Tries to break on sentence boundaries (। . ? !) to avoid
    cutting mid-sentence — important for QA quality.
    """
    if not text:
        return []

    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            for sep in ["।", ".", "?", "!", "\n"]:
                idx = chunk.rfind(sep)
                if idx > chunk_size * 0.5:
                    chunk = chunk[:idx + 1]
                    break

        raw_len = len(chunk)
        chunk   = chunk.strip()
        if chunk:
            chunks.append(chunk)
        advance = raw_len - overlap
        start  += max(advance, max(1, chunk_size // 2))

    return [c for c in chunks if len(c) > MIN_CHUNK_CHARS]


def run_chunking() -> None:
    manifest = load_json(MANIFEST_PATH)
    if not manifest:
        log.error("manifest.json not found — run step 2 first")
        return

    all_chunks = []
    chunk_id   = 0

    for key, entry in manifest.items():
        if entry.get("status") != "extracted":
            log.warning("SKIP (not extracted): %s", key)
            continue

        extracted_path = entry.get("extracted_path")
        if not extracted_path or not Path(extracted_path).exists():
            log.error("extracted_path missing or not found for %s", key)
            continue

        data = load_json(extracted_path)
        if not data:
            log.error("Could not load extracted JSON: %s", extracted_path)
            continue

        pages = data.get("pages", [])
        meta  = data.get("meta", entry)

        subject_raw  = meta.get("subject", "")
        lang_code    = meta.get("lang_code", "en")
        class_num    = meta.get("class_num", 0)
        subject_code = SUBJECT_CODE_MAP.get(subject_raw, subject_raw.lower())
        grade_band   = GRADE_BAND_MAP.get(class_num, "unknown")
        script       = SCRIPT_MAP.get(lang_code, "latin")

        file_chunks = 0
        for page in pages:
            page_text = page.get("text", "").strip()
            if not page_text:
                continue

            for chunk_text in split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
                all_chunks.append({
                    "chunk_id":     chunk_id,
                    "text":         chunk_text,
                    "char_count":   len(chunk_text),
                    "source":       key,
                    "subject":      subject_raw,
                    "subject_code": subject_code,
                    "language":     lang_code,
                    "lang_code":    lang_code,
                    "class_num":    class_num,
                    "grade_band":   grade_band,
                    "script":       script,
                    "page_num":     page.get("page_num"),
                    "folder_key":   meta.get("folder_key"),
                })
                chunk_id    += 1
                file_chunks += 1

        log.info("✓ %s → %d chunks", key, file_chunks)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / "all_chunks.json"
    save_json(out_path, all_chunks)
    log.info("Chunking complete: %d total chunks → %s", len(all_chunks), out_path)


if __name__ == "__main__":
    run_chunking()
