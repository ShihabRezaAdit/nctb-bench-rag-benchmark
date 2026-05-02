# utils.py
import json
import logging
import re
import tempfile
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Filename Parser ───────────────────────────────────────────────────────────
# Matches: ICT_ben_class6.pdf / Science_eng_class8.pdf
_FNAME_RE = re.compile(
    r'^(?P<subject>.+?)_(?P<lang>ben|eng)_class(?P<cls>\d+)\.pdf$',
    re.IGNORECASE
)
_LANG_MAP = {"ben": "bn", "eng": "en"}


def parse_filename(filename: str) -> dict | None:
    m = _FNAME_RE.match(filename)
    if not m:
        logger.warning("Filename did not match expected pattern: %s", filename)
        return None
    lang      = m.group("lang").lower()
    cls       = int(m.group("cls"))
    lang_code = _LANG_MAP[lang]
    return {
        "subject":    m.group("subject"),
        "lang":       lang,
        "lang_code":  lang_code,
        "class_num":  cls,
        "folder_key": f"class{cls}_{lang_code}",
    }


def get_all_pdfs(nctb_root: str | Path) -> list[dict]:
    root = Path(nctb_root)
    results = []
    for pdf_path in sorted(root.rglob("*.pdf")):
        parsed = parse_filename(pdf_path.name)
        if parsed is None:
            continue
        results.append({
            **parsed,
            "filename":  pdf_path.name,
            "full_path": str(pdf_path),
        })
    logger.info("Total PDFs discovered: %d", len(results))
    return results


# ── Atomic JSON Write ─────────────────────────────────────────────────────────
# Write to a temp file first, then rename (atomic on same filesystem).
# Prevents corrupt files if process crashes mid-write.
def save_json(path: str | Path, data, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)   # atomic on Windows (same drive)
    except Exception:
        os.unlink(tmp)
        raise


def load_json(path: str | Path) -> dict | list | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Output Path Builder ───────────────────────────────────────────────────────
def make_output_path(base_dir: str | Path, folder_key: str,
                     stem: str, ext: str) -> Path:
    out_dir = Path(base_dir) / folder_key
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}{ext}"


# ── Logging Setup ─────────────────────────────────────────────────────────────
def setup_logging(step_name: str, logs_dir: Path) -> logging.Logger:
    """
    Each step gets its own dated log file AND writes to console.
    Using a unique logger name per step prevents handlers stacking up
    if setup_logging is called more than once in the same process.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{step_name}_{date_str}.log"

    logger_name = f"pipeline.{step_name}"
    log = logging.getLogger(logger_name)

    # Guard: don't add duplicate handlers if called twice
    if log.handlers:
        return log

    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)
    return log
