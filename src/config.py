import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the repo root (where run_pipeline.sh lives)
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
load_dotenv(_ROOT / ".env")

# ── Base Paths (relative to repo root) ───────────────────────────────────────
BASE_DIR       = _ROOT
NCTB_ROOT      = BASE_DIR / "data" / "pdfs"

# Output Paths
EXTRACTED_DIR  = BASE_DIR / "data" / "extracted_text"
OCR_DIR        = BASE_DIR / "data" / "ocr_results"
CHUNKS_DIR     = BASE_DIR / "data" / "chunks"
QA_DIR         = BASE_DIR / "data" / "qa_pairs"
EMBEDDINGS_DIR = BASE_DIR / "chromadb"
LOGS_DIR       = BASE_DIR / "logs"
MANIFEST_PATH  = BASE_DIR / "data" / "manifest.json"
VALIDATED_DIR  = BASE_DIR / "data" / "qa_pairs"

ALL_OUTPUT_DIRS = [
    EXTRACTED_DIR, OCR_DIR, CHUNKS_DIR,
    QA_DIR, EMBEDDINGS_DIR, LOGS_DIR,
]

# ── Pilot Scope ───────────────────────────────────────────────────────────────
PILOT_CLASSES   = [6, 7, 8, 9]
PILOT_SUBJECTS  = ["ICT", "Science"]
PILOT_LANGUAGES = ["bn", "en"]

SUBJECT_CODE_MAP = {
    "ICT":     "ict",
    "Science": "science",
}

GRADE_BAND_MAP = {
    6: "junior_sec",
    7: "junior_sec",
    8: "junior_sec",
    9: "ssc",
    10: "ssc",
}

SCRIPT_MAP = {
    "bn": "bengali",
    "en": "latin",
}

# ── Together AI ───────────────────────────────────────────────────────────────
TOGETHER_API_KEY          = os.getenv("TOGETHER_API_KEY")
TOGETHER_BASE_URL         = "https://api.together.xyz/v1"
TOGETHER_MODEL_PRIMARY    = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TOGETHER_MODEL_SECONDARY  = "Qwen/Qwen2.5-7B-Instruct-Turbo"
TOGETHER_RPM_LIMIT        = 500
TOGETHER_MAX_RETRIES      = 4
TOGETHER_RETRY_BASE_DELAY = 1.0

# ── Extraction (EasyOCR) ──────────────────────────────────────────────────────
OCR_LANGUAGES   = ["en", "bn"]
OCR_GPU         = True
OCR_BATCH_SIZE  = 4
PDF_DPI         = 200
MIN_TEXT_LENGTH = 30

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
MIN_CHUNK_CHARS = 50

# ── QA Generation ─────────────────────────────────────────────────────────────
QA_PER_CHUNK  = 2
QA_DAY_LIMIT  = 99999

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "intfloat/multilingual-e5-large"
EMBED_BATCH_SIZE  = 64
EMBED_PREFIX      = "passage: "
QUERY_PREFIX      = "query: "
CHROMA_COLLECTION = "nctb_pilot"

# ── AMSV ──────────────────────────────────────────────────────────────────────
AMSV_OVERLAP_THRESHOLD = 0.25
AMSV_RETRIEVAL_TOP_K   = 3
AMSV_DEDUP_THRESHOLD   = 0.97
GOLD_TEST_SIZE         = 200
CMAR_TARGET            = 0.80


def validate_config() -> None:
    if not TOGETHER_API_KEY:
        raise EnvironmentError(
            "TOGETHER_API_KEY not found.\n"
            "Create a .env file in the repo root with:\n"
            "  TOGETHER_API_KEY=your_key_here"
        )
    if not NCTB_ROOT.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {NCTB_ROOT}\n"
            "Place NCTB PDF files under data/pdfs/"
        )
