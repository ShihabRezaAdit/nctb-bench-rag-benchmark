# step6_amsv_validate.py
"""
AMSV (Automated Multi-Signal Validation) for NCTBench QA pairs.

6-signal pipeline:
  Signal 3: Answer extractability (overlap ratio, no API)
  Signal 4: Language integrity    (langdetect, no API)
  Signal 5: Structural quality    (heuristics, no API)
  Signal 2: LLM verification      (Together AI primary model)
  Signal 1: Round-trip retrieval  (ChromaDB cosine similarity)
  Signal 6: Cross-model verify    (Together AI secondary model)

Then: deduplication → gold test set selection.
"""
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import chromadb
from langdetect import detect, LangDetectException
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    VALIDATED_DIR, QA_DIR, EMBEDDINGS_DIR, LOGS_DIR,
    TOGETHER_API_KEY, TOGETHER_BASE_URL,
    TOGETHER_MODEL_PRIMARY, TOGETHER_MODEL_SECONDARY,
    TOGETHER_RPM_LIMIT, TOGETHER_MAX_RETRIES, TOGETHER_RETRY_BASE_DELAY,
    AMSV_OVERLAP_THRESHOLD, AMSV_RETRIEVAL_TOP_K, AMSV_DEDUP_THRESHOLD,
    GOLD_TEST_SIZE, CMAR_TARGET,
    EMBEDDING_MODEL, EMBED_PREFIX, QUERY_PREFIX, CHROMA_COLLECTION,
)
from utils import load_json, save_json, setup_logging

log = setup_logging("amsv_validate", LOGS_DIR)


# ── Rate Limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm
        self._last_call   = 0.0

    def wait(self):
        elapsed  = time.monotonic() - self._last_call
        wait_for = self.min_interval - elapsed
        if wait_for > 0:
            time.sleep(wait_for)
        self._last_call = time.monotonic()


# ── Signal 3: Answer Extractability ──────────────────────────────────────────
def signal3_overlap(pair: dict) -> tuple[bool, float]:
    """Token overlap between answer and source_text."""
    answer  = str(pair.get("answer", "")).lower().split()
    source  = str(pair.get("source_text", "")).lower().split()
    if not answer or not source:
        return False, 0.0
    source_set = set(source)
    overlap    = sum(1 for w in answer if w in source_set) / len(answer)
    return overlap >= AMSV_OVERLAP_THRESHOLD, overlap


# ── Signal 4: Language Integrity ──────────────────────────────────────────────
def signal4_language(pair: dict) -> tuple[bool, str]:
    """Detected language must match declared language."""
    declared = pair.get("language", pair.get("lang_code", "en"))
    text     = pair.get("question", "") + " " + pair.get("answer", "")
    try:
        detected = detect(text)
    except LangDetectException:
        return False, "detect_failed"
    # langdetect returns "bn" for Bengali, "en" for English
    lang_map = {"bn": "bn", "en": "en"}
    declared_norm = lang_map.get(declared, declared)
    return detected == declared_norm, detected


# ── Signal 5: Structural Quality ─────────────────────────────────────────────
def signal5_structure(pair: dict) -> tuple[bool, str]:
    """Heuristic checks on question and answer quality."""
    q = pair.get("question", "").strip()
    a = pair.get("answer", "").strip()
    ev = pair.get("evidence", "")
    e = (ev[0] if isinstance(ev, list) and ev else ev if isinstance(ev, str) else "").strip()

    if len(q) < 10:
        return False, "question_too_short"
    if len(a) < 5:
        return False, "answer_too_short"
    if not e:
        return False, "missing_evidence"
    if q.lower() == a.lower():
        return False, "question_equals_answer"
    if not any(q.endswith(c) for c in ("?", "।", "?")):
        # Allow questions without explicit mark but must have question word
        question_words_en = {"what", "which", "who", "when", "where", "why", "how"}
        question_words_bn = {"কী", "কি", "কোন", "কেন", "কোথায়", "কখন", "কীভাবে"}
        words = set(q.lower().split())
        if not (words & question_words_en or words & question_words_bn):
            return False, "not_a_question"
    return True, "ok"


# ── LLM Verify (Signals 2 and 6) ─────────────────────────────────────────────
def llm_verify(pair: dict, client: OpenAI, model: str,
               limiter: RateLimiter) -> tuple[bool, str]:
    """Ask the LLM if the QA pair is factually grounded in the source text."""
    prompt = (
        "You are a strict QA validator. Given a source text and a question-answer pair, "
        "reply with exactly one word: PASS or FAIL.\n"
        "PASS if: the answer is factually correct and directly supported by the source text.\n"
        "FAIL if: the answer contradicts the source, is hallucinated, or is not answerable "
        "from the source text alone.\n\n"
        f"Source text:\n{pair.get('source_text', '')[:600]}\n\n"
        f"Question: {pair.get('question', '')}\n"
        f"Answer: {pair.get('answer', '')}\n\n"
        "Your verdict (PASS or FAIL):"
    )

    for attempt in range(TOGETHER_MAX_RETRIES):
        limiter.wait()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            verdict = response.choices[0].message.content.strip().upper()
            passed  = verdict.startswith("PASS")
            return passed, verdict
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429:
                delay = TOGETHER_RETRY_BASE_DELAY * (2 ** attempt)
                log.warning("Rate limited (429). Waiting %.1fs...", delay)
                time.sleep(delay)
            else:
                log.warning("LLM verify attempt %d failed: %s", attempt + 1, e)

    return False, "api_failed"


# ── Signal 1: Round-trip Retrieval ────────────────────────────────────────────
def signal1_retrieval(pairs: list[dict], embed_model: SentenceTransformer,
                      collection) -> list[dict]:
    """
    Embed each question and retrieve top-K chunks from ChromaDB.
    Pair passes if the source chunk appears in top-K results.
    """
    passed = []
    for pair in pairs:
        q_vec = embed_model.encode(
            f"{QUERY_PREFIX}{pair['question']}",
            normalize_embeddings=True,
        ).tolist()
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=AMSV_RETRIEVAL_TOP_K,
            include=["metadatas"],
        )
        retrieved_ids = [
            m.get("chunk_id", "") for m in results["metadatas"][0]
        ]
        source_id = str(pair.get("chunk_id", ""))
        if source_id in retrieved_ids:
            pair["signal1_passed"] = True
            passed.append(pair)
        else:
            pair["signal1_passed"] = False
    return [p for p in pairs if p.get("signal1_passed")]


# ── Deduplication ─────────────────────────────────────────────────────────────
def deduplicate(pairs: list[dict], embed_model: SentenceTransformer) -> list[dict]:
    """Remove near-duplicate questions by cosine similarity threshold."""
    if not pairs:
        return pairs

    questions = [p["question"] for p in pairs]
    vecs = embed_model.encode(
        questions, normalize_embeddings=True, show_progress_bar=False
    )

    kept    = []
    dropped = set()
    for i, pair in enumerate(pairs):
        if i in dropped:
            continue
        kept.append(pair)
        for j in range(i + 1, len(pairs)):
            if j in dropped:
                continue
            sim = float(vecs[i] @ vecs[j])
            if sim >= AMSV_DEDUP_THRESHOLD:
                dropped.add(j)

    log.info("Dedup: %d → %d pairs (dropped %d duplicates)",
             len(pairs), len(kept), len(dropped))
    return kept


# ── Main AMSV Runner ──────────────────────────────────────────────────────────
def run_amsv() -> None:
    if not TOGETHER_API_KEY:
        log.error("TOGETHER_API_KEY not set.")
        return

    # Load all per-chunk QA files
    qa_files = sorted(QA_DIR.glob("qa_*.json"))
    if not qa_files:
        log.error("No qa_*.json files found — run step 5 first")
        return

    all_pairs = []
    for f in qa_files:
        pairs = load_json(f)
        if pairs:
            all_pairs.extend(pairs)

    log.info("Loaded %d raw QA pairs from %d files", len(all_pairs), len(qa_files))
    total_raw = len(all_pairs)

    # ── Signals 3, 4, 5 (no API cost) ────────────────────────────────────────
    log.info("Running Signals 3, 4, 5 (no API)...")
    after_345 = []
    for pair in all_pairs:
        ok3, _  = signal3_overlap(pair)
        ok4, _  = signal4_language(pair)
        ok5, _  = signal5_structure(pair)
        if ok3 and ok4 and ok5:
            after_345.append(pair)

    log.info("After S3+S4+S5: %d / %d", len(after_345), total_raw)

    # ── Signal 2: LLM verify (primary model) ─────────────────────────────────
    log.info("Running Signal 2 (LLM verify, primary model: %s)...",
             TOGETHER_MODEL_PRIMARY)
    client  = OpenAI(api_key=TOGETHER_API_KEY, base_url=TOGETHER_BASE_URL)
    limiter = RateLimiter(TOGETHER_RPM_LIMIT)

    after_s2 = []
    for i, pair in enumerate(after_345):
        passed, verdict = llm_verify(pair, client, TOGETHER_MODEL_PRIMARY, limiter)
        if passed:
            after_s2.append(pair)
        if (i + 1) % 50 == 0:
            log.info("Signal 2 progress: %d / %d", i + 1, len(after_345))

    log.info("After S2: %d / %d", len(after_s2), len(after_345))

    # ── Signal 1: Round-trip retrieval (ChromaDB) ─────────────────────────────
    log.info("Running Signal 1 (round-trip retrieval)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
    try:
        collection = chroma_client.get_collection(CHROMA_COLLECTION)
    except Exception:
        log.error("ChromaDB collection '%s' not found — run step 4 first",
                  CHROMA_COLLECTION)
        return

    after_s1 = signal1_retrieval(after_s2, embed_model, collection)
    log.info("After S1: %d / %d", len(after_s1), len(after_s2))

    # ── Signal 6: Cross-model verify (secondary model) ────────────────────────
    log.info("Running Signal 6 (cross-model verify, secondary: %s)...",
             TOGETHER_MODEL_SECONDARY)
    after_s6 = []
    for i, pair in enumerate(after_s1):
        passed, verdict = llm_verify(pair, client, TOGETHER_MODEL_SECONDARY, limiter)
        if passed:
            after_s6.append(pair)
        if (i + 1) % 50 == 0:
            log.info("Signal 6 progress: %d / %d", i + 1, len(after_s1))

    log.info("After S6: %d / %d", len(after_s6), len(after_s1))

    # ── Deduplication ─────────────────────────────────────────────────────────
    after_dedup = deduplicate(after_s6, embed_model)
    log.info("After dedup: %d", len(after_dedup))

    # ── Gold Test Set ─────────────────────────────────────────────────────────
    gold_size  = min(GOLD_TEST_SIZE, len(after_dedup))
    test_set   = random.sample(after_dedup, gold_size)
    cmar       = len(after_dedup) / max(total_raw, 1)
    cmar_met   = cmar >= CMAR_TARGET
    reject_rate = 1.0 - cmar

    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    save_json(VALIDATED_DIR / "validated_pairs.json", after_dedup)
    save_json(VALIDATED_DIR / "test_set.json", test_set)

    report = {
        "total_raw":           total_raw,
        "after_s3_s4_s5":      len(after_345),
        "after_s2":            len(after_s2),
        "after_s1":            len(after_s1),
        "after_s6":            len(after_s6),
        "after_dedup":         len(after_dedup),
        "gold_test_set_size":  gold_size,
        "cmar":                round(cmar, 4),
        "cmar_target":         CMAR_TARGET,
        "cmar_met":            cmar_met,
        "rejection_rate":      round(reject_rate, 4),
    }
    save_json(VALIDATED_DIR / "validation_report.json", report)

    log.info("AMSV complete. CMAR=%.4f (target=%.2f, met=%s). "
             "Gold test set: %d pairs.",
             cmar, CMAR_TARGET, cmar_met, gold_size)


if __name__ == "__main__":
    run_amsv()
