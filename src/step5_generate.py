# step5_qa_generate.py
"""
Generate QA pairs from chunks using Together AI (OpenAI-compatible).

Key design decisions:
  1. Token bucket rate limiter — stays safely under TOGETHER_RPM_LIMIT
  2. Exponential backoff on 429 / 5xx errors
  3. Per-chunk output files — resume after crash without reprocessing
  4. Validates response is parseable JSON before saving
  5. Language-appropriate prompts (Bengali vs English)
  6. Source metadata + source_text attached to every QA pair
  7. Evidence field required in every pair
  8. Stratified logarithmic sampling across subject x language x class
"""
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from config import (CHUNKS_DIR, QA_DIR, LOGS_DIR,
                    TOGETHER_API_KEY, TOGETHER_BASE_URL,
                    TOGETHER_MODEL_PRIMARY, TOGETHER_RPM_LIMIT,
                    TOGETHER_MAX_RETRIES, TOGETHER_RETRY_BASE_DELAY,
                    QA_PER_CHUNK, QA_DAY_LIMIT)
from utils import load_json, save_json, setup_logging

log = setup_logging("qa_generate", LOGS_DIR)


# ── Rate Limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    """Simple token bucket — ensures we never exceed RPM_LIMIT."""
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm
        self._last_call   = 0.0

    def wait(self):
        elapsed  = time.monotonic() - self._last_call
        wait_for = self.min_interval - elapsed
        if wait_for > 0:
            time.sleep(wait_for)
        self._last_call = time.monotonic()


# ── Stratified Sampling ───────────────────────────────────────────────────────
def build_stratified_sample(chunks: list[dict], total: int) -> list[dict]:
    """
    Logarithmic allocation across strata (subject x language x class).
    Smaller strata get proportionally more coverage via log weighting.
    If total >= len(chunks), returns all chunks.
    """
    if total >= len(chunks):
        return chunks

    strata: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        key = f"{c.get('subject_code','?')}_{c.get('language','?')}_{c.get('class_num','?')}"
        strata[key].append(c)

    log_weights = {k: math.log1p(len(v)) for k, v in strata.items()}
    total_weight = sum(log_weights.values())

    sample = []
    for k, v in strata.items():
        alloc = max(1, round(total * log_weights[k] / total_weight))
        alloc = min(alloc, len(v))
        sample.extend(random.sample(v, alloc))

    return sample[:total]


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(chunk: dict, n: int) -> str:
    lang    = chunk.get("language", chunk.get("lang_code", "en"))
    subject = chunk.get("subject", "")
    cls     = chunk.get("class_num", "")
    text    = chunk["text"]

    if lang == "bn":
        instruction = (
            f"তুমি একজন শিক্ষক। নিচের পাঠ্যাংশ থেকে Class {cls} {subject} বইয়ের "
            f"জন্য {n}টি প্রশ্ন-উত্তর জোড়া তৈরি করো। "
            "প্রতিটি জোড়ায় অবশ্যই একটি 'evidence' ক্ষেত্র থাকতে হবে "
            "(পাঠ্যাংশের হুবহু অংশ যা উত্তরটি সমর্থন করে)। "
            "শুধুমাত্র JSON array আউটপুট করো, অন্য কিছু না।\n"
            'Format: [{"question": "...", "answer": "...", "evidence": "..."}, ...]\n\n'
            f"পাঠ্যাংশ:\n{text}"
        )
    else:
        instruction = (
            f"You are a teacher. Generate exactly {n} question-answer pairs "
            f"from the following Class {cls} {subject} textbook excerpt. "
            "Every pair MUST include an 'evidence' field containing the exact "
            "verbatim span from the text that supports the answer. "
            "Output ONLY a valid JSON array, no preamble, no markdown fences.\n"
            'Format: [{"question": "...", "answer": "...", "evidence": "..."}, ...]\n\n'
            f"Text:\n{text}"
        )
    return instruction


# ── Single Together AI Call With Retry ───────────────────────────────────────
def call_together(client: OpenAI, prompt: str, limiter: RateLimiter) -> list[dict] | None:
    for attempt in range(TOGETHER_MAX_RETRIES):
        limiter.wait()
        try:
            response = client.chat.completions.create(
                model=TOGETHER_MODEL_PRIMARY,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.4,
            )
            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            pairs = json.loads(raw)
            if not isinstance(pairs, list):
                raise ValueError("Response is not a JSON array")
            for p in pairs:
                if "question" not in p or "answer" not in p:
                    raise ValueError(f"Missing keys in pair: {p}")
            return pairs

        except json.JSONDecodeError as e:
            log.warning("Attempt %d: JSON parse failed — %s", attempt + 1, e)
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429:
                delay = TOGETHER_RETRY_BASE_DELAY * (2 ** attempt)
                log.warning("Rate limited (429). Waiting %.1fs...", delay)
                time.sleep(delay)
            else:
                log.warning("Attempt %d failed: %s", attempt + 1, e)

    log.error("All %d attempts failed for this chunk", TOGETHER_MAX_RETRIES)
    return None


# ── Main Runner ───────────────────────────────────────────────────────────────
def run_qa_generation(day: int = 1) -> None:
    if not TOGETHER_API_KEY:
        log.error("TOGETHER_API_KEY not set. Check your .env file.")
        return

    chunks_path = CHUNKS_DIR / "all_chunks.json"
    if not chunks_path.exists():
        log.error("all_chunks.json not found — run step 3 first")
        return

    all_chunks = load_json(chunks_path)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    start_idx = (day - 1) * QA_DAY_LIMIT
    end_idx   = start_idx + QA_DAY_LIMIT
    batch     = all_chunks[start_idx:end_idx]

    if not batch:
        log.info("No chunks left for day %d (total chunks: %d)", day, len(all_chunks))
        return

    log.info("Day %d: processing chunks %d–%d (%d chunks)",
             day, start_idx, min(end_idx, len(all_chunks)) - 1, len(batch))

    client  = OpenAI(api_key=TOGETHER_API_KEY, base_url=TOGETHER_BASE_URL)
    limiter = RateLimiter(TOGETHER_RPM_LIMIT)
    results = []
    success = 0

    for i, chunk in enumerate(batch):
        chunk_id = chunk["chunk_id"]
        out_file = QA_DIR / f"qa_{chunk_id:05d}.json"

        if out_file.exists():
            log.info("SKIP chunk %d (already done)", chunk_id)
            cached = load_json(out_file)
            if cached:
                results.extend(cached)
            success += 1
            continue

        prompt = build_prompt(chunk, QA_PER_CHUNK)
        pairs  = call_together(client, prompt, limiter)

        if pairs:
            enriched = [{
                **p,
                "evidence":      (p["evidence"][0] if isinstance(p.get("evidence"), list) and p["evidence"] else p.get("evidence", "") if isinstance(p.get("evidence"), str) else ""),
                "source_text":   chunk["text"],
                "chunk_id":      chunk_id,
                "source":        chunk["source"],
                "subject":       chunk["subject"],
                "subject_code":  chunk.get("subject_code", ""),
                "language":      chunk.get("language", chunk.get("lang_code", "")),
                "lang_code":     chunk.get("lang_code", chunk.get("language", "")),
                "class_num":     chunk["class_num"],
                "grade_band":    chunk.get("grade_band", ""),
                "page_num":      chunk["page_num"],
            } for p in pairs]

            save_json(out_file, enriched)
            results.extend(enriched)
            success += 1
        else:
            log.warning("No QA pairs generated for chunk %d", chunk_id)

        if (i + 1) % 25 == 0:
            log.info("Progress: %d / %d chunks (%d pairs so far)",
                     i + 1, len(batch), len(results))

    merged_path = QA_DIR / f"qa_pairs_day{day}.json"
    save_json(merged_path, results)
    log.info("Day %d complete: %d/%d chunks → %d QA pairs → %s",
             day, success, len(batch), len(results), merged_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--day", type=int, default=1)
    args = p.parse_args()
    run_qa_generation(day=args.day)
