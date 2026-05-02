import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd

_EVAL_DIR = Path(__file__).resolve().parent
_SRC_DIR  = _EVAL_DIR.parent / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_EVAL_DIR))
from config import (
    TOGETHER_MODEL_PRIMARY, TOGETHER_MODEL_SECONDARY,
    VALIDATED_DIR, CHUNKS_DIR,
)
from metrics import token_f1, exact_match, recall_at_k, compute_all_metrics
from retrievers import BM25Retriever, DenseRetriever, HybridRRFRetriever
from generators import RAGGenerator

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_SET_PATH = VALIDATED_DIR / "test_set.json"
RESULTS_DIR   = _EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 12 Configurations ─────────────────────────────────────────────────────────
CONFIGS = [
    {"id":  1, "retriever": "bm25",   "generator": "llama",   "prompt": "grounded"},
    {"id":  2, "retriever": "bm25",   "generator": "llama",   "prompt": "open"},
    {"id":  3, "retriever": "bm25",   "generator": "qwen", "prompt": "grounded"},
    {"id":  4, "retriever": "bm25",   "generator": "qwen", "prompt": "open"},
    {"id":  5, "retriever": "dense",  "generator": "llama",   "prompt": "grounded"},
    {"id":  6, "retriever": "dense",  "generator": "llama",   "prompt": "open"},
    {"id":  7, "retriever": "dense",  "generator": "qwen", "prompt": "grounded"},
    {"id":  8, "retriever": "dense",  "generator": "qwen", "prompt": "open"},
    {"id":  9, "retriever": "hybrid", "generator": "llama",   "prompt": "grounded"},
    {"id": 10, "retriever": "hybrid", "generator": "llama",   "prompt": "open"},
    {"id": 11, "retriever": "hybrid", "generator": "qwen", "prompt": "grounded"},
    {"id": 12, "retriever": "hybrid", "generator": "qwen", "prompt": "open"},
]

MODEL_MAP = {
    "llama": TOGETHER_MODEL_PRIMARY,
    "qwen":  TOGETHER_MODEL_SECONDARY,
}


def load_test_set() -> list[dict]:
    data = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    print(f"Test set loaded: {len(data)} pairs")
    return data


def run_config(cfg: dict, test_set: list,
               retrievers: dict, generator: RAGGenerator) -> dict:
    """
    Run one configuration across all test questions.
    Saves results per question immediately (crash-safe).
    Returns aggregated metrics dict.
    """
    cfg_id      = cfg["id"]
    retriever   = retrievers[cfg["retriever"]]
    model       = MODEL_MAP[cfg["generator"]]
    prompt_type = cfg["prompt"]
    result_file = RESULTS_DIR / f"config_{cfg_id:02d}_results.json"

    # Load existing results (idempotent resume)
    existing = {}
    if result_file.exists():
        saved    = json.loads(result_file.read_text(encoding="utf-8"))
        existing = {str(r["chunk_id"]): r for r in saved}
        if existing:
            print(f"  Config {cfg_id}: {len(existing)} results already saved, resuming...")

    all_results = list(existing.values())

    for i, pair in enumerate(test_set):
        chunk_id = pair.get("chunk_id", f"pair_{i}")

        if str(chunk_id) in existing:
            continue

        question      = pair.get("question", "")
        gold_answer   = pair.get("answer", "")
        gold_chunk_id = pair.get("chunk_id", "")
        language      = pair.get("language", "en")
        grade_band    = pair.get("grade_band", "unknown")

        # Retrieve top-10, pass top-5 to generator, score against top-10
        chunks        = retriever.retrieve(question, k=10)
        retrieved_ids = [c["chunk_id"] for c in chunks]
        gen_chunks    = chunks[:5]

        # Generate
        if prompt_type == "grounded":
            predicted = generator.generate_grounded(question, gen_chunks, model)
        else:
            predicted = generator.generate_open(question, model)

        # Open answer for hallucination comparison (grounded configs only)
        open_answer = ""
        if prompt_type == "grounded":
            open_answer = generator.generate_open(question, model)

        # Compute metrics
        em = exact_match(predicted, gold_answer)
        f1 = token_f1(predicted, gold_answer)
        r1 = recall_at_k(retrieved_ids, gold_chunk_id, 1)
        r3 = recall_at_k(retrieved_ids, gold_chunk_id, 3)
        r5 = recall_at_k(retrieved_ids, gold_chunk_id, 5)
        r10 = recall_at_k(retrieved_ids, gold_chunk_id, 10)

        # Hallucination flag: open answer less accurate than grounded by >0.10
        hall_flag = 0
        if prompt_type == "grounded" and open_answer:
            open_f1 = token_f1(open_answer, gold_answer)
            if open_f1 < f1 - 0.10:
                hall_flag = 1

        result = {
            "chunk_id":        chunk_id,
            "question":        question,
            "gold_answer":     gold_answer,
            "predicted":       predicted,
            "open_answer":     open_answer,
            "retrieved_ids":   retrieved_ids,
            "gold_chunk_id":   gold_chunk_id,
            "language":        language,
            "grade_band":      grade_band,
            "subject_code":    pair.get("subject_code", ""),
            "class_num":       pair.get("class_num", ""),
            "em":              em,
            "token_f1":        f1,
            "recall_at_1":     r1,
            "recall_at_3":     r3,
            "recall_at_5":     r5,
            "recall_at_10":    r10,
            "hallucination":   hall_flag,
        }
        all_results.append(result)

        # Save immediately (crash-safe)
        result_file.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        if (i + 1) % 20 == 0:
            done    = len(all_results)
            total   = len(test_set)
            avg_f1  = sum(r["token_f1"] for r in all_results) / max(done, 1)
            print(f"    Progress: {done}/{total}  avg_f1={avg_f1:.3f}")

    # Aggregate metrics
    n = len(all_results)

    def avg(key):
        return round(sum(r[key] for r in all_results) / max(n, 1), 4)

    def avg_f1_sub(subset):
        return round(sum(r["token_f1"] for r in subset) / len(subset), 4) if subset else 0.0

    def avg_em_sub(subset):
        return round(sum(r["em"] for r in subset) / len(subset), 4) if subset else 0.0

    bn_results  = [r for r in all_results if r["language"] == "bn"]
    en_results  = [r for r in all_results if r["language"] == "en"]
    js_results  = [r for r in all_results if r["grade_band"] == "junior_sec"]
    ssc_results = [r for r in all_results if r["grade_band"] == "ssc"]

    return {
        "config_id":          cfg_id,
        "retriever":          cfg["retriever"],
        "generator":          cfg["generator"],
        "prompt_type":        prompt_type,
        "n":                  n,
        "recall_at_1":        avg("recall_at_1"),
        "recall_at_3":        avg("recall_at_3"),
        "recall_at_5":        avg("recall_at_5"),
        "recall_at_10":       avg("recall_at_10"),
        "em":                 avg("em"),
        "token_f1":           avg("token_f1"),
        "hallucination_rate": avg("hallucination"),
        "bn_f1":              avg_f1_sub(bn_results),
        "bn_em":              avg_em_sub(bn_results),
        "en_f1":              avg_f1_sub(en_results),
        "en_em":              avg_em_sub(en_results),
        "junior_sec_f1":      avg_f1_sub(js_results),
        "ssc_f1":             avg_f1_sub(ssc_results),
    }


def main():
    print("=" * 65)
    print("NCTBench-Pilot RAG Evaluation v2 (Improved)")
    print("12 configurations x 200 test questions")
    print("Improvements: few-shot prompts, k=10 retrieval, weighted RRF, preamble stripping")
    print("=" * 65)

    test_set = load_test_set()

    print("\nInitializing retrievers...")
    bm25   = BM25Retriever()
    dense  = DenseRetriever()
    hybrid = HybridRRFRetriever(bm25, dense)
    retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}
    print("All retrievers ready.")

    generator   = RAGGenerator()
    all_metrics = []

    for cfg in CONFIGS:
        print(f"\n{'-'*65}")
        print(f"Config {cfg['id']:2d}: "
              f"{cfg['retriever']:6s} | "
              f"{cfg['generator']:7s} | "
              f"{cfg['prompt']}")
        print(f"{'-'*65}")
        metrics = run_config(cfg, test_set, retrievers, generator)
        all_metrics.append(metrics)
        print(f"  recall@3={metrics['recall_at_3']:.3f}  "
              f"recall@10={metrics['recall_at_10']:.3f}  "
              f"f1={metrics['token_f1']:.3f}  "
              f"em={metrics['em']:.3f}  "
              f"hall={metrics['hallucination_rate']:.3f}")

    # Save summary table
    df           = pd.DataFrame(all_metrics)
    summary_path = RESULTS_DIR / "summary_table.csv"
    df.to_csv(summary_path, index=False)

    print(f"\n{'='*65}")
    print("EVALUATION COMPLETE")
    print(f"{'='*65}")
    print(f"\nSummary table: {summary_path}")
    print(df[["config_id", "retriever", "generator", "prompt_type",
              "recall_at_3", "recall_at_10", "token_f1", "em", "hallucination_rate"]]
          .to_string(index=False))

    best = df.loc[df["token_f1"].idxmax()]
    print(f"\nBest config by Token F1:")
    print(f"  Config {int(best['config_id'])}: "
          f"{best['retriever']} | {best['generator']} | {best['prompt_type']}")
    print(f"  Recall@3={best['recall_at_3']:.3f}  "
          f"F1={best['token_f1']:.3f}  "
          f"EM={best['em']:.3f}  "
          f"Hall={best['hallucination_rate']:.3f}")

    print(f"\nLanguage gap (best config):")
    print(f"  Bengali F1 : {best['bn_f1']:.3f}")
    print(f"  English F1 : {best['en_f1']:.3f}")
    print(f"  Gap (en-bn): {best['en_f1'] - best['bn_f1']:+.3f}")

    # Language and grade breakdown CSVs
    lang_rows, grade_rows = [], []
    for row in all_metrics:
        for lang, f1, em in [("bn", row["bn_f1"], row["bn_em"]),
                              ("en", row["en_f1"], row["en_em"])]:
            lang_rows.append({"config_id": row["config_id"],
                               "retriever": row["retriever"],
                               "generator": row["generator"],
                               "prompt_type": row["prompt_type"],
                               "language": lang, "token_f1": f1, "em": em})
        for grade, f1 in [("junior_sec", row["junior_sec_f1"]),
                           ("ssc", row["ssc_f1"])]:
            grade_rows.append({"config_id": row["config_id"],
                                "retriever": row["retriever"],
                                "generator": row["generator"],
                                "prompt_type": row["prompt_type"],
                                "grade_band": grade, "token_f1": f1})

    pd.DataFrame(lang_rows).to_csv(RESULTS_DIR / "language_breakdown.csv", index=False)
    pd.DataFrame(grade_rows).to_csv(RESULTS_DIR / "grade_breakdown.csv", index=False)
    print(f"\nLanguage breakdown : {RESULTS_DIR}/language_breakdown.csv")
    print(f"Grade breakdown    : {RESULTS_DIR}/grade_breakdown.csv")

    dense.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
