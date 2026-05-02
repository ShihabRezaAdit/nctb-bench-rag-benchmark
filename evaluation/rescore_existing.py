"""
Enhancement C — Re-score all existing configs (1-12) with:
  - ROUGE-L
  - BERTScore (multilingual-e5-large cosine similarity)
  - Std dev + CI95 for token_f1
  - Per-subject breakdown (ICT vs Science)

No API calls required. Reads existing result JSONs from evaluation/results/.
Writes enriched_summary.csv with all new columns added.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR.parent / "src"))
sys.path.insert(0, str(_EVAL_DIR))

from metrics import token_f1, exact_match, rouge_l, BERTScorer, _std, _ci95

RESULTS_DIR  = _EVAL_DIR / "results"
OUTPUT_CSV   = RESULTS_DIR / "enriched_summary.csv"
CONFIG_IDS   = list(range(1, 13))   # existing configs 1-12


def load_config_results(cfg_id: int) -> list[dict]:
    path = RESULTS_DIR / f"config_{cfg_id:02d}_results.json"
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping.")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"  Config {cfg_id:2d}: {len(data)} results loaded")
    return data


def rescore_config(cfg_id: int, results: list[dict],
                   bert_scorer: BERTScorer) -> dict:
    """
    Compute ROUGE-L, BERTScore, std, CI95, and subject breakdown
    for one config's result list. Returns a flat metrics dict.
    """
    if not results:
        return {}

    predictions = [r.get("predicted", "") for r in results]
    golds       = [r.get("gold_answer", "") for r in results]
    languages   = [r.get("language", "en") for r in results]
    grade_bands = [r.get("grade_band", "") for r in results]
    subjects    = [r.get("subject_code", "") for r in results]

    # ROUGE-L (pure Python, fast)
    rouge_scores = [rouge_l(p, g) for p, g in zip(predictions, golds)]

    # BERTScore (GPU batch, ~30s for 200 pairs)
    print(f"    Computing BERTScore for config {cfg_id}...")
    bert_scores = bert_scorer.score(predictions, golds)

    # Token F1 (re-compute for std/CI95)
    f1_scores = [token_f1(p, g) for p, g in zip(predictions, golds)]
    em_scores = [exact_match(p, g) for p, g in zip(predictions, golds)]

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def sub_mean(vals, mask):
        filtered = [v for v, m in zip(vals, mask) if m]
        return round(sum(filtered) / len(filtered), 4) if filtered else 0.0

    bn_mask  = [l == "bn" for l in languages]
    en_mask  = [l == "en" for l in languages]
    js_mask  = [g == "junior_sec" for g in grade_bands]
    ssc_mask = [g == "ssc" for g in grade_bands]
    ict_mask = [s == "ict" for s in subjects]
    sci_mask = [s == "science" for s in subjects]

    # Load existing summary row for this config
    existing_csv = RESULTS_DIR / "summary_table.csv"
    base_row = {}
    if existing_csv.exists():
        df = pd.read_csv(existing_csv)
        rows = df[df["config_id"] == cfg_id]
        if not rows.empty:
            base_row = rows.iloc[0].to_dict()

    return {
        **base_row,
        # New metrics
        "rouge_l":          mean(rouge_scores),
        "bert_score":       mean(bert_scores),
        # Std dev + CI95 for token_f1
        "token_f1_std":     _std(f1_scores),
        "token_f1_ci95":    _ci95(f1_scores),
        "rouge_l_std":      _std(rouge_scores),
        "bert_score_std":   _std(bert_scores),
        # Per-language ROUGE-L
        "bn_rouge_l":       sub_mean(rouge_scores, bn_mask),
        "en_rouge_l":       sub_mean(rouge_scores, en_mask),
        # Per-language BERTScore
        "bn_bert_score":    sub_mean(bert_scores, bn_mask),
        "en_bert_score":    sub_mean(bert_scores, en_mask),
        # Per-grade ROUGE-L
        "junior_sec_rouge_l": sub_mean(rouge_scores, js_mask),
        "ssc_rouge_l":        sub_mean(rouge_scores, ssc_mask),
        # Per-subject (new breakdown)
        "ict_f1":           sub_mean(f1_scores, ict_mask),
        "science_f1":       sub_mean(f1_scores, sci_mask),
        "ict_rouge_l":      sub_mean(rouge_scores, ict_mask),
        "science_rouge_l":  sub_mean(rouge_scores, sci_mask),
        "ict_bert_score":   sub_mean(bert_scores, ict_mask),
        "science_bert_score": sub_mean(bert_scores, sci_mask),
    }


def print_comparison(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("ENRICHED RESULTS — Token F1 vs ROUGE-L vs BERTScore")
    print("=" * 80)
    cols = ["config_id", "retriever", "generator", "prompt_type",
            "token_f1", "rouge_l", "bert_score",
            "token_f1_std", "token_f1_ci95"]
    show = [c for c in cols if c in df.columns]
    print(df[show].to_string(index=False))

    print("\n" + "=" * 80)
    print("SUBJECT BREAKDOWN (best grounded configs)")
    print("=" * 80)
    sub_cols = ["config_id", "retriever", "generator",
                "ict_f1", "science_f1", "ict_rouge_l", "science_rouge_l"]
    show_sub = [c for c in sub_cols if c in df.columns]
    grounded = df[df["prompt_type"] == "grounded"] if "prompt_type" in df.columns else df
    print(grounded[show_sub].to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY FINDING: ROUGE-L vs Token F1 gap (grounded configs)")
    print("  A large ROUGE-L > Token F1 gap means models give correct but re-ordered answers")
    print("=" * 80)
    if "rouge_l" in df.columns and "token_f1" in df.columns:
        df2 = df.copy()
        df2["rouge_gap"] = (df2["rouge_l"] - df2["token_f1"]).round(4)
        cols2 = ["config_id", "retriever", "generator", "prompt_type",
                 "token_f1", "rouge_l", "rouge_gap"]
        show2 = [c for c in cols2 if c in df2.columns]
        print(df2[show2].to_string(index=False))


def main():
    print("=" * 65)
    print("Enhancement C — ROUGE-L + BERTScore + Extended Breakdown")
    print(f"Re-scoring {len(CONFIG_IDS)} existing configs (no API calls)")
    print("=" * 65)

    print("\nLoading BERTScorer (multilingual-e5-large)...")
    bert_scorer = BERTScorer.get()
    print("BERTScorer ready.")

    all_rows = []
    for cfg_id in CONFIG_IDS:
        print(f"\nConfig {cfg_id}:")
        results = load_config_results(cfg_id)
        if not results:
            continue
        row = rescore_config(cfg_id, results, bert_scorer)
        all_rows.append(row)
        print(f"  rouge_l={row.get('rouge_l', 0):.4f}  "
              f"bert_score={row.get('bert_score', 0):.4f}  "
              f"token_f1_std={row.get('token_f1_std', 0):.4f}  "
              f"ict_f1={row.get('ict_f1', 0):.4f}  "
              f"science_f1={row.get('science_f1', 0):.4f}")

    df = pd.DataFrame(all_rows)
    if "config_id" in df.columns:
        df = df.sort_values("config_id").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nEnriched summary saved: {OUTPUT_CSV}")

    print_comparison(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
