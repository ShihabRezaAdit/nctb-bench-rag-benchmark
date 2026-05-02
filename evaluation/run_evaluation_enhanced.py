"""
NCTBench-Pilot — Enhanced Evaluation (Configs 13-18)
=====================================================
Combines all three enhancements:
  A. Cross-encoder reranking + MMR (RerankingRetriever)
  B. Stricter grounded prompt (already updated in generators.py)
  C. ROUGE-L + BERTScore + std/CI95 + subject breakdown metrics

Configs:
  13 = Dense + Rerank + MMR + Llama   + grounded  (best retrieval quality)
  14 = Hybrid + Rerank + MMR + Llama  + grounded
  15 = BM25 + Rerank + MMR + Llama   + grounded  (fixes BM25 grounded failure)
  16 = Dense + Rerank + NoMMR + Llama + grounded  (MMR ablation)
  17 = Hybrid + Rerank + MMR + Qwen + grounded (faithfulness champion + rerank)
  18 = Dense + Rerank + MMR + Qwen + grounded

Results saved to evaluation/results/ (same folder as configs 1-12).
Final enriched_summary_all.csv covers all 18 configs with all metrics.
"""

import json
import sys
import time
import math
from pathlib import Path
from collections import defaultdict

import pandas as pd

_EVAL_DIR = Path(__file__).resolve().parent
_SRC_DIR  = _EVAL_DIR.parent / "src"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_EVAL_DIR))

from config import (
    TOGETHER_MODEL_PRIMARY, TOGETHER_MODEL_SECONDARY,
    VALIDATED_DIR,
)
from metrics import (
    token_f1, exact_match, recall_at_k,
    rouge_l, BERTScorer, _std, _ci95,
)
from retrievers import (
    BM25Retriever, DenseRetriever, HybridRRFRetriever,
    CrossEncoderReranker, RerankingRetriever,
)
from generators import RAGGenerator

# Paths
TEST_SET_PATH = VALIDATED_DIR / "test_set.json"
RESULTS_DIR   = _EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 6 Enhanced Configurations
CONFIGS = [
    {"id": 13, "retriever": "rerank_dense",      "generator": "llama",
     "prompt": "grounded", "desc": "Dense+Rerank+MMR+Llama"},
    {"id": 14, "retriever": "rerank_hybrid",     "generator": "llama",
     "prompt": "grounded", "desc": "Hybrid+Rerank+MMR+Llama"},
    {"id": 15, "retriever": "rerank_bm25",       "generator": "llama",
     "prompt": "grounded", "desc": "BM25+Rerank+MMR+Llama"},
    {"id": 16, "retriever": "rerank_dense_nomr", "generator": "llama",
     "prompt": "grounded", "desc": "Dense+Rerank+NoMMR+Llama"},
    {"id": 17, "retriever": "rerank_hybrid",     "generator": "qwen",
     "prompt": "grounded", "desc": "Hybrid+Rerank+MMR+Qwen"},
    {"id": 18, "retriever": "rerank_dense",      "generator": "qwen",
     "prompt": "grounded", "desc": "Dense+Rerank+MMR+Qwen"},
]

MODEL_MAP = {
    "llama": TOGETHER_MODEL_PRIMARY,
    "qwen":  TOGETHER_MODEL_SECONDARY,
}


def load_test_set() -> list[dict]:
    data = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    print(f"Test set: {len(data)} pairs")
    return data


def run_config(cfg: dict, test_set: list,
               retrievers: dict, generator: RAGGenerator) -> list[dict]:
    """Run one config, return all per-sample results (resumable)."""
    cfg_id      = cfg["id"]
    retriever   = retrievers[cfg["retriever"]]
    model       = MODEL_MAP[cfg["generator"]]
    prompt_type = cfg["prompt"]
    result_file = RESULTS_DIR / f"config_{cfg_id:02d}_results.json"

    existing = {}
    if result_file.exists():
        saved    = json.loads(result_file.read_text(encoding="utf-8"))
        existing = {str(r["chunk_id"]): r for r in saved}
        if existing:
            print(f"  Config {cfg_id}: {len(existing)} cached, resuming...")

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
        subject_code  = pair.get("subject_code", "")
        class_num     = pair.get("class_num", "")

        t0     = time.perf_counter()
        chunks = retriever.retrieve(question, k=10)
        t_ret  = round(time.perf_counter() - t0, 3)

        retrieved_ids = [c["chunk_id"] for c in chunks]
        gen_chunks    = chunks[:5]

        t1        = time.perf_counter()
        predicted = generator.generate_grounded(question, gen_chunks, model)
        t_gen     = round(time.perf_counter() - t1, 3)

        # Open answer for hallucination scoring
        open_answer = generator.generate_open(question, model)

        em   = exact_match(predicted, gold_answer)
        f1   = token_f1(predicted, gold_answer)
        rl   = rouge_l(predicted, gold_answer)
        r1   = recall_at_k(retrieved_ids, gold_chunk_id, 1)
        r3   = recall_at_k(retrieved_ids, gold_chunk_id, 3)
        r5   = recall_at_k(retrieved_ids, gold_chunk_id, 5)
        r10  = recall_at_k(retrieved_ids, gold_chunk_id, 10)

        # Hallucination: open F1 < grounded F1 by >0.10
        open_f1   = token_f1(open_answer, gold_answer)
        hall_flag = int(open_f1 < f1 - 0.10)

        result = {
            "chunk_id":       chunk_id,
            "question":       question,
            "gold_answer":    gold_answer,
            "predicted":      predicted,
            "open_answer":    open_answer,
            "retrieved_ids":  retrieved_ids,
            "gold_chunk_id":  gold_chunk_id,
            "language":       language,
            "grade_band":     grade_band,
            "subject_code":   subject_code,
            "class_num":      class_num,
            "em":             em,
            "token_f1":       f1,
            "rouge_l":        rl,
            "recall_at_1":    r1,
            "recall_at_3":    r3,
            "recall_at_5":    r5,
            "recall_at_10":   r10,
            "hallucination":  hall_flag,
            "retrieval_ms":   int(t_ret * 1000),
            "generation_ms":  int(t_gen * 1000),
        }
        all_results.append(result)

        result_file.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        if (i + 1) % 20 == 0:
            done   = len(all_results)
            avg_f1 = sum(r["token_f1"] for r in all_results) / max(done, 1)
            avg_rl = sum(r.get("rouge_l", 0.0) for r in all_results) / max(done, 1)
            print(f"    {done}/{len(test_set)}  f1={avg_f1:.3f}  rouge_l={avg_rl:.3f}")

    return all_results


def aggregate(cfg: dict, results: list[dict],
              bert_scorer: BERTScorer) -> dict:
    """Compute full aggregate metrics including ROUGE-L, BERTScore, std, CI95, breakdowns."""
    n = len(results)
    if n == 0:
        return {}

    predictions = [r["predicted"]   for r in results]
    golds       = [r["gold_answer"] for r in results]
    languages   = [r["language"]    for r in results]
    grade_bands = [r["grade_band"]  for r in results]
    subjects    = [r["subject_code"]for r in results]

    f1s   = [r["token_f1"]    for r in results]
    ems   = [r["em"]          for r in results]
    # Compute rouge_l on-the-fly if not already cached (backward compat)
    rls   = [r.get("rouge_l", rouge_l(r.get("predicted", ""), r.get("gold_answer", "")))
             for r in results]
    r1s   = [r["recall_at_1"] for r in results]
    r3s   = [r["recall_at_3"] for r in results]
    r5s   = [r.get("recall_at_5",  0.0) for r in results]
    r10s  = [r.get("recall_at_10", 0.0) for r in results]
    halls = [r["hallucination"] for r in results]

    print(f"  Computing BERTScore for config {cfg['id']}...")
    bert_scores = bert_scorer.score(predictions, golds)

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def sub_mean(vals, mask):
        filtered = [v for v, m in zip(vals, mask) if m]
        return round(sum(filtered) / len(filtered), 4) if filtered else 0.0

    bn_mask  = [l == "bn"         for l in languages]
    en_mask  = [l == "en"         for l in languages]
    js_mask  = [g == "junior_sec" for g in grade_bands]
    ssc_mask = [g == "ssc"        for g in grade_bands]
    ict_mask = [s == "ict"        for s in subjects]
    sci_mask = [s == "science"    for s in subjects]

    ret_ms = [r.get("retrieval_ms", 0) for r in results]
    gen_ms = [r.get("generation_ms", 0) for r in results]

    return {
        "config_id":           cfg["id"],
        "desc":                cfg["desc"],
        "retriever":           cfg["retriever"],
        "generator":           cfg["generator"],
        "prompt_type":         cfg["prompt"],
        "n":                   n,
        # Core metrics
        "recall_at_1":         mean(r1s),
        "recall_at_3":         mean(r3s),
        "recall_at_5":         mean(r5s),
        "recall_at_10":        mean(r10s),
        "em":                  mean(ems),
        "token_f1":            mean(f1s),
        "hallucination_rate":  mean(halls),
        # New metrics
        "rouge_l":             mean(rls),
        "bert_score":          mean(bert_scores),
        # Variance
        "token_f1_std":        _std(f1s),
        "token_f1_ci95":       _ci95(f1s),
        "rouge_l_std":         _std(rls),
        "bert_score_std":      _std(bert_scores),
        # Language breakdown
        "bn_f1":               sub_mean(f1s,  bn_mask),
        "en_f1":               sub_mean(f1s,  en_mask),
        "bn_rouge_l":          sub_mean(rls,  bn_mask),
        "en_rouge_l":          sub_mean(rls,  en_mask),
        "bn_bert_score":       sub_mean(bert_scores, bn_mask),
        "en_bert_score":       sub_mean(bert_scores, en_mask),
        # Grade breakdown
        "junior_sec_f1":       sub_mean(f1s, js_mask),
        "ssc_f1":              sub_mean(f1s, ssc_mask),
        "junior_sec_rouge_l":  sub_mean(rls, js_mask),
        "ssc_rouge_l":         sub_mean(rls, ssc_mask),
        # Subject breakdown (new)
        "ict_f1":              sub_mean(f1s,  ict_mask),
        "science_f1":          sub_mean(f1s,  sci_mask),
        "ict_rouge_l":         sub_mean(rls,  ict_mask),
        "science_rouge_l":     sub_mean(rls,  sci_mask),
        "ict_bert_score":      sub_mean(bert_scores, ict_mask),
        "science_bert_score":  sub_mean(bert_scores, sci_mask),
        # Timing
        "avg_retrieval_ms":    mean(ret_ms),
        "avg_generation_ms":   mean(gen_ms),
    }


def print_results(all_metrics: list[dict]):
    df = pd.DataFrame(all_metrics)
    print("\n" + "=" * 75)
    print("ENHANCED CONFIGS (13-18) - ALL METRICS")
    print("=" * 75)
    cols = ["config_id", "desc", "recall_at_3", "token_f1",
            "rouge_l", "bert_score", "hallucination_rate"]
    show = [c for c in cols if c in df.columns]
    print(df[show].to_string(index=False))

    print("\n" + "=" * 75)
    print("SUBJECT BREAKDOWN")
    print("=" * 75)

    sub_cols = ["config_id", "desc", "ict_f1", "science_f1",
                "ict_rouge_l", "science_rouge_l"]
    show_sub = [c for c in sub_cols if c in df.columns]
    print(df[show_sub].to_string(index=False))

    best = df.loc[df["token_f1"].idxmax()]
    print(f"\nBest by Token F1: Config {int(best['config_id'])} — {best.get('desc','')}")
    print(f"  F1={best['token_f1']:.4f}  ROUGE-L={best['rouge_l']:.4f}  "
          f"BERTScore={best['bert_score']:.4f}  Hall={best['hallucination_rate']:.3f}")


def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge enriched_summary.csv (configs 1-12) with new configs 13-18."""
    enriched = RESULTS_DIR / "enriched_summary.csv"
    existing = RESULTS_DIR / "summary_table.csv"

    if enriched.exists():
        old_df = pd.read_csv(enriched)
    elif existing.exists():
        old_df = pd.read_csv(existing)
    else:
        return new_df

    old_df = old_df[~old_df["config_id"].isin(new_df["config_id"])]
    merged = pd.concat([old_df, new_df], ignore_index=True).sort_values("config_id")
    return merged


def main():
    print("=" * 75)
    print("NCTBench-Pilot — Enhanced Evaluation (Configs 13-18)")
    print("Enhancements: Reranking + Stricter Prompt + ROUGE-L + BERTScore")
    print("=" * 75)

    test_set = load_test_set()

    print("\nInitializing retrievers...")
    bm25   = BM25Retriever()
    dense  = DenseRetriever()
    hybrid = HybridRRFRetriever(bm25, dense)

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker()

    retrievers = {
        "rerank_dense":   RerankingRetriever(dense,  reranker, dense=dense, bm25=bm25, use_mmr=True),
        "rerank_hybrid":  RerankingRetriever(hybrid, reranker, dense=dense, bm25=bm25, use_mmr=True),
        "rerank_bm25":    RerankingRetriever(bm25,   reranker, dense=dense, bm25=bm25, use_mmr=True),
        "rerank_dense_nomr": RerankingRetriever(dense, reranker, dense=dense, bm25=bm25, use_mmr=False),
    }
    print("All retrievers ready.\n")

    print("Loading BERTScorer...")
    bert_scorer = BERTScorer.get()
    print("BERTScorer ready.\n")

    generator   = RAGGenerator()
    all_metrics = []

    for cfg in CONFIGS:
        print(f"\n{'-'*75}")
        print(f"Config {cfg['id']:2d}: {cfg['desc']}")
        print(f"{'-'*75}")
        results = run_config(cfg, test_set, retrievers, generator)
        metrics = aggregate(cfg, results, bert_scorer)
        all_metrics.append(metrics)
        print(f"  recall@3={metrics['recall_at_3']:.3f}  "
              f"f1={metrics['token_f1']:.3f}  rouge_l={metrics['rouge_l']:.3f}  "
              f"bert={metrics['bert_score']:.3f}  hall={metrics['hallucination_rate']:.3f}")

    # Save configs 13-18 results
    new_df = pd.DataFrame(all_metrics)
    new_df.to_csv(RESULTS_DIR / "enhanced_summary.csv", index=False)
    print(f"\nConfigs 13-18 saved: {RESULTS_DIR}/enhanced_summary.csv")

    # Merge all configs into one master table
    full_df = merge_with_existing(new_df)
    full_df.to_csv(RESULTS_DIR / "enriched_summary_all.csv", index=False)
    print(f"Full 18-config table: {RESULTS_DIR}/enriched_summary_all.csv")

    print_results(all_metrics)

    # Overall best across all 18
    if not full_df.empty and "token_f1" in full_df.columns:
        best = full_df.loc[full_df["token_f1"].idxmax()]
        print(f"\nBest overall (all 18 configs): Config {int(best['config_id'])}")
        print(f"  {best.get('desc', best.get('retriever',''))} | "
              f"F1={best['token_f1']:.4f}  ROUGE-L={best.get('rouge_l', 0):.4f}  "
              f"BERTScore={best.get('bert_score', 0):.4f}")

    dense.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
