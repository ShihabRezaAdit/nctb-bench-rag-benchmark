"""
Finalize results with all valid configs (1-15).
Configs 16-18 dropped due to API credit exhaustion mid-run.
Produces the definitive enriched_summary_all.csv.
"""
import json, sys
from pathlib import Path
_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR.parent / "src"))
sys.path.insert(0, str(_EVAL_DIR))

from metrics import token_f1, exact_match, recall_at_k, rouge_l, BERTScorer, _std, _ci95
import pandas as pd

RESULTS_DIR = _EVAL_DIR / "results"

# Configs 13-15 metadata (16-18 dropped)
NEW_CONFIGS = [
    {'id': 13, 'retriever': 'rerank_dense',  'generator': 'llama',
     'prompt': 'grounded', 'desc': 'Dense+Rerank+MMR+Llama'},
    {'id': 14, 'retriever': 'rerank_hybrid', 'generator': 'llama',
     'prompt': 'grounded', 'desc': 'Hybrid+Rerank+MMR+Llama'},
    {'id': 15, 'retriever': 'rerank_bm25',   'generator': 'llama',
     'prompt': 'grounded', 'desc': 'BM25+Rerank+MMR+Llama'},
]

print("Loading BERTScorer...")
bert_scorer = BERTScorer.get()
print("Ready.\n")

rows = []
for cfg in NEW_CONFIGS:
    cfg_id = cfg['id']
    fpath  = RESULTS_DIR / f"config_{cfg_id:02d}_results.json"
    results = json.loads(fpath.read_text(encoding='utf-8'))
    n = len(results)

    preds  = [r.get('predicted', '')    for r in results]
    golds  = [r.get('gold_answer', '')  for r in results]
    langs  = [r.get('language', 'en')   for r in results]
    subjs  = [r.get('subject_code', '') for r in results]
    grades = [r.get('grade_band', '')   for r in results]

    f1s   = [r['token_f1']    for r in results]
    ems   = [r['em']          for r in results]
    r1s   = [r['recall_at_1'] for r in results]
    r3s   = [r['recall_at_3'] for r in results]
    halls = [r['hallucination'] for r in results]
    rls   = [rouge_l(p, g) for p, g in zip(preds, golds)]

    print(f"BERTScore config {cfg_id}...")
    berts = bert_scorer.score(preds, golds)

    def mean(v):
        return round(sum(v) / len(v), 4) if v else 0.0

    def sub(v, mask):
        filtered = [x for x, m in zip(v, mask) if m]
        return mean(filtered)

    bn  = [l == 'bn'      for l in langs]
    en  = [l == 'en'      for l in langs]
    ict = [s == 'ict'     for s in subjs]
    sci = [s == 'science' for s in subjs]
    js  = [g == 'junior_sec' for g in grades]
    ssc = [g == 'ssc'    for g in grades]

    row = {
        'config_id':          cfg_id,
        'desc':               cfg['desc'],
        'retriever':          cfg['retriever'],
        'generator':          cfg['generator'],
        'prompt_type':        cfg['prompt'],
        'n':                  n,
        'recall_at_1':        mean(r1s),
        'recall_at_3':        mean(r3s),
        'em':                 mean(ems),
        'token_f1':           mean(f1s),
        'rouge_l':            mean(rls),
        'bert_score':         mean(berts),
        'hallucination_rate': mean(halls),
        'token_f1_std':       _std(f1s),
        'token_f1_ci95':      _ci95(f1s),
        'rouge_l_std':        _std(rls),
        'bert_score_std':     _std(berts),
        'bn_f1':              sub(f1s,  bn),
        'en_f1':              sub(f1s,  en),
        'bn_rouge_l':         sub(rls,  bn),
        'en_rouge_l':         sub(rls,  en),
        'bn_bert_score':      sub(berts, bn),
        'en_bert_score':      sub(berts, en),
        'junior_sec_f1':      sub(f1s, js),
        'ssc_f1':             sub(f1s, ssc),
        'ict_f1':             sub(f1s,  ict),
        'science_f1':         sub(f1s,  sci),
        'ict_rouge_l':        sub(rls,  ict),
        'science_rouge_l':    sub(rls,  sci),
        'ict_bert_score':     sub(berts, ict),
        'science_bert_score': sub(berts, sci),
    }
    rows.append(row)
    print(f"  f1={row['token_f1']:.4f}  rouge_l={row['rouge_l']:.4f}  "
          f"bert={row['bert_score']:.4f}  hall={row['hallucination_rate']:.3f}")

new_df = pd.DataFrame(rows)

# Merge with enriched configs 1-12
old_df = pd.read_csv(RESULTS_DIR / 'enriched_summary.csv')
old_df = old_df[~old_df['config_id'].isin(new_df['config_id'])]
full_df = pd.concat([old_df, new_df], ignore_index=True).sort_values('config_id').reset_index(drop=True)
full_df.to_csv(RESULTS_DIR / 'enriched_summary_all.csv', index=False)
print(f"\nSaved enriched_summary_all.csv ({len(full_df)} configs)")

print("\n=== FINAL RESULTS (all valid configs) ===")
cols = ['config_id', 'retriever', 'generator', 'prompt_type',
        'recall_at_3', 'token_f1', 'rouge_l', 'bert_score', 'hallucination_rate']
show = [c for c in cols if c in full_df.columns]
print(full_df[show].to_string(index=False))

print("\n=== TOP 5 GROUNDED CONFIGS BY TOKEN F1 ===")
grounded = full_df[full_df['prompt_type'] == 'grounded'].sort_values('token_f1', ascending=False)
top5 = grounded.head(5)
print(top5[show].to_string(index=False))

best = grounded.iloc[0]
print(f"\nBEST: Config {int(best['config_id'])} ({best.get('desc', best['retriever'])})")
print(f"  Token F1  = {best['token_f1']:.4f} (+-{best.get('token_f1_ci95', 0):.4f} CI95)")
print(f"  ROUGE-L   = {best['rouge_l']:.4f}")
print(f"  BERTScore = {best['bert_score']:.4f}")
print(f"  Hall Rate = {best['hallucination_rate']:.3f}")
print(f"  ICT F1    = {best.get('ict_f1', 0):.4f}")
print(f"  Science F1= {best.get('science_f1', 0):.4f}")
print(f"  Bengali F1= {best.get('bn_f1', 0):.4f}")
print(f"  English F1= {best.get('en_f1', 0):.4f}")
print("\nDone.")
