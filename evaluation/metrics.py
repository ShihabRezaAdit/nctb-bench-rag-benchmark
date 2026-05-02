import re
import string
import math
from collections import Counter
from typing import Optional


_BN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

_PREAMBLE = re.compile(
    r'^(based on (the )?(passage|context|text)[^,]*,?\s*|'
    r'according to (the )?(passage|context|text)[^,]*,?\s*|'
    r'the answer is[:\s]*|'
    r'from the (passage|context|text)[^,]*,?\s*|'
    r'as (mentioned|stated|given) in (the )?(passage|context|text)[^,]*,?\s*)',
    re.IGNORECASE,
)


def strip_preamble(text: str) -> str:
    """Remove common model preamble phrases before scoring."""
    return _PREAMBLE.sub('', text).strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for metric computation.
    Works for both Bengali and English.
    Steps: strip preamble, Bengali→ASCII digits, lowercase,
           remove punctuation, collapse whitespace.
    """
    text = strip_preamble(text)
    text = text.translate(_BN_DIGITS)
    text = text.lower()
    # Remove punctuation (both ASCII and common Unicode)
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~।,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 if normalized texts match exactly, else 0.0."""
    if not prediction or not gold:
        return 0.0
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """
    SQuAD-style token F1 score.
    Tokenizes on whitespace after normalization.
    Handles Bengali whitespace-separated tokens correctly.
    """
    if not prediction or not gold:
        return 0.0
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def recall_at_k(retrieved_ids: list, gold_chunk_id, k: int) -> float:
    """
    Return 1.0 if gold_chunk_id appears in retrieved_ids[:k].
    Handles both int and string chunk IDs.
    """
    if not retrieved_ids or gold_chunk_id is None:
        return 0.0
    # Normalize to string for comparison
    str_ids   = [str(i) for i in retrieved_ids[:k]]
    str_gold  = str(gold_chunk_id)
    # Also check ChromaDB-style "chunk_XXXXXX" format
    padded_gold = f"chunk_{int(gold_chunk_id):06d}" if str(gold_chunk_id).isdigit() else str_gold
    return 1.0 if (str_gold in str_ids or padded_gold in str_ids) else 0.0


def coverage_score(prediction: str, passages: list[str]) -> float:
    """
    Fraction of answer tokens that appear in any retrieved passage.
    Measures lexical grounding — zero cost, no model needed.
    """
    if not prediction or not passages:
        return 0.0
    pred_tokens = set(normalize_text(prediction).split())
    if not pred_tokens:
        return 0.0
    passage_tokens = set()
    for p in passages:
        passage_tokens.update(normalize_text(p).split())
    covered = pred_tokens & passage_tokens
    return round(len(covered) / len(pred_tokens), 4)


class NLIScorer:
    """
    NLI-based faithfulness scorer using cross-encoder/nli-deberta-v3-small.
    Loaded once; scores answer entailment against each passage.
    Returns max entailment probability across all passages (0.0–1.0).
    Label order for this model: contradiction=0, entailment=1, neutral=2
    """
    _MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

    def __init__(self):
        from transformers import pipeline
        self._pipe = pipeline(
            "text-classification",
            model=self._MODEL_NAME,
            device=-1,          # CPU; change to 0 for GPU
            top_k=None,
        )

    def score(self, answer: str, passages: list[str]) -> tuple[float, int]:
        """
        Returns (max_entailment_score, best_passage_rank_1indexed).
        best_passage_rank is the 1-based index of the passage with highest entailment.
        """
        if not answer or not passages:
            return 0.0, -1
        answer = strip_preamble(answer)
        best_score = 0.0
        best_rank  = 1
        for i, passage in enumerate(passages[:5]):
            premise = passage[:512]
            try:
                results = self._pipe(f"{premise} [SEP] {answer}", truncation=True)
                # results is list of dicts with 'label' and 'score'
                label_scores = {r["label"].lower(): r["score"] for r in results[0]}
                entail = label_scores.get("entailment", 0.0)
            except Exception:
                entail = 0.0
            if entail > best_score:
                best_score = entail
                best_rank  = i + 1
        return round(best_score, 4), best_rank

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a batch of (answer, passage) pairs for entailment."""
        if not pairs:
            return []
        inputs = [f"{p[:512]} [SEP] {a}" for a, p in pairs]
        try:
            results = self._pipe(inputs, truncation=True, batch_size=16)
            scores = []
            for r in results:
                label_scores = {x["label"].lower(): x["score"] for x in r}
                scores.append(round(label_scores.get("entailment", 0.0), 4))
            return scores
        except Exception:
            return [0.0] * len(pairs)


def rouge_l(prediction: str, gold: str) -> float:
    """
    ROUGE-L: longest common subsequence F1.
    Better than token F1 for order-sensitive answers.
    """
    if not prediction or not gold:
        return 0.0
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    m, n = len(gold_tokens), len(pred_tokens)
    # DP table for LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gold_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall    = lcs / m
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


class BERTScorer:
    """
    BERTScore using multilingual-e5-large (already in-env, GPU-accelerated).
    Uses cosine similarity between contextualized embeddings.
    Handles Bengali + English naturally via multilingual model.
    Loaded once and reused.
    """
    _instance = None

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = SentenceTransformer(
            "intfloat/multilingual-e5-large", device=self.device
        )

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def score(self, predictions: list[str], references: list[str]) -> list[float]:
        """
        Returns token-level BERTScore F1 per pair using embedding cosine similarity.
        Uses sentence-level embeddings as a fast approximation.
        """
        if not predictions or not references:
            return []
        # Prefix required by multilingual-e5
        pred_texts = [f"query: {normalize_text(p)}" for p in predictions]
        ref_texts  = [f"query: {normalize_text(r)}" for r in references]

        import numpy as np
        pred_vecs = self.model.encode(pred_texts, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        ref_vecs  = self.model.encode(ref_texts,  normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        # Cosine similarity (vectors are already normalized)
        scores = [round(float(np.dot(p, r)), 4)
                  for p, r in zip(pred_vecs, ref_vecs)]
        return scores

    def score_pair(self, prediction: str, reference: str) -> float:
        results = self.score([prediction], [reference])
        return results[0] if results else 0.0


def _ci95(values: list[float]) -> float:
    """95% confidence interval half-width using normal approximation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    return round(1.96 * std / math.sqrt(n), 4)


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return round(math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)), 4)


def compute_all_metrics(predictions: list[dict]) -> dict:
    """
    Compute all metrics across a list of prediction dicts.
    Each dict must have keys:
      predicted_answer, gold_answer,
      retrieved_ids, gold_chunk_id
    Returns dict with mean of each metric.
    """
    if not predictions:
        return {"em": 0.0, "token_f1": 0.0,
                "recall_at_1": 0.0, "recall_at_3": 0.0,
                "n": 0}
    ems, f1s, r1s, r3s = [], [], [], []
    for p in predictions:
        pred   = p.get("predicted_answer", "")
        gold   = p.get("gold_answer", "")
        r_ids  = p.get("retrieved_ids", [])
        g_id   = p.get("gold_chunk_id", "")
        ems.append(exact_match(pred, gold))
        f1s.append(token_f1(pred, gold))
        r1s.append(recall_at_k(r_ids, g_id, 1))
        r3s.append(recall_at_k(r_ids, g_id, 3))
    n = len(predictions)
    return {
        "em":          round(sum(ems) / n, 4),
        "token_f1":    round(sum(f1s) / n, 4),
        "recall_at_1": round(sum(r1s) / n, 4),
        "recall_at_3": round(sum(r3s) / n, 4),
        "n":           n,
    }
