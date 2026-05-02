import json
import sys
import re
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load pipeline config — src/ is the pipeline directory in this repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    CHUNKS_DIR, EMBEDDINGS_DIR, CHROMA_COLLECTION,
    EMBEDDING_MODEL, QUERY_PREFIX
)

# Load all chunks once at module level
_CHUNKS_PATH = CHUNKS_DIR / "all_chunks.json"
ALL_CHUNKS   = json.loads(_CHUNKS_PATH.read_text(encoding="utf-8"))
CHUNK_BY_ID      = {c["chunk_id"]: c for c in ALL_CHUNKS}
CHUNK_BY_STR_ID  = {f"chunk_{c['chunk_id']:06d}": c for c in ALL_CHUNKS}

# Bengali digit normalization for query expansion
_BN_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

# Bengali suffix stemming — strip common inflectional suffixes
_BN_SUFFIXES = re.compile(
    r'(টির?|টা|গুলো(র|কে|তে)?|এর|কে|তে|তেই|র|ের|য়ের|কেই|'
    r'ওয়া|আনো|ানো|িয়ে|য়ে|ছে|ছেন|গেছে|হয়েছে|করেছে|'
    r'tion|ing|ness|ment|ity|ies|ed|er|est|ly)$'
)


def _stem(token: str) -> str:
    """Strip Bengali/English inflectional suffixes for better BM25 matching."""
    if len(token) > 5:
        stemmed = _BN_SUFFIXES.sub('', token)
        return stemmed if len(stemmed) >= 3 else token
    return token


def _resolve_chunk(chunk_id) -> dict | None:
    """Resolve a chunk by int id or 'chunk_XXXXXX' string."""
    if chunk_id in CHUNK_BY_ID:
        return CHUNK_BY_ID[chunk_id]
    if chunk_id in CHUNK_BY_STR_ID:
        return CHUNK_BY_STR_ID[chunk_id]
    return None


def _expand_query(query: str) -> str:
    """
    Bengali query expansion:
    - Normalize Bengali digits to ASCII
    - Strip common Bengali question words that add noise to BM25
    - Append transliterated version for English fallback
    """
    expanded = query.translate(_BN_DIGITS)
    # Remove common Bengali question-word prefixes that don't add lexical signal
    expanded = re.sub(
        r'^(কী|কি|কেন|কীভাবে|কোথায়|কখন|কে|কাকে|কার|কোন|what is|what are|'
        r'why is|how does|how is|where is|who is)\s+',
        '', expanded, flags=re.IGNORECASE
    ).strip()
    # Keep original too — return both joined
    if expanded != query:
        return query + " " + expanded
    return query


class BM25Retriever:
    """
    Sparse lexical retrieval using BM25Okapi.
    Tokenizes on whitespace with Bengali query expansion.
    """
    def __init__(self, chunks: list[dict] = None):
        self.chunks   = chunks or ALL_CHUNKS
        tokenized     = [[_stem(t) for t in c["text"].lower().split()]
                         for c in self.chunks]
        self.bm25     = BM25Okapi(tokenized)

    def _tokenize(self, query: str) -> list[str]:
        tokens = _expand_query(query).lower().split()
        return [_stem(t) for t in tokens]

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        scores  = self.bm25.get_scores(self._tokenize(query))
        top_idx = sorted(range(len(scores)),
                         key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_idx]

    def retrieve_with_scores(self, query: str, k: int = 50) -> list[tuple]:
        """Returns list of (chunk, score) sorted by score desc."""
        scores  = self.bm25.get_scores(self._tokenize(query))
        top_idx = sorted(range(len(scores)),
                         key=lambda i: scores[i], reverse=True)[:k]
        max_s = max(scores[top_idx[0]], 1e-9)
        return [(self.chunks[i], scores[i] / max_s) for i in top_idx]

    def retrieve_ids(self, query: str, k: int = 50) -> list:
        scores  = self.bm25.get_scores(self._tokenize(query))
        top_idx = sorted(range(len(scores)),
                         key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i]["chunk_id"] for i in top_idx]


class DenseRetriever:
    """
    Dense retrieval using intfloat/multilingual-e5-large + ChromaDB.
    "query: " prefix is REQUIRED by this model.
    """
    def __init__(self):
        self.model      = SentenceTransformer(EMBEDDING_MODEL)
        db_client       = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
        self.collection = db_client.get_collection(CHROMA_COLLECTION)
        self.prefix     = QUERY_PREFIX

    def _embed(self, query: str) -> list[float]:
        return self.model.encode(
            f"{self.prefix}{query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q_vec   = self._embed(query)
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            include=["metadatas", "distances"],
        )
        chunks = []
        for cid in results["ids"][0]:
            chunk = _resolve_chunk(cid)
            if chunk:
                chunks.append(chunk)
        return chunks

    def retrieve_with_scores(self, query: str, k: int = 50) -> list[tuple]:
        """Returns list of (chunk, cosine_similarity) sorted desc."""
        q_vec   = self._embed(query)
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            include=["metadatas", "distances"],
        )
        chunks_scores = []
        for cid, dist in zip(results["ids"][0], results["distances"][0]):
            chunk = _resolve_chunk(cid)
            if chunk:
                sim = 1.0 - dist   # ChromaDB returns L2 distance for normalized vecs
                sim = max(0.0, min(1.0, sim))
                chunks_scores.append((chunk, sim))
        return chunks_scores

    def retrieve_ids(self, query: str, k: int = 50) -> list:
        q_vec   = self._embed(query)
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            include=["metadatas"],
        )
        return results["ids"][0]

    def close(self):
        import gc
        del self.model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


class CrossEncoderReranker:
    """
    Cross-encoder reranker using cross-encoder/ms-marco-MiniLM-L-6-v2.
    Scores (query, passage) pairs jointly — much more accurate than bi-encoder.
    Used to rerank a candidate pool retrieved by any upstream retriever.
    """
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        self.model = CrossEncoder(self.MODEL_NAME)

    def rerank(self, query: str, chunks: list[dict],
               top_k: int = 10) -> list[tuple]:
        """
        Rerank chunks by cross-encoder score.
        Returns list of (chunk, score) sorted by score desc, top_k results.
        """
        if not chunks:
            return []
        pairs  = [(query, c["text"][:512]) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(chunks, scores),
                        key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def rerank_chunks(self, query: str, chunks: list[dict],
                      top_k: int = 10) -> list[dict]:
        """Convenience: returns only chunks (no scores)."""
        return [c for c, _ in self.rerank(query, chunks, top_k)]


def _mmr(chunks_scores: list[tuple], query_vec: np.ndarray,
         embed_fn, top_k: int = 10, lambda_mmr: float = 0.6) -> list[dict]:
    """
    Maximal Marginal Relevance diversification.
    lambda_mmr=0.6: balance relevance vs diversity (0=pure diversity, 1=pure relevance).
    Reduces redundant passages sent to the generator.
    """
    if not chunks_scores:
        return []

    selected   = []
    candidates = list(chunks_scores)   # [(chunk, rel_score)]

    while candidates and len(selected) < top_k:
        if not selected:
            # First pick: highest relevance
            best = max(candidates, key=lambda x: x[1])
            selected.append(best[0])
            candidates.remove(best)
            continue

        # Embed selected passages
        sel_texts = [c["text"][:256] for c in selected]
        sel_vecs  = embed_fn(sel_texts)   # (n_sel, dim)

        best_score = -1e9
        best_item  = None
        for chunk, rel in candidates:
            cand_vec  = embed_fn([chunk["text"][:256]])[0]
            # Max similarity to any already-selected chunk
            sims      = [float(np.dot(cand_vec, sv) /
                               (np.linalg.norm(cand_vec) * np.linalg.norm(sv) + 1e-9))
                         for sv in sel_vecs]
            max_sim   = max(sims)
            mmr_score = lambda_mmr * rel - (1 - lambda_mmr) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_item  = (chunk, rel)

        if best_item:
            selected.append(best_item[0])
            candidates.remove(best_item)

    return selected


class HybridRRFRetriever:
    """
    Hybrid retrieval using weighted Reciprocal Rank Fusion (RRF).
    Dense weight=0.7, BM25 weight=0.3.
    score(d) = w_dense/(k+rank_dense) + w_bm25/(k+rank_bm25)
    """
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever,
                 rrf_k: int = 60, dense_weight: float = 0.7,
                 bm25_weight: float = 0.3):
        self.bm25         = bm25
        self.dense        = dense
        self.rrf_k        = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight  = bm25_weight

    def _rrf_scores(self, query: str, pool: int = 50) -> dict:
        bm25_ids  = [str(i) for i in self.bm25.retrieve_ids(query, k=pool)]
        dense_ids = [str(i) for i in self.dense.retrieve_ids(query, k=pool)]
        all_ids   = set(bm25_ids) | set(dense_ids)
        scores    = {}
        for cid in all_ids:
            score = 0.0
            if cid in bm25_ids:
                score += self.bm25_weight / (self.rrf_k + bm25_ids.index(cid) + 1)
            if cid in dense_ids:
                score += self.dense_weight / (self.rrf_k + dense_ids.index(cid) + 1)
            scores[cid] = score
        return scores

    def retrieve(self, query: str, k: int = 3, top_k: int = None) -> list[dict]:
        top_k  = top_k if top_k is not None else k
        scores = self._rrf_scores(query)
        top_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [c for cid in top_ids
                if (c := _resolve_chunk(cid)) is not None]

    def retrieve_with_scores(self, query: str, k: int = 50) -> list[tuple]:
        scores  = self._rrf_scores(query)
        top_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        result  = []
        max_s   = max(scores.values(), default=1e-9)
        for cid in top_ids:
            chunk = _resolve_chunk(cid)
            if chunk:
                result.append((chunk, scores[cid] / max_s))
        return result

    def retrieve_ids(self, query: str, k: int = 50) -> list:
        scores  = self._rrf_scores(query)
        top_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        return [_resolve_chunk(cid)["chunk_id"]
                for cid in top_ids if _resolve_chunk(cid)]


class RerankingRetriever:
    """
    Two-stage retriever: first-stage retrieval + cross-encoder reranking + MMR.

    Pipeline:
      1. First-stage retriever fetches top-50 candidates
      2. Cross-encoder reranks to top-15
      3. Optional MMR diversification to final top_k
      4. Weighted score fusion: 0.5*reranker + 0.3*dense + 0.2*bm25

    This is the best-quality retriever for paper experiments.
    """
    def __init__(self, base_retriever,
                 reranker: CrossEncoderReranker,
                 dense: DenseRetriever = None,
                 bm25: BM25Retriever = None,
                 use_mmr: bool = True,
                 mmr_lambda: float = 0.6):
        self.base      = base_retriever
        self.reranker  = reranker
        self.dense     = dense
        self.bm25      = bm25
        self.use_mmr   = use_mmr
        self.mmr_lambda = mmr_lambda

    def _fused_score(self, chunk, reranker_score: float,
                     bm25_scores: dict, dense_scores: dict) -> float:
        cid   = str(chunk["chunk_id"])
        b_s   = bm25_scores.get(cid, 0.0)
        d_s   = dense_scores.get(cid, 0.0)
        # Normalize reranker score to 0-1 via sigmoid
        import math
        r_s   = 1.0 / (1.0 + math.exp(-reranker_score / 5.0))
        return 0.5 * r_s + 0.3 * d_s + 0.2 * b_s

    def retrieve(self, query: str, k: int = 10, top_k: int = None) -> list[dict]:
        top_k = top_k if top_k is not None else k

        # Stage 1: first-stage retrieval (top-50)
        candidates = self.base.retrieve(query, k=50)

        # Stage 2: cross-encoder rerank to top-15
        reranked = self.reranker.rerank(query, candidates, top_k=15)

        if not reranked:
            return candidates[:top_k]

        # Stage 3: weighted score fusion
        bm25_scores, dense_scores = {}, {}
        if self.bm25:
            for c, s in self.bm25.retrieve_with_scores(query, k=50):
                bm25_scores[str(c["chunk_id"])] = s
        if self.dense:
            for c, s in self.dense.retrieve_with_scores(query, k=50):
                dense_scores[str(c["chunk_id"])] = s

        fused = [(c, self._fused_score(c, rs, bm25_scores, dense_scores))
                 for c, rs in reranked]
        fused.sort(key=lambda x: x[1], reverse=True)

        # Stage 4: MMR diversification
        if self.use_mmr and self.dense and len(fused) > top_k:
            def embed_fn(texts):
                return self.dense.model.encode(
                    texts, normalize_embeddings=True, convert_to_numpy=True
                )
            return _mmr(fused, None, embed_fn, top_k=top_k,
                        lambda_mmr=self.mmr_lambda)

        return [c for c, _ in fused[:top_k]]

    def retrieve_ids(self, query: str, k: int = 50) -> list:
        results = self.retrieve(query, top_k=k)
        return [c["chunk_id"] for c in results]


class HierarchicalRetriever:
    """
    Two-level hierarchical retrieval: retrieve child chunks, return parent context.

    Pipeline:
      1. Dense retrieval on child chunks (small, precise)
      2. Map children back to their parent chunks (large, context-rich)
      3. Deduplicate parents (multiple children may map to same parent)
      4. Optional: rerank parents with cross-encoder
      5. Return parent chunks to generator (full context)

    Benefits:
      - Retrieval precision of small chunks
      - Generation quality of large context windows
      - Title-prefixed embeddings improve child retrieval accuracy
    """

    def __init__(self, dense_model: SentenceTransformer,
                 reranker: CrossEncoderReranker = None,
                 use_rerank: bool = True):
        self.model      = dense_model
        self.reranker   = reranker
        self.use_rerank = use_rerank
        self.prefix     = QUERY_PREFIX

        # Load child and parent chunk maps
        child_path  = CHUNKS_DIR / "all_chunks_child.json"
        parent_path = CHUNKS_DIR / "all_chunks_parent.json"

        if not child_path.exists() or not parent_path.exists():
            raise FileNotFoundError(
                "Hierarchical chunks not found. Run step3b_rechunk_hier.py first."
            )

        children = json.loads(child_path.read_text(encoding="utf-8"))
        parents  = json.loads(parent_path.read_text(encoding="utf-8"))

        self.child_by_id  = {c["chunk_id"]: c for c in children}
        self.parent_by_id = {p["chunk_id"]: p for p in parents}

        # Connect to child ChromaDB collection
        db_client        = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
        self.collection  = db_client.get_collection("nctb_pilot_child")

    def _embed(self, query: str) -> list[float]:
        return self.model.encode(
            f"{self.prefix}{query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def _children_to_parents(self, child_ids: list) -> list[dict]:
        """Map child chunk IDs to unique parent chunks, preserving order."""
        seen_parents = set()
        parents      = []
        for cid in child_ids:
            # child IDs from ChromaDB are "child_XXXXXX"
            if isinstance(cid, str) and cid.startswith("child_"):
                idx = int(cid.split("_")[1])
            else:
                idx = int(cid)
            child = self.child_by_id.get(idx)
            if not child:
                continue
            pid = child["parent_id"]
            if pid not in seen_parents:
                seen_parents.add(pid)
                parent = self.parent_by_id.get(pid)
                if parent:
                    parents.append(parent)
        return parents

    def retrieve(self, query: str, k: int = 10, top_k: int = None) -> list[dict]:
        top_k = top_k if top_k is not None else k

        # Retrieve top-50 children
        q_vec   = self._embed(query)
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=min(50, self.collection.count()),
            include=["metadatas"],
        )
        child_ids = results["ids"][0]

        # Map to parent chunks (deduplicated)
        parents = self._children_to_parents(child_ids)

        # Rerank parents if reranker available
        if self.use_rerank and self.reranker and len(parents) > top_k:
            parents = self.reranker.rerank_chunks(query, parents, top_k=top_k)
        else:
            parents = parents[:top_k]

        return parents

    def retrieve_ids(self, query: str, k: int = 50) -> list:
        return [c["chunk_id"] for c in self.retrieve(query, top_k=k)]
