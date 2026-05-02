# step4_embed_index.py
"""
Embed all chunks and index into ChromaDB for retrieval.

Key design decisions:
  1. Batched embedding (EMBED_BATCH_SIZE=64) — avoids VRAM OOM
  2. ChromaDB upsert instead of add — safe to re-run without duplicate errors
  3. passage: prefix required by intfloat/multilingual-e5-large
  4. normalize_embeddings=True required for cosine similarity
  5. GPU memory released after embedding before ChromaDB write
  6. Stores text_preview in metadata for human-readable retrieval results
"""
import gc
import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import (CHUNKS_DIR, EMBEDDINGS_DIR, EMBEDDING_MODEL,
                    EMBED_BATCH_SIZE, EMBED_PREFIX, CHROMA_COLLECTION, LOGS_DIR)
from utils import load_json, setup_logging

log = setup_logging("embed_index", LOGS_DIR)


def run_embedding() -> None:
    chunks_path = CHUNKS_DIR / "all_chunks.json"
    if not chunks_path.exists():
        log.error("all_chunks.json not found — run step 3 first")
        return

    all_chunks = load_json(chunks_path)
    if not all_chunks:
        log.error("all_chunks.json is empty")
        return

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    texts     = []
    ids       = []
    metadatas = []

    for chunk in all_chunks:
        # passage: prefix is required by multilingual-e5-large
        prefixed = f"{EMBED_PREFIX}{chunk['text']}"
        uid      = f"chunk_{chunk['chunk_id']:06d}"
        texts.append(prefixed)
        ids.append(uid)
        metadatas.append({
            "chunk_id":     str(chunk["chunk_id"]),
            "source_pdf":   str(chunk.get("source", "")),
            "subject":      str(chunk.get("subject", "")),
            "subject_code": str(chunk.get("subject_code", "")),
            "language":     str(chunk.get("language", "")),
            "class_num":    str(chunk.get("class_num", "")),
            "grade_band":   str(chunk.get("grade_band", "")),
            "page_num":     str(chunk.get("page_num", "")),
            "text_preview": chunk["text"][:200],
        })

    log.info("Embedding %d chunks in batches of %d...", len(texts), EMBED_BATCH_SIZE)
    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        vecs  = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
        all_embeddings.extend(vecs)
        log.info("Embedded %d / %d", min(i + EMBED_BATCH_SIZE, len(texts)), len(texts))

    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    log.info("Indexing into ChromaDB collection '%s'...", CHROMA_COLLECTION)
    CHROMA_BATCH = 500
    for i in range(0, len(ids), CHROMA_BATCH):
        collection.upsert(
            ids=ids[i: i + CHROMA_BATCH],
            embeddings=all_embeddings[i: i + CHROMA_BATCH],
            metadatas=metadatas[i: i + CHROMA_BATCH],
            documents=texts[i: i + CHROMA_BATCH],
        )
        log.info("Upserted %d / %d", min(i + CHROMA_BATCH, len(ids)), len(ids))

    log.info("✓ Indexing complete. Collection '%s' has %d records.",
             CHROMA_COLLECTION, collection.count())


if __name__ == "__main__":
    run_embedding()
