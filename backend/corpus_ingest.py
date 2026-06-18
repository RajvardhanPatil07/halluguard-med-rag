"""
CLI and indexing helpers for the future corpus-backed retrieval layer.

This module is intentionally separate from retrieval.py so the current graph
backed system keeps working during migration.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

genai = None
QdrantClient = None
qdrant_models = None

from .corpus_loader import (
    corpus_status,
    ingest_medlineplus_xml,
    load_corpus_chunks,
    write_source_registry,
)
from .corpus_schema import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_MANIFEST_PATH,
    INDEX_DIR,
    ensure_corpus_dirs,
    write_jsonl,
    write_manifest,
)
from .settings import (
    GOOGLE_API_KEY,
    GOOGLE_EMBEDDING_DIM,
    GOOGLE_EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from .structured_log import log_event


DEFAULT_CORPUS_COLLECTION = os.getenv("CORPUS_QDRANT_COLLECTION", "halluguard_med_corpus_v1")
BM25_INDEX_PATH = INDEX_DIR / "bm25_corpus.pkl"
BM25_MANIFEST_PATH = INDEX_DIR / "bm25_manifest.json"
QDRANT_MANIFEST_PATH = INDEX_DIR / "qdrant_manifest.json"
DEFAULT_QDRANT_BATCH_SIZE = int(os.getenv("CORPUS_QDRANT_BATCH_SIZE", "8"))
DEFAULT_EMBED_REQUESTS_PER_MINUTE = int(os.getenv("CORPUS_EMBED_REQUESTS_PER_MINUTE", "80"))
DEFAULT_MAX_EMBED_RETRIES = int(os.getenv("CORPUS_MAX_EMBED_RETRIES", "6"))
DEFAULT_RETRY_SLEEP_SECONDS = int(os.getenv("CORPUS_RETRY_SLEEP_SECONDS", "65"))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qdrant_collection"


def _checkpoint_path(collection_name: str) -> Path:
    return INDEX_DIR / f"qdrant_checkpoint_{_safe_filename(collection_name)}.json"


def _load_checkpoint(collection_name: str) -> dict[str, Any]:
    path = _checkpoint_path(collection_name)
    if not path.exists():
        return {
            "collection": collection_name,
            "indexed_chunk_ids": [],
            "last_updated": None,
        }
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    data.setdefault("collection", collection_name)
    data.setdefault("indexed_chunk_ids", [])
    return data


def _save_checkpoint(collection_name: str, indexed_ids: set[str]) -> None:
    ensure_corpus_dirs()
    payload = {
        "collection": collection_name,
        "indexed_count": len(indexed_ids),
        "indexed_chunk_ids": sorted(indexed_ids),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _checkpoint_path(collection_name).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2, sort_keys=True)


def _is_resource_exhausted(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "resource_exhausted" in message
        or "quota" in message
        or "429" in message
        or "requestsperminute" in message
        or "rate limit" in message
    )


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    size = max(1, int(batch_size))
    return [items[index:index + size] for index in range(0, len(items), size)]


def _require_bm25() -> None:
    if BM25Okapi is None:
        raise RuntimeError("rank-bm25 is not installed")


def _require_embedding_stack() -> None:
    global genai, QdrantClient, qdrant_models
    missing = []
    if genai is None:
        try:
            import google.generativeai as imported_genai
            genai = imported_genai
        except ImportError:
            pass
    if QdrantClient is None or qdrant_models is None:
        try:
            from qdrant_client import QdrantClient as ImportedQdrantClient
            from qdrant_client.http import models as imported_qdrant_models
            QdrantClient = ImportedQdrantClient
            qdrant_models = imported_qdrant_models
        except ImportError:
            pass
    if genai is None:
        missing.append("google-generativeai")
    if QdrantClient is None or qdrant_models is None:
        missing.append("qdrant-client")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if missing:
        raise RuntimeError(f"Missing corpus indexing dependencies: {', '.join(missing)}")


def build_bm25_index(chunks_path: Path = DEFAULT_CHUNKS_PATH) -> dict[str, Any]:
    _require_bm25()
    ensure_corpus_dirs()
    chunks = load_corpus_chunks(chunks_path)
    if not chunks:
        raise RuntimeError(f"No corpus chunks found at {chunks_path}")

    tokenized = [_tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    with BM25_INDEX_PATH.open("wb") as file:
        pickle.dump(payload, file)

    manifest = {
        "ok": True,
        "chunks_path": str(chunks_path),
        "index_path": str(BM25_INDEX_PATH),
        "chunks_count": len(chunks),
        "tokenizer": "regex:[a-z0-9]+",
    }
    with BM25_MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=True, indent=2, sort_keys=True)
    return manifest


def _embed_one_text(
    text: str,
    request_delay_seconds: float,
    max_retries: int,
    retry_sleep_seconds: int,
) -> list[float]:
    _require_embedding_stack()
    genai.configure(api_key=GOOGLE_API_KEY)

    for attempt in range(max_retries + 1):
        try:
            response = genai.embed_content(
                model=GOOGLE_EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document",
                output_dimensionality=GOOGLE_EMBEDDING_DIM,
            )
            vector = response.get("embedding") if isinstance(response, dict) else getattr(response, "embedding", None)
            if not vector:
                raise RuntimeError("Google embedding API returned an empty corpus embedding")
            if len(vector) != GOOGLE_EMBEDDING_DIM:
                raise RuntimeError(
                    f"Corpus embedding dimension mismatch: expected {GOOGLE_EMBEDDING_DIM}, got {len(vector)}"
                )
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
            return [float(value) for value in vector]
        except Exception as exc:
            if not _is_resource_exhausted(exc) or attempt >= max_retries:
                raise
            sleep_seconds = retry_sleep_seconds * (attempt + 1)
            print(
                f"Embedding quota/rate limit hit. Sleeping {sleep_seconds}s "
                f"before retry {attempt + 1}/{max_retries}...",
                flush=True,
            )
            log_event(
                "corpus",
                "embedding_retry_after_resource_exhausted",
                "warning",
                attempt=attempt + 1,
                sleep_seconds=sleep_seconds,
                error=str(exc),
            )
            time.sleep(sleep_seconds)
    raise RuntimeError("Embedding retry loop exited unexpectedly")


def _embed_texts(
    texts: list[str],
    request_delay_seconds: float = 0.0,
    max_retries: int = DEFAULT_MAX_EMBED_RETRIES,
    retry_sleep_seconds: int = DEFAULT_RETRY_SLEEP_SECONDS,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vectors.append(_embed_one_text(
            text=text,
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        ))
    return vectors


def _existing_qdrant_ids(client, collection_name: str, chunk_ids: list[str]) -> set[str]:
    if not chunk_ids:
        return set()
    try:
        records = client.retrieve(
            collection_name=collection_name,
            ids=chunk_ids,
            with_payload=False,
            with_vectors=False,
        )
    except Exception:
        return set()
    return {str(record.id) for record in records}


def index_qdrant_corpus(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    collection_name: str = DEFAULT_CORPUS_COLLECTION,
    max_chunks: int | None = None,
    batch_size: int = DEFAULT_QDRANT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_EMBED_RETRIES,
    retry_sleep_seconds: int = DEFAULT_RETRY_SLEEP_SECONDS,
    embed_requests_per_minute: int = DEFAULT_EMBED_REQUESTS_PER_MINUTE,
) -> dict[str, Any]:
    _require_embedding_stack()
    ensure_corpus_dirs()
    chunks = load_corpus_chunks(chunks_path)
    if max_chunks is not None:
        chunks = chunks[:max(0, int(max_chunks))]
    if not chunks:
        raise RuntimeError(f"No corpus chunks found at {chunks_path}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = client.get_collections().collections
    exists = any(collection.name == collection_name for collection in collections)
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=GOOGLE_EMBEDDING_DIM,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    checkpoint = _load_checkpoint(collection_name)
    indexed_ids = set(str(chunk_id) for chunk_id in checkpoint.get("indexed_chunk_ids", []))
    total = len(chunks)
    target_ids = {chunk.chunk_id for chunk in chunks}
    indexed_ids &= target_ids
    request_delay_seconds = 60.0 / max(1, int(embed_requests_per_minute))
    skipped_count = 0
    indexed_this_run = 0

    for batch_number, batch in enumerate(_batched(chunks, batch_size), start=1):
        batch_ids = [chunk.chunk_id for chunk in batch]
        qdrant_existing = _existing_qdrant_ids(client, collection_name, batch_ids)
        indexed_ids.update(qdrant_existing)
        pending = [chunk for chunk in batch if chunk.chunk_id not in indexed_ids]
        skipped_count += len(batch) - len(pending)

        completed_before_batch = len(indexed_ids)
        if not pending:
            print(
                f"Batch {batch_number}: skipped {len(batch)} already indexed chunks "
                f"({len(indexed_ids)}/{total}).",
                flush=True,
            )
            _save_checkpoint(collection_name, indexed_ids)
            continue

        print(
            f"Batch {batch_number}: embedding {len(pending)} chunks "
            f"({completed_before_batch}/{total} already indexed).",
            flush=True,
        )
        vectors = _embed_texts(
            [chunk.text for chunk in pending],
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        points = [
            qdrant_models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk.to_dict(),
            )
            for chunk, vector in zip(pending, vectors)
        ]
        client.upsert(collection_name=collection_name, points=points)
        indexed_ids.update(chunk.chunk_id for chunk in pending)
        indexed_this_run += len(pending)
        _save_checkpoint(collection_name, indexed_ids)

        print(
            f"Indexed {len(indexed_ids)}/{total} chunks "
            f"(+{len(pending)} this batch, {indexed_this_run} this run).",
            flush=True,
        )
        log_event(
            "corpus",
            "qdrant_index_progress",
            collection=collection_name,
            indexed_count=len(indexed_ids),
            total_count=total,
            indexed_this_run=indexed_this_run,
        )

    manifest = {
        "ok": True,
        "chunks_path": str(chunks_path),
        "collection": collection_name,
        "qdrant_url": QDRANT_URL,
        "embedding_model": GOOGLE_EMBEDDING_MODEL,
        "embedding_dim": GOOGLE_EMBEDDING_DIM,
        "chunks_count": total,
        "indexed_count": len(indexed_ids),
        "indexed_this_run": indexed_this_run,
        "skipped_already_indexed": skipped_count,
        "batch_size": batch_size,
        "embed_requests_per_minute": embed_requests_per_minute,
        "checkpoint_path": str(_checkpoint_path(collection_name)),
    }
    with QDRANT_MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=True, indent=2, sort_keys=True)
    return manifest


def initialize_corpus() -> dict[str, Any]:
    ensure_corpus_dirs()
    write_source_registry()
    chunks = load_corpus_chunks(DEFAULT_CHUNKS_PATH)
    if not DEFAULT_CHUNKS_PATH.exists():
        write_jsonl(chunks, DEFAULT_CHUNKS_PATH)
    manifest = write_manifest(chunks, DEFAULT_MANIFEST_PATH)
    return {
        "ok": True,
        "status": "initialized",
        "manifest": manifest.to_dict(),
        "corpus": corpus_status(),
    }


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="HalluGuard-Med corpus ingestion utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create local corpus folders and metadata files")

    medline = subparsers.add_parser("ingest-medlineplus", help="Ingest a MedlinePlus health topic XML export")
    medline.add_argument("--xml", required=True, type=_path, help="Path to MedlinePlus XML file")
    medline.add_argument("--output", default=str(DEFAULT_CHUNKS_PATH), type=_path)
    medline.add_argument("--append", action="store_true")

    bm25 = subparsers.add_parser("build-bm25", help="Build a standalone BM25 corpus index")
    bm25.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH), type=_path)

    qdrant = subparsers.add_parser("index-qdrant", help="Index corpus chunks into a separate Qdrant collection")
    qdrant.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH), type=_path)
    qdrant.add_argument("--collection", default=DEFAULT_CORPUS_COLLECTION)
    qdrant.add_argument("--max-chunks", type=int, default=None, help="Index only the first N chunks for testing")
    qdrant.add_argument("--batch-size", type=int, default=DEFAULT_QDRANT_BATCH_SIZE, help="Chunks to embed/upsert per checkpoint")
    qdrant.add_argument("--max-retries", type=int, default=DEFAULT_MAX_EMBED_RETRIES, help="Retries after embedding quota errors")
    qdrant.add_argument("--retry-sleep", type=int, default=DEFAULT_RETRY_SLEEP_SECONDS, help="Base seconds to sleep after quota errors")
    qdrant.add_argument(
        "--embed-requests-per-minute",
        type=int,
        default=DEFAULT_EMBED_REQUESTS_PER_MINUTE,
        help="Throttle Google embedding requests below project quota",
    )

    subparsers.add_parser("status", help="Show corpus status")

    args = parser.parse_args()
    if args.command == "init":
        result = initialize_corpus()
    elif args.command == "ingest-medlineplus":
        chunks = ingest_medlineplus_xml(args.xml, args.output, append=args.append)
        result = {"ok": True, "chunks_count": len(chunks), "output": str(args.output)}
    elif args.command == "build-bm25":
        result = build_bm25_index(args.chunks)
    elif args.command == "index-qdrant":
        result = index_qdrant_corpus(
            chunks_path=args.chunks,
            collection_name=args.collection,
            max_chunks=args.max_chunks,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep,
            embed_requests_per_minute=args.embed_requests_per_minute,
        )
    elif args.command == "status":
        result = corpus_status()
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
