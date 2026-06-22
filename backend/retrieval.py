"""
Hybrid retrieval primitives for HalluGuard-Med.

Primary source: corpus chunks in medical_corpus/processed/corpus_chunks.jsonl
and Qdrant collection halluguard_med_corpus_v1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    BM25_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_EMBEDDINGS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    qdrant_models = None
    QDRANT_AVAILABLE = False

try:
    from .settings import (
        GOOGLE_API_KEY,
        GOOGLE_EMBEDDING_DIM,
        GOOGLE_EMBEDDING_MODEL,
        QDRANT_API_KEY,
        QDRANT_URL,
        RAG_RRF_K,
        RAG_TOP_K_EACH,
        RAG_TOP_K_FINAL,
        RAG_VERIFIED_THRESHOLD,
    )
    from .structured_log import log_event
    from .corpus_schema import DEFAULT_CHUNKS_PATH, read_jsonl
    from .medical_entities import ENTITY_DISEASE, MedicalEntityExtractor
except ImportError:
    from settings import (
        GOOGLE_API_KEY,
        GOOGLE_EMBEDDING_DIM,
        GOOGLE_EMBEDDING_MODEL,
        QDRANT_API_KEY,
        QDRANT_URL,
        RAG_RRF_K,
        RAG_TOP_K_EACH,
        RAG_TOP_K_FINAL,
        RAG_VERIFIED_THRESHOLD,
    )
    from structured_log import log_event
    from corpus_schema import DEFAULT_CHUNKS_PATH, read_jsonl
    from medical_entities import ENTITY_DISEASE, MedicalEntityExtractor


CORPUS_QDRANT_COLLECTION = os.getenv("CORPUS_QDRANT_COLLECTION", "halluguard_med_corpus_v1")
_bm25_index = None
_chunks: list["EvidenceChunk"] = []
_chunk_by_id: dict[str, "EvidenceChunk"] = {}
_qdrant_client = None
_last_error = None
_indexed_signature = None
_retrieval_source = None
_active_qdrant_collection = None
_entity_extractor = None
_embedding_cache: dict[str, list[float]] = {}


class RetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    text: str
    source: str
    source_type: str
    condition: str
    section: str
    citation_id: str


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    source: str
    source_type: str
    condition: str
    section: str
    citation_id: str
    rank: int
    rrf_score: float
    normalized_score: float
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None


def _expand_medical_query(query: str) -> str:
    expanded = query
    if re.search(r"\banti[\s-]?d\b", query, re.IGNORECASE):
        expanded += (
            " Rh immune globulin Rho(D) immune globulin Rh incompatibility "
            "Rh negative pregnancy injection Rh antibodies"
        )
    if re.search(r"\b(paracetamol|acetaminophen|tylenol)\b", query, re.IGNORECASE):
        expanded += " paracetamol acetaminophen tylenol pain reliever analgesic dose dosage adult"
    return expanded


def _normalize_token_text(text: str) -> str:
    text = re.sub(r"\banti[\s-]?d\b", "antid", text, flags=re.IGNORECASE)
    text = re.sub(r"\brho\s*\(\s*d\s*\)", "rhod", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:acetaminophen|tylenol)\b", "paracetamol", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:rupture|ruptured|bursts|bursting)\b", "burst", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:inflamed|inflammation)\b", "inflame", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:vomiting|vomited|throwing up|throws up|emesis)\b", "vomit", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:nauseous|feeling sick|queasy)\b", "nausea", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhaemorrhagic\b", "hemorrhagic", text, flags=re.IGNORECASE)
    return text.lower()


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalize_token_text(text))
    return [
        token
        for token in tokens
        if len(token) > 1 or token.isdigit()
    ]


def _condition_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    tokens = []
    for token in normalized.split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def _get_entity_extractor() -> MedicalEntityExtractor:
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = MedicalEntityExtractor()
    return _entity_extractor


def _query_disease_terms(query: str) -> set[str]:
    entities = _get_entity_extractor().extract(query, source="query")
    return {
        _condition_key(entity.normalized)
        for entity in entities
        if entity.label == ENTITY_DISEASE
    }


def _condition_related_to_query(query: str, hit: RetrievalHit, disease_terms: set[str]) -> bool:
    if not disease_terms:
        return True
    condition = _condition_key(hit.condition)
    haystack = _condition_key(f"{hit.condition} {hit.text[:500]}")
    for term in disease_terms:
        if not term:
            continue
        term_tokens = set(term.split())
        condition_tokens = set(condition.split())
        if term == condition or term in condition or condition in term:
            return True
        if term_tokens and condition_tokens and len(term_tokens & condition_tokens) / len(term_tokens) >= 0.6:
            return True
        if term in haystack:
            return True
    return False


def build_chunks_from_corpus() -> list[EvidenceChunk]:
    corpus_chunks = read_jsonl(DEFAULT_CHUNKS_PATH)
    chunks = [
        EvidenceChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            source_type=chunk.source_type,
            condition=chunk.condition,
            section=chunk.section,
            citation_id=chunk.citation_id,
        )
        for chunk in corpus_chunks
    ]
    if not chunks:
        raise RetrievalError(f"No corpus chunks found at: {DEFAULT_CHUNKS_PATH}")
    return chunks


def _chunks_signature(chunks: list[EvidenceChunk], source_name: str) -> str:
    payload = json.dumps([asdict(chunk) for chunk in chunks], sort_keys=True)
    return hashlib.sha256(f"{source_name}:{payload}".encode("utf-8")).hexdigest()


def _load_primary_chunks() -> tuple[str, list[EvidenceChunk], str]:
    try:
        chunks = build_chunks_from_corpus()
        return "corpus", chunks, CORPUS_QDRANT_COLLECTION
    except Exception as corpus_exc:
        raise RetrievalError(
            f"Corpus retrieval source is unavailable: {corpus_exc}. "
            f"Expected corpus chunks at: {DEFAULT_CHUNKS_PATH}"
        ) from corpus_exc


def _require_dependencies() -> None:
    missing = []
    if not BM25_AVAILABLE:
        missing.append("rank-bm25")
    if not GOOGLE_EMBEDDINGS_AVAILABLE:
        missing.append("google-generativeai")
    if not QDRANT_AVAILABLE:
        missing.append("qdrant-client")
    if missing:
        raise RetrievalError(f"Missing Hybrid RAG dependencies: {', '.join(missing)}")
    if not GOOGLE_API_KEY:
        raise RetrievalError(f"GOOGLE_API_KEY is not configured for {GOOGLE_EMBEDDING_MODEL}")


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    _require_dependencies()
    genai.configure(api_key=GOOGLE_API_KEY)
    vectors: list[list[float]] = []
    for text in texts:
        response = genai.embed_content(
            model=GOOGLE_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
            output_dimensionality=GOOGLE_EMBEDDING_DIM,
        )
        vector = _extract_embedding(response)
        if not vector:
            raise RetrievalError("Google embedding API returned an empty embedding")
        if len(vector) != GOOGLE_EMBEDDING_DIM:
            raise RetrievalError(
                f"Embedding dimension mismatch: expected {GOOGLE_EMBEDDING_DIM}, got {len(vector)}"
            )
        vectors.append([float(value) for value in vector])
    return vectors


def embed_query(query: str) -> list[float]:
    _require_dependencies()
    cache_key = f"{GOOGLE_EMBEDDING_MODEL}:{GOOGLE_EMBEDDING_DIM}:{query}"
    cached = _embedding_cache.get(cache_key)
    if cached is not None:
        log_event(
            "retrieval",
            "embedding_generation_completed",
            duration_ms=0.0,
            cache_hit=True,
            input_chars=len(query),
        )
        return cached
    start = perf_counter()
    genai.configure(api_key=GOOGLE_API_KEY)
    response = genai.embed_content(
        model=GOOGLE_EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
        output_dimensionality=GOOGLE_EMBEDDING_DIM,
    )
    vector = _extract_embedding(response)
    if not vector:
        raise RetrievalError("Google embedding API returned an empty query embedding")
    if len(vector) != GOOGLE_EMBEDDING_DIM:
        raise RetrievalError(
            f"Query embedding dimension mismatch: expected {GOOGLE_EMBEDDING_DIM}, got {len(vector)}"
        )
    embedding = [float(value) for value in vector]
    _embedding_cache[cache_key] = embedding
    log_event(
        "retrieval",
        "embedding_generation_completed",
        duration_ms=round((perf_counter() - start) * 1000.0, 2),
        cache_hit=False,
        input_chars=len(query),
        dimensions=len(embedding),
    )
    return embedding


def _extract_embedding(response: Any) -> list[float] | None:
    if isinstance(response, dict):
        return response.get("embedding")
    return getattr(response, "embedding", None)


def _score_signal(value: float | None, cap: float | None = None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    score = max(0.0, float(value))
    if cap is not None and cap > 0:
        score = min(score / cap, 1.0)
    return min(score, 1.0)


def _rrf_signal(hit: RetrievalHit) -> float:
    max_possible = 2.0 / (RAG_RRF_K + 1)
    if hit.rrf_score and max_possible:
        return _score_signal(float(hit.rrf_score) / max_possible)
    return _score_signal(hit.normalized_score)


def calculate_dynamic_rag_score(hits: list[RetrievalHit]) -> float:
    if not hits:
        return 0.0

    top_hit = hits[0]
    top_rrf = _rrf_signal(top_hit)
    mean_rrf = sum(_rrf_signal(hit) for hit in hits) / len(hits)
    both_legs = sum(
        1
        for hit in hits
        if hit.bm25_rank is not None and hit.dense_rank is not None
    ) / len(hits)

    condition_diversity = len({hit.condition for hit in hits}) / len(hits)
    best_dense = max((_score_signal(hit.dense_score) for hit in hits), default=0.0)
    best_bm25 = max((_score_signal(hit.bm25_score, cap=10.0) for hit in hits), default=0.0)

    score = (
        (top_rrf * 0.35)
        + (mean_rrf * 0.20)
        + (both_legs * 0.25)
        + (best_dense * 0.10)
        + (best_bm25 * 0.05)
        + (condition_diversity * 0.05)
    )
    return round(max(0.0, min(score, 1.0)), 4)


def build_retrieval_summary(hits: list[RetrievalHit], bm25_count: int, dense_count: int) -> dict[str, Any]:
    if not hits:
        return {
            "hits_count": 0,
            "bm25_count": bm25_count,
            "dense_count": dense_count,
            "source_agreement_ratio": 0.0,
            "top_conditions": [],
            "top_sections": [],
            "top_citations": [],
        }
    source_agreement = [
        hit for hit in hits
        if hit.bm25_rank is not None and hit.dense_rank is not None
    ]
    conditions: dict[str, int] = {}
    sections: dict[str, int] = {}
    for hit in hits:
        conditions[hit.condition] = conditions.get(hit.condition, 0) + 1
        sections[hit.section] = sections.get(hit.section, 0) + 1
    return {
        "hits_count": len(hits),
        "bm25_count": bm25_count,
        "dense_count": dense_count,
        "source_agreement_ratio": round(len(source_agreement) / len(hits), 4),
        "top_conditions": [
            {"condition": key, "count": value}
            for key, value in sorted(conditions.items(), key=lambda item: item[1], reverse=True)[:5]
        ],
        "top_sections": [
            {"section": key, "count": value}
            for key, value in sorted(sections.items(), key=lambda item: item[1], reverse=True)[:5]
        ],
        "top_citations": [
            {
                "citation_id": hit.citation_id,
                "condition": hit.condition,
                "score": hit.normalized_score,
                "bm25_rank": hit.bm25_rank,
                "dense_rank": hit.dense_rank,
            }
            for hit in hits[:5]
        ],
    }


def _ensure_qdrant_collection(client, collection_name: str, create_if_missing: bool = False) -> None:
    collections = client.get_collections().collections
    exists = any(collection.name == collection_name for collection in collections)
    if exists:
        return
    if not create_if_missing:
        raise RetrievalError(
            f"Qdrant collection '{collection_name}' does not exist. "
            "Run corpus indexing before using corpus retrieval."
        )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=GOOGLE_EMBEDDING_DIM,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def _ensure_indexes() -> None:
    global _bm25_index, _chunks, _chunk_by_id, _indexed_signature, _last_error
    global _retrieval_source, _active_qdrant_collection
    _require_dependencies()
    source_name, chunks, collection_name = _load_primary_chunks()
    signature = _chunks_signature(chunks, source_name)
    if _bm25_index is not None and _indexed_signature == signature:
        return

    try:
        _bm25_index = BM25Okapi([_tokenize(chunk.text) for chunk in chunks])
        _chunks = chunks
        _chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        client = _get_qdrant_client()
        _ensure_qdrant_collection(client, collection_name, create_if_missing=False)
        _indexed_signature = signature
        _retrieval_source = source_name
        _active_qdrant_collection = collection_name
        _last_error = None
        log_event(
            "retrieval",
            "retrieval_indexes_ready",
            retrieval_source=source_name,
            qdrant_collection=collection_name,
            chunks_count=len(chunks),
            corpus_chunks_path=str(DEFAULT_CHUNKS_PATH),
        )
    except Exception as exc:
        _last_error = str(exc)
        raise RetrievalError(f"Hybrid RAG indexing failed: {exc}") from exc


def _bm25_search(query: str, top_k: int) -> list[dict[str, Any]]:
    start = perf_counter()
    if _bm25_index is None:
        raise RetrievalError("BM25 index is not initialized")
    scores = _bm25_index.get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
    results = []
    for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
        if not math.isfinite(float(score)):
            continue
        chunk = _chunks[idx]
        results.append({
            "chunk_id": chunk.chunk_id,
            "rank": rank,
            "score": float(score),
        })
    log_event(
        "retrieval",
        "bm25_search_completed",
        duration_ms=round((perf_counter() - start) * 1000.0, 2),
        retrieval_source=_retrieval_source,
        hits_count=len(results),
        top_hits=[
            {
                "chunk_id": result["chunk_id"],
                "rank": result["rank"],
                "score": round(float(result["score"]), 4),
            }
            for result in results[:5]
        ],
    )
    return results


def _payload_to_chunk(payload: dict[str, Any], point_id: Any) -> EvidenceChunk | None:
    if not payload:
        return None
    text = str(payload.get("text") or "").strip()
    if not text:
        return None
    chunk_id = str(payload.get("chunk_id") or point_id)
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        source=str(payload.get("source") or "qdrant_payload"),
        source_type=str(payload.get("source_type") or "corpus"),
        condition=str(payload.get("condition") or payload.get("title") or "unknown"),
        section=str(payload.get("section") or "unknown"),
        citation_id=str(payload.get("citation_id") or chunk_id),
    )


def _dense_search(query: str, top_k: int) -> list[dict[str, Any]]:
    start = perf_counter()
    client = _get_qdrant_client()
    query_vector = embed_query(query)
    collection_name = _active_qdrant_collection or CORPUS_QDRANT_COLLECTION
    qdrant_start = perf_counter()
    if hasattr(client, "search"):
        points = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
    else:
        points = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
    qdrant_duration_ms = round((perf_counter() - qdrant_start) * 1000.0, 2)
    results = []
    for rank, point in enumerate(points, start=1):
        payload = getattr(point, "payload", None) or {}
        chunk = _payload_to_chunk(payload, getattr(point, "id", ""))
        if chunk is not None:
            _chunk_by_id.setdefault(chunk.chunk_id, chunk)
        chunk_id = chunk.chunk_id if chunk is not None else str(point.id)
        results.append({
            "chunk_id": chunk_id,
            "rank": rank,
            "score": float(point.score),
            "payload": payload,
        })
    log_event(
        "retrieval",
        "dense_search_completed",
        duration_ms=round((perf_counter() - start) * 1000.0, 2),
        qdrant_duration_ms=qdrant_duration_ms,
        retrieval_source=_retrieval_source,
        qdrant_collection=collection_name,
        hits_count=len(results),
        top_hits=[
            {
                "chunk_id": result["chunk_id"],
                "rank": result["rank"],
                "score": round(float(result["score"]), 4),
                "source": (result.get("payload") or {}).get("source"),
                "section": (result.get("payload") or {}).get("section"),
            }
            for result in results[:5]
        ],
    )
    return results


def rrf_fuse(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    final_k: int,
) -> list[RetrievalHit]:
    start = perf_counter()
    fused: dict[str, dict[str, Any]] = {}

    for result in bm25_results:
        item = fused.setdefault(result["chunk_id"], {"rrf_score": 0.0})
        item["rrf_score"] += 1.0 / (RAG_RRF_K + result["rank"])
        item["bm25_rank"] = result["rank"]
        item["bm25_score"] = result["score"]

    for result in dense_results:
        item = fused.setdefault(result["chunk_id"], {"rrf_score": 0.0})
        item["rrf_score"] += 1.0 / (RAG_RRF_K + result["rank"])
        item["dense_rank"] = result["rank"]
        item["dense_score"] = result["score"]
        if "payload" in result:
            item["payload"] = result["payload"]

    max_possible = 2.0 / (RAG_RRF_K + 1)
    ranked = sorted(
        fused.items(),
        key=lambda item: item[1]["rrf_score"],
        reverse=True,
    )

    hits: list[RetrievalHit] = []
    seen_texts: set[str] = set()
    for chunk_id, values in ranked:
        chunk = _chunk_by_id.get(chunk_id)
        if chunk is None and isinstance(values.get("payload"), dict):
            chunk = _payload_to_chunk(values["payload"], chunk_id)
            if chunk is not None:
                _chunk_by_id[chunk.chunk_id] = chunk
        if chunk is None:
            continue
        text_key = re.sub(r"\s+", " ", chunk.text.strip().lower())
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        normalized = values["rrf_score"] / max_possible if max_possible else 0.0
        rank = len(hits) + 1
        hits.append(RetrievalHit(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            source_type=chunk.source_type,
            condition=chunk.condition,
            section=chunk.section,
            citation_id=chunk.citation_id,
            rank=rank,
            rrf_score=round(float(values["rrf_score"]), 6),
            normalized_score=round(float(normalized), 4),
            bm25_rank=values.get("bm25_rank"),
            bm25_score=round(float(values["bm25_score"]), 4) if "bm25_score" in values else None,
            dense_rank=values.get("dense_rank"),
            dense_score=round(float(values["dense_score"]), 4) if "dense_score" in values else None,
        ))
        if len(hits) >= final_k:
            break
    log_event(
        "retrieval",
        "rrf_fusion_completed",
        duration_ms=round((perf_counter() - start) * 1000.0, 2),
        retrieval_source=_retrieval_source,
        bm25_count=len(bm25_results),
        dense_count=len(dense_results),
        fused_count=len(hits),
        top_hits=[
            {
                "citation_id": hit.citation_id,
                "condition": hit.condition,
                "section": hit.section,
                "score": hit.normalized_score,
                "bm25_rank": hit.bm25_rank,
                "dense_rank": hit.dense_rank,
            }
            for hit in hits
        ],
    )
    return hits


def _rerank_domain_specific(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    if not re.search(r"\banti[\s-]?d\b", query, re.IGNORECASE):
        return hits

    positive_terms = (
        "rh incompatibility",
        "rh immune globulin",
        "rho(d)",
        "rh-negative",
        "rh negative",
        "rh antibodies",
    )
    negative_terms = ("vitamin d", "vitamin d deficiency")

    def score(hit: RetrievalHit) -> float:
        haystack = f"{hit.condition} {hit.section} {hit.text}".lower()
        bonus = sum(0.18 for term in positive_terms if term in haystack)
        penalty = sum(0.35 for term in negative_terms if term in haystack)
        return hit.normalized_score + bonus - penalty

    ranked = sorted(hits, key=score, reverse=True)
    non_vitamin = [
        hit for hit in ranked
        if "vitamin d" not in f"{hit.condition} {hit.text}".lower()
    ]
    if len(non_vitamin) >= 3:
        ranked = non_vitamin + [hit for hit in ranked if hit not in non_vitamin]
    reranked = []
    for rank, hit in enumerate(ranked, start=1):
        hit.rank = rank
        reranked.append(hit)
    return reranked


def _rerank_exact_condition_matches(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    disease_terms = _query_disease_terms(query)
    if not disease_terms:
        return hits

    exact_conditions = {
        _condition_key(hit.condition)
        for hit in hits
        if _condition_key(hit.condition) in disease_terms
    }

    def score(hit: RetrievalHit) -> float:
        condition = _condition_key(hit.condition)
        section = str(hit.section or "").lower()
        text_key = _condition_key(hit.text[:220])
        bonus = 0.0
        condition_relevant = False
        if condition in disease_terms:
            bonus += 0.55
            condition_relevant = True
        elif any(term and term in condition.split(" ") for term in disease_terms):
            bonus += 0.08
            condition_relevant = True
        elif any(term and term in condition for term in disease_terms):
            bonus += 0.18
            condition_relevant = True
        if exact_conditions and condition not in exact_conditions and any(term in condition for term in disease_terms):
            bonus -= 0.12
        if any(term and term in text_key for term in disease_terms):
            bonus += 0.08
        if condition_relevant and section == "overview":
            bonus += 0.05
        elif section in {"related_topics", "topic_groups"}:
            bonus -= 0.20
        if not _condition_related_to_query(query, hit, disease_terms):
            bonus -= 0.60
        return hit.normalized_score + bonus

    ranked_with_scores = sorted(
        [(score(hit), hit) for hit in hits],
        key=lambda item: item[0],
        reverse=True,
    )
    reranked = []
    for rank, (_adjusted_score, hit) in enumerate(ranked_with_scores, start=1):
        hit.rank = rank
        reranked.append(hit)
    return reranked


def _prefer_primary_condition_hits(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    disease_terms = _query_disease_terms(query)
    if not disease_terms:
        return hits
    related = [hit for hit in hits if _condition_related_to_query(query, hit, disease_terms)]
    unrelated = [hit for hit in hits if hit not in related]
    # Keep broad recall during BM25/dense search, but do not pass severe condition
    # mismatches into final context when the primary condition was retrieved.
    unrelated = [hit for hit in unrelated if hit.normalized_score >= 0.15]
    ordered = related + unrelated
    for rank, hit in enumerate(ordered, start=1):
        hit.rank = rank
    return ordered


def _rerank_topical_matches(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return hits

    def topical_score(hit: RetrievalHit) -> float:
        condition_terms = set(_tokenize(hit.condition))
        section_terms = set(_tokenize(hit.section))
        text_terms = set(_tokenize(hit.text))
        condition_overlap = len(query_terms & condition_terms) / len(query_terms)
        text_overlap = len(query_terms & text_terms) / len(query_terms)
        section_overlap = len(query_terms & section_terms) / len(query_terms)
        source_agreement = 0.08 if hit.bm25_rank is not None and hit.dense_rank is not None else 0.0
        return (
            hit.normalized_score
            + (0.28 * condition_overlap)
            + (0.18 * text_overlap)
            + (0.06 * section_overlap)
            + source_agreement
        )

    ranked = sorted(hits, key=topical_score, reverse=True)
    for rank, hit in enumerate(ranked, start=1):
        hit.rank = rank
    return ranked


def _rerank_retrieval_hits(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    hits = _rerank_topical_matches(query, hits)
    hits = _rerank_exact_condition_matches(query, hits)
    hits = _rerank_domain_specific(query, hits)
    hits = _prefer_primary_condition_hits(query, hits)
    return hits


def hybrid_retrieve(query: str) -> dict[str, Any]:
    global _last_error
    start = perf_counter()
    try:
        _ensure_indexes()
        retrieval_query = _expand_medical_query(query)
        bm25_results = _bm25_search(retrieval_query, RAG_TOP_K_EACH)
        dense_results = _dense_search(retrieval_query, RAG_TOP_K_EACH)
        fused_hits = rrf_fuse(bm25_results, dense_results, RAG_TOP_K_FINAL)
        hits = _rerank_retrieval_hits(query, fused_hits)
        if not hits and fused_hits:
            hits = fused_hits
            for rank, hit in enumerate(hits, start=1):
                hit.rank = rank
            log_event(
                "retrieval",
                "rerank_returned_no_hits_using_fused_fallback",
                "warning",
                query_chars=len(query),
                fused_count=len(fused_hits),
                disease_terms=sorted(_query_disease_terms(query)),
                fallback_citations=[hit.citation_id for hit in hits],
            )
        if not hits:
            raise RetrievalError("Hybrid RAG returned no evidence")
        score = calculate_dynamic_rag_score(hits)
        retrieval_summary = build_retrieval_summary(hits, len(bm25_results), len(dense_results))
        verified = score >= RAG_VERIFIED_THRESHOLD
        _last_error = None
        log_event(
            "retrieval",
            "hybrid_retrieve_success",
            duration_ms=round((perf_counter() - start) * 1000.0, 2),
            query_chars=len(query),
            expanded_query=retrieval_query if retrieval_query != query else None,
            retrieval_source=_retrieval_source,
            qdrant_collection=_active_qdrant_collection,
            rag_score=score,
            rag_verified=verified,
            hits_count=len(hits),
            bm25_count=len(bm25_results),
            dense_count=len(dense_results),
            fusion_results=[
                {
                    "citation_id": hit.citation_id,
                    "condition": hit.condition,
                    "section": hit.section,
                    "score": hit.normalized_score,
                }
                for hit in hits
            ],
        )
        return {
            "ok": True,
            "score": round(float(score), 4),
            "verified": verified,
            "hits": [asdict(hit) for hit in hits],
            "bm25_count": len(bm25_results),
            "dense_count": len(dense_results),
            "retrieval_summary": retrieval_summary,
            "retrieval_source": _retrieval_source,
            "qdrant_collection": _active_qdrant_collection,
            "expanded_query": retrieval_query if retrieval_query != query else None,
            "error": None,
        }
    except Exception as exc:
        _last_error = str(exc)
        log_event(
            "retrieval",
            "hybrid_retrieve_failed",
            "error",
            query_chars=len(query),
            error=str(exc),
        )
        return {
            "ok": False,
            "score": None,
            "verified": None,
            "hits": [],
            "bm25_count": 0,
            "dense_count": 0,
            "retrieval_source": _retrieval_source,
            "qdrant_collection": _active_qdrant_collection,
            "error": str(exc),
        }


def setup_qdrant_index() -> dict[str, Any]:
    try:
        _ensure_indexes()
        return {
            "ok": True,
            "status": "ready",
            "chunks_count": len(_chunks),
            "collection": _active_qdrant_collection,
            "retrieval_source": _retrieval_source,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "chunks_count": len(_chunks),
            "collection": _active_qdrant_collection or CORPUS_QDRANT_COLLECTION,
            "retrieval_source": _retrieval_source,
            "error": str(exc),
        }


def get_retrieval_status() -> dict[str, Any]:
    return {
        "bm25_available": BM25_AVAILABLE,
        "google_embeddings_available": GOOGLE_EMBEDDINGS_AVAILABLE,
        "qdrant_available": QDRANT_AVAILABLE,
        "google_api_key_configured": bool(GOOGLE_API_KEY),
        "embedding_model": GOOGLE_EMBEDDING_MODEL,
        "embedding_dim": GOOGLE_EMBEDDING_DIM,
        "qdrant_url": QDRANT_URL,
        "qdrant_collection": _active_qdrant_collection or CORPUS_QDRANT_COLLECTION,
        "corpus_qdrant_collection": CORPUS_QDRANT_COLLECTION,
        "retrieval_source": _retrieval_source,
        "corpus_chunks_path": str(DEFAULT_CHUNKS_PATH),
        "corpus_chunks_exists": DEFAULT_CHUNKS_PATH.exists(),
        "bm25_index_built": _bm25_index is not None,
        "qdrant_indexed": _indexed_signature is not None,
        "chunks_count": len(_chunks),
        "top_k_each": RAG_TOP_K_EACH,
        "top_k_final": RAG_TOP_K_FINAL,
        "rrf_k": RAG_RRF_K,
        "verified_threshold": RAG_VERIFIED_THRESHOLD,
        "last_error": _last_error,
    }
