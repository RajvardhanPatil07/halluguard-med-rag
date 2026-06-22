"""
Grounded Hybrid RAG orchestration for HalluGuard-Med.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

try:
    from .retrieval import get_retrieval_status, hybrid_retrieve
    from .safety_pipeline import build_safety_context_note, run_pre_generation_safety
    from .structured_log import log_event
except ImportError:
    from retrieval import get_retrieval_status, hybrid_retrieve
    from safety_pipeline import build_safety_context_note, run_pre_generation_safety
    from structured_log import log_event


class RagPipelineError(RuntimeError):
    pass


def build_grounded_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            "\n".join([
                f"Source: {hit['source']}",
                f"Condition: {hit['condition']}",
                f"Section: {hit['section']}",
                f"Evidence: {hit['text']}",
            ])
        )
    return "\n\n".join(blocks)


def build_grounded_query(query: str, context: str, safety_note: str = "") -> str:
    safety_block = f"{safety_note}\n" if safety_note else ""
    return (
        "Use only the retrieved medical evidence below when answering. "
        "Do not invent facts, diagnoses, doses, risks, or treatment recommendations. "
        "Every important medical statement must be supported by retrieved evidence. "
        "When evidence is weak or insufficient, explicitly say that the retrieved evidence is insufficient. "
        "Mention uncertainty when evidence is weak. "
        "Never claim an unsupported diagnosis. "
        "If the evidence is insufficient, say that the retrieved evidence is insufficient. "
        "Do not present the answer as a final diagnosis or treatment order. "
        "Cite supporting evidence using human-readable source and condition names, such as (MedlinePlus - Diabetes Complications). "
        "Do not include internal Citation IDs such as [M-CONDITION-SECTION-001] in the answer. "
        "Do not copy "
        "or reproduce the full Source/Condition/Section/Evidence blocks in the answer. "
        "Give a detailed, patient-friendly response rather than a one-line answer. "
        "Use this structure when possible: Overview, Key points, What the evidence supports, "
        "and Safety notes or when to seek medical care. "
        "Explain important terms briefly. "
        "Do not add facts, doses, diagnoses, or treatment recommendations that are not supported by the retrieved evidence.\n\n"
        f"{safety_block}"
        f"Retrieved evidence:\n{context}\n\n"
        f"User query: {query}"
    )


def run_rag_pipeline(query: str) -> dict[str, Any]:
    pipeline_start = perf_counter()
    retrieval_start = perf_counter()
    retrieval = hybrid_retrieve(query)
    retrieval_duration_ms = round((perf_counter() - retrieval_start) * 1000.0, 2)
    if not retrieval.get("ok"):
        raise RagPipelineError(
            retrieval.get("error") or "Hybrid RAG retrieval failed"
        )

    safety_start = perf_counter()
    safety_precheck = run_pre_generation_safety(query, retrieval)
    safety_duration_ms = round((perf_counter() - safety_start) * 1000.0, 2)
    context_start = perf_counter()
    hits = safety_precheck["filtered_hits"]
    context = build_grounded_context(hits)
    safety_note = build_safety_context_note(safety_precheck)
    grounded_query = build_grounded_query(query, context, safety_note)
    context_duration_ms = round((perf_counter() - context_start) * 1000.0, 2)
    citation_start = perf_counter()
    citations = [
        {
            "id": hit["citation_id"],
            "source": hit["source"],
            "source_type": hit["source_type"],
            "condition": hit["condition"],
            "section": hit["section"],
            "text": hit["text"],
            "rank": hit["rank"],
            "score": hit["normalized_score"],
            "bm25_rank": hit["bm25_rank"],
            "bm25_score": hit["bm25_score"],
            "dense_rank": hit["dense_rank"],
            "dense_score": hit["dense_score"],
            "rrf_score": hit["rrf_score"],
        }
        for hit in hits
    ]
    citation_duration_ms = round((perf_counter() - citation_start) * 1000.0, 2)
    timing_ms = {
        "retrieval": retrieval_duration_ms,
        "pre_generation_safety": safety_duration_ms,
        "context_build": context_duration_ms,
        "citation_selection": citation_duration_ms,
        "total": round((perf_counter() - pipeline_start) * 1000.0, 2),
    }
    log_event(
        "rag",
        "rag_pipeline_completed",
        duration_ms=timing_ms["total"],
        retrieval_ms=retrieval_duration_ms,
        pre_generation_safety_ms=safety_duration_ms,
        context_build_ms=context_duration_ms,
        citation_selection_ms=citation_duration_ms,
        hits_count=len(retrieval.get("hits", [])),
        filtered_hits_count=len(hits),
    )
    return {
        "query": query,
        "grounded_query": grounded_query,
        "context": context,
        "citations": citations,
        "rag_score": retrieval["score"],
        "rag_verified": retrieval["verified"],
        "retrieval_summary": retrieval.get("retrieval_summary", {}),
        "retrieval": retrieval,
        "safety_precheck": safety_precheck,
        "timing_ms": timing_ms,
    }


def get_rag_pipeline_status() -> dict[str, Any]:
    return get_retrieval_status()
