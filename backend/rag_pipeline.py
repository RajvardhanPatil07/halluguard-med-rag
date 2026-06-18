"""
Grounded Hybrid RAG orchestration for HalluGuard-Med.
"""

from __future__ import annotations

from typing import Any

try:
    from .retrieval import get_retrieval_status, hybrid_retrieve
    from .safety_pipeline import build_safety_context_note, run_pre_generation_safety
except ImportError:
    from retrieval import get_retrieval_status, hybrid_retrieve
    from safety_pipeline import build_safety_context_note, run_pre_generation_safety


class RagPipelineError(RuntimeError):
    pass


def build_grounded_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            "\n".join([
                f"[{hit['citation_id']}]",
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
        "If the evidence is insufficient, say that the retrieved evidence is insufficient. "
        "Do not present the answer as a final diagnosis or treatment order. "
        "Cite supporting evidence using citation IDs such as [E1], but do not copy "
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
    retrieval = hybrid_retrieve(query)
    if not retrieval.get("ok"):
        raise RagPipelineError(
            retrieval.get("error") or "Hybrid RAG retrieval failed"
        )

    safety_precheck = run_pre_generation_safety(query, retrieval)
    hits = safety_precheck["filtered_hits"]
    context = build_grounded_context(hits)
    safety_note = build_safety_context_note(safety_precheck)
    grounded_query = build_grounded_query(query, context, safety_note)
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
    }


def get_rag_pipeline_status() -> dict[str, Any]:
    return get_retrieval_status()
