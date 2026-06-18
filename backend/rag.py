"""
Compatibility wrapper for the Hybrid RAG pipeline.
"""

from typing import Any

try:
    from .rag_pipeline import get_rag_pipeline_status, run_rag_pipeline
except ImportError:
    from rag_pipeline import get_rag_pipeline_status, run_rag_pipeline


def retrieve_context(query: str, response: str | None = None) -> dict[str, Any]:
    """
    Compatibility entry point for older verifier code.
    Uses Hybrid RAG and returns explicit failure state on errors.
    """
    try:
        result = run_rag_pipeline(query)
        return {
            "score": result["rag_score"],
            "verified": result["rag_verified"],
            "context": [citation["text"] for citation in result["citations"]],
            "citations": result["citations"],
            "error": None,
        }
    except Exception as exc:
        return {
            "score": None,
            "verified": None,
            "context": [],
            "citations": [],
            "error": str(exc),
        }


def preload_rag() -> bool:
    try:
        run_rag_pipeline("medical evidence readiness check")
        return True
    except Exception:
        return False


def get_rag_status() -> dict[str, Any]:
    return get_rag_pipeline_status()
