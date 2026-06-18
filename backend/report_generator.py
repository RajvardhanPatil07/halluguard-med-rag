from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _percent(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if 0 <= numeric <= 1:
        numeric *= 100
    return f"{numeric:.1f}%"


def build_report_data(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a HalluGuard-Med response payload into a report-friendly shape.

    This module intentionally does not call MedGemma, RAG, Qdrant, verification,
    radiology, or confidence scoring. It only formats fields already returned by
    the existing chat endpoint and submitted by the frontend.
    """
    analysis = _as_dict(payload.get("analysis"))
    confidence = _as_dict(analysis.get("confidence"))
    nli = _as_dict(analysis.get("nli"))
    imaging = _as_dict(analysis.get("imaging"))
    safety = _as_dict(analysis.get("safety"))
    pre_generation = _as_dict(safety.get("pre_generation"))
    post_generation = _as_dict(safety.get("post_generation"))
    claims_summary = _as_dict(analysis.get("claims_summary") or post_generation.get("claims_summary"))
    retrieval_summary = _as_dict(analysis.get("retrieval_summary"))
    final_assessment = _as_dict(analysis.get("final_assessment"))
    meta = _as_dict(payload.get("meta"))

    generated_at = datetime.now(timezone.utc)
    case_id = payload.get("case_id") or f"HGM-{generated_at.strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}"

    return {
        "case_id": case_id,
        "timestamp": generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "query": payload.get("query") or payload.get("user_query") or "N/A",
        "final_response": payload.get("final_response") or "N/A",
        "summary": {
            "risk_tier": analysis.get("risk_tier", "N/A"),
            "risk_score": analysis.get("risk_score", "N/A"),
            "confidence_label": confidence.get("label", "N/A"),
            "confidence_percentage": _percent(confidence.get("score")),
        },
        "analysis_results": {
            "kg": analysis.get("kg", "N/A"),
            "nli": nli.get("label", "N/A"),
            "nli_confidence": _percent(nli.get("confidence")),
            "rag_score": _percent(analysis.get("rag_score")),
            "rag_verified": analysis.get("rag_verified", "N/A"),
            "rag_error": analysis.get("rag_error"),
            "imaging": imaging.get("status", "N/A"),
        },
        "confidence_breakdown": _as_dict(confidence.get("breakdown")),
        "claims_summary": claims_summary,
        "retrieval_summary": retrieval_summary,
        "timing_metrics": _as_dict(meta.get("timing_ms")),
        "final_assessment": final_assessment,
        "claims": _as_list(analysis.get("claim_verification")),
        "citations": _as_list(analysis.get("citations") or payload.get("citations")),
        "evidence_scores": _as_list(pre_generation.get("evidence_scores")),
        "source_conflicts": _as_list(pre_generation.get("source_conflicts")),
        "imaging": {
            "status": imaging.get("status", "N/A"),
            "findings": _as_list(imaging.get("findings")),
            "percentage_scores": _as_dict(imaging.get("percentage_scores")),
            "critical": _as_list(imaging.get("critical")),
            "normal_score": imaging.get("normal_score"),
            "warnings": _as_list(imaging.get("warnings")),
        },
        "risk_findings": {
            "risk_reasons": _as_list(analysis.get("risk_reasons")),
            "warnings": _as_list(payload.get("warnings")),
        },
        "recommendations": _as_list(payload.get("suggestions")),
        "matched_conditions": _as_list(payload.get("matched_conditions")),
        "image_uploaded": bool(payload.get("image_uploaded")),
        "meta": meta,
    }
