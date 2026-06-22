from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


INTERNAL_CITATION_RE = re.compile(
    r"\s*\[[A-Z0-9][A-Z0-9_-]{2,}(?:-[A-Z0-9_-]+)*\]",
    re.IGNORECASE,
)
INTERNAL_CITATION_ID_RE = re.compile(
    r"^[A-Z0-9][A-Z0-9_-]{2,}(?:-[A-Z0-9_-]+)*$",
    re.IGNORECASE,
)


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
    if 0 < numeric < 0.1:
        return "<0.1%"
    if 99.9 < numeric < 100:
        return ">99.9%"
    return f"{numeric:.1f}%"


def _title_label(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip()
    return re.sub(r"\s+", " ", text).title() if text else ""


def _display_citation_label(item: dict[str, Any] | None) -> str:
    item = item or {}
    source = str(item.get("source") or "Retrieved evidence").strip()
    condition = _title_label(item.get("condition") or item.get("section") or "Medical Evidence")
    return f"{source} - {condition}" if condition else source


def _strip_internal_citation_ids(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(INTERNAL_CITATION_RE, "", value).strip()
    return value


def _label_for_citation_id(raw_id: Any, labels: dict[str, str]) -> str:
    raw_text = str(raw_id or "")
    if raw_text == "IMG1":
        return "Uploaded Image Analysis"
    if INTERNAL_CITATION_ID_RE.match(raw_text):
        return labels.get(raw_text) or "Retrieved Evidence"
    return labels.get(raw_text) or _title_label(raw_text)


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
    citations = _as_list(analysis.get("citations") or payload.get("citations"))
    citation_labels = {
        str(item.get("id")): _display_citation_label(item)
        for item in citations
        if isinstance(item, dict) and item.get("id")
    }
    display_citations = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        display = dict(item)
        display["id"] = _display_citation_label(item)
        display["text"] = _strip_internal_citation_ids(display.get("text"))
        display_citations.append(display)
    display_claims = []
    for item in _as_list(analysis.get("claim_verification")):
        if not isinstance(item, dict):
            continue
        display = dict(item)
        display["claim"] = _strip_internal_citation_ids(display.get("claim"))
        display["best_evidence"] = _strip_internal_citation_ids(display.get("best_evidence"))
        raw_citation = display.get("best_citation_id")
        if raw_citation:
            display["best_citation_id"] = _label_for_citation_id(raw_citation, citation_labels)
        display_claims.append(display)
    display_evidence_scores = []
    for item in _as_list(pre_generation.get("evidence_scores")):
        if not isinstance(item, dict):
            continue
        display = dict(item)
        raw_citation = display.get("citation_id")
        if raw_citation:
            display["citation_id"] = _label_for_citation_id(raw_citation, citation_labels)
        display_evidence_scores.append(display)
    display_retrieval_summary = dict(retrieval_summary)
    display_top_citations = []
    for item in _as_list(retrieval_summary.get("top_citations")):
        if not isinstance(item, dict):
            continue
        display = dict(item)
        raw_citation = display.get("citation_id")
        if raw_citation:
            display["citation_id"] = _label_for_citation_id(raw_citation, citation_labels)
        display_top_citations.append(display)
    if display_top_citations:
        display_retrieval_summary["top_citations"] = display_top_citations
    display_source_conflicts = []
    for item in _as_list(pre_generation.get("source_conflicts")):
        if not isinstance(item, dict):
            continue
        display = dict(item)
        for key in ("citation_a", "citation_b"):
            raw_citation = display.get(key)
            if raw_citation:
                display[key] = _label_for_citation_id(raw_citation, citation_labels)
        display_source_conflicts.append(display)

    return {
        "case_id": case_id,
        "timestamp": generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "query": payload.get("query") or payload.get("user_query") or "N/A",
        "final_response": _strip_internal_citation_ids(payload.get("final_response")) or "N/A",
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
        "retrieval_summary": display_retrieval_summary,
        "timing_metrics": _as_dict(meta.get("timing_ms")),
        "final_assessment": final_assessment,
        "claims": display_claims,
        "citations": display_citations,
        "evidence_scores": display_evidence_scores,
        "source_conflicts": display_source_conflicts,
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
