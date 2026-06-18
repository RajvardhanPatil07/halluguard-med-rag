"""
Evidence-gated clinical safety pipeline for HalluGuard-Med.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

try:
    from .clinical_claims import (
        ClinicalClaimVerifier,
        ClaimVerificationResult,
        claim_results_to_dicts,
        claims_to_dicts,
        get_claim_verification_status,
    )
    from .confidence_fusion import ConfidenceFusion
    from .evidence_scoring import EvidenceScore, EvidenceScorer, evidence_scores_to_dicts
    from .medical_entities import MedicalEntityExtractor, entities_to_dicts
    from .radiology_analyzer import format_finding_assessment
    from .source_conflicts import SourceConflict, SourceConflictDetector, conflicts_to_dicts
    from .structured_log import log_event
except ImportError:
    from clinical_claims import (
        ClinicalClaimVerifier,
        ClaimVerificationResult,
        claim_results_to_dicts,
        claims_to_dicts,
        get_claim_verification_status,
    )
    from confidence_fusion import ConfidenceFusion
    from evidence_scoring import EvidenceScore, EvidenceScorer, evidence_scores_to_dicts
    from medical_entities import MedicalEntityExtractor, entities_to_dicts
    from radiology_analyzer import format_finding_assessment
    from source_conflicts import SourceConflict, SourceConflictDetector, conflicts_to_dicts
    from structured_log import log_event


_entity_extractor = MedicalEntityExtractor()
_conflict_detector = SourceConflictDetector()
_evidence_scorer = EvidenceScorer(_entity_extractor)
_claim_verifier = ClinicalClaimVerifier(_entity_extractor)
_confidence_fusion = ConfidenceFusion()


def _build_claims_summary(claim_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(claim_rows)
    by_status: dict[str, int] = {}
    support_scores = []
    for row in claim_rows:
        status = str(row.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        try:
            support_scores.append(float(row.get("support_score") or 0.0))
        except (TypeError, ValueError):
            pass
    return {
        "total": total,
        "supported": by_status.get("supported", 0),
        "weak_support": by_status.get("weak_support", 0),
        "unsupported": by_status.get("unsupported", 0),
        "insufficient": by_status.get("insufficient", 0),
        "contradicted": by_status.get("contradicted", 0),
        "mean_support": round(sum(support_scores) / len(support_scores), 4) if support_scores else 0.0,
        "by_status": by_status,
    }


def _hits_by_passed_score(hits: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passed = {
        row["citation_id"]: row
        for row in score_rows
        if row.get("passed")
    }
    filtered = [
        hit
        for hit in hits
        if str(hit.get("citation_id", "")) in passed
    ]
    if filtered:
        return filtered
    ranked_scores = sorted(score_rows, key=lambda row: row.get("final_score", 0.0), reverse=True)
    keep = {row["citation_id"] for row in ranked_scores[:2]}
    return [hit for hit in hits if str(hit.get("citation_id", "")) in keep] or hits[:2]


def run_pre_generation_safety(query: str, retrieval: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter()
    hits = retrieval.get("hits", [])
    query_entities = _entity_extractor.extract(query, source="query")
    evidence_entities = _entity_extractor.extract_from_hits(hits)
    conflicts = _conflict_detector.detect(hits, evidence_entities)
    evidence_start = perf_counter()
    evidence_scores = _evidence_scorer.score_hits(
        hits=hits,
        query_entities=query_entities,
        evidence_entities=evidence_entities,
        conflicts=conflicts,
        query_text=query,
    )
    evidence_duration_ms = round((perf_counter() - evidence_start) * 1000.0, 2)
    evidence_score_rows = evidence_scores_to_dicts(evidence_scores)
    if any(entity.label == "disease" for entity in query_entities):
        disease_terms = {
            entity.normalized
            for entity in query_entities
            if entity.label == "disease"
        }
        for row in evidence_score_rows:
            hit = next((item for item in hits if item.get("citation_id") == row["citation_id"]), None)
            if hit and hit.get("condition", "").lower() not in disease_terms and row.get("entity_overlap", 0) < 0.75:
                row["passed"] = False
                row.setdefault("reasons", []).append("condition_not_aligned_with_detected_disease")
    filtered_hits = _hits_by_passed_score(hits, evidence_score_rows)

    high_conflicts = [conflict for conflict in conflicts if conflict.severity == "high"]
    warnings = []
    if high_conflicts:
        warnings.append("Retrieved sources contain high-severity medical conflicts.")
    if len(filtered_hits) < len(hits):
        warnings.append("Some retrieved chunks were excluded by evidence quality scoring.")

    log_event(
        "safety",
        "pre_generation_completed",
        duration_ms=round((perf_counter() - start) * 1000.0, 2),
        evidence_scoring_duration_ms=evidence_duration_ms,
        query_entities_count=len(query_entities),
        hits_count=len(hits),
        filtered_hits_count=len(filtered_hits),
        conflicts_count=len(conflicts),
    )

    return {
        "query_entities": entities_to_dicts(query_entities),
        "evidence_entities": {
            citation_id: entities_to_dicts(entities)
            for citation_id, entities in evidence_entities.items()
        },
        "source_conflicts": conflicts_to_dicts(conflicts),
        "evidence_scores": evidence_score_rows,
        "filtered_hits": filtered_hits,
        "warnings": warnings,
    }


def run_post_generation_safety(
    query: str,
    answer: str,
    rag_result: dict[str, Any],
    precheck: dict[str, Any],
    imaging_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = perf_counter()
    filtered_hits = precheck.get("filtered_hits") or rag_result.get("retrieval", {}).get("hits", [])
    if imaging_result and imaging_result.get("status") == "Analyzed":
        imaging_evidence = _build_imaging_evidence_hit(imaging_result)
        filtered_hits = [imaging_evidence, *filtered_hits]
    claim_start = perf_counter()
    claims = _claim_verifier.extract_claims(answer)
    claim_extraction_duration_ms = round((perf_counter() - claim_start) * 1000.0, 2)
    nli_start = perf_counter()
    claim_results = _claim_verifier.verify_claims(claims, filtered_hits)
    nli_duration_ms = round((perf_counter() - nli_start) * 1000.0, 2)

    # Rehydrate lightweight dict data into the dataclasses expected by the fusion layer.
    evidence_scores = [EvidenceScore(**row) for row in precheck.get("evidence_scores", [])]
    conflicts = [SourceConflict(**row) for row in precheck.get("source_conflicts", [])]
    claim_rows = claim_results_to_dicts(claim_results)
    confidence = _confidence_fusion.fuse(
        rag_score=rag_result.get("rag_score"),
        evidence_scores=evidence_scores,
        conflicts=conflicts,
        claim_results=claim_results,
        imaging_result=imaging_result,
    )

    warnings = []
    if confidence.should_refuse:
        warnings.append("Clinical safety layer recommends refusing this answer.")
    elif not confidence.should_answer:
        warnings.append("Clinical safety layer found insufficient support for a confident answer.")

    log_event(
        "safety",
        "post_generation_completed",
        duration_ms=round((perf_counter() - start) * 1000.0, 2),
        claim_extraction_duration_ms=claim_extraction_duration_ms,
        nli_verification_duration_ms=nli_duration_ms,
        claims_count=len(claims),
        confidence_score=confidence.score,
        confidence_label=confidence.label,
        should_refuse=confidence.should_refuse,
    )

    return {
        "query": query,
        "claims": claims_to_dicts(claims),
        "claims_summary": _build_claims_summary(claim_rows),
        "claim_verification": claim_rows,
        "confidence": confidence.to_dict(),
        "warnings": warnings,
    }


def _build_imaging_evidence_hit(imaging_result: dict[str, Any]) -> dict[str, Any]:
    findings = imaging_result.get("findings") or []
    critical = imaging_result.get("critical") or []
    scores = imaging_result.get("percentage_scores") or {}
    assessments = [
        format_finding_assessment(name, float(value))
        for name, value in sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    ]
    score_text = ", ".join(
        f"{name} {value}%"
        for name, value in sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    )
    text_parts = []
    if assessments:
        text_parts.append(f"Radiology image analysis assessment: {' '.join(assessments[:4])}")
    elif findings:
        text_parts.append(f"Radiology image analysis detected findings: {', '.join(findings)}.")
    if critical:
        text_parts.append(f"Likely high-confidence radiology findings: {', '.join(critical)}.")
    if score_text:
        text_parts.append(f"Radiology model scores: {score_text}.")
    return {
        "citation_id": "IMG1",
        "chunk_id": "uploaded_image_analysis",
        "text": " ".join(text_parts) or "Radiology image analysis was performed.",
        "source": "uploaded image",
        "source_type": "radiology_model",
        "condition": ", ".join(critical or findings) or "imaging",
        "section": "image_analysis",
        "rank": 0,
        "normalized_score": 1.0,
    }


def build_safety_context_note(precheck: dict[str, Any]) -> str:
    conflicts = precheck.get("source_conflicts", [])
    evidence_scores = precheck.get("evidence_scores", [])
    rejected = [row for row in evidence_scores if not row.get("passed")]
    note_parts = []
    if conflicts:
        note_parts.append(
            "Retrieved source conflicts were detected; avoid resolving conflicts from memory."
        )
    if rejected:
        note_parts.append(
            "Some retrieved evidence was excluded for low clinical support."
        )
    if not note_parts:
        return ""
    return "Safety note: " + " ".join(note_parts)


def get_safety_pipeline_status() -> dict[str, Any]:
    return {
        "entity_extraction": "medical_corpus_and_regex",
        "source_conflict_detection": "rule_based",
        "evidence_scoring": "enabled",
        "claim_verification": get_claim_verification_status(),
    }
