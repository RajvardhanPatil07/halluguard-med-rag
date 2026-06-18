"""
Strict answer gating for HalluGuard-Med.

MedGemma may produce a candidate answer, but this module decides what text is
safe enough to expose as final_response.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


THRESHOLDS_VERSION = "strict-evidence-gate-v1"
BLOCKING_CLAIM_STATUSES = {"weak_support", "unsupported", "insufficient", "contradicted"}
REFUSAL_CLAIM_STATUSES = {"contradicted"}


def _normalize_citation(hit: dict[str, Any]) -> dict[str, Any]:
    citation_id = str(hit.get("id") or hit.get("citation_id") or hit.get("chunk_id") or "EVIDENCE")
    return {
        "id": citation_id,
        "text": str(hit.get("text") or "").strip(),
        "source": str(hit.get("source") or "retrieved evidence"),
        "condition": str(hit.get("condition") or "medical topic"),
        "section": str(hit.get("section") or "evidence"),
        "rank": hit.get("rank"),
    }


def _evidence_hits(rag_result: dict[str, Any]) -> list[dict[str, Any]]:
    hits = rag_result.get("safety_precheck", {}).get("filtered_hits") or []
    if not hits:
        hits = rag_result.get("citations") or []
    normalized = []
    seen = set()
    for hit in hits:
        citation = _normalize_citation(hit)
        if not citation["text"] or citation["id"] in seen:
            continue
        seen.add(citation["id"])
        normalized.append(citation)
    return normalized[:5]


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text


def _snippet(text: str, limit: int = 280) -> str:
    clean = _clean_text(text)
    if not clean:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    selected = ""
    for sentence in sentences:
        candidate = f"{selected} {sentence}".strip() if selected else sentence
        if len(candidate) > limit:
            break
        selected = candidate
        if len(selected) >= 140:
            break
    selected = selected or clean[:limit]
    if len(selected) > limit:
        selected = selected[: limit - 3].rstrip() + "..."
    return selected


def _claim_rows(safety_result: dict[str, Any]) -> list[dict[str, Any]]:
    return list(safety_result.get("claim_verification") or [])


def _claim_status_counts(claim_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in claim_rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _has_high_source_conflict(rag_result: dict[str, Any]) -> bool:
    conflicts = rag_result.get("safety_precheck", {}).get("source_conflicts") or []
    return any(str(conflict.get("severity") or "").lower() == "high" for conflict in conflicts)


def _has_inline_citation(text: str) -> bool:
    return bool(re.search(r"\[[A-Za-z0-9][A-Za-z0-9_.:-]*\]", text or ""))


def _policy_base(
    decision: str,
    final_response_source: str,
    raw_candidate_hidden: bool,
    claim_rows: list[dict[str, Any]],
    reasons: list[str],
) -> dict[str, Any]:
    counts = _claim_status_counts(claim_rows)
    blocked_count = sum(counts.get(status, 0) for status in BLOCKING_CLAIM_STATUSES)
    return {
        "decision": decision,
        "final_response_source": final_response_source,
        "raw_candidate_hidden": raw_candidate_hidden,
        "blocked_claims_count": blocked_count,
        "supported_claims_count": counts.get("supported", 0),
        "claim_status_counts": counts,
        "thresholds_version": THRESHOLDS_VERSION,
        "reasons": reasons,
    }


def _build_evidence_summary(query: str, rag_result: dict[str, Any], reason: str) -> str:
    hits = _evidence_hits(rag_result)
    lines = [
        "What retrieved evidence supports",
    ]
    if hits:
        for hit in hits[:4]:
            snippet = _snippet(hit["text"])
            if snippet:
                lines.append(f"- [{hit['id']}] {snippet}")
    else:
        lines.append("- The retrieval step did not return usable evidence for a grounded answer.")

    lines.extend([
        "",
        "What is not established",
        (
            "- The generated draft was hidden because the safety gate did not find enough "
            f"support for every clinical claim ({reason})."
        ),
        "- Do not treat this as a diagnosis, prescription, or complete management plan.",
        "",
        "When to seek care",
        (
            "- Seek urgent medical care for severe, sudden, worsening, or life-threatening "
            "symptoms, or if a clinician has advised emergency evaluation."
        ),
    ])
    return "\n".join(lines)


def _build_refusal(rag_result: dict[str, Any], reasons: list[str]) -> str:
    hits = _evidence_hits(rag_result)
    reason_text = "; ".join(reasons) if reasons else "safety verification did not pass"
    lines = [
        "I cannot provide a medical answer from the generated draft because the safety gate found a high-risk verification issue.",
        f"Reason: {reason_text}.",
        "",
        "Retrieved citations available for clinician review",
    ]
    if hits:
        for hit in hits[:4]:
            lines.append(f"- [{hit['id']}] {hit['source']} - {hit['condition']} / {hit['section']}")
    else:
        lines.append("- No usable retrieved citations were available.")
    lines.extend([
        "",
        "Please use clinician review or trusted medical guidance before acting on this topic.",
    ])
    return "\n".join(lines)


def enforce_answer_policy(
    *,
    query: str,
    candidate_answer: str,
    verification: dict[str, Any],
    rag_result: dict[str, Any],
    safety_result: dict[str, Any],
    imaging_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    claim_rows = _claim_rows(safety_result)
    confidence = safety_result.get("confidence") or {}
    nli = verification.get("nli") or {}
    nli_label = str(nli.get("label") or "")
    risk_tier = str(verification.get("risk_tier") or "")
    imaging_status = str(verification.get("imaging") or "")
    rag_verified = verification.get("rag_verified")
    statuses = {str(row.get("status") or "unknown") for row in claim_rows}

    refusal_reasons = []
    if statuses & REFUSAL_CLAIM_STATUSES:
        refusal_reasons.append("contradicted_clinical_claim")
    if confidence.get("should_refuse"):
        refusal_reasons.append("confidence_fusion_refusal")
    if _has_high_source_conflict(rag_result):
        refusal_reasons.append("high_severity_source_conflict")
    if risk_tier == "Tier 3":
        refusal_reasons.append("tier_3_risk")
    if imaging_status == "Mismatch":
        refusal_reasons.append("imaging_response_mismatch")
    if nli_label == "Contradicted":
        refusal_reasons.append("nli_contradiction")

    if refusal_reasons:
        policy = _policy_base(
            decision="refuse",
            final_response_source="safety_refusal",
            raw_candidate_hidden=True,
            claim_rows=claim_rows,
            reasons=sorted(set(refusal_reasons)),
        )
        return {
            "final_response": _build_refusal(rag_result, policy["reasons"]),
            "answer_policy": policy,
        }

    answer_reasons = []
    if rag_verified is not True:
        answer_reasons.append("rag_not_verified")
    if risk_tier != "Tier 1":
        answer_reasons.append("risk_tier_not_low")
    if nli_label != "Entailed":
        answer_reasons.append("nli_not_entailed")
    if confidence.get("should_answer") is not True:
        answer_reasons.append("confidence_fusion_did_not_allow_answer")
    if not claim_rows:
        answer_reasons.append("no_claims_verified")
    if statuses - {"supported"}:
        answer_reasons.append("not_all_claims_supported")
    if not _has_inline_citation(candidate_answer):
        answer_reasons.append("missing_inline_citations")

    if not answer_reasons:
        policy = _policy_base(
            decision="answer",
            final_response_source="medgemma_verified",
            raw_candidate_hidden=False,
            claim_rows=claim_rows,
            reasons=[],
        )
        return {
            "final_response": candidate_answer,
            "answer_policy": policy,
        }

    reason = ", ".join(answer_reasons)
    policy = _policy_base(
        decision="evidence_summary",
        final_response_source="extractive_evidence",
        raw_candidate_hidden=True,
        claim_rows=claim_rows,
        reasons=answer_reasons,
    )
    return {
        "final_response": _build_evidence_summary(query, rag_result, reason),
        "answer_policy": policy,
    }


def sanitize_hidden_candidate_safety(
    safety_result: dict[str, Any],
    answer_policy: dict[str, Any],
) -> dict[str, Any]:
    if not answer_policy.get("raw_candidate_hidden"):
        return safety_result

    sanitized = deepcopy(safety_result)
    sanitized["claims"] = [
        {
            "claim_id": row.get("claim_id"),
            "claim_type": row.get("claim_type"),
        }
        for row in sanitized.get("claims", [])
    ]
    sanitized["claim_verification"] = [
        {
            "claim_id": row.get("claim_id"),
            "status": row.get("status"),
            "support_score": row.get("support_score"),
            "contradiction_score": row.get("contradiction_score"),
            "best_citation_id": row.get("best_citation_id"),
            "reason": row.get("reason"),
        }
        for row in sanitized.get("claim_verification", [])
    ]
    sanitized["raw_candidate_hidden"] = True
    return sanitized


def sanitize_hidden_candidate_verification(
    verification: dict[str, Any],
    answer_policy: dict[str, Any],
) -> dict[str, Any]:
    if not answer_policy.get("raw_candidate_hidden"):
        return verification

    sanitized = deepcopy(verification)
    nli = dict(sanitized.get("nli") or {})
    nli["claims"] = [
        {
            "status": row.get("status"),
            "entailment": row.get("entailment"),
            "contradiction": row.get("contradiction"),
            "neutral": row.get("neutral"),
            "source": row.get("source"),
        }
        for row in nli.get("claims", [])
    ]
    sanitized["nli"] = nli
    sanitized["raw_candidate_hidden"] = True
    return sanitized
