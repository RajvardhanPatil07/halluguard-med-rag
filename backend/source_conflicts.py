"""
Source-to-source medical conflict detection.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass
from typing import Any

try:
    from .medical_entities import MedicalEntity
except ImportError:
    from medical_entities import MedicalEntity


POSITIVE_PATTERNS = [
    r"\brecommended\b",
    r"\bindicated\b",
    r"\bstandard treatment\b",
    r"\bsafe\b",
    r"\bcan be used\b",
    r"\bshould be given\b",
]

NEGATIVE_PATTERNS = [
    r"\bcontraindicat(?:ed|ion|ions)?\b",
    r"\bavoid\b",
    r"\bnot recommended\b",
    r"\bshould not\b",
    r"\bunsafe\b",
    r"\bwithhold\b",
]

ABSOLUTE_PATTERNS = [
    r"\balways\b",
    r"\bnever\b",
    r"\bguaranteed\b",
    r"\b100%\b",
]


@dataclass(frozen=True)
class SourceConflict:
    conflict_id: str
    citation_a: str
    citation_b: str
    claim_a: str
    claim_b: str
    contradiction_score: float
    entities: list[str]
    severity: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _entity_keys(entities: list[MedicalEntity]) -> set[str]:
    return {
        entity.normalized
        for entity in entities
        if entity.label in {"disease", "drug", "treatment", "symptom", "risk_factor"}
    }


def _short(text: str, limit: int = 220) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


def _is_comparable_medical_claim(hit: dict[str, Any]) -> bool:
    section = str(hit.get("section") or "").lower()
    text = str(hit.get("text") or "").lower()
    if section in {"symptoms", "overview", "xray_findings"}:
        return False
    return any(
        word in text
        for word in (
            "treatment",
            "dose",
            "dosage",
            "recommended",
            "contraindicated",
            "avoid",
            "safe",
            "unsafe",
            "should not",
        )
    )


class SourceConflictDetector:
    def detect(
        self,
        hits: list[dict[str, Any]],
        evidence_entities: dict[str, list[MedicalEntity]],
        max_pairs: int = 12,
    ) -> list[SourceConflict]:
        conflicts: list[SourceConflict] = []
        top_hits = hits[:6]

        for index, (left, right) in enumerate(itertools.combinations(top_hits, 2), start=1):
            if index > max_pairs:
                break
            if not (_is_comparable_medical_claim(left) and _is_comparable_medical_claim(right)):
                continue
            citation_a = str(left.get("citation_id", ""))
            citation_b = str(right.get("citation_id", ""))
            left_text = left.get("text", "")
            right_text = right.get("text", "")
            left_entities = _entity_keys(evidence_entities.get(citation_a, []))
            right_entities = _entity_keys(evidence_entities.get(citation_b, []))
            shared = sorted(left_entities & right_entities)
            if not shared:
                continue

            left_positive = _has_any(left_text, POSITIVE_PATTERNS)
            right_positive = _has_any(right_text, POSITIVE_PATTERNS)
            left_negative = _has_any(left_text, NEGATIVE_PATTERNS)
            right_negative = _has_any(right_text, NEGATIVE_PATTERNS)
            left_absolute = _has_any(left_text, ABSOLUTE_PATTERNS)
            right_absolute = _has_any(right_text, ABSOLUTE_PATTERNS)

            score = 0.0
            reason = ""
            if (left_positive and right_negative) or (left_negative and right_positive):
                score = 0.78
                reason = "recommendation_vs_contraindication"
            elif left_absolute != right_absolute and (left_positive or right_positive):
                score = 0.55
                reason = "absolute_claim_mismatch"

            if score <= 0:
                continue

            severity = "high" if score >= 0.75 else "medium"
            conflicts.append(SourceConflict(
                conflict_id=f"SC{len(conflicts) + 1}",
                citation_a=citation_a,
                citation_b=citation_b,
                claim_a=_short(left_text),
                claim_b=_short(right_text),
                contradiction_score=round(score, 4),
                entities=shared[:8],
                severity=severity,
                reason=reason,
            ))

        return conflicts


def conflicts_to_dicts(conflicts: list[SourceConflict]) -> list[dict[str, Any]]:
    return [conflict.to_dict() for conflict in conflicts]
