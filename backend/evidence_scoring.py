"""
Evidence quality scoring for retrieved medical chunks.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from .medical_entities import MedicalEntity, MedicalEntityExtractor
    from .source_conflicts import SourceConflict
except ImportError:
    from medical_entities import MedicalEntity, MedicalEntityExtractor
    from source_conflicts import SourceConflict


DEFAULT_EVIDENCE_THRESHOLD = 0.35

SOURCE_AUTHORITY = {
    "consumer_health": 0.86,
    "guideline": 0.95,
    "textbook": 0.82,
    "journal": 0.88,
    "public_health": 0.90,
    "corpus": 0.80,
}


def _normalized_disease_terms(entities: list[MedicalEntity]) -> set[str]:
    return {
        entity.normalized
        for entity in entities
        if entity.label == "disease"
    }


def _condition_aligned(hit: dict[str, Any], disease_terms: set[str]) -> bool | None:
    if not disease_terms:
        return None
    condition = str(hit.get("condition") or "").strip().lower()
    if not condition:
        return None
    return condition in disease_terms


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "if", "in", "is", "it", "its", "may", "of", "on", "or", "that", "the",
    "their", "this", "to", "used", "uses", "with", "your", "you", "can", "should",
    "what", "when", "where", "why", "medical", "patient",
    "warning", "sign", "signs", "seek", "care", "someone", "key", "overview",
}

CONCEPT_SYNONYMS = {
    "rupture": "burst",
    "ruptured": "burst",
    "bursts": "burst",
    "inflammation": "inflame",
    "inflamed": "inflame",
    "nauseous": "nausea",
    "sick": "nausea",
    "queasy": "nausea",
    "vomiting": "vomit",
    "vomited": "vomit",
    "throwing": "vomit",
    "emesis": "vomit",
    "haemorrhagic": "hemorrhagic",
    "thrombotic": "clot",
    "thrombus": "clot",
    "blocked": "block",
    "blockage": "block",
    "hyperglycemia": "glucose",
    "hypoglycemia": "glucose",
    "diplopia": "vision",
    "ptosis": "eyelid",
    "medication": "medicine",
    "medications": "medicine",
    "medicines": "medicine",
    "managed": "manage",
    "managing": "manage",
    "gradually": "time",
    "physical": "activity",
    "exercise": "activity",
}

PHRASE_SYNONYMS = {
    "blood clot": ["blood clot", "clot", "blocked artery", "blockage", "thrombus", "thrombotic"],
    "bleeding in brain": ["bleeding in the brain", "brain bleed", "hemorrhage in the brain", "ruptured blood vessel"],
    "high blood sugar": ["high blood sugar", "high blood glucose", "hyperglycemia", "elevated glucose"],
    "low blood sugar": ["low blood sugar", "low blood glucose", "hypoglycemia"],
    "insulin resistance": ["insulin resistance", "resistant to insulin", "body does not use insulin well"],
    "nausea and vomiting": ["nausea and vomiting", "nausea or vomiting", "feeling sick", "throwing up"],
    "lifestyle changes": ["lifestyle changes", "diet and exercise", "healthy eating and physical activity", "eating fewer calories", "physically active", "physical activity"],
    "diabetes medication": ["medication", "medications", "medicine", "medicines", "oral medicines", "insulin", "injectable medicines"],
    "develops over time": ["develops gradually", "gradually over time", "over time", "prevent or delay"],
}


def _content_terms(text: str) -> set[str]:
    text = re.sub(r"\b(?:acetaminophen|tylenol)\b", "paracetamol", str(text or ""), flags=re.IGNORECASE)
    terms = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) < 2 or token in STOPWORDS:
            continue
        token = CONCEPT_SYNONYMS.get(token, token)
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        terms.add(token)
    return terms


def _lexical_relevance(query_text: str, hit: dict[str, Any]) -> float:
    query_terms = _content_terms(query_text)
    if not query_terms:
        return 0.0
    haystack = " ".join([
        str(hit.get("condition") or ""),
        str(hit.get("section") or ""),
        str(hit.get("text") or ""),
    ])
    hit_terms = _content_terms(haystack)
    overlap = len(query_terms & hit_terms) / len(query_terms)
    condition_terms = _content_terms(str(hit.get("condition") or ""))
    condition_bonus = 0.0
    if condition_terms and query_terms & condition_terms:
        condition_bonus = 0.18
    return round(min(1.0, overlap + condition_bonus), 4)


def _semantic_relevance(query_text: str, hit: dict[str, Any]) -> float:
    query_terms = _content_terms(query_text)
    if not query_terms:
        return 0.0
    haystack = " ".join([
        str(hit.get("condition") or ""),
        str(hit.get("section") or ""),
        str(hit.get("text") or ""),
    ])
    hit_terms = _content_terms(haystack)
    coverage = len(query_terms & hit_terms) / len(query_terms)
    query_lower = query_text.lower()
    hit_lower = haystack.lower()
    phrase_matches = 0
    for variants in PHRASE_SYNONYMS.values():
        if any(variant in query_lower for variant in variants) and any(variant in hit_lower for variant in variants):
            phrase_matches += 1
    if phrase_matches:
        coverage = max(coverage, min(0.78, 0.50 + (0.14 * phrase_matches)))
    condition_terms = _content_terms(str(hit.get("condition") or ""))
    if condition_terms and query_terms & condition_terms:
        coverage = max(coverage, 0.62)
    return round(min(1.0, coverage), 4)


def _matched_query_terms(query_text: str, hit: dict[str, Any]) -> tuple[list[str], list[str]]:
    query_terms = _content_terms(query_text)
    haystack = " ".join([
        str(hit.get("condition") or ""),
        str(hit.get("section") or ""),
        str(hit.get("text") or ""),
    ])
    hit_terms = _content_terms(haystack)
    matched = sorted(query_terms & hit_terms)
    missing = sorted(query_terms - hit_terms)
    return matched, missing


@dataclass(frozen=True)
class EvidenceScore:
    citation_id: str
    chunk_id: str
    retrieval_score: float
    entity_overlap: float
    lexical_relevance: float
    source_authority: float
    recency_score: float
    contradiction_penalty: float
    final_score: float
    passed: bool
    reasons: list[str]
    matched_query_terms: list[str] = field(default_factory=list)
    missing_query_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceScorer:
    def __init__(
        self,
        extractor: MedicalEntityExtractor | None = None,
        threshold: float = DEFAULT_EVIDENCE_THRESHOLD,
    ) -> None:
        self.extractor = extractor or MedicalEntityExtractor()
        self.threshold = threshold

    def score_hits(
        self,
        hits: list[dict[str, Any]],
        query_entities: list[MedicalEntity],
        evidence_entities: dict[str, list[MedicalEntity]],
        conflicts: list[SourceConflict] | None = None,
        query_text: str = "",
    ) -> list[EvidenceScore]:
        conflict_penalties = self._build_conflict_penalties(conflicts or [])
        scores: list[EvidenceScore] = []

        for hit in hits:
            citation_id = str(hit.get("citation_id", ""))
            chunk_id = str(hit.get("chunk_id") or citation_id)
            retrieval_score = float(hit.get("normalized_score") or hit.get("score") or 0.0)
            source_type = str(hit.get("source_type") or "").lower()
            source_authority = SOURCE_AUTHORITY.get(source_type, 0.68)
            recency_score = 0.65
            entity_overlap = self.extractor.entity_overlap(
                query_entities,
                evidence_entities.get(chunk_id) or evidence_entities.get(citation_id, []),
            )
            lexical_relevance = _lexical_relevance(query_text, hit)
            semantic_relevance = _semantic_relevance(query_text, hit)
            matched_terms, missing_terms = _matched_query_terms(query_text, hit)
            contradiction_penalty = conflict_penalties.get(citation_id, 0.0)
            disease_terms = _normalized_disease_terms(query_entities)
            condition_alignment = _condition_aligned(hit, disease_terms)

            final_score = (
                (0.24 * retrieval_score)
                + (0.18 * entity_overlap)
                + (0.18 * lexical_relevance)
                + (0.16 * semantic_relevance)
                + (0.16 * source_authority)
                + (0.08 * recency_score)
                - (0.10 * contradiction_penalty)
            )
            if condition_alignment is True:
                final_score += 0.12
            elif condition_alignment is False:
                final_score *= 0.45
            if query_entities and entity_overlap <= 0.0:
                final_score *= 0.70 if semantic_relevance >= 0.55 else 0.55
            elif query_entities and entity_overlap < 0.5:
                final_score *= 0.88 if semantic_relevance >= 0.55 else 0.75
            if not query_entities:
                final_score *= 0.95
            final_score = round(max(0.0, min(final_score, 1.0)), 4)
            reasons = []
            if not query_entities:
                reasons.append("no_query_entities_detected")
            if entity_overlap < 0.25 and query_entities:
                reasons.append("low_entity_overlap")
            if contradiction_penalty > 0:
                reasons.append("source_conflict_penalty")
            if retrieval_score < 0.25:
                reasons.append("weak_retrieval_score")
            if lexical_relevance < 0.25:
                reasons.append("low_lexical_relevance")
            if semantic_relevance >= 0.55:
                reasons.append("semantic_relevance_match")
            if condition_alignment is True:
                reasons.append("condition_aligned")
            elif condition_alignment is False:
                reasons.append("condition_mismatch")

            scores.append(EvidenceScore(
                citation_id=citation_id,
                chunk_id=chunk_id,
                retrieval_score=round(retrieval_score, 4),
                entity_overlap=round(entity_overlap, 4),
                lexical_relevance=round(lexical_relevance, 4),
                source_authority=round(source_authority, 4),
                recency_score=round(recency_score, 4),
                contradiction_penalty=round(contradiction_penalty, 4),
                final_score=final_score,
                passed=final_score >= self.threshold,
                reasons=reasons,
                matched_query_terms=matched_terms[:12],
                missing_query_terms=missing_terms[:12],
            ))

        return scores

    def _build_conflict_penalties(self, conflicts: list[SourceConflict]) -> dict[str, float]:
        penalties: dict[str, float] = {}
        for conflict in conflicts:
            penalty = 1.0 if conflict.severity == "high" else 0.6
            for citation_id in (conflict.citation_a, conflict.citation_b):
                penalties[citation_id] = max(penalties.get(citation_id, 0.0), penalty)
        return penalties


def evidence_scores_to_dicts(scores: list[EvidenceScore]) -> list[dict[str, Any]]:
    return [score.to_dict() for score in scores]
