"""
Medical entity extraction for HalluGuard-Med.

This module is intentionally lightweight: it uses the local MedlinePlus corpus
plus clinical regex patterns instead of another large model.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CORPUS_CHUNKS_PATH = Path(__file__).resolve().parents[1] / "medical_corpus" / "processed" / "corpus_chunks.jsonl"

ENTITY_DISEASE = "disease"
ENTITY_SYMPTOM = "symptom"
ENTITY_TREATMENT = "treatment"
ENTITY_TEST = "test"
ENTITY_DRUG = "drug"
ENTITY_DOSAGE = "dosage"
ENTITY_RISK_FACTOR = "risk_factor"
ENTITY_LAB_VALUE = "lab_value"
ENTITY_CONTRAINDICATION = "contraindication"

DRUG_TERMS = {
    "acetaminophen",
    "tylenol",
    "metformin",
    "insulin",
    "aspirin",
    "paracetamol",
    "heparin",
    "warfarin",
    "isoniazid",
    "rifampicin",
    "pyrazinamide",
    "ethambutol",
    "salbutamol",
    "montelukast",
    "digoxin",
    "nitroglycerin",
    "anti-d",
    "anti d",
    "rh immune globulin",
    "rho(d) immune globulin",
}

TEST_TERMS = {
    "x-ray",
    "xray",
    "ct",
    "mri",
    "ecg",
    "ekg",
    "troponin",
    "cd4",
    "blood pressure",
    "glucose",
    "hemoglobin",
    "haemoglobin",
}

RISK_FACTOR_TERMS = {
    "smoking",
    "pregnancy",
    "obesity",
    "hypertension",
    "diabetes",
    "hiv",
    "immunocompromised",
    "surgery",
    "prolonged immobility",
}

CONTRAINDICATION_PATTERNS = [
    r"\bcontraindicat(?:ed|ion|ions)?\b",
    r"\bavoid\b",
    r"\bnot recommended\b",
    r"\bshould not\b",
    r"\bunsafe\b",
]

DISEASE_ALIASES = {
    "chronic obstructive pulmonary disease": "copd",
    "chronic kidney disease": "chronic kidney disease",
    "ckd": "chronic kidney disease",
    "myocardial infarction": "heart attack",
    "mi": "heart attack",
    "cerebrovascular accident": "stroke",
    "cva": "stroke",
}

BROAD_TAG_TERMS = {
    "children",
    "children and teenagers",
    "diagnosis and therapy",
    "disorders",
    "female reproductive system",
    "genetics/birth defects",
    "health",
    "infections",
    "injuries and wounds",
    "men",
    "older adults",
    "other resources",
    "people",
    "pregnancy and reproduction",
    "prevention and risk factors",
    "seniors",
    "specific populations",
    "teenagers",
    "women",
}

DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?|iu|%)"
    r"(?:\s?(?:/|per)\s?(?:day|dose|kg|hour|hr))?\b",
    re.IGNORECASE,
)
LAB_RE = re.compile(
    r"\b(?:hb|hemoglobin|haemoglobin|glucose|troponin|cd4|bp|blood pressure|spo2|creatinine)"
    r"\s*(?:is|=|:)?\s*\d+(?:\.\d+)?(?:/\d+)?\s*(?:mg/dl|g/dl|mmhg|%|ng/l|cells/mm3)?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MedicalEntity:
    text: str
    label: str
    normalized: str
    source: str
    confidence: float
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _singular_variant(term: str) -> str | None:
    normalized = _normalize(term)
    if len(normalized) > 4 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if len(normalized) > 4 and normalized.endswith("s"):
        return normalized[:-1]
    return None


def _load_corpus_terms() -> dict[str, dict[str, set[str]]]:
    if not CORPUS_CHUNKS_PATH.exists():
        return {}
    terms_by_condition: dict[str, dict[str, set[str]]] = {}
    try:
        with CORPUS_CHUNKS_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                condition = _normalize(str(row.get("condition") or ""))
                if not condition:
                    continue
                bucket = terms_by_condition.setdefault(
                    condition,
                    {"condition": set(), "title": set(), "aliases": set(), "tags": set()},
                )
                bucket["condition"].add(condition)
                title = _normalize(str(row.get("title") or ""))
                if title:
                    bucket["title"].add(title)
                for alias in row.get("aliases") or []:
                    clean = _normalize(str(alias))
                    if clean:
                        bucket["aliases"].add(clean)
                for tag in row.get("tags") or []:
                    clean = _normalize(str(tag))
                    if clean and clean not in BROAD_TAG_TERMS and len(clean) >= 4:
                        bucket["tags"].add(clean)
    except Exception:
        return terms_by_condition
    return terms_by_condition


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


class MedicalEntityExtractor:
    def __init__(self, graph: dict[str, Any] | None = None) -> None:
        self.corpus_terms = _load_corpus_terms()
        self.lexicon = self._build_lexicon()
        self.lexicon_patterns = [
            (_term_pattern(term), term, label, normalized, confidence)
            for term, label, normalized, confidence in self.lexicon
        ]
        self._extract_cache: dict[tuple[str, str], list[MedicalEntity]] = {}

    def _build_lexicon(self) -> list[tuple[str, str, str, float]]:
        terms: dict[tuple[str, str], tuple[str, str, str, float]] = {}

        def add(term: str, label: str, normalized: str, confidence: float) -> None:
            clean = term.strip()
            if len(clean) < 2:
                return
            key = (_normalize(clean), label)
            terms[key] = (clean, label, _normalize(normalized), confidence)

        for condition_name, groups in self.corpus_terms.items():
            for term in groups["condition"]:
                add(term, ENTITY_DISEASE, condition_name, 0.92)
                singular = _singular_variant(term)
                if singular and singular != term:
                    add(singular, ENTITY_DISEASE, condition_name, 0.86)
            for term in groups["title"]:
                add(term, ENTITY_DISEASE, condition_name, 0.88)
                singular = _singular_variant(term)
                if singular and singular != term:
                    add(singular, ENTITY_DISEASE, condition_name, 0.84)
            for term in groups["aliases"]:
                add(term, ENTITY_DISEASE, condition_name, 0.86)
            for term in groups["tags"]:
                if term == condition_name or term in groups["condition"] or term in groups["title"] or term in groups["aliases"]:
                    continue
                label = ENTITY_DRUG if term in DRUG_TERMS else ENTITY_RISK_FACTOR
                add(term, label, term if label != ENTITY_RISK_FACTOR else term, 0.64)

        for drug in DRUG_TERMS:
            normalized = "paracetamol" if drug in {"acetaminophen", "tylenol"} else drug
            add(drug, ENTITY_DRUG, normalized, 0.88)
        for alias, normalized in DISEASE_ALIASES.items():
            add(alias, ENTITY_DISEASE, normalized, 0.90)
        for test in TEST_TERMS:
            add(test, ENTITY_TEST, test, 0.78)
        for risk in RISK_FACTOR_TERMS:
            add(risk, ENTITY_RISK_FACTOR, risk, 0.72)

        return sorted(terms.values(), key=lambda row: len(row[0]), reverse=True)

    def extract(self, text: str, source: str = "query") -> list[MedicalEntity]:
        if not text:
            return []
        cache_key = (source, text)
        cached = self._extract_cache.get(cache_key)
        if cached is not None:
            return cached

        entities: list[MedicalEntity] = []
        seen: set[tuple[int | None, int | None, str, str]] = set()

        for pattern, _term, label, normalized, confidence in self.lexicon_patterns:
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), label, normalized)
                if key in seen:
                    continue
                seen.add(key)
                entities.append(MedicalEntity(
                    text=match.group(0),
                    label=label,
                    normalized=normalized,
                    source=source,
                    confidence=confidence,
                    start=match.start(),
                    end=match.end(),
                ))

        for regex, label, confidence in (
            (DOSAGE_RE, ENTITY_DOSAGE, 0.90),
            (LAB_RE, ENTITY_LAB_VALUE, 0.86),
        ):
            for match in regex.finditer(text):
                normalized = _normalize(match.group(0))
                key = (match.start(), match.end(), label, normalized)
                if key in seen:
                    continue
                seen.add(key)
                entities.append(MedicalEntity(
                    text=match.group(0),
                    label=label,
                    normalized=normalized,
                    source=source,
                    confidence=confidence,
                    start=match.start(),
                    end=match.end(),
                ))

        for pattern in CONTRAINDICATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                normalized = _normalize(match.group(0))
                key = (match.start(), match.end(), ENTITY_CONTRAINDICATION, normalized)
                if key in seen:
                    continue
                seen.add(key)
                entities.append(MedicalEntity(
                    text=match.group(0),
                    label=ENTITY_CONTRAINDICATION,
                    normalized=normalized,
                    source=source,
                    confidence=0.84,
                    start=match.start(),
                    end=match.end(),
                ))

        result = _drop_nested_same_label_entities(entities)
        if len(self._extract_cache) > 512:
            self._extract_cache.clear()
        self._extract_cache[cache_key] = result
        return result

    def extract_from_hits(self, hits: list[dict[str, Any]]) -> dict[str, list[MedicalEntity]]:
        output: dict[str, list[MedicalEntity]] = {}
        for hit in hits:
            citation_id = str(hit.get("citation_id") or hit.get("chunk_id") or "")
            chunk_id = str(hit.get("chunk_id") or citation_id)
            entities = self.extract(hit.get("text", ""), source=chunk_id)
            output[chunk_id] = entities
            if citation_id and citation_id != chunk_id:
                output[citation_id] = entities
        return output

    def entity_overlap(
        self,
        query_entities: list[MedicalEntity],
        evidence_entities: list[MedicalEntity],
    ) -> float:
        query_set = {
            (entity.label, entity.normalized)
            for entity in query_entities
            if entity.label not in {ENTITY_DOSAGE, ENTITY_LAB_VALUE}
        }
        evidence_set = {
            (entity.label, entity.normalized)
            for entity in evidence_entities
            if entity.label not in {ENTITY_DOSAGE, ENTITY_LAB_VALUE}
        }
        if not query_set:
            return 0.0
        return round(len(query_set & evidence_set) / len(query_set), 4)


def entities_to_dicts(entities: list[MedicalEntity]) -> list[dict[str, Any]]:
    return [entity.to_dict() for entity in entities]


def _drop_nested_same_label_entities(entities: list[MedicalEntity]) -> list[MedicalEntity]:
    ordered = sorted(
        entities,
        key=lambda entity: (
            entity.start is None,
            entity.start or 0,
            -(entity.end or 0) + (entity.start or 0),
        ),
    )
    keep: list[MedicalEntity] = []
    for entity in ordered:
        if entity.start is None or entity.end is None:
            keep.append(entity)
            continue
        nested = False
        for existing in keep:
            if existing.label != entity.label or existing.start is None or existing.end is None:
                continue
            existing_len = existing.end - existing.start
            entity_len = entity.end - entity.start
            if existing.start <= entity.start and existing.end >= entity.end and existing_len > entity_len:
                nested = True
                break
        if not nested:
            keep.append(entity)
    return sorted(keep, key=lambda entity: (entity.start is None, entity.start or 0, -len(entity.text)))
