"""
Verification pipeline for HalluGuard-Med.

Layers:
1. Corpus condition matching
2. Hybrid RAG evidence verification
3. NLI consistency check
4. Imaging consistency check
"""

import re
from time import perf_counter
from typing import Any

try:
    from .corpus_schema import DEFAULT_CHUNKS_PATH, read_jsonl
    from .clinical_claims import SUPPORTED_THRESHOLD, WEAK_SUPPORT_THRESHOLD, classify_support_status
    from .nli_verifier import NLI_CONTRADICTION_THRESHOLD, NLI_SUPPORT_THRESHOLD, get_nli_status
    from .radiology_analyzer import format_finding_assessment
    from .structured_log import log_event
except ImportError:
    from corpus_schema import DEFAULT_CHUNKS_PATH, read_jsonl
    from clinical_claims import SUPPORTED_THRESHOLD, WEAK_SUPPORT_THRESHOLD, classify_support_status
    from nli_verifier import NLI_CONTRADICTION_THRESHOLD, NLI_SUPPORT_THRESHOLD, get_nli_status
    from radiology_analyzer import format_finding_assessment
    from structured_log import log_event

_corpus_condition_cache = None
NLI_ENGINE = "deberta_v3_mnli_with_local_fallback"

NLI_LABEL_CONTRADICTION = 0
NLI_LABEL_ENTAILMENT = 1
NLI_LABEL_NEUTRAL = 2
NLI_ENTAILMENT_THRESHOLD = NLI_SUPPORT_THRESHOLD
NLI_NEUTRAL_THRESHOLD = 0.45
NLI_CONTRADICTION_MARGIN = 0.10
MAX_NLI_CLAIMS = 8
MAX_NLI_PREMISES_PER_CLAIM = 5
MAX_NLI_PREMISE_CHARS = 520

UNSAFE_PHRASES = [
    "guaranteed to cure",
    "always works",
    "never fails",
    "definitely cured",
    "no need to consult",
    "stop taking your medication",
    "cure immediately",
    "100% effective",
    "no side effects whatsoever",
    "safe for everyone without exception",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "if", "in", "is", "it", "its", "may", "of", "on", "or", "that", "the",
    "their", "this", "to", "used", "uses", "with", "your", "you",
    "given", "during", "typically", "usually", "considered", "requiring", "standard",
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
    "nausea and vomiting": ["nausea and vomiting", "nausea or vomiting", "feeling sick and throwing up", "throwing up"],
    "loss of appetite": ["loss of appetite", "lost appetite", "decreased appetite", "poor appetite"],
    "burst appendix": ["burst appendix", "appendix can burst", "ruptured appendix", "appendiceal rupture"],
    "lifestyle changes": ["lifestyle changes", "diet and exercise", "healthy eating and physical activity", "eating fewer calories", "physically active", "physical activity"],
    "diabetes medication": ["medication", "medications", "medicine", "medicines", "oral medicines", "insulin", "injectable medicines"],
    "develops over time": ["develops gradually", "gradually over time", "over time", "prevent or delay"],
}

DOMAIN_PHRASES = (
    "a1c",
    "appendicitis",
    "appendix",
    "blockage",
    "medical emergency",
    "removing the appendix",
    "removal of the appendix",
    "appendectomy",
    "treatment",
    "anti-d",
    "anti d",
    "rh immune globulin",
    "rh incompatibility",
    "rh-negative",
    "rh negative",
    "rh-positive",
    "rh positive",
    "blood glucose",
    "blood sugar",
    "prediabetes",
    "diabetes",
    "pregnancy",
    "injection",
)


def _contains_any(text: str, values: list[str]) -> bool:
    for value in values:
        if _term_matches(text, value):
            return True
    return False


def _singular_variant(term: str) -> str | None:
    normalized = _normalize_term(term)
    if len(normalized) > 4 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if len(normalized) > 4 and normalized.endswith("s"):
        return normalized[:-1]
    return None


def _term_variants(term: str) -> list[str]:
    normalized = _normalize_term(term)
    if not normalized:
        return []
    variants = [normalized]
    singular = _singular_variant(normalized)
    if singular and singular != normalized:
        variants.append(singular)
    return variants


def _term_matches(text: str, term: str) -> bool:
    lowered = text.lower()
    for variant in _term_variants(term):
        escaped = re.escape(variant).replace(r"\ ", r"\s+")
        if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lowered):
            return True
    return False


def _condition_match_score(condition: dict[str, Any], text: str, include_tags: bool = False) -> float:
    score = 0.0
    if _term_matches(text, condition.get("name", "")):
        score = max(score, 1.0)
    for alias in condition.get("aliases", []):
        if not _term_matches(text, alias):
            continue
        token_count = len(re.findall(r"[a-z0-9]+", alias.lower()))
        if token_count >= 2:
            score = max(score, 0.86)
        elif len(alias) >= 8:
            score = max(score, 0.70)
        else:
            score = max(score, 0.35)
    if include_tags:
        for tag in condition.get("tags", []):
            if _term_matches(text, tag):
                score = max(score, 0.45)
    return score


def _condition_matches_text(condition: dict[str, Any], text: str, include_tags: bool = False) -> bool:
    match_requires = condition.get("match_requires", [])
    if match_requires:
        return _contains_any(text, match_requires)
    return _condition_match_score(condition, text, include_tags) > 0.0


def _normalize_term(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _load_corpus_conditions() -> list[dict[str, Any]]:
    global _corpus_condition_cache
    if _corpus_condition_cache is not None:
        return _corpus_condition_cache

    grouped: dict[str, dict[str, Any]] = {}
    for chunk in read_jsonl(DEFAULT_CHUNKS_PATH):
        name = _normalize_term(chunk.condition)
        if not name:
            continue
        row = grouped.setdefault(
            name,
            {
                "name": name,
                "title": chunk.title or chunk.condition,
                "aliases": set(),
                "tags": set(),
                "source": "medical_corpus/processed/corpus_chunks.jsonl",
                "source_type": "corpus",
                "sections": set(),
            },
        )
        if chunk.title:
            row["aliases"].add(_normalize_term(chunk.title))
        for alias in chunk.aliases:
            clean = _normalize_term(alias)
            if clean:
                row["aliases"].add(clean)
        for tag in chunk.tags:
            clean = _normalize_term(tag)
            if clean:
                row["tags"].add(clean)
        if chunk.section:
            row["sections"].add(_normalize_term(chunk.section))

    _corpus_condition_cache = [
        {
            **row,
            "aliases": sorted(row["aliases"] - {name}),
            "tags": sorted(row["tags"]),
            "sections": sorted(row["sections"]),
        }
        for name, row in sorted(grouped.items())
    ]
    return _corpus_condition_cache


def _split_response_claims(response: str) -> list[str]:
    response = re.split(
        r"\n\s*(?:\*\*)?citations?(?:\*\*)?\s*(?:\n|$)",
        response,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    response = re.sub(
        r"(?im)^\s*\*\*(overview|key points|what the evidence supports|safety notes(?: or when to seek medical care)?)\*\*\s*$",
        "",
        response,
    )
    lines = []
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line or re.match(r"^\s*#{1,6}\s+", line):
            continue
        if re.match(r"^\s*(overview|key points?|warning signs?|symptoms?|diagnosis|treatment|emergency care)\s*:?\s*$", line, re.IGNORECASE):
            continue
        line = re.sub(r"^[\s*\-•\d.)]+", "", line).strip()
        if line:
            lines.append(line)
    text = " ".join(line for line in lines if line)
    candidates = re.split(r"(?<=[.!?])\s+", text)
    claims = []
    for candidate in candidates:
        claim = candidate.strip()
        if len(claim) < 25:
            continue
        if claim.lower().startswith(("consult ", "seek ", "talk to ", "please ")):
            continue
        claims.append(claim)
    return claims[:MAX_NLI_CLAIMS]


def _content_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) < 2 or token in STOPWORDS:
            continue
        token = CONCEPT_SYNONYMS.get(token, token)
        if token == "appendicitis":
            token = "appendiciti"
        elif token in {"caused", "causes", "causing"}:
            token = "cause"
        elif token in {"preventing", "prevents", "prevented"}:
            token = "prevent"
        elif token in {"draining", "drains", "drained"}:
            token = "drain"
        elif token in {"removing", "removed", "removal"}:
            token = "remove"
        elif token in {"infected", "infection", "infections"}:
            token = "infect"
        elif token in {"inflamed", "inflammation"}:
            token = "inflame"
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        terms.add(token)
    return terms


def _lexical_support_score(claim: str, premise: str) -> float:
    claim_terms = _content_terms(claim)
    if not claim_terms:
        return 0.0
    premise_terms = _content_terms(premise)
    overlap = len(claim_terms & premise_terms) / len(claim_terms)
    phrase_bonus = 0.0
    for phrase in DOMAIN_PHRASES:
        if phrase in claim.lower() and phrase in premise.lower():
            phrase_bonus += 0.08
    explicit = 0.0
    claim_lower = claim.lower()
    premise_lower = premise.lower()
    for variants in PHRASE_SYNONYMS.values():
        if any(variant in claim_lower for variant in variants) and any(variant in premise_lower for variant in variants):
            explicit = max(explicit, 0.92)
    return round(min(1.0, max(overlap + min(0.24, phrase_bonus), explicit)), 4)


def _split_premise_windows(text: str, max_chars: int = MAX_NLI_PREMISE_CHARS) -> list[str]:
    sentences = _split_response_claims(text)
    if not sentences:
        clean = re.sub(r"\s+", " ", text).strip()
        return [clean] if clean else []
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            windows.append(current)
        current = sentence
    if current:
        windows.append(current)
    return windows


def _select_premises_for_claim(claim: str, premises: list[str]) -> list[str]:
    windows: list[str] = []
    for premise in premises:
        windows.extend(_split_premise_windows(premise) or [premise])
    ranked = sorted(
        [(window, _lexical_support_score(claim, window)) for window in windows if window],
        key=lambda item: item[1],
        reverse=True,
    )
    selected = [window for window, score in ranked if score > 0][:MAX_NLI_PREMISES_PER_CLAIM]
    return selected or windows[:MAX_NLI_PREMISES_PER_CLAIM]


def _is_low_value_image_claim(claim: str) -> bool:
    patterns = [
        r"\bclear visualization of the lungs\b",
        r"\bclear visualization of .*mediastinum\b",
        r"\bdoes not show any other specific findings\b",
        r"\bwould suggest other diagnoses\b",
    ]
    return any(re.search(pattern, claim, re.IGNORECASE) for pattern in patterns)


def _run_corpus_condition_verification(
    query: str,
    response: str,
    corpus_conditions: list[dict[str, Any]],
) -> tuple[str, list[str], list[dict]]:
    warnings: list[str] = []
    query_matched_conditions: list[tuple[float, dict[str, Any]]] = []
    response_matched_conditions: list[tuple[float, dict[str, Any]]] = []

    for condition in corpus_conditions:
        query_score = _condition_match_score(condition, query)
        if query_score > 0:
            query_matched_conditions.append((query_score, condition))
            continue
        response_score = _condition_match_score(condition, response)
        if response_score > 0:
            response_matched_conditions.append((response_score, condition))

    # Query-time condition matches are authoritative. Falling back to response text
    # is useful for vague queries, but mixing both can leak unrelated conditions
    # into NLI premises when citations mention another condition.
    scored_matches = query_matched_conditions or response_matched_conditions
    scored_matches.sort(key=lambda item: (item[0], len(item[1].get("name", ""))), reverse=True)
    if scored_matches:
        best_score = scored_matches[0][0]
        matched_conditions = [
            condition
            for score, condition in scored_matches
            if score >= max(0.70, best_score - 0.15)
        ][:3]
    else:
        matched_conditions = []

    kg_status = "Match" if matched_conditions else "Neutral"
    for condition in matched_conditions:
        condition_name = condition.get("name", "")
        if condition_name and condition_name in query.lower() and condition_name not in response.lower():
            warnings.append(
                f"Corpus matched '{condition_name}' but the response did not clearly name it."
            )

    return kg_status, warnings, matched_conditions


def _run_rag_verification(rag_result: dict[str, Any] | None) -> dict[str, Any]:
    if not rag_result:
        return {
            "rag_score": None,
            "rag_verified": None,
            "rag_context": [],
            "citations": [],
            "error": "Hybrid RAG result was not supplied to verification",
        }

    retrieval = rag_result.get("retrieval", {})
    if not retrieval.get("ok", False):
        return {
            "rag_score": None,
            "rag_verified": None,
            "rag_context": [],
            "citations": [],
            "error": retrieval.get("error") or "Hybrid RAG retrieval failed",
        }

    citations = rag_result.get("citations", [])
    return {
        "rag_score": rag_result.get("rag_score"),
        "rag_verified": rag_result.get("rag_verified"),
        "rag_context": [citation.get("text", "") for citation in citations],
        "citations": citations,
        "error": None,
    }


def _run_nli_verification(
    response: str,
    matched_conditions: list[dict],
    rag_context: list[str],
    imaging_result: dict[str, Any] | None = None,
) -> tuple[str, float | None, list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    if _contains_any(response, UNSAFE_PHRASES):
        warnings.append("Unsafe absolute medical claim detected in response.")
        return "Contradicted", 1.0, warnings, []

    premises = [chunk for chunk in rag_context if chunk]
    if imaging_result and imaging_result.get("status") == "Analyzed":
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
        imaging_parts = []
        if assessments:
            imaging_parts.append(f"Uploaded chest X-ray image analysis assessment: {' '.join(assessments[:4])}")
        elif findings:
            imaging_parts.append(f"Uploaded chest X-ray image analysis detected findings: {', '.join(findings)}.")
        if critical:
            imaging_parts.append(f"Likely high-confidence imaging findings: {', '.join(critical)}.")
        if score_text:
            imaging_parts.append(f"Radiology model scores: {score_text}.")
        if imaging_parts:
            premises.insert(0, " ".join(imaging_parts))
    for condition in matched_conditions:
        symptoms = condition.get("symptoms", [])
        treatments = condition.get("treatments", [])
        name = condition.get("name", "")
        if symptoms:
            premises.append(
                f"{name} is associated with symptoms including: {', '.join(symptoms[:4])}."
            )
        if treatments:
            premises.append(
                f"Standard treatments for {name} include: {', '.join(treatments[:3])}."
            )

    claims = _split_response_claims(response)
    claims = [
        claim for claim in claims
        if not _is_low_value_image_claim(claim)
    ]
    if not claims and response.strip():
        claims = [response.strip()]

    if not premises:
        warnings.append("Local verification could not run because no evidence premises were available.")
        return "Insufficient Evidence", None, warnings, []

    claim_results: list[dict[str, Any]] = []
    for claim in claims:
        claim_premises = _select_premises_for_claim(claim, premises)
        scored = [
            (premise, _lexical_support_score(claim, premise))
            for premise in claim_premises
        ]
        best_premise, support = max(scored, key=lambda item: item[1], default=("", 0.0))
        neutral = round(max(0.0, 1.0 - support), 4)
        claim_results.append({
            "claim": claim,
            "status": classify_support_status(float(support)),
            "entailment": round(float(support), 4),
            "contradiction": 0.0,
            "neutral": neutral,
            "best_premise": best_premise,
            "lexical_support": round(float(support), 4),
            "source": NLI_ENGINE,
        })

    avg_support = sum(item["entailment"] for item in claim_results) / len(claim_results)
    unsupported = [
        item for item in claim_results
        if item["entailment"] < WEAK_SUPPORT_THRESHOLD
    ]
    weak_supported = [
        item for item in claim_results
        if WEAK_SUPPORT_THRESHOLD <= item["entailment"] < SUPPORTED_THRESHOLD
    ]
    if unsupported:
        warnings.append(
            f"Local verification found {len(unsupported)} unsupported claim(s)."
        )
    if weak_supported:
        warnings.append(
            f"Local verification found {len(weak_supported)} claim(s) with weak evidence support."
        )
    if avg_support >= SUPPORTED_THRESHOLD and not unsupported and not weak_supported:
        return "Entailed", round(avg_support, 4), warnings, claim_results
    return "Neutral", round(avg_support, 4), warnings, claim_results


def _run_nli_from_claim_verification(
    safety_result: dict[str, Any] | None,
) -> tuple[str, float | None, list[str], list[dict[str, Any]]] | None:
    if not safety_result:
        return None
    claim_results = safety_result.get("claim_verification") or []
    if not claim_results:
        return None

    warnings: list[str] = []
    contradictions = [
        item for item in claim_results
        if item.get("status") == "contradicted"
        and float(item.get("contradiction_score") or 0.0) >= NLI_CONTRADICTION_THRESHOLD
        and float(item.get("contradiction_score") or 0.0) > float(item.get("support_score") or 0.0) + NLI_CONTRADICTION_MARGIN
    ]
    supported = [item for item in claim_results if item.get("status") == "supported"]
    weak_supported = [item for item in claim_results if item.get("status") == "weak_support"]
    weak = [
        item for item in claim_results
        if item.get("status") in {"unsupported", "insufficient"}
    ]

    nli_claims = [
        {
            "claim": item.get("claim"),
            "entailment": round(float(item.get("support_score") or 0.0), 4),
            "contradiction": round(float(item.get("contradiction_score") or 0.0), 4),
            "neutral": round(max(0.0, 1.0 - max(
                float(item.get("support_score") or 0.0),
                float(item.get("contradiction_score") or 0.0),
            )), 4),
            "best_premise": item.get("best_evidence"),
            "lexical_support": None,
            "source": "post_generation_claim_verification",
            "nli": item.get("nli"),
        }
        for item in claim_results
    ]

    if contradictions:
        worst = max(contradictions, key=lambda item: float(item.get("contradiction_score") or 0.0))
        warnings.append("NLI reused claim verification and found contradicted clinical claim(s).")
        return "Contradicted", round(float(worst.get("contradiction_score") or 0.0), 4), warnings, nli_claims

    support_ratio = (len(supported) + (0.5 * len(weak_supported))) / len(claim_results)
    mean_support = sum(float(item.get("support_score") or 0.0) for item in claim_results) / len(claim_results)
    if weak:
        warnings.append(f"Claim verification found {len(weak)} unsupported claim(s).")
    if weak_supported:
        warnings.append(f"Claim verification found {len(weak_supported)} weakly supported claim(s).")
    if support_ratio >= 0.80 and mean_support >= SUPPORTED_THRESHOLD and not weak and not weak_supported:
        return "Entailed", round(mean_support, 4), warnings, nli_claims
    return "Neutral", round(mean_support, 4), warnings, nli_claims


def _aggregate_nli_metadata(nli_claims: list[dict[str, Any]]) -> dict[str, Any]:
    enabled_claims = [
        item.get("nli") or {}
        for item in nli_claims
        if (item.get("nli") or {}).get("enabled")
    ]
    status = get_nli_status()
    if not enabled_claims:
        return {
            "enabled": False,
            "model": status.get("model"),
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 0.0,
        }
    return {
        "enabled": True,
        "model": enabled_claims[0].get("model") or status.get("model"),
        "entailment": round(sum(float(item.get("entailment") or 0.0) for item in enabled_claims) / len(enabled_claims), 4),
        "contradiction": round(max(float(item.get("contradiction") or 0.0) for item in enabled_claims), 4),
        "neutral": round(sum(float(item.get("neutral") or 0.0) for item in enabled_claims) / len(enabled_claims), 4),
    }


def _run_imaging_verification(
    response: str,
    imaging_result: dict[str, Any] | None,
    matched_conditions: list[dict],
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not imaging_result:
        return "N/A", warnings

    warnings.extend(imaging_result.get("warnings", []))
    status = imaging_result.get("status", "Neutral")
    if status in ("Error", "Neutral"):
        return status, warnings

    graph_terms = imaging_result.get("graph_terms", [])
    findings = imaging_result.get("findings", [])
    if not findings:
        return "Neutral", warnings

    response_lower = response.lower()
    finding_terms = [str(finding).lower() for finding in findings]
    mentioned_findings = [
        term for term in [*graph_terms, *finding_terms]
        if term and term.lower() in response_lower
    ]
    expected_xray_terms = []
    for condition in matched_conditions:
        expected_xray_terms.extend(condition.get("xray_findings", []))

    expected_lower = [term.lower() for term in expected_xray_terms]
    graph_match = any(
        term and term.lower() in expected_lower
        for term in graph_terms
    )

    if mentioned_findings or graph_match:
        return "Match", warnings

    scores = imaging_result.get("percentage_scores") or {}
    assessments = [
        format_finding_assessment(name, float(scores.get(name)))
        for name in findings[:4]
        if scores.get(name) is not None
    ]
    finding_text = " ".join(assessments) if assessments else f"Radiology detected: {', '.join(findings[:4])}."
    warnings.append(
        f"{finding_text} "
        "These findings were not reflected in the AI response. "
        "Radiologist review recommended."
    )
    return "Mismatch", warnings


def _calculate_risk_tier(
    kg_status: str,
    nli_status: str,
    imaging_status: str,
    rag_verified: bool | None,
    warnings: list[str],
    rag_score: float | None = None,
    nli_confidence: float | None = None,
    imaging_result: dict[str, Any] | None = None,
) -> str:
    return _calculate_risk_assessment(
        kg_status,
        nli_status,
        imaging_status,
        rag_verified,
        warnings,
        rag_score,
        nli_confidence,
        imaging_result,
    )["tier"]


def _calculate_risk_assessment(
    kg_status: str,
    nli_status: str,
    imaging_status: str,
    rag_verified: bool | None,
    warnings: list[str],
    rag_score: float | None,
    nli_confidence: float | None,
    imaging_result: dict[str, Any] | None,
    safety_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []
    claim_results = (safety_result or {}).get("claim_verification") or []
    confidence = ((safety_result or {}).get("confidence") or {}).get("score")
    confidence_signal = float(confidence) if confidence is not None else None
    if claim_results:
        support_values = [float(item.get("support_score") or 0.0) for item in claim_results]
        contradiction_values = [float(item.get("contradiction_score") or 0.0) for item in claim_results]
        supported_count = len([item for item in claim_results if item.get("status") == "supported"])
        weak_support_count = len([item for item in claim_results if item.get("status") == "weak_support"])
        support_ratio = (supported_count + (0.5 * weak_support_count)) / len(claim_results)
        mean_support = sum(support_values) / len(support_values)
        max_contradiction = max(contradiction_values or [0.0])
    else:
        support_ratio = 0.0
        mean_support = float(nli_confidence or 0.0) if nli_status == "Entailed" else 0.0
        max_contradiction = float(nli_confidence or 0.0) if nli_status == "Contradicted" else 0.0

    if nli_status == "Contradicted":
        weight = 2.4 + (1.8 * max(float(nli_confidence or 0.0), max_contradiction))
        score += weight
        reasons.append("NLI contradiction detected")
    elif nli_status == "Error":
        score += 2.0
        reasons.append("NLI verification failed")
    elif nli_status in {"Neutral", "Insufficient Evidence"}:
        score += 0.35 + (1.15 * (1.0 - max(mean_support, support_ratio)))
        reasons.append("NLI support is not conclusive")

    if imaging_status == "Mismatch":
        score += 3.0
        reasons.append("Imaging findings conflict with the response")
    elif imaging_status == "Error":
        score += 1.0
        reasons.append("Imaging analysis failed")

    critical_findings = []
    if imaging_result:
        critical_findings = imaging_result.get("critical") or []
    if critical_findings:
        score += 1.0
        reasons.append("Critical imaging finding detected")

    if rag_verified is False:
        score += 1.5
        reasons.append("Hybrid RAG evidence score is below verification threshold")
    elif rag_verified is None:
        score += 2.0
        reasons.append("Hybrid RAG verification is unavailable")

    if rag_score is not None:
        rag_gap = max(0.0, 0.55 - float(rag_score))
        score += min(1.6, rag_gap * 2.4)
        if rag_score < 0.25:
            reasons.append("Retrieved evidence is weak")
        elif rag_score < 0.40:
            reasons.append("Retrieved evidence is borderline")
    if claim_results:
        unsupported_ratio = len([
            item for item in claim_results
            if item.get("status") in {"unsupported", "insufficient"}
        ]) / len(claim_results)
        weak_support_ratio = len([
            item for item in claim_results
            if item.get("status") == "weak_support"
        ]) / len(claim_results)
        score += 1.45 * unsupported_ratio
        score += 0.55 * weak_support_ratio
        if unsupported_ratio:
            reasons.append("Some generated claims are unsupported")
        if weak_support_ratio:
            reasons.append("Some generated claims have weak support")
        score += 1.8 * max_contradiction if max_contradiction >= 0.55 else 0.35 * max_contradiction
        if max_contradiction >= 0.55:
            reasons.append("Claim contradiction signal is elevated")
    if confidence_signal is not None:
        score += max(0.0, 0.70 - confidence_signal) * 1.6
        if confidence_signal < 0.45:
            reasons.append("Clinical confidence score is low")

    if kg_status == "Neutral":
        score += 0.5
        reasons.append("Corpus condition matcher did not match a condition")

    if warnings:
        score += min(1.2, 0.18 * len(warnings))
        reasons.append("Verification warnings were generated")

    if score >= 4:
        tier = "Tier 3"
    elif score >= 1:
        tier = "Tier 2"
    else:
        tier = "Tier 1"

    return {
        "tier": tier,
        "score": round(score, 2),
        "reasons": reasons,
    }


def _build_final_assessment(
    risk_assessment: dict[str, Any],
    nli_label: str,
    rag_verified: bool | None,
    safety_result: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    confidence = (safety_result or {}).get("confidence") or {}
    claims_summary = (safety_result or {}).get("claims_summary") or {}
    tier = risk_assessment.get("tier")
    if tier == "Tier 3":
        verdict = "High review priority"
        recommendation = "Do not rely on this response without clinician review."
    elif tier == "Tier 2":
        verdict = "Needs cross-check"
        recommendation = "Use cautiously and verify unsupported or weakly matched claims."
    else:
        verdict = "Low detected verification risk"
        recommendation = "Response is broadly supported by retrieved evidence, but remains decision support only."
    if rag_verified is False:
        recommendation = "Retrieved evidence is weak or insufficient; cross-check before use."
    if claims_summary.get("unsupported") or claims_summary.get("insufficient") or claims_summary.get("weak_support"):
        recommendation = "Some claims lack strong evidence match; review claim details and citations."
    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "risk_tier": tier,
        "risk_score": risk_assessment.get("score"),
        "confidence_label": confidence.get("label"),
        "confidence_score": confidence.get("score"),
        "nli_label": nli_label,
        "rag_verified": rag_verified,
        "warnings_count": len(warnings),
    }


def verify_response(
    query: str,
    response: str,
    imaging_result: dict[str, Any] | None = None,
    rag_result: dict[str, Any] | None = None,
    safety_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = perf_counter()
    corpus_conditions = _load_corpus_conditions()
    all_warnings: list[str] = []

    suggestions: list[str] = [
        "Use this output as decision support only, not as a final diagnosis.",
        "Verify clinically important claims against current evidence-based guidelines.",
        "Consult a licensed physician before making any medical decisions.",
    ]

    kg_status, kg_warnings, matched_conditions = _run_corpus_condition_verification(
        query, response, corpus_conditions
    )
    all_warnings.extend(kg_warnings)

    rag_verification = _run_rag_verification(rag_result)
    rag_score = rag_verification["rag_score"]
    rag_verified = rag_verification["rag_verified"]
    rag_context = rag_verification["rag_context"]
    citations = rag_verification["citations"]
    rag_error = rag_verification["error"]
    if rag_error:
        all_warnings.append(f"RAG: {rag_error}")
    all_warnings.extend(
        rag_result.get("safety_precheck", {}).get("warnings", [])
        if rag_result else []
    )

    nli_start = perf_counter()
    reused_nli = _run_nli_from_claim_verification(safety_result)
    if reused_nli is not None:
        nli_label, nli_confidence, nli_warnings, nli_claims = reused_nli
        nli_source = "post_generation_claim_verification"
    else:
        nli_label, nli_confidence, nli_warnings, nli_claims = _run_nli_verification(
            response, matched_conditions, rag_context, imaging_result
        )
        nli_source = NLI_ENGINE
    nli_duration_ms = round((perf_counter() - nli_start) * 1000.0, 2)
    all_warnings.extend(nli_warnings)

    all_warnings.extend((safety_result or {}).get("warnings", []))

    imaging_status, imaging_warnings = _run_imaging_verification(
        response, imaging_result, matched_conditions
    )
    all_warnings.extend(imaging_warnings)

    risk_assessment = _calculate_risk_assessment(
        kg_status,
        nli_label,
        imaging_status,
        rag_verified,
        all_warnings,
        rag_score,
        nli_confidence,
        imaging_result,
        safety_result,
    )
    risk_tier = risk_assessment["tier"]
    final_assessment = _build_final_assessment(
        risk_assessment,
        nli_label,
        rag_verified,
        safety_result,
        all_warnings,
    )
    if risk_tier == "Tier 3":
        suggestions.insert(0, "High risk detected. Do not act on this output without professional review.")
    elif risk_tier == "Tier 2":
        suggestions.insert(0, "Moderate risk detected. Cross-check this response with a second source.")

    log_event(
        "verification",
        "verification_completed",
        duration_ms=round((perf_counter() - start) * 1000.0, 2),
        nli_verification_duration_ms=nli_duration_ms,
        nli_source=nli_source,
        risk_tier=risk_tier,
        risk_score=risk_assessment["score"],
        kg_status=kg_status,
        nli_label=nli_label,
        nli_confidence=nli_confidence,
        rag_score=rag_score,
        rag_verified=rag_verified,
        imaging_status=imaging_status,
        warnings_count=len(all_warnings),
        citations_count=len(citations),
    )
    nli_metadata = _aggregate_nli_metadata(nli_claims)

    return {
        "risk_tier": risk_tier,
        "kg": kg_status,
        "nli": {
            "label": nli_label,
            "confidence": nli_confidence,
            "claims": nli_claims,
            **nli_metadata,
        },
        "risk_score": risk_assessment["score"],
        "risk_reasons": risk_assessment["reasons"],
        "final_assessment": final_assessment,
        "rag_score": rag_score,
        "rag_verified": rag_verified,
        "rag_error": rag_error,
        "citations": citations,
        "imaging": imaging_status,
        "warnings": all_warnings,
        "suggestions": suggestions,
        "matched_conditions": [c["name"] for c in matched_conditions],
    }


def get_verification_status() -> dict[str, Any]:
    nli_status = get_nli_status()
    return {
        "nli_available": nli_status.get("available", False),
        "nli_model_loaded": nli_status.get("model_loaded", False),
        "nli_load_error": nli_status.get("load_error"),
        "nli_model": nli_status.get("model"),
        "nli_engine": NLI_ENGINE,
        "nli_contradiction_threshold": NLI_CONTRADICTION_THRESHOLD,
        "condition_source": "medical_corpus",
        "corpus_chunks_path": str(DEFAULT_CHUNKS_PATH),
        "corpus_chunks_exists": DEFAULT_CHUNKS_PATH.exists(),
        "corpus_conditions_loaded": _corpus_condition_cache is not None,
    }
