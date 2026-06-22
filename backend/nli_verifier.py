"""
DeBERTa-v3 MNLI claim verification.

This module is intentionally isolated so model loading stays lazy and any
runtime/model failure can fall back to the existing rule-based verifier.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

try:
    from .structured_log import log_event
except ImportError:
    from structured_log import log_event


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", DEFAULT_NLI_MODEL)
NLI_SUPPORT_THRESHOLD = 0.65
NLI_CONTRADICTION_THRESHOLD = 0.75

_tokenizer = None
_model = None
_model_loaded = False
_load_error: str | None = None
_label_map: dict[str, int] = {}


INLINE_CITATION_RE = re.compile(
    r"\s*\[[A-Z0-9][A-Z0-9_-]{2,}(?:-[A-Z0-9_-]+)*\]",
    re.IGNORECASE,
)


class NliUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class NliEvidenceResult:
    entailment: float
    contradiction: float
    neutral: float
    label: str
    premise: str
    citation_id: str | None
    model: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model": self.model,
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral,
            "label": self.label,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_label(label: str) -> str:
    lowered = label.lower()
    if "entail" in lowered:
        return "entailment"
    if "contrad" in lowered:
        return "contradiction"
    if "neutral" in lowered:
        return "neutral"
    return lowered


def _load_model() -> tuple[Any, Any]:
    global _tokenizer, _model, _model_loaded, _load_error, _label_map

    if _model_loaded and _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    if _load_error:
        raise NliUnavailableError(_load_error)

    start = perf_counter()
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        _model.eval()
        if torch.cuda.is_available() and os.getenv("NLI_DEVICE", "").lower() == "cuda":
            _model.to("cuda")

        raw_labels = getattr(_model.config, "id2label", {}) or {}
        _label_map = {
            _normalise_label(label): int(index)
            for index, label in raw_labels.items()
        }
        missing = {"entailment", "contradiction", "neutral"} - set(_label_map)
        if missing:
            raise RuntimeError(f"NLI model labels missing: {sorted(missing)}")

        _model_loaded = True
        log_event(
            "nli",
            "model_loaded",
            model=NLI_MODEL_NAME,
            duration_ms=round((perf_counter() - start) * 1000.0, 2),
        )
        return _tokenizer, _model
    except Exception as exc:
        _load_error = str(exc)
        log_event(
            "nli",
            "model_load_failed",
            "warning",
            model=NLI_MODEL_NAME,
            error=_load_error,
        )
        raise NliUnavailableError(_load_error) from exc


def is_available() -> bool:
    try:
        _load_model()
        return True
    except NliUnavailableError:
        return False


def _classify_scores(entailment: float, contradiction: float) -> str:
    if contradiction >= NLI_CONTRADICTION_THRESHOLD:
        return "contradicted"
    if entailment >= NLI_SUPPORT_THRESHOLD:
        return "supported"
    return "unsupported"


def _clean_hypothesis_for_nli(claim: str) -> str:
    # Inline citation IDs are grounding metadata, not clinical claim content.
    return re.sub(INLINE_CITATION_RE, "", claim).strip()


def verify_claim_against_evidence(
    claim: str,
    evidence_hits: list[dict[str, Any]],
    *,
    max_evidence: int = 3,
) -> NliEvidenceResult:
    tokenizer, model = _load_model()
    candidates = [
        hit for hit in evidence_hits[:max_evidence]
        if (hit.get("_premise_text") or hit.get("text"))
    ]
    if not candidates:
        raise NliUnavailableError("No evidence supplied for NLI verification")

    premises = [str(hit.get("_premise_text") or hit.get("text") or "") for hit in candidates]
    clean_claim = _clean_hypothesis_for_nli(claim)
    hypotheses = [clean_claim for _ in premises]

    import torch

    device = next(model.parameters()).device
    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=int(os.getenv("NLI_MAX_LENGTH", "512")),
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    start = perf_counter()
    with torch.inference_mode():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu()

    scored: list[tuple[float, float, float, int]] = []
    for index, row in enumerate(probs):
        entailment = float(row[_label_map["entailment"]])
        contradiction = float(row[_label_map["contradiction"]])
        neutral = float(row[_label_map["neutral"]])
        scored.append((entailment, contradiction, neutral, index))

    best_contradiction = max(scored, key=lambda item: item[1])
    best_entailment = max(scored, key=lambda item: item[0])
    if best_contradiction[1] >= NLI_CONTRADICTION_THRESHOLD:
        chosen = best_contradiction
    else:
        chosen = best_entailment

    entailment, contradiction, neutral, index = chosen
    label = _classify_scores(entailment, contradiction)
    hit = candidates[index]
    result = NliEvidenceResult(
        entailment=round(entailment, 4),
        contradiction=round(contradiction, 4),
        neutral=round(neutral, 4),
        label=label,
        premise=premises[index],
        citation_id=hit.get("citation_id"),
        model=NLI_MODEL_NAME,
    )
    log_event(
        "nli",
        "claim_verified",
        model=NLI_MODEL_NAME,
        duration_ms=round((perf_counter() - start) * 1000.0, 2),
        evidence_count=len(candidates),
        label=label,
        entailment=result.entailment,
        contradiction=result.contradiction,
        neutral=result.neutral,
    )
    return result


def disabled_metadata() -> dict[str, Any]:
    return {
        "enabled": False,
        "model": NLI_MODEL_NAME,
        "entailment": 0.0,
        "contradiction": 0.0,
        "neutral": 0.0,
    }


def get_nli_status() -> dict[str, Any]:
    return {
        "enabled": _load_error is None,
        "available": _model_loaded,
        "model": NLI_MODEL_NAME,
        "model_loaded": _model_loaded,
        "load_error": _load_error,
        "entailment_threshold": NLI_SUPPORT_THRESHOLD,
        "support_threshold": NLI_SUPPORT_THRESHOLD,
        "contradiction_threshold": NLI_CONTRADICTION_THRESHOLD,
    }
