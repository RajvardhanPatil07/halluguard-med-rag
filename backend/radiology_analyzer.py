"""
radiology_analyzer.py
─────────────────────
Uses TorchXRayVision (DenseNet121) to analyze uploaded chest X-ray images.

Returns:
- All pathology confidence scores as percentages (for bar chart)
- Critical findings list (above high threshold)
- Normal score
- graph_terms for verification.py comparison
- No substitute values — null/error returned explicitly if fails
"""

from io import BytesIO
from typing import Any
import contextlib
import io

try:
    from .structured_log import log_event
except ImportError:
    from structured_log import log_event

# ──────────────────────────────────────────────
# Optional imports
# ──────────────────────────────────────────────
try:
    import numpy as np
    import torch
    import torchxrayvision as xrv
    from PIL import Image
    import torchvision.transforms as transforms
    RADIOLOGY_AVAILABLE = True
except ImportError:
    RADIOLOGY_AVAILABLE = False
    np = None
    torch = None
    xrv = None
    Image = None
    transforms = None


# ──────────────────────────────────────────────
# Model singleton
# ──────────────────────────────────────────────
_model = None
_model_load_error = None


def _load_model():
    """Load DenseNet121. Cached after first call."""
    global _model, _model_load_error

    if _model is not None:
        return _model

    if not RADIOLOGY_AVAILABLE:
        raise RuntimeError(
            "TorchXRayVision is not installed. "
            "Run: pip install torchxrayvision"
        )

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _model = xrv.models.DenseNet(weights="densenet121-res224-all")
        _model.eval()
        _model_load_error = None
        return _model
    except Exception as exc:
        _model_load_error = str(exc)
        raise RuntimeError(f"Failed to load radiology model: {exc}") from exc


# ──────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────

# Findings above this shown in bar chart
DISPLAY_THRESHOLD = 0.30

# Findings above this marked as detected/active
CONFIDENCE_THRESHOLD = 0.60

# Findings above this marked as critical (red banner)
CRITICAL_THRESHOLD = 0.80


def finding_confidence_label(score_percent: float) -> str:
    if score_percent >= 80.0:
        return "likely"
    if score_percent >= 60.0:
        return "possible"
    return "low_confidence"


def format_finding_assessment(label: str, score_percent: float) -> str:
    readable = label.replace("_", " ").lower()
    confidence = finding_confidence_label(score_percent)
    if confidence == "likely":
        return f"Likely {readable} finding ({score_percent:.1f}%)."
    if confidence == "possible":
        return f"Possible {readable} detected ({score_percent:.1f}%). Further evaluation recommended."
    return f"Low-confidence {readable} signal ({score_percent:.1f}%). Do not treat as confirmed."


# ──────────────────────────────────────────────
# Mapping to radiology verification terms
# ──────────────────────────────────────────────
LABEL_TO_GRAPH_TERMS = {
    "Atelectasis":                ["atelectasis", "collapse", "volume loss"],
    "Consolidation":              ["consolidation", "lung opacity", "infiltrates"],
    "Infiltration":               ["infiltrates", "increased density", "lung opacity"],
    "Pneumothorax":               ["pneumothorax", "air in pleural space", "collapsed lung"],
    "Edema":                      ["pulmonary edema", "fluid overload", "Kerley B lines"],
    "Emphysema":                  ["hyperinflation", "bullae", "increased AP diameter"],
    "Fibrosis":                   ["fibrosis", "reticular pattern", "honeycombing"],
    "Effusion":                   ["pleural effusion", "fluid in pleural space", "blunted costophrenic angle"],
    "Pneumonia":                  ["consolidation", "air bronchogram", "lung opacity", "infiltrates"],
    "Pleural Thickening":         ["pleural thickening", "pleural disease"],
    "Pleural_Thickening":         ["pleural thickening", "pleural disease"],
    "Cardiomegaly":               ["cardiomegaly", "enlarged cardiac silhouette", "cardiothoracic ratio greater than 0.5"],
    "Nodule":                     ["nodules", "pulmonary nodule"],
    "Mass":                       ["mass lesion", "lung tumor", "pulmonary mass"],
    "Hernia":                     ["hernia", "diaphragmatic hernia"],
    "Lung Lesion":                ["lung lesion", "pulmonary lesion"],
    "Fracture":                   ["fracture line", "cortical break", "rib fracture"],
    "Lung Opacity":               ["lung opacity", "consolidation", "infiltrates"],
    "Enlarged Cardiomediastinum": ["mediastinal widening", "enlarged cardiac silhouette"],
}


# ──────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────
def _preprocess_image(image_bytes: bytes):
    """
    Converts image bytes to TorchXRayVision expected format:
    float32, range [-1024, 1024], shape [1, 1, 224, 224]
    """
    image = Image.open(BytesIO(image_bytes)).convert("L")  # grayscale

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    tensor = transform(image)           # [1, 224, 224]
    tensor = tensor * 2048.0 - 1024.0  # normalize to [-1024, 1024]
    tensor = tensor.unsqueeze(0)        # [1, 1, 224, 224]
    return tensor


# ──────────────────────────────────────────────
# Core analysis function
# ──────────────────────────────────────────────
def analyze_image(image_bytes: bytes | None) -> dict[str, Any] | None:
    """
    Analyzes a chest X-ray using TorchXRayVision DenseNet121.

    Returns None if no image provided.
    Returns structured dict with:
        status            — "Analyzed", "Neutral", or "Error"
        findings          — conditions above CONFIDENCE_THRESHOLD
        critical          — conditions above CRITICAL_THRESHOLD (red banner)
        normal_score      — percentage likelihood image is normal
        percentage_scores — all pathologies with % scores (for bar chart)
        graph_terms       — radiology verification terms
        warnings          — error messages if any
    """

    if not image_bytes:
        log_event("imaging", "image_not_supplied")
        return None

    if not RADIOLOGY_AVAILABLE:
        log_event(
            "imaging",
            "dependency_unavailable",
            "error",
            error="TorchXRayVision is not installed",
        )
        return {
            "status": "Error",
            "findings": None,
            "critical": None,
            "normal_score": None,
            "percentage_scores": None,
            "graph_terms": None,
            "warnings": [
                "TorchXRayVision is not installed — imaging analysis unavailable."
            ],
        }

    try:
        model = _load_model()
        tensor = _preprocess_image(image_bytes)

        with torch.no_grad():
            output = model(tensor)  # [1, num_pathologies]

        model_labels = model.pathologies
        scores_raw = output[0].cpu().numpy()

        # ── Raw scores dict ──
        raw_scores = {
            label: float(score)
            for label, score in zip(model_labels, scores_raw)
            if label is not None
        }

        # ── Percentage scores for bar chart (all above display threshold) ──
        percentage_scores = {
            label: round(score * 100, 1)
            for label, score in raw_scores.items()
            if score >= DISPLAY_THRESHOLD
        }

        # Sort by score descending
        percentage_scores = dict(
            sorted(percentage_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # ── Active findings (above confidence threshold) ──
        findings = [
            label
            for label, score in raw_scores.items()
            if score >= CONFIDENCE_THRESHOLD
        ]

        # ── Critical findings (above critical threshold) ──
        critical = [
            label
            for label, score in raw_scores.items()
            if score >= CRITICAL_THRESHOLD
        ]

        # ── Normal score ──
        # Computed as inverse of max finding score
        # If all findings are low → image is likely normal
        max_finding_score = max(raw_scores.values()) if raw_scores else 0.0
        normal_score = round((1.0 - max_finding_score) * 100, 1)
        normal_score = max(0.0, min(100.0, normal_score))  # clamp 0-100

        # ── Graph terms for verification.py ──
        graph_terms = []
        for finding in findings:
            terms = LABEL_TO_GRAPH_TERMS.get(finding, [finding.lower()])
            graph_terms.extend([finding.lower(), *terms])
        graph_terms = list(set(graph_terms))

        status = "Analyzed" if findings else "Neutral"

        result = {
            "status": status,
            "findings": findings,
            "critical": critical,
            "normal_score": normal_score,
            "percentage_scores": percentage_scores,
            "graph_terms": graph_terms,
            "warnings": [],
        }
        log_event(
            "imaging",
            "analysis_completed",
            status=status,
            findings_count=len(findings),
            critical_count=len(critical),
            normal_score=normal_score,
        )
        return result

    except Exception as exc:
        log_event(
            "imaging",
            "analysis_failed",
            "error",
            error=str(exc),
        )
        return {
            "status": "Error",
            "findings": None,
            "critical": None,
            "normal_score": None,
            "percentage_scores": None,
            "graph_terms": None,
            "warnings": [
                f"Radiology analysis failed: {str(exc)}"
            ],
        }


# ──────────────────────────────────────────────
# Status check
# ──────────────────────────────────────────────
def get_radiology_status() -> dict[str, Any]:
    return {
        "available": RADIOLOGY_AVAILABLE,
        "model_loaded": _model is not None,
        "last_load_error": _model_load_error,
        "display_threshold": DISPLAY_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "critical_threshold": CRITICAL_THRESHOLD,
    }
