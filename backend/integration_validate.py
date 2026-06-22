"""
Runtime integration validator for HalluGuard-Med.

This script performs real checks only. It does not use fallback responses,
mock embeddings, or synthetic model outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .medgemma import generate_response, get_model_status
    from .radiology_analyzer import analyze_image
    from .rag_pipeline import run_rag_pipeline
    from .runtime_checks import get_runtime_status
    from .verification import verify_response
except ImportError:
    from medgemma import generate_response, get_model_status
    from radiology_analyzer import analyze_image
    from rag_pipeline import run_rag_pipeline
    from runtime_checks import get_runtime_status
    from verification import verify_response


QA_QUERY = "What are common symptoms and first-line management options for pneumonia?"
HALLUCINATION_QUERY = "Can pneumonia be guaranteed to cure immediately without antibiotics or clinician review?"


def _result(ok: bool, **fields: Any) -> dict[str, Any]:
    return {"ok": ok, **fields}


def validate_text_pipeline(query: str, run_generation: bool) -> dict[str, Any]:
    try:
        rag_result = run_rag_pipeline(query)
    except Exception as exc:
        return _result(False, stage="rag", error=str(exc))

    if not run_generation:
        return _result(
            True,
            stage="rag",
            rag_score=rag_result.get("rag_score"),
            citations_count=len(rag_result.get("citations", [])),
        )

    try:
        response = generate_response(rag_result["grounded_query"])
    except Exception as exc:
        return _result(False, stage="generation", rag_score=rag_result.get("rag_score"), error=str(exc))

    verification = verify_response(query, response, rag_result=rag_result)
    return _result(
        True,
        stage="full_text",
        rag_score=verification.get("rag_score"),
        risk_tier=verification.get("risk_tier"),
        risk_score=verification.get("risk_score"),
        nli=verification.get("nli"),
        citations_count=len(verification.get("citations", [])),
        response_chars=len(response),
    )


def validate_image(image_path: str | None) -> dict[str, Any]:
    if not image_path:
        return _result(False, stage="imaging", error="No image path supplied")

    path = Path(image_path)
    if not path.exists():
        return _result(False, stage="imaging", error=f"Image not found: {path}")

    try:
        result = analyze_image(path.read_bytes())
        return _result(
            result is not None and result.get("status") != "Error",
            stage="imaging",
            result=result,
        )
    except Exception as exc:
        return _result(False, stage="imaging", error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Optional chest X-ray image path")
    parser.add_argument("--run-generation", action="store_true", help="Load MedGemma and generate real text")
    args = parser.parse_args()

    report = {
        "runtime": get_runtime_status(),
        "model_status": get_model_status(),
        "medical_qa": validate_text_pipeline(QA_QUERY, args.run_generation),
        "hallucination_query": validate_text_pipeline(HALLUCINATION_QUERY, args.run_generation),
        "image": validate_image(args.image),
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
