from __future__ import annotations

from io import BytesIO
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

try:
    from .pdf_templates import generate_pdf
    from .report_generator import build_report_data
    from .structured_log import log_event
except ImportError:
    from pdf_templates import generate_pdf
    from report_generator import build_report_data
    from structured_log import log_event


router = APIRouter(prefix="/api", tags=["reports"])


@router.post("/download-report")
async def download_report(payload: dict[str, Any]) -> StreamingResponse:
    """
    Generate a HalluGuard-Med PDF report from an existing chat response payload.

    This endpoint performs report formatting only. It does not invoke or alter the
    MedGemma, RAG, Qdrant, verification, confidence, or radiology pipelines.
    """
    if not isinstance(payload, dict) or not payload.get("analysis"):
        raise HTTPException(status_code=400, detail="Valid HalluGuard analysis payload is required.")

    report_start = perf_counter()
    build_start = perf_counter()
    report = build_report_data(payload)
    build_duration_ms = round((perf_counter() - build_start) * 1000.0, 2)
    pdf_start = perf_counter()
    pdf_bytes = generate_pdf(report)
    pdf_duration_ms = round((perf_counter() - pdf_start) * 1000.0, 2)
    log_event(
        "runtime",
        "stage_timing",
        stage_name="report_generation",
        duration_ms=round((perf_counter() - report_start) * 1000.0, 2),
        report_data_build_ms=build_duration_ms,
        pdf_generation_ms=pdf_duration_ms,
        pdf_bytes=len(pdf_bytes),
    )
    filename = f"halluguard_report_{report['case_id']}.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
