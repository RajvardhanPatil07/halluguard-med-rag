from __future__ import annotations

from io import BytesIO
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

try:
    from .pdf_templates import generate_pdf
    from .report_generator import build_report_data
except ImportError:
    from pdf_templates import generate_pdf
    from report_generator import build_report_data


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

    report = build_report_data(payload)
    pdf_bytes = generate_pdf(report)
    filename = f"halluguard_report_{report['case_id']}.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
