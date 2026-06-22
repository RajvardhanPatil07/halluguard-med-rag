from __future__ import annotations

from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BLUE = colors.HexColor("#2563eb")
DARK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
BORDER = colors.HexColor("#dbe3ef")
LIGHT_BLUE = colors.HexColor("#eff6ff")
LIGHT_GRAY = colors.HexColor("#f8fafc")


def _safe(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _list_text(values: list[Any]) -> str:
    if not values:
        return "N/A"
    return "\n".join(f"- {_safe(item)}" for item in values)


def _dict_text(values: dict[str, Any]) -> str:
    if not values:
        return "N/A"
    return "\n".join(f"- {key}: {_safe(value)}" for key, value in values.items())


def _status_label(value: Any) -> str:
    labels = {
        "supported": "SUPPORTED",
        "weak_support": "WEAK SUPPORT",
        "unsupported": "UNSUPPORTED",
        "insufficient": "INSUFFICIENT",
        "contradicted": "CONTRADICTED",
    }
    return labels.get(str(value or "").lower(), _safe(value))


def _percent(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if 0 <= numeric <= 1:
        numeric *= 100
    if 0 < numeric < 0.1:
        return "<0.1%"
    if 99.9 < numeric < 100:
        return ">99.9%"
    return f"{numeric:.1f}%"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=22,
        ),
        "section": ParagraphStyle(
            "SectionHeader",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=BLUE,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=DARK,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=DARK,
        ),
        "muted": ParagraphStyle(
            "Muted",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(_safe(text)).replace("\n", "<br/>"), style)


def _section(story: list[Any], title: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph(title, styles["section"]))


def _key_value_table(rows: list[tuple[str, Any]], styles: dict[str, ParagraphStyle]) -> Table:
    table_data = [[_p(label, styles["small"]), _p(value, styles["small"])] for label, value in rows]
    table = Table(table_data, colWidths=[1.8 * inch, 4.8 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _data_table(headers: list[str], rows: list[list[Any]], widths: list[float], styles: dict[str, ParagraphStyle]) -> Table:
    body_rows = rows or [["N/A" for _ in headers]]
    table_data = [[_p(header, styles["small"]) for header in headers]]
    table_data.extend([[_p(cell, styles["small"]) for cell in row] for row in body_rows])
    table = Table(table_data, colWidths=[width * inch for width in widths], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.line(doc.leftMargin, 0.55 * inch, A4[0] - doc.rightMargin, 0.55 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(
        A4[0] / 2,
        0.36 * inch,
        f"Generated by HalluGuard-Med | AI Medical Verification System | Page {doc.page}",
    )
    canvas.restoreState()


def generate_pdf(report: dict[str, Any]) -> bytes:
    """Render a professional multi-page HalluGuard-Med PDF report in memory."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.75 * inch,
        title=f"HalluGuard-Med Report {report.get('case_id', '')}",
    )
    styles = _styles()
    story: list[Any] = []

    story.append(Paragraph("HalluGuard-Med", styles["title"]))
    story.append(Paragraph("AI Medical Verification Report", styles["subtitle"]))
    story.append(_key_value_table([
        ("Generated Case ID", report.get("case_id")),
        ("Report Timestamp", report.get("timestamp")),
        ("Image Uploaded", report.get("image_uploaded")),
    ], styles))
    story.append(Spacer(1, 16))

    _section(story, "Medical Query", styles)
    story.append(_p(report.get("query"), styles["body"]))

    _section(story, "MedGemma Response", styles)
    story.append(_p(report.get("final_response"), styles["body"]))

    _section(story, "Verification Summary", styles)
    summary = report.get("summary", {})
    story.append(_key_value_table([
        ("Risk Tier", summary.get("risk_tier")),
        ("Risk Score", summary.get("risk_score")),
        ("Confidence Label", summary.get("confidence_label")),
        ("Confidence Percentage", summary.get("confidence_percentage")),
    ], styles))

    _section(story, "Analysis Results", styles)
    analysis = report.get("analysis_results", {})
    story.append(_key_value_table([
        ("KG Verification", analysis.get("kg")),
        ("NLI Verification", f"{analysis.get('nli')} ({analysis.get('nli_confidence')})"),
        ("RAG Verification", f"{analysis.get('rag_score')} | Verified: {_safe(analysis.get('rag_verified'))}"),
        ("RAG Error", analysis.get("rag_error")),
        ("Imaging Verification", analysis.get("imaging")),
    ], styles))
    if report.get("confidence_breakdown"):
        story.append(Spacer(1, 8))
        story.append(_key_value_table([
            ("Confidence Breakdown", _dict_text(report.get("confidence_breakdown", {}))),
        ], styles))
    if report.get("timing_metrics"):
        story.append(Spacer(1, 8))
        story.append(_key_value_table([
            ("Timing Metrics", _dict_text(report.get("timing_metrics", {}))),
        ], styles))

    story.append(PageBreak())

    _section(story, "Claim Verification", styles)
    claims_summary = report.get("claims_summary", {})
    if claims_summary:
        story.append(_key_value_table([
            ("Claims Summary", _dict_text(claims_summary)),
        ], styles))
        story.append(Spacer(1, 8))
    claim_rows = []
    for claim in report.get("claims", []):
        support_detail = "\n".join([
            _percent(claim.get("support_score")),
            f"Reason: {_safe(claim.get('reason'))}",
            f"Matched: {_list_text(claim.get('matched_concepts', []))}",
            f"Missing: {_list_text(claim.get('missing_concepts', []))}",
            f"Breakdown: {_dict_text(claim.get('support_breakdown', {}))}",
        ])
        claim_rows.append([
            claim.get("claim_id", "N/A"),
            claim.get("claim", "N/A"),
            _status_label(claim.get("status")),
            support_detail,
        ])
    story.append(_data_table(["ID", "Extracted Claim", "Status", "Support"], claim_rows, [0.7, 3.9, 1.0, 1.0], styles))

    _section(story, "Retrieval Summary", styles)
    retrieval_summary = report.get("retrieval_summary", {})
    story.append(_key_value_table([
        ("Retrieval Summary", _dict_text({
            "hits_count": retrieval_summary.get("hits_count"),
            "bm25_count": retrieval_summary.get("bm25_count"),
            "dense_count": retrieval_summary.get("dense_count"),
            "source_agreement_ratio": retrieval_summary.get("source_agreement_ratio"),
        })),
        ("Top Conditions", _list_text([
            f"{item.get('condition')} ({item.get('count')})"
            for item in retrieval_summary.get("top_conditions", [])
        ])),
        ("Top Citations", _list_text([
            f"{item.get('citation_id')} | {item.get('condition')} | score {item.get('score')}"
            for item in retrieval_summary.get("top_citations", [])
        ])),
    ], styles))

    _section(story, "Evidence Section", styles)
    evidence_by_id = {
        row.get("citation_id"): row
        for row in report.get("evidence_scores", [])
        if isinstance(row, dict)
    }
    evidence_rows = []
    for citation in report.get("citations", []):
        evidence_score = evidence_by_id.get(citation.get("id")) or {}
        score_detail = "\n".join([
            f"Evidence score: {_percent(evidence_score.get('final_score'))}",
            f"Lexical relevance: {_percent(evidence_score.get('lexical_relevance'))}",
            f"Matched terms: {_list_text(evidence_score.get('matched_query_terms', []))}",
        ])
        evidence_rows.append([
            citation.get("id", "N/A"),
            citation.get("condition", "N/A"),
            citation.get("section", "N/A"),
            f"{citation.get('text', 'N/A')}\n\n{score_detail}",
        ])
    story.append(_data_table(["Evidence ID", "Condition", "Section", "Retrieved Evidence"], evidence_rows, [0.9, 1.25, 1.0, 3.45], styles))

    _section(story, "Source Conflict Section", styles)
    conflict_rows = []
    for conflict in report.get("source_conflicts", []):
        conflict_rows.append([
            conflict.get("conflict_id", "N/A"),
            conflict.get("severity", "N/A"),
            conflict.get("reason", "N/A"),
        ])
    story.append(_data_table(["Conflict ID", "Severity", "Reason"], conflict_rows, [1.0, 1.1, 4.5], styles))

    story.append(PageBreak())

    _section(story, "Imaging Findings", styles)
    imaging = report.get("imaging", {})
    score_rows = [[name, f"{score}%"] for name, score in imaging.get("percentage_scores", {}).items()]
    story.append(_key_value_table([
        ("Imaging Status", imaging.get("status")),
        ("Findings", _list_text(imaging.get("findings", []))),
        ("Critical Findings", _list_text(imaging.get("critical", []))),
        ("Normal Score", _percent(imaging.get("normal_score"))),
        ("Imaging Warnings", _list_text(imaging.get("warnings", []))),
    ], styles))
    story.append(Spacer(1, 8))
    story.append(_data_table(["Finding", "Percentage Score"], score_rows, [4.8, 1.8], styles))

    _section(story, "Risk Findings", styles)
    risk = report.get("risk_findings", {})
    story.append(_key_value_table([
        ("Risk Reasons", _list_text(risk.get("risk_reasons", []))),
        ("Warnings", _list_text(risk.get("warnings", []))),
    ], styles))

    _section(story, "Final Assessment", styles)
    final_assessment = report.get("final_assessment", {})
    story.append(_key_value_table([
        ("Verdict", final_assessment.get("verdict")),
        ("Recommendation", final_assessment.get("recommendation")),
        ("Risk Tier / Score", f"{_safe(final_assessment.get('risk_tier'))} / {_safe(final_assessment.get('risk_score'))}"),
        ("Confidence", f"{_safe(final_assessment.get('confidence_label'))} ({_percent(final_assessment.get('confidence_score'))})"),
        ("RAG Verified", final_assessment.get("rag_verified")),
    ], styles))

    _section(story, "Recommendations", styles)
    story.append(_p(_list_text(report.get("recommendations", [])), styles["body"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Generated by HalluGuard-Med | AI Medical Verification System",
        styles["footer"],
    ))
    story.append(Paragraph(
        "Clinical decision-support disclaimer: This report is for medical AI verification and review. "
        "It is not a substitute for diagnosis, treatment, or judgment by a licensed clinician.",
        styles["footer"],
    ))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
