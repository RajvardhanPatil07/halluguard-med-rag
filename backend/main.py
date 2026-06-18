import hashlib
from time import perf_counter
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from .audit_log import write_audit_event
    from .medgemma import MedGemmaError, generate_response, get_model_status
    from .radiology_analyzer import analyze_image, format_finding_assessment, get_radiology_status
    from .verification import verify_response, get_verification_status
    from .rag import get_rag_status
    from .rag_pipeline import RagPipelineError, run_rag_pipeline
    from .report_routes import router as report_router
    from .runtime_checks import get_runtime_status
    from .safety_pipeline import get_safety_pipeline_status, run_post_generation_safety
    from .safety_gate import (
        enforce_answer_policy,
        sanitize_hidden_candidate_safety,
        sanitize_hidden_candidate_verification,
    )
    from .structured_log import log_event
except ImportError:
    from audit_log import write_audit_event
    from medgemma import MedGemmaError, generate_response, get_model_status
    from radiology_analyzer import analyze_image, format_finding_assessment, get_radiology_status
    from verification import verify_response, get_verification_status
    from rag import get_rag_status
    from rag_pipeline import RagPipelineError, run_rag_pipeline
    from report_routes import router as report_router
    from runtime_checks import get_runtime_status
    from safety_pipeline import get_safety_pipeline_status, run_post_generation_safety
    from safety_gate import (
        enforce_answer_policy,
        sanitize_hidden_candidate_safety,
        sanitize_hidden_candidate_verification,
    )
    from structured_log import log_event


app = FastAPI(title="HalluGuard-Med API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Report export integration: registers PDF generation routes without changing
# chat, MedGemma, RAG, Qdrant, verification, confidence, or radiology logic.
app.include_router(report_router)


def _build_image_augmented_query(query: str, imaging_result: dict[str, Any] | None) -> str:
    if not imaging_result or imaging_result.get("status") != "Analyzed":
        return query

    findings = imaging_result.get("findings") or []
    critical = imaging_result.get("critical") or []
    scores = imaging_result.get("percentage_scores") or {}
    assessments = [
        format_finding_assessment(name, float(score))
        for name, score in sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    ]
    parts = [query]
    if assessments:
        parts.append(f"Radiology model assessment: {' '.join(assessments[:4])}")
    elif findings:
        parts.append(f"Detected radiology findings: {', '.join(findings)}.")
    if critical:
        parts.append(f"Likely high-confidence radiology findings: {', '.join(critical)}.")
    if scores:
        top_scores = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)[:4]
        score_text = ", ".join(f"{name} {value}%" for name, value in top_scores)
        parts.append(f"Radiology model scores: {score_text}.")
    parts.append(
        "Prioritize the uploaded image findings. Do not list unrelated X-ray diagnoses unless clearly stated as differential possibilities."
    )
    return " ".join(parts)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <html>
      <head><title>HalluGuard-Med API</title></head>
      <body style="font-family: Arial, sans-serif; margin: 40px;">
        <h1>🛡️ HalluGuard-Med API</h1>
        <p>Backend is running.</p>
        <ul>
          <li><a href="/docs">Open API docs</a></li>
          <li><a href="/health">Check health</a></li>
          <li><a href="/api/model-status">Check model status</a></li>
        </ul>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, Any]:
    runtime = get_runtime_status()
    return {
        "status": runtime["status"],
        "model": runtime["medgemma"],
        "failures": runtime["failures"],
    }


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return get_runtime_status()


@app.get("/api/model-status")
def model_status() -> dict[str, Any]:
    """
    Returns full status of all system modules.
    Useful for debugging in Cell 7 of Kaggle notebook.
    """
    return {
        "medgemma": get_model_status(),
        "radiology": get_radiology_status(),
        "verification": get_verification_status(),
        "safety_pipeline": get_safety_pipeline_status(),
        "rag": get_rag_status(),
    }


@app.get("/api/retrieval-status")
def retrieval_status() -> dict[str, Any]:
    runtime = get_runtime_status()
    return {
        "status": "ok" if runtime["qdrant"]["available"] and runtime["retrieval"].get("google_api_key_configured") else "unavailable",
        "retrieval": runtime["retrieval"],
        "qdrant": runtime["qdrant"],
        "google_api_key_configured": runtime["checks"]["google_api_key"]["ok"],
    }


@app.post("/api/chat", response_model=None)
async def chat(
    query: str = Form(...),
    image: UploadFile | None = File(default=None),
) -> dict[str, Any] | JSONResponse:
    """
    Main chat endpoint.

    Request:
        query — medical question (text)
        image — optional chest X-ray image

    Response format:
        final_response  — policy-controlled answer visible to the user
        analysis:
            risk_tier       — Tier 1 / Tier 2 / Tier 3
            kg              — Match / Neutral
            nli:
                label       — Entailed / Contradicted / Neutral
                confidence  — float 0-1 or null
            rag_score       — float 0-1 or null
            rag_verified    — bool or null
            imaging:
                status          — Analyzed / Neutral / N/A / Error
                findings        — list of detected pathologies
                critical        — list of critical findings
                normal_score    — float % or null
                percentage_scores — dict {pathology: %} or null
        warnings        — list of warning strings
        suggestions     — list of suggestion strings
        matched_conditions — list of matched condition names
        image_uploaded  — bool
    """

    # ── Read image ──
    chat_start = perf_counter()
    image_bytes = await image.read() if image else None
    image_uploaded = image_bytes is not None and len(image_bytes) > 0
    imaging_result = analyze_image(image_bytes) if image_uploaded else None
    retrieval_query = _build_image_augmented_query(query, imaging_result)

    # Retrieve evidence before generation. Never generate without grounded context.
    retrieval_start = perf_counter()
    try:
        rag_result = run_rag_pipeline(retrieval_query)
    except RagPipelineError as exc:
        log_event(
            "runtime",
            "chat_failed_rag",
            "error",
            query_chars=len(query),
            image_uploaded=image_uploaded,
            error=str(exc),
        )
        write_audit_event({
            "query": query,
            "image_uploaded": image_uploaded,
            "model_available": None,
            "rag_available": False,
            "rag_error": str(exc),
            "generated_response": False,
        })
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "RAG_RETRIEVAL_FAILED",
                    "message": (
                        "Hybrid RAG retrieval failed. "
                        "No AI medical response was generated."
                    ),
                    "detail": str(exc),
                    "rag_status": get_rag_status(),
                }
            },
        )

    retrieval_duration_ms = round((perf_counter() - retrieval_start) * 1000.0, 2)
    log_event(
        "runtime",
        "stage_timing",
        stage_name="rag_retrieval_and_pre_generation_safety",
        duration_ms=retrieval_duration_ms,
        query_chars=len(retrieval_query),
    )

    # Generate MedGemma response. Never continue with synthetic text.
    generation_start = perf_counter()
    try:
        ai_response = generate_response(rag_result["grounded_query"], image_bytes)
    except MedGemmaError as exc:
        model_status_data = get_model_status()
        log_event(
            "runtime",
            "chat_failed_generation",
            "error",
            query_chars=len(query),
            image_uploaded=image_uploaded,
            error=str(exc),
            model_status=model_status_data.get("status"),
        )
        write_audit_event({
            "query": query,
            "image_uploaded": image_uploaded,
            "model_available": False,
            "model_status": model_status_data.get("status"),
            "model_error": str(exc),
            "generated_response": False,
        })
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "MEDGEMMA_UNAVAILABLE",
                    "message": (
                        "MedGemma is not available. "
                        "No AI medical response was generated."
                    ),
                    "model_status": model_status_data,
                }
            },
        )

    # ── Radiology analysis ──
    # ── Clinical claim safety layer ──
    generation_duration_ms = round((perf_counter() - generation_start) * 1000.0, 2)
    log_event(
        "runtime",
        "stage_timing",
        stage_name="response_generation",
        duration_ms=generation_duration_ms,
        prompt_chars=len(rag_result["grounded_query"]),
        response_chars=len(ai_response),
    )

    post_start = perf_counter()
    post_generation_safety = run_post_generation_safety(
        query=query,
        answer=ai_response,
        rag_result=rag_result,
        precheck=rag_result.get("safety_precheck", {}),
        imaging_result=imaging_result,
    )
    post_duration_ms = round((perf_counter() - post_start) * 1000.0, 2)
    log_event(
        "runtime",
        "stage_timing",
        stage_name="claim_extraction_and_nli_verification",
        duration_ms=post_duration_ms,
        claims_count=len(post_generation_safety.get("claims", [])),
    )

    # ── Verification pipeline ──
    verification_start = perf_counter()
    verification = verify_response(
        query,
        ai_response,
        imaging_result,
        rag_result,
        safety_result=post_generation_safety,
    )
    verification_duration_ms = round((perf_counter() - verification_start) * 1000.0, 2)
    log_event(
        "runtime",
        "stage_timing",
        stage_name="risk_scoring",
        duration_ms=verification_duration_ms,
        risk_score=verification["risk_score"],
        risk_tier=verification["risk_tier"],
    )
    verification["warnings"].extend(
        rag_result.get("safety_precheck", {}).get("warnings", [])
    )
    verification["warnings"].extend(post_generation_safety.get("warnings", []))

    answer_gate = enforce_answer_policy(
        query=query,
        candidate_answer=ai_response,
        verification=verification,
        rag_result=rag_result,
        safety_result=post_generation_safety,
        imaging_result=imaging_result,
    )
    final_response = answer_gate["final_response"]
    answer_policy = answer_gate["answer_policy"]
    public_safety_result = sanitize_hidden_candidate_safety(
        post_generation_safety,
        answer_policy,
    )
    public_verification = sanitize_hidden_candidate_verification(
        verification,
        answer_policy,
    )
    candidate_response_hash = hashlib.sha256(ai_response.encode("utf-8")).hexdigest()

    # ── Build imaging section for response ──
    if imaging_result is None:
        imaging_data = {
            "status": "N/A",
            "findings": None,
            "critical": None,
            "normal_score": None,
            "percentage_scores": None,
            "warnings": [],
        }
    else:
        imaging_data = {
            "status": imaging_result.get("status"),
            "findings": imaging_result.get("findings"),
            "critical": imaging_result.get("critical"),
            "normal_score": imaging_result.get("normal_score"),
            "percentage_scores": imaging_result.get("percentage_scores"),
            "warnings": imaging_result.get("warnings", []),
        }

    # ── Build full response ──
    result = {
        "final_response": final_response,
        "analysis": {
            "risk_tier": public_verification["risk_tier"],
            "risk_score": public_verification["risk_score"],
            "risk_reasons": public_verification["risk_reasons"],
            "final_assessment": public_verification.get("final_assessment"),
            "kg": public_verification["kg"],
            "nli": public_verification["nli"],
            "rag_score": public_verification["rag_score"],
            "rag_verified": public_verification["rag_verified"],
            "rag_error": public_verification["rag_error"],
            "retrieval_summary": rag_result.get("retrieval_summary", {}),
            "citations": public_verification["citations"],
            "answer_policy": answer_policy,
            "imaging": imaging_data,
            "safety": {
                "pre_generation": {
                    "query_entities": rag_result.get("safety_precheck", {}).get("query_entities", []),
                    "evidence_scores": rag_result.get("safety_precheck", {}).get("evidence_scores", []),
                    "source_conflicts": rag_result.get("safety_precheck", {}).get("source_conflicts", []),
                },
                "post_generation": public_safety_result,
            },
            "confidence": public_safety_result.get("confidence"),
            "claims_summary": public_safety_result.get("claims_summary", {}),
            "claim_verification": public_safety_result.get("claim_verification", []),
        },
        "warnings": public_verification["warnings"],
        "suggestions": public_verification["suggestions"],
        "matched_conditions": public_verification["matched_conditions"],
        "citations": public_verification["citations"],
        "image_uploaded": image_uploaded,
        "meta": {
            "response_chars": len(final_response),
            "candidate_response_chars": len(ai_response),
            "candidate_response_sha256": candidate_response_hash,
            "timing_ms": {
                "retrieval_and_pre_generation_safety": retrieval_duration_ms,
                "response_generation": generation_duration_ms,
                "claim_extraction_and_verification": post_duration_ms,
                "risk_scoring": verification_duration_ms,
                "total": round((perf_counter() - chat_start) * 1000.0, 2),
            },
        },
    }

    # ── Audit log ──
    write_audit_event({
        "query": query,
        "image_uploaded": image_uploaded,
        "model_available": True,
        "risk_tier": verification["risk_tier"],
        "risk_score": verification["risk_score"],
        "risk_reasons": verification["risk_reasons"],
        "answer_policy": answer_policy,
        "safety_confidence": post_generation_safety.get("confidence"),
        "claims_summary": post_generation_safety.get("claims_summary", {}),
        "claim_status_counts": answer_policy.get("claim_status_counts", {}),
        "candidate_response_chars": len(ai_response),
        "candidate_response_sha256": candidate_response_hash,
        "final_response_chars": len(final_response),
        "source_conflicts": rag_result.get("safety_precheck", {}).get("source_conflicts", []),
        "rag_score": verification["rag_score"],
        "rag_verified": verification["rag_verified"],
        "rag_error": verification["rag_error"],
        "citations": verification["citations"],
        "nli_label": verification["nli"]["label"],
        "matched_conditions": verification["matched_conditions"],
        "imaging_status": imaging_data["status"],
        "critical_findings": imaging_data["critical"],
    })
    log_event(
        "runtime",
        "chat_completed",
        duration_ms=round((perf_counter() - chat_start) * 1000.0, 2),
        query_chars=len(query),
        image_uploaded=image_uploaded,
        response_chars=len(final_response),
        candidate_response_chars=len(ai_response),
        answer_policy_decision=answer_policy.get("decision"),
        risk_tier=verification["risk_tier"],
        rag_score=verification["rag_score"],
        nli_label=verification["nli"]["label"],
        imaging_status=imaging_data["status"],
    )

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
