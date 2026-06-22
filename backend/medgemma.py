"""
medgemma.py

Generates medical responses through local Ollama MedGemma.

Important safety rule:
- This module never returns synthetic medical text.
- If Ollama MedGemma is unavailable, callers receive an explicit exception.
"""

import re
from datetime import datetime, timezone
from io import BytesIO
from time import perf_counter

try:
    from .ollama_client import (
        OllamaError,
        OllamaGenerationError,
        OllamaUnavailableError,
        check_model_available,
        generate_text,
    )
    from .settings import (
        MAX_NEW_TOKENS,
        OLLAMA_HOST,
        OLLAMA_MODEL,
        OLLAMA_TIMEOUT_SECONDS,
    )
    from .structured_log import log_event
except ImportError:
    from ollama_client import (
        OllamaError,
        OllamaGenerationError,
        OllamaUnavailableError,
        check_model_available,
        generate_text,
    )
    from settings import (
        MAX_NEW_TOKENS,
        OLLAMA_HOST,
        OLLAMA_MODEL,
        OLLAMA_TIMEOUT_SECONDS,
    )
    from structured_log import log_event


_model_status = "unavailable"
_last_load_attempt = None
_last_load_error = None
_last_generation_info = None

GENERATION_IMAGE_MAX_SIDE = 1024


class MedGemmaError(RuntimeError):
    pass


class MedGemmaLoadError(MedGemmaError):
    pass


class MedGemmaUnavailableError(MedGemmaError):
    pass


class MedGemmaGenerationError(MedGemmaError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_response(text: str) -> str:
    """
    Strips internal thinking/reasoning blocks from MedGemma output.
    These model-internal tags should never reach the user.
    """
    if not text:
        return text

    text = re.sub(
        r"<thought>.*?</thought>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<unused\d+>thought.*?<unused\d+>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<unused\d+>", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(Thinking Process|Planning|Drafting|Self-Correction|Confidence Score|Strategizing).*?(?=\n\n|\Z)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*(?:thought|thinking)\s*\n(?:\d+\.\s+.*?(?:\n\s*\n|$))+",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*(?:thought|thinking)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    skip_patterns = [
        "**identify the core question",
        "**scan the provided evidence",
        "identify the core question",
        "scan the provided evidence",
        "thinking process",
        "thought",
        "confidence score:",
        "strategizing complete",
        "self-correction",
        "constraint checklist",
        "plan:",
        "drafting:",
        "refining",
        "revised plan",
    ]
    cleaned_lines = []
    for line in text.split("\n"):
        lower_line = line.strip().lower()
        if any(pattern in lower_line for pattern in skip_patterns):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ensure_model_loaded():
    """
    Checks that local Ollama MedGemma is reachable.
    Ollama owns model loading; this function validates availability only.
    """
    global _model_status, _last_load_attempt, _last_load_error

    _model_status = "checking"
    _last_load_attempt = _utc_now()
    status = check_model_available(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
        timeout=min(OLLAMA_TIMEOUT_SECONDS, 10.0),
    )
    if status.get("available"):
        _model_status = "loaded"
        _last_load_error = None
        return status

    _model_status = "failed"
    _last_load_error = status.get("error") or "Ollama MedGemma is unavailable"
    log_event(
        "generation",
        "model_load_failed",
        "error",
        model_ref=OLLAMA_MODEL,
        host=OLLAMA_HOST,
        error=_last_load_error,
    )
    raise MedGemmaUnavailableError(
        f"MedGemma is not available: {_last_load_error}"
    )


def is_model_available() -> bool:
    return _model_status == "loaded"


def _build_prompt(query: str) -> str:
    return f"""
You are MedGemma, an expert medical AI assistant.

Instructions:
- Provide comprehensive and detailed medical explanations.
- Use clear section headings.
- Use bullet points when appropriate.
- Explain symptoms, causes, risk factors, diagnosis, treatment options, prevention, and prognosis whenever relevant.
- Expand each section with sufficient detail.
- Do not provide one-paragraph answers.
- Do not give extremely short summaries.
- If information is uncertain, clearly state the uncertainty.
- Do not provide a final diagnosis.
- Do not provide a treatment prescription.
- Answer directly without showing internal reasoning.

Question:
{query}
"""


def _prepare_image_for_generation(image_bytes: bytes | None) -> tuple[bytes | None, dict]:
    timings: dict[str, float | int | bool | str] = {
        "input_bytes": len(image_bytes or b""),
        "resized": False,
    }
    if not image_bytes:
        return None, timings

    start = perf_counter()
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            timings["input_width"], timings["input_height"] = image.size
            max_side = max(image.size)
            if max_side <= GENERATION_IMAGE_MAX_SIDE:
                timings["output_bytes"] = len(image_bytes)
                timings["prepare_ms"] = round((perf_counter() - start) * 1000.0, 2)
                return image_bytes, timings

            image = image.convert("RGB")
            image.thumbnail((GENERATION_IMAGE_MAX_SIDE, GENERATION_IMAGE_MAX_SIDE))
            output = BytesIO()
            image.save(output, format="JPEG", quality=92, optimize=True)
            prepared = output.getvalue()
            timings.update({
                "resized": True,
                "output_width": image.size[0],
                "output_height": image.size[1],
                "output_bytes": len(prepared),
            })
            timings["prepare_ms"] = round((perf_counter() - start) * 1000.0, 2)
            return prepared, timings
    except Exception as exc:
        timings["fallback_reason"] = str(exc)
        timings["output_bytes"] = len(image_bytes)
        timings["prepare_ms"] = round((perf_counter() - start) * 1000.0, 2)
        return image_bytes, timings


def _encode_image(image_bytes: bytes | None) -> tuple[list[str] | None, dict]:
    prepared, timings = _prepare_image_for_generation(image_bytes)
    encode_start = perf_counter()
    if not prepared:
        timings["base64_encode_ms"] = 0.0
        return None, timings
    if not image_bytes:
        timings["base64_encode_ms"] = 0.0
        return None, timings
    import base64

    encoded = base64.b64encode(prepared).decode("ascii")
    timings["base64_chars"] = len(encoded)
    timings["base64_encode_ms"] = round((perf_counter() - encode_start) * 1000.0, 2)
    return [encoded], timings


def generate_response(
    query: str,
    image_bytes: bytes | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """
    Generates a medical response from local Ollama MedGemma.
    Raises if Ollama cannot serve the model or generation fails.
    """
    global _model_status, _last_generation_info

    token_limit = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS
    prompt = _build_prompt(query)
    start = perf_counter()
    encode_start = perf_counter()
    images, image_timing = _encode_image(image_bytes)
    image_timing["total_image_encode_ms"] = round((perf_counter() - encode_start) * 1000.0, 2)

    try:
        ollama_start = perf_counter()
        data = generate_text(
            prompt=prompt,
            images=images,
            model=OLLAMA_MODEL,
            host=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT_SECONDS,
            max_new_tokens=token_limit,
        )
        ollama_request_ms = round((perf_counter() - ollama_start) * 1000.0, 2)
    except OllamaUnavailableError as exc:
        _model_status = "failed"
        _last_generation_info = None
        log_event(
            "generation",
            "generation_failed",
            "error",
            image_supplied=image_bytes is not None,
            query_chars=len(query),
            host=OLLAMA_HOST,
            model=OLLAMA_MODEL,
            error=str(exc),
            image_timing_ms=image_timing,
        )
        raise MedGemmaUnavailableError(
            f"MedGemma is not available: {exc}"
        ) from exc
    except OllamaGenerationError as exc:
        log_event(
            "generation",
            "generation_failed",
            "error",
            image_supplied=image_bytes is not None,
            query_chars=len(query),
            host=OLLAMA_HOST,
            model=OLLAMA_MODEL,
            error=str(exc),
            image_timing_ms=image_timing,
        )
        raise MedGemmaGenerationError(
            f"MedGemma generation failed: {exc}"
        ) from exc
    except OllamaError as exc:
        raise MedGemmaGenerationError(
            f"MedGemma generation failed: {exc}"
        ) from exc

    _model_status = "loaded"
    raw_text = str(data.get("response") or "").strip()
    clean_text = _clean_response(raw_text)
    _last_generation_info = {
        "max_new_tokens": token_limit,
        "token_count": data.get("eval_count"),
        "reached_eos": bool(data.get("done")),
        "raw_chars": len(raw_text),
        "clean_chars": len(clean_text),
        "thinking_blocks_removed": len(raw_text) - len(clean_text),
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "eval_duration": data.get("eval_duration"),
        "ollama_request_ms": ollama_request_ms,
        "image_timing_ms": image_timing,
    }
    log_event(
        "generation",
        "generation_success",
        duration_ms=round((perf_counter() - start) * 1000.0, 2),
        image_supplied=image_bytes is not None,
        query_chars=len(query),
        response_chars=len(clean_text),
        token_count=data.get("eval_count"),
        reached_eos=bool(data.get("done")),
        host=OLLAMA_HOST,
        model=OLLAMA_MODEL,
        ollama_request_ms=ollama_request_ms,
        image_timing_ms=image_timing,
    )
    return clean_text


def get_model_status() -> dict:
    status = _model_status
    return {
        "status": status,
        "loaded": status == "loaded",
        "loading": status == "checking",
        "failed": status == "failed",
        "unavailable": status == "unavailable",
        "provider": "ollama",
        "host": OLLAMA_HOST,
        "model_ref": OLLAMA_MODEL,
        "model_ref_exists": status == "loaded",
        "local_files_only": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "timeout_seconds": OLLAMA_TIMEOUT_SECONDS,
        "last_load_attempt": _last_load_attempt,
        "last_load_error": _last_load_error,
        "last_generation": _last_generation_info,
    }
