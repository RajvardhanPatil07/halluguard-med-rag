"""
Ollama client for local MedGemma generation.

This module contains only transport logic for the local Ollama API.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    pass


class OllamaUnavailableError(OllamaError):
    pass


class OllamaGenerationError(OllamaError):
    pass


def _generate_url(host: str) -> str:
    return f"{host.rstrip('/')}/api/generate"


def generate_text(
    *,
    prompt: str,
    images: list[str] | None = None,
    model: str = "medgemma",
    host: str = "http://localhost:11434",
    timeout: float = 120.0,
    max_new_tokens: int = 1024,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": 0,
            "top_p": 1,
        },
    }
    if images:
        payload["images"] = images
    request = Request(
        _generate_url(host),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise OllamaUnavailableError(
                f"Ollama model '{model}' is not available: {detail or exc.reason}"
            ) from exc
        raise OllamaGenerationError(
            f"Ollama returned HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise OllamaUnavailableError(
            f"Ollama request timed out after {timeout} seconds"
        ) from exc
    except URLError as exc:
        raise OllamaUnavailableError(
            f"Ollama is unavailable at {host}: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise OllamaUnavailableError(
            f"Ollama is unavailable at {host}: {exc}"
        ) from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaGenerationError("Ollama returned invalid JSON") from exc

    if data.get("error"):
        message = str(data["error"])
        if "not found" in message.lower() or "pull" in message.lower():
            raise OllamaUnavailableError(
                f"Ollama model '{model}' is not loaded: {message}"
            )
        raise OllamaGenerationError(f"Ollama generation failed: {message}")

    text = str(data.get("response") or "").strip()
    if not text:
        raise OllamaGenerationError("Ollama returned an empty response")

    return data


def check_model_available(
    *,
    model: str = "medgemma",
    host: str = "http://localhost:11434",
    timeout: float = 10.0,
) -> dict:
    try:
        result = generate_text(
            prompt="Respond with: ok",
            model=model,
            host=host,
            timeout=timeout,
            max_new_tokens=4,
        )
        return {
            "available": True,
            "error": None,
            "response_chars": len(str(result.get("response") or "")),
        }
    except OllamaError as exc:
        return {
            "available": False,
            "error": str(exc),
            "response_chars": 0,
        }
