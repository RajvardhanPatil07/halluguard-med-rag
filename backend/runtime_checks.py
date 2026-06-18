from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None

try:
    from .medgemma import get_model_status
    from .ollama_client import check_model_available
    from .radiology_analyzer import get_radiology_status
    from .retrieval import QDRANT_AVAILABLE, QDRANT_API_KEY, QDRANT_URL, get_retrieval_status
    from .settings import GOOGLE_API_KEY, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS, get_environment_info
    from .verification import get_verification_status
except ImportError:
    from medgemma import get_model_status
    from ollama_client import check_model_available
    from radiology_analyzer import get_radiology_status
    from retrieval import QDRANT_AVAILABLE, QDRANT_API_KEY, QDRANT_URL, get_retrieval_status
    from settings import GOOGLE_API_KEY, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS, get_environment_info
    from verification import get_verification_status


def check_gpu() -> dict[str, Any]:
    if torch is None:
        return {"available": False, "error": "PyTorch is not installed"}

    try:
        cuda_available = bool(torch.cuda.is_available())
        devices = []
        for index in range(torch.cuda.device_count() if cuda_available else 0):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            })
        return {
            "available": cuda_available,
            "device_count": len(devices),
            "devices": devices,
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 2) if cuda_available else 0,
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 2) if cuda_available else 0,
            "error": None,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def check_qdrant() -> dict[str, Any]:
    if not QDRANT_AVAILABLE:
        return {
            "available": False,
            "url": QDRANT_URL,
            "error": "qdrant-client is not installed",
        }

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=2)
        collections = client.get_collections().collections
        return {
            "available": True,
            "url": QDRANT_URL,
            "collections": [collection.name for collection in collections],
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "url": QDRANT_URL,
            "error": str(exc),
        }


def check_ollama_medgemma() -> dict[str, Any]:
    status = check_model_available(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
        timeout=min(OLLAMA_TIMEOUT_SECONDS, 3.0),
    )
    return {
        "available": bool(status.get("available")),
        "host": OLLAMA_HOST,
        "model": OLLAMA_MODEL,
        "error": status.get("error"),
    }


def get_runtime_status() -> dict[str, Any]:
    env = get_environment_info()
    model = get_model_status()
    ollama = check_ollama_medgemma()
    model = {
        **model,
        "status": "loaded" if ollama["available"] else "failed",
        "loaded": bool(ollama["available"]),
        "failed": not bool(ollama["available"]),
        "unavailable": not bool(ollama["available"]),
        "model_ref_exists": bool(ollama["available"]),
        "last_load_error": ollama.get("error"),
    }
    retrieval = get_retrieval_status()
    qdrant = check_qdrant()
    verification = get_verification_status()
    radiology = get_radiology_status()
    gpu = check_gpu()

    missing_retrieval_deps = []
    if not retrieval.get("bm25_available"):
        missing_retrieval_deps.append("rank-bm25")
    if not retrieval.get("google_embeddings_available"):
        missing_retrieval_deps.append("google-generativeai")
    if not retrieval.get("qdrant_available"):
        missing_retrieval_deps.append("qdrant-client")

    checks = {
        "google_api_key": {
            "ok": bool(GOOGLE_API_KEY),
            "error": None if GOOGLE_API_KEY else "GOOGLE_API_KEY is not configured",
        },
        "ollama_medgemma": {
            "ok": ollama["available"],
            "error": ollama.get("error"),
        },
        "qdrant": {
            "ok": qdrant["available"],
            "error": qdrant.get("error"),
        },
        "hybrid_retrieval_dependencies": {
            "ok": not missing_retrieval_deps,
            "error": None if not missing_retrieval_deps else f"Missing dependencies: {', '.join(missing_retrieval_deps)}",
        },
        "nli": {
            "ok": bool(verification.get("nli_engine")),
            "error": verification.get("nli_load_error"),
        },
        "torchxrayvision": {
            "ok": radiology.get("available"),
            "error": radiology.get("last_load_error") or (
                None if radiology.get("available") else "torchxrayvision is not installed"
            ),
        },
        "gpu": {
            "ok": gpu.get("available"),
            "error": gpu.get("error"),
        },
    }

    failures = [
        {"check": name, "error": details.get("error")}
        for name, details in checks.items()
        if not details.get("ok")
    ]
    critical = {
        "google_api_key",
        "ollama_medgemma",
        "qdrant",
        "hybrid_retrieval_dependencies",
    }
    critical_failures = [failure for failure in failures if failure["check"] in critical]
    status = "ok"
    if critical_failures:
        status = "unavailable"
    elif failures:
        status = "degraded"

    return {
        "status": status,
        "checks": checks,
        "failures": failures,
        "environment": env,
        "medgemma": model,
        "ollama": ollama,
        "retrieval": retrieval,
        "qdrant": qdrant,
        "verification": verification,
        "radiology": radiology,
        "gpu": gpu,
        "project_root": str(Path(__file__).resolve().parents[1]),
    }
