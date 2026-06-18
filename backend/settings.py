import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "medgemma"


def is_kaggle() -> bool:
    """Returns True if code is running inside a Kaggle notebook."""
    return (
        os.path.exists("/kaggle")
        or os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None
        or os.getenv("KAGGLE_URL_BASE") is not None
    )


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


OLLAMA_HOST = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
MAX_NEW_TOKENS = int(os.getenv("MEDGEMMA_MAX_NEW_TOKENS", "384"))
MAX_GENERATION_ROUNDS = int(os.getenv("MEDGEMMA_MAX_GENERATION_ROUNDS", "1"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
GOOGLE_EMBEDDING_DIM = int(os.getenv("GOOGLE_EMBEDDING_DIM", "768"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "halluguard_med_chunks")
RAG_TOP_K_EACH = int(os.getenv("RAG_TOP_K_EACH", "8"))
RAG_TOP_K_FINAL = int(os.getenv("RAG_TOP_K_FINAL", "5"))
RAG_RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RAG_VERIFIED_THRESHOLD = float(os.getenv("RAG_VERIFIED_THRESHOLD", "0.40"))


def get_model_ref() -> str:
    """
    Returns the MedGemma model reference used by the runtime provider.
    """
    return OLLAMA_MODEL


def should_use_local_files_only() -> bool:
    """
    Ollama generation is local-service based, not HuggingFace file based.
    """
    return False


def get_environment_info() -> dict:
    """
    Returns a summary of current environment for debugging.
    Visible in /api/model-status response.
    """
    return {
        "is_kaggle": is_kaggle(),
        "model_provider": "ollama",
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "ollama_timeout_seconds": OLLAMA_TIMEOUT_SECONDS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_generation_rounds": MAX_GENERATION_ROUNDS,
        "google_embedding_model": GOOGLE_EMBEDDING_MODEL,
        "google_embedding_dim": GOOGLE_EMBEDDING_DIM,
        "qdrant_url": QDRANT_URL,
        "qdrant_collection": QDRANT_COLLECTION,
    }
