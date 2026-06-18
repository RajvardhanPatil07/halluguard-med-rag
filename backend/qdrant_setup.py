"""
Qdrant setup entry point for HalluGuard-Med Hybrid RAG.

Run from the project root after setting GOOGLE_API_KEY and starting Qdrant:
    python -m backend.qdrant_setup
"""

import json

try:
    from .retrieval import setup_qdrant_index
except ImportError:
    from retrieval import setup_qdrant_index


def main() -> None:
    result = setup_qdrant_index()
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
