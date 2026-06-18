"""
Shared schema for the future HalluGuard-Med medical corpus.

These records are intentionally compatible with the retrieval.EvidenceChunk
shape while carrying enough metadata for citations and future filtering.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "halluguard-med-corpus-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT / "medical_corpus"
RAW_DIR = CORPUS_ROOT / "raw"
PROCESSED_DIR = CORPUS_ROOT / "processed"
INDEX_DIR = CORPUS_ROOT / "indexes"
DEFAULT_CHUNKS_PATH = PROCESSED_DIR / "corpus_chunks.jsonl"
DEFAULT_MANIFEST_PATH = PROCESSED_DIR / "corpus_manifest.json"
DEFAULT_SOURCE_REGISTRY_PATH = PROCESSED_DIR / "source_registry.json"


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    text: str
    source: str
    source_type: str
    condition: str
    section: str
    citation_id: str
    title: str = ""
    source_url: str = ""
    language: str = "en"
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    identifiers: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""
    license_note: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_evidence_dict(self) -> dict[str, Any]:
        """Return the subset expected by the current retrieval pipeline."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "condition": self.condition,
            "section": self.section,
            "citation_id": self.citation_id,
        }


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: str
    generated_at: str
    chunks_count: int
    sources: list[str]
    chunks_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_corpus_dirs() -> None:
    for path in (
        CORPUS_ROOT,
        RAW_DIR,
        RAW_DIR / "medlineplus",
        RAW_DIR / "who",
        RAW_DIR / "nih_medgen",
        PROCESSED_DIR,
        INDEX_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "medical-topic"


def stable_chunk_id(source: str, title: str, section: str, index: int, text: str) -> str:
    raw = "|".join([source, title, section, str(index), text[:240]])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def stable_citation_id(source: str, title: str, section: str, index: int) -> str:
    prefix = "".join(part[:1] for part in re.findall(r"[A-Za-z0-9]+", source)).upper()
    prefix = prefix[:4] or "SRC"
    title_slug = slugify(title).replace("-", "_")[:32].upper()
    section_slug = slugify(section).replace("-", "_")[:20].upper()
    return f"{prefix}-{title_slug}-{section_slug}-{index:03d}"


def chunk_from_dict(data: dict[str, Any]) -> CorpusChunk:
    required = ("chunk_id", "text", "source", "source_type", "condition", "section", "citation_id")
    missing = [field_name for field_name in required if not data.get(field_name)]
    if missing:
        raise ValueError(f"Corpus chunk missing required fields: {', '.join(missing)}")
    return CorpusChunk(
        chunk_id=str(data["chunk_id"]),
        text=normalize_text(str(data["text"])),
        source=str(data["source"]),
        source_type=str(data["source_type"]),
        condition=str(data["condition"]),
        section=str(data["section"]),
        citation_id=str(data["citation_id"]),
        title=str(data.get("title") or ""),
        source_url=str(data.get("source_url") or ""),
        language=str(data.get("language") or "en"),
        tags=list(data.get("tags") or []),
        aliases=list(data.get("aliases") or []),
        identifiers=dict(data.get("identifiers") or {}),
        updated_at=str(data.get("updated_at") or ""),
        license_note=str(data.get("license_note") or ""),
        schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
    )


def validate_chunks(chunks: Iterable[CorpusChunk]) -> list[CorpusChunk]:
    seen: set[str] = set()
    valid = []
    for chunk in chunks:
        if not chunk.text:
            raise ValueError(f"Corpus chunk has empty text: {chunk.chunk_id}")
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate corpus chunk_id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        valid.append(chunk)
    return valid


def read_jsonl(path: Path = DEFAULT_CHUNKS_PATH) -> list[CorpusChunk]:
    if not path.exists():
        return []
    chunks = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            clean = line.strip()
            if not clean:
                continue
            try:
                chunks.append(chunk_from_dict(json.loads(clean)))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return validate_chunks(chunks)


def write_jsonl(chunks: Iterable[CorpusChunk], path: Path = DEFAULT_CHUNKS_PATH) -> None:
    ensure_corpus_dirs()
    valid = validate_chunks(chunks)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in valid:
            file.write(json.dumps(chunk.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")


def write_manifest(chunks: list[CorpusChunk], path: Path = DEFAULT_MANIFEST_PATH) -> CorpusManifest:
    ensure_corpus_dirs()
    manifest = CorpusManifest(
        schema_version=SCHEMA_VERSION,
        generated_at=now_iso(),
        chunks_count=len(chunks),
        sources=sorted({chunk.source for chunk in chunks}),
        chunks_path=str(DEFAULT_CHUNKS_PATH),
    )
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest.to_dict(), file, ensure_ascii=True, indent=2, sort_keys=True)
    return manifest
