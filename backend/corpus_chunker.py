"""
Chunking utilities for normalized medical source documents.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any

from .corpus_schema import (
    CorpusChunk,
    normalize_text,
    stable_chunk_id,
    stable_citation_id,
)


DEFAULT_MAX_CHARS = 1800
DEFAULT_MIN_CHARS = 220


def strip_markup(text: str | None) -> str:
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = unescape(clean)
    return normalize_text(clean)


def split_into_chunks(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    sentences = re.split(r"(?<=[.!?])\s+", clean)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = sentence

    if current:
        chunks.append(current)

    compacted: list[str] = []
    for chunk in chunks:
        if compacted and len(chunk) < min_chars and len(compacted[-1]) + len(chunk) + 1 <= max_chars:
            compacted[-1] = f"{compacted[-1]} {chunk}"
        else:
            compacted.append(chunk)
    return compacted


def _section_chunks(topic: dict[str, Any]) -> list[tuple[str, str]]:
    sections = []
    summary = strip_markup(topic.get("summary"))
    if summary:
        sections.append(("overview", summary))

    also_called = ", ".join(topic.get("aliases") or [])
    if also_called:
        sections.append(("aliases", f"Also called: {also_called}."))

    groups = ", ".join(topic.get("groups") or [])
    if groups:
        sections.append(("topic_groups", f"Health topic groups: {groups}."))

    related = ", ".join(topic.get("related_topics") or [])
    if related:
        sections.append(("related_topics", f"Related health topics: {related}."))

    return sections


def chunks_from_medlineplus_topic(topic: dict[str, Any]) -> list[CorpusChunk]:
    title = normalize_text(topic.get("title"))
    if not title:
        return []

    source = "MedlinePlus"
    source_type = "consumer_health"
    source_url = str(topic.get("url") or "")
    updated_at = str(topic.get("updated_at") or "")
    aliases = list(topic.get("aliases") or [])
    identifiers = {
        "mesh": list(topic.get("mesh_ids") or []),
        "language_mapped_url": topic.get("language_mapped_url") or "",
    }
    tags = sorted({
        title.lower(),
        *(alias.lower() for alias in aliases),
        *(group.lower() for group in topic.get("groups") or []),
    })

    chunks: list[CorpusChunk] = []
    local_index = 1
    for section, section_text in _section_chunks(topic):
        for part in split_into_chunks(section_text):
            chunk_id = stable_chunk_id(source, title, section, local_index, part)
            citation_id = stable_citation_id(source, title, section, local_index)
            chunks.append(CorpusChunk(
                chunk_id=chunk_id,
                text=part,
                source=source,
                source_type=source_type,
                condition=title.lower(),
                section=section,
                citation_id=citation_id,
                title=title,
                source_url=source_url,
                language="en",
                tags=tags,
                aliases=aliases,
                identifiers=identifiers,
                updated_at=updated_at,
                license_note="MedlinePlus public health topic metadata; preserve source URL in citations.",
            ))
            local_index += 1
    return chunks
