"""
Local corpus loading and MedlinePlus XML normalization.

This module does not participate in the live retrieval path yet. It prepares
the corpus files that will later replace graph-derived chunks.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .corpus_chunker import chunks_from_medlineplus_topic
from .corpus_schema import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SOURCE_REGISTRY_PATH,
    INDEX_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    CorpusChunk,
    ensure_corpus_dirs,
    normalize_text,
    read_jsonl,
    write_jsonl,
    write_manifest,
)


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_text(" ".join(element.itertext()))


def _child_values(topic: ET.Element, tag_name: str, attr_name: str = "title") -> list[str]:
    values = []
    for child in topic.findall(tag_name):
        value = child.attrib.get(attr_name) or _element_text(child)
        if value:
            values.append(normalize_text(value))
    return values


def _mesh_ids(topic: ET.Element) -> list[str]:
    ids = []
    for descriptor in topic.findall(".//mesh-heading/descriptor"):
        mesh_id = descriptor.attrib.get("id")
        if mesh_id:
            ids.append(mesh_id)
    return ids


def parse_medlineplus_xml(xml_path: Path) -> list[dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    topics = []
    for topic in root.findall(".//health-topic"):
        title = topic.attrib.get("title") or _element_text(topic.find("title"))
        if not title:
            continue
        topics.append({
            "title": normalize_text(title),
            "url": topic.attrib.get("url") or topic.findtext("url") or "",
            "updated_at": topic.attrib.get("date-created") or topic.attrib.get("date-updated") or "",
            "language_mapped_url": topic.attrib.get("language-mapped-url") or "",
            "summary": _element_text(topic.find("full-summary")),
            "aliases": _child_values(topic, "also-called"),
            "groups": _child_values(topic, "group"),
            "related_topics": _child_values(topic, "related-topic"),
            "mesh_ids": _mesh_ids(topic),
        })
    return topics


def ingest_medlineplus_xml(
    xml_path: Path,
    output_path: Path = DEFAULT_CHUNKS_PATH,
    append: bool = False,
) -> list[CorpusChunk]:
    ensure_corpus_dirs()
    topics = parse_medlineplus_xml(xml_path)
    new_chunks: list[CorpusChunk] = []
    for topic in topics:
        new_chunks.extend(chunks_from_medlineplus_topic(topic))

    chunks = read_jsonl(output_path) if append and output_path.exists() else []
    existing_ids = {chunk.chunk_id for chunk in chunks}
    chunks.extend(chunk for chunk in new_chunks if chunk.chunk_id not in existing_ids)
    write_jsonl(chunks, output_path)
    write_manifest(chunks, DEFAULT_MANIFEST_PATH)
    return chunks


def load_corpus_chunks(path: Path = DEFAULT_CHUNKS_PATH) -> list[CorpusChunk]:
    return read_jsonl(path)


def write_source_registry(path: Path = DEFAULT_SOURCE_REGISTRY_PATH) -> None:
    ensure_corpus_dirs()
    registry = {
        "primary_source": "MedlinePlus",
        "sources": [
            {
                "name": "MedlinePlus",
                "source_type": "consumer_health",
                "raw_dir": str(RAW_DIR / "medlineplus"),
                "notes": "Use exported MedlinePlus health topic XML as the seed corpus.",
            },
            {
                "name": "WHO",
                "source_type": "public_health",
                "raw_dir": str(RAW_DIR / "who"),
                "notes": "Reserved for later public health fact sheets and disease pages.",
            },
            {
                "name": "NIH MedGen",
                "source_type": "ontology",
                "raw_dir": str(RAW_DIR / "nih_medgen"),
                "notes": "Reserved for disease aliases and identifiers, not primary patient-facing answers.",
            },
        ],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=True, indent=2, sort_keys=True)


def corpus_status(path: Path = DEFAULT_CHUNKS_PATH) -> dict[str, Any]:
    ensure_corpus_dirs()
    chunks = read_jsonl(path)
    return {
        "corpus_root": str(path.parents[1]),
        "raw_dir": str(RAW_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "index_dir": str(INDEX_DIR),
        "chunks_path": str(path),
        "chunks_exists": path.exists(),
        "chunks_count": len(chunks),
        "sources": sorted({chunk.source for chunk in chunks}),
        "manifest_path": str(DEFAULT_MANIFEST_PATH),
        "source_registry_path": str(DEFAULT_SOURCE_REGISTRY_PATH),
    }
