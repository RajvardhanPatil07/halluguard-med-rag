# HalluGuard-Med Medical Corpus

This folder is the future replacement knowledge layer for graph-derived retrieval.
It is not wired into `backend/retrieval.py` yet, so the current system remains unchanged.

## Layout

- `raw/medlineplus/` stores MedlinePlus XML exports.
- `raw/who/` is reserved for WHO public health pages and fact sheets.
- `raw/nih_medgen/` is reserved for NIH/NCBI MedGen alias and identifier data.
- `processed/corpus_chunks.jsonl` is the normalized retrieval corpus.
- `indexes/` stores standalone BM25 and Qdrant indexing manifests.

## Commands

Initialize folders and metadata:

```powershell
python -m backend.corpus_ingest init
```

Ingest a MedlinePlus XML file:

```powershell
python -m backend.corpus_ingest ingest-medlineplus --xml medical_corpus\raw\medlineplus\health_topics.xml
```

Build the future BM25 corpus index:

```powershell
python -m backend.corpus_ingest build-bm25
```

Index into a separate Qdrant collection that does not affect the current app:

```powershell
python -m backend.corpus_ingest index-qdrant --collection halluguard_med_corpus_v1
```

## Chunk Schema

Each line in `processed/corpus_chunks.jsonl` is a JSON object with at least:

- `chunk_id`
- `text`
- `source`
- `source_type`
- `condition`
- `section`
- `citation_id`

Additional citation metadata such as `source_url`, `title`, `aliases`, `tags`, and
`updated_at` is preserved in the same JSON object and will become Qdrant payload.
