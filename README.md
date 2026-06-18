# HalluGuard-Med RAG

Private FastAPI project for a medical RAG assistant that combines hybrid retrieval,
MedGemma generation through Ollama, clinical safety checks, PDF reports, and optional
chest X-ray screening with TorchXRayVision.

This project is for research and educational use. It is not a medical device and must
not be used as a substitute for professional medical judgment, diagnosis, treatment, or
emergency care.

## What It Does

- Serves a FastAPI backend with `/api/chat`, health checks, model status, retrieval
  status, and PDF report generation.
- Retrieves medical evidence from a local corpus using BM25/Qdrant-backed RAG assets.
- Calls MedGemma through an Ollama-compatible local service.
- Screens uploaded chest X-ray images with TorchXRayVision and treats image output as
  supportive screening context, not diagnosis.
- Runs post-generation safety, claim verification, risk scoring, citation handling, and
  answer policy enforcement.
- Keeps a simple static frontend under `frontend/`.

## Repository Layout

```text
backend/                 FastAPI app, RAG, model, safety, verification, and report code
frontend/                Static HTML/CSS/JS frontend
medical_corpus/          Source corpus, processed chunks, and retrieval indexes
tests/                   Focused safety-gate tests
requirements.txt         Python dependencies
```

Runtime outputs such as `.env`, logs, generated reports, images, and cache folders are
ignored by git.

## Prerequisites

- Python 3.10 or newer
- Ollama running locally with the configured MedGemma model available
- Qdrant available locally or remotely
- A Google API key for embedding-backed retrieval

Useful local defaults:

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=medgemma
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=halluguard_med_chunks
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_google_api_key
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=medgemma
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=halluguard_med_chunks
```

Start Qdrant locally if needed:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

Start or verify Ollama separately:

```bash
ollama serve
ollama list
```

## Run The App

Start the backend:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Open the frontend from `frontend/index.html` or `frontend/assistant.html`.

## API Endpoints

- `GET /` - basic backend landing page
- `GET /health` - compact health status
- `GET /api/health` - full runtime health status
- `GET /api/model-status` - MedGemma, radiology, verification, safety, and RAG status
- `GET /api/retrieval-status` - Qdrant and embedding retrieval status
- `POST /api/chat` - medical chat endpoint with `query` and optional `image`
- `POST /api/download-report` - generate a PDF report from a chat response payload

Example chat request:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -F "query=What are warning signs of appendicitis?"
```

With an image:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -F "query=Can you screen this chest X-ray?" \
  -F "image=@/path/to/chest-xray.png"
```

## Corpus And Retrieval

The `medical_corpus/` folder contains raw sources, normalized chunks, and index
artifacts. The corpus tooling can initialize folders, ingest MedlinePlus XML, build a
BM25 corpus file, and index chunks into Qdrant.

Common commands:

```bash
python -m backend.corpus_ingest init
python -m backend.corpus_ingest build-bm25
python -m backend.corpus_ingest index-qdrant --collection halluguard_med_corpus_v1
```

## Tests

Run the focused unit tests:

```bash
python3 -m unittest discover -s tests
```

Compile-check the backend:

```bash
python3 -m compileall -q backend
```

## Safety Notes

- Answers are generated only when retrieval and MedGemma are available.
- The system should not invent substitute medical text if the model backend fails.
- X-ray findings are screening signals only and require clinician review.
- Emergency symptoms should be treated as urgent real-world care situations.
- Generated reports are summaries of the assistant output, not clinical records.
