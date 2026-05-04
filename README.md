---
title: Medical Assistant RAG
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Medical Assistant RAG

A Retrieval-Augmented Generation (RAG) system for medical question-answering, built with **Mistral-7B-Instruct**, **LangChain**, and **Chroma**. Supports three user personas with distinct response styles and LLM parameters.

https://huggingface.co/spaces/Shawmamish/MedicAI 

## Personas

| Persona | Style | Key Parameters |
|---------|-------|----------------|
| **Research Scientist** | Technical depth, mechanisms, clinical evidence | `max_tokens=1536`, `temp=0.1`, `k=10` |
| **Physician** | Bedside manner, empathetic, structured | `max_tokens=768`, `temp=0.35`, `k=5` |
| **Patient FAQ** | Plain language, 3–5 bullet points | `max_tokens=256`, `temp=0.2`, `k=3` |

## Project Structure

```
├── medical_assistant/      # Core Python package
│   ├── config/             # Settings (env vars) + persona parameters
│   ├── core/               # LLM, embeddings, vector store singletons
│   ├── rag/                # Document loader, retriever, RAG pipeline
│   ├── prompts/            # Persona-specific system prompt templates
│   └── evaluation/         # Groundedness & relevance raters
├── api/                    # FastAPI backend
│   ├── main.py             # App entry point + startup lifecycle
│   ├── routes/chat.py      # POST /chat  GET /health
│   └── schemas/models.py   # Pydantic request/response models
├── frontend/               # Vanilla HTML/CSS/JS UI
├── data/                   # PDF knowledge base (Merck Manuals)
├── models/                 # GGUF model cache (downloaded at runtime)
├── Dockerfile
└── docker-compose.yml
```

## Quick Start (local)

```bash
# 1. Clone and set up env
git clone <your-repo>
cd MedicalAssistant
cp .env.example .env

# 2. Edit .env — set N_GPU_LAYERS=35 if you have an RTX 3070
#    Leave at 0 for CPU-only

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run (model downloads automatically on first run)
uvicorn api.main:app --reload --port 7860
```

Open http://localhost:7860

## Docker

```bash
# CPU-only (default)
docker-compose up --build

# GPU (RTX 3070) — edit GPU=1 in docker-compose.yml build args
# and set N_GPU_LAYERS=35 in the environment section
docker-compose up --build
```

## GPU Setup (Windows 11 + RTX 3070)

1. Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) + Ubuntu
2. Install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) inside WSL2
3. In `.env`: set `N_GPU_LAYERS=35` and `MODEL_FILENAME=mistral-7b-instruct-v0.2.Q6_K.gguf` for best quality
4. Build with `GPU=1`: `docker build --build-arg GPU=1 -t medical-assistant .`

## API

```
POST /chat
  { "question": "What are the symptoms of appendicitis?", "persona": "doctor" }
  → { "answer": "...", "persona": "doctor", "sources": [...] }

GET /health
  → { "status": "ok", "model_loaded": true, "vector_store_ready": true }
```

## Deploying to Hugging Face Spaces

1. Create a new Space → SDK: **Docker**
2. Push this repo — the `README.md` YAML header is already configured
3. The model downloads automatically on first container start (~4 GB)
4. Free tier uses CPU (N_GPU_LAYERS=0) — upgrade to GPU Space for faster inference

## Previous Experimental Approach

Three progressively improved strategies were evaluated against five clinical test queries:

| # | Query Topic |
|---|-------------|
| 1 | Sepsis management protocol (critical care) |
| 2 | Appendicitis symptoms and surgical procedure |
| 3 | Sudden patchy hair loss — causes and treatments |
| 4 | Brain tissue injury — treatment recommendations |
| 5 | Leg fracture — field precautions and recovery |

### Strategy 1 — Baseline LLM

Direct model inference with no retrieval or system prompt. Responses were observed to be generic and often truncated, lacking specific clinical detail (dosages, procedure names, timing).

### Strategy 2 — LLM with Prompt Engineering

A structured system message instructed the model to act as a medical assistant delivering "accurate and compendious diagnoses and treatment plans." Responses became more specific and clinically toned. Several rounds of parameter tuning followed:

| Fine-tune | Parameters | Observation |
|-----------|-----------|-------------|
| Baseline | `max_tokens=256, temp=0, top_p=0.95, top_k=50` | Truncated answers |
| FT-1 | `max_tokens=512` | Addressed truncation; most answers completed |
| FT-2 | `max_tokens=512, temp=0.2` | Added warmth/tone while maintaining accuracy |
| FT-3 | `max_tokens=512, top_p=0.98, top_k=75` | Richer medical vocabulary; clinical feel |
| FT-4 | `max_tokens=512, top_k=15` | Tighter precision; strong readability balance |
| FT-5 | `max_tokens=1024` | No meaningful gain over 512 without retrieval |

### Strategy 3 — RAG (LLM + Vector Database)

Retrieved document chunks are injected as context alongside the user query before generation. This produced the strongest results even at baseline settings.

---

## RAG Data Pipeline

### Chunking

```python
RecursiveCharacterTextSplitter(
    encoding_name='cl100k_base',    # tiktoken
    chunk_size=512,                 # also tested: 1024
    chunk_overlap=50                # also tested: 100
)
```

### Embeddings

```python
SentenceTransformerEmbeddings(model_name='thenlper/gte-large')
# Embedding dimension: 1024
```

### Vector Database

**ChromaDB** — persisted locally; two stores created:
- `chunksize_512` (overlap=50)
- `chunksize_1024` (overlap=100)

### Retriever

```python
vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}   # also tested: 5, 10, 15
)
```

---

## RAG Fine-tuning Experiments

| Fine-tune | chunk_size | retriever k | max_tokens | temperature |         Notes         |
|-----------|------------|-------------|------------|-------------|-----------------------|
| Baseline  | 512        | 3           | 512        | 0           | Strong out-of-the-box |
| FT-1      | 512        | 3           | 1024       | 0, top_k=15 | Longer generation benefited RAG (unlike LLM-only) |
| FT-2      | 1024       | 5           | 256        | 0           | Answers became high-level; too much context diluted precision |
| FT-3      | 1024       | 5           | 512        | 0.1         | Best verbosity + tone; production candidate |
| FT-4      | 1024       | 10          | 512        | 0           | Good context saturation |
| FT-5      | 1024       | 15          | 512        | 0           | Solid context generation |

---

## Evaluation — LLM-as-a-Judge

The same Mistral model evaluated each RAG configuration using two scoring dimensions (1–5 scale):

- **Groundedness** — factual accuracy relative to retrieved document context
- **Relevance** — how directly the answer addresses the posed question

### Score Summary Across 5 Queries

|       Configuration           |      Avg Groundedness      |     Avg Relevance     |  
|-------------------------------|-----------------|---------------|
| RAG Baseline (chunk=512, k=3) |       4.6       |      4.6      |
| FT-1 (max_tokens=1024, top_k=15) |    4.0       |      4.0      |
| FT-2 (chunk=1024, k=5)        |       3.8       |      4.6      |
| **FT-3 (chunk=1024, k=5, temp=0.1)** | **4.8**  |    **5.0**    |
| FT-4 (chunk=1024, k=10)       |       4.6       |      4.8      |
| FT-5 (chunk=1024, k=15)       |       4.8       |      4.8      |

**Best overall**: FT-3 (`chunk_size=1024`, `max_tokens=512`, `temperature=0.1`) — consistently scored 5/5 on relevance across all queries.

---

> **Disclaimer**: For educational and research purposes only. Not a substitute for professional medical advice.
