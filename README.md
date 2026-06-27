# RAG Knowledge Assistant

> Enterprise document Q&A powered by Retrieval-Augmented Generation (RAG)

Ask questions about your documents in plain English. Every answer is grounded in retrieved context and comes with source citations — no hallucinations.

---

## Architecture

```
Documents (PDF/DOCX/Web)
    ↓
[Loader] → [Cleaner] → [Chunker] → [Embedder] → [Vector Store]
                                                       ↓
User Query → [Query Embedder] → [Retriever] → [Reranker] → [LLM] → Answer + Citations
```

**Ingestion pipeline**: documents are loaded, cleaned, split into 512-char chunks, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB (or Pinecone in production).

**Query pipeline**: the user's question is embedded, the top-5 similar chunks are retrieved, optionally re-ranked by Cohere, injected into a structured prompt, and sent to Claude/GPT-4o to generate a grounded answer.

---

## Quickstart (local dev, no API keys needed)

```bash
# 1. Clone and install
git clone https://github.com/your-username/rag-knowledge-assistant
cd rag-knowledge-assistant
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: add your LLM API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)
# Everything else works with defaults (HuggingFace embeddings + ChromaDB)

# 3. Launch the chat UI
streamlit run ui/app.py

# 4. Or run the REST API
uvicorn api.main:app --reload
# → API docs at http://localhost:8000/docs
```

---

## Tech stack

| Layer | Choice | Alternatives |
|---|---|---|
| Language | Python 3.11 | — |
| RAG framework | LangChain | LlamaIndex |
| Embeddings | `all-MiniLM-L6-v2` (free) | `text-embedding-3-small` (OpenAI) |
| Vector DB | ChromaDB (dev) / Pinecone (prod) | Weaviate, Qdrant |
| LLM | Claude 3 Haiku | GPT-4o-mini, Llama 3 |
| Reranker | Cohere Rerank v3 | cross-encoder (local) |
| API | FastAPI + Mangum | Flask |
| UI | Streamlit | Gradio, Next.js |
| Cloud | AWS Lambda | Azure Container Apps |
| CI/CD | GitHub Actions | — |
| Evaluation | RAGAS | TruLens |

---

## Project structure

```
rag-knowledge-assistant/
├── config.py               # Central config from .env
├── ingestion/
│   ├── loader.py           # PDF, DOCX, web, text loaders
│   ├── chunker.py          # Recursive splitting + cleaning
│   └── embedder.py         # Embedding model + vector store factory
├── retrieval/
│   ├── retriever.py        # Semantic similarity search
│   └── reranker.py         # Cohere / cross-encoder reranking
├── chain/
│   ├── prompt.py           # System prompt + LCEL templates
│   ├── rag_chain.py        # Full query pipeline + streaming
│   └── memory.py           # Sliding-window conversation memory
├── api/
│   ├── main.py             # FastAPI app (query, ingest, health)
│   └── schemas.py          # Pydantic request/response models
├── ui/
│   └── app.py              # Streamlit chat interface
├── evaluation/
│   └── ragas_eval.py       # RAGAS metrics evaluation
├── tests/
│   ├── test_ingestion.py
│   └── test_retrieval.py
├── .github/workflows/
│   └── deploy.yml          # CI/CD: lint → test → build → deploy
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | System status, uptime, config |
| POST | `/query` | Ask a question, returns answer + citations |
| POST | `/query/stream` | Same, but streams tokens (SSE) |
| POST | `/ingest` | Upload a file or URL to the knowledge base |
| POST | `/reset` | Clear conversation history |

Example:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?"}'
```

---

## Evaluation results (RAGAS)

Run against 5 questions on the built-in sample corpus:

| Metric | Score | Target |
|---|---|---|
| Faithfulness | — | ≥ 0.85 |
| Answer relevancy | — | ≥ 0.80 |
| Context recall | — | ≥ 0.75 |
| Context precision | — | ≥ 0.70 |

Run the evaluation yourself:
```bash
python evaluation/ragas_eval.py
```

---

## Configuration

All settings are in `.env`. Key options:

```bash
LLM_PROVIDER=anthropic          # anthropic | openai
VECTOR_DB=chroma                # chroma | pinecone
EMBEDDING_PROVIDER=huggingface  # huggingface | openai
USE_RERANKER=false              # true requires COHERE_API_KEY
CHUNK_SIZE=512
TOP_K_RETRIEVAL=5
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Deployment (AWS Lambda)

1. Add AWS secrets to GitHub repository settings
2. Push to `main` → GitHub Actions runs lint → test → build → deploy automatically
3. The Lambda function is updated with the new container image

For Azure Container Apps, change the deploy step in `.github/workflows/deploy.yml` to use `az containerapp update`.

---

## Built for

This project demonstrates skills from the **Senior Consultant – AI & Analytics** job description:
- LLM + GenAI development (RAG pipeline with Claude/GPT-4)
- Agentic AI frameworks (LangChain LCEL chains)
- Cloud deployment (AWS Lambda + Docker)
- MLOps (CI/CD, RAGAS evaluation, monitoring)
- NLP and vector search
- Python, FastAPI, REST APIs
- Data storytelling (source citations, latency metrics)
# RAG_bot
