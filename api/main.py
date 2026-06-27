"""FastAPI application — REST API for the RAG knowledge assistant."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

import config
from api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ResetResponse,
)
from chain.rag_chain import RAGChain
from ingestion import run_ingestion_pipeline


# ── App state (initialised at startup) ───────────────────────────────────────

_rag_chain: RAGChain | None = None
_start_time = time.time()
_query_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG chain with sample documents on startup."""
    global _rag_chain
    logger.info("Starting RAG Knowledge Assistant API...")
    vector_store = run_ingestion_pipeline(use_sample=True)
    _rag_chain = RAGChain(vector_store)
    logger.info("API ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="RAG Knowledge Assistant",
    description=(
        "Enterprise knowledge assistant powered by Retrieval-Augmented Generation. "
        "Upload documents and ask questions — answers are always grounded in your docs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth dependency ───────────────────────────────────────────────────────────

def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Simple API key check. Skip in development mode."""
    if config.APP_ENV == "development":
        return
    if x_api_key != config.API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check — returns uptime, query count, and config summary."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
        query_count=_query_count,
        vector_db=config.VECTOR_DB,
        llm_provider=config.LLM_PROVIDER,
        embedding_provider=config.EMBEDDING_PROVIDER,
        reranker_enabled=config.USE_RERANKER,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest, _: None = Depends(verify_api_key)):
    """
    Ask a question. Returns an answer grounded in the ingested documents,
    along with source citations and performance metadata.
    """
    global _query_count, _rag_chain
    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialised")

    logger.info(f"Query: '{request.question[:80]}'")
    result = _rag_chain.query(request.question)
    _query_count += 1

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieved_chunks=result["retrieved_chunks"],
        latency_ms=result["latency_ms"],
    )


@app.post("/query/stream", tags=["RAG"])
async def query_stream(request: QueryRequest, _: None = Depends(verify_api_key)):
    """
    Ask a question with streaming response — tokens arrive as Server-Sent Events.
    """
    global _rag_chain
    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialised")

    def generate():
        for token in _rag_chain.stream(request.question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = None,
    _: None = Depends(verify_api_key),
):
    """
    Ingest a document (PDF/DOCX/TXT upload) or a web URL into the knowledge base.
    After ingestion, queries will include this new content.
    """
    global _rag_chain

    if file is None and url is None:
        raise HTTPException(
            status_code=400, detail="Provide either a file upload or a url parameter"
        )

    import tempfile, os
    from pathlib import Path

    try:
        if file is not None:
            suffix = Path(file.filename or "doc.txt").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            vector_store = run_ingestion_pipeline(
                source=tmp_path,
                original_filename=file.filename
            )
            os.unlink(tmp_path)
            source_desc = file.filename or "uploaded file"
        else:
            vector_store = run_ingestion_pipeline(source=url)
            source_desc = url

        _rag_chain = RAGChain(vector_store)
        return IngestResponse(
            status="success",
            message=f"Ingested: {source_desc}. Knowledge base updated.",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", response_model=ResetResponse, tags=["System"])
async def reset_memory(_: None = Depends(verify_api_key)):
    """Clear conversation history and vector database. Useful for starting a new session."""
    global _rag_chain
    if _rag_chain:
        _rag_chain.reset_memory()
    from ingestion.embedder import clear_vector_store
    clear_vector_store()
    return ResetResponse(status="ok", message="Conversation history and vector store cleared")
