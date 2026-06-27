"""Ingestion package — exposes a single run_ingestion_pipeline() entry point."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from loguru import logger

from ingestion.chunker import chunk_documents, clean_documents
from ingestion.embedder import ingest_documents
from ingestion.loader import load_directory, load_sample_documents, load_url


def run_ingestion_pipeline(
    source: str | None = None,
    use_sample: bool = False,
    original_filename: str | None = None,
) -> VectorStore:
    """
    Full ingestion pipeline: load → clean → chunk → embed → store.

    Args:
        source: Path to a file/directory OR a URL to scrape.
                Pass None with use_sample=True to use built-in demo docs.
        use_sample: If True, ignore source and load built-in sample documents.
        original_filename: Clean name to associate with this source (avoids exposing temp folders).

    Returns:
        Populated VectorStore ready for retrieval.
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    if use_sample or source is None:
        raw_docs = load_sample_documents()
    elif source.startswith("http://") or source.startswith("https://"):
        raw_docs = load_url(source)
    else:
        from pathlib import Path
        p = Path(source)
        if p.is_dir():
            raw_docs = load_directory(p)
        elif p.suffix.lower() == ".pdf":
            from ingestion.loader import load_pdf
            raw_docs = load_pdf(p)
        elif p.suffix.lower() in {".docx", ".doc"}:
            from ingestion.loader import load_docx
            raw_docs = load_docx(p)
        else:
            from ingestion.loader import load_text
            raw_docs = load_text(p)

    # Clean up file source metadata to avoid absolute temporary paths
    import os
    for doc in raw_docs:
        if original_filename:
            doc.metadata["source"] = original_filename
        else:
            source_val = doc.metadata.get("source", "")
            if source_val and ("/" in source_val or "\\" in source_val):
                doc.metadata["source"] = os.path.basename(source_val)

    logger.info(f"Loaded {len(raw_docs)} raw documents")

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    cleaned = clean_documents(raw_docs)

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    chunks = chunk_documents(cleaned)

    # ── 4. Embed + Store ──────────────────────────────────────────────────────
    vector_store = ingest_documents(chunks)

    logger.info("Pipeline complete — vector store ready")
    return vector_store


__all__ = ["run_ingestion_pipeline"]
