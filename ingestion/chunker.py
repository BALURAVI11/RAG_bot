"""Chunker — splits documents into overlapping chunks for embedding."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

import config


def chunk_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Uses RecursiveCharacterTextSplitter which splits on:
    paragraph → sentence → word → character boundaries,
    preserving semantic coherence as much as possible.

    Args:
        documents: Raw documents from the loader.
        chunk_size: Maximum characters per chunk (default from config).
        chunk_overlap: Character overlap between adjacent chunks.

    Returns:
        List of chunked Documents with preserved + enriched metadata.
    """
    if not documents:
        logger.warning("chunk_documents called with empty document list")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata: add chunk index within each source doc
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[source] = idx + 1

    logger.info(
        f"Chunking complete: {len(documents)} docs → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def clean_text(text: str) -> str:
    """Remove excessive whitespace, null bytes, and common PDF artifacts."""
    import re

    text = text.replace("\x00", "")                        # null bytes
    text = re.sub(r"\n{3,}", "\n\n", text)                 # collapse 3+ newlines
    text = re.sub(r"[ \t]{2,}", " ", text)                 # collapse spaces/tabs
    text = re.sub(r"-\n([a-z])", r"\1", text)              # fix hyphenated line breaks
    return text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    """Apply clean_text to every document's page_content in place."""
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    # Filter out documents that are now empty or too short
    cleaned = [d for d in documents if len(d.page_content) > 50]
    removed = len(documents) - len(cleaned)
    if removed:
        logger.debug(f"Removed {removed} near-empty documents after cleaning")
    return cleaned
