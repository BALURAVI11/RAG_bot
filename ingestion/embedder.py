"""Embedder — generates embeddings and upserts chunks into the vector store."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger

import config


# ── Embedding model factory ───────────────────────────────────────────────────

def get_embedding_model() -> Embeddings:
    """
    Return the configured embedding model.

    Supported providers:
      - huggingface: free, runs locally, good for dev (default)
      - openai: cloud, higher quality, costs money
    """
    if config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Using OpenAI embeddings: {config.EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )

    # Default: HuggingFace sentence-transformers (free, local)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info(f"Using HuggingFace embeddings: {config.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Vector store factory ──────────────────────────────────────────────────────

def get_vector_store(embedding_model: Embeddings | None = None) -> VectorStore:
    """
    Return the configured vector store (existing index, not ingested yet).

    Supported backends:
      - chroma: local, no infra, great for dev (default)
      - pinecone: managed cloud, production-grade
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    if config.VECTOR_DB == "pinecone":
        return _get_pinecone_store(embedding_model)

    return _get_chroma_store(embedding_model)


def _get_chroma_store(embedding_model: Embeddings) -> VectorStore:
    from langchain_community.vectorstores import Chroma
    logger.info(f"Using ChromaDB at: {config.CHROMA_PERSIST_DIR}")
    return Chroma(
        collection_name="rag_knowledge_base",
        embedding_function=embedding_model,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
    )


def _get_pinecone_store(embedding_model: Embeddings) -> VectorStore:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index_names = [i.name for i in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in index_names:
        logger.info(f"Creating Pinecone index: {config.PINECONE_INDEX_NAME}")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=384,   # all-MiniLM-L6-v2 dimension; change to 1536 for OpenAI
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    logger.info(f"Using Pinecone index: {config.PINECONE_INDEX_NAME}")
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embedding_model,
        pinecone_api_key=config.PINECONE_API_KEY,
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_documents(
    chunks: List[Document],
    batch_size: int = 100,
) -> VectorStore:
    """
    Embed chunks and upsert them into the configured vector store.

    Args:
        chunks: Pre-chunked LangChain Documents.
        batch_size: Number of chunks to embed and upsert per batch.

    Returns:
        The populated VectorStore instance.
    """
    if not chunks:
        raise ValueError("No chunks to ingest — check your document loader")

    logger.info(f"Starting ingestion: {len(chunks)} chunks, batch_size={batch_size}")
    embedding_model = get_embedding_model()

    if config.VECTOR_DB == "pinecone":
        # Pinecone: add in batches to avoid request size limits
        store = _get_pinecone_store(embedding_model)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            store.add_documents(batch)
            logger.debug(f"  Upserted batch {i//batch_size + 1} ({len(batch)} chunks)")
    else:
        # ChromaDB: from_documents handles batching internally
        from langchain_community.vectorstores import Chroma
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="rag_knowledge_base",
            persist_directory=str(config.CHROMA_PERSIST_DIR),
        )

    logger.info(f"Ingestion complete: {len(chunks)} chunks stored in {config.VECTOR_DB}")
    return store


def clear_vector_store() -> None:
    """Clear all documents from the vector store by deleting the local database directory or Pinecone vectors."""
    import shutil
    import os

    if config.VECTOR_DB == "pinecone":
        from pinecone import Pinecone
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index_names = [i.name for i in pc.list_indexes()]
        if config.PINECONE_INDEX_NAME in index_names:
            logger.info(f"Clearing Pinecone index: {config.PINECONE_INDEX_NAME}")
            index = pc.Index(config.PINECONE_INDEX_NAME)
            index.delete(delete_all=True)
    else:
        if os.path.exists(config.CHROMA_PERSIST_DIR):
            logger.info(f"Clearing ChromaDB directory: {config.CHROMA_PERSIST_DIR}")
            shutil.rmtree(config.CHROMA_PERSIST_DIR)
