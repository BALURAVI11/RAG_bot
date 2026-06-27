"""Reranker — re-scores retrieved chunks for improved relevance."""
from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from loguru import logger

import config


def rerank(
    query: str,
    docs_with_scores: List[Tuple[Document, float]],
    top_k: int = config.TOP_K_RERANK,
) -> List[Document]:
    """
    Re-rank retrieved chunks using a cross-encoder or Cohere Rerank.

    Falls back to returning the top-k by original similarity score if
    no reranker is configured (USE_RERANKER=false in .env).

    Args:
        query: The user's question.
        docs_with_scores: Output from retriever.retrieve().
        top_k: Number of chunks to return after reranking.

    Returns:
        Reranked list of Documents (best first), truncated to top_k.
    """
    if not docs_with_scores:
        return []

    if not config.USE_RERANKER:
        # No reranker: return top-k by original retrieval score
        logger.debug("Reranker disabled — using retrieval scores directly")
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:top_k]]

    if config.COHERE_API_KEY:
        return _rerank_cohere(query, docs_with_scores, top_k)

    return _rerank_cross_encoder(query, docs_with_scores, top_k)


def _rerank_cohere(
    query: str,
    docs_with_scores: List[Tuple[Document, float]],
    top_k: int,
) -> List[Document]:
    """Rerank using Cohere Rerank API (best quality, requires API key)."""
    import cohere

    client = cohere.Client(config.COHERE_API_KEY)
    docs = [doc for doc, _ in docs_with_scores]
    texts = [doc.page_content for doc in docs]

    logger.debug(f"Cohere reranking {len(texts)} chunks")
    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_k,
    )

    reranked = [docs[result.index] for result in response.results]
    logger.debug(f"Cohere rerank complete: {len(reranked)} chunks returned")
    return reranked


def _rerank_cross_encoder(
    query: str,
    docs_with_scores: List[Tuple[Document, float]],
    top_k: int,
) -> List[Document]:
    """Rerank using a local HuggingFace cross-encoder (no API key needed)."""
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        docs = [doc for doc, _ in docs_with_scores]
        pairs = [(query, doc.page_content) for doc in docs]

        logger.debug(f"Cross-encoder reranking {len(pairs)} chunks")
        scores = model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    except ImportError:
        logger.warning(
            "sentence-transformers not available for cross-encoder reranking. "
            "Falling back to original retrieval order."
        )
        return [doc for doc, _ in docs_with_scores[:top_k]]
