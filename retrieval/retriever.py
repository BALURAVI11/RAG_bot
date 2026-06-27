"""Retriever — semantic similarity search over the vector store."""
from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from loguru import logger

import config


def retrieve(
    query: str,
    vector_store: VectorStore,
    top_k: int = config.TOP_K_RETRIEVAL,
) -> List[Tuple[Document, float]]:
    """
    Retrieve the top-K most semantically similar chunks for a query.

    Args:
        query: The user's question.
        vector_store: Populated VectorStore.
        top_k: Number of chunks to retrieve.

    Returns:
        List of (Document, similarity_score) tuples, highest score first.
    """
    logger.debug(f"Retrieving top-{top_k} chunks for: '{query[:80]}...' ")

    results: List[Tuple[Document, float]] = (
        vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    )

    logger.debug(
        f"Retrieved {len(results)} chunks | "
        f"scores: {[round(s, 3) for _, s in results]}"
    )
    return results


def get_retriever(vector_store: VectorStore, top_k: int = config.TOP_K_RETRIEVAL):
    """
    Return a LangChain-compatible Retriever object for use in LCEL chains.

    Useful when wiring directly into a LangChain RAG chain via `|` operators.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
