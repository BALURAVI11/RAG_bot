"""Tests for the retrieval pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.documents import Document

from retrieval.reranker import rerank


class TestReranker:
    """Test reranker with USE_RERANKER=false (default) so no API keys needed."""

    def _make_docs(self, n=5):
        return [
            (
                Document(page_content=f"Document {i} about topic {i}", metadata={"id": i}),
                float(n - i) / n,  # descending similarity scores
            )
            for i in range(n)
        ]

    def test_rerank_returns_top_k(self):
        docs = self._make_docs(5)
        result = rerank("test query", docs, top_k=3)
        assert len(result) == 3

    def test_rerank_orders_by_score(self):
        docs = self._make_docs(5)
        result = rerank("test query", docs, top_k=5)
        # First result should have highest score (score = (5-0)/5 = 1.0)
        assert result[0].metadata["id"] == 0

    def test_rerank_empty_input(self):
        result = rerank("test query", [], top_k=3)
        assert result == []

    def test_rerank_fewer_docs_than_k(self):
        docs = self._make_docs(2)
        result = rerank("test query", docs, top_k=5)
        assert len(result) == 2


class TestEndToEnd:
    """Integration test: full ingestion → retrieval pipeline (no LLM)."""

    def test_ingest_and_retrieve(self, tmp_path):
        """Ingest sample docs, run a retrieval, get results back."""
        from ingestion import run_ingestion_pipeline
        from retrieval.retriever import retrieve

        vector_store = run_ingestion_pipeline(use_sample=True)
        results = retrieve("What is RAG?", vector_store, top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        # Each result is (Document, float)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert len(doc.page_content) > 0

    def test_retrieval_relevance(self):
        """Check that the most relevant chunk is returned for a specific query."""
        from ingestion import run_ingestion_pipeline
        from retrieval.retriever import retrieve

        vector_store = run_ingestion_pipeline(use_sample=True)
        results = retrieve("vector database pinecone chromadb", vector_store, top_k=3)

        top_doc = results[0][0]
        # The top result should mention vector databases
        assert any(
            kw in top_doc.page_content.lower()
            for kw in ["vector", "pinecone", "chromadb", "weaviate"]
        )
