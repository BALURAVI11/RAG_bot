"""Tests for the ingestion pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.documents import Document

from ingestion.chunker import chunk_documents, clean_documents, clean_text
from ingestion.loader import load_sample_documents


class TestLoader:
    def test_load_sample_documents_returns_documents(self):
        docs = load_sample_documents()
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_sample_documents_have_metadata(self):
        docs = load_sample_documents()
        for doc in docs:
            assert "source" in doc.metadata
            assert "topic" in doc.metadata

    def test_sample_documents_have_content(self):
        docs = load_sample_documents()
        for doc in docs:
            assert len(doc.page_content) > 100


class TestCleaner:
    def test_clean_text_removes_null_bytes(self):
        result = clean_text("hello\x00world")
        assert "\x00" not in result
        assert "helloworld" in result

    def test_clean_text_collapses_whitespace(self):
        result = clean_text("hello   world")
        assert "  " not in result

    def test_clean_text_collapses_newlines(self):
        result = clean_text("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_clean_documents_removes_short_docs(self):
        docs = [
            Document(page_content="x", metadata={}),        # too short
            Document(page_content="a" * 100, metadata={}),  # ok
        ]
        cleaned = clean_documents(docs)
        assert len(cleaned) == 1

    def test_clean_documents_preserves_good_docs(self):
        docs = load_sample_documents()
        cleaned = clean_documents(docs)
        assert len(cleaned) == len(docs)  # all sample docs are long enough


class TestChunker:
    def test_chunk_documents_produces_chunks(self):
        docs = load_sample_documents()
        chunks = chunk_documents(docs)
        assert len(chunks) >= len(docs)

    def test_chunks_have_chunk_index_metadata(self):
        docs = load_sample_documents()
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata

    def test_chunks_respect_size_limit(self):
        docs = load_sample_documents()
        chunks = chunk_documents(docs, chunk_size=200)
        for chunk in chunks:
            assert len(chunk.page_content) <= 250  # allow slight overshoot at word boundaries

    def test_empty_input_returns_empty(self):
        result = chunk_documents([])
        assert result == []
