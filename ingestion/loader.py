"""Document loader — ingests PDF, DOCX, web pages, and plain text into LangChain Documents."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from loguru import logger

import config


def load_pdf(file_path: str | Path) -> List[Document]:
    """Load a PDF file and return a list of Documents (one per page)."""
    from langchain_community.document_loaders import PyPDFLoader

    path = str(file_path)
    logger.info(f"Loading PDF: {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()
    logger.info(f"  → {len(docs)} pages loaded")
    return docs


def load_docx(file_path: str | Path) -> List[Document]:
    """Load a Word document."""
    from langchain_community.document_loaders import Docx2txtLoader

    path = str(file_path)
    logger.info(f"Loading DOCX: {path}")
    loader = Docx2txtLoader(path)
    docs = loader.load()
    logger.info(f"  → {len(docs)} document(s) loaded")
    return docs


def load_text(file_path: str | Path) -> List[Document]:
    """Load a plain-text file."""
    from langchain_community.document_loaders import TextLoader

    path = str(file_path)
    logger.info(f"Loading text: {path}")
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    return docs


def load_url(url: str) -> List[Document]:
    """Scrape a web page and return its text as Documents."""
    from langchain_community.document_loaders import WebBaseLoader

    logger.info(f"Loading URL: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    logger.info(f"  → {len(docs)} page(s) scraped")
    return docs


def load_directory(directory: str | Path, glob: str = "**/*") -> List[Document]:
    """Recursively load all supported files from a directory."""
    from langchain_community.document_loaders import DirectoryLoader

    path = str(directory)
    logger.info(f"Loading directory: {path} (glob={glob})")
    all_docs: List[Document] = []

    dir_path = Path(directory)
    for file in dir_path.rglob("*"):
        if file.suffix.lower() == ".pdf":
            all_docs.extend(load_pdf(file))
        elif file.suffix.lower() in {".docx", ".doc"}:
            all_docs.extend(load_docx(file))
        elif file.suffix.lower() in {".txt", ".md"}:
            all_docs.extend(load_text(file))

    logger.info(f"Directory load complete: {len(all_docs)} total documents")
    return all_docs


def load_sample_documents() -> List[Document]:
    """
    Load built-in sample documents for demo / testing when no real corpus is provided.
    Returns synthetic documents about AI topics — good for showcasing the system.
    """
    logger.info("Loading built-in sample documents")
    samples = [
        Document(
            page_content=(
                "Retrieval-Augmented Generation (RAG) is an AI framework that combines "
                "information retrieval with large language model generation. Instead of relying "
                "solely on parametric knowledge stored in model weights, RAG systems retrieve "
                "relevant context from an external knowledge base at query time and inject it "
                "into the prompt before generating a response. This approach significantly "
                "reduces hallucinations and keeps answers grounded in verifiable sources."
            ),
            metadata={"source": "sample", "topic": "RAG Overview", "page": 1},
        ),
        Document(
            page_content=(
                "Vector databases store high-dimensional embeddings and support approximate "
                "nearest-neighbour (ANN) search. Popular options include Pinecone (managed, "
                "cloud-native), Weaviate (open-source with hybrid search), ChromaDB (lightweight, "
                "local-first), and Qdrant (Rust-based, high performance). The choice depends on "
                "scale, latency requirements, and whether a managed service is preferred. "
                "For production systems handling millions of vectors, Pinecone or Weaviate are "
                "typically recommended; for local development, ChromaDB requires no infrastructure."
            ),
            metadata={"source": "sample", "topic": "Vector Databases", "page": 1},
        ),
        Document(
            page_content=(
                "Chunking strategy significantly affects RAG quality. Recursive character splitting "
                "divides documents by paragraph, then sentence, then character boundaries — "
                "preserving semantic coherence. Chunk size of 256–512 tokens balances context "
                "completeness with retrieval precision. Overlap of 10–15% between adjacent chunks "
                "prevents important context from falling on a boundary split. Semantic chunking "
                "(splitting at embedding similarity drops) is more sophisticated but computationally "
                "expensive. For most use cases, recursive splitting with 512-token chunks and "
                "50-token overlap is the recommended starting point."
            ),
            metadata={"source": "sample", "topic": "Chunking Strategy", "page": 1},
        ),
        Document(
            page_content=(
                "LangChain is a framework for building LLM-powered applications. Its core "
                "abstractions include: Chains (composable pipelines of LLM calls and tool use), "
                "Agents (LLMs that decide which tools to invoke), Memory (conversation history "
                "management), and Retrievers (interfaces to vector stores and search engines). "
                "LangChain Expression Language (LCEL) enables declarative composition of chains "
                "using the pipe operator (|), making it easy to build, trace, and deploy complex "
                "multi-step AI workflows."
            ),
            metadata={"source": "sample", "topic": "LangChain", "page": 1},
        ),
        Document(
            page_content=(
                "RAGAS (Retrieval Augmented Generation Assessment) is an evaluation framework "
                "for RAG systems. It measures four key metrics without requiring human annotations: "
                "Faithfulness (does the answer follow from the retrieved context?), Answer Relevance "
                "(does the answer address the question?), Context Recall (did retrieval surface the "
                "right chunks?), and Context Precision (are the retrieved chunks relevant?). "
                "Target scores: Faithfulness > 0.85, Answer Relevance > 0.80, Context Recall > 0.75. "
                "RAGAS uses an LLM-as-judge approach, making it cost-effective to run at scale."
            ),
            metadata={"source": "sample", "topic": "RAGAS Evaluation", "page": 1},
        ),
        Document(
            page_content=(
                "Prompt engineering for RAG systems involves three key components: a system prompt "
                "that defines the assistant's role and instructs it to answer only from provided "
                "context, retrieved document chunks injected as context, and the user's question. "
                "Best practices: instruct the model to cite sources by name, ask it to say "
                "'I don't have information on that' when context is insufficient (preventing "
                "hallucination), and limit context to 3–5 top chunks to stay within token limits. "
                "Chain-of-thought prompting can improve reasoning on complex multi-hop questions."
            ),
            metadata={"source": "sample", "topic": "Prompt Engineering", "page": 1},
        ),
        Document(
            page_content=(
                "MLOps for RAG systems involves several production concerns: model versioning "
                "(track which embedding model and LLM version is in use), data versioning "
                "(version the vector store index alongside the document corpus), monitoring "
                "(log query latency, token usage, and retrieval quality metrics), and CI/CD "
                "(automate re-ingestion when documents are updated, re-run RAGAS evaluations "
                "on each deployment). Tools: MLflow for experiment tracking, GitHub Actions "
                "for CI/CD, and AWS CloudWatch or Grafana for production monitoring."
            ),
            metadata={"source": "sample", "topic": "MLOps", "page": 1},
        ),
        Document(
            page_content=(
                "Agentic AI systems extend basic RAG by giving the LLM the ability to decide "
                "what actions to take. Instead of a fixed retrieve-then-generate pipeline, an "
                "agent can call multiple tools (search, calculator, code executor, API calls) "
                "in a loop until it has enough information to answer. Frameworks include "
                "LangGraph (graph-based agent orchestration), AutoGen (multi-agent conversations), "
                "and CrewAI (role-based agent teams). Agentic RAG can handle complex multi-hop "
                "questions that require synthesizing information from multiple sources."
            ),
            metadata={"source": "sample", "topic": "Agentic AI", "page": 1},
        ),
    ]
    logger.info(f"  → {len(samples)} sample documents loaded")
    return samples
