"""RAG chain — the full query → retrieve → generate pipeline."""
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from loguru import logger

import config
from chain.memory import ConversationMemory
from chain.prompt import RAG_PROMPT, format_docs
from retrieval.reranker import rerank
from retrieval.retriever import retrieve


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_llm(streaming: bool = False) -> BaseChatModel:
    """Return the configured LLM."""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        logger.info("Using OpenAI LLM: gpt-4o-mini")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=streaming,
            openai_api_key=config.OPENAI_API_KEY,
        )

    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("Using Gemini LLM: gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=config.GOOGLE_API_KEY,
        )

    # Default: Anthropic Claude
    from langchain_anthropic import ChatAnthropic
    logger.info("Using Anthropic LLM: claude-3-haiku-20240307")
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0,
        streaming=streaming,
        anthropic_api_key=config.ANTHROPIC_API_KEY,
        max_tokens=1024,
    )


# ── RAG Chain ─────────────────────────────────────────────────────────────────

class RAGChain:
    """
    End-to-end RAG pipeline with conversation memory.

    Flow per query:
      1. Retrieve top-K chunks from vector store
      2. Rerank chunks (if enabled)
      3. Format chunks into context string
      4. Build prompt with system message + history + question
      5. Call LLM and return answer + source citations
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.memory = ConversationMemory(max_history=config.MAX_CONVERSATION_HISTORY)
        self.llm = get_llm(streaming=False)
        self.streaming_llm = get_llm(streaming=True)
        logger.info("RAGChain initialised")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run a query through the full RAG pipeline.

        Returns:
            {
                "answer": str,
                "sources": List[dict],      # source metadata for each cited chunk
                "retrieved_chunks": int,    # number of chunks retrieved
                "latency_ms": float,
            }
        """
        start = time.perf_counter()

        # 1. Retrieve
        docs_with_scores = retrieve(question, self.vector_store)

        # 2. Rerank
        reranked_docs = rerank(question, docs_with_scores)

        # 3. Format context
        context = format_docs(reranked_docs)

        # 4. Build prompt
        messages = RAG_PROMPT.format_messages(
            context=context,
            question=question,
            chat_history=self.memory.get_history(),
        )

        # 5. Call LLM
        response = self.llm.invoke(messages)
        answer = response.content

        # 6. Update memory
        self.memory.add_turn(question, answer)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Query answered in {latency_ms:.0f}ms | chunks used: {len(reranked_docs)}")

        return {
            "answer": answer,
            "sources": _extract_sources(reranked_docs),
            "retrieved_chunks": len(reranked_docs),
            "latency_ms": round(latency_ms, 1),
        }

    def stream(self, question: str) -> Iterator[str]:
        """
        Stream tokens as they arrive from the LLM.

        Yields individual text tokens for real-time display.
        Also updates conversation memory after the full response is assembled.
        """
        docs_with_scores = retrieve(question, self.vector_store)
        reranked_docs = rerank(question, docs_with_scores)
        context = format_docs(reranked_docs)

        messages = RAG_PROMPT.format_messages(
            context=context,
            question=question,
            chat_history=self.memory.get_history(),
        )

        full_answer = ""
        for chunk in self.streaming_llm.stream(messages):
            token = chunk.content
            full_answer += token
            yield token

        self.memory.add_turn(question, full_answer)

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """Extract clean source metadata from retrieved documents."""
    seen = set()
    sources = []
    for doc in docs:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.metadata.get("topic", ""),
        )
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "topic": doc.metadata.get("topic", ""),
                    "page": doc.metadata.get("page", ""),
                    "preview": doc.page_content[:200] + "...",
                }
            )
    return sources
