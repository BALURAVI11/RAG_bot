"""Prompt templates for the RAG chain."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable enterprise knowledge assistant. \
Your job is to answer questions accurately and concisely based ONLY on the \
provided context documents.

Guidelines:
- Answer using information from the provided context only.
- If the context doesn't contain enough information to fully answer the question, \
say: "I don't have enough information in the provided documents to answer that completely."
- Always cite your sources by mentioning the document or topic name from the metadata.
- Keep answers clear and structured. Use bullet points for lists.
- Do not make up facts or use knowledge outside the provided context.
- If asked something outside the document scope, say so politely.

Context documents:
{context}
"""

# ── Chat prompt (with history support) ───────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ]
)

# ── Standalone question prompt (for query rewriting with history) ─────────────
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and a follow-up question, rewrite the follow-up "
            "question to be a standalone question that captures all necessary context. "
            "Return ONLY the rewritten question, nothing else.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Follow-up question: {question}"),
    ]
)


def format_docs(docs) -> str:
    """
    Format a list of retrieved Documents into a single context string.

    Each chunk is labelled with its source for easy citation.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        topic = doc.metadata.get("topic", "")
        page = doc.metadata.get("page", "")

        label_parts = [f"[{i}]"]
        if topic:
            label_parts.append(f"Topic: {topic}")
        label_parts.append(f"Source: {source}")
        if page:
            label_parts.append(f"Page: {page}")

        header = " | ".join(label_parts)
        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)
