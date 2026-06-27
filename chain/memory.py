"""Conversation memory — sliding window of recent chat turns."""
from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from loguru import logger


class ConversationMemory:
    """
    Maintains a sliding window of the last N conversation turns.

    Each turn = one HumanMessage + one AIMessage.
    Older turns are dropped to stay within the LLM's context limit.
    """

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self._history: List[BaseMessage] = []

    def add_turn(self, question: str, answer: str) -> None:
        """Append a human question + AI answer to history."""
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        # Keep only the last max_history turns (2 messages per turn)
        max_messages = self.max_history * 2
        if len(self._history) > max_messages:
            dropped = len(self._history) - max_messages
            self._history = self._history[dropped:]
            logger.debug(f"Memory trimmed: dropped {dropped} old messages")

    def get_history(self) -> List[BaseMessage]:
        """Return the current message history (for injection into the prompt)."""
        return list(self._history)

    def clear(self) -> None:
        """Reset history."""
        self._history = []

    def __len__(self) -> int:
        return len(self._history) // 2  # number of turns
