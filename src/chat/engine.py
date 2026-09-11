"""OpenAI gpt-4o streaming chat engine with RAG-grounded farm assistant."""

from __future__ import annotations

import logging
import time
from collections.abc import Generator

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from config.settings import SYSTEM_PROMPT, Settings
from src.models import ChatMessage, ChatTurnResult
from src.rag.retriever import ContextRetriever

logger = logging.getLogger("agrivoltaics.chat")


class ChatError(Exception):
    """Typed chat failure so the UI can show an appropriate message.

    ``kind`` is one of: ``auth``, ``busy``, ``connection``, ``unknown``.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class FarmChatEngine:
    """Farm assistant that grounds responses in the local knowledge base via RAG."""

    def __init__(
        self,
        settings: Settings,
        openai_client: OpenAI,
        retriever: ContextRetriever,
    ) -> None:
        self._settings = settings
        self._client = openai_client
        self._retriever = retriever
        self._history: list[ChatMessage] = []
        self._topic: str | None = None

    def set_topic(self, topic: str | None) -> None:
        """Set the current project topic used to steer answers."""
        self._topic = (topic or "").strip() or None

    def prepare_turn(self, user_query: str) -> ChatTurnResult:
        """Retrieve relevant document chunks and prepare for streaming."""
        try:
            self._history.append(ChatMessage(role="user", content=user_query))
            retrieval = self._retriever.retrieve(user_query)
            return ChatTurnResult(assistant_text="", retrieval=retrieval)
        except Exception as exc:  # noqa: BLE001
            logger.exception("prepare_turn failed: %s", exc)
            return ChatTurnResult(assistant_text=f"System error: {exc}")

    @staticmethod
    def build_system_content(turn: ChatTurnResult, topic: str | None = None) -> str:
        """Assemble the system message, injecting topic, RAG context, or a low-confidence note."""
        base = SYSTEM_PROMPT
        if topic:
            base += (
                f"\n\nThis conversation is part of a project about: {topic}. "
                "Keep your answers focused on and relevant to this topic when appropriate."
            )

        has_context = bool(turn.retrieval and turn.retrieval.chunks)
        if has_context:
            rag_context = turn.retrieval.formatted_context()
            return (
                base
                + "\n\n[Private background notes — use to inform your answer, "
                "never mention these to the farmer]\n"
                + rag_context
            )
        return (
            base
            + "\n\n[No specific background notes matched this question. Answer from "
            "general, well-established knowledge only. Keep specifics general, avoid "
            "inventing exact figures, and gently note that details depend on the "
            "farmer's own situation. Do not mention notes or sources.]"
        )

    def stream_response(self, user_query: str, turn: ChatTurnResult) -> Generator[str, None, None]:
        """Stream gpt-4o tokens with RAG context injected."""
        try:
            system_content = self.build_system_content(turn, self._topic)

            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
            for msg in self._history[:-1]:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": user_query})

            stream = self._open_stream(messages)
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except ChatError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming generation failed: %s", exc)
            raise ChatError("unknown", str(exc)) from exc

    def _open_stream(self, messages: list[dict[str, str]], max_attempts: int = 4):
        """Open a streaming completion, retrying transient failures with backoff."""
        for attempt in range(max_attempts):
            try:
                return self._client.chat.completions.create(
                    model=self._settings.chat_model,
                    messages=messages,
                    stream=True,
                    temperature=0.3,
                )
            except AuthenticationError as exc:
                logger.error("OpenAI authentication failed: %s", exc)
                raise ChatError("auth", str(exc)) from exc
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                if attempt == max_attempts - 1:
                    kind = "busy" if isinstance(exc, RateLimitError) else "connection"
                    logger.error("Chat stream failed after retries (%s): %s", kind, exc)
                    raise ChatError(kind, str(exc)) from exc
                wait = 2**attempt
                logger.warning("Transient chat error, retrying in %ss: %s", wait, exc)
                time.sleep(wait)
        raise ChatError("unknown", "Exhausted chat retries.")

    def complete_turn(self, _user_query: str, assistant_text: str) -> None:
        """Append assistant reply to conversation history."""
        try:
            self._history.append(ChatMessage(role="assistant", content=assistant_text))
        except Exception as exc:  # noqa: BLE001
            logger.exception("complete_turn failed: %s", exc)

    def discard_last_turn(self) -> None:
        """Drop a trailing user message that never received an assistant reply.

        Called after a failed turn so the unanswered question does not pollute the
        context sent on the next request.
        """
        if self._history and self._history[-1].role == "user":
            self._history.pop()

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def load_history(self, messages: list[dict[str, str]]) -> None:
        """Restore conversation context from stored messages.

        Skips messages flagged as errors so they don't pollute the LLM context.
        """
        self._history.clear()
        for msg in messages:
            if msg.get("error"):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role in {"user", "assistant"} and content:
                self._history.append(ChatMessage(role=role, content=content))


AgrivoltaicsChatEngine = FarmChatEngine
