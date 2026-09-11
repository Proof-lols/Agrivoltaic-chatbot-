"""Tests for chat engine system-message assembly and error typing."""

from __future__ import annotations

from src.chat.engine import ChatError, FarmChatEngine
from src.models import ChatMessage, ChatTurnResult, RetrievalChunk, RetrievalResult


def test_system_content_includes_context_when_chunks_present() -> None:
    turn = ChatTurnResult(
        assistant_text="",
        retrieval=RetrievalResult(
            query="row spacing",
            chunks=[
                RetrievalChunk(
                    document_id="d1",
                    content="Row spacing must be 20-30 ft for tractors.",
                    source="site.md",
                    page=1,
                    score=0.12,
                )
            ],
        ),
    )
    content = FarmChatEngine.build_system_content(turn)
    assert "Row spacing must be 20-30 ft" in content
    assert "Private background notes" in content


def test_system_content_low_confidence_when_no_chunks() -> None:
    turn = ChatTurnResult(
        assistant_text="",
        retrieval=RetrievalResult(query="anything", chunks=[]),
    )
    content = FarmChatEngine.build_system_content(turn)
    assert "No specific background notes" in content
    assert "Private background notes" not in content


def test_system_content_includes_project_topic() -> None:
    turn = ChatTurnResult(
        assistant_text="",
        retrieval=RetrievalResult(query="anything", chunks=[]),
    )
    content = FarmChatEngine.build_system_content(turn, topic="Solar grazing with sheep")
    assert "Solar grazing with sheep" in content
    assert "part of a project about" in content


def test_chat_error_carries_kind() -> None:
    err = ChatError("busy", "rate limited")
    assert err.kind == "busy"
    assert "rate limited" in str(err)


def test_discard_last_turn_removes_dangling_user() -> None:
    engine = FarmChatEngine(settings=None, openai_client=None, retriever=None)
    engine._history = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content="unanswered"),
    ]
    engine.discard_last_turn()
    assert [m.role for m in engine._history] == ["user", "assistant"]


def test_discard_last_turn_keeps_completed_turn() -> None:
    engine = FarmChatEngine(settings=None, openai_client=None, retriever=None)
    engine._history = [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", content="a"),
    ]
    engine.discard_last_turn()
    assert len(engine._history) == 2
