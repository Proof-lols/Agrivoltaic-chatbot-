"""Tests for persistent conversation storage."""

from __future__ import annotations

from src.storage.conversation_store import ConversationStore


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "How much row spacing do I need?"},
        {"role": "assistant", "content": "About 20-30 ft for tractors.", "error": False},
    ]


def test_save_and_load_roundtrip(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    cid = store.new_id()
    store.save(cid, _messages())

    loaded = store.load(cid)
    assert loaded is not None
    assert loaded["id"] == cid
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][0]["content"].startswith("How much row spacing")


def test_title_derived_from_first_user_message(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    cid = store.new_id()
    record = store.save(cid, _messages())
    assert record is not None
    assert "row spacing" in record["title"].lower()


def test_list_conversations_sorted_newest_first(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    first = store.new_id()
    store.save(first, [{"role": "user", "content": "first"}])
    second = store.new_id()
    store.save(second, [{"role": "user", "content": "second"}])

    listed = store.list_conversations()
    assert len(listed) == 2
    assert listed[0]["updated_at"] >= listed[1]["updated_at"]


def test_delete_removes_conversation(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    cid = store.new_id()
    store.save(cid, _messages())
    assert store.delete(cid) is True
    assert store.load(cid) is None


def test_created_at_preserved_on_update(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    cid = store.new_id()
    first = store.save(cid, [{"role": "user", "content": "hi"}])
    second = store.save(cid, _messages())
    assert first is not None and second is not None
    assert first["created_at"] == second["created_at"]
