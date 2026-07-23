"""Tests for project storage and project-scoped conversation filtering."""

from __future__ import annotations

from src.storage.conversation_store import ConversationStore
from src.storage.project_store import ProjectStore


def test_create_and_list_projects(tmp_path) -> None:
    store = ProjectStore(tmp_path)
    store.create_project("Solar Grazing", "Sheep under panels")
    store.create_project("Berry Trials", "Shade-tolerant berries")

    projects = store.list_projects()
    assert len(projects) == 2
    names = {p["name"] for p in projects}
    assert names == {"Solar Grazing", "Berry Trials"}


def test_get_and_delete_project(tmp_path) -> None:
    store = ProjectStore(tmp_path)
    created = store.create_project("Irrigation", "Water planning")
    assert created is not None
    assert store.get(created["id"])["topic"] == "Water planning"
    assert store.delete(created["id"]) is True
    assert store.get(created["id"]) is None


def test_ensure_default_creates_general(tmp_path) -> None:
    store = ProjectStore(tmp_path)
    default = store.ensure_default()
    assert default["name"] == "General"
    assert len(store.list_projects()) == 1


def test_conversations_filtered_by_project(tmp_path) -> None:
    convo = ConversationStore(tmp_path / "conversations")
    a = convo.new_id()
    b = convo.new_id()
    convo.save(a, [{"role": "user", "content": "in project 1"}], project_id="p1")
    convo.save(b, [{"role": "user", "content": "in project 2"}], project_id="p2")

    p1 = convo.list_conversations(project_id="p1")
    assert len(p1) == 1
    assert p1[0]["id"] == a
    assert len(convo.list_conversations(project_id="p2")) == 1
    assert len(convo.list_conversations()) == 2


def test_delete_by_project_removes_all(tmp_path) -> None:
    convo = ConversationStore(tmp_path / "conversations")
    convo.save(convo.new_id(), [{"role": "user", "content": "x"}], project_id="p1")
    convo.save(convo.new_id(), [{"role": "user", "content": "y"}], project_id="p1")
    convo.save(convo.new_id(), [{"role": "user", "content": "z"}], project_id="p2")

    removed = convo.delete_by_project("p1")
    assert removed == 2
    assert convo.list_conversations(project_id="p1") == []
    assert len(convo.list_conversations(project_id="p2")) == 1
