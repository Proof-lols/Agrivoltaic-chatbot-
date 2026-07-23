"""Persistent storage for chat conversations as local JSON files."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("agrivoltaics.storage")

_TITLE_MAX_LEN = 48


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_title(text: str) -> str:
    """Derive a short, human-friendly title from the first user message."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "New conversation"
    if len(cleaned) <= _TITLE_MAX_LEN:
        return cleaned
    return cleaned[:_TITLE_MAX_LEN].rstrip() + "..."


class ConversationStore:
    """Save, load, list, and delete chat conversations on disk.

    Each conversation is a JSON file ``<id>.json`` in ``storage_dir`` shaped as::

        {
          "id": str,
          "title": str,
          "created_at": iso8601,
          "updated_at": iso8601,
          "messages": [{"role": str, "content": str, "audit": [...], "error": bool}]
        }
    """

    def __init__(self, storage_dir: Path) -> None:
        self._dir = Path(storage_dir)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to create conversation storage dir: %s", exc)

    def _path(self, conversation_id: str) -> Path:
        return self._dir / f"{conversation_id}.json"

    def new_id(self) -> str:
        """Generate a sortable, unique conversation id."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:6]}"

    def save(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Persist a conversation, creating or updating its file."""
        try:
            path = self._path(conversation_id)
            first_user = next(
                (m["content"] for m in messages if m.get("role") == "user"),
                "New conversation",
            )
            created_at = _now_iso()
            existing_project = project_id
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    created_at = existing.get("created_at", created_at)
                    if existing_project is None:
                        existing_project = existing.get("project_id")
                except Exception:  # noqa: BLE001
                    pass

            record = {
                "id": conversation_id,
                "project_id": existing_project,
                "title": _make_title(first_user),
                "created_at": created_at,
                "updated_at": _now_iso(),
                "messages": messages,
            }
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
            return record
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to save conversation %s: %s", conversation_id, exc)
            return None

    def load(self, conversation_id: str) -> dict[str, Any] | None:
        """Load a single conversation by id."""
        try:
            path = self._path(conversation_id)
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load conversation %s: %s", conversation_id, exc)
            return None

    def list_conversations(self, project_id: str | None = None) -> list[dict[str, Any]]:
        """Return conversation metadata, newest first.

        If ``project_id`` is provided, only conversations in that project are returned.
        """
        conversations: list[dict[str, Any]] = []
        try:
            for path in self._dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if project_id is not None and data.get("project_id") != project_id:
                        continue
                    conversations.append(
                        {
                            "id": data.get("id", path.stem),
                            "project_id": data.get("project_id"),
                            "title": data.get("title", "Untitled"),
                            "updated_at": data.get("updated_at", ""),
                            "message_count": len(data.get("messages", [])),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping unreadable conversation %s: %s", path.name, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to list conversations: %s", exc)

        conversations.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        return conversations

    def delete_by_project(self, project_id: str) -> int:
        """Delete all conversations belonging to a project. Returns count removed."""
        removed = 0
        for conv in self.list_conversations(project_id=project_id):
            if self.delete(conv["id"]):
                removed += 1
        return removed

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation file."""
        try:
            path = self._path(conversation_id)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to delete conversation %s: %s", conversation_id, exc)
            return False
