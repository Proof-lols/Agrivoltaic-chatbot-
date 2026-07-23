"""Persistent storage for projects (topic-scoped conversation groups)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("agrivoltaics.storage.projects")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    """Manage projects in a single JSON index file.

    Each project is a dict::

        {"id": str, "name": str, "topic": str, "created_at": iso8601}
    """

    def __init__(self, storage_dir: Path) -> None:
        self._dir = Path(storage_dir)
        self._index = self._dir / "projects.json"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to create project storage dir: %s", exc)

    def _read_all(self) -> list[dict[str, Any]]:
        try:
            if not self._index.exists():
                return []
            return json.loads(self._index.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to read projects index: %s", exc)
            return []

    def _write_all(self, projects: list[dict[str, Any]]) -> bool:
        try:
            tmp = self._index.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(projects, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._index)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to write projects index: %s", exc)
            return False

    def new_id(self) -> str:
        """Generate a sortable, unique project id."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"proj-{stamp}-{uuid.uuid4().hex[:6]}"

    def list_projects(self) -> list[dict[str, Any]]:
        """Return all projects, newest first."""
        projects = self._read_all()
        projects.sort(key=lambda p: p.get("created_at", ""), reverse=True)
        return projects

    def create_project(self, name: str, topic: str = "") -> dict[str, Any] | None:
        """Create and persist a new project."""
        try:
            clean_name = (name or "").strip() or "Untitled project"
            record = {
                "id": self.new_id(),
                "name": clean_name,
                "topic": (topic or "").strip(),
                "created_at": _now_iso(),
            }
            projects = self._read_all()
            projects.append(record)
            self._write_all(projects)
            return record
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to create project: %s", exc)
            return None

    def get(self, project_id: str) -> dict[str, Any] | None:
        """Return a single project by id."""
        for project in self._read_all():
            if project.get("id") == project_id:
                return project
        return None

    def delete(self, project_id: str) -> bool:
        """Remove a project from the index."""
        projects = self._read_all()
        remaining = [p for p in projects if p.get("id") != project_id]
        if len(remaining) == len(projects):
            return False
        return self._write_all(remaining)

    def ensure_default(self) -> dict[str, Any]:
        """Return an existing project or create a default 'General' one."""
        projects = self.list_projects()
        if projects:
            return projects[0]
        created = self.create_project(
            "General", "General agrivoltaics questions and farm planning."
        )
        return created or {"id": "", "name": "General", "topic": "", "created_at": _now_iso()}
