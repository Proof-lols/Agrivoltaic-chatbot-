"""Tests for document reading and chunking in the ingestion pipeline."""

from __future__ import annotations

from config.settings import Settings
from src.rag.ingestion import KnowledgeBaseIngester


class _DummyClient:
    """Placeholder OpenAI client; ingestion file-reading never calls it."""


def _ingester(tmp_path) -> KnowledgeBaseIngester:
    settings = Settings(
        openai_api_key="test-key",
        knowledge_base_dir=tmp_path,
        chroma_persist_dir=tmp_path / "chroma",
        chunk_size=200,
        chunk_overlap=20,
    )
    return KnowledgeBaseIngester(settings, _DummyClient())


def test_read_markdown_returns_text_without_page(tmp_path) -> None:
    md = tmp_path / "note.md"
    md.write_text("# Title\n\nSome agrivoltaics content.", encoding="utf-8")
    ingester = _ingester(tmp_path)
    pages = ingester._read_file(md)
    assert len(pages) == 1
    text, page = pages[0]
    assert "agrivoltaics content" in text
    assert page is None


def test_read_txt_returns_text(tmp_path) -> None:
    txt = tmp_path / "data.txt"
    txt.write_text("Land lease rates vary by region.", encoding="utf-8")
    ingester = _ingester(tmp_path)
    pages = ingester._read_file(txt)
    assert pages[0][0].startswith("Land lease")


def test_unsupported_extension_returns_empty(tmp_path) -> None:
    other = tmp_path / "image.png"
    other.write_bytes(b"\x89PNG")
    ingester = _ingester(tmp_path)
    assert ingester._read_file(other) == []


def test_long_text_splits_into_multiple_chunks(tmp_path) -> None:
    ingester = _ingester(tmp_path)
    long_text = ("Agrivoltaics improves land use. " * 100).strip()
    chunks = ingester._splitter.split_text(long_text)
    assert len(chunks) > 1
