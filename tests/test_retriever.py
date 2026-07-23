"""Tests for relevance-threshold filtering in the RAG retriever."""

from __future__ import annotations

from types import SimpleNamespace

from config.settings import Settings
from src.rag.retriever import retrieve_context


class _FakeEmbeddings:
    def create(self, model, input):  # noqa: A002 - mirror OpenAI signature
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


class _FakeCollection:
    def __init__(self, payload: dict, count: int = 10) -> None:
        self._payload = payload
        self._count = count

    def count(self) -> int:
        return self._count

    def query(self, query_embeddings, n_results, include):
        return self._payload


def _settings(**overrides) -> Settings:
    base = {
        "openai_api_key": "test-key",
        "retrieval_top_k": 2,
        "retrieval_distance_threshold": 0.5,
        "retrieval_candidate_multiplier": 3,
    }
    base.update(overrides)
    return Settings(**base)


def test_filters_out_chunks_above_threshold() -> None:
    payload = {
        "ids": [["a", "b", "c", "d"]],
        "documents": [["near", "far", "mid", "beyond"]],
        "metadatas": [
            [
                {"source": "near.md", "page": 1},
                {"source": "far.md", "page": 2},
                {"source": "mid.md", "page": 3},
                {"source": "beyond.md", "page": 4},
            ]
        ],
        "distances": [[0.10, 0.90, 0.30, 0.60]],
    }
    result = retrieve_context(
        query="q",
        settings=_settings(),
        openai_client=_FakeClient(),
        top_k=2,
        collection=_FakeCollection(payload),
    )
    sources = [c.source for c in result.chunks]
    # 0.90 and 0.60 exceed threshold 0.5 and are dropped; 0.10 and 0.30 kept, sorted ascending.
    assert sources == ["near.md", "mid.md"]


def test_respects_top_k_after_filtering() -> None:
    payload = {
        "ids": [["a", "b", "c"]],
        "documents": [["one", "two", "three"]],
        "metadatas": [
            [
                {"source": "one.md", "page": 1},
                {"source": "two.md", "page": 1},
                {"source": "three.md", "page": 1},
            ]
        ],
        "distances": [[0.10, 0.20, 0.30]],
    }
    result = retrieve_context(
        query="q",
        settings=_settings(retrieval_top_k=2),
        openai_client=_FakeClient(),
        top_k=2,
        collection=_FakeCollection(payload),
    )
    assert len(result.chunks) == 2
    assert [c.source for c in result.chunks] == ["one.md", "two.md"]


def test_empty_collection_returns_no_chunks() -> None:
    result = retrieve_context(
        query="q",
        settings=_settings(),
        openai_client=_FakeClient(),
        top_k=2,
        collection=_FakeCollection({}, count=0),
    )
    assert result.chunks == []
