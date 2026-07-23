"""Local RAG retrieval pipeline with relevance filtering and citation metadata."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from config.settings import Settings
from src.models import RetrievalChunk, RetrievalResult

logger = logging.getLogger("agrivoltaics.rag.retriever")

_COLLECTION_NAME = "agrivoltaics_kb"


def retrieve_context(
    query: str,
    settings: Settings,
    openai_client: OpenAI,
    top_k: int = 3,
    collection: Any | None = None,
) -> RetrievalResult:
    """
    Route a farmer query to the local ChromaDB vector store.

    Embeds the query with the configured embedding model, retrieves a candidate
    set, discards chunks whose cosine distance exceeds
    ``settings.retrieval_distance_threshold``, and returns the best ``top_k``
    with source/page metadata for transparent citations.

    A pre-built ``collection`` may be supplied to avoid reconstructing the
    ChromaDB client on every call.
    """
    try:
        if collection is None:
            chroma = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = chroma.get_or_create_collection(name=_COLLECTION_NAME)

        if collection.count() == 0:
            logger.warning("Vector store empty - run ingestion first.")
            return RetrievalResult(query=query, chunks=[])

        embedding_response = openai_client.embeddings.create(
            model=settings.embedding_model,
            input=[query],
        )
        query_embedding = embedding_response.data[0].embedding

        candidate_count = max(top_k, top_k * settings.retrieval_candidate_multiplier)
        results: dict[str, Any] = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        threshold = settings.retrieval_distance_threshold
        candidates: list[RetrievalChunk] = []
        for doc_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            if not content:
                continue
            score = float(distance) if distance is not None else None
            if score is not None and score > threshold:
                continue
            meta = metadata or {}
            candidates.append(
                RetrievalChunk(
                    document_id=doc_id,
                    content=content,
                    source=str(meta.get("source", "unknown")),
                    page=meta.get("page"),
                    score=score,
                )
            )

        candidates.sort(key=lambda c: (c.score is None, c.score if c.score is not None else 0.0))
        chunks = candidates[:top_k]

        logger.info(
            "Retrieved %s/%s candidate chunks passed threshold %.2f for query.",
            len(chunks),
            len(documents),
            threshold,
        )
        return RetrievalResult(query=query, chunks=chunks)
    except Exception as exc:  # noqa: BLE001
        logger.exception("retrieve_context failed: %s", exc)
        return RetrievalResult(query=query, chunks=[])


class ContextRetriever:
    """Object-oriented wrapper that reuses one ChromaDB client across queries."""

    def __init__(self, settings: Settings, openai_client: OpenAI) -> None:
        self._settings = settings
        self._client = openai_client
        try:
            self._chroma = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._chroma.get_or_create_collection(name=_COLLECTION_NAME)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize ChromaDB collection: %s", exc)
            self._collection = None

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve relevance-filtered context chunks with citation metadata."""
        k = top_k or self._settings.retrieval_top_k
        return retrieve_context(
            query=query,
            settings=self._settings,
            openai_client=self._client,
            top_k=k,
            collection=self._collection,
        )
