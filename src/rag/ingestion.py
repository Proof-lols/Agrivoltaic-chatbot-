"""ChromaDB ingestion utilities for the local agrivoltaics knowledge base."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, RateLimitError
from pypdf import PdfReader

from config.settings import Settings

logger = logging.getLogger("agrivoltaics.rag.ingestion")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class KnowledgeBaseIngester:
    """Loads, chunks, embeds, and persists documents into ChromaDB."""

    def __init__(self, settings: Settings, openai_client: OpenAI) -> None:
        self._settings = settings
        self._client = openai_client
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._chroma = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name="agrivoltaics_kb",
            metadata={"hnsw:space": "cosine"},
        )

    def _read_file(self, path: Path) -> list[tuple[str, int | None]]:
        """Return list of (text, page_number) tuples from a supported file."""
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [(text, None)]

        if suffix == ".pdf":
            reader = PdfReader(str(path))
            pages: list[tuple[str, int | None]] = []
            for index, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append((page_text, index))
            return pages

        return []

    def _embed(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        """Generate embeddings via OpenAI in batches with rate-limit retries."""
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for attempt in range(5):
                try:
                    response = self._client.embeddings.create(
                        model=self._settings.embedding_model,
                        input=batch,
                    )
                    all_embeddings.extend(item.embedding for item in response.data)
                    break
                except RateLimitError:
                    wait = 2 ** attempt
                    logger.warning("Rate limit hit, retrying in %ss...", wait)
                    time.sleep(wait)
            else:
                raise RuntimeError("Exceeded embedding retries after repeated rate limits.")
            time.sleep(0.5)
        return all_embeddings

    def ingest_directory(self, force_rebuild: bool = False) -> dict[str, Any]:
        """
        Scan knowledge_base directory, chunk documents, and upsert into ChromaDB.

        Returns ingestion statistics for UI feedback.
        """
        stats: dict[str, Any] = {
            "files_processed": 0,
            "chunks_indexed": 0,
            "errors": [],
        }

        try:
            kb_dir = self._settings.knowledge_base_dir
            kb_dir.mkdir(parents=True, exist_ok=True)

            if force_rebuild:
                logger.info("Rebuilding Chroma collection from scratch.")
                self._chroma.delete_collection("agrivoltaics_kb")
                self._collection = self._chroma.get_or_create_collection(
                    name="agrivoltaics_kb",
                    metadata={"hnsw:space": "cosine"},
                )

            files = [
                path
                for path in kb_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            if not files:
                logger.warning("No ingestible files found in %s", kb_dir)
                return stats

            for file_path in files:
                try:
                    raw_pages = self._read_file(file_path)
                    if not raw_pages:
                        continue

                    documents: list[str] = []
                    metadatas: list[dict[str, Any]] = []
                    ids: list[str] = []

                    for page_text, page_num in raw_pages:
                        chunks = self._splitter.split_text(page_text)
                        for chunk_index, chunk in enumerate(chunks):
                            doc_id = f"{file_path.stem}-p{page_num or 0}-c{chunk_index}"
                            documents.append(chunk)
                            metadatas.append(
                                {
                                    "source": file_path.name,
                                    "page": page_num if page_num is not None else "N/A",
                                    "path": str(file_path),
                                }
                            )
                            ids.append(doc_id)

                    if not documents:
                        continue

                    embeddings = self._embed(documents)
                    self._collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings,
                    )
                    stats["files_processed"] += 1
                    stats["chunks_indexed"] += len(documents)
                    logger.info("Indexed %s chunks from %s", len(documents), file_path.name)
                except Exception as file_exc:  # noqa: BLE001
                    msg = f"{file_path.name}: {file_exc}"
                    logger.exception("Failed to ingest file: %s", msg)
                    stats["errors"].append(msg)

            return stats
        except Exception as exc:  # noqa: BLE001
            logger.exception("Directory ingestion failed: %s", exc)
            stats["errors"].append(str(exc))
            return stats

    @property
    def collection_count(self) -> int:
        """Return number of indexed chunks."""
        try:
            return int(self._collection.count())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to read collection count: %s", exc)
            return 0
