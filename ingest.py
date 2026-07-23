"""Ingest knowledge_base into ChromaDB for RAG retrieval."""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(override=True)

from config.settings import Settings
from openai import OpenAI
from src.rag.ingestion import KnowledgeBaseIngester


def main() -> None:
    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)
    ingester = KnowledgeBaseIngester(settings, client)
    print("Starting full ingestion (rebuild)...")
    stats = ingester.ingest_directory(force_rebuild=True)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Chunks indexed: {stats['chunks_indexed']}")
    print(f"Total chunks in DB: {ingester.collection_count}")
    if stats["errors"]:
        print("Errors:")
        for err in stats["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
