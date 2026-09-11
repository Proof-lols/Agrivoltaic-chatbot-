"""One-off audit: compare knowledge_base files vs ChromaDB index."""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(override=True)

from config.settings import Settings
from openai import OpenAI
from src.rag.ingestion import KnowledgeBaseIngester, SUPPORTED_EXTENSIONS


def main() -> None:
    settings = Settings()
    kb = settings.knowledge_base_dir
    files = sorted(p for p in kb.rglob("*") if p.is_file())
    supported = [p for p in files if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    unsupported = [p for p in files if p.suffix.lower() not in SUPPORTED_EXTENSIONS]

    print("=== KNOWLEDGE BASE FILES ===")
    if not files:
        print("(empty folder)")
    for path in supported:
        print(f"OK   {path.name} ({path.suffix}, {path.stat().st_size} bytes)")
    for path in unsupported:
        ext = path.suffix or "no extension"
        print(f"SKIP {path.name} ({ext} — unsupported format)")

    client = OpenAI(api_key=settings.openai_api_key)
    ingester = KnowledgeBaseIngester(settings, client)
    count = ingester.collection_count
    print("\n=== VECTOR STORE ===")
    print(f"Indexed chunks in ChromaDB: {count}")

    if count == 0 and supported:
        print("\nNo chunks indexed yet — running ingestion...")
        stats = ingester.ingest_directory(force_rebuild=False)
        print(f"Files processed: {stats['files_processed']}")
        print(f"Chunks indexed: {stats['chunks_indexed']}")
        if stats["errors"]:
            print("Errors:")
            for err in stats["errors"]:
                print(f"  - {err}")
        count = ingester.collection_count
        print(f"New chunk count: {count}")

    if count > 0:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        chroma = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        col = chroma.get_collection("agrivoltaics_kb")
        meta = col.get(include=["metadatas"])
        sources = sorted({m.get("source", "?") for m in meta["metadatas"]})
        print("\nIndexed source files:")
        for source in sources:
            print(f"  - {source}")
        missing = [p.name for p in supported if p.name not in sources]
        if missing:
            print("\nNOT yet indexed (click Ingest or re-run this script):")
            for name in missing:
                print(f"  - {name}")
        elif supported:
            print("\nAll supported files appear indexed.")
        elif not supported:
            print("\nNo supported files to index.")


if __name__ == "__main__":
    main()
