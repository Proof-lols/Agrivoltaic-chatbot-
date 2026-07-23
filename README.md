# Agrivoltaics RAG Decision-Support Chatbot

Production-grade Streamlit application combining **OpenAI GPT-4o**, **ChromaDB RAG**, and **vetted .edu/.org web search** for interactive agrivoltaics farm planning.

## Features

- **Two-step evaluation loop** — validates four core pillars (acreage, soil, microclimate, capital) before final recommendations
- **Proactive question escalation** — advances consultation toward tracking, tilt, net-metering, and cultivar optimization
- **Local RAG pipeline** — `retrieve_context()` with `text-embedding-3-small` and source/page citations
- **Credibility-constrained web search** — Tavily queries append `site:.edu OR site:.org` filters
- **Streaming GPT-4o** — structured Markdown sections with hardcoded system prompt matrix

## Quick Start

```bash
cd C:\Users\noahm\Projects\agrivoltaics-rag-chatbot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your OPENAI_API_KEY (and optional TAVILY_API_KEY)
streamlit run app.py
```

1. Open the sidebar and click **Ingest / Rebuild Vector Store**
2. Add `.txt`, `.md`, or `.pdf` files to `knowledge_base/`
3. Chat with the assistant — missing pillar data triggers the collection form

## Project Structure

```
app.py                          # Streamlit UI entry point
config/settings.py              # Environment + system prompt constants
src/
  models.py                     # Pydantic domain models
  validation/pillar_validator.py
  conversation/state_tracker.py
  rag/ingestion.py              # Vector store builder
  rag/retriever.py              # retrieve_context()
  search/web_search.py          # Domain-filtered Tavily wrapper
  chat/engine.py                # GPT-4o streaming orchestrator
knowledge_base/                 # Local documents for RAG
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Chat (gpt-4o) and embeddings |
| `TAVILY_API_KEY` | No | Live .edu/.org web search fallback |
| `KNOWLEDGE_BASE_DIR` | No | Default `./knowledge_base` |
| `CHROMA_PERSIST_DIR` | No | Default `./chroma_db` |

## Response Sections

Every assistant reply is formatted into:

- Agricultural Analysis & Localized Site Context
- Solar System Geometry & Shading Optimization
- Botanical Assessment & Cultivar Selection
- Economic Feasibility & Capital Requirements Analysis
- Suggested Forward-Moving Planning Actions

When pillars are incomplete, a **CRITICAL DATA NEEDED** section and sidebar form appear automatically.
