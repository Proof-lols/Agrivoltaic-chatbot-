# Agrivoltaics Farm Assistant

A friendly Streamlit chatbot that helps farmers with agrivoltaics questions. It uses a
local RAG (retrieval-augmented generation) knowledge base silently in the background to
give more grounded answers, organizes chats into **Projects**, and keeps a history of past
conversations.

## Features

- Plain-language farm assistant powered by OpenAI.
- Background RAG over a local ChromaDB vector store (no jargon shown to users).
- Projects to group conversations by topic, plus saved conversation history.
- Hidden researcher/audit mode via `?researcher=1` for inspecting retrieved sources.

## Run locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI key
copy .env.example .env         # then edit .env and set OPENAI_API_KEY

# 4. (Optional) Rebuild the vector store from knowledge_base/
python ingest.py

# 5. Launch the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already done).
2. Go to https://share.streamlit.io → **New app** → select this repo, branch `main`,
   main file `app.py`.
3. In **Advanced settings → Secrets**, add your key:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Click **Deploy**.

Notes:
- The prebuilt vector store in `chroma_db/` is committed, so RAG works immediately on
  deploy with no extra build step.
- `pysqlite3-binary` is installed on Linux hosts to satisfy ChromaDB's SQLite version
  requirement (handled automatically via `requirements.txt` and a shim in `app.py`).
- Saved conversations and projects are written to the local filesystem, which is
  **ephemeral** on Streamlit Cloud — they reset when the app restarts. Use a host with a
  persistent disk (e.g. Render, Railway) if you need them to persist.

## Configuration

Settings are read from environment variables / `.env` (see `config/settings.py`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | _(required)_ | OpenAI API key |
| `TAVILY_API_KEY` | _(optional)_ | Optional web search |
| `CHAT_MODEL` | `gpt-4o` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |

## Tests

```bash
pytest -q
```
