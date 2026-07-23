"""Streamlit dashboard — simple farm chat (RAG runs silently in the background)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from config.settings import Settings  # noqa: E402
from src.chat.engine import ChatError, FarmChatEngine  # noqa: E402
from src.logging_config import configure_logging  # noqa: E402
from src.rag.retriever import ContextRetriever  # noqa: E402
from src.storage.conversation_store import ConversationStore  # noqa: E402
from src.storage.project_store import ProjectStore  # noqa: E402

logger = logging.getLogger("agrivoltaics.app")

CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
PROJECTS_DIR = PROJECT_ROOT / "projects"

# Bump when submodules change so a running Streamlit session reloads cached modules
# (Streamlit reruns app.py but does not re-import already-loaded submodules).
_APP_SCHEMA_VERSION = 3


def _heal_stale_modules() -> None:
    """Force-reload updated submodules and drop stale cached instances (once per version)."""
    if st.session_state.get("_schema_version") == _APP_SCHEMA_VERSION:
        return
    import importlib

    import src.chat.engine as engine_mod
    import src.rag.retriever as retriever_mod
    import src.storage.conversation_store as conv_mod
    import src.storage.project_store as project_mod

    for module in (conv_mod, project_mod, retriever_mod, engine_mod):
        try:
            importlib.reload(module)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not reload %s: %s", module.__name__, exc)

    for key in ("engine", "store", "project_store"):
        st.session_state.pop(key, None)

    st.session_state._schema_version = _APP_SCHEMA_VERSION
    st.rerun()

ERROR_MESSAGES = {
    "auth": (
        "I'm not set up correctly right now (there's a configuration problem). "
        "Please contact your administrator."
    ),
    "busy": (
        "I'm getting a lot of questions at the moment and hit a usage limit. "
        "Please wait a minute and try again."
    ),
    "connection": (
        "I'm having trouble connecting right now. Please check your internet "
        "and try again in a moment."
    ),
    "unknown": "Sorry, I had trouble answering that. Please try again.",
}


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "store" not in st.session_state:
        st.session_state.store = ConversationStore(CONVERSATIONS_DIR)
    if "project_store" not in st.session_state:
        st.session_state.project_store = ProjectStore(PROJECTS_DIR)
    if "current_project_id" not in st.session_state:
        st.session_state.current_project_id = None


def _load_settings() -> Settings | None:
    try:
        return Settings()
    except Exception as exc:  # noqa: BLE001
        st.error("Something went wrong starting up. Please contact your administrator.")
        logger.exception("Settings load failed: %s", exc)
        return None


def _build_engine(settings: Settings) -> FarmChatEngine:
    client = OpenAI(api_key=settings.openai_api_key)
    retriever = ContextRetriever(settings, client)
    return FarmChatEngine(settings, client, retriever)


def _researcher_enabled() -> bool:
    """Hidden audit mode via ?researcher=1 query param (not shown to farmers)."""
    value = st.query_params.get("researcher")
    return str(value).lower() in {"1", "true", "yes"}


def _audit_from_turn(turn) -> list[dict]:
    """Extract retrieved source/page/score info for the researcher audit view."""
    if not turn.retrieval or not turn.retrieval.chunks:
        return []
    audit: list[dict] = []
    for chunk in turn.retrieval.chunks:
        audit.append(
            {
                "source": chunk.source,
                "page": chunk.page,
                "score": chunk.score,
                "preview": chunk.content[:300] + ("..." if len(chunk.content) > 300 else ""),
            }
        )
    return audit


def _render_audit(audit: list[dict]) -> None:
    """Render the retrieved-sources audit panel (researcher mode only)."""
    if not audit:
        st.caption("Researcher mode: no knowledge-base chunks passed the relevance threshold.")
        return
    with st.expander(f"Researcher mode: {len(audit)} source chunk(s) used", expanded=False):
        for i, item in enumerate(audit, 1):
            page = item.get("page")
            page_label = f", page {page}" if page not in (None, "N/A") else ""
            score = item.get("score")
            score_label = f" - distance {score:.3f}" if isinstance(score, (int, float)) else ""
            st.markdown(f"**[{i}] `{item['source']}`{page_label}**{score_label}")
            st.caption(item.get("preview", ""))


def _start_new_conversation(engine: FarmChatEngine) -> None:
    """Clear the active chat without leaving the current project."""
    engine.reset()
    st.session_state.messages = []
    st.session_state.current_conversation_id = None


def _render_projects_section(
    engine: FarmChatEngine, project_store: ProjectStore
) -> dict | None:
    """Render the project picker and creator. Returns the active project."""
    with st.sidebar:
        st.header("Projects")

        projects = project_store.list_projects()
        if not projects:
            default = project_store.ensure_default()
            projects = [default]

        ids = [p["id"] for p in projects]
        if st.session_state.current_project_id not in ids:
            st.session_state.current_project_id = ids[0]

        names = {p["id"]: p["name"] for p in projects}
        selected = st.selectbox(
            "Current project",
            options=ids,
            index=ids.index(st.session_state.current_project_id),
            format_func=lambda pid: names.get(pid, "Untitled"),
        )
        if selected != st.session_state.current_project_id:
            st.session_state.current_project_id = selected
            _start_new_conversation(engine)
            st.rerun()

        active = next((p for p in projects if p["id"] == selected), projects[0])
        if active.get("topic"):
            st.caption(f"Topic: {active['topic']}")

        with st.expander("New project"):
            new_name = st.text_input("Project name", key="new_project_name")
            new_topic = st.text_input(
                "Topic (optional)",
                key="new_project_topic",
                help="A short description of what this project is about.",
            )
            if st.button("Create project", use_container_width=True):
                if new_name.strip():
                    created = project_store.create_project(new_name, new_topic)
                    if created:
                        st.session_state.current_project_id = created["id"]
                        _start_new_conversation(engine)
                        st.rerun()
                else:
                    st.warning("Please enter a project name.")

        if len(projects) > 1:
            if st.button("Delete this project", use_container_width=True):
                st.session_state.store.delete_by_project(active["id"])
                project_store.delete(active["id"])
                st.session_state.current_project_id = None
                _start_new_conversation(engine)
                st.rerun()

        return active


def _render_conversation_sidebar(
    engine: FarmChatEngine, store: ConversationStore, project_id: str
) -> None:
    """Sidebar to start a new chat and reopen past conversations in this project."""
    with st.sidebar:
        st.divider()
        st.header("Conversations")
        if st.button("New conversation", use_container_width=True, type="primary"):
            _start_new_conversation(engine)
            st.rerun()

        conversations = store.list_conversations(project_id=project_id)
        if not conversations:
            st.caption("Your conversations in this project will appear here.")
            return

        current_id = st.session_state.current_conversation_id
        for conv in conversations:
            open_col, delete_col = st.columns([0.82, 0.18])
            prefix = "> " if conv["id"] == current_id else ""
            with open_col:
                if st.button(
                    f"{prefix}{conv['title']}",
                    key=f"load_{conv['id']}",
                    use_container_width=True,
                    help=f"{conv['message_count']} messages",
                ):
                    data = store.load(conv["id"])
                    if data:
                        st.session_state.messages = data.get("messages", [])
                        st.session_state.current_conversation_id = conv["id"]
                        engine.load_history(st.session_state.messages)
                        st.rerun()
            with delete_col:
                if st.button("Delete", key=f"del_{conv['id']}", use_container_width=True):
                    store.delete(conv["id"])
                    if current_id == conv["id"]:
                        _start_new_conversation(engine)
                    st.rerun()


LOGO_PATH = str(PROJECT_ROOT / "assets" / "solar_logo.png")


def main() -> None:
    st.set_page_config(page_title="AgriBot", page_icon=LOGO_PATH, layout="centered")

    settings = _load_settings()
    if settings is None:
        st.stop()

    configure_logging(settings.log_level)
    _heal_stale_modules()
    _init_session_state()

    if st.session_state.engine is None or not hasattr(st.session_state.engine, "set_topic"):
        st.session_state.engine = _build_engine(settings)

    engine: FarmChatEngine = st.session_state.engine
    store: ConversationStore = st.session_state.store
    project_store: ProjectStore = st.session_state.project_store
    researcher = _researcher_enabled()

    active_project = _render_projects_section(engine, project_store)
    project_id = active_project["id"] if active_project else None
    if hasattr(engine, "set_topic"):
        engine.set_topic(active_project.get("topic") if active_project else None)
    _render_conversation_sidebar(engine, store, project_id)

    header_col1, header_col2 = st.columns([1, 6], vertical_alignment="center")
    with header_col1:
        st.image(LOGO_PATH, width=70)
    with header_col2:
        st.title("Agrivoltaics AI Assistant")
    st.markdown("Ask me anything about agrivoltaics implementation!")

    if researcher:
        st.caption(
            f"Researcher mode ON | model: {settings.chat_model} | "
            f"top_k: {settings.retrieval_top_k} | "
            f"distance threshold: {settings.retrieval_distance_threshold}"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if researcher and message.get("audit") is not None:
                _render_audit(message["audit"])

    user_input = st.chat_input("What's on your mind?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        turn = engine.prepare_turn(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed_text = ""
            failed = False
            try:
                for token in engine.stream_response(user_input, turn):
                    streamed_text += token
                    placeholder.markdown(streamed_text + "▌")
                placeholder.markdown(streamed_text)
            except ChatError as exc:
                logger.warning("Chat error (%s): %s", exc.kind, exc)
                streamed_text = ERROR_MESSAGES.get(exc.kind, ERROR_MESSAGES["unknown"])
                failed = True
                placeholder.markdown(streamed_text)
            except Exception as exc:  # noqa: BLE001
                logger.exception("UI streaming error: %s", exc)
                streamed_text = ERROR_MESSAGES["unknown"]
                failed = True
                placeholder.markdown(streamed_text)

            audit = _audit_from_turn(turn) if not failed else []
            if researcher:
                _render_audit(audit)

        if not failed:
            engine.complete_turn(user_input, streamed_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": streamed_text, "audit": audit, "error": failed}
        )

        if st.session_state.current_conversation_id is None:
            st.session_state.current_conversation_id = store.new_id()
        store.save(
            st.session_state.current_conversation_id,
            st.session_state.messages,
            project_id=project_id,
        )
        st.rerun()


if __name__ == "__main__":
    main()
