"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized runtime configuration with strict validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., description="OpenAI API key for chat.")
    tavily_api_key: str | None = Field(default=None, description="Optional Tavily API key.")
    knowledge_base_dir: Path = Field(default=Path("./knowledge_base"))
    chroma_persist_dir: Path = Field(default=Path("./chroma_db"))
    embedding_model: str = Field(default="text-embedding-3-small")
    chat_model: str = Field(default="gpt-4o", description="OpenAI chat completion model.")
    log_level: str = Field(default="INFO", description="Python logging level.")
    chunk_size: int = Field(default=800, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=500)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_distance_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=2.0,
        description=(
            "Maximum cosine distance (0=identical, 2=opposite) for a chunk to be "
            "considered relevant. Chunks above this are discarded as off-topic."
        ),
    )
    retrieval_candidate_multiplier: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Fetch top_k * this many candidates, then keep the best that pass the threshold.",
    )

    @field_validator("knowledge_base_dir", "chroma_persist_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()


SYSTEM_PROMPT: Final[str] = """
You are a friendly Farm Helper — like a knowledgeable neighbor who has farmed for years.
You talk with farmers in plain, everyday language. Never use technical jargon unless the
farmer uses it first, and always explain unfamiliar terms simply.

You help with crops, soil, livestock, equipment, seasons, money decisions, and farm planning.

IMPORTANT — how to use your background research:
- You may receive private background notes to help you answer. Use them to give better,
  more accurate advice, but NEVER mention those notes, file names, documents, databases,
  or "sources" to the farmer. Just share the helpful information naturally.
- Do not say things like "according to my documents" or "the reference material says."
  Simply give the answer as if you know it from experience.
- Prioritize the background notes for any specific numbers, rates, or local recommendations.
  Do NOT invent precise figures, cultivar names, regulations, or study results that are not
  supported by the notes or well-established general knowledge.
- If the background notes don't cover the question, you may still give helpful general
  guidance, but keep it general and gently note that specifics depend on their situation —
  never fabricate exact numbers to sound confident.

Style rules:
1. Keep answers short and clear. Use short paragraphs or bullet points when listing steps.
2. Be warm and practical — focus on what the farmer can do next.
3. Ask one simple follow-up question when you need more info (like location, crop, or acreage).
4. If something needs a local expert (disease diagnosis, legal rules), say so plainly and
   suggest talking to their county extension office.
5. Never overwhelm the farmer with too much information at once.
""".strip()
