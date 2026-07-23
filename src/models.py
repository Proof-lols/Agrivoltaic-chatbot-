"""Strict Pydantic domain models for agrivoltaics consultation state."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class CorePillar(str, Enum):
    """Four baseline pillars required for final recommendations."""

    PARCEL_ACREAGE = "parcel_acreage"
    SOIL_PROFILE = "soil_profile"
    MICROCLIMATE = "microclimate"
    CAPITAL = "capital"


class AdvancedTopic(str, Enum):
    """Secondary optimization layers for proactive escalation."""

    PANEL_TRACKING = "panel_tracking"
    SOLAR_TILT = "solar_tilt"
    NET_METERING = "net_metering"
    SHADE_CULTIVARS = "shade_cultivars"


PILLAR_LABELS: dict[CorePillar, str] = {
    CorePillar.PARCEL_ACREAGE: "Parcel acreage (total farmable acres)",
    CorePillar.SOIL_PROFILE: "Soil profile & biology (texture, pH, organic matter, drainage)",
    CorePillar.MICROCLIMATE: "Microclimate & shading limits (DNI/GHI, frost dates, wind)",
    CorePillar.CAPITAL: "Capital availability (budget range, financing, incentives sought)",
}

ADVANCED_TOPIC_PROMPTS: dict[AdvancedTopic, str] = {
    AdvancedTopic.PANEL_TRACKING: (
        "What panel mounting style are you considering (fixed-tilt, single-axis tracking, "
        "elevated stilt structures)?"
    ),
    AdvancedTopic.SOLAR_TILT: (
        "Do you know your target solar tilt angle and row spacing constraints for crop machinery?"
    ),
    AdvancedTopic.NET_METERING: (
        "What are your utility net-metering rules, interconnection limits, and export tariffs?"
    ),
    AdvancedTopic.SHADE_CULTIVARS: (
        "Which shade-tolerant cultivars or crop rotations are you evaluating under partial shade?"
    ),
}


class PillarExtraction(BaseModel):
    """Detected pillar values parsed from natural language input."""

    parcel_acreage: str | None = None
    soil_profile: str | None = None
    microclimate: str | None = None
    capital: str | None = None

    def missing_pillars(self) -> list[CorePillar]:
        """Return pillars not yet supplied by the farmer."""
        missing: list[CorePillar] = []
        if not self.parcel_acreage:
            missing.append(CorePillar.PARCEL_ACREAGE)
        if not self.soil_profile:
            missing.append(CorePillar.SOIL_PROFILE)
        if not self.microclimate:
            missing.append(CorePillar.MICROCLIMATE)
        if not self.capital:
            missing.append(CorePillar.CAPITAL)
        return missing

    def satisfied(self) -> bool:
        """True when all four core pillars contain values."""
        return len(self.missing_pillars()) == 0


class ValidationResult(BaseModel):
    """Output of the two-step interactive evaluation pre-processor."""

    extraction: PillarExtraction
    missing_pillars: list[CorePillar]
    can_finalize_recommendation: bool
    intermediate_allowed: bool = True
    validation_notes: list[str] = Field(default_factory=list)


class RetrievalChunk(BaseModel):
    """Single RAG chunk with transparent citation metadata."""

    document_id: str
    content: str
    source: str
    page: int | str | None = None
    score: float | None = None


class RetrievalResult(BaseModel):
    """Aggregated local retrieval output."""

    query: str
    chunks: list[RetrievalChunk] = Field(default_factory=list)

    def formatted_context(self) -> str:
        """Render chunks for LLM context injection."""
        if not self.chunks:
            return "No local knowledge-base matches were found."

        blocks: list[str] = []
        for index, chunk in enumerate(self.chunks, start=1):
            page_label = f", page {chunk.page}" if chunk.page is not None else ""
            blocks.append(
                f"[Local Source {index}] {chunk.source}{page_label}\n{chunk.content}"
            )
        return "\n\n".join(blocks)


class WebSearchHit(BaseModel):
    """Vetted external reference constrained to .edu / .org domains."""

    title: str
    url: HttpUrl
    snippet: str
    domain: str


class WebSearchResult(BaseModel):
    """Filtered web search payload."""

    original_query: str
    filtered_query: str
    hits: list[WebSearchHit] = Field(default_factory=list)

    def formatted_context(self) -> str:
        """Render web hits for LLM grounding."""
        if not self.hits:
            return "No vetted .edu/.org web references were retrieved."

        blocks: list[str] = []
        for index, hit in enumerate(self.hits, start=1):
            blocks.append(
                f"[Web Source {index}] {hit.title} ({hit.domain})\n"
                f"URL: {hit.url}\n{hit.snippet}"
            )
        return "\n\n".join(blocks)


class ChatMessage(BaseModel):
    """Conversation turn."""

    role: str
    content: str


class ConversationState(BaseModel):
    """Tracks farmer parameters and escalation progress."""

    profile: PillarExtraction = Field(default_factory=PillarExtraction)
    addressed_advanced: list[AdvancedTopic] = Field(default_factory=list)
    history: list[ChatMessage] = Field(default_factory=list)
    block_final_recommendation: bool = True

    def merge_extraction(self, extraction: PillarExtraction) -> None:
        """Merge newly parsed pillar values into persistent profile."""
        for field_name in PillarExtraction.model_fields:
            incoming = getattr(extraction, field_name)
            if incoming:
                setattr(self.profile, field_name, incoming)

        self.block_final_recommendation = not self.profile.satisfied()

    def next_advanced_prompts(self, limit: int = 2) -> list[str]:
        """Return proactive follow-up questions not yet asked."""
        pending = [
            ADVANCED_TOPIC_PROMPTS[topic]
            for topic in AdvancedTopic
            if topic not in self.addressed_advanced
        ]
        return pending[:limit]

    def mark_advanced_addressed(self, topics: list[AdvancedTopic]) -> None:
        """Record advanced topics covered in the latest turn."""
        for topic in topics:
            if topic not in self.addressed_advanced:
                self.addressed_advanced.append(topic)


class ChatTurnResult(BaseModel):
    """Structured result returned after each chat cycle."""

    assistant_text: str
    validation: ValidationResult | None = None
    retrieval: RetrievalResult | None = None
    web_search: WebSearchResult | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    escalation_questions: list[str] = Field(default_factory=list)
