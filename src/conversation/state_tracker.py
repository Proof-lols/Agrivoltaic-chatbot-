"""Conversation state tracking and proactive question escalation."""

from __future__ import annotations

import logging
import re

from src.models import AdvancedTopic, ChatMessage, ConversationState, PillarExtraction
from src.validation.pillar_validator import PillarValidator

logger = logging.getLogger("agrivoltaics.conversation")

ADVANCED_KEYWORDS: dict[AdvancedTopic, re.Pattern[str]] = {
    AdvancedTopic.PANEL_TRACKING: re.compile(
        r"\b(tracking|single-axis|dual-axis|fixed-tilt|stilt|elevated structure)\b",
        re.I,
    ),
    AdvancedTopic.SOLAR_TILT: re.compile(r"\b(tilt|azimuth|row spacing|pitch)\b", re.I),
    AdvancedTopic.NET_METERING: re.compile(
        r"\b(net metering|interconnection|export tariff|utility rate|PPA)\b",
        re.I,
    ),
    AdvancedTopic.SHADE_CULTIVARS: re.compile(
        r"\b(cultivar|variety|shade-tolerant|crop rotation|botanical)\b",
        re.I,
    ),
}


class ConversationStateTracker:
    """Maintains multi-turn farmer context and drives proactive escalation."""

    def __init__(self) -> None:
        self._validator = PillarValidator()
        self.state = ConversationState()

    def ingest_user_message(self, message: str) -> ConversationState:
        """Update persistent profile and advanced-topic coverage from user text."""
        try:
            extraction = self._validator.parse_query(message)
            self.state.merge_extraction(extraction)

            for topic, pattern in ADVANCED_KEYWORDS.items():
                if pattern.search(message):
                    self.state.mark_advanced_addressed([topic])

            self.state.history.append(ChatMessage(role="user", content=message))
            logger.debug("Conversation state updated: %s", self.state.model_dump())
            return self.state
        except Exception as exc:  # noqa: BLE001
            logger.exception("State ingestion failed: %s", exc)
            return self.state

    def apply_form_submission(self, form_data: dict[str, str]) -> ConversationState:
        """Merge sidebar/form collected pillar metrics into session profile."""
        try:
            extraction = PillarExtraction(
                parcel_acreage=form_data.get("parcel_acreage") or self.state.profile.parcel_acreage,
                soil_profile=form_data.get("soil_profile") or self.state.profile.soil_profile,
                microclimate=form_data.get("microclimate") or self.state.profile.microclimate,
                capital=form_data.get("capital") or self.state.profile.capital,
            )
            self.state.merge_extraction(extraction)
            return self.state
        except Exception as exc:  # noqa: BLE001
            logger.exception("Form merge failed: %s", exc)
            return self.state

    def escalation_questions(self, limit: int = 2) -> list[str]:
        """Return proactive advanced-layer questions when basics are satisfied."""
        if self.state.block_final_recommendation:
            return []
        return self.state.next_advanced_prompts(limit=limit)

    def reset(self) -> None:
        """Clear session state for a new consultation."""
        self.state = ConversationState()
