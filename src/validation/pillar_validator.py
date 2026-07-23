"""Input pre-processing validation for the four core agrivoltaics pillars."""

from __future__ import annotations

import logging
import re
from typing import Final

from src.models import CorePillar, PillarExtraction, ValidationResult

logger = logging.getLogger("agrivoltaics.validation")

ACREAGE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:acres?|acreage|ha|hectares?)\b", re.I),
    re.compile(r"\bparcel\s*(?:size|acreage)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b", re.I),
]

SOIL_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"\b(soil|clay|sandy|loam|silt|organic matter|pH|drainage|CEC|biology|microbiome)\b",
        re.I,
    ),
]

MICROCLIMATE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"\b(shade|shading|microclimate|GHI|DNI|frost|wind|humidity|irradiance|"
        r"solar access|canopy|partial shade)\b",
        re.I,
    ),
]

CAPITAL_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(\$[\d,]+(?:\.\d{2})?(?:\s*(?:k|m|million|thousand))?|\b\d+\s*(?:k|million)\b|"
        r"\b(budget|capital|financing|investment|ROI|payback|incentive|grant|loan)\b)",
        re.I,
    ),
]


def _first_match(patterns: list[re.Pattern[str]], text: str) -> str | None:
    """Return the first regex match group or full match span as evidence string."""
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            if match.groups():
                return match.group(0).strip()
            return match.group(0).strip()
    return None


def _extract_sentence_evidence(text: str, keyword_pattern: re.Pattern[str]) -> str | None:
    """Pull the sentence containing a keyword for richer pillar capture."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if keyword_pattern.search(sentence):
            return sentence.strip()
    return None


class PillarValidator:
    """Validates farmer queries against the four core decision pillars."""

    def parse_query(self, query: str) -> PillarExtraction:
        """
        Parse natural-language farmer input and extract pillar evidence.

        Uses regex heuristics plus sentence-level capture for partial metrics.
        """
        try:
            normalized = query.strip()
            if not normalized:
                return PillarExtraction()

            acreage = _first_match(ACREAGE_PATTERNS, normalized)
            soil = _first_match(SOIL_PATTERNS, normalized) or _extract_sentence_evidence(
                normalized, SOIL_PATTERNS[0]
            )
            microclimate = _first_match(MICROCLIMATE_PATTERNS, normalized) or _extract_sentence_evidence(
                normalized, MICROCLIMATE_PATTERNS[0]
            )
            capital = _first_match(CAPITAL_PATTERNS, normalized) or _extract_sentence_evidence(
                normalized, CAPITAL_PATTERNS[0]
            )

            extraction = PillarExtraction(
                parcel_acreage=acreage,
                soil_profile=soil,
                microclimate=microclimate,
                capital=capital,
            )
            logger.debug("Pillar extraction: %s", extraction.model_dump())
            return extraction
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse query pillars: %s", exc)
            return PillarExtraction()

    def validate(self, query: str, known_profile: PillarExtraction | None = None) -> ValidationResult:
        """
        Execute the two-step interactive evaluation pre-check.

        Merges session-known profile values with newly parsed query content.
        """
        try:
            parsed = self.parse_query(query)
            merged = PillarExtraction(
                parcel_acreage=parsed.parcel_acreage or (known_profile.parcel_acreage if known_profile else None),
                soil_profile=parsed.soil_profile or (known_profile.soil_profile if known_profile else None),
                microclimate=parsed.microclimate or (known_profile.microclimate if known_profile else None),
                capital=parsed.capital or (known_profile.capital if known_profile else None),
            )
            missing = merged.missing_pillars()
            notes: list[str] = []

            if missing:
                notes.append(
                    "Partial parameters detected. Intermediate guidance permitted; "
                    "absolute final recommendations blocked until all pillars are supplied."
                )
            else:
                notes.append("All four core pillars satisfied. Final recommendations enabled.")

            return ValidationResult(
                extraction=merged,
                missing_pillars=missing,
                can_finalize_recommendation=len(missing) == 0,
                intermediate_allowed=True,
                validation_notes=notes,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Validation failure: %s", exc)
            empty = known_profile or PillarExtraction()
            return ValidationResult(
                extraction=empty,
                missing_pillars=empty.missing_pillars() or list(CorePillar),
                can_finalize_recommendation=False,
                intermediate_allowed=True,
                validation_notes=[f"Validation error intercepted: {exc}"],
            )

    @staticmethod
    def missing_pillar_labels(missing: list[CorePillar]) -> list[str]:
        """Human-readable labels for data collection forms."""
        from src.models import PILLAR_LABELS

        return [PILLAR_LABELS[pillar] for pillar in missing]
