"""Vetted web search with strict .edu / .org domain filtering."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from src.models import WebSearchHit, WebSearchResult

logger = logging.getLogger("agrivoltaics.search")

ALLOWED_TLD_PATTERN = re.compile(r"\.(edu|org)(?:/|$)", re.I)
BLOCKED_COMMERCIAL_HINTS = re.compile(
    r"\b(buy now|shop|coupon|affiliate|sponsored|\.com/blog)\b",
    re.I,
)

DOMAIN_FILTER_SUFFIX = " site:.edu OR site:.org"


class VettedWebSearch:
    """
    Web search API wrapper that enforces university/extension credibility.

    Uses Tavily when configured; gracefully degrades when unavailable.
    """

    def __init__(self, tavily_api_key: str | None) -> None:
        self._api_key = tavily_api_key
        self._client = None
        if tavily_api_key:
            try:
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=tavily_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tavily client initialization failed: %s", exc)

    @staticmethod
    def build_filtered_query(query: str) -> str:
        """Append mandatory domain constraints to outbound search queries."""
        base = query.strip()
        if DOMAIN_FILTER_SUFFIX.lower() in base.lower():
            return base
        return f"{base} {DOMAIN_FILTER_SUFFIX}"

    @staticmethod
    def _is_credible_url(url: str) -> bool:
        """Accept only .edu and .org domains; reject commercial blog patterns."""
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            if not host:
                return False
            if not ALLOWED_TLD_PATTERN.search(host):
                return False
            if host.endswith(".com") or host.endswith(".net"):
                return False
            return True
        except Exception:  # noqa: BLE001
            return False

    def search(self, query: str, max_results: int = 5) -> WebSearchResult:
        """
        Execute filtered web search and discard non-credible sources.

        Returns only university agricultural extensions and research foundations.
        """
        filtered_query = self.build_filtered_query(query)
        hits: list[WebSearchHit] = []

        if self._client is None:
            logger.warning("Web search unavailable — Tavily API key not configured.")
            return WebSearchResult(
                original_query=query,
                filtered_query=filtered_query,
                hits=[],
            )

        try:
            response = self._client.search(
                query=filtered_query,
                search_depth="advanced",
                max_results=max_results * 3,
            )

            for item in response.get("results", []):
                url = str(item.get("url", ""))
                title = str(item.get("title", "Untitled"))
                snippet = str(item.get("content", ""))

                if not self._is_credible_url(url):
                    continue
                if BLOCKED_COMMERCIAL_HINTS.search(f"{title} {snippet}"):
                    continue

                domain = urlparse(url).netloc
                hits.append(
                    WebSearchHit(
                        title=title,
                        url=url,  # type: ignore[arg-type]
                        snippet=snippet,
                        domain=domain,
                    )
                )
                if len(hits) >= max_results:
                    break

            logger.info("Web search returned %s vetted hits.", len(hits))
            return WebSearchResult(
                original_query=query,
                filtered_query=filtered_query,
                hits=hits,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Web search failed: %s", exc)
            return WebSearchResult(
                original_query=query,
                filtered_query=filtered_query,
                hits=[],
            )

    @staticmethod
    def needs_web_fallback(query: str, local_chunk_count: int) -> bool:
        """Heuristic: trigger web search when local RAG coverage is thin or query is regulatory."""
        regulatory = re.search(
            r"\b(regulation|policy|incentive|grant|net metering|utility tariff|latest|current)\b",
            query,
            re.I,
        )
        return local_chunk_count == 0 or bool(regulatory)
