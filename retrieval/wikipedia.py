"""
Lightweight Wikipedia retrieval for the Vanilla ReAct baseline.

Implements:
- Search for page titles
- Fetch page extracts
- Simple in-memory caching

Uses the official MediaWiki API.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import requests

from config import REQUEST_TIMEOUT, USER_AGENT

API_URL = "https://en.wikipedia.org/w/api.php"


@dataclass(slots=True)
class SearchResult:
    title: str
    snippet: str
    pageid: int


class WikipediaRetriever:
    """
    Simple MediaWiki API wrapper.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT
            }
        )

    @lru_cache(maxsize=4096)
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[SearchResult]:
        """
        Search Wikipedia.
        """

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }

        response = self.session.get(
            API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get("query", {}).get("search", []):

            results.append(
                SearchResult(
                    title=item["title"],
                    snippet=item.get("snippet", ""),
                    pageid=item["pageid"],
                )
            )

        return results

    @lru_cache(maxsize=4096)
    def get_page(
        self,
        title: str,
    ) -> Optional[str]:
        """
        Fetch plain-text page extract.
        """

        params = {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "format": "json",
            "explaintext": 1,
            "redirects": 1,
        }

        response = self.session.get(
            API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        pages = response.json()["query"]["pages"]

        for page in pages.values():

            extract = page.get("extract")

            if extract:
                return extract

        return None

    def lookup(
        self,
        title: str,
        keyword: str,
        window: int = 2,
    ) -> Optional[str]:
        """
        Return a small evidence window around
        the first matching paragraph.
        """

        page = self.get_page(title)

        if page is None:
            return None

        paragraphs = [
            p.strip()
            for p in page.split("\n")
            if p.strip()
        ]

        keyword_lower = keyword.lower()

        for i, paragraph in enumerate(paragraphs):

            if keyword_lower in paragraph.lower():

                start = max(0, i - window)
                end = min(len(paragraphs), i + window + 1)

                return "\n\n".join(
                    paragraphs[start:end]
                )

        return None