"""
Wikipedia retrieval module for Vanilla ReAct.

Implements:

Search -> Open Page -> Lookup

using the official MediaWiki API.
"""

from __future__ import annotations

import re
from functools import lru_cache

import requests

from config import (
    REQUEST_TIMEOUT,
    MAX_SEARCH_RESULTS,
    MAX_OBSERVATION_CHARS,
)

API_URL = "https://en.wikipedia.org/w/api.php"


class WikipediaRetriever:

    def __init__(self):

        self.session = requests.Session()

        self.current_title = None
        self.current_page = None

    # -------------------------------------------------------------

    @lru_cache(maxsize=2048)
    def search(self, query: str):
        """
        Search Wikipedia.

        Returns
        -------
        list[str]
            Candidate page titles.
        """

        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": query,
            "srlimit": MAX_SEARCH_RESULTS,
        }

        response = self.session.get(
            API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data["query"]["search"]:
            results.append(item["title"])

        return results

    # -------------------------------------------------------------

    @lru_cache(maxsize=2048)
    def open_page(self, title: str):
        """
        Download a Wikipedia page.
        """

        params = {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "format": "json",
            "redirects": 1,
            "explaintext": 1,
        }

        response = self.session.get(
            API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        pages = response.json()["query"]["pages"]

        page = next(iter(pages.values()))

        extract = page.get("extract", "")

        self.current_title = title
        self.current_page = extract

        return extract

    # -------------------------------------------------------------

    def lookup(self, keyword: str):
        """
        Return paragraph(s) containing keyword.
        """

        if self.current_page is None:
            return "No page opened."

        paragraphs = [
            p.strip()
            for p in self.current_page.split("\n")
            if p.strip()
        ]

        keyword = keyword.lower()

        hits = []

        for paragraph in paragraphs:

            if keyword in paragraph.lower():
                hits.append(paragraph)

        if not hits:
            return "Keyword not found."

        observation = "\n\n".join(hits)

        return observation[:MAX_OBSERVATION_CHARS]

    # -------------------------------------------------------------

    def search_action(self, query: str):
        """
        Execute Search[query].

        Returns formatted observation.
        """

        titles = self.search(query)

        if not titles:
            return "No Wikipedia page found."

        title = titles[0]

        page = self.open_page(title)

        summary = page.split("\n")[0]

        observation = (
            f"Title: {title}\n\n"
            f"{summary}"
        )

        return observation[:MAX_OBSERVATION_CHARS]

    # -------------------------------------------------------------

    def lookup_action(self, keyword: str):
        """
        Execute Lookup[keyword].
        """

        return self.lookup(keyword)

    # -------------------------------------------------------------

    def reset(self):

        self.current_title = None
        self.current_page = None