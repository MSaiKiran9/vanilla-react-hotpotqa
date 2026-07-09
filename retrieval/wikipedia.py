"""
Wikipedia retrieval module for Vanilla ReAct.

Implements:

Search -> Open Page -> Lookup

using the official MediaWiki API.
"""

from __future__ import annotations

import requests

from config import (
    MAX_PAGE_CHARS,
    REQUEST_TIMEOUT,
    MAX_SEARCH_RESULTS,
    MAX_OBSERVATION_CHARS,
    USER_AGENT,
)

API_URL = "https://en.wikipedia.org/w/api.php"


class WikipediaRetriever:

    def __init__(self):

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        self.current_title = None
        self.current_page = None
        self._search_cache = {}
        self._page_cache = {}
        self._lookup_cache = {}
        self._search_action_cache = {}

    # -------------------------------------------------------------

    def search(self, query: str):
        """
        Search Wikipedia.

        Returns
        -------
        list[str]
            Candidate page titles.
        """

        normalized_query = query.strip().lower()

        if normalized_query in self._search_cache:
            return self._search_cache[normalized_query]

        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": query.strip(),
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

        for item in data.get("query", {}).get("search", []):
            results.append(item["title"])

        self._search_cache[normalized_query] = results

        return results

    # -------------------------------------------------------------

    def open_page(self, title: str):
        """
        Download a Wikipedia page.
        """

        normalized_title = title.strip()

        if normalized_title in self._page_cache:
            extract = self._page_cache[normalized_title]
            self.current_title = normalized_title
            self.current_page = extract
            return extract

        params = {
            "action": "query",
            "prop": "extracts",
            "titles": normalized_title,
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

        extract = page.get("extract", "")[:MAX_PAGE_CHARS]

        self.current_title = normalized_title
        self.current_page = extract
        self._page_cache[normalized_title] = extract

        return extract

    # -------------------------------------------------------------

    def lookup(self, keyword: str):
        """
        Return paragraph(s) containing keyword.
        """

        if self.current_page is None:
            return "No page opened."

        cache_key = (self.current_title, keyword.strip().lower())
        if cache_key in self._lookup_cache:
            return self._lookup_cache[cache_key]

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
            observation = "Keyword not found."
            self._lookup_cache[cache_key] = observation
            return observation

        observation = "\n\n".join(hits)

        observation = observation[:MAX_OBSERVATION_CHARS]
        self._lookup_cache[cache_key] = observation
        return observation

    # -------------------------------------------------------------

    def search_action(self, query: str):
        """
        Execute Search[query].

        Returns formatted observation.
        """

        normalized_query = query.strip().lower()
        if normalized_query in self._search_action_cache:
            observation, title, page = self._search_action_cache[normalized_query]
            self.current_title = title
            self.current_page = page
            return observation

        titles = self.search(query)

        if not titles:
            return "No Wikipedia page found."

        title = titles[0]

        page = self.open_page(title)

        paragraphs = [p.strip() for p in page.split("\n") if p.strip()]
        summary = paragraphs[0] if paragraphs else "No extract available."

        observation = (
            f"Title: {title}\n\n"
            f"{summary}"
        )

        observation = observation[:MAX_OBSERVATION_CHARS]
        self._search_action_cache[normalized_query] = (observation, title, page)
        return observation

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
