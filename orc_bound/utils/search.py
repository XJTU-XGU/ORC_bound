"""
Web search utilities via SerpAPI.

Used to fetch ORC-related literature and references from Google Scholar.
Requires a SerpAPI key: https://serpapi.com/
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from google_search_results import GoogleSearchResults
    _HAS_SERPAPI = True
except ImportError:
    _HAS_SERPAPI = False


def _get_serpapi_key(serpapi_key: Optional[str] = None) -> str:
    """
    Resolve SerpAPI key from argument or environment.

    Raises
    ------
    EnvironmentError
        If no key is available.
    """
    key = serpapi_key or os.environ.get("SERPAPI_KEY", "")
    if not key:
        raise EnvironmentError(
            "SerpAPI key not provided and SERPAPI_KEY environment variable is not set. "
            "Get a free key at https://serpapi.com/"
        )
    return key


def search_orc_literature(
    query: str,
    num_results: int = 10,
    serpapi_key: Optional[str] = None,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    Search for ORC-related literature via Google Scholar through SerpAPI.

    Parameters
    ----------
    query : str
        Search query string. Example: ``"optimal rank canonical graph curvature"``.
    num_results : int, default=10
        Number of results to return (max 100).
    serpapi_key : str or None
        SerpAPI key. If None, reads from the ``SERPAPI_KEY`` environment variable.
    lang : str, default="en"
        Language code for results (e.g., ``"en"``, ``"zh"``).

    Returns
    -------
    List[Dict[str, Any]]
        List of search results. Each dict contains:
        - ``title``: Paper title.
        - ``link``: URL to the paper.
        - ``snippet``: Short abstract or description.
        - ``position``: Rank in results.

    Raises
    ------
    EnvironmentError
        If the SerpAPI key is not available.
    ImportError
        If the ``google-search-results`` package is not installed.
        Install with: ``pip install google-search-results``
    RuntimeError
        If the SerpAPI request fails.

    Examples
    --------
    >>> results = search_orc_literature(
    ...     query="Ricci curvature graph networks",
    ...     num_results=5,
    ...     serpapi_key="your-key-here",
    ... )
    >>> for r in results:
    ...     print(r["title"])
    """
    if not _HAS_SERPAPI:
        raise ImportError(
            "The 'google-search-results' package is required for web search. "
            "Install it with: pip install google-search-results"
        )

    key = _get_serpapi_key(serpapi_key)
    client = GoogleSearchResults({"serpapi_api_key": key})

    params = {
        "q": query,
        "num": min(num_results, 100),
        "hl": lang,
        "engine": "google_scholar",
    }
    client.set_dict(params)

    data = client.get_dict()

    if "organic_results" not in data:
        error = data.get("error", "Unknown error")
        raise RuntimeError(f"SerpAPI request failed: {error}")

    results = []
    for item in data["organic_results"]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "position": item.get("position", len(results) + 1),
        })

    return results


def search_general(
    query: str,
    num_results: int = 10,
    serpapi_key: Optional[str] = None,
    engine: str = "google",
) -> List[Dict[str, Any]]:
    """
    General web search via SerpAPI.

    Parameters
    ----------
    query : str
        Search query.
    num_results : int, default=10
        Number of results (max 100).
    serpapi_key : str or None
        SerpAPI key. If None, reads from ``SERPAPI_KEY`` env var.
    engine : str, default="google"
        Search engine: ``"google"``, ``"bing"``, ``"duckduckgo"``, etc.

    Returns
    -------
    List[Dict[str, Any]]
        List of results with ``title``, ``link``, and ``snippet``.

    Examples
    --------
    >>> results = search_general(
    ...     "residual shell Ricci curvature graph",
    ...     num_results=5,
    ...     serpapi_key="your-key-here",
    ... )
    """
    if not _HAS_SERPAPI:
        raise ImportError(
            "The 'google-search-results' package is required for web search. "
            "Install it with: pip install google-search-results"
        )

    key = _get_serpapi_key(serpapi_key)
    client = GoogleSearchResults({"serpapi_api_key": key})

    params = {
        "q": query,
        "num": min(num_results, 100),
        "engine": engine,
    }
    client.set_dict(params)

    data = client.get_dict()

    key_name = "organic_results"
    if key_name not in data:
        error = data.get("error", "Unknown error")
        raise RuntimeError(f"SerpAPI request failed: {error}")

    results = []
    for item in data[key_name]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", item.get("description", "")),
            "position": item.get("position", len(results) + 1),
        })

    return results
